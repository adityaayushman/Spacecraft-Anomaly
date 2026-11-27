"""Detectors behind one interface.

Every model exposes fit(fit_x, es_x) and score(x) -> array of len(x), where a
higher value means "more anomalous" and the first W entries (no context yet)
are padded with the median score. Forecasters score = |prediction error|.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge

from .data import windows

W = 100          # context length for forecasters
AE_W = 30        # window for the autoencoder
torch.set_num_threads(2)


def _pad(score, w, total):
    out = np.empty(total, dtype=np.float64)
    out[w:] = score
    out[:w] = np.median(score)
    return out


# ------------------------------------------------------------------ baselines
class Persistence:
    name = "persistence"

    def fit(self, fit_x, es_x, seed=0):
        return self

    def score(self, x):
        return _pad(np.abs(x[W:, 0] - x[W - 1:-1, 0]), W, len(x))


class MovingAverage:
    name = "moving_avg"

    def fit(self, fit_x, es_x, seed=0):
        return self

    def score(self, x):
        c = np.r_[0.0, np.cumsum(x[:, 0])]               # c[i] = sum(x[:i])
        k, T = 10, len(x)
        ma = (c[W:T] - c[W - k:T - k]) / k               # mean of x[t-k:t] for t = W..T-1
        return _pad(np.abs(x[W:, 0] - ma), W, T)


class RidgeAR:
    """Linear autoregression on the last 20 steps of all features."""
    name = "ridge_ar"
    L = 20

    def fit(self, fit_x, es_x, seed=0):
        X, y = windows(fit_x, self.L)
        self.m = Ridge(alpha=1.0).fit(X.reshape(len(X), -1), y)
        return self

    def score(self, x):
        X, y = windows(x, self.L)
        err = np.abs(self.m.predict(X.reshape(len(X), -1)) - y)
        return _pad(err, self.L, len(x))


class IForest:
    """Isolation Forest on windows of the telemetry value (no forecasting)."""
    name = "iforest"
    L = 20

    def fit(self, fit_x, es_x, seed=0):
        self.m = IsolationForest(n_estimators=100, random_state=seed, n_jobs=2).fit(self._feat(fit_x))
        return self

    def _feat(self, x):
        v = x[:, 0]
        idx = np.arange(len(v) - self.L + 1)[:, None] + np.arange(self.L)[None, :]
        w = v[idx]
        return np.c_[w, np.diff(w, axis=1).std(1)]

    def score(self, x):
        s = -self.m.score_samples(self._feat(x))
        out = np.empty(len(x))
        out[self.L - 1:] = s
        out[: self.L - 1] = np.median(s)
        return out


# ------------------------------------------------------------------ neural forecasters
class _LSTM(nn.Module):
    def __init__(self, f, h=64, cell=nn.LSTM):
        super().__init__()
        self.rnn = cell(f, h, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        return self.head(self.rnn(x)[0][:, -1]).squeeze(-1)


class _CNN(nn.Module):
    """Causal dilated 1D CNN (TCN-style)."""

    def __init__(self, f, h=32):
        super().__init__()
        self.dil = [1, 2, 4, 8, 16]
        self.convs = nn.ModuleList(
            [nn.Conv1d(f if i == 0 else h, h, 3, dilation=d) for i, d in enumerate(self.dil)])
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        for conv, d in zip(self.convs, self.dil):
            x = F.relu(conv(F.pad(x, (2 * d, 0))))
        return self.head(x[:, :, -1]).squeeze(-1)


class Forecaster:
    """Next-step forecaster. hidden / w (context length) / cmd (use command features) are
    exposed for the sensitivity study; defaults reproduce the main-table models."""

    def __init__(self, arch, hidden=64, w=W, cmd=True):
        self.arch, self.hidden, self.w, self.cmd = arch, hidden, w, cmd

    def _x(self, x):
        return x if self.cmd else x[:, :1]

    def _net(self, f):
        return {"lstm": lambda: _LSTM(f, h=self.hidden, cell=nn.LSTM),
                "gru": lambda: _LSTM(f, h=self.hidden, cell=nn.GRU),
                "cnn": lambda: _CNN(f)}[self.arch]()

    def fit(self, fit_x, es_x, seed=0, epochs=15, patience=3):
        torch.manual_seed(seed)
        X, y = map(torch.from_numpy, windows(self._x(fit_x), self.w, stride=2))
        Xv, yv = map(torch.from_numpy, windows(self._x(es_x), self.w, stride=2))
        net = self._net(X.shape[2])
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        best, state, bad = 1e9, None, 0
        for _ in range(epochs):
            net.train()
            for i in torch.randperm(len(X)).split(128):
                opt.zero_grad()
                F.mse_loss(net(X[i]), y[i]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = F.mse_loss(net(Xv), yv).item()
            if v < best:
                best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(state)
        self.net = net.eval()
        return self

    @torch.no_grad()
    def score(self, x):
        X, y = windows(self._x(x), self.w)
        pred = torch.cat([self.net(torch.from_numpy(b)) for b in np.array_split(X, max(1, len(X) // 1024))])
        return _pad(np.abs(pred.numpy() - y), self.w, len(x))


# ------------------------------------------------------------------ LSTM autoencoder
class _AE(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.enc = nn.LSTM(1, h, batch_first=True)
        self.dec = nn.LSTM(h, h, batch_first=True)
        self.out = nn.Linear(h, 1)

    def forward(self, x):
        _, (hn, _) = self.enc(x)
        z = hn[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        return self.out(self.dec(z)[0])


class LSTMAutoencoder:
    """Univariate sequence autoencoder; score = mean abs reconstruction error of the
    window ending at t."""
    name = "lstm_ae"

    @staticmethod
    def _win(x):
        v = x[:, 0].astype(np.float32)
        idx = np.arange(len(v) - AE_W + 1)[:, None] + np.arange(AE_W)[None, :]
        return torch.from_numpy(v[idx]).unsqueeze(-1)

    def fit(self, fit_x, es_x, seed=0, epochs=15, patience=3):
        torch.manual_seed(seed)
        X, Xv = self._win(fit_x)[::2], self._win(es_x)[::2]
        net = _AE()
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        best, state, bad = 1e9, None, 0
        for _ in range(epochs):
            net.train()
            for i in torch.randperm(len(X)).split(128):
                opt.zero_grad()
                F.mse_loss(net(X[i]), X[i]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = F.mse_loss(net(Xv), Xv).item()
            if v < best:
                best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(state)
        self.net = net.eval()
        return self

    @torch.no_grad()
    def score(self, x):
        X = self._win(x)
        err = torch.cat([(self.net(b) - b).abs().mean((1, 2)) for b in X.split(1024)]).numpy()
        out = np.empty(len(x))
        out[AE_W - 1:] = err
        out[: AE_W - 1] = np.median(err)
        return out


MODELS = {
    "persistence": Persistence, "moving_avg": MovingAverage, "ridge_ar": RidgeAR,
    "iforest": IForest, "lstm": lambda: Forecaster("lstm"), "gru": lambda: Forecaster("gru"),
    "cnn": lambda: Forecaster("cnn"), "lstm_ae": LSTMAutoencoder,
    # sensitivity variants of the GRU (not part of the main table)
    "gru_h32": lambda: Forecaster("gru", hidden=32), "gru_h128": lambda: Forecaster("gru", hidden=128),
    "gru_h256": lambda: Forecaster("gru", hidden=256),
    "gru_w50": lambda: Forecaster("gru", w=50), "gru_w200": lambda: Forecaster("gru", w=200),
    "gru_nocmd": lambda: Forecaster("gru", cmd=False),
}


def run_channel(name, train, test, seed=0):
    """Train on the clean train split; return (test_score, calibration_score).

    train split: first 80% fit, next 10% early stopping, last 10% calibration
    (clean data used only to set static thresholds).
    """
    n = len(train)
    a, b = int(0.8 * n), int(0.9 * n)
    m = MODELS[name]()
    w = getattr(m, "w", W)
    m.fit(train[:a], train[a - w:b], seed=seed)
    cal = m.score(train)[b:]
    return m.score(test), cal
