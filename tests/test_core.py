import numpy as np

from sad.data import load_labels, windows
from sad.detect import Config, detect, dynamic_threshold, prune, to_sequences
from sad.metrics import aggregate, channel_counts, prf


def test_labels_merge_p2_and_counts():
    df = load_labels()
    assert len(df) == 81
    assert sum(len(s) for s in df.seqs) == 105
    assert df.spacecraft.value_counts().to_dict() == {"SMAP": 54, "MSL": 27}


def test_windows_alignment():
    x = np.arange(20, dtype=float)[:, None]
    X, y = windows(x, 5)
    assert X.shape == (15, 5, 1) and y[0] == 5 and X[0, -1, 0] == 4 and y[-1] == 19


def test_to_sequences_merges_and_pads():
    m = np.zeros(300, bool)
    m[[10, 11, 40, 200]] = True
    assert to_sequences(m, pad=0, merge_gap=50) == [[10, 40], [200, 200]]
    assert to_sequences(m, pad=5, merge_gap=50)[0] == [5, 45]


def test_strict_counts():
    c = channel_counts(pred=[[10, 20], [100, 110]], true=[[15, 30], [200, 210]], n=300)
    assert (c["tp"], c["fp"], c["fn"]) == (1, 1, 1)
    # point-adjusted credits the whole detected true segment (16 points)
    assert c["pa_tp"] == 16 and c["pt_tp"] == 6


def test_prf():
    assert prf(1, 1, 1) == (0.5, 0.5, 0.5)
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_aggregate_pools_counts():
    a = channel_counts([[0, 5]], [[0, 5]], 100)
    b = channel_counts([], [[0, 5]], 100)
    agg = aggregate([a, b])
    assert (agg["tp"], agg["fp"], agg["fn"]) == (1, 0, 1)


def test_dynamic_threshold_finds_spike_and_prune_keeps_it():
    rng = np.random.default_rng(0)
    e = rng.normal(1, 0.05, 2000)
    e[1000:1020] += 3
    cfg = Config(1, "dynamic", 0.0, 0.13)
    seqs = detect(e, e[:200], cfg)
    assert len(seqs) == 1 and seqs[0][0] <= 1000 and seqs[0][1] >= 1019
    assert np.isfinite(dynamic_threshold(e))


def test_prune_drops_marginal_sequences():
    seqs = [[10, 10], [50, 50]]
    # second peak (1.05) barely above the highest normal value (1.0): pruned
    e = np.ones(100)
    e[10], e[50] = 5.0, 1.05
    assert prune(seqs, e, 1.04, 0.13) == [[10, 10]]
    # second peak clearly above normal: both kept
    e[50] = 2.1
    assert prune(seqs, e, 1.5, 0.13) == seqs
    assert prune(seqs, e, 1.5, 0.0) == seqs   # pruning off


def test_static_k_uses_calibration_only():
    cal = np.random.default_rng(1).normal(0, 1, 500)
    test = np.zeros(300)
    test[100] = 50
    assert detect(test, cal, Config(1, "static_k", 8)) != []
    assert detect(np.zeros(300), cal, Config(1, "static_k", 8)) == []
