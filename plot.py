"""Plot forecast error + detections for a few channels -> results/examples.png"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
chans = sys.argv[1:] or ["S-1", "D-16", "F-8"]
fig, ax = plt.subplots(len(chans), 1, figsize=(10, 2.6 * len(chans)), squeeze=False)
for a, c in zip(ax[:, 0], chans):
    d = np.load(f"results/{c}.npz"); x = np.arange(len(d["err"])) + 100
    a.plot(x, d["err"], lw=.8, color="#33658a", label="smoothed forecast error")
    a.axhline(d["eps"], ls="--", color="#999", lw=.8, label="dynamic threshold")
    for i, (s, e) in enumerate(d["true"]): a.axvspan(s, e, color="#2a9d8f", alpha=.25, label="expert label" if i == 0 else None)
    for i, (s, e) in enumerate(d["found"]): a.axvspan(s, e, ymax=.12, color="#e63946", label="detected" if i == 0 else None)
    a.set_title(c, loc="left", fontsize=10); a.set_xlabel("test timestep")
ax[0, 0].legend(fontsize=7, loc="upper left"); plt.tight_layout(); plt.savefig("results/examples.png", dpi=130)
