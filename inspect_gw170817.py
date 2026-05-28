"""
Storytelling Q-transform for GW170817: H1 (clean), L1 raw (showing glitch),
L1 after gating (chirp emerges).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries

OUT = Path(__file__).parent / "output"
GPS = 1187008882.41
GLITCH = 1187008881.4
HW = 4.0


def fetch(det):
    return TimeSeries.fetch_open_data(
        det, GPS - HW - 16, GPS + HW + 16,
        sample_rate=4096, cache=True, verbose=False,
    ).highpass(15.0)


def gate(ts, gate_time, half=0.5, taper=0.25):
    data = ts.value.copy()
    sr = float(ts.sample_rate.value)
    t0 = float(ts.t0.value)
    n = len(data)
    center = int((gate_time - t0) * sr)
    h = int(half * sr)
    tn = int(taper * sr)
    w = np.ones_like(data)
    w[max(0, center - h):min(n, center + h)] = 0.0
    for i in range(tn):
        alpha = 0.5 * (1 - np.cos(np.pi * i / tn))
        for idx in (center - h - tn + i, center + h + tn - i):
            if 0 <= idx < n:
                w[idx] = min(w[idx], 1 - alpha)
    new = ts.copy()
    new[:] = data * w
    return new


def qspec(ts):
    return ts.whiten(4, 2).q_transform(
        qrange=(4, 64), frange=(20, 1024),
        outseg=(GPS - HW, GPS + HW), whiten=False,
    )


def panel(ax, q, title, vmax_pct=99):
    times = q.times.value - GPS
    freqs = q.frequencies.value
    ax.pcolormesh(times, freqs, q.value.T, shading="auto", cmap="viridis",
                  vmin=0, vmax=np.percentile(q.value, vmax_pct))
    ax.set_yscale("log")
    ax.set_ylim(20, 1024)
    ax.set_ylabel(f"{title}\nfreq [Hz]")
    ax.axvline(0, color="white", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(GLITCH - GPS, color="red", lw=0.8, ls=":", alpha=0.5)


def main():
    h1 = fetch("H1")
    l1_raw = fetch("L1")
    l1_gated = gate(l1_raw, GLITCH)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    panel(axes[0], qspec(h1), "H1 (raw — clean)", vmax_pct=99.5)
    panel(axes[1], qspec(l1_raw), "L1 (raw — glitch dominates)", vmax_pct=99)
    panel(axes[2], qspec(l1_gated), "L1 (after gating glitch — chirp emerges)", vmax_pct=99.5)
    axes[-1].set_xlabel(f"t − {GPS:.3f} (s)   (red dotted line = glitch time)")
    fig.suptitle("GW170817 — binary neutron star, 17 Aug 2017 12:41:04 UTC")
    fig.tight_layout()
    out = OUT / "qscan_GW170817_BNS_3panel.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
