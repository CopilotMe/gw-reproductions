"""
Render Q-transform spectrograms for a candidate trigger in both detectors.

Usage:
  python inspect_trigger.py --gps <GPS> --label <name> [--window 8]

A genuine BBH chirp shows as a sweeping curve from low to high frequency in
both detectors, with the L1 image time-shifted relative to H1 by < 10 ms.
A glitch typically appears in only one detector, or as a vertical line / blob.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries

OUT = Path(__file__).parent / "output"


def render(detector, gps, half_window, ax):
    pad = 16
    ts = TimeSeries.fetch_open_data(
        detector, gps - half_window - pad, gps + half_window + pad,
        sample_rate=4096, cache=True, verbose=False,
    )
    ts = ts.highpass(15.0)
    white = ts.whiten(4, 2)
    qspec = white.q_transform(
        qrange=(4, 64),
        frange=(20, 1024),
        outseg=(gps - half_window, gps + half_window),
        whiten=False,
    )
    times = qspec.times.value - gps
    freqs = qspec.frequencies.value
    img = ax.pcolormesh(
        times, freqs, qspec.value.T,
        shading="auto", cmap="viridis",
        vmin=0, vmax=np.percentile(qspec.value, 99.5),
    )
    ax.set_yscale("log")
    ax.set_ylim(20, 1024)
    ax.set_ylabel(f"{detector}  freq [Hz]")
    ax.axvline(0, color="white", lw=0.8, ls="--", alpha=0.6)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gps", type=float, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--window", type=float, default=4.0,
                   help="half-window in seconds (default 4)")
    args = p.parse_args()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, det in zip(axes, ("H1", "L1")):
        render(det, args.gps, args.window, ax)
    axes[-1].set_xlabel(f"t − {args.gps:.3f} (s)")
    fig.suptitle(f"Q-transform — {args.label}  (GPS {args.gps:.3f})")
    fig.tight_layout()
    out = OUT / f"qscan_{args.label}.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
