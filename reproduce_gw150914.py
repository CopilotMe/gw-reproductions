"""
Reproduce the GW150914 detection from public LIGO strain data.

Pulls 32 s of H1 and L1 strain around the event from GWOSC, whitens it,
runs a matched filter against a GR template (SEOBNRv4) for the published
component masses, and writes diagnostic plots to ./output/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycbc.catalog import Merger
from pycbc.filter import matched_filter, resample_to_delta_t, highpass
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform import get_td_waveform

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

EVENT = "GW150914"
DETECTORS = ("H1", "L1")
SAMPLE_RATE = 2048
SEGMENT_SECONDS = 32
MASS1 = 36.0
MASS2 = 29.0


def load_strain(detector: str):
    merger = Merger(EVENT)
    strain = merger.strain(detector)
    strain = resample_to_delta_t(highpass(strain, 15.0), 1.0 / SAMPLE_RATE)
    strain = strain.crop(2, 2)
    return merger, strain


def whiten(strain):
    psd = strain.psd(4)
    psd = interpolate(psd, strain.delta_f)
    psd = inverse_spectrum_truncation(
        psd, int(4 * strain.sample_rate), low_frequency_cutoff=20
    )
    return psd


def make_template(delta_t, flen):
    hp, _ = get_td_waveform(
        approximant="SEOBNRv4_opt",
        mass1=MASS1,
        mass2=MASS2,
        delta_t=delta_t,
        f_lower=20,
    )
    hp.resize(flen)
    template = hp.cyclic_time_shift(hp.start_time)
    return template


def main():
    fig_strain, axes_strain = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig_snr, axes_snr = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig_white, axes_white = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    results = {}
    merger_gps = Merger(EVENT).time

    for idx, det in enumerate(DETECTORS):
        print(f"[{det}] fetching strain...")
        _, strain = load_strain(det)
        psd = whiten(strain)

        flen = len(strain) // 2 + 1
        template = make_template(strain.delta_t, len(strain))

        snr = matched_filter(
            template, strain, psd=psd, low_frequency_cutoff=20
        )
        snr = snr.crop(4 + 4, 4)

        peak = abs(snr).numpy().argmax()
        peak_time = snr.sample_times[peak]
        peak_snr = abs(snr[peak])
        results[det] = (peak_time, peak_snr)
        print(f"[{det}] peak |SNR|={peak_snr:.2f} at GPS {peak_time:.4f}")

        t_rel_raw = strain.sample_times - merger_gps
        axes_strain[idx].plot(t_rel_raw, strain, lw=0.5)
        axes_strain[idx].set_xlim(-0.5, 0.2)
        axes_strain[idx].set_ylabel(f"{det} strain")
        axes_strain[idx].grid(alpha=0.3)

        white = (strain.to_frequencyseries() / psd**0.5).to_timeseries()
        white = white.highpass_fir(30, 512).lowpass_fir(300, 512)
        white = white.time_slice(merger_gps - 0.2, merger_gps + 0.1)
        axes_white[idx].plot(
            white.sample_times - merger_gps, white, color="C1", lw=0.8
        )
        axes_white[idx].set_ylabel(f"{det} whitened\n(30–300 Hz)")
        axes_white[idx].grid(alpha=0.3)

        axes_snr[idx].plot(
            snr.sample_times - merger_gps, abs(snr), color="C2"
        )
        axes_snr[idx].axvline(
            peak_time - merger_gps,
            color="red",
            ls="--",
            label=f"peak |SNR|={peak_snr:.1f}",
        )
        axes_snr[idx].set_xlim(-0.3, 0.3)
        axes_snr[idx].set_ylabel(f"{det} |SNR|")
        axes_snr[idx].legend(loc="upper right")
        axes_snr[idx].grid(alpha=0.3)

    for ax in (axes_strain[-1], axes_white[-1], axes_snr[-1]):
        ax.set_xlabel(f"t − t_event (s)   [event GPS={merger_gps:.4f}]")

    fig_strain.suptitle(f"{EVENT}: raw strain (highpassed @ 15 Hz, 2048 Hz)")
    fig_white.suptitle(f"{EVENT}: whitened, band-passed 30–300 Hz")
    fig_snr.suptitle(
        f"{EVENT}: matched-filter |SNR| vs template "
        f"(m1={MASS1} M☉, m2={MASS2} M☉, SEOBNRv4)"
    )

    fig_strain.tight_layout()
    fig_white.tight_layout()
    fig_snr.tight_layout()

    fig_strain.savefig(OUT / "1_raw_strain.png", dpi=140)
    fig_white.savefig(OUT / "2_whitened_strain.png", dpi=140)
    fig_snr.savefig(OUT / "3_matched_filter_snr.png", dpi=140)

    delta_t = abs(results["H1"][0] - results["L1"][0]) * 1000
    print()
    print(f"H1 vs L1 peak time difference: {delta_t:.2f} ms")
    print("(LIGO published value: ~6.9 ms — light travel time between sites is 10 ms)")
    print(f"Plots written to {OUT}/")


if __name__ == "__main__":
    main()
