"""
Mini coincident-trigger matched-filter search for LIGO O3 open data.

Three phases, selected with --phase:
  smoke  10 min near a known event; sanity check that the pipeline finds it
  known  1 hour around GW200129_065458; full validation
  blind  1 hour in a window with no GWTC-1/2/2.1/3 events; methodology demo
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.signal import find_peaks

from pycbc.catalog import Catalog
from pycbc.filter import highpass, matched_filter, resample_to_delta_t
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.types import TimeSeries as PyCBCts
from pycbc.waveform import get_td_waveform

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

SAMPLE_RATE = 2048
F_LOWER = 20.0
APPROXIMANT = "IMRPhenomD"


def fetch_strain(detector: str, gps_start: int, duration: int) -> PyCBCts:
    print(f"  [{detector}] fetching {duration}s @ GPS {gps_start}...", flush=True)
    t0 = time.time()
    ts = TimeSeries.fetch_open_data(
        detector, gps_start, gps_start + duration,
        sample_rate=4096, cache=True, verbose=False,
    )
    pyts = PyCBCts(ts.value, delta_t=float(ts.dt.value), epoch=float(ts.t0.value))
    pyts = resample_to_delta_t(highpass(pyts, 15.0), 1.0 / SAMPLE_RATE)
    print(f"    fetched in {time.time()-t0:.1f}s, {len(pyts)} samples", flush=True)
    return pyts


def gate_strain(strain: PyCBCts, gate_time: float, half_width: float = 0.5, taper: float = 0.25) -> PyCBCts:
    """Zero out a window around gate_time with cosine taper on each side.
    Used for the well-known L1 glitch 1.1 s before GW170817."""
    data = strain.numpy().copy()
    sr = 1.0 / strain.delta_t
    t0 = float(strain.start_time)
    center = int((gate_time - t0) * sr)
    half = int(half_width * sr)
    taper_n = int(taper * sr)
    window = np.ones_like(data)
    window[max(0, center - half):min(len(data), center + half)] = 0.0
    for i in range(taper_n):
        alpha = 0.5 * (1 - np.cos(np.pi * i / taper_n))
        li, ri = center - half - taper_n + i, center + half + taper_n - i
        if 0 <= li < len(window):
            window[li] = min(window[li], 1 - alpha)
        if 0 <= ri < len(window):
            window[ri] = min(window[ri], 1 - alpha)
    return PyCBCts(data * window, delta_t=strain.delta_t, epoch=strain.start_time)


def estimate_psd(strain: PyCBCts):
    psd = strain.psd(4)
    psd = interpolate(psd, strain.delta_f)
    return inverse_spectrum_truncation(
        psd, int(4 * strain.sample_rate),
        low_frequency_cutoff=F_LOWER,
    )


def build_bank(m_min: float, m_max: float, n_per_side: int):
    masses = np.linspace(m_min, m_max, n_per_side)
    return [(float(m1), float(m2)) for m1 in masses for m2 in masses if m1 >= m2]


def make_template(m1: float, m2: float, delta_t: float, flen: int):
    hp, _ = get_td_waveform(
        approximant=APPROXIMANT, mass1=m1, mass2=m2,
        delta_t=delta_t, f_lower=F_LOWER,
    )
    raw_duration = len(hp) * delta_t
    hp.resize(flen)
    return hp.cyclic_time_shift(hp.start_time), raw_duration


def find_triggers(snr_series, threshold: float, min_separation_s: float = 1.0):
    abs_snr = abs(snr_series).numpy()
    distance = max(1, int(min_separation_s / snr_series.delta_t))
    peaks, _ = find_peaks(abs_snr, height=threshold, distance=distance)
    times = snr_series.sample_times.numpy()
    return [(float(times[p]), float(abs_snr[p])) for p in peaks]


def search_detector(detector, strain, psd, bank, threshold, crop_s=4):
    triggers = []
    flen = len(strain)
    t0 = time.time()
    for i, (m1, m2) in enumerate(bank):
        try:
            tmpl, tmpl_dur = make_template(m1, m2, strain.delta_t, flen)
            snr = matched_filter(tmpl, strain, psd=psd, low_frequency_cutoff=F_LOWER)
            crop = max(crop_s, tmpl_dur + 2.0)
            snr = snr.crop(crop, crop)
            for t, s in find_triggers(snr, threshold):
                triggers.append({"gps": t, "snr": s, "m1": m1, "m2": m2})
        except Exception as e:
            print(f"    [{detector}] template ({m1:.1f},{m2:.1f}) failed: {e}", flush=True)
        if (i + 1) % 10 == 0 or i == len(bank) - 1:
            print(f"    [{detector}] {i+1}/{len(bank)} templates, {len(triggers)} triggers, {time.time()-t0:.1f}s", flush=True)
    return triggers


def coincidence(h1_trigs, l1_trigs, window_ms=15.0):
    window_s = window_ms / 1000.0
    l1_sorted = sorted(l1_trigs, key=lambda x: x["gps"])
    l1_times = np.array([t["gps"] for t in l1_sorted])
    coincs = []
    for h in h1_trigs:
        lo = np.searchsorted(l1_times, h["gps"] - window_s)
        hi = np.searchsorted(l1_times, h["gps"] + window_s)
        for l in l1_sorted[lo:hi]:
            coincs.append({
                "gps_h1": h["gps"], "gps_l1": l["gps"],
                "dt_ms": (h["gps"] - l["gps"]) * 1000,
                "snr_h1": h["snr"], "snr_l1": l["snr"],
                "snr_network": float(np.sqrt(h["snr"]**2 + l["snr"]**2)),
                "m1_h1": h["m1"], "m2_h1": h["m2"],
                "m1_l1": l["m1"], "m2_l1": l["m2"],
            })
    return coincs


def cluster_coincs(coincs, window_s=1.0):
    """Collapse coincs within window_s into the loudest per cluster."""
    if not coincs:
        return []
    sorted_c = sorted(coincs, key=lambda c: c["gps_h1"])
    clusters = [[sorted_c[0]]]
    for c in sorted_c[1:]:
        if c["gps_h1"] - clusters[-1][-1]["gps_h1"] < window_s:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    return [max(cl, key=lambda c: c["snr_network"]) for cl in clusters]


def all_known_events():
    known = {}
    for source in ["gwtc-1", "gwtc-2", "gwtc-2.1", "gwtc-3"]:
        try:
            cat = Catalog(source=source)
            for name in cat.names:
                try:
                    known[name] = float(cat[name].time)
                except Exception:
                    pass
        except Exception:
            pass
    return known


def cross_check(coincs, known, tolerance_s=2.0):
    for c in coincs:
        t_event = (c["gps_h1"] + c["gps_l1"]) / 2
        match = None
        for name, t in known.items():
            if abs(t - t_event) < tolerance_s:
                match = name
                break
        c["catalog_match"] = match
    return coincs


PHASES = {
    "smoke": dict(
        gps_start=1264315816,    # 5 min before GW200129_065458 (GPS 1264316116)
        duration=600,
        bank=(15, 45, 7),         # 28 templates
        threshold=5.5,
    ),
    "known": dict(
        gps_start=1264314316,    # 30 min before GW200129
        duration=3600,
        bank=(10, 50, 9),         # 45 templates
        threshold=5.5,
    ),
    "blind": dict(
        gps_start=1264298116,    # 5h before GW200129; same observing run, no catalog events in 1h
        duration=3600,
        bank=(10, 50, 9),
        threshold=5.5,
    ),
    "gw250114": dict(
        gps_start=1420877841,    # 5 min before GW250114_082203 (GPS 1420878141.2)
        duration=600,
        bank=(15, 45, 7),         # 28 templates
        threshold=5.5,
    ),
    "gw170817": dict(
        gps_start=1187008370,    # ~8.5 min before GW170817 (GPS 1187008882.4) — BNS, 2017-08-17
        duration=1024,            # power of 2; fits 256s BNS templates @ f_lower=30
        bank=(1.2, 1.7, 6),       # 21 BNS templates in 1.2-1.7 M_sun
        threshold=5.5,
        f_lower=30.0,             # BNS templates are huge at 20 Hz; LIGO BNS analyses use 23-30 Hz
        gate_l1=(1187008881.4, 0.5, 0.25),  # famous L1 glitch 1.1s before merger
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=list(PHASES))
    args = parser.parse_args()
    cfg = PHASES[args.phase]

    if "f_lower" in cfg:
        global F_LOWER
        F_LOWER = cfg["f_lower"]

    bank = build_bank(*cfg["bank"])
    gps_start = cfg["gps_start"]
    duration = cfg["duration"]
    threshold = cfg["threshold"]

    print(f"=== Phase {args.phase} ===")
    print(f"Window:    GPS {gps_start} -> {gps_start+duration}  ({duration}s)")
    print(f"Bank:      {len(bank)} templates, m=[{cfg['bank'][0]},{cfg['bank'][1]}] M_sun")
    print(f"Threshold: |SNR| > {threshold}")
    print(f"Approx:    {APPROXIMANT}\n", flush=True)

    known = all_known_events()
    in_window = {n: t for n, t in known.items() if gps_start <= t < gps_start + duration}
    print(f"Catalog events in window: {len(in_window)}")
    for n, t in sorted(in_window.items(), key=lambda kv: kv[1]):
        print(f"  {n}  at GPS {t:.2f}  (+{t-gps_start:.0f}s)")
    print(flush=True)

    h1 = fetch_strain("H1", gps_start, duration)
    h1_psd = estimate_psd(h1)
    l1 = fetch_strain("L1", gps_start, duration)
    if "gate_l1" in cfg:
        gt, hw, tp = cfg["gate_l1"]
        print(f"  [L1] gating glitch at GPS {gt} (±{hw}s, {tp}s taper)")
        l1 = gate_strain(l1, gt, hw, tp)
    l1_psd = estimate_psd(l1)

    print(f"\nSearching H1...", flush=True)
    h1_trigs = search_detector("H1", h1, h1_psd, bank, threshold)
    print(f"Searching L1...", flush=True)
    l1_trigs = search_detector("L1", l1, l1_psd, bank, threshold)

    print(f"\nH1 triggers: {len(h1_trigs)}   L1 triggers: {len(l1_trigs)}")

    coincs = coincidence(h1_trigs, l1_trigs, window_ms=15)
    print(f"Raw coincident pairs: {len(coincs)}")
    coincs = cluster_coincs(coincs, window_s=1.0)
    print(f"After clustering:     {len(coincs)}")

    coincs = cross_check(coincs, known)
    coincs.sort(key=lambda c: -c["snr_network"])

    summary = {
        "phase": args.phase,
        "window": {"gps_start": gps_start, "duration": duration},
        "bank_size": len(bank),
        "threshold": threshold,
        "known_events_in_window": in_window,
        "h1_triggers": len(h1_trigs),
        "l1_triggers": len(l1_trigs),
        "n_coincident_clusters": len(coincs),
        "top_coincs": coincs[:25],
        "top_h1_triggers": sorted(h1_trigs, key=lambda t: -t["snr"])[:15],
        "top_l1_triggers": sorted(l1_trigs, key=lambda t: -t["snr"])[:15],
    }
    out_json = OUT / f"phase_{args.phase}_results.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"\n=== Top {min(15, len(coincs))} coincident clusters ===")
    print(f"{'#':>3} {'GPS (mid)':>14} {'dt(ms)':>8} {'SNR_H1':>7} {'SNR_L1':>7} {'SNR_net':>8} {'m1':>5} {'m2':>5}  catalog")
    for i, c in enumerate(coincs[:15]):
        cat = c["catalog_match"] or "-"
        t_mid = (c["gps_h1"] + c["gps_l1"]) / 2
        print(f"{i+1:>3} {t_mid:>14.2f} {c['dt_ms']:>+8.2f} {c['snr_h1']:>7.2f} {c['snr_l1']:>7.2f} {c['snr_network']:>8.2f} {c['m1_h1']:>5.1f} {c['m2_h1']:>5.1f}  {cat}")

    print(f"\nResults written to {out_json}")


if __name__ == "__main__":
    main()
