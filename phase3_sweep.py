"""
phase3_sweep.py — per-asset key level sweep detection.

This is the non-SMT analog of phase3.py. For each asset (MNQ, MES) independently:
  - For every alive key level
  - Find the FIRST 1m bar that strictly crosses it (by >= 1 tick)
  - Record a sweep event with direction inferred from the level type

Direction rules:
  HIGH level types (PDH, PWH, LondonHigh, AsiaKZHigh, AsiaONHigh, 4H_SwingHigh)
    → sweep is BEARISH (price reversed expected after taking liquidity above)
  LOW level types (PDL, PWL, LondonLow, AsiaKZLow, AsiaONLow, 4H_SwingLow)
    → sweep is BULLISH

A level can be swept ONCE per its alive lifetime. After it's swept, it's "dead"
(per Phase 2's first_touch_ts which already records this).

Output: SWEEP_events_<start>_<end>_<MNQ|MES>.csv

Run:
  python3 phase3_sweep.py
"""

import os
import pandas as pd
import numpy as np

# ============================================================
# USER CONFIG
# ============================================================
DATA_FOLDER = "/Users/voligle/workspace/Trading/Trading_code/clause Backtester"
START_DATE  = "2025-03-01"
END_DATE    = "2025-12-30"
TICK_SIZE   = 0.25
EPS         = 1e-9
# ============================================================

OUT_FOLDER = os.path.join(DATA_FOLDER, "New/out")

HIGH_LEVELS = {"PDH", "PWH", "LondonHigh", "AsiaKZHigh", "AsiaONHigh", "4H_SwingHigh"}
LOW_LEVELS  = {"PDL", "PWL", "LondonLow",  "AsiaKZLow",  "AsiaONLow",  "4H_SwingLow"}
ALL_LEVELS  = HIGH_LEVELS | LOW_LEVELS

LEVEL_RANK = {
    "PWH": 4, "PWL": 4,
    "PDH": 3, "PDL": 3,
    "4H_SwingHigh": 2, "4H_SwingLow": 2,
    "LondonHigh": 1, "LondonLow": 1,
    "AsiaKZHigh": 1, "AsiaKZLow": 1,
    "AsiaONHigh": 1, "AsiaONLow": 1,
}

SWEEP_COLS = [
    "level_id", "birth_ts",
    "level_type", "direction", "asset",
    "level_price", "sweep_price", "sweep_overshoot_ticks",
    "level_age_minutes", "level_rank",
]


def load_levels(asset):
    path = os.path.join(OUT_FOLDER, f"{asset}_levels.csv")
    df = pd.read_csv(path)
    for c in ("ts_formed", "valid_from", "valid_until"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    if "first_touch_ts" in df.columns:
        df["first_touch_ts"] = pd.to_datetime(df["first_touch_ts"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    df = df[df["level_type"].isin(ALL_LEVELS)]
    return df.reset_index(drop=True)


def load_1m_bars(asset):
    path = os.path.join(OUT_FOLDER, f"{asset}_1m.csv")
    d = pd.read_csv(path)
    d["ts_ny"] = pd.to_datetime(d["ts_ny"], utc=True).dt.tz_convert("America/New_York")
    return d.set_index("ts_ny").sort_index()[["open","high","low","close"]]


def find_first_strict_cross(bars_1m, level_price, level_type, valid_from, valid_until):
    """For a given level, find the FIRST 1m bar within [valid_from, valid_until]
    that strictly crosses it (by >= 1 tick).

    Returns (birth_ts, sweep_price, overshoot_ticks) or (None, None, None).
    """
    sub = bars_1m.loc[valid_from:valid_until]
    if len(sub) == 0:
        return None, None, None

    if level_type in HIGH_LEVELS:
        # Bullish-pierce of a high level (price goes ABOVE)
        threshold = level_price + TICK_SIZE
        mask = sub["high"] >= threshold - EPS
        if not mask.any():
            return None, None, None
        first = mask.idxmax()
        sweep_high = float(sub.loc[first]["high"])
        overshoot_ticks = (sweep_high - level_price) / TICK_SIZE
        return first, sweep_high, overshoot_ticks
    else:
        # Bearish-pierce of a low level (price goes BELOW)
        threshold = level_price - TICK_SIZE
        mask = sub["low"] <= threshold + EPS
        if not mask.any():
            return None, None, None
        first = mask.idxmax()
        sweep_low = float(sub.loc[first]["low"])
        overshoot_ticks = (level_price - sweep_low) / TICK_SIZE
        return first, sweep_low, overshoot_ticks


def detect_sweeps_for_asset(asset, levels, bars_1m, win_start, win_end):
    """Return list of sweep event dicts for this asset."""
    events = []
    # Only consider levels with valid_from in or before our window AND valid_until after window start
    alive = levels[(levels["valid_from"] < win_end) & (levels["valid_until"] > win_start)].copy()

    for _, lv in alive.iterrows():
        level_type = lv["level_type"]
        vf = lv["valid_from"]
        vu = lv["valid_until"]
        # Limit scan to our window (no point scanning beyond)
        scan_start = max(vf, win_start)
        scan_end = min(vu, win_end)
        if scan_end <= scan_start:
            continue

        birth_ts, sweep_price, overshoot = find_first_strict_cross(
            bars_1m, float(lv["price"]), level_type, scan_start, scan_end
        )
        if birth_ts is None:
            continue

        # Direction: HIGH sweep → bearish, LOW sweep → bullish
        direction = "bearish" if level_type in HIGH_LEVELS else "bullish"
        age_minutes = int((birth_ts - vf).total_seconds() / 60)

        events.append({
            "level_id": lv["level_id"],
            "birth_ts": birth_ts,
            "level_type": level_type,
            "direction": direction,
            "asset": asset,
            "level_price": float(lv["price"]),
            "sweep_price": sweep_price,
            "sweep_overshoot_ticks": float(overshoot),
            "level_age_minutes": age_minutes,
            "level_rank": LEVEL_RANK.get(level_type, 0),
        })
    return events


def save_csv(df, path):
    cols = SWEEP_COLS
    if len(df) == 0:
        for c in cols:
            if c not in df.columns:
                df[c] = pd.Series(dtype="object")
        df = df[cols]
    else:
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[cols + [c for c in df.columns if c not in cols]]
    df.to_csv(path, index=False)


def main():
    win_start = pd.Timestamp(START_DATE, tz="America/New_York") - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
    win_end   = pd.Timestamp(END_DATE,   tz="America/New_York") + pd.Timedelta(hours=17)
    print(f"Window: {win_start} → {win_end}")
    print(f"Output folder: {OUT_FOLDER}\n")

    for asset in ("MNQ", "MES"):
        print(f"Processing {asset}...")
        levels = load_levels(asset)
        print(f"  Levels: {len(levels):,}")
        bars_1m = load_1m_bars(asset)
        print(f"  1m bars: {len(bars_1m):,}")

        events = detect_sweeps_for_asset(asset, levels, bars_1m, win_start, win_end)
        print(f"  Sweeps detected: {len(events):,}")

        df = pd.DataFrame(events)
        if len(df):
            df = df.sort_values("birth_ts").reset_index(drop=True)
            # Per-level-type breakdown
            print("  By level type:")
            for lt, n in df["level_type"].value_counts().items():
                print(f"    {lt:>14}: {n}")
        out_path = os.path.join(OUT_FOLDER, f"SWEEP_events_{START_DATE}_{END_DATE}_{asset}.csv")
        save_csv(df, out_path)
        print(f"  → {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()