"""
phase4_sweep.py — measure MSS outcomes for each per-asset sweep event.

For each sweep event from phase3_sweep:
  Asset = the one that did the sweep
  Direction = bullish (LOW sweep) / bearish (HIGH sweep)
  Birth time = the sweep timestamp

For each (TF in 1m/5m/15m, anchor in before_smt/after_smt, mss_rule in wick/close):
  - Detect swings in the sweeper asset's bars
  - Pick anchor swing
  - Find MSS (break of swing in trade direction)
  - Walk 1m bars from MSS bar close to session end:
      - SL = swing low/high (sl_leg_extreme)
      - TP1..TP5 = entry ± n*R
      - First touch decides outcome (same-bar TP+SL → SL wins)

Realism rules (same as phase4.py):
  - MSS bar must CLOSE before we walk for SL/TP
  - Same-bar SL+TP → SL wins (conservative)
  - Walk starts at MSS_ts + TF_minutes

No opposing kills, no SMT invalidation, no auto-pick:
  - Each sweep produces 1 trade row per (TF × anchor × rule)
  - Managed/entry-skipped fields not used (no opposing logic)

Output: SWEEP_measurements_<start>_<end>.csv

Run:
  python3 phase4_sweep.py
"""

import os
import glob
import pandas as pd
import numpy as np

# ============================================================
# USER CONFIG
# ============================================================
DATA_FOLDER = "/Users/voligle/workspace/Trading/Trading_code/clause Backtester"
START_DATE  = "2025-03-01"
END_DATE    = "2025-05-30"
TICK_SIZE   = 0.25
EPS         = 1e-9
# ============================================================

OUT_FOLDER = os.path.join(DATA_FOLDER, "New/out")

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15}
TFS = ["1m", "5m", "15m"]
ANCHORS = ["before_smt", "after_smt"]
MSS_RULES = ["wick_break", "close_break"]

HIGH_LEVELS = {"PDH","PWH","LondonHigh","AsiaKZHigh","AsiaONHigh","4H_SwingHigh"}
LOW_LEVELS  = {"PDL","PWL","LondonLow","AsiaKZLow","AsiaONLow","4H_SwingLow"}

MEASUREMENT_COLS = [
    "level_id", "smt_birth_ts", "level_type", "direction", "asset",
    "level_price", "sweep_price", "sweep_overshoot_ticks",
    "measured_asset", "mss_timeframe", "anchor", "mss_rule",
    "swing_found", "swing_ts", "swing_price",
    "mss_found", "mss_ts", "mss_bar_close",
    "entry_price", "sl_price", "r_points",
    "max_favorable_r", "max_adverse_r",
    "hit_1r", "hit_2r", "hit_3r", "hit_4r", "hit_5r",
    "hit_1r_ts", "hit_2r_ts", "hit_3r_ts", "hit_4r_ts", "hit_5r_ts",
    "sl_hit_ts", "time_to_mfe_minutes", "session",
    "final_outcome", "final_r",
    "r_tp1", "r_tp2", "r_tp3", "r_tp4", "r_tp5", "r_no_tp",
    # Managed columns kept as raw equivalents (no opposing kill in sweep pipeline)
    "r_tp1_mgd", "r_tp2_mgd", "r_tp3_mgd", "r_tp4_mgd", "r_tp5_mgd", "r_no_tp_mgd",
    "managed_exit_ts", "managed_exit_reason",
    "entry_skipped", "entry_skip_reason",
    "auto_picked", "auto_pick_reason",
    "trade_exit_ts_if_held",
]


def session_end_ts(ts):
    if ts.hour >= 18:
        return ts.normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=17)
    return ts.normalize() + pd.Timedelta(hours=17)


def session_day_of(ts):
    if ts.hour >= 18:
        return (ts.normalize() + pd.Timedelta(days=1)).date()
    return ts.normalize().date()


def classify_session(ts):
    h = ts.hour; m = ts.minute
    mins = h * 60 + m
    if 2 * 60 <= mins < 5 * 60: return "london"
    if 9 * 60 + 30 <= mins < 12 * 60: return "ny_am"
    if 13 * 60 <= mins < 16 * 60: return "ny_pm"
    return "outside"


def detect_swings_on_bars(bars, swing_kind, n=1):
    """Detect swing highs/lows. Returns list of (confirm_ts, pivot_ts, price)."""
    swings = []
    if swing_kind == "High":
        for i in range(n, len(bars) - n):
            pivot = bars.iloc[i]["high"]
            left = bars.iloc[i-n:i]["high"]
            right = bars.iloc[i+1:i+1+n]["high"]
            if (left < pivot - EPS).all() and (right < pivot - EPS).all():
                pivot_ts = bars.index[i]
                confirm_ts = bars.index[i+n]
                swings.append((confirm_ts, pivot_ts, float(pivot)))
    else:
        for i in range(n, len(bars) - n):
            pivot = bars.iloc[i]["low"]
            left = bars.iloc[i-n:i]["low"]
            right = bars.iloc[i+1:i+1+n]["low"]
            if (left > pivot + EPS).all() and (right > pivot + EPS).all():
                pivot_ts = bars.index[i]
                confirm_ts = bars.index[i+n]
                swings.append((confirm_ts, pivot_ts, float(pivot)))
    return swings


def pick_before_smt(swings, sweep_ts):
    """Last swing confirmed BEFORE the sweep."""
    candidates = [s for s in swings if s[0] < sweep_ts]
    if not candidates:
        return None
    return candidates[-1]


def pick_after_smt(swings, sweep_ts):
    """First swing pivot AT or AFTER sweep_ts (confirm can be slightly after)."""
    candidates = [s for s in swings if s[1] >= sweep_ts]
    if not candidates:
        return None
    return candidates[0]


def find_mss(bars, anchor_confirm_ts, swing_price, direction, mss_rule, session_end):
    """Find the first bar that breaks the swing in the trade direction.
    Returns (mss_ts, mss_close) or (None, None).
    """
    sub = bars.loc[anchor_confirm_ts:session_end]
    if len(sub) <= 1:
        return None, None
    # Skip the confirm bar itself; look at bars after it
    sub = sub.iloc[1:]
    if mss_rule == "wick_break":
        if direction == "bullish":
            mask = sub["high"] >= swing_price - EPS
        else:
            mask = sub["low"] <= swing_price + EPS
    else:  # close_break
        if direction == "bullish":
            mask = sub["close"] >= swing_price - EPS
        else:
            mask = sub["close"] <= swing_price + EPS
    if not mask.any():
        return None, None
    first = mask.idxmax()
    return first, float(sub.loc[first]["close"])


def track_outcome(bars_1m, entry_ts, entry_price, sl_price, direction, session_end, tf):
    """Walk 1m bars after MSS bar closes. Same-bar SL+TP → SL wins."""
    out = {
        "max_favorable_r": 0.0, "max_adverse_r": 0.0,
        "hit_1r": False, "hit_2r": False, "hit_3r": False, "hit_4r": False, "hit_5r": False,
        "hit_1r_ts": None, "hit_2r_ts": None, "hit_3r_ts": None,
        "hit_4r_ts": None, "hit_5r_ts": None,
        "sl_hit_ts": None,
        "time_to_mfe_minutes": None,
        "final_outcome": None, "final_r": None,
    }
    r_size = abs(entry_price - sl_price)
    if r_size < TICK_SIZE - EPS:
        out["final_outcome"] = "invalid_r"
        return out

    tf_min = TF_MINUTES.get(tf, 1)
    walk_start = entry_ts + pd.Timedelta(minutes=tf_min)
    sub = bars_1m.loc[walk_start:session_end]
    if len(sub) == 0:
        out["final_outcome"] = "session_end"
        out["final_r"] = 0.0
        return out

    mfe_r = 0.0; mae_r = 0.0; mfe_ts = entry_ts
    for ts, bar in sub.iterrows():
        if direction == "bullish":
            favorable = (bar["high"] - entry_price) / r_size
            adverse = (entry_price - bar["low"]) / r_size
            hit_sl = bar["low"] <= sl_price + EPS
        else:
            favorable = (entry_price - bar["low"]) / r_size
            adverse = (bar["high"] - entry_price) / r_size
            hit_sl = bar["high"] >= sl_price - EPS

        if not hit_sl:
            for lvl in (1, 2, 3, 4, 5):
                key = f"hit_{lvl}r"
                if not out[key] and favorable >= lvl - EPS:
                    out[key] = True
                    out[f"hit_{lvl}r_ts"] = ts

        if favorable > mfe_r:
            mfe_r = favorable; mfe_ts = ts
        if adverse > mae_r:
            mae_r = adverse

        if hit_sl:
            out["sl_hit_ts"] = ts
            out["max_favorable_r"] = mfe_r
            out["max_adverse_r"] = max(mae_r, 1.0)
            out["final_outcome"] = "sl_hit"
            out["final_r"] = -1.0
            out["time_to_mfe_minutes"] = int((mfe_ts - entry_ts).total_seconds() / 60)
            return out

    last_close = sub.iloc[-1]["close"]
    if direction == "bullish":
        final_r = (last_close - entry_price) / r_size
    else:
        final_r = (entry_price - last_close) / r_size
    out["max_favorable_r"] = mfe_r
    out["max_adverse_r"] = mae_r
    out["final_outcome"] = "session_end"
    out["final_r"] = final_r
    out["time_to_mfe_minutes"] = int((mfe_ts - entry_ts).total_seconds() / 60)
    return out


def compute_tp_r(outcome, final_r):
    """Apply TP levels: trade exits at +Nr if hit, else final_r at session end / SL."""
    res = {}
    for n in (1, 2, 3, 4, 5):
        if outcome["final_outcome"] == "sl_hit":
            res[f"r_tp{n}"] = -1.0
        elif outcome.get(f"hit_{n}r"):
            res[f"r_tp{n}"] = float(n)
        else:
            res[f"r_tp{n}"] = final_r
    res["r_no_tp"] = final_r
    return res


def load_sweeps():
    files = sorted(glob.glob(os.path.join(OUT_FOLDER, "SWEEP_events_*.csv")))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        if len(d) == 0: continue
        d["birth_ts"] = pd.to_datetime(d["birth_ts"], utc=True).dt.tz_convert("America/New_York")
        dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    start = pd.Timestamp(START_DATE, tz="America/New_York") - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
    end   = pd.Timestamp(END_DATE,   tz="America/New_York") + pd.Timedelta(hours=17)
    df = df[(df["birth_ts"] >= start) & (df["birth_ts"] <= end)]
    df = df.drop_duplicates(subset=["asset","birth_ts","level_type","level_price"]).reset_index(drop=True)
    df = df.sort_values("birth_ts").reset_index(drop=True)
    return df


def load_bars(asset, tf):
    path = os.path.join(OUT_FOLDER, f"{asset}_{tf}.csv")
    df = pd.read_csv(path)
    df["ts_ny"] = pd.to_datetime(df["ts_ny"], utc=True).dt.tz_convert("America/New_York")
    return df.set_index("ts_ny").sort_index()[["open","high","low","close"]]


def measure_sweep(sweep, bars_by_asset_tf, bars_1m_by_asset):
    rows = []
    asset = sweep["asset"]
    direction = sweep["direction"]
    sweep_ts = sweep["birth_ts"]
    end = session_end_ts(sweep_ts)
    swing_kind = "High" if direction == "bullish" else "Low"

    base = {
        "level_id": sweep["level_id"],
        "smt_birth_ts": sweep_ts,
        "level_type": sweep["level_type"],
        "direction": direction,
        "asset": asset,
        "level_price": float(sweep["level_price"]),
        "sweep_price": float(sweep["sweep_price"]),
        "sweep_overshoot_ticks": float(sweep.get("sweep_overshoot_ticks", 0)),
    }

    bars_1m = bars_1m_by_asset[asset]

    for tf in TFS:
        bars_tf = bars_by_asset_tf[(asset, tf)]
        slice_start = sweep_ts - pd.Timedelta(days=5)
        bars_tf_slice = bars_tf.loc[slice_start:end]
        swings = detect_swings_on_bars(bars_tf_slice, swing_kind, n=1)

        for anchor in ANCHORS:
            picked = pick_before_smt(swings, sweep_ts) if anchor == "before_smt" \
                     else pick_after_smt(swings, sweep_ts)

            for mss_rule in MSS_RULES:
                row = {**base,
                       "measured_asset": asset,
                       "mss_timeframe": tf,
                       "anchor": anchor,
                       "mss_rule": mss_rule,
                       "swing_found": picked is not None}

                if picked is None:
                    row["final_outcome"] = "no_swing"
                    rows.append(row); continue

                confirm_ts, pivot_ts, swing_price = picked
                row["swing_ts"] = pivot_ts
                row["swing_price"] = swing_price

                mss_ts, mss_close = find_mss(
                    bars_tf_slice, confirm_ts, swing_price, direction, mss_rule, end
                )
                if mss_ts is None:
                    row["mss_found"] = False
                    row["final_outcome"] = "no_mss"
                    rows.append(row); continue

                row["mss_found"] = True
                row["mss_ts"] = mss_ts
                row["mss_bar_close"] = mss_close
                row["entry_price"] = mss_close
                row["session"] = classify_session(mss_ts)

                # SL = swing pivot extreme (the level that defined the leg)
                # For bullish: SL = the swing low BEFORE the broken swing high
                # For bearish: SL = the swing high BEFORE the broken swing low
                # In before_smt anchoring, the swing being broken IS the anchor itself
                # (sweep + reversal + break of swing in opposite direction)
                # Wait: re-think. anchor_swing_price is what we BREAK. SL should be the
                # extreme that formed the leg toward the break.
                #
                # Simplest: for bullish trade (break of a swing HIGH from below),
                #   the leg low is the swing low BEFORE the swing high that got broken.
                #   We use the LOW of bars between [confirm_ts..mss_ts] as the SL.
                # For bearish: HIGH of bars between [confirm_ts..mss_ts].
                leg_bars = bars_tf_slice.loc[confirm_ts:mss_ts]
                if direction == "bullish":
                    sl_price = float(leg_bars["low"].min())
                else:
                    sl_price = float(leg_bars["high"].max())
                row["sl_price"] = sl_price
                row["r_points"] = abs(mss_close - sl_price)

                if abs(mss_close - sl_price) < TICK_SIZE - EPS:
                    row["final_outcome"] = "invalid_r"
                    rows.append(row); continue

                outcome = track_outcome(
                    bars_1m, mss_ts, mss_close, sl_price, direction, end, tf
                )
                row.update(outcome)
                row.update(compute_tp_r(outcome, outcome.get("final_r", 0.0)))
                row["trade_exit_ts_if_held"] = outcome.get("sl_hit_ts") or end

                # No managed mode in sweep pipeline → managed values = raw values
                for n in (1, 2, 3, 4, 5):
                    row[f"r_tp{n}_mgd"] = row.get(f"r_tp{n}")
                row["r_no_tp_mgd"] = row.get("r_no_tp")
                row["managed_exit_ts"] = row["trade_exit_ts_if_held"]
                row["managed_exit_reason"] = outcome["final_outcome"]
                row["entry_skipped"] = False
                row["entry_skip_reason"] = ""
                row["auto_picked"] = True   # single-asset pipeline → always picked
                row["auto_pick_reason"] = "single_asset"

                rows.append(row)

    return rows


def save_csv(df, path):
    cols = MEASUREMENT_COLS
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
    print(f"Window: {START_DATE} → {END_DATE}")
    print(f"Output folder: {OUT_FOLDER}")

    sweeps = load_sweeps()
    print(f"Sweep events loaded: {len(sweeps):,}")
    if len(sweeps) == 0:
        print("Nothing to do. Run phase3_sweep.py first.")
        return

    print("Loading bars for both assets and all TFs...")
    bars_by_asset_tf = {}
    bars_1m_by_asset = {}
    for asset in ("MNQ", "MES"):
        for tf in TFS:
            bars_by_asset_tf[(asset, tf)] = load_bars(asset, tf)
        bars_1m_by_asset[asset] = bars_by_asset_tf[(asset, "1m")]
        print(f"  {asset}: 1m={len(bars_1m_by_asset[asset]):,}")

    print("\nMeasuring outcomes per sweep...")
    all_rows = []
    every_n = max(1, len(sweeps) // 20)
    for i, (_, sweep) in enumerate(sweeps.iterrows(), 1):
        all_rows.extend(measure_sweep(sweep, bars_by_asset_tf, bars_1m_by_asset))
        if i % every_n == 0 or i == len(sweeps):
            print(f"  [{i:>5}/{len(sweeps)}] {i/len(sweeps)*100:5.1f}%")

    df = pd.DataFrame(all_rows)
    print(f"\nTotal measurement rows: {len(df):,}")
    if len(df):
        # Summary breakdown
        print(f"\nBy final_outcome:")
        print(df["final_outcome"].value_counts().to_string())
        print(f"\nBy measured_asset × TF × anchor (mss_found=True):")
        sub = df[df["mss_found"] == True]
        if len(sub):
            cnt = sub.groupby(["measured_asset","mss_timeframe","anchor"]).size().to_frame("n").reset_index()
            print(cnt.to_string(index=False))

    out_path = os.path.join(OUT_FOLDER, f"SWEEP_measurements_{START_DATE}_{END_DATE}.csv")
    save_csv(df, out_path)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()