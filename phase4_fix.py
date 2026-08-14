"""
Phase 4: MSS detection + R-measurement (with entry filter + during-trade management).

NEW IN THIS VERSION:
  - Removed:  opposing_smt_during_trade, opposing_smt_mss_during_trade
  - Added:
      opposing_smt_at_entry          bool
      opposing_smt_at_entry_rank     int (0 if none)
      opposing_mss_at_entry          bool
      entry_skipped                  bool
      entry_skip_reason              str: "" / "opp_smt_higher_rank" / "opp_mss_at_entry"
      managed_exit_ts                ts
      managed_exit_reason            str: "tp_hit" / "sl_hit" / "opp_smt_kill" /
                                          "opp_mss_kill" / "smt_invalidation_kill" /
                                          "session_end"
      r_tp1_mgd ... r_tp5_mgd, r_no_tp_mgd
                                     R values when applying at-entry filter
                                     and during-trade kill rules

  ENTRY RULES:
    no opposing alive             -> enter
    opposing rank >= ours alive   -> skip
    opposing rank < ours, no MSS  -> enter
    opposing rank < ours, has MSS -> skip

  DURING-TRADE RULES (only if entered):
    opposing rank >= ours born  -> close at that bar's close (kill)
    opposing rank < ours born   -> hold; if it produces MSS (same strategy)
                                    later -> close at that bar's close (kill)

  Original r_tp1..r_tp5 / r_no_tp columns preserved for comparison.

Usage: python3 phase4.py
"""

import os
import glob
import pandas as pd
import numpy as np

# ============================================================
# USER CONFIG
# ============================================================
DATA_FOLDER = "/Users/voligle/workspace/Trading/Trading_code/clause Backtester"
START_DATE  = "2023-03-01"
END_DATE    = "2025-12-30"
TICK_SIZE   = 0.25
EPS         = 1e-9
# ============================================================

OUT_FOLDER = os.path.join(DATA_FOLDER, "New/out")

HIGH_LEVELS = {"PDH","PWH","LondonHigh","AsiaKZHigh","AsiaONHigh","4H_SwingHigh"}
LOW_LEVELS  = {"PDL","PWL","LondonLow","AsiaKZLow","AsiaONLow","4H_SwingLow"}

LEVEL_RANK = {
    "PWH": 4, "PWL": 4,
    "PDH": 3, "PDL": 3,
    "4H_SwingHigh": 2, "4H_SwingLow": 2,
    "LondonHigh": 1, "LondonLow": 1,
    "AsiaKZHigh": 1, "AsiaKZLow": 1,
    "AsiaONHigh": 1, "AsiaONLow": 1,
}

MEASUREMENT_COLS = [
    "level_id", "smt_birth_ts", "smt_death_ts", "smt_status",
    "level_type", "direction",
    "sweeper", "non_confirmer", "separation_ticks", "opposing_live",
    "session_day",
    "measured_asset", "mss_timeframe", "anchor", "mss_rule",
    "swing_found", "swing_ts", "swing_price",
    "mss_found", "mss_ts", "mss_bar_close", "entry_price", "session",
    "sl_leg_extreme_ts", "sl_leg_extreme_price", "sl_price", "r_points",
    "max_favorable_r", "max_adverse_r",
    "hit_1r", "hit_2r", "hit_3r", "hit_4r", "hit_5r",
    "hit_1r_ts", "hit_2r_ts", "hit_3r_ts", "hit_4r_ts", "hit_5r_ts",
    "sl_hit_ts", "trade_exit_ts_if_held",
    "time_to_mfe_minutes",
    "final_outcome", "final_r",
    "r_tp1", "r_tp2", "r_tp3", "r_tp4", "r_tp5", "r_no_tp",
    "opposing_smt_at_entry", "opposing_smt_at_entry_rank", "opposing_mss_at_entry",
    "entry_skipped", "entry_skip_reason",
    "managed_exit_ts", "managed_exit_reason",
    "r_tp1_mgd", "r_tp2_mgd", "r_tp3_mgd", "r_tp4_mgd", "r_tp5_mgd", "r_no_tp_mgd",
    # NEW: auto-pick (per (level_id, tf, anchor, rule) -> True for the asset whose MSS fires first
    # AND whose entry is not skipped. Used when user selects asset = "Auto" in dashboard.
    "auto_picked",  # bool
    "auto_pick_reason",  # "first_mss" | "fallback_other_skipped" | "" (empty if not picked)
]


def save_csv(df, path, cols):
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


def load_bars(asset, tf):
    path = os.path.join(OUT_FOLDER, f"{asset}_{tf}.csv")
    df = pd.read_csv(path)
    df["ts_ny"] = pd.to_datetime(df["ts_ny"], utc=True).dt.tz_convert("America/New_York")
    return df.set_index("ts_ny").sort_index()[["open","high","low","close"]]


def load_smts_in_range():
    all_smts = []
    for p in glob.glob(os.path.join(OUT_FOLDER, "SMT_events_*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if len(df) == 0:
            continue
        df["birth_ts"] = pd.to_datetime(df["birth_ts"], utc=True).dt.tz_convert("America/New_York")
        df["death_ts"] = pd.to_datetime(df["death_ts"], utc=True).dt.tz_convert("America/New_York")
        all_smts.append(df)
    if not all_smts:
        return pd.DataFrame()
    df = pd.concat(all_smts, ignore_index=True)
    # FIX A: keep BOTH session_end and invalidated_strict SMTs.
    # An invalidated SMT can still have produced a valid MSS BEFORE its death.
    # measure_smt rejects MSSs that fire at or after smt_death_ts.
    df = df[df["status"].isin(["session_end", "invalidated_strict"])]
    start = pd.Timestamp(START_DATE, tz="America/New_York") - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
    end   = pd.Timestamp(END_DATE,   tz="America/New_York") + pd.Timedelta(hours=17)
    df = df[(df["birth_ts"] >= start) & (df["birth_ts"] <= end)]
    df = df.drop_duplicates(
        subset=["birth_ts","sweeper","level_type","level_price_sweeper"]
    ).reset_index(drop=True)
    df["rank"] = df["level_type"].map(LEVEL_RANK).fillna(0).astype(int)
    return df


def session_day_of(birth_ts):
    if birth_ts.hour >= 18:
        return (birth_ts.normalize() + pd.Timedelta(days=1)).date()
    return birth_ts.normalize().date()


def session_end_ts(birth_ts):
    d = session_day_of(birth_ts)
    return pd.Timestamp(d, tz="America/New_York") + pd.Timedelta(hours=17)


def classify_session(ts):
    h = ts.hour; m = ts.minute
    mins = h * 60 + m
    if 2 * 60 <= mins < 5 * 60: return "london"
    if 9 * 60 + 30 <= mins < 12 * 60: return "ny_am"
    if 13 * 60 <= mins < 16 * 60: return "ny_pm"
    return "outside"


# ============================================================
# SWING / MSS / SL
# ============================================================
def detect_swings_on_bars(bars, swing_kind, n=1):
    swings = []
    if len(bars) < 2 * n + 1: return swings
    highs = bars["high"].values; lows = bars["low"].values; times = bars.index
    for i in range(n, len(bars) - n):
        if swing_kind == "High":
            v = highs[i]
            is_pivot = all(v > highs[i - k] for k in range(1, n + 1)) and all(v > highs[i + k] for k in range(1, n + 1))
        else:
            v = lows[i]
            is_pivot = all(v < lows[i - k] for k in range(1, n + 1)) and all(v < lows[i + k] for k in range(1, n + 1))
        if is_pivot:
            swings.append((times[i + n], times[i], v))
    return swings


def pick_before_smt(swings, sweep_ts):
    c = [s for s in swings if s[0] < sweep_ts]
    return c[-1] if c else None


def pick_after_smt(swings, sweep_ts):
    c = [s for s in swings if s[0] > sweep_ts]
    return c[0] if c else None


def find_mss(bars, anchor_confirm_ts, swing_price, direction, mss_rule, session_end):
    sub = bars.loc[anchor_confirm_ts:session_end]
    if len(sub) == 0: return None, None
    sub = sub.iloc[1:] if len(sub) > 1 else sub.iloc[1:0]
    if len(sub) == 0: return None, None
    if direction == "bullish":
        mask = sub["high"] >= swing_price + TICK_SIZE - EPS if mss_rule == "wick_break" else sub["close"] >= swing_price + TICK_SIZE - EPS
    else:
        mask = sub["low"] <= swing_price - TICK_SIZE + EPS if mss_rule == "wick_break" else sub["close"] <= swing_price - TICK_SIZE + EPS
    if not mask.any(): return None, None
    idx = mask.argmax()
    return sub.index[idx], sub.iloc[idx]["close"]


def determine_sl(bars, anchor_confirm_ts, mss_ts, direction):
    sub = bars.loc[anchor_confirm_ts:mss_ts]
    if len(sub) == 0: return None, None, None
    if direction == "bullish":
        idx = sub["low"].argmin()
        return sub.index[idx], sub.iloc[idx]["low"], sub.iloc[idx]["low"] - TICK_SIZE
    idx = sub["high"].argmax()
    return sub.index[idx], sub.iloc[idx]["high"], sub.iloc[idx]["high"] + TICK_SIZE


TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15}


def track_outcome(bars_1m, entry_ts, entry_price, sl_price, direction, session_end, tf):
    """
    Walk 1m bars after the MSS bar CLOSES to compute SL/TP outcomes.

    Realism rules:
      - Walk starts at entry_ts + TF_minutes (= MSS bar close = realistic entry time)
      - Same-bar TP + SL collision: SL wins (conservative; intrabar sequence unknown)
    """
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
    walk_start = entry_ts + pd.Timedelta(minutes=tf_min)  # MSS bar close
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

        # SAME-BAR COLLISION: if SL also hit on this bar, do NOT record TP hits.
        # Intrabar sequence is unknown; conservative rule is SL wins.
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
    result = {}
    sl_hit_ts = outcome.get("sl_hit_ts")
    hit_ts_map = {n: outcome.get(f"hit_{n}r_ts") for n in (1,2,3,4,5)}
    for n in (1, 2, 3, 4, 5):
        tp_ts = hit_ts_map.get(n)
        if tp_ts is None:
            result[f"r_tp{n}"] = -1.0 if sl_hit_ts is not None else final_r
        else:
            if sl_hit_ts is not None and sl_hit_ts < tp_ts:
                result[f"r_tp{n}"] = -1.0
            else:
                result[f"r_tp{n}"] = float(n)
    mfe = outcome.get("max_favorable_r", 0.0)
    result["r_no_tp"] = -1.0 if sl_hit_ts is not None else float(mfe)
    return result


def measure_smt(smt, bars_by_asset_tf, bars_1m_by_asset):
    rows = []
    direction = smt["direction"]
    sweep_ts = smt["birth_ts"]
    smt_death_ts = smt.get("death_ts")
    smt_status = smt.get("status", "session_end")
    end = session_end_ts(sweep_ts)
    swing_kind = "High" if direction == "bullish" else "Low"

    # FIX A: for invalidated SMTs, MSS must fire BEFORE death_ts.
    # Cap the MSS search window at min(session_end, smt_death_ts).
    mss_deadline = end
    if smt_status == "invalidated_strict" and pd.notna(smt_death_ts):
        mss_deadline = min(end, smt_death_ts)

    base_context = {
        "level_id": smt.get("level_id", ""),
        "smt_birth_ts": sweep_ts,
        "smt_death_ts": smt_death_ts,
        "smt_status": smt_status,
        "level_type": smt["level_type"],
        "direction": direction,
        "sweeper": smt["sweeper"],
        "non_confirmer": smt["non_confirmer"],
        "separation_ticks": smt["separation_ticks"],
        "opposing_live": smt.get("opposing_live", False),
        "session_day": session_day_of(sweep_ts),
    }

    for measured_asset in ("MNQ", "MES"):
        bars_1m = bars_1m_by_asset[measured_asset]
        for tf in ("15m", "5m", "1m"):
            bars_tf = bars_by_asset_tf[(measured_asset, tf)]
            slice_start = sweep_ts - pd.Timedelta(days=5)
            bars_tf_slice = bars_tf.loc[slice_start:end]
            swings = detect_swings_on_bars(bars_tf_slice, swing_kind, n=1)

            for anchor in ("before_smt", "after_smt"):
                picked = pick_before_smt(swings, sweep_ts) if anchor == "before_smt" \
                         else pick_after_smt(swings, sweep_ts)

                for mss_rule in ("wick_break", "close_break"):
                    row = {**base_context,
                           "measured_asset": measured_asset,
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

                    # MSS search capped at mss_deadline (= death_ts for invalidated SMTs)
                    mss_ts, mss_close = find_mss(
                        bars_tf_slice, confirm_ts, swing_price, direction, mss_rule, mss_deadline
                    )
                    if mss_ts is None:
                        row["mss_found"] = False
                        if smt_status == "invalidated_strict" and mss_deadline < end:
                            row["final_outcome"] = "no_mss_before_smt_death"
                        else:
                            row["final_outcome"] = "no_mss"
                        rows.append(row); continue

                    # Defensive: MSS must be strictly before death
                    if smt_status == "invalidated_strict" and pd.notna(smt_death_ts):
                        if mss_ts >= smt_death_ts:
                            row["mss_found"] = False
                            row["final_outcome"] = "mss_at_or_after_smt_death"
                            rows.append(row); continue

                    row["mss_found"] = True
                    row["mss_ts"] = mss_ts
                    row["mss_bar_close"] = mss_close
                    row["entry_price"] = mss_close
                    row["session"] = classify_session(mss_ts)

                    ext_ts, ext_price, sl_price = determine_sl(
                        bars_tf_slice, confirm_ts, mss_ts, direction
                    )
                    if sl_price is None:
                        row["final_outcome"] = "no_sl"
                        rows.append(row); continue

                    row["sl_leg_extreme_ts"] = ext_ts
                    row["sl_leg_extreme_price"] = ext_price
                    row["sl_price"] = sl_price
                    row["r_points"] = abs(mss_close - sl_price)

                    outcome = track_outcome(
                        bars_1m, mss_ts, mss_close, sl_price, direction, end, tf
                    )
                    row.update(outcome)
                    row.update(compute_tp_r(outcome, outcome.get("final_r", 0.0)))
                    row["trade_exit_ts_if_held"] = outcome.get("sl_hit_ts") or end

                    rows.append(row)

    return rows


# ============================================================
# PASS 2 — ENTRY FILTER + DURING-TRADE MANAGEMENT
# ============================================================
def find_kill_ts(my_row, my_rank, all_smts, mss_lookup):
    """
    For a trade (row): determine the earliest "kill" timestamp.
    Sources of kills:
      A) Self-SMT invalidation (FIX B): if this trade's own SMT was
         invalidated_strict, exit at the SMT's death_ts (when the non-confirmer
         also crossed the level).
      B) Opposing SMT with rank >= ours: kill at opposing's birth_ts
         (or my entry if it was alive at entry).
      C) Opposing SMT with rank < ours that produces an MSS in same strategy:
         kill at opposing's MSS time.
    Returns: (kill_ts, kill_reason) or (None, None) if no kill.
    """
    my_dir = my_row["direction"]
    my_entry = my_row["mss_ts"]
    my_session_end = session_end_ts(my_entry)

    earliest_kill = None
    earliest_reason = None

    # A) Self-SMT invalidation kill (FIX B)
    smt_status = my_row.get("smt_status", "session_end")
    smt_death_ts = my_row.get("smt_death_ts")
    if smt_status == "invalidated_strict" and pd.notna(smt_death_ts):
        # death must be after entry (MSS-at-or-after-death already rejected in measure_smt)
        # and within session window
        if smt_death_ts > my_entry and smt_death_ts <= my_session_end:
            earliest_kill = smt_death_ts
            earliest_reason = "smt_invalidation_kill"

    # B + C) Opposing SMTs
    opp_dir = "bearish" if my_dir == "bullish" else "bullish"
    opps = all_smts[all_smts["direction"] == opp_dir]
    if len(opps) == 0:
        return earliest_kill, earliest_reason

    relevant = opps[
        (opps["birth_ts"] <= my_session_end)
        & (opps["death_ts"] >= my_entry)
    ]
    if len(relevant) == 0:
        return earliest_kill, earliest_reason

    for _, opp in relevant.iterrows():
        opp_rank = opp["rank"]
        opp_birth = opp["birth_ts"]

        if opp_rank >= my_rank:
            kill_ts = max(opp_birth, my_entry)
            if kill_ts > my_session_end:
                continue
            if earliest_kill is None or kill_ts < earliest_kill:
                earliest_kill = kill_ts
                earliest_reason = "opp_smt_kill"
        else:
            key = (opp["level_id"], my_row["measured_asset"],
                   my_row["mss_timeframe"], my_row["anchor"], my_row["mss_rule"])
            opp_mss_ts = mss_lookup.get(key)
            if opp_mss_ts is None:
                continue
            if opp_mss_ts <= my_entry:
                continue
            if opp_mss_ts > my_session_end:
                continue
            if earliest_kill is None or opp_mss_ts < earliest_kill:
                earliest_kill = opp_mss_ts
                earliest_reason = "opp_mss_kill"

    return earliest_kill, earliest_reason


def compute_entry_skip(my_row, my_rank, all_smts, mss_lookup):
    """
    Determine if entry should be skipped per the entry rules.
    Returns: (skip:bool, reason:str, opp_at_entry:bool, opp_at_entry_rank:int, opp_mss_at_entry:bool)
    """
    my_dir = my_row["direction"]
    my_entry = my_row["mss_ts"]
    opp_dir = "bearish" if my_dir == "bullish" else "bullish"

    opps = all_smts[all_smts["direction"] == opp_dir]
    if len(opps) == 0:
        return False, "", False, 0, False

    # Opposing SMTs alive AT entry
    alive_at_entry = opps[
        (opps["birth_ts"] <= my_entry)
        & (opps["death_ts"] >= my_entry)
    ]
    if len(alive_at_entry) == 0:
        return False, "", False, 0, False

    opp_at_entry = True
    max_rank = int(alive_at_entry["rank"].max())

    # Check rank rule first: any rank >= mine => skip
    if max_rank >= my_rank:
        # Determine if MSS at entry too (for the column flag)
        opp_mss = False
        for _, opp in alive_at_entry.iterrows():
            key = (opp["level_id"], my_row["measured_asset"],
                   my_row["mss_timeframe"], my_row["anchor"], my_row["mss_rule"])
            opp_mss_ts = mss_lookup.get(key)
            if opp_mss_ts is not None and opp_mss_ts <= my_entry:
                opp_mss = True
                break
        return True, "opp_smt_higher_rank", True, max_rank, opp_mss

    # All opposing have rank < mine. Check if any has MSS at or before entry (same strategy).
    opp_mss_at_entry = False
    for _, opp in alive_at_entry.iterrows():
        key = (opp["level_id"], my_row["measured_asset"],
               my_row["mss_timeframe"], my_row["anchor"], my_row["mss_rule"])
        opp_mss_ts = mss_lookup.get(key)
        if opp_mss_ts is not None and opp_mss_ts <= my_entry:
            opp_mss_at_entry = True
            break

    if opp_mss_at_entry:
        return True, "opp_mss_at_entry", True, max_rank, True

    # Lower rank, no MSS yet -> enter
    return False, "", True, max_rank, False


def compute_managed_outcome(my_row, kill_ts, bars_1m_by_asset):
    """
    Compute managed outcome columns for a trade given a possible kill timestamp.
    Mechanics:
      walk 1m bars from entry+1 -> session_end
      track TP hits and SL hits (same as raw outcome)
      if kill_ts is set and we reach it without TP/SL: exit at that bar's CLOSE
    Returns dict with r_tp1_mgd .. r_no_tp_mgd and managed_exit_ts/reason.
    """
    direction = my_row["direction"]
    entry = my_row["entry_price"]
    sl = my_row["sl_price"]
    r_size = abs(entry - sl)
    if r_size < TICK_SIZE - EPS:
        return {f"r_tp{n}_mgd": np.nan for n in (1,2,3,4,5)} | \
               {"r_no_tp_mgd": np.nan, "managed_exit_ts": pd.NaT, "managed_exit_reason": "invalid_r"}

    end = session_end_ts(my_row["mss_ts"])
    bars_1m = bars_1m_by_asset[my_row["measured_asset"]]
    sub = bars_1m.loc[my_row["mss_ts"] + pd.Timedelta(minutes=1):end]

    if len(sub) == 0:
        return {f"r_tp{n}_mgd": 0.0 for n in (1,2,3,4,5)} | \
               {"r_no_tp_mgd": 0.0, "managed_exit_ts": end, "managed_exit_reason": "session_end"}

    # Track per-TP outcomes independently. Each TP "trade" has its own exit decision:
    # whichever of {TP, SL, kill, session_end} comes first.
    # We walk once and record TP/SL/kill timestamps if reached.
    sl_ts = None
    tp_hits = {n: None for n in (1,2,3,4,5)}
    mfe = 0.0
    kill_close_at_kill = None  # close price at kill bar
    exited_by_kill_ts = None

    for ts, bar in sub.iterrows():
        # Kill check first (at the bar containing kill_ts, exit at close of that bar)
        if kill_ts is not None and exited_by_kill_ts is None and ts >= kill_ts:
            exited_by_kill_ts = ts
            kill_close_at_kill = bar["close"]
            # Don't break — we still want to know if SL/TP would have hit FIRST
            # within this same bar. But since we treat kill as exit on this bar's close,
            # and SL/TP intra-bar would be earlier wicks, we have to make a choice.
            # Conservative: if SL would have been hit on a prior bar (already recorded),
            # that takes precedence. If TP/SL hit on the SAME bar as kill, we say kill wins
            # because we're using close price.
            # The walk logic below already records per-bar TP/SL hits BEFORE this point,
            # so we naturally handle "earlier bars".

        if direction == "bullish":
            favorable = (bar["high"] - entry) / r_size
            adverse = (entry - bar["low"]) / r_size
            hit_sl = bar["low"] <= sl + EPS
        else:
            favorable = (entry - bar["low"]) / r_size
            adverse = (bar["high"] - entry) / r_size
            hit_sl = bar["high"] >= sl - EPS

        if favorable > mfe:
            mfe = favorable

        for n in (1,2,3,4,5):
            if tp_hits[n] is None and favorable >= n - EPS:
                tp_hits[n] = ts

        if hit_sl and sl_ts is None:
            sl_ts = ts

        # Stop walking if we have an exit reason set
        if exited_by_kill_ts is not None:
            break
        if sl_ts is not None:
            # SL stop, but keep walking only to record higher TP timestamps for OTHER TP strategies
            # Actually no — once SL hit, the trade is dead for ALL TP strategies that haven't
            # hit yet. So we can break.
            break

    # Compute per-TP managed result
    result = {}
    for n in (1,2,3,4,5):
        tp_ts = tp_hits[n]
        # For this TP strategy, what was the FIRST event among {tp_n, sl, kill, session_end}?
        events = []
        if tp_ts is not None: events.append((tp_ts, "tp", float(n)))
        if sl_ts is not None: events.append((sl_ts, "sl", -1.0))
        if exited_by_kill_ts is not None:
            # R at kill exit = (kill_close - entry) / r_size for bullish, neg for bearish
            if direction == "bullish":
                kill_r = (kill_close_at_kill - entry) / r_size
            else:
                kill_r = (entry - kill_close_at_kill) / r_size
            events.append((exited_by_kill_ts, "kill", kill_r))

        if not events:
            # No exit during walk -> session_end
            last_close = sub.iloc[-1]["close"]
            if direction == "bullish":
                end_r = (last_close - entry) / r_size
            else:
                end_r = (entry - last_close) / r_size
            result[f"r_tp{n}_mgd"] = end_r
        else:
            events.sort(key=lambda x: x[0])
            _, _, r_value = events[0]
            result[f"r_tp{n}_mgd"] = r_value

    # r_no_tp_mgd: same logic but no TP target
    events_no_tp = []
    if sl_ts is not None: events_no_tp.append((sl_ts, "sl", -1.0))
    if exited_by_kill_ts is not None:
        if direction == "bullish":
            kill_r = (kill_close_at_kill - entry) / r_size
        else:
            kill_r = (entry - kill_close_at_kill) / r_size
        events_no_tp.append((exited_by_kill_ts, "kill", kill_r))
    if not events_no_tp:
        result["r_no_tp_mgd"] = float(mfe)  # nothing exited us, use MFE
    else:
        events_no_tp.sort(key=lambda x: x[0])
        _, _, r_value = events_no_tp[0]
        if events_no_tp[0][1] == "sl":
            result["r_no_tp_mgd"] = -1.0
        else:
            result["r_no_tp_mgd"] = r_value

    # Determine overall exit ts/reason — use the EARLIEST event among ANY (prefer reporting
    # the kill if it triggered, otherwise SL, otherwise session_end).
    if exited_by_kill_ts is not None:
        result["managed_exit_ts"] = exited_by_kill_ts
        # Distinguish kill reason: rank-based or MSS-based — passed in via kill_reason
        # We don't have that here cleanly; caller will set this.
        result["managed_exit_reason"] = "kill"  # caller can refine
    elif sl_ts is not None:
        result["managed_exit_ts"] = sl_ts
        result["managed_exit_reason"] = "sl_hit"
    else:
        result["managed_exit_ts"] = end
        result["managed_exit_reason"] = "session_end"

    return result


def fill_entry_and_managed_columns(all_rows_df, all_smts, bars_1m_by_asset):
    n = len(all_rows_df)

    # Build mss lookup
    mss_lookup = {}
    for _, row in all_rows_df.iterrows():
        if row.get("mss_found") is not True:
            continue
        if pd.isna(row.get("mss_ts")):
            continue
        key = (row["level_id"], row["measured_asset"],
               row["mss_timeframe"], row["anchor"], row["mss_rule"])
        mss_lookup[key] = row["mss_ts"]

    # Add ranks to SMTs
    if "rank" not in all_smts.columns:
        all_smts = all_smts.copy()
        all_smts["rank"] = all_smts["level_type"].map(LEVEL_RANK).fillna(0).astype(int)

    # Initialize columns
    out_cols = {
        "opposing_smt_at_entry": [False]*n,
        "opposing_smt_at_entry_rank": [0]*n,
        "opposing_mss_at_entry": [False]*n,
        "entry_skipped": [False]*n,
        "entry_skip_reason": [""]*n,
        "managed_exit_ts": [pd.NaT]*n,
        "managed_exit_reason": [""]*n,
    }
    for tp in ("r_tp1_mgd","r_tp2_mgd","r_tp3_mgd","r_tp4_mgd","r_tp5_mgd","r_no_tp_mgd"):
        out_cols[tp] = [np.nan]*n

    rows_iter = list(all_rows_df.iterrows())
    total_with_mss = sum(1 for _, r in rows_iter if r.get("mss_found") is True and pd.notna(r.get("mss_ts")))
    print(f"  filling entry+managed columns for {total_with_mss:,} executed rows...")

    processed = 0
    for i, (_, row) in enumerate(rows_iter):
        if row.get("mss_found") is not True:
            continue
        if pd.isna(row.get("mss_ts")):
            continue

        my_rank = LEVEL_RANK.get(row["level_type"], 0)

        # Entry filter
        skip, reason, opp_at_entry, opp_rank, opp_mss_at_entry = compute_entry_skip(
            row, my_rank, all_smts, mss_lookup
        )
        out_cols["opposing_smt_at_entry"][i] = opp_at_entry
        out_cols["opposing_smt_at_entry_rank"][i] = opp_rank
        out_cols["opposing_mss_at_entry"][i] = opp_mss_at_entry
        out_cols["entry_skipped"][i] = skip
        out_cols["entry_skip_reason"][i] = reason

        if skip:
            # Managed columns stay NaN — entry was skipped
            processed += 1
            if processed % 1000 == 0:
                print(f"    processed {processed:,}/{total_with_mss:,}")
            continue

        # During-trade kill check
        kill_ts, kill_reason = find_kill_ts(row, my_rank, all_smts, mss_lookup)

        # Compute managed outcomes
        mgd = compute_managed_outcome(row, kill_ts, bars_1m_by_asset)
        for tp in ("r_tp1_mgd","r_tp2_mgd","r_tp3_mgd","r_tp4_mgd","r_tp5_mgd","r_no_tp_mgd"):
            out_cols[tp][i] = mgd[tp]
        out_cols["managed_exit_ts"][i] = mgd["managed_exit_ts"]
        # Refine reason if it was a kill
        if mgd["managed_exit_reason"] == "kill" and kill_reason:
            out_cols["managed_exit_reason"][i] = kill_reason
        else:
            out_cols["managed_exit_reason"][i] = mgd["managed_exit_reason"]

        processed += 1
        if processed % 1000 == 0:
            print(f"    processed {processed:,}/{total_with_mss:,}")

    for c, vals in out_cols.items():
        all_rows_df[c] = vals
    return all_rows_df


def fill_auto_pick_columns(df):
    """
    Pass 3: For each (level_id, mss_timeframe, anchor, mss_rule) group of two rows
    (one per asset MNQ/MES), pick the asset whose MSS fired first AND whose entry
    is not skipped. Falls through to the other asset if the first-MSS one is skipped.

    Marks chosen row with auto_picked=True, the other with False.
    Rows with no MSS get auto_picked=False (no entry to take).
    """
    n = len(df)
    auto_picked = [False] * n
    auto_reason = [""] * n

    # Group by (level_id, mss_timeframe, anchor, mss_rule)
    grp_cols = ["level_id", "mss_timeframe", "anchor", "mss_rule"]

    # Index for fast lookup
    df_indexed = df.reset_index().rename(columns={"index": "_orig_idx"})

    for keys, grp in df_indexed.groupby(grp_cols):
        # grp typically has 1 or 2 rows (one per asset)
        # Filter to rows that actually have MSS
        with_mss = grp[grp["mss_found"] == True]
        if len(with_mss) == 0:
            continue
        # Filter to rows that are not skipped at entry
        candidates = with_mss[with_mss["entry_skipped"] != True].copy()

        if len(candidates) == 0:
            # All possible entries skipped — no auto pick
            continue

        # Pick the row with earliest mss_ts. If tie, prefer non-confirmer side.
        # Sort by (mss_ts asc, then sweeper position so non-confirmer wins)
        candidates["_is_nonconfirmer"] = (candidates["measured_asset"] != candidates["sweeper"])
        # We want non-confirmer FIRST when ties — so sort DESCENDING on _is_nonconfirmer (True first)
        candidates = candidates.sort_values(
            by=["mss_ts", "_is_nonconfirmer"],
            ascending=[True, False],
        )

        winner_idx = candidates.iloc[0]["_orig_idx"]
        # Determine reason: if there were skipped rows that fired EARLIER, this is fallback
        all_with_mss_sorted = with_mss.sort_values("mss_ts")
        earliest_idx = all_with_mss_sorted.iloc[0]["_orig_idx"]
        if winner_idx == earliest_idx:
            reason = "first_mss"
        else:
            reason = "fallback_other_skipped"

        auto_picked[winner_idx] = True
        auto_reason[winner_idx] = reason

    df["auto_picked"] = auto_picked
    df["auto_pick_reason"] = auto_reason
    return df


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Phase 4 MSS measurement: {START_DATE} -> {END_DATE}")
    smts = load_smts_in_range()
    print(f"SMTs in range: {len(smts):,}")
    if len(smts) == 0:
        print("No SMTs. Run Phase 3 first for the full range.")
        return

    print("\nLoading bar data...")
    bars_by_asset_tf = {}
    bars_1m_by_asset = {}
    for asset in ("MNQ", "MES"):
        bars_1m_by_asset[asset] = load_bars(asset, "1m")
        for tf in ("15m", "5m", "1m"):
            bars_by_asset_tf[(asset, tf)] = load_bars(asset, tf)

    print(f"\nPass 1: computing 24 measurements for {len(smts):,} SMTs...")
    all_rows = []
    for i, (_, smt) in enumerate(smts.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(smts)}")
        all_rows.extend(measure_smt(smt, bars_by_asset_tf, bars_1m_by_asset))

    df = pd.DataFrame(all_rows)
    print(f"Pass 1 produced {len(df):,} rows")

    print("\nPass 2: entry filter + during-trade management...")
    df = fill_entry_and_managed_columns(df, smts, bars_1m_by_asset)
    print("Done.")

    print("\nPass 3: auto-pick asset (first MSS, fallback if skipped)...")
    df = fill_auto_pick_columns(df)
    n_auto = int(df["auto_picked"].sum())
    print(f"  Auto-picked rows: {n_auto:,}")
    if "auto_pick_reason" in df.columns:
        reasons = df[df["auto_picked"] == True]["auto_pick_reason"].value_counts()
        for r, c in reasons.items():
            print(f"    {r}: {c:,}")
    print("Done.")

    print(f"\nTotal rows: {len(df):,}")
    if len(df):
        print("\nBy final_outcome:")
        print(df["final_outcome"].value_counts().to_string())
        valid = df[df["final_outcome"].isin(["sl_hit","session_end"])]
        print(f"\nValid trades: {len(valid):,}")
        if len(valid):
            kept = valid[valid["entry_skipped"] == False]
            skipped = valid[valid["entry_skipped"] == True]
            print(f"  Entry kept: {len(kept):,} ({len(kept)/len(valid)*100:.1f}%)")
            print(f"  Entry skipped: {len(skipped):,} ({len(skipped)/len(valid)*100:.1f}%)")
            if len(kept):
                print("\nManaged exits among kept:")
                print(kept["managed_exit_reason"].value_counts().to_string())
                for strat in ("r_tp2","r_tp2_mgd","r_tp3","r_tp3_mgd"):
                    if strat in kept.columns:
                        m = kept[strat].mean()
                        wr = (kept[strat] > 0).sum() / len(kept) * 100
                        print(f"  {strat}: mean {m:.2f}R, win {wr:.1f}%")

    out = os.path.join(OUT_FOLDER, f"SMT_measurements_{START_DATE}_{END_DATE}.csv")
    save_csv(df, out, MEASUREMENT_COLS)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()