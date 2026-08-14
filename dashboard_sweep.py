"""
SMT Strategy Dashboard.

- Top filter bar: asset, TF, anchor, rule, TP, session, date range + Apply button
- Summary metrics: Total PnL $, Account Balance, Win Rate, Total Trades, Breakeven
- Equity curve chart
- RR metrics: Avg RR, Max RR, Ideal Avg RR, Max Ideal RR
- Expectancy + Profit Factor
- Winners / Losers panels
- Performance by side (bullish / bearish donuts)
- Monthly calendar of trading days with $ PnL and trade counts
- Click a calendar day: list of trades
- Click a trade: modal with chart + position tool

Usage: streamlit run dashboard.py --server.port 8505
"""

import os
import glob
import io
import zipfile
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIG
# ============================================================
DATA_FOLDER = "/Users/voligle/workspace/Trading/Trading_code/clause Backtester"
OUT_FOLDER = os.path.join(DATA_FOLDER, "New/out")
TICK_SIZE = 0.25
EPS = 1e-9

STARTING_BALANCE = 25000.0
RISK_PER_TRADE = 100.0

SHORT_LABEL = {
    "PDH": "PDH", "PDL": "PDL", "PWH": "PWH", "PWL": "PWL",
    "LondonHigh": "LH", "LondonLow": "LL",
    "AsiaKZHigh": "AKH", "AsiaKZLow": "AKL",
    "AsiaONHigh": "AOH", "AsiaONLow": "AOL",
    "4H_SwingHigh": "SWH", "4H_SwingLow": "SWL",
}
HIGH_LEVELS = {"PDH","PWH","LondonHigh","AsiaKZHigh","AsiaONHigh","4H_SwingHigh"}
LOW_LEVELS  = {"PDL","PWL","LondonLow","AsiaKZLow","AsiaONLow","4H_SwingLow"}
ALL_LEVELS  = HIGH_LEVELS | LOW_LEVELS
PERSISTENT_TYPES = {"PDH","PDL","PWH","PWL","4H_SwingHigh","4H_SwingLow"}

TP_COL = {
    "no_TP": "r_no_tp", "TP1": "r_tp1", "TP2": "r_tp2",
    "TP3": "r_tp3", "TP4": "r_tp4", "TP5": "r_tp5",
}

st.set_page_config(page_title="Sweep Dashboard", layout="wide")

# ============================================================
# CACHED LOADERS
# ============================================================
@st.cache_data
def load_bars(asset, tf):
    path = os.path.join(OUT_FOLDER, f"{asset}_{tf}.csv")
    df = pd.read_csv(path)
    df["ts_ny"] = pd.to_datetime(df["ts_ny"], utc=True).dt.tz_convert("America/New_York")
    return df.set_index("ts_ny").sort_index()[["open","high","low","close"]]


@st.cache_data
def load_levels(asset):
    path = os.path.join(OUT_FOLDER, f"{asset}_levels.csv")
    df = pd.read_csv(path)
    for c in ("ts_formed","valid_from","valid_until"):
        df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert("America/New_York")
    if "first_touch_ts" in df.columns:
        df["first_touch_ts"] = pd.to_datetime(df["first_touch_ts"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    df = df[df["level_type"].isin(ALL_LEVELS)]
    return df.reset_index(drop=True)


@st.cache_data
def load_measurements(path, mtime):
    try:
        if os.path.getsize(path) == 0:
            return pd.DataFrame()
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if len(df) == 0:
        return df
    ts_cols = ["smt_birth_ts","swing_ts","mss_ts","sl_leg_extreme_ts",
                "hit_1r_ts","hit_2r_ts","hit_3r_ts","hit_4r_ts","hit_5r_ts",
                "sl_hit_ts","trade_exit_ts_if_held","managed_exit_ts"]
    for c in ts_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    if "session_day" in df.columns:
        df["session_day"] = pd.to_datetime(df["session_day"], errors="coerce")
    return df


# ============================================================
# SMT DETECTION (same as chart_app)
# ============================================================
def session_bounds(date_str):
    start = pd.Timestamp(date_str, tz="America/New_York") - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
    end = pd.Timestamp(date_str, tz="America/New_York") + pd.Timedelta(hours=17)
    return start, end


def session_end_for(ts):
    if ts.hour >= 18:
        return ts.normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=17)
    return ts.normalize() + pd.Timedelta(hours=17)


def first_strict_cross_ts(bars, start, end, level, is_high):
    sub = bars.loc[start:end]
    if len(sub) == 0:
        return None
    if is_high:
        mask = sub["high"] >= level + TICK_SIZE - EPS
    else:
        mask = sub["low"] <= level - TICK_SIZE + EPS
    if not mask.any():
        return None
    return sub.index[mask.argmax()]


def is_sunday_evening(ts):
    return ts.dayofweek == 6 and ts.hour >= 18


def formation_window(level_type, ts_formed, valid_from):
    vf = valid_from
    if level_type in ("LondonHigh","LondonLow"):
        d = vf.normalize()
        return (d + pd.Timedelta(hours=2), d + pd.Timedelta(hours=5))
    if level_type in ("AsiaKZHigh","AsiaKZLow"):
        d = vf.normalize()
        return (d + pd.Timedelta(hours=19), d + pd.Timedelta(hours=22))
    if level_type in ("AsiaONHigh","AsiaONLow"):
        return (vf - pd.Timedelta(hours=8), vf)
    if level_type in ("PDH","PDL"):
        return (ts_formed, ts_formed + pd.Timedelta(hours=23))
    if level_type in ("PWH","PWL"):
        return (ts_formed, ts_formed + pd.Timedelta(days=7))
    if level_type in ("4H_SwingHigh","4H_SwingLow"):
        return (ts_formed, ts_formed + pd.Timedelta(hours=4))
    return (ts_formed, ts_formed + pd.Timedelta(minutes=1))


def find_extreme(bars_1m, start, end, find_high):
    sub = bars_1m.loc[start:end]
    if len(sub) == 0:
        return None
    return sub["high"].idxmax() if find_high else sub["low"].idxmin()


def detect_smts_for_window(win_start, win_end, mnq_1m, mes_1m, mnq_lv, mes_lv):
    smts = []
    mnq_by_id = {r["level_id"]: r for _, r in mnq_lv.iterrows()}
    mes_by_id = {r["level_id"]: r for _, r in mes_lv.iterrows()}
    common_ids = set(mnq_by_id.keys()) & set(mes_by_id.keys())

    for level_id in common_ids:
        mnq_row = mnq_by_id[level_id]; mes_row = mes_by_id[level_id]
        lt = mnq_row["level_type"]
        is_high = lt in HIGH_LEVELS

        vf = max(mnq_row["valid_from"], mes_row["valid_from"], win_start)
        vu = min(mnq_row["valid_until"], mes_row["valid_until"], win_end)
        if vf >= vu: continue
        if lt in PERSISTENT_TYPES:
            ft = mnq_row.get("first_touch_ts")
            if pd.notna(ft) and ft < win_start: continue

        lp_mnq = mnq_row["price"]; lp_mes = mes_row["price"]
        mnq_cross = first_strict_cross_ts(mnq_1m, vf, vu, lp_mnq, is_high)
        mes_cross = first_strict_cross_ts(mes_1m, vf, vu, lp_mes, is_high)
        if mnq_cross is None and mes_cross is None: continue

        if mnq_cross is None:
            sweep_ts = mes_cross; sweeper = "MES"; nonconf = "MNQ"
            sweep_bars = mes_1m; nonconf_bars = mnq_1m
            lp_sweeper = lp_mes; lp_nonconf = lp_mnq
        elif mes_cross is None:
            sweep_ts = mnq_cross; sweeper = "MNQ"; nonconf = "MES"
            sweep_bars = mnq_1m; nonconf_bars = mes_1m
            lp_sweeper = lp_mnq; lp_nonconf = lp_mes
        else:
            if mnq_cross <= mes_cross:
                sweep_ts = mnq_cross; sweeper = "MNQ"; nonconf = "MES"
                sweep_bars = mnq_1m; nonconf_bars = mes_1m
                lp_sweeper = lp_mnq; lp_nonconf = lp_mes
            else:
                sweep_ts = mes_cross; sweeper = "MES"; nonconf = "MNQ"
                sweep_bars = mes_1m; nonconf_bars = mnq_1m
                lp_sweeper = lp_mes; lp_nonconf = lp_mnq

        if is_sunday_evening(sweep_ts): continue
        if sweep_ts not in nonconf_bars.index: continue
        nc_bar = nonconf_bars.loc[sweep_ts]
        if is_high:
            nc_extreme = nc_bar["high"]
            if nc_extreme >= lp_nonconf + TICK_SIZE - EPS: continue
        else:
            nc_extreme = nc_bar["low"]
            if nc_extreme <= lp_nonconf - TICK_SIZE + EPS: continue

        if is_high:
            sep_ticks = (lp_nonconf - nc_extreme) / TICK_SIZE
            sweep_price = sweep_bars.loc[sweep_ts]["high"]
        else:
            sep_ticks = (nc_extreme - lp_nonconf) / TICK_SIZE
            sweep_price = sweep_bars.loc[sweep_ts]["low"]

        smt_deadline = session_end_for(sweep_ts)
        nc_forward = nonconf_bars.loc[sweep_ts + pd.Timedelta(minutes=1):smt_deadline]
        death_reason = "session_end"; death_ts = smt_deadline
        if len(nc_forward) > 0:
            if is_high:
                strict_mask = nc_forward["high"] >= lp_nonconf + TICK_SIZE - EPS
            else:
                strict_mask = nc_forward["low"] <= lp_nonconf - TICK_SIZE + EPS
            if strict_mask.any():
                death_ts = nc_forward.index[strict_mask.argmax()]
                death_reason = "invalidated_strict"

        smts.append({
            "level_id": level_id, "birth_ts": sweep_ts, "death_ts": death_ts,
            "status": death_reason, "level_type": lt,
            "direction": "bearish" if is_high else "bullish",
            "sweeper": sweeper, "non_confirmer": nonconf,
            "level_price_sweeper": lp_sweeper, "level_price_nonconf": lp_nonconf,
            "sweep_price": sweep_price, "nonconf_price_at_birth": nc_extreme,
            "separation_ticks": sep_ticks,
            "mnq_ts_formed": mnq_row["ts_formed"], "mnq_valid_from": mnq_row["valid_from"],
            "mes_ts_formed": mes_row["ts_formed"], "mes_valid_from": mes_row["valid_from"],
        })
    df = pd.DataFrame(smts)
    if len(df):
        df = df.sort_values("birth_ts").reset_index(drop=True)
    return df


# ============================================================
# MSS / SL helpers
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


def track_trade(bars_1m, entry_ts, entry_price, sl_price, r_points, direction, session_end, tp_target):
    sub = bars_1m.loc[entry_ts + pd.Timedelta(minutes=1):session_end]
    if len(sub) == 0:
        return session_end, entry_price, "session_end"
    target_price = None
    if tp_target is not None:
        if direction == "bullish":
            target_price = entry_price + tp_target * r_points
        else:
            target_price = entry_price - tp_target * r_points
    for ts, bar in sub.iterrows():
        if direction == "bullish":
            hit_sl = bar["low"] <= sl_price + EPS
            hit_tp = target_price is not None and bar["high"] >= target_price - EPS
        else:
            hit_sl = bar["high"] >= sl_price - EPS
            hit_tp = target_price is not None and bar["low"] <= target_price + EPS
        if hit_sl and hit_tp: return ts, sl_price, "sl_hit"
        if hit_sl: return ts, sl_price, "sl_hit"
        if hit_tp: return ts, target_price, "tp_hit"
    last = sub.iloc[-1]
    return last.name, last["close"], "session_end"


# ============================================================
# REALISTIC MODE TRIM
# ============================================================
def realistic_trim(trades, tp_col, single_position=False):
    """
    Walk trades chronologically per (asset, tf, anchor, rule). Skip overlapping ones.
    If single_position=True, group by (tf, anchor, rule) only — both assets share
    one position slot (used for Auto-pick mode).
    """
    if len(trades) == 0:
        return trades
    t = trades.copy()
    is_managed = tp_col.endswith("_mgd")
    raw_tp = tp_col.replace("_mgd","")

    exit_series = []
    for _, r in t.iterrows():
        if is_managed and pd.notna(r.get("managed_exit_ts")):
            exit_ts = r["managed_exit_ts"]
        else:
            sl_ts = r.get("sl_hit_ts")
            session_end = r.get("trade_exit_ts_if_held")
            if raw_tp == "r_no_tp":
                exit_ts = sl_ts if pd.notna(sl_ts) else session_end
            else:
                n = int(raw_tp.replace("r_tp",""))
                tp_ts = r.get(f"hit_{n}r_ts")
                candidates = []
                if pd.notna(tp_ts): candidates.append(tp_ts)
                if pd.notna(sl_ts): candidates.append(sl_ts)
                if not candidates: candidates.append(session_end)
                exit_ts = min(candidates)
        exit_series.append(exit_ts)
    t["_exit_ts"] = exit_series

    t = t.sort_values("mss_ts").reset_index(drop=True)
    keep = []
    groups = {}
    for idx, row in t.iterrows():
        if single_position:
            key = (row["mss_timeframe"], row["anchor"], row["mss_rule"])
        else:
            key = (row["measured_asset"], row["mss_timeframe"], row["anchor"], row["mss_rule"])
        last_exit = groups.get(key)
        entry = row["mss_ts"]
        if last_exit is not None and entry < last_exit:
            keep.append(False)
            continue
        keep.append(True)
        groups[key] = row["_exit_ts"]
    t["_keep"] = keep
    return t[t["_keep"]].drop(columns=["_keep","_exit_ts"])


def simulate_retraced_entries(trades_df, retrace_r, tp_multiplier, tp_col_name, strategy_tf):
    """
    For each trade, simulate "wait for price to retrace `retrace_r` * R against
    the original MSS-close entry, then fill". Compute new entry, R distance,
    TP target (using tp_multiplier R from new_entry), and walk 1m bars.

    `tp_multiplier` corresponds to the selected TP target:
        TP1 -> 1.0, TP2 -> 2.0, ... TP5 -> 5.0
        no_TP -> None (only SL / kill / session_end can exit)

    `strategy_tf` is the strategy timeframe ("1m"/"5m"/"15m"). The retracement
    search window starts at mss_ts + TF_minutes (the MSS bar's close), so we
    only look for retracement AFTER the MSS bar has finished.

    SL stays at the original sl_price.
    Phase 4 kill timestamps (opp_smt_kill, opp_mss_kill) are preserved and acted
    on identically. Risk is fixed in $ via user-selected RISK_PER_TRADE;
    position size implicitly scales.

    REALISM RULES:
      - Search starts at MSS bar close (mss_ts + TF_minutes), not mss_ts + 1min
      - Same-bar retracement-fill + SL collision: trade FILLS at retracement,
        immediately stops out at -1R (limit order would have been triggered
        before SL was reached in real life)
      - Post-fill same-bar TP + SL collision: SL wins (conservative)

    Trades whose price never retraces (and don't get stopped) are EXCLUDED.
    Returns: a DataFrame with rewritten entry_price_sim, r_points_sim, outcome_r,
    outcome_reason, fill_ts, exit_ts_sim.
    """
    TF_MIN = {"1m": 1, "5m": 5, "15m": 15}
    tf_minutes = TF_MIN.get(strategy_tf, 1)

    if retrace_r == 0:
        out = trades_df.copy()
        out["fill_ts"] = out["mss_ts"]
        out["entry_price_sim"] = out["entry_price"]
        out["r_points_sim"] = out["r_points"]
        out["outcome_r"] = out[tp_col_name]
        out["outcome_reason"] = out.get("managed_exit_reason", pd.Series([""] * len(out))).fillna("session_end")
        out["exit_ts_sim"] = out.get("managed_exit_ts", out.get("trade_exit_ts_if_held"))
        return out

    bars_1m = {a: load_bars(a, "1m") for a in ("MNQ","MES")}

    rows_out = []
    for _, t in trades_df.iterrows():
        asset = t["measured_asset"]
        direction = t["direction"]
        orig_entry = float(t["entry_price"])
        sl = float(t["sl_price"])
        orig_r = abs(orig_entry - sl)
        if orig_r < TICK_SIZE - EPS:
            continue
        mss_ts = t["mss_ts"]
        if pd.isna(mss_ts):
            continue
        sess_end = session_end_for(mss_ts)

        # Preserve Phase 4 kill timestamp if any
        kill_ts = None
        kill_reason = None
        mgd_reason = t.get("managed_exit_reason", "")
        mgd_exit_ts = t.get("managed_exit_ts")
        if mgd_reason in ("opp_smt_kill", "opp_mss_kill", "smt_invalidation_kill") and pd.notna(mgd_exit_ts):
            kill_ts = mgd_exit_ts
            kill_reason = mgd_reason

        # Retracement target
        if direction == "bullish":
            retrace_price = orig_entry - retrace_r * orig_r
        else:
            retrace_price = orig_entry + retrace_r * orig_r

        # FIX 1: Search starts at MSS bar close (mss_ts + TF_minutes), not +1min
        search_start = mss_ts + pd.Timedelta(minutes=tf_minutes)
        try:
            sub = bars_1m[asset].loc[search_start:sess_end]
        except KeyError:
            sub = bars_1m[asset].iloc[0:0]
        if len(sub) == 0:
            continue

        # Find fill bar
        fill_ts = None
        fill_then_stopped = False  # FIX 2 flag
        for ts, bar in sub.iterrows():
            if kill_ts is not None and ts >= kill_ts:
                fill_ts = None
                break

            if direction == "bullish":
                hits_retrace = bar["low"] <= retrace_price + EPS
                hits_sl = bar["low"] <= sl + EPS
            else:
                hits_retrace = bar["high"] >= retrace_price - EPS
                hits_sl = bar["high"] >= sl - EPS

            if hits_retrace and hits_sl:
                # FIX 2: Same-bar fill + SL → trade fills at retracement, then stops out.
                # (Limit order would have been triggered before price reached the stop.)
                fill_ts = ts
                fill_then_stopped = True
                break
            if hits_sl and not hits_retrace:
                # SL alone on this bar (without touching retracement) → no fill
                fill_ts = None
                break
            if hits_retrace:
                fill_ts = ts
                break

        if fill_ts is None:
            continue

        new_entry = retrace_price
        new_r = abs(new_entry - sl)
        if new_r < TICK_SIZE - EPS:
            continue

        # Same-bar fill + SL: record immediate stop-out
        if fill_then_stopped:
            new_row = t.to_dict()
            new_row["fill_ts"] = fill_ts
            new_row["entry_price_sim"] = new_entry
            new_row["r_points_sim"] = new_r
            new_row["outcome_r"] = -1.0
            new_row["outcome_reason"] = "sl_hit"
            new_row["exit_ts_sim"] = fill_ts
            rows_out.append(new_row)
            continue

        # TP target (None for no_TP)
        new_tp = None
        if tp_multiplier is not None:
            if direction == "bullish":
                new_tp = new_entry + tp_multiplier * new_r
            else:
                new_tp = new_entry - tp_multiplier * new_r

        # Walk to outcome (start at next 1m bar after fill)
        try:
            sub2 = bars_1m[asset].loc[fill_ts + pd.Timedelta(minutes=1):sess_end]
        except KeyError:
            sub2 = bars_1m[asset].iloc[0:0]

        outcome_r = None
        outcome_reason = None
        outcome_exit_ts = None
        for ts, bar in sub2.iterrows():
            if kill_ts is not None and ts >= kill_ts:
                if direction == "bullish":
                    outcome_r = (bar["close"] - new_entry) / new_r
                else:
                    outcome_r = (new_entry - bar["close"]) / new_r
                outcome_reason = kill_reason or "kill"
                outcome_exit_ts = ts
                break

            if direction == "bullish":
                hit_tp = (new_tp is not None) and (bar["high"] >= new_tp - EPS)
                hit_sl = bar["low"] <= sl + EPS
            else:
                hit_tp = (new_tp is not None) and (bar["low"] <= new_tp + EPS)
                hit_sl = bar["high"] >= sl - EPS

            # FIX 3: Same-bar TP + SL → SL wins (conservative, intrabar sequence unknown)
            if hit_sl and hit_tp:
                outcome_r = -1.0
                outcome_reason = "sl_hit"
                outcome_exit_ts = ts
                break
            if hit_sl:
                outcome_r = -1.0
                outcome_reason = "sl_hit"
                outcome_exit_ts = ts
                break
            if hit_tp:
                outcome_r = float(tp_multiplier)
                outcome_reason = "tp_hit"
                outcome_exit_ts = ts
                break

        if outcome_r is None:
            if len(sub2) > 0:
                last_close = sub2.iloc[-1]["close"]
                if direction == "bullish":
                    outcome_r = (last_close - new_entry) / new_r
                else:
                    outcome_r = (new_entry - last_close) / new_r
                outcome_exit_ts = sub2.index[-1]
            else:
                outcome_r = 0.0
                outcome_exit_ts = fill_ts
            outcome_reason = "session_end"

        new_row = t.to_dict()
        new_row["fill_ts"] = fill_ts
        new_row["entry_price_sim"] = new_entry
        new_row["r_points_sim"] = new_r
        new_row["outcome_r"] = round(outcome_r, 3)
        new_row["outcome_reason"] = outcome_reason
        new_row["exit_ts_sim"] = outcome_exit_ts
        rows_out.append(new_row)

    return pd.DataFrame(rows_out)


# ============================================================
# LOAD DATA
# ============================================================
measurement_files = sorted(glob.glob(os.path.join(OUT_FOLDER, "SWEEP_measurements_*.csv")))
if not measurement_files:
    st.error(f"No SWEEP_measurements_*.csv found in {OUT_FOLDER}. Run phase3_sweep.py then phase4_sweep.py first.")
    st.stop()

# ============================================================
# TOP FILTER BAR
# ============================================================
st.markdown("### Strategy filters")

r1c1, r1c2, r1c3, r1c4, r1c5, r1c6, r1c7 = st.columns([1.2, 1, 1, 1.2, 1, 1.2, 1])

with r1c1:
    ds_name = st.selectbox("Dataset",
                             [os.path.basename(f) for f in measurement_files],
                             index=len(measurement_files)-1, key="ds")
with r1c2:
    asset_sel = st.selectbox("Asset", ["MNQ","MES","Both"], key="asset",
                              help="Sweep pipeline: each asset trades independently. "
                                   "'Both' shows the combined PnL from MNQ + MES trading in parallel.")
with r1c3:
    tf_sel = st.selectbox("TF", ["1m","5m","15m"], index=1, key="tf")
with r1c4:
    anchor_sel = st.selectbox("Anchor", ["before_smt","after_smt"], key="anchor")
with r1c5:
    rule_sel = st.selectbox("Rule", ["wick_break","close_break"], index=1, key="rule")
with r1c6:
    tp_sel = st.selectbox("TP", ["TP1","TP2","TP3","TP4","TP5","no_TP"], index=1, key="tp")
with r1c7:
    apply_clicked = st.button("Apply", type="primary", use_container_width=True)

r2c1, r2c2, r2c3 = st.columns([2, 1, 1])
with r2c1:
    sessions_sel = st.multiselect("Session", ["london","ny_am","ny_pm","outside"],
                                    default=["london","ny_am","ny_pm"], key="sessions")
with r2c2:
    risk_sel = st.number_input("Risk per trade ($)", min_value=1, value=int(RISK_PER_TRADE), step=10, key="risk")
with r2c3:
    starting_sel = st.number_input("Starting balance ($)", min_value=100, value=int(STARTING_BALANCE), step=1000, key="start")

r3c1, r3c2 = st.columns(2)
with r3c1:
    use_managed_sel = st.checkbox("Use at-entry filter + during-trade management",
                                    value=True, key="use_managed",
                                    help="When ON: skip trades with opposing-MSS at entry or opposing rank ≥ ours. "
                                         "Close trade if an opposing SMT/MSS triggers a kill during the trade. "
                                         "Uses r_tpN_mgd columns. When OFF: raw signal performance, no rules applied.")
with r3c2:
    hide_opp_live_sel = st.checkbox("Hide opposing-live SMTs (at SMT birth)",
                                      value=False, key="hide_opp_live",
                                      help="Advanced filter: exclude SMTs born when an opposing SMT of ≥ rank was already alive.")

# Level type multiselect
ALL_LEVEL_TYPES = sorted(HIGH_LEVELS | LOW_LEVELS)
level_types_sel = st.multiselect(
    "Include SMTs from these level types",
    ALL_LEVEL_TYPES,
    default=ALL_LEVEL_TYPES,
    key="level_types",
    help="Uncheck level types you don't want to trade.",
)

# Entry mode selector
entry_mode_sel = st.radio(
    "Entry mode",
    ["Standard (MSS bar close)", "FVG tap (5m FVG)", "FVG tap (15m FVG)"],
    horizontal=True,
    key="entry_mode",
    help="Standard: enter at the MSS bar close (Phase 4 default). "
         "FVG tap: requires an FVG to form within the 15m MSS leg AND price to retrace 1 tick into the FVG zone after MSS.",
)

# Entry retracement simulator
entry_retrace_sel = st.selectbox(
    "Entry retracement (R)",
    [0.0, 0.25, 0.5, 0.75],
    index=0,
    format_func=lambda x: f"{x:.2f}R" + (" (normal MSS-bar entry)" if x == 0.0 else " (wait for retracement)"),
    key="entry_retrace",
    help="Simulate waiting for price to retrace X·R against the MSS-close entry "
         "before filling. SL stays at the leg extreme; TP recomputes from the new "
         "entry using the selected TP target. Trades that never retrace this much "
         "are excluded entirely. Only applies when Entry mode = Standard.",
)

st.divider()

if "applied" not in st.session_state:
    st.session_state.applied = False

if apply_clicked:
    st.session_state.applied = True
    st.session_state.applied_ds = ds_name
    st.session_state.applied_asset = asset_sel
    st.session_state.applied_tf = tf_sel
    st.session_state.applied_anchor = anchor_sel
    st.session_state.applied_rule = rule_sel
    st.session_state.applied_tp = tp_sel
    st.session_state.applied_sessions = sessions_sel
    st.session_state.applied_risk = risk_sel
    st.session_state.applied_start = starting_sel
    st.session_state.applied_hide_opp_live = hide_opp_live_sel
    st.session_state.applied_use_managed = use_managed_sel
    st.session_state.applied_level_types = level_types_sel
    st.session_state.applied_entry_mode = entry_mode_sel
    st.session_state.applied_entry_retrace = entry_retrace_sel

if not st.session_state.applied:
    st.info("👆 Select a combination above and click **Apply**.")
    st.stop()

ds_name     = st.session_state.applied_ds
asset_sel   = st.session_state.applied_asset
tf_sel      = st.session_state.applied_tf
anchor_sel  = st.session_state.applied_anchor
rule_sel    = st.session_state.applied_rule
tp_sel      = st.session_state.applied_tp
sessions_sel = st.session_state.applied_sessions
risk_sel    = st.session_state.applied_risk
starting_sel = st.session_state.applied_start
hide_opp_live = st.session_state.get("applied_hide_opp_live", False)
use_managed = st.session_state.get("applied_use_managed", True)
level_types = st.session_state.get("applied_level_types", sorted(HIGH_LEVELS | LOW_LEVELS))
entry_mode = st.session_state.get("applied_entry_mode", "Standard (MSS bar close)")
entry_retrace_R = st.session_state.get("applied_entry_retrace", 0.0)

# ============================================================
# LOAD + FILTER
# ============================================================
csv_path = os.path.join(OUT_FOLDER, ds_name)
df_all = load_measurements(csv_path, os.path.getmtime(csv_path))

fvg_mode = entry_mode != "Standard (MSS bar close)"
fvg_tf = "5m" if "5m FVG" in entry_mode else "15m" if "15m FVG" in entry_mode else None

if fvg_mode:
    fvg_files = sorted(glob.glob(os.path.join(OUT_FOLDER, "FVG_outcomes_*.csv")))
    if not fvg_files:
        st.error("FVG outcomes CSV not found. Run fvg_outcomes.py first.")
        st.stop()
    fvg_path = fvg_files[-1]
    fvg_df = pd.read_csv(fvg_path)
    fvg_ts_cols = ["smt_birth_ts","mss_ts","fvg_15m_tap_ts","fvg_5m_tap_ts",
                    "fvg_15m_entry_ts","fvg_5m_entry_ts",
                    "fvg_15m_managed_exit_ts","fvg_5m_managed_exit_ts"]
    for c in fvg_ts_cols:
        if c in fvg_df.columns:
            fvg_df[c] = pd.to_datetime(fvg_df[c], utc=True, errors="coerce").dt.tz_convert("America/New_York")

    if tf_sel != "15m":
        st.warning(f"FVG entry mode requires 15m MSS detection. Switching from {tf_sel} to 15m.")
    tf_sel = "15m"

    df = fvg_df[
        (fvg_df["measured_asset"] == asset_sel)
        & (fvg_df["anchor"] == anchor_sel)
        & (fvg_df["mss_rule"] == rule_sel)
        & (fvg_df["session"].isin(sessions_sel))
        & (fvg_df["level_type"].isin(level_types))
    ].copy()

    prefix = f"fvg_{fvg_tf}"
    df["mss_ts_original"] = df["mss_ts"]
    df["entry_ts"] = df[f"{prefix}_entry_ts"]
    df["entry_price"] = df[f"{prefix}_entry_price"]
    df["r_points"] = df[f"{prefix}_r_points"]
    df["mss_ts"] = df["entry_ts"]

    for n in (1,2,3,4,5):
        df[f"r_tp{n}"] = df[f"r_tp{n}_{prefix}"]
        df[f"r_tp{n}_mgd"] = df[f"r_tp{n}_{prefix}_mgd"]
    df["r_no_tp"] = df[f"r_no_tp_{prefix}"]
    df["r_no_tp_mgd"] = df[f"r_no_tp_{prefix}_mgd"]

    df["entry_skipped"] = df[f"{prefix}_entry_skipped"]
    df["entry_skip_reason"] = df[f"{prefix}_entry_skip_reason"]
    df["managed_exit_ts"] = df[f"{prefix}_managed_exit_ts"]
    df["managed_exit_reason"] = df[f"{prefix}_managed_exit_reason"]

    df["mss_timeframe"] = "15m"
    df["mss_found"] = True
    df["final_outcome"] = "sl_hit"
    df["max_favorable_r"] = 0.0
    df["max_adverse_r"] = 0.0
    for col_n in (1,2,3,4,5):
        if f"hit_{col_n}r" not in df.columns:
            df[f"hit_{col_n}r"] = (df[f"r_tp{col_n}"] >= col_n - 0.001).fillna(False)
        if f"hit_{col_n}r_ts" not in df.columns:
            df[f"hit_{col_n}r_ts"] = pd.NaT
    if "sl_hit_ts" not in df.columns:
        df["sl_hit_ts"] = pd.NaT
    if "trade_exit_ts_if_held" not in df.columns:
        df["trade_exit_ts_if_held"] = df["managed_exit_ts"]

    df = df.dropna(subset=["entry_price","r_points","r_tp2"])

else:
    # Sweep pipeline: each asset trades independently. "Both" includes both.
    both_mode = (asset_sel == "Both")
    if both_mode:
        df = df_all[
            (df_all["mss_timeframe"] == tf_sel)
            & (df_all["anchor"] == anchor_sel)
            & (df_all["mss_rule"] == rule_sel)
            & (df_all["mss_found"] == True)
            & (df_all["final_outcome"].isin(["sl_hit","session_end"]))
            & (df_all["session"].isin(sessions_sel))
            & (df_all["level_type"].isin(level_types))
        ].copy()
    else:
        df = df_all[
            (df_all["measured_asset"] == asset_sel)
            & (df_all["mss_timeframe"] == tf_sel)
            & (df_all["anchor"] == anchor_sel)
            & (df_all["mss_rule"] == rule_sel)
            & (df_all["mss_found"] == True)
            & (df_all["final_outcome"].isin(["sl_hit","session_end"]))
            & (df_all["session"].isin(sessions_sel))
            & (df_all["level_type"].isin(level_types))
        ].copy()

total_signals = len(df)
n_skipped_at_entry = 0  # sweep pipeline has no entry skips

# Sweep pipeline has no opposing_live; managed mode same as raw
# (kept for column compatibility but no filtering effect)

tp_col_raw = TP_COL[tp_sel]
if use_managed:
    tp_col = tp_col_raw + "_mgd"
    if tp_col not in df.columns:
        st.error(f"Column '{tp_col}' not found. Falling back to raw column.")
        tp_col = tp_col_raw
else:
    tp_col = tp_col_raw

df = df.dropna(subset=[tp_col])

# Sequential trim: each asset has its own queue, so trim per asset independently
if both_mode:
    if len(df) > 0:
        df = pd.concat([
            realistic_trim(df[df["measured_asset"] == "MNQ"], tp_col, single_position=True),
            realistic_trim(df[df["measured_asset"] == "MES"], tp_col, single_position=True),
        ]).sort_values("mss_ts").reset_index(drop=True)
else:
    df = realistic_trim(df, tp_col, single_position=True)

# Entry retracement simulation (only for Standard entry mode)
n_skipped_no_retrace = 0
if entry_retrace_R > 0 and not fvg_mode and len(df) > 0:
    # Derive TP multiplier from tp_sel
    tp_multiplier = None
    if tp_sel != "no_TP":
        tp_multiplier = float(tp_sel.replace("TP", ""))
    with st.spinner(f"Simulating {entry_retrace_R:.2f}R retraced entries..."):
        df_sim = simulate_retraced_entries(df, entry_retrace_R, tp_multiplier, tp_col, tf_sel)
    n_original = len(df)
    n_filled = len(df_sim)
    n_skipped_no_retrace = n_original - n_filled
    if len(df_sim) == 0:
        st.error(f"No trades retraced to {entry_retrace_R:.2f}R. Try a smaller value.")
        st.stop()
    # Override columns so downstream stats use the simulated outcomes
    df = df_sim.copy()
    df["entry_price"] = df["entry_price_sim"]
    df["r_points"] = df["r_points_sim"]
    df["mss_ts"] = df["fill_ts"]
    df[tp_col] = df["outcome_r"]
    df["managed_exit_reason"] = df["outcome_reason"]
    df["managed_exit_ts"] = df["exit_ts_sim"]
    # Update hit flag for the selected TP (used in TP hit rate metric)
    if tp_multiplier is not None:
        hit_col = f"hit_{int(tp_multiplier)}r"
        df[hit_col] = (df["outcome_r"] >= tp_multiplier - 0.001)
    # max_favorable_r and max_adverse_r are no longer meaningful after simulation
    if "max_favorable_r" not in df.columns:
        df["max_favorable_r"] = 0.0
    if "max_adverse_r" not in df.columns:
        df["max_adverse_r"] = 0.0

if len(df) == 0:
    st.warning("No trades match this combination / filter set.")
    st.stop()

df["pnl_usd"] = df[tp_col] * risk_sel
df = df.sort_values("mss_ts").reset_index(drop=True)
df["cum_pnl"] = df["pnl_usd"].cumsum()
df["balance"] = starting_sel + df["cum_pnl"]

# ============================================================
# HEADER + FILTER TAGS
# ============================================================
tag_parts = [f"**{asset_sel}**", f"TF {tf_sel}", anchor_sel, rule_sel, tp_sel,
              "+".join(sessions_sel), ds_name]
if fvg_mode: tag_parts.append(f"📍 FVG entry ({fvg_tf})")
if use_managed: tag_parts.append("✅ managed")
else: tag_parts.append("⚠️ raw")
if hide_opp_live: tag_parts.append("🚫 opp-live")
if entry_retrace_R > 0:
    tag_parts.append(f"📐 retrace {entry_retrace_R:.2f}R")
all_levels_set = set(HIGH_LEVELS | LOW_LEVELS)
excluded_levels = all_levels_set - set(level_types)
if excluded_levels:
    short_excluded = ", ".join(sorted(excluded_levels))
    tag_parts.append(f"🚫 levels: {short_excluded}")
st.markdown(" · ".join(tag_parts))

if use_managed and n_skipped_at_entry > 0:
    skip_pct = n_skipped_at_entry / max(total_signals, 1) * 100
    st.caption(f"📋 At-entry filter: {n_skipped_at_entry} of {total_signals} signals skipped ({skip_pct:.1f}%) "
               f"due to opposing rank ≥ ours or opposing MSS at entry.")

if entry_retrace_R > 0 and n_skipped_no_retrace > 0:
    n_orig_for_retrace = len(df) + n_skipped_no_retrace
    fill_pct = len(df) / max(n_orig_for_retrace, 1) * 100
    st.caption(f"📐 Retracement filter: {len(df)} of {n_orig_for_retrace} trades filled at {entry_retrace_R:.2f}R "
               f"({fill_pct:.1f}%). {n_skipped_no_retrace} skipped (price never retraced, or SL/kill hit before fill).")

st.divider()

# ============================================================
# SUMMARY BAR + EQUITY CURVE
# ============================================================
st.markdown("## Profit and loss")
st.caption("Over time")

wins = df[df[tp_col] > 0]
losses = df[df[tp_col] <= 0]
win_rate = len(wins) / len(df) * 100 if len(df) else 0
total_pnl = df["pnl_usd"].sum()
final_balance = starting_sel + total_pnl
pct_return = (total_pnl / starting_sel) * 100

if tp_sel == "no_TP":
    tp_wr_label = "TP Touch (5R)"
    tp_hit_count = int(df["hit_5r"].sum()) if "hit_5r" in df.columns else 0
else:
    n = int(tp_sel.replace("TP",""))
    tp_wr_label = f"TP Hit Rate ({tp_sel})"
    tp_hit_count = int(df[f"hit_{n}r"].sum()) if f"hit_{n}r" in df.columns else 0
tp_hit_rate = tp_hit_count / len(df) * 100 if len(df) else 0
gap_pct = win_rate - tp_hit_rate

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total PnL", f"${total_pnl:,.2f}", f"{pct_return:+.2f}%")
c2.metric("Account Balance", f"${final_balance:,.2f}", f"{pct_return:+.2f}%")
c3.metric("Win Rate", f"{win_rate:.2f}%")
c4.metric(tp_wr_label, f"{tp_hit_rate:.2f}%",
            f"{gap_pct:+.1f}% drift" if gap_pct > 0 else None)
c5.metric("Total Trades", f"{len(df)}", f"{len(wins)}W / {len(losses)}L")
c6.metric("Avg Win / Avg Loss",
           f"${wins['pnl_usd'].mean():.0f}" if len(wins) else "—",
           f"${losses['pnl_usd'].mean():.0f}" if len(losses) else "—")

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=df["mss_ts"], y=df["balance"], mode="lines+markers",
    line=dict(color="#1976d2", width=2),
    marker=dict(size=5, color="#1976d2"),
    name="Balance",
))
fig_eq.add_hline(y=starting_sel, line_dash="dot", line_color="#888", annotation_text="Start")
fig_eq.update_layout(
    height=350, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    xaxis=dict(showgrid=False, linecolor="#444", tickfont=dict(color="#ccc")),
    yaxis=dict(showgrid=True, gridcolor="#222", linecolor="#444",
                tickfont=dict(color="#ccc"), tickprefix="$"),
    font=dict(color="#ccc"),
    margin=dict(l=50, r=30, t=20, b=40),
    showlegend=False,
)
st.plotly_chart(fig_eq, use_container_width=True)

st.divider()

# ============================================================
# RR METRICS
# ============================================================
st.markdown("## Risk-Reward metrics")

avg_rr = df[tp_col].mean()
max_rr = df[tp_col].max()
ideal_avg_rr = df["max_favorable_r"].mean()
max_ideal_rr = df["max_favorable_r"].max()

m1, m2, m3 = st.columns(3)
with m1:
    with st.container(border=True):
        sub = st.columns(2)
        sub[0].metric("Average R", f"{avg_rr:.2f}")
        sub[1].metric("Max R", f"{max_rr:.2f}")
        spark = go.Figure()
        spark.add_trace(go.Scatter(
            x=list(range(len(df))), y=df[tp_col].cumsum(),
            mode="lines", line=dict(color="#1976d2", width=1.5),
            fill="tozeroy", fillcolor="rgba(25,118,210,0.15)",
        ))
        spark.update_layout(
            height=80, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        )
        st.plotly_chart(spark, use_container_width=True)

with m2:
    with st.container(border=True):
        sub = st.columns(2)
        sub[0].metric("Ideal Avg R (MFE)", f"{ideal_avg_rr:.2f}")
        sub[1].metric("Max Ideal R", f"{max_ideal_rr:.2f}")
        spark2 = go.Figure()
        spark2.add_trace(go.Scatter(
            x=list(range(len(df))), y=df["max_favorable_r"].cumsum(),
            mode="lines", line=dict(color="#1976d2", width=1.5),
            fill="tozeroy", fillcolor="rgba(25,118,210,0.15)",
        ))
        spark2.update_layout(
            height=80, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        )
        st.plotly_chart(spark2, use_container_width=True)

with m3:
    with st.container(border=True):
        expectancy_usd = avg_rr * risk_sel
        avg_win_usd = wins["pnl_usd"].mean() if len(wins) else 0
        avg_loss_usd = losses["pnl_usd"].mean() if len(losses) else 0
        pf = (wins["pnl_usd"].sum() / abs(losses["pnl_usd"].sum())) if len(losses) and losses["pnl_usd"].sum() != 0 else float("inf")
        st.metric("Expectancy", f"${expectancy_usd:,.2f}",
                   f"${avg_win_usd:,.0f} / ${avg_loss_usd:,.0f}")
        st.metric("Profit Factor", f"{pf:.2f}" if pf != float("inf") else "∞")

st.divider()

# ============================================================
# WINNERS / LOSERS
# ============================================================
st.markdown("## Winners and Losers")


def consecutive_streaks(series):
    max_streak = 0
    cur = 0
    streaks = []
    for v in series:
        if v:
            cur += 1
            if cur > max_streak: max_streak = cur
        else:
            if cur > 0: streaks.append(cur)
            cur = 0
    if cur > 0: streaks.append(cur)
    avg = sum(streaks)/len(streaks) if streaks else 0
    return max_streak, avg


def trade_duration_minutes(row, tp_col, use_managed):
    entry = row["mss_ts"]
    if use_managed and pd.notna(row.get("managed_exit_ts")):
        exit_ts = row["managed_exit_ts"]
        if pd.isna(exit_ts) or pd.isna(entry):
            return None
        return (exit_ts - entry).total_seconds() / 60

    sl_ts = row.get("sl_hit_ts")
    tp_ts = None
    raw_tp = tp_col.replace("_mgd","")
    if raw_tp != "r_no_tp":
        n = int(raw_tp.replace("r_tp",""))
        tp_ts = row.get(f"hit_{n}r_ts")
    candidates = []
    if pd.notna(sl_ts): candidates.append(sl_ts)
    if tp_ts is not None and pd.notna(tp_ts): candidates.append(tp_ts)
    if not candidates:
        exit_ts = row.get("trade_exit_ts_if_held")
    else:
        exit_ts = min(candidates)
    if pd.isna(exit_ts) or pd.isna(entry): return None
    return (exit_ts - entry).total_seconds() / 60


df["_duration_min"] = df.apply(lambda r: trade_duration_minutes(r, tp_col, use_managed), axis=1)
wins = df[df[tp_col] > 0]
losses = df[df[tp_col] <= 0]

is_win_series = (df[tp_col] > 0).values
is_loss_series = (df[tp_col] <= 0).values
max_win_streak, avg_win_streak = consecutive_streaks(is_win_series)
max_loss_streak, avg_loss_streak = consecutive_streaks(is_loss_series)

def fmt_duration(m):
    if m is None or pd.isna(m): return "—"
    h = int(m // 60); mm = int(m % 60)
    return f"{h}h {mm}m"

wl1, wl2 = st.columns(2)
with wl1:
    with st.container(border=True):
        st.markdown("### 🟢 Winners")
        st.markdown(f"**Total winners:** {len(wins)}")
        st.markdown(f"**Best win:** {wins[tp_col].max():.2f}R (${wins['pnl_usd'].max():,.0f})" if len(wins) else "**Best win:** —")
        st.markdown(f"**Average win:** {wins[tp_col].mean():.2f}R (${wins['pnl_usd'].mean():,.0f})" if len(wins) else "**Average win:** —")
        st.markdown(f"**Average duration:** {fmt_duration(wins['_duration_min'].mean())}" if len(wins) else "**Average duration:** —")
        st.markdown(f"**Max consecutive wins:** {max_win_streak}")
        st.markdown(f"**Avg consecutive wins:** {avg_win_streak:.2f}")

with wl2:
    with st.container(border=True):
        st.markdown("### 🔴 Losers")
        st.markdown(f"**Total losers:** {len(losses)}")
        st.markdown(f"**Worst loss:** {losses[tp_col].min():.2f}R (${losses['pnl_usd'].min():,.0f})" if len(losses) else "**Worst loss:** —")
        st.markdown(f"**Average loss:** {losses[tp_col].mean():.2f}R (${losses['pnl_usd'].mean():,.0f})" if len(losses) else "**Average loss:** —")
        st.markdown(f"**Average duration:** {fmt_duration(losses['_duration_min'].mean())}" if len(losses) else "**Average duration:** —")
        st.markdown(f"**Max consecutive losses:** {max_loss_streak}")
        st.markdown(f"**Avg consecutive losses:** {avg_loss_streak:.2f}")

st.divider()

# ============================================================
# PERFORMANCE BY SIDE
# ============================================================
st.markdown("## Performance by side")

bulls = df[df["direction"] == "bullish"]
bears = df[df["direction"] == "bearish"]
bulls_wr = (bulls[tp_col] > 0).sum() / len(bulls) * 100 if len(bulls) else 0
bears_wr = (bears[tp_col] > 0).sum() / len(bears) * 100 if len(bears) else 0

sd1, sd2 = st.columns(2)
with sd1:
    with st.container(border=True):
        st.markdown("### Total Trades")
        fig_d1 = go.Figure(data=[go.Pie(
            labels=["buy (bullish)", "sell (bearish)"],
            values=[len(bulls), len(bears)],
            hole=0.55,
            marker=dict(colors=["#26a69a","#1976d2"]),
            textinfo="percent", textfont=dict(size=14),
        )])
        fig_d1.update_layout(
            height=280, paper_bgcolor="#0e1117",
            font=dict(color="#ccc"),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True, legend=dict(orientation="h", y=-0.05),
        )
        st.plotly_chart(fig_d1, use_container_width=True)

with sd2:
    with st.container(border=True):
        st.markdown("### Win Rate by Side")
        fig_d2 = go.Figure(data=[go.Pie(
            labels=[f"buy {bulls_wr:.0f}%", f"sell {bears_wr:.0f}%"],
            values=[bulls_wr, bears_wr],
            hole=0.55,
            marker=dict(colors=["#26a69a","#1976d2"]),
            textinfo="label", textfont=dict(size=12),
        )])
        fig_d2.update_layout(
            height=280, paper_bgcolor="#0e1117",
            font=dict(color="#ccc"),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True, legend=dict(orientation="h", y=-0.05),
        )
        st.plotly_chart(fig_d2, use_container_width=True)

st.divider()

# ============================================================
# CALENDAR
# ============================================================
st.markdown("## Calendar")

df["trade_date"] = df["mss_ts"].dt.date

all_months = sorted(df["mss_ts"].dt.to_period("M").unique())
if len(all_months) == 0:
    st.warning("No trades to show in calendar.")
    st.stop()

month_options = [str(p) for p in all_months]

if "cal_idx" not in st.session_state:
    st.session_state.cal_idx = len(month_options) - 1
st.session_state.cal_idx = max(0, min(st.session_state.cal_idx, len(month_options) - 1))

mc1, mc2, mc3, mc4 = st.columns([1,3,1,3])
with mc1:
    if st.button("←", key="prev_month", use_container_width=True):
        st.session_state.cal_idx = max(0, st.session_state.cal_idx - 1)
with mc2:
    st.session_state.cal_idx = month_options.index(
        st.selectbox("Month", month_options, index=st.session_state.cal_idx, label_visibility="collapsed")
    )
with mc3:
    if st.button("→", key="next_month", use_container_width=True):
        st.session_state.cal_idx = min(len(month_options) - 1, st.session_state.cal_idx + 1)

selected_month = all_months[st.session_state.cal_idx]
month_start = pd.Timestamp(selected_month.to_timestamp())
month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()

month_df = df[df["mss_ts"].dt.to_period("M") == selected_month]
by_day = month_df.groupby("trade_date").agg(
    pnl=("pnl_usd","sum"), trades=("pnl_usd","size"),
).to_dict("index")

m_total_trades = len(month_df)
if m_total_trades > 0:
    m_wins = month_df[month_df[tp_col] > 0]
    m_losses = month_df[month_df[tp_col] <= 0]
    m_pnl = month_df["pnl_usd"].sum()
    m_wr = len(m_wins) / m_total_trades * 100
    m_total_r = month_df[tp_col].sum()

    sorted_pnl = month_df.sort_values("mss_ts")["pnl_usd"].values
    cum = np.cumsum(sorted_pnl)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], cum)))
    cum_with_zero = np.concatenate(([0.0], cum))
    drawdowns = cum_with_zero - running_peak
    m_max_dd_usd = float(drawdowns.min()) if len(drawdowns) else 0.0
    m_max_dd_r = m_max_dd_usd / risk_sel if risk_sel > 0 else 0.0
else:
    m_wins = m_losses = month_df
    m_pnl = 0.0
    m_wr = 0.0
    m_total_r = 0.0
    m_max_dd_usd = 0.0
    m_max_dd_r = 0.0

mm1, mm2, mm3, mm4, mm5 = st.columns(5)
mm1.metric("Month PnL", f"${m_pnl:,.2f}", f"{m_total_r:+.2f}R")
mm2.metric("Win Rate", f"{m_wr:.1f}%")
mm3.metric("Total Trades", f"{m_total_trades}",
            f"{len(m_wins)}W / {len(m_losses)}L" if m_total_trades else "—")
mm4.metric("Trading days", f"{len(by_day)}")
mm5.metric("Max Drawdown", f"${m_max_dd_usd:,.2f}", f"{m_max_dd_r:.2f}R",
            delta_color="inverse")

first_day = month_start
dow_of_first = first_day.weekday()
days_in_month = (month_end - month_start).days + 1

dow_headers = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
hdr_cols = st.columns(7)
for i, h in enumerate(dow_headers):
    hdr_cols[i].markdown(f"<div style='color:#888;font-size:12px;text-align:center'>{h}</div>",
                          unsafe_allow_html=True)

cal_rows = []
current_row = [None]*dow_of_first
for d in range(1, days_in_month + 1):
    date = (month_start + pd.Timedelta(days=d - 1)).date()
    current_row.append(date)
    if len(current_row) == 7:
        cal_rows.append(current_row); current_row = []
if current_row:
    while len(current_row) < 7: current_row.append(None)
    cal_rows.append(current_row)

if "selected_day" not in st.session_state:
    st.session_state.selected_day = None

for row in cal_rows:
    cols = st.columns(7)
    for i, date in enumerate(row):
        with cols[i]:
            if date is None:
                st.markdown("<div style='height:100px'></div>", unsafe_allow_html=True)
                continue
            data = by_day.get(date)
            day_num = date.day
            if data is None:
                st.markdown(
                    f"<div style='height:100px;border:1px solid #222;border-radius:6px;padding:6px;color:#555'>"
                    f"<div style='text-align:right;font-size:12px'>{day_num}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                pnl = data["pnl"]
                trades_n = data["trades"]
                if pnl > 0:
                    color = "#0a2818"; border = "#26a69a"; txt_col = "#4caf50"; sign = "+"
                elif pnl < 0:
                    color = "#2b0f0f"; border = "#d32f2f"; txt_col = "#ef5350"; sign = ""
                else:
                    color = "#2a2410"; border = "#d4a017"; txt_col = "#d4a017"; sign = ""
                html = (
                    f"<div style='height:100px;background:{color};border:1px solid {border};"
                    f"border-radius:6px;padding:6px;display:flex;flex-direction:column;justify-content:space-between;'>"
                    f"<div style='display:flex;justify-content:space-between'>"
                    f"<span style='font-size:11px;color:#bbb'>{trades_n} trade{'s' if trades_n>1 else ''}</span>"
                    f"<span style='font-size:12px;color:#ddd'>{day_num}</span>"
                    f"</div>"
                    f"<div style='text-align:center;font-weight:600;color:{txt_col};font-size:13px'>"
                    f"{sign}${abs(pnl):,.2f}</div>"
                    f"</div>"
                )
                st.markdown(html, unsafe_allow_html=True)
                if st.button(" ", key=f"daybtn_{date}", help=f"{date} · {trades_n} trades · ${pnl:+.2f}"):
                    st.session_state.selected_day = date

# ============================================================
# TRADES FOR SELECTED DAY
# ============================================================
if st.session_state.selected_day is not None:
    st.divider()
    sd = st.session_state.selected_day
    st.markdown(f"### Trades on {sd}")

    day_trades = df[df["trade_date"] == sd].sort_values("mss_ts").reset_index(drop=True)

    for i, row in day_trades.iterrows():
        with st.container(border=True):
            c1, c2, c3, c4, c5, c6 = st.columns([1, 2, 1, 1, 1, 1])
            c1.write(f"**#{i+1}**")
            c2.write(f"{row['mss_ts'].strftime('%H:%M')} · {row['direction']} · {row['level_type']}")
            c3.write(f"Entry: {row['entry_price']:.2f}")
            c4.write(f"SL: {row['sl_price']:.2f}")
            outcome_color = "🟢" if row[tp_col] > 0 else "🔴"
            c5.write(f"{outcome_color} {row[tp_col]:+.2f}R")
            c6.write(f"${row['pnl_usd']:+,.2f}")
            if st.button("📈 View chart", key=f"view_{sd}_{i}"):
                st.session_state.selected_trade = row.to_dict()


# ============================================================
# DOWNLOAD COMBINATION DATA
# ============================================================
st.divider()
st.markdown("## Download combination data")
st.caption("Export this strategy's complete data — all trades + summary metrics + monthly stats — as CSV files.")

trades_export = df.copy()
trades_export["combo_asset"] = asset_sel
trades_export["combo_tf"] = tf_sel
trades_export["combo_anchor"] = anchor_sel
trades_export["combo_rule"] = rule_sel
trades_export["combo_tp"] = tp_sel
trades_export["combo_sessions"] = "+".join(sessions_sel)
trades_export["combo_managed"] = use_managed
trades_export["combo_entry_mode"] = entry_mode
trades_export["combo_levels"] = ",".join(level_types)
trades_export["combo_risk_per_trade_usd"] = risk_sel
trades_export["combo_starting_balance_usd"] = starting_sel
trades_export["combo_dataset"] = ds_name

for c in ("entry_price","sl_price","r_points","max_favorable_r","max_adverse_r",
          tp_col,"pnl_usd","cum_pnl","balance"):
    if c in trades_export.columns:
        trades_export[c] = trades_export[c].astype(float).round(2)

core_cols = [
    "combo_asset","combo_tf","combo_anchor","combo_rule","combo_tp","combo_sessions",
    "combo_managed","combo_entry_mode","combo_levels",
    "combo_dataset","combo_risk_per_trade_usd","combo_starting_balance_usd",
    "smt_birth_ts","mss_ts","direction","level_type","sweeper","measured_asset",
    "session","entry_price","sl_price","r_points",
    tp_col, "pnl_usd","cum_pnl","balance",
    "max_favorable_r","max_adverse_r",
    "hit_1r","hit_2r","hit_3r","hit_4r","hit_5r",
    "managed_exit_ts","managed_exit_reason",
    "opposing_smt_at_entry","opposing_smt_at_entry_rank","opposing_mss_at_entry",
    "entry_skipped","entry_skip_reason",
    "_duration_min",
]
core_cols = [c for c in core_cols if c in trades_export.columns]
trades_export = trades_export[core_cols + [c for c in trades_export.columns if c not in core_cols]]

summary_data = {
    "combo_asset": asset_sel,
    "combo_tf": tf_sel,
    "combo_anchor": anchor_sel,
    "combo_rule": rule_sel,
    "combo_tp": tp_sel,
    "combo_sessions": "+".join(sessions_sel),
    "combo_managed": use_managed,
    "combo_entry_mode": entry_mode,
    "combo_levels_included": ",".join(level_types),
    "combo_levels_excluded": ",".join(sorted(set(HIGH_LEVELS | LOW_LEVELS) - set(level_types))) or "(none)",
    "dataset": ds_name,
    "risk_per_trade_usd": risk_sel,
    "starting_balance_usd": starting_sel,
    "total_trades": len(df),
    "total_signals_before_skip": total_signals,
    "skipped_at_entry": n_skipped_at_entry,
    "skip_pct": round(n_skipped_at_entry / max(total_signals,1) * 100, 2),
    "wins": int((df[tp_col] > 0).sum()),
    "losses": int((df[tp_col] <= 0).sum()),
    "win_rate_pct": round(win_rate, 2),
    "tp_hit_rate_pct": round(tp_hit_rate, 2),
    "drift_gap_pct": round(gap_pct, 2),
    "total_pnl_usd": round(total_pnl, 2),
    "final_balance_usd": round(final_balance, 2),
    "pct_return": round(pct_return, 2),
    "mean_r": round(df[tp_col].mean(), 3),
    "median_r": round(df[tp_col].median(), 3),
    "std_r": round(df[tp_col].std(), 3) if len(df) > 1 else 0.0,
    "total_r": round(df[tp_col].sum(), 2),
    "ideal_avg_r_mfe": round(ideal_avg_rr, 3) if not pd.isna(ideal_avg_rr) else 0.0,
    "max_ideal_r_mfe": round(max_ideal_rr, 2) if not pd.isna(max_ideal_rr) else 0.0,
    "expectancy_usd": round(expectancy_usd, 2),
    "avg_win_usd": round(avg_win_usd, 2) if avg_win_usd else 0.0,
    "avg_loss_usd": round(avg_loss_usd, 2) if avg_loss_usd else 0.0,
    "profit_factor": round(pf, 2) if pf != float("inf") else 99.99,
    "max_consecutive_wins": int(max_win_streak),
    "max_consecutive_losses": int(max_loss_streak),
    "avg_consecutive_wins": round(avg_win_streak, 2),
    "avg_consecutive_losses": round(avg_loss_streak, 2),
}

if len(df) > 0:
    sorted_pnl = df.sort_values("mss_ts")["pnl_usd"].values
    cum = np.cumsum(sorted_pnl)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], cum)))
    cum_with_zero = np.concatenate(([0.0], cum))
    drawdowns = cum_with_zero - running_peak
    summary_data["global_max_dd_usd"] = round(float(drawdowns.min()), 2)
    summary_data["global_max_dd_r"] = round(float(drawdowns.min()) / risk_sel, 2) if risk_sel else 0.0
else:
    summary_data["global_max_dd_usd"] = 0.0
    summary_data["global_max_dd_r"] = 0.0

bulls = df[df["direction"] == "bullish"]
bears = df[df["direction"] == "bearish"]
summary_data["bullish_trades"] = len(bulls)
summary_data["bearish_trades"] = len(bears)
summary_data["bullish_win_rate_pct"] = round((bulls[tp_col] > 0).sum() / len(bulls) * 100, 2) if len(bulls) else 0.0
summary_data["bearish_win_rate_pct"] = round((bears[tp_col] > 0).sum() / len(bears) * 100, 2) if len(bears) else 0.0
summary_data["bullish_mean_r"] = round(bulls[tp_col].mean(), 3) if len(bulls) else 0.0
summary_data["bearish_mean_r"] = round(bears[tp_col].mean(), 3) if len(bears) else 0.0

summary_export = pd.DataFrame([summary_data])

monthly_rows = []
df_sorted = df.sort_values("mss_ts").copy()
df_sorted["_month"] = df_sorted["mss_ts"].dt.to_period("M").astype(str)
for month, m_df in df_sorted.groupby("_month"):
    m_n = len(m_df)
    if m_n == 0: continue
    m_pnl_arr = m_df["pnl_usd"].values
    cum = np.cumsum(m_pnl_arr)
    peak = np.maximum.accumulate(np.concatenate(([0.0], cum)))
    cum0 = np.concatenate(([0.0], cum))
    dd = cum0 - peak
    monthly_rows.append({
        "month": month,
        "total_trades": m_n,
        "wins": int((m_df[tp_col] > 0).sum()),
        "losses": int((m_df[tp_col] <= 0).sum()),
        "win_rate_pct": round((m_df[tp_col] > 0).sum() / m_n * 100, 2),
        "total_pnl_usd": round(m_df["pnl_usd"].sum(), 2),
        "total_r": round(m_df[tp_col].sum(), 2),
        "mean_r": round(m_df[tp_col].mean(), 3),
        "trading_days": m_df["mss_ts"].dt.date.nunique(),
        "max_drawdown_usd": round(float(dd.min()), 2),
        "max_drawdown_r": round(float(dd.min()) / risk_sel, 2) if risk_sel else 0.0,
    })
monthly_export = pd.DataFrame(monthly_rows)

filter_tag = f"{asset_sel}_{tf_sel}_{anchor_sel}_{rule_sel}_{tp_sel}"
filter_tag = filter_tag.replace(" ","").replace("(","").replace(")","")

dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button(
        "📥 Download trades (CSV)",
        data=trades_export.to_csv(index=False).encode("utf-8"),
        file_name=f"trades_{filter_tag}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with dl2:
    st.download_button(
        "📥 Download summary (CSV)",
        data=summary_export.to_csv(index=False).encode("utf-8"),
        file_name=f"summary_{filter_tag}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with dl3:
    st.download_button(
        "📥 Download monthly stats (CSV)",
        data=monthly_export.to_csv(index=False).encode("utf-8"),
        file_name=f"monthly_{filter_tag}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# FULL STRATEGY REPORT (ZIP archive)
# ============================================================
def build_strategy_zip():
    """
    Build a ZIP archive containing 15 files documenting this strategy across
    the entire 3-year dataset. See README for file list.
    """
    buf = io.BytesIO()

    # --- Per-week aggregation ---
    df_w = df.sort_values("mss_ts").copy()
    df_w["_week"] = df_w["mss_ts"].dt.to_period("W-SUN").astype(str)
    weekly_rows = []
    for week, w_df in df_w.groupby("_week"):
        if len(w_df) == 0: continue
        cum = np.cumsum(w_df["pnl_usd"].values)
        peak = np.maximum.accumulate(np.concatenate(([0.0], cum)))
        cum0 = np.concatenate(([0.0], cum))
        dd = cum0 - peak
        weekly_rows.append({
            "week": week,
            "total_trades": len(w_df),
            "wins": int((w_df[tp_col] > 0).sum()),
            "losses": int((w_df[tp_col] <= 0).sum()),
            "win_rate_pct": round((w_df[tp_col] > 0).sum() / len(w_df) * 100, 2),
            "total_pnl_usd": round(w_df["pnl_usd"].sum(), 2),
            "total_r": round(w_df[tp_col].sum(), 2),
            "mean_r": round(w_df[tp_col].mean(), 3),
            "trading_days": w_df["mss_ts"].dt.date.nunique(),
            "max_drawdown_usd": round(float(dd.min()), 2),
        })
    weekly_export = pd.DataFrame(weekly_rows)

    # --- Per-day aggregation (trading days only) ---
    df_d = df.sort_values("mss_ts").copy()
    df_d["_day"] = df_d["mss_ts"].dt.date
    daily_rows = []
    for day, d_df in df_d.groupby("_day"):
        if len(d_df) == 0: continue
        daily_rows.append({
            "date": str(day),
            "dow": pd.Timestamp(day).day_name(),
            "total_trades": len(d_df),
            "wins": int((d_df[tp_col] > 0).sum()),
            "losses": int((d_df[tp_col] <= 0).sum()),
            "total_pnl_usd": round(d_df["pnl_usd"].sum(), 2),
            "total_r": round(d_df[tp_col].sum(), 2),
            "first_trade_time": d_df["mss_ts"].min().strftime("%H:%M"),
            "last_trade_time": d_df["mss_ts"].max().strftime("%H:%M"),
        })
    daily_export = pd.DataFrame(daily_rows)

    # --- Per-year aggregation ---
    df_y = df.sort_values("mss_ts").copy()
    df_y["_year"] = df_y["mss_ts"].dt.year
    yearly_rows = []
    for year, y_df in df_y.groupby("_year"):
        if len(y_df) == 0: continue
        cum = np.cumsum(y_df["pnl_usd"].values)
        peak = np.maximum.accumulate(np.concatenate(([0.0], cum)))
        cum0 = np.concatenate(([0.0], cum))
        dd = cum0 - peak
        wins_y = (y_df[tp_col] > 0).sum()
        losses_y = (y_df[tp_col] <= 0).sum()
        pf_y = (y_df[y_df[tp_col] > 0]["pnl_usd"].sum()
                / abs(y_df[y_df[tp_col] <= 0]["pnl_usd"].sum())) \
                if losses_y > 0 else float("inf")
        yearly_rows.append({
            "year": int(year),
            "total_trades": len(y_df),
            "wins": int(wins_y),
            "losses": int(losses_y),
            "win_rate_pct": round(wins_y / len(y_df) * 100, 2),
            "total_pnl_usd": round(y_df["pnl_usd"].sum(), 2),
            "total_r": round(y_df[tp_col].sum(), 2),
            "mean_r": round(y_df[tp_col].mean(), 3),
            "profit_factor": round(pf_y, 2) if pf_y != float("inf") else 99.99,
            "trading_days": y_df["mss_ts"].dt.date.nunique(),
            "max_drawdown_usd": round(float(dd.min()), 2),
        })
    yearly_export = pd.DataFrame(yearly_rows)

    # --- Quarterly aggregation ---
    df_q = df.sort_values("mss_ts").copy()
    df_q["_quarter"] = df_q["mss_ts"].dt.to_period("Q").astype(str)
    quarterly_rows = []
    for q, q_df in df_q.groupby("_quarter"):
        if len(q_df) == 0: continue
        wins_q = (q_df[tp_col] > 0).sum()
        quarterly_rows.append({
            "quarter": q,
            "total_trades": len(q_df),
            "wins": int(wins_q),
            "win_rate_pct": round(wins_q / len(q_df) * 100, 2),
            "total_pnl_usd": round(q_df["pnl_usd"].sum(), 2),
            "total_r": round(q_df[tp_col].sum(), 2),
            "mean_r": round(q_df[tp_col].mean(), 3),
        })
    quarterly_export = pd.DataFrame(quarterly_rows)

    # --- Drawdown periods ---
    sorted_df = df.sort_values("mss_ts").reset_index(drop=True).copy()
    sorted_df["cum"] = sorted_df["pnl_usd"].cumsum()
    sorted_df["peak"] = sorted_df["cum"].cummax()
    sorted_df["dd"] = sorted_df["cum"] - sorted_df["peak"]
    # Identify drawdown periods: from a peak that gets surpassed later
    dd_periods = []
    in_dd = False
    dd_start_idx = None
    peak_val = 0.0
    for i, row in sorted_df.iterrows():
        if row["dd"] < 0 and not in_dd:
            in_dd = True
            dd_start_idx = i
            peak_val = row["peak"]
        elif row["dd"] >= 0 and in_dd:
            # Recovery
            in_dd = False
            dd_slice = sorted_df.iloc[dd_start_idx:i+1]
            trough_idx = dd_slice["dd"].idxmin()
            dd_periods.append({
                "start_date": sorted_df.iloc[dd_start_idx]["mss_ts"].strftime("%Y-%m-%d"),
                "trough_date": sorted_df.loc[trough_idx]["mss_ts"].strftime("%Y-%m-%d"),
                "recovery_date": sorted_df.iloc[i]["mss_ts"].strftime("%Y-%m-%d"),
                "trough_dd_usd": round(float(sorted_df["dd"].iloc[dd_slice.index[0]:dd_slice.index[-1]+1].min()), 2),
                "duration_days": (sorted_df.iloc[i]["mss_ts"] - sorted_df.iloc[dd_start_idx]["mss_ts"]).days,
                "trades_in_dd": i - dd_start_idx + 1,
                "peak_at_start_usd": round(peak_val, 2),
            })
    # If still in DD at end
    if in_dd:
        dd_slice = sorted_df.iloc[dd_start_idx:]
        trough_idx = dd_slice["dd"].idxmin()
        dd_periods.append({
            "start_date": sorted_df.iloc[dd_start_idx]["mss_ts"].strftime("%Y-%m-%d"),
            "trough_date": sorted_df.loc[trough_idx]["mss_ts"].strftime("%Y-%m-%d"),
            "recovery_date": "(not yet recovered)",
            "trough_dd_usd": round(float(dd_slice["dd"].min()), 2),
            "duration_days": (sorted_df.iloc[-1]["mss_ts"] - sorted_df.iloc[dd_start_idx]["mss_ts"]).days,
            "trades_in_dd": len(dd_slice),
            "peak_at_start_usd": round(peak_val, 2),
        })
    drawdowns_export = pd.DataFrame(dd_periods).sort_values("trough_dd_usd") if dd_periods else pd.DataFrame()

    # --- Streaks ---
    sorted_df["is_win"] = sorted_df[tp_col] > 0
    streak_rows = []
    cur_kind = None
    cur_start = None
    cur_count = 0
    cur_pnl = 0.0
    for i, row in sorted_df.iterrows():
        kind = "win" if row["is_win"] else "loss"
        if kind != cur_kind:
            if cur_kind is not None and cur_count > 0:
                streak_rows.append({
                    "kind": cur_kind,
                    "count": cur_count,
                    "start_date": cur_start,
                    "end_date": sorted_df.iloc[i-1]["mss_ts"].strftime("%Y-%m-%d %H:%M"),
                    "total_pnl_usd": round(cur_pnl, 2),
                })
            cur_kind = kind
            cur_start = row["mss_ts"].strftime("%Y-%m-%d %H:%M")
            cur_count = 1
            cur_pnl = float(row["pnl_usd"])
        else:
            cur_count += 1
            cur_pnl += float(row["pnl_usd"])
    if cur_kind is not None and cur_count > 0:
        streak_rows.append({
            "kind": cur_kind,
            "count": cur_count,
            "start_date": cur_start,
            "end_date": sorted_df.iloc[-1]["mss_ts"].strftime("%Y-%m-%d %H:%M"),
            "total_pnl_usd": round(cur_pnl, 2),
        })
    streaks_export = pd.DataFrame(streak_rows)

    # --- Performance breakdowns ---
    def grp_perf(df_g, col):
        rows = []
        for k, g in df_g.groupby(col):
            wins_g = (g[tp_col] > 0).sum()
            rows.append({
                col: k,
                "trades": len(g),
                "wins": int(wins_g),
                "losses": int((g[tp_col] <= 0).sum()),
                "win_rate_pct": round(wins_g / len(g) * 100, 2),
                "mean_r": round(g[tp_col].mean(), 3),
                "total_r": round(g[tp_col].sum(), 2),
                "total_pnl_usd": round(g["pnl_usd"].sum(), 2),
            })
        return pd.DataFrame(rows).sort_values("total_pnl_usd", ascending=False) if rows else pd.DataFrame()

    df_b = df.copy()
    df_b["dow"] = df_b["mss_ts"].dt.day_name()
    df_b["entry_hour"] = df_b["mss_ts"].dt.hour

    by_level_export = grp_perf(df_b, "level_type")
    by_dow_export = grp_perf(df_b, "dow")
    by_hour_export = grp_perf(df_b, "entry_hour")
    by_direction_export = grp_perf(df_b, "direction")
    by_session_export = grp_perf(df_b, "session") if "session" in df_b.columns else pd.DataFrame()
    by_sweeper_export = grp_perf(df_b, "sweeper") if "sweeper" in df_b.columns else pd.DataFrame()

    # --- MAE buckets ---
    mae_buckets_export = pd.DataFrame()
    if "max_adverse_r" in df_b.columns and df_b["max_adverse_r"].notna().any():
        BUCKETS = [
            ("0 to 0.25R",    0.00, 0.25),
            ("0.25 to 0.5R",  0.25, 0.50),
            ("0.5 to 0.75R",  0.50, 0.75),
            ("0.75 to 0.99R", 0.75, 0.99),
        ]
        df_b["_group"] = "loser"
        df_b.loc[df_b[tp_col] > 0, "_group"] = "winner"
        rows = []
        for grp_name in ("winner", "loser", "all"):
            sub = df_b if grp_name == "all" else df_b[df_b["_group"] == grp_name]
            if len(sub) == 0: continue
            mae = sub["max_adverse_r"].clip(upper=0.99)
            n = len(mae)
            for label, lo, hi in BUCKETS:
                if hi >= 0.99:
                    mask = (mae >= lo) & (mae <= hi)
                else:
                    mask = (mae >= lo) & (mae < hi)
                rows.append({
                    "outcome_group": grp_name,
                    "mae_bucket": label,
                    "trades": int(mask.sum()),
                    "pct_of_group": round(mask.sum() / n * 100, 2),
                })
        mae_buckets_export = pd.DataFrame(rows)

    # --- README ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    excluded_str = ", ".join(sorted(excluded_levels)) if excluded_levels else "(none)"
    n_orig = len(df) + n_skipped_no_retrace if entry_retrace_R > 0 else len(df)
    readme = f"""STRATEGY REPORT
Generated: {timestamp}
Dashboard: dashboard.py
Dataset:   {ds_name}

STRATEGY PARAMETERS
  Asset         : {asset_sel}
  Timeframe     : {tf_sel}
  Anchor        : {anchor_sel}
  MSS rule      : {rule_sel}
  TP target     : {tp_sel}
  Sessions      : {', '.join(sessions_sel)}
  Levels excl   : {excluded_str}
  Managed mode  : {use_managed}
  Hide opp-live : {hide_opp_live}
  Entry mode    : {entry_mode}
  Retracement   : {entry_retrace_R:.2f}R
  Risk/trade $  : {risk_sel}
  Starting bal  : {starting_sel}

HEADLINE RESULTS
  Total trades       : {len(df)}
  Wins / Losses      : {len(wins)} / {len(losses)}
  Win rate           : {win_rate:.2f}%
  TP hit rate ({tp_sel}) : {tp_hit_rate:.2f}%
  Drift gap          : {gap_pct:+.2f}%
  Total PnL          : ${total_pnl:,.2f}
  Pct return         : {pct_return:+.2f}%
  Mean R             : {df[tp_col].mean():.3f}
  Profit factor      : {pf:.2f}
  Max DD ($)         : {summary_data.get('global_max_dd_usd', 0):,.2f}

NOTES
  - All numbers reflect the strategy as configured at generation time.
  - "Trades" = entries after all filters (managed/skip/realistic-trim/retracement).
  - "managed_exit_reason" in 01_trades.csv shows how each trade exited.
  - If retracement > 0, {n_skipped_no_retrace} of {n_orig} original trades were skipped (never retraced).

FILE INDEX
  README.txt              this file
  01_trades.csv           one row per trade (full detail)
  02_summary.csv          single-row overall stats
  03_monthly.csv          per-month aggregation
  04_weekly.csv           per-week aggregation
  05_daily.csv            per-day aggregation (trading days only)
  06_yearly.csv           per-year aggregation
  07_quarterly.csv        per-quarter aggregation
  08_drawdowns.csv        every drawdown period (start, trough, recovery)
  09_streaks.csv          every win/loss streak (start, end, count, $)
  10_by_level_type.csv    performance per level type
  11_by_dow.csv           performance per day-of-week
  12_by_hour.csv          performance per entry hour
  13_by_direction.csv     bullish vs bearish
  14_by_session.csv       london vs ny_am
  15_by_sweeper.csv       MNQ-sweeper vs MES-sweeper
  16_mae_buckets.csv      MAE distribution (winners vs losers)
"""

    # --- Build ZIP ---
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", readme)
        z.writestr("01_trades.csv", trades_export.to_csv(index=False))
        z.writestr("02_summary.csv", summary_export.to_csv(index=False))
        z.writestr("03_monthly.csv", monthly_export.to_csv(index=False))
        z.writestr("04_weekly.csv", weekly_export.to_csv(index=False))
        z.writestr("05_daily.csv", daily_export.to_csv(index=False))
        z.writestr("06_yearly.csv", yearly_export.to_csv(index=False))
        z.writestr("07_quarterly.csv", quarterly_export.to_csv(index=False))
        z.writestr("08_drawdowns.csv", drawdowns_export.to_csv(index=False))
        z.writestr("09_streaks.csv", streaks_export.to_csv(index=False))
        z.writestr("10_by_level_type.csv", by_level_export.to_csv(index=False))
        z.writestr("11_by_dow.csv", by_dow_export.to_csv(index=False))
        z.writestr("12_by_hour.csv", by_hour_export.to_csv(index=False))
        z.writestr("13_by_direction.csv", by_direction_export.to_csv(index=False))
        z.writestr("14_by_session.csv", by_session_export.to_csv(index=False))
        z.writestr("15_by_sweeper.csv", by_sweeper_export.to_csv(index=False))
        z.writestr("16_mae_buckets.csv", mae_buckets_export.to_csv(index=False))
    buf.seek(0)
    return buf.getvalue()

# Build the ZIP only when user clicks. To avoid heavy work on every rerun,
# use a button + cache the bytes in session state.
zip_filename = f"strategy_report_{filter_tag}"
if entry_retrace_R > 0:
    zip_filename += f"_retrace{entry_retrace_R:.2f}R".replace(".","p")
zip_filename += f"_{datetime.datetime.now().strftime('%Y%m%d')}.zip"

st.markdown("### 📦 Full strategy report")
st.caption("ZIP archive with 16 files: trades, summary, time-based aggregations, drawdowns, streaks, "
           "and performance breakdowns by level type / day / hour / direction / session / sweeper / MAE.")

if st.button("Build report (may take a few seconds)", key="build_report"):
    with st.spinner("Building ZIP report..."):
        st.session_state.report_zip_bytes = build_strategy_zip()
        st.session_state.report_zip_filename = zip_filename

if st.session_state.get("report_zip_bytes"):
    st.download_button(
        "📥 Download full strategy report (ZIP)",
        data=st.session_state.report_zip_bytes,
        file_name=st.session_state.report_zip_filename,
        mime="application/zip",
        use_container_width=True,
    )


# ============================================================
# CHART DIALOG (for selected trade)
# ============================================================
@st.dialog("Trade chart")
def show_trade_chart(trade):
    smt_ts = trade["smt_birth_ts"]
    if pd.isna(smt_ts):
        st.error("Missing SMT birth timestamp")
        return

    session_date_str = (smt_ts.normalize() if smt_ts.hour < 18 else (smt_ts + pd.Timedelta(days=1)).normalize()).strftime("%Y-%m-%d")
    s_start, s_end = session_bounds(session_date_str)

    mnq_bars = load_bars("MNQ", tf_sel).loc[s_start:s_end]
    mes_bars = load_bars("MES", tf_sel).loc[s_start:s_end]
    mnq_lv = load_levels("MNQ"); mes_lv = load_levels("MES")
    mnq_1m = load_bars("MNQ","1m"); mes_1m = load_bars("MES","1m")

    smts_day = detect_smts_for_window(s_start, s_end, mnq_1m, mes_1m, mnq_lv, mes_lv)
    smts_drawn = smts_day[smts_day["status"] == "session_end"] if len(smts_day) else smts_day

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.04,
                          subplot_titles=(f"MNQ · {tf_sel}", f"MES · {tf_sel}"))

    def add_candles(df_bars, col):
        fig.add_trace(go.Candlestick(
            x=df_bars.index, open=df_bars["open"], high=df_bars["high"],
            low=df_bars["low"], close=df_bars["close"],
            increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
            decreasing=dict(line=dict(color="#000000"), fillcolor="#000000"),
            showlegend=False, hoverinfo="skip"), row=1, col=col)

    add_candles(mnq_bars, 1)
    add_candles(mes_bars, 2)

    for _, s in smts_drawn.iterrows():
        lt = s["level_type"]; find_high = lt in HIGH_LEVELS
        sweep_ts = s["birth_ts"]
        if s["sweeper"] == "MNQ":
            sw_1m_df, nc_1m_df = mnq_1m, mes_1m; sw_col, nc_col = 1, 2
            sw_ts_formed = s["mnq_ts_formed"]; sw_vf = s["mnq_valid_from"]
            nc_ts_formed = s["mes_ts_formed"]; nc_vf = s["mes_valid_from"]
        else:
            sw_1m_df, nc_1m_df = mes_1m, mnq_1m; sw_col, nc_col = 2, 1
            sw_ts_formed = s["mes_ts_formed"]; sw_vf = s["mes_valid_from"]
            nc_ts_formed = s["mnq_ts_formed"]; nc_vf = s["mnq_valid_from"]
        sw_win = formation_window(lt, sw_ts_formed, sw_vf)
        nc_win = formation_window(lt, nc_ts_formed, nc_vf)
        sw_t = find_extreme(sw_1m_df, sw_win[0], sw_win[1], find_high)
        nc_t = find_extreme(nc_1m_df, nc_win[0], nc_win[1], find_high)
        if sw_t is None or nc_t is None: continue
        sw_start = max(sw_t, s_start); nc_start = max(nc_t, s_start)
        yshift = 10 if find_high else -10
        yanchor = "bottom" if find_high else "top"
        for (ts0, y, col) in [(sw_start, s["level_price_sweeper"], sw_col),
                               (nc_start, s["level_price_nonconf"], nc_col)]:
            fig.add_shape(type="line", x0=ts0, x1=sweep_ts, y0=y, y1=y,
                           line=dict(color="black", width=1), row=1, col=col)
            fig.add_annotation(x=sweep_ts, y=y, text=f" {SHORT_LABEL.get(lt,lt)} ",
                                showarrow=False, font=dict(size=9, color="black"),
                                xanchor="left", yanchor=yanchor, yshift=yshift,
                                row=1, col=col)

    measured_asset = trade["measured_asset"]
    direction = trade["direction"]
    mss_ts = trade["mss_ts"]
    entry = trade["entry_price"]
    sl = trade["sl_price"]
    r_pts = abs(entry - sl)
    if measured_asset == "MNQ":
        bars_1m_asset = mnq_1m; col = 1
    else:
        bars_1m_asset = mes_1m; col = 2
    end = session_end_for(mss_ts)
    tp_target = None if tp_sel == "no_TP" else int(tp_sel[2:])

    mgd_exit_ts = trade.get("managed_exit_ts")
    mgd_reason = trade.get("managed_exit_reason")
    if use_managed and pd.notna(mgd_exit_ts):
        exit_ts = mgd_exit_ts
        try:
            exit_price = bars_1m_asset.loc[mgd_exit_ts]["close"]
        except KeyError:
            exit_price = entry
        outcome = mgd_reason if mgd_reason in ("sl_hit","tp_hit","session_end") else "kill"
    else:
        exit_ts, exit_price, outcome = track_trade(
            bars_1m_asset, mss_ts, entry, sl, r_pts, direction, end, tp_target
        )
    visual_end = exit_ts
    max_r = tp_target if tp_target is not None else 3
    r_levels = []
    for n in range(1, max_r + 1):
        r_levels.append((n, entry + n * r_pts if direction == "bullish" else entry - n * r_pts))
    if direction == "bullish":
        tp_top = r_levels[-1][1] if r_levels else entry; tp_bot = entry
        sl_top = entry; sl_bot = sl
    else:
        tp_top = entry; tp_bot = r_levels[-1][1] if r_levels else entry
        sl_top = sl; sl_bot = entry

    fig.add_shape(type="rect", x0=mss_ts, x1=visual_end, y0=sl_bot, y1=sl_top,
                   fillcolor="rgba(239,83,80,0.25)", line=dict(width=0), row=1, col=col)
    fig.add_shape(type="rect", x0=mss_ts, x1=visual_end, y0=tp_bot, y1=tp_top,
                   fillcolor="rgba(38,166,154,0.20)", line=dict(width=0), row=1, col=col)
    fig.add_shape(type="line", x0=mss_ts, x1=visual_end, y0=entry, y1=entry,
                   line=dict(color="#1976d2", width=1.5), row=1, col=col)
    fig.add_shape(type="line", x0=mss_ts, x1=visual_end, y0=sl, y1=sl,
                   line=dict(color="#d32f2f", width=1.5), row=1, col=col)
    for n, rv in r_levels:
        fig.add_shape(type="line", x0=mss_ts, x1=visual_end, y0=rv, y1=rv,
                       line=dict(color="#388e3c", width=1, dash="dash"), row=1, col=col)

    label_offset = pd.Timedelta(minutes=2)
    fig.add_annotation(x=visual_end + label_offset, y=entry,
                        text=f"Entry {entry:.2f}", showarrow=False,
                        font=dict(size=9, color="#1976d2"),
                        xanchor="left", yanchor="middle", row=1, col=col)
    fig.add_annotation(x=visual_end + label_offset, y=sl,
                        text=f"SL {sl:.2f}", showarrow=False,
                        font=dict(size=9, color="#d32f2f"),
                        xanchor="left", yanchor="middle", row=1, col=col)
    for n, rv in r_levels:
        fig.add_annotation(x=visual_end + label_offset, y=rv,
                            text=f"{n}R", showarrow=False,
                            font=dict(size=9, color="#388e3c"),
                            xanchor="left", yanchor="middle", row=1, col=col)
    exit_color_map = {
        "sl_hit": "#d32f2f",
        "tp_hit": "#388e3c",
        "session_end": "#757575",
        "opp_smt_kill": "#9c27b0",
        "opp_mss_kill": "#9c27b0",
        "kill": "#9c27b0",
    }
    exit_color = exit_color_map.get(outcome, "#757575")
    fig.add_shape(type="line", x0=exit_ts, x1=exit_ts, y0=sl_bot, y1=tp_top,
                   line=dict(color=exit_color, width=2, dash="dot"), row=1, col=col)

    fig.update_layout(
        height=600, paper_bgcolor="white", plot_bgcolor="white",
        xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False,
        hovermode=False, showlegend=False,
        margin=dict(l=50, r=90, t=40, b=40),
        font=dict(color="black"), title_font=dict(color="black"),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor="black",
                      tickfont=dict(color="black", size=9),
                      rangebreaks=[dict(bounds=[17, 18], pattern="hour")])
    fig.update_yaxes(showgrid=False, showline=True, linecolor="black",
                      tickfont=dict(color="black", size=9))

    st.plotly_chart(fig, use_container_width=True)

    # KEY LEVEL INFO
    level_type = trade.get("level_type", "—")
    sweeper_asset = trade.get("sweeper", "")
    level_id = trade.get("level_id", "")

    level_formed_str = "—"
    try:
        sweeper_levels = load_levels(sweeper_asset)
        sweeper_lvl = sweeper_levels[sweeper_levels["level_id"] == level_id]
        if len(sweeper_lvl):
            ts_formed = sweeper_lvl.iloc[0]["ts_formed"]
            if pd.notna(ts_formed):
                level_formed_str = ts_formed.strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    sweep_ts = trade.get("smt_birth_ts")
    sweep_ts_str = pd.to_datetime(sweep_ts).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M") \
        if pd.notna(sweep_ts) else "—"

    with st.container(border=True):
        st.markdown(
            f"""
            <div style='display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap'>
              <div style='flex:1;min-width:160px'>
                <div style='color:#888;font-size:12px;margin-bottom:4px'>Key level type</div>
                <div style='font-size:18px;font-weight:600'>{level_type}</div>
              </div>
              <div style='flex:1;min-width:200px'>
                <div style='color:#888;font-size:12px;margin-bottom:4px'>Level created at</div>
                <div style='font-size:18px;font-weight:600'>{level_formed_str}</div>
              </div>
              <div style='flex:1;min-width:200px'>
                <div style='color:#888;font-size:12px;margin-bottom:4px'>Swept by {sweeper_asset} at</div>
                <div style='font-size:18px;font-weight:600'>{sweep_ts_str}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("#### Trade details")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Direction", direction)
    t2.metric("Entry", f"{entry:.2f}")
    t3.metric("SL", f"{sl:.2f}")
    t4.metric("Outcome", outcome)
    t5, t6, t7, t8 = st.columns(4)
    t5.metric("R result", f"{trade[tp_col]:+.2f}R")
    t6.metric("PnL", f"${trade['pnl_usd']:+,.2f}")
    t7.metric("MFE", f"{trade['max_favorable_r']:.2f}R")
    t8.metric("MAE", f"{trade['max_adverse_r']:.2f}R")


if "selected_trade" in st.session_state:
    show_trade_chart(st.session_state.selected_trade)
    del st.session_state.selected_trade