"""
voligle_15m_after — production dashboard for the second locked strategy.

LOCKED STRATEGY:
  Asset       = Auto (first MSS, fallback if skipped)
  TF          = 15m
  Anchor      = after_smt
  Rule        = wick_break
  TP          = TP2  (managed mode ON)
  Sessions    = london + ny_am
  Levels      = ALL except AsiaON High/Low, PWH, PWL
  Risk        = $250 per trade
  Starting    = $25,000

Run:
  streamlit run voligle_15m_after.py --server.port 8508
"""

import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# LOCKED CONFIG — do not change here without intent
# ============================================================
DATA_FOLDER = "/Users/voligle/workspace/Trading/Trading_code/clause Backtester"
OUT_FOLDER = os.path.join(DATA_FOLDER, "out")
TICK_SIZE = 0.25
EPS = 1e-9

# Strategy params (LOCKED)
ASSET            = "Auto (first MSS)"
TF               = "15m"
ANCHOR           = "after_smt"
RULE             = "wick_break"
TP_LEVEL         = 2                   # TP2
TP_COL_RAW       = "r_tp2"
TP_COL_MGD       = "r_tp2_mgd"
HIT_COL          = "hit_2r"
HIT_TS_COL       = "hit_2r_ts"
SESSIONS         = ["london", "ny_am"]
EXCLUDE_LEVELS   = {"AsiaONHigh", "AsiaONLow", "PWH", "PWL"}
RISK_PER_TRADE   = 250.0
STARTING_BAL     = 25000.0

# Prop firm thresholds
DAILY_DD_LIMIT   = STARTING_BAL * 0.05    # 5% = $1,250 for $25k account
TRAILING_DD_LIM  = STARTING_BAL * 0.10    # 10% = $2,500
# ============================================================

HIGH_LEVELS = {"PDH","PWH","LondonHigh","AsiaKZHigh","AsiaONHigh","4H_SwingHigh"}
LOW_LEVELS  = {"PDL","PWL","LondonLow","AsiaKZLow","AsiaONLow","4H_SwingLow"}
ALL_LEVELS  = HIGH_LEVELS | LOW_LEVELS
LEVEL_TYPES_USED = sorted(ALL_LEVELS - EXCLUDE_LEVELS)

SHORT_LABEL = {
    "PDH": "PDH", "PDL": "PDL", "PWH": "PWH", "PWL": "PWL",
    "LondonHigh": "LH", "LondonLow": "LL",
    "AsiaKZHigh": "AKH", "AsiaKZLow": "AKL",
    "AsiaONHigh": "AOH", "AsiaONLow": "AOL",
    "4H_SwingHigh": "SWH", "4H_SwingLow": "SWL",
}
PERSISTENT_TYPES = {"PDH","PDL","PWH","PWL","4H_SwingHigh","4H_SwingLow"}

st.set_page_config(page_title="voligle 15m after", layout="wide")

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
# CORE — load data + apply locked filters
# ============================================================
measurement_files = sorted(glob.glob(os.path.join(OUT_FOLDER, "SMT_measurements_*.csv")))
if not measurement_files:
    st.error(f"No SMT_measurements_*.csv found in {OUT_FOLDER}. Run phase4.py first.")
    st.stop()
csv_path = measurement_files[-1]
df_all = load_measurements(csv_path, os.path.getmtime(csv_path))

# Apply LOCKED filters
df = df_all[
    (df_all["mss_timeframe"] == TF)
    & (df_all["anchor"] == ANCHOR)
    & (df_all["mss_rule"] == RULE)
    & (df_all["mss_found"] == True)
    & (df_all["final_outcome"].isin(["sl_hit","session_end"]))
    & (df_all["session"].isin(SESSIONS))
    & (df_all["level_type"].isin(LEVEL_TYPES_USED))
    & (df_all["auto_picked"] == True)
].copy()

# Use managed TP2 column (Pass 3 already filtered out skipped entries from auto_picked)
TP_COL = TP_COL_MGD if TP_COL_MGD in df.columns else TP_COL_RAW
df = df.dropna(subset=[TP_COL])

# Realistic mode: single position slot across both assets (Auto picks one per SMT)
def realistic_trim_single(trades, tp_col):
    if len(trades) == 0: return trades
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
                cands = []
                if pd.notna(tp_ts): cands.append(tp_ts)
                if pd.notna(sl_ts): cands.append(sl_ts)
                if not cands: cands.append(session_end)
                exit_ts = min(cands)
        exit_series.append(exit_ts)
    t["_exit_ts"] = exit_series
    t = t.sort_values("mss_ts").reset_index(drop=True)
    keep = []
    last_exit = None
    for _, row in t.iterrows():
        entry = row["mss_ts"]
        if last_exit is not None and entry < last_exit:
            keep.append(False); continue
        keep.append(True); last_exit = row["_exit_ts"]
    t["_keep"] = keep
    return t[t["_keep"]].drop(columns=["_keep","_exit_ts"])

df = realistic_trim_single(df, TP_COL)

if len(df) == 0:
    st.error("No trades match this strategy on the current dataset.")
    st.stop()

df["pnl_usd"] = df[TP_COL] * RISK_PER_TRADE
df = df.sort_values("mss_ts").reset_index(drop=True)
df["cum_pnl"] = df["pnl_usd"].cumsum()
df["balance"] = STARTING_BAL + df["cum_pnl"]
df["trade_date"] = df["mss_ts"].dt.date


# ============================================================
# ENTRY RETRACEMENT DROPDOWN (defined BEFORE simulation runs)
# ============================================================
st.markdown("# 🔒 voligle_15m_after — Production Strategy Dashboard")

ec1, ec2 = st.columns([1, 3])
with ec1:
    entry_retrace_R = st.selectbox(
        "Entry retracement (R)",
        [0.0, 0.25, 0.5, 0.75],
        index=0,
        format_func=lambda x: f"{x:.2f}R" + (" (normal MSS-bar entry)" if x == 0.0 else " (wait for retracement)"),
        help="Simulate waiting for price to retrace X·R against the original entry "
             "before filling. SL stays at the leg extreme; TP2 recomputes from the "
             "new entry. Trades that never retrace to this level are skipped.",
    )
with ec2:
    if entry_retrace_R > 0:
        st.info(
            f"📐 Simulation: enter at MSS_close − {entry_retrace_R:.2f}R for bullish "
            f"(or +{entry_retrace_R:.2f}R for bearish). SL unchanged; TP2 = new_entry ± 2·new_R. "
            f"Trades that don't retrace to this level are excluded entirely.",
            icon="🧪",
        )


# ============================================================
# ENTRY-RETRACEMENT SIMULATION
# ============================================================
def session_end_of(ts):
    """Return 17:00 NY of the session day for a timestamp."""
    if ts.hour >= 18:
        return ts.normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=17)
    return ts.normalize() + pd.Timedelta(hours=17)


def simulate_retraced_entries(trades_df, retrace_r):
    """
    For each trade, simulate "wait for price to retrace `retrace_r` * R against
    the original entry, then fill". Compute new entry, R distance, TP2 target,
    and walk 1m bars to determine outcome.

    SL stays at the original sl_price.
    Kill timestamps from Phase 4 are preserved (they're independent of our entry).
    Risk is fixed at $RISK_PER_TRADE; position size implicitly scales.

    Returns: a new DataFrame with the same shape as df, but with rewritten:
      - entry_price, r_points
      - mss_ts (kept as the original signal time, used for sorting/calendar)
      - "fill_ts" added (when the simulated fill actually happened)
      - new "outcome_r" column (the R outcome for THIS trade given retraced entry)
      - new "outcome_reason" column (tp2_hit, sl_hit, opp_smt_kill, opp_mss_kill, session_end, no_fill)

    Trades that never retrace are EXCLUDED from the result.
    """
    if retrace_r == 0:
        # No retracement — return df with placeholder columns
        out = trades_df.copy()
        out["fill_ts"] = out["mss_ts"]
        out["entry_price_sim"] = out["entry_price"]
        out["r_points_sim"] = out["r_points"]
        out["outcome_r"] = out[TP_COL]
        # Map the existing managed_exit_reason
        out["outcome_reason"] = out["managed_exit_reason"].fillna("session_end")
        return out

    # Pre-load 1m bars
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
        sess_end = session_end_of(mss_ts)

        # Determine the existing kill timestamp (if any) from Phase 4 columns.
        # Kill applies as a HARD exit at that bar's close, regardless of the simulated entry.
        kill_ts = None
        kill_close = None
        kill_reason = None
        mgd_reason = t.get("managed_exit_reason", "")
        mgd_exit_ts = t.get("managed_exit_ts")
        if mgd_reason in ("opp_smt_kill", "opp_mss_kill") and pd.notna(mgd_exit_ts):
            kill_ts = mgd_exit_ts
            kill_reason = mgd_reason

        # Compute retracement target price (where we'd want to be filled)
        if direction == "bullish":
            retrace_price = orig_entry - retrace_r * orig_r
        else:
            retrace_price = orig_entry + retrace_r * orig_r

        # Walk 1m bars from MSS+1m onward: find first bar that touches retrace_price
        try:
            sub = bars_1m[asset].loc[mss_ts + pd.Timedelta(minutes=1):sess_end]
        except KeyError:
            sub = bars_1m[asset].iloc[0:0]
        if len(sub) == 0:
            continue

        fill_ts = None
        for ts, bar in sub.iterrows():
            # Check kill BEFORE checking fill. If a kill fires before we're filled,
            # we never enter the trade.
            if kill_ts is not None and ts >= kill_ts:
                fill_ts = None
                break
            # Check SL touch BEFORE fill (SL hit before fill = no trade)
            if direction == "bullish":
                if bar["low"] <= sl + EPS:
                    fill_ts = None
                    break
                if bar["low"] <= retrace_price + EPS:
                    fill_ts = ts
                    break
            else:
                if bar["high"] >= sl - EPS:
                    fill_ts = None
                    break
                if bar["high"] >= retrace_price - EPS:
                    fill_ts = ts
                    break

        if fill_ts is None:
            continue

        # New entry, new R, new TP2 target
        new_entry = retrace_price
        new_r = abs(new_entry - sl)
        if new_r < TICK_SIZE - EPS:
            continue
        if direction == "bullish":
            new_tp2 = new_entry + 2 * new_r
        else:
            new_tp2 = new_entry - 2 * new_r

        # Walk bars from fill_ts onward: first of {SL, TP2, kill, session_end}
        try:
            sub2 = bars_1m[asset].loc[fill_ts + pd.Timedelta(minutes=1):sess_end]
        except KeyError:
            sub2 = bars_1m[asset].iloc[0:0]

        outcome_r = None
        outcome_reason = None
        outcome_exit_ts = None  # the actual exit timestamp of the simulated trade

        for ts, bar in sub2.iterrows():
            # Kill exits at bar close (inferior to TP/SL precedence on same bar)
            if kill_ts is not None and ts >= kill_ts:
                if direction == "bullish":
                    outcome_r = (bar["close"] - new_entry) / new_r
                else:
                    outcome_r = (new_entry - bar["close"]) / new_r
                outcome_reason = kill_reason or "kill"
                outcome_exit_ts = ts
                break

            if direction == "bullish":
                hit_tp = bar["high"] >= new_tp2 - EPS
                hit_sl = bar["low"] <= sl + EPS
            else:
                hit_tp = bar["low"] <= new_tp2 + EPS
                hit_sl = bar["high"] >= sl - EPS

            if hit_tp and hit_sl:
                # Same bar both TP and SL touched — conservative: SL first
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
                outcome_r = 2.0
                outcome_reason = "tp2_hit"
                outcome_exit_ts = ts
                break

        if outcome_r is None:
            # Reached session end without TP/SL
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


# Apply simulation if user picked a non-zero retracement
if entry_retrace_R > 0:
    with st.spinner(f"Simulating {entry_retrace_R:.2f}R retraced entries..."):
        df_sim = simulate_retraced_entries(df, entry_retrace_R)
    if len(df_sim) == 0:
        st.error(f"No trades retraced to {entry_retrace_R:.2f}R. Try a smaller retracement value.")
        st.stop()
    n_original = len(df)
    n_filled = len(df_sim)
    n_skipped_no_retrace = n_original - n_filled
    st.caption(
        f"📊 Of {n_original} original trades, **{n_filled} filled** at the {entry_retrace_R:.2f}R retraced entry "
        f"({n_filled/n_original*100:.1f}%). **{n_skipped_no_retrace} trades skipped** (price never retraced "
        f"or SL/kill hit before fill)."
    )

    # Replace df with simulated trades. Override key columns so all downstream
    # analyses (MAE, calendar, etc.) use the simulated outcomes.
    df = df_sim.copy()
    df["entry_price"] = df["entry_price_sim"]
    df["r_points"] = df["r_points_sim"]
    df["mss_ts"] = df["fill_ts"]  # use fill time as the new "entry timestamp"
    df[TP_COL] = df["outcome_r"]
    df["managed_exit_reason"] = df["outcome_reason"]
    df["managed_exit_ts"] = df["exit_ts_sim"]  # override so duration is correct
    # Update hit_2r flag to reflect new outcome
    df[HIT_COL] = (df["outcome_r"] >= 2.0 - 0.001)
    # max_favorable_r and max_adverse_r are not directly comparable post-sim;
    # set to zero placeholders so existing UI code doesn't crash
    if "max_favorable_r" not in df.columns:
        df["max_favorable_r"] = 0.0
    if "max_adverse_r" not in df.columns:
        df["max_adverse_r"] = 0.0
else:
    # No simulation, df stays as-is
    pass

df["pnl_usd"] = df[TP_COL] * RISK_PER_TRADE
df = df.sort_values("mss_ts").reset_index(drop=True)
df["cum_pnl"] = df["pnl_usd"].cumsum()
df["balance"] = STARTING_BAL + df["cum_pnl"]
df["trade_date"] = df["mss_ts"].dt.date


# ============================================================
# COMPUTE MAE PROPERLY — from 1m bars, between entry and outcome ts
# ============================================================
@st.cache_data
def compute_mae_per_trade(df_trades_hash, csv_path_hash, mtime):
    """
    For each trade, compute true MAE in points from entry to its 'outcome timestamp':
      - TP2 hit:    MAE from mss_ts to hit_2r_ts
      - Drift win:  MAE from mss_ts to managed_exit_ts (= session_end since no SL/TP/kill)
      - Loser:      MAE from mss_ts to managed_exit_ts (= sl_hit_ts or kill_ts)
    Returns a DataFrame indexed by mss_ts with columns:
      mae_points, mae_ticks, mae_usd, outcome_group ('tp2_hit'|'drift_win'|'loser'),
      mae_window_ts (the timestamp used as end of MAE window)
    """
    # Re-derive df from CSV to keep cache key simple (mtime-based)
    base = load_measurements(csv_path_hash, mtime)
    rows_dict = {pd.Timestamp(r["mss_ts"]): r for _, r in df_trades_hash.iterrows()}

    # Pre-load 1m bars per asset
    bars_1m = {a: load_bars(a, "1m") for a in ("MNQ","MES")}

    out = []
    for _, t in df_trades_hash.iterrows():
        asset = t["measured_asset"]
        direction = t["direction"]
        entry = float(t["entry_price"])
        sl = float(t["sl_price"])
        r_pts = abs(entry - sl)
        mss_ts = t["mss_ts"]
        if pd.isna(mss_ts) or r_pts < TICK_SIZE - EPS:
            continue

        # Determine outcome group + end timestamp for MAE window
        hit_tp2 = bool(t.get(HIT_COL, False))
        sl_ts = t.get("sl_hit_ts")
        managed_exit_ts = t.get("managed_exit_ts")
        managed_reason = t.get("managed_exit_reason", "")
        tp_r_value = float(t[TP_COL]) if pd.notna(t[TP_COL]) else 0.0

        if hit_tp2 and tp_r_value > 0:
            # TP2 hit — measure MAE from entry to TP2 hit time
            end_ts = t.get(HIT_TS_COL)
            group = "tp2_hit"
        elif tp_r_value > 0:
            # Drift win (managed exit at session_end with positive R, never hit TP2)
            end_ts = managed_exit_ts if pd.notna(managed_exit_ts) else t.get("trade_exit_ts_if_held")
            group = "drift_win"
        else:
            # Loser — MAE from entry to managed_exit_ts (SL hit or kill)
            end_ts = managed_exit_ts if pd.notna(managed_exit_ts) else sl_ts
            group = "loser"

        if pd.isna(end_ts):
            continue

        # Walk 1m bars [mss_ts+1m, end_ts] and compute MAE in points
        try:
            sub = bars_1m[asset].loc[mss_ts + pd.Timedelta(minutes=1):end_ts]
        except KeyError:
            continue
        if len(sub) == 0:
            mae_pts = 0.0
        else:
            if direction == "bullish":
                # Adverse = how far BELOW entry price went
                worst_low = sub["low"].min()
                mae_pts = max(0.0, entry - worst_low)
            else:
                # Adverse = how far ABOVE entry price went
                worst_high = sub["high"].max()
                mae_pts = max(0.0, worst_high - entry)

        mae_ticks = mae_pts / TICK_SIZE
        mae_r = mae_pts / r_pts
        mae_usd = mae_r * RISK_PER_TRADE

        # MAE before reaching +1R: from mss_ts to hit_1r_ts (only if 1R was reached)
        mae_before_1r_pts = None
        mae_before_1r_r = None
        hit_1r_ts = t.get("hit_1r_ts")
        if pd.notna(hit_1r_ts):
            try:
                sub_1r = bars_1m[asset].loc[mss_ts + pd.Timedelta(minutes=1):hit_1r_ts]
            except KeyError:
                sub_1r = bars_1m[asset].iloc[0:0]
            if len(sub_1r) > 0:
                if direction == "bullish":
                    worst_low_1r = sub_1r["low"].min()
                    mae_before_1r_pts = max(0.0, entry - worst_low_1r)
                else:
                    worst_high_1r = sub_1r["high"].max()
                    mae_before_1r_pts = max(0.0, worst_high_1r - entry)
                mae_before_1r_r = mae_before_1r_pts / r_pts

        out.append({
            "mss_ts": mss_ts,
            "outcome_group": group,
            "mae_points": round(mae_pts, 2),
            "mae_ticks": round(mae_ticks, 1),
            "mae_r": round(mae_r, 3),
            "mae_usd": round(mae_usd, 2),
            "mae_window_end_ts": end_ts,
            "r_points": round(r_pts, 2),
            "reached_1r": pd.notna(hit_1r_ts),
            "mae_before_1r_pts": round(mae_before_1r_pts, 2) if mae_before_1r_pts is not None else None,
            "mae_before_1r_r": round(mae_before_1r_r, 3) if mae_before_1r_r is not None else None,
        })
    return pd.DataFrame(out)

# Compute MAE
mae_df = compute_mae_per_trade(df, csv_path, os.path.getmtime(csv_path))
df = df.merge(
    mae_df[["mss_ts","outcome_group","mae_points","mae_ticks","mae_r","mae_usd",
            "reached_1r","mae_before_1r_pts","mae_before_1r_r"]],
    on="mss_ts", how="left",
)


# ============================================================
# STRATEGY HEADER (LOCKED PARAMS)
# ============================================================
st.markdown(
    f"""
    <div style='background:#0e2030;border:1px solid #1976d2;border-radius:8px;padding:12px 16px;margin-bottom:12px'>
      <div style='display:flex;flex-wrap:wrap;gap:24px;font-size:14px'>
        <div><span style='color:#888'>Asset:</span> <b>{ASSET}</b></div>
        <div><span style='color:#888'>TF:</span> <b>{TF}</b></div>
        <div><span style='color:#888'>Anchor:</span> <b>{ANCHOR}</b></div>
        <div><span style='color:#888'>Rule:</span> <b>{RULE}</b></div>
        <div><span style='color:#888'>Target:</span> <b>TP{TP_LEVEL} (managed)</b></div>
        <div><span style='color:#888'>Sessions:</span> <b>{' + '.join(SESSIONS)}</b></div>
        <div><span style='color:#888'>Risk:</span> <b>${int(RISK_PER_TRADE)}/trade</b></div>
        <div><span style='color:#888'>Start:</span> <b>${int(STARTING_BAL):,}</b></div>
      </div>
      <div style='color:#888;font-size:12px;margin-top:8px'>
        Levels excluded: {', '.join(sorted(EXCLUDE_LEVELS))}
        &nbsp;·&nbsp; Dataset: {os.path.basename(csv_path)}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 2. HEADLINE METRICS
# ============================================================
st.markdown("## Headline metrics")

wins = df[df[TP_COL] > 0]
losses = df[df[TP_COL] <= 0]
win_rate = len(wins) / len(df) * 100 if len(df) else 0
total_pnl = df["pnl_usd"].sum()
final_balance = STARTING_BAL + total_pnl
pct_return = (total_pnl / STARTING_BAL) * 100
tp_hit_count = int(df[HIT_COL].sum()) if HIT_COL in df.columns else 0
tp_hit_rate = tp_hit_count / len(df) * 100
drift_gap = win_rate - tp_hit_rate

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total PnL", f"${total_pnl:,.2f}", f"{pct_return:+.2f}%")
c2.metric("Account Balance", f"${final_balance:,.2f}", f"{pct_return:+.2f}%")
c3.metric("Win Rate", f"{win_rate:.2f}%")
c4.metric(f"TP{TP_LEVEL} Hit Rate", f"{tp_hit_rate:.2f}%",
            f"{drift_gap:+.1f}% drift")
c5.metric("Total Trades", f"{len(df)}", f"{len(wins)}W / {len(losses)}L")
c6.metric("Avg Win / Avg Loss",
           f"${wins['pnl_usd'].mean():.0f}" if len(wins) else "—",
           f"${losses['pnl_usd'].mean():.0f}" if len(losses) else "—")

# ============================================================
# 3. EQUITY CURVE
# ============================================================
st.markdown("## Equity curve")
fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=df["mss_ts"], y=df["balance"], mode="lines+markers",
    line=dict(color="#1976d2", width=2),
    marker=dict(size=4, color="#1976d2"),
    name="Balance",
))
# Prop firm DD reference lines
fig_eq.add_hline(y=STARTING_BAL, line_dash="dot", line_color="#888",
                   annotation_text=f"Start ${int(STARTING_BAL):,}")
fig_eq.add_hline(y=STARTING_BAL - TRAILING_DD_LIM, line_dash="dash", line_color="#d32f2f",
                   annotation_text=f"−10% prop fail (${int(STARTING_BAL - TRAILING_DD_LIM):,})",
                   annotation_position="bottom right")
fig_eq.add_hline(y=STARTING_BAL - DAILY_DD_LIMIT, line_dash="dot", line_color="#ef5350",
                   annotation_text=f"−5% daily ($-{int(DAILY_DD_LIMIT):,})",
                   annotation_position="bottom right")
fig_eq.update_layout(
    height=400, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    xaxis=dict(showgrid=False, linecolor="#444", tickfont=dict(color="#ccc")),
    yaxis=dict(showgrid=True, gridcolor="#222", linecolor="#444",
                tickfont=dict(color="#ccc"), tickprefix="$"),
    font=dict(color="#ccc"),
    margin=dict(l=50, r=30, t=20, b=40),
    showlegend=False,
)
st.plotly_chart(fig_eq, use_container_width=True)

# ============================================================
# 4. R-MULTIPLE DISTRIBUTION
# ============================================================
st.markdown("## R-multiple distribution")
st.caption("How trades distribute across R outcomes. The TP2 spike (+2R) is what hit-rate is about.")

r_values = df[TP_COL].values
fig_hist = go.Figure(data=[go.Histogram(
    x=r_values, nbinsx=40,
    marker=dict(color="#1976d2", line=dict(color="#444", width=1)),
)])
fig_hist.add_vline(x=0, line_dash="dash", line_color="#888")
fig_hist.add_vline(x=2.0, line_dash="dash", line_color="#26a69a",
                     annotation_text="TP2", annotation_position="top right")
fig_hist.add_vline(x=-1.0, line_dash="dash", line_color="#d32f2f",
                     annotation_text="SL", annotation_position="top right")
fig_hist.update_layout(
    height=300, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    xaxis=dict(title="R outcome", linecolor="#444",
                tickfont=dict(color="#ccc"), gridcolor="#222"),
    yaxis=dict(title="# trades", linecolor="#444",
                tickfont=dict(color="#ccc"), gridcolor="#222"),
    font=dict(color="#ccc"),
    margin=dict(l=50, r=30, t=20, b=50),
    showlegend=False,
)
st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================
# 5. MAE ANALYSIS BY OUTCOME GROUP
# ============================================================
st.markdown("## MAE analysis — heat before outcome")
st.caption("Maximum Adverse Excursion: how far against you the trade went before its final outcome. "
            "Measured in points/ticks/$ from entry until the outcome timestamp (TP2 hit, session end, or SL).")

groups = {
    "tp2_hit": {"label": "TP2 hits", "color": "#26a69a",
                  "desc": "Trades where price physically touched TP2. MAE measured entry → TP2 hit."},
    "drift_win": {"label": "Drift wins", "color": "#1976d2",
                    "desc": "Wins that closed positive at session end without touching TP2."},
    "loser":    {"label": "Losers",    "color": "#d32f2f",
                  "desc": "Trades that hit SL or were killed by opposing-SMT/MSS rules."},
    "all":      {"label": "All trades", "color": "#888",
                  "desc": "Combined view across all outcomes."},
}

def mae_stats(sub_df):
    if len(sub_df) == 0:
        return None
    return {
        "n": len(sub_df),
        "median_pts":  sub_df["mae_points"].median(),
        "mean_pts":    sub_df["mae_points"].mean(),
        "p25_pts":     sub_df["mae_points"].quantile(0.25),
        "p75_pts":     sub_df["mae_points"].quantile(0.75),
        "p90_pts":     sub_df["mae_points"].quantile(0.90),
        "max_pts":     sub_df["mae_points"].max(),
        "median_ticks": sub_df["mae_ticks"].median(),
        "mean_ticks":   sub_df["mae_ticks"].mean(),
        "p75_ticks":    sub_df["mae_ticks"].quantile(0.75),
        "p90_ticks":    sub_df["mae_ticks"].quantile(0.90),
        "median_usd":  sub_df["mae_usd"].median(),
        "mean_usd":    sub_df["mae_usd"].mean(),
        "p75_usd":     sub_df["mae_usd"].quantile(0.75),
        "p90_usd":     sub_df["mae_usd"].quantile(0.90),
        "median_r":    sub_df["mae_r"].median(),
        "p75_r":       sub_df["mae_r"].quantile(0.75),
        "p90_r":       sub_df["mae_r"].quantile(0.90),
    }

stats_by_group = {}
for g_key in ("tp2_hit","drift_win","loser"):
    sub = df[df["outcome_group"] == g_key]
    stats_by_group[g_key] = mae_stats(sub)
stats_by_group["all"] = mae_stats(df.dropna(subset=["mae_points"]))

# Bucket MAE in R into 4 ranges: 0 to -0.25, -0.25 to -0.5, -0.5 to -0.75, -0.75 to -0.99
# (we measure MAE as positive number representing how far adverse, so buckets are 0-0.25, etc.)
MAE_BUCKETS = [
    ("0 to 0.25R",    0.00, 0.25),
    ("0.25 to 0.5R",  0.25, 0.50),
    ("0.5 to 0.75R",  0.50, 0.75),
    ("0.75 to 0.99R", 0.75, 0.99),
]

def bucket_distribution(sub_df):
    """Return list of (label, count, pct) for each MAE bucket. Cap MAE at 0.99R."""
    if len(sub_df) == 0:
        return []
    # Cap MAE at 0.99R for losers (since SL = 1.0R is the loss case, not adverse to a winner)
    mae = sub_df["mae_r"].clip(upper=0.99)
    n = len(mae)
    out = []
    for label, lo, hi in MAE_BUCKETS:
        # Inclusive lo, exclusive hi (except last bucket includes 0.99)
        if hi >= 0.99:
            mask = (mae >= lo) & (mae <= hi)
        else:
            mask = (mae >= lo) & (mae < hi)
        cnt = int(mask.sum())
        pct = cnt / n * 100 if n else 0
        out.append((label, cnt, pct))
    return out

# Render four panels (TP2, drift win, loser, all) as bucket distributions
mae_cols = st.columns(4)
for i, g_key in enumerate(("tp2_hit","drift_win","loser","all")):
    info = groups[g_key]
    if g_key == "all":
        sub = df.dropna(subset=["mae_r"])
    else:
        sub = df[df["outcome_group"] == g_key]
    buckets = bucket_distribution(sub)
    with mae_cols[i]:
        with st.container(border=True):
            st.markdown(f"### {info['label']}")
            st.caption(info["desc"])
            if len(sub) == 0:
                st.write("No trades.")
                continue
            st.markdown(f"**Total trades:** {len(sub)}")
            # Build colored bucket rows
            rows_html = []
            for label, cnt, pct in buckets:
                # Bar fill width proportional to pct (max 100%)
                bar_w = min(100, pct)
                rows_html.append(
                    f"""
                    <div style='margin:6px 0'>
                      <div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:2px'>
                        <span>{label}</span>
                        <span><b>{cnt}</b> trades · {pct:.1f}%</span>
                      </div>
                      <div style='background:#1a1a1a;border-radius:3px;height:8px;overflow:hidden'>
                        <div style='background:{info["color"]};width:{bar_w}%;height:100%'></div>
                      </div>
                    </div>
                    """
                )
            st.markdown("\n".join(rows_html), unsafe_allow_html=True)

# ============================================================
# 5b. MAE BEFORE REACHING +1R (entry quality analysis)
# ============================================================
st.markdown("## MAE before reaching +1R")
st.caption("How far against you the trade went BEFORE first touching +1R. "
            "Useful for thinking about better entry prices — if heat-before-1R is small, "
            "you can potentially enter closer to the SL leg extreme for better R/R.")

# Two subsets:
#   A. TP2 hits only (winners that ran to +2R)
#   B. ALL trades that reached +1R (broader sample, including some that gave back)
sub_a = df[(df["outcome_group"] == "tp2_hit") & (df["reached_1r"] == True) & df["mae_before_1r_r"].notna()]
sub_b = df[(df["reached_1r"] == True) & df["mae_before_1r_r"].notna()]

def bucket_mae_before_1r(sub_df):
    if len(sub_df) == 0:
        return []
    mae = sub_df["mae_before_1r_r"].clip(upper=0.99)
    n = len(mae)
    out = []
    for label, lo, hi in MAE_BUCKETS:
        if hi >= 0.99:
            mask = (mae >= lo) & (mae <= hi)
        else:
            mask = (mae >= lo) & (mae < hi)
        cnt = int(mask.sum())
        pct = cnt / n * 100 if n else 0
        out.append((label, cnt, pct))
    return out

panels = [
    ("A. TP2 hits only", "#26a69a", sub_a,
        "Winners that ran all the way to +2R. MAE measured entry → first touch of +1R."),
    ("B. All trades that hit +1R", "#1976d2", sub_b,
        "Every trade where price touched +1R, regardless of final outcome. Includes trades that gave back to SL or drifted."),
]

mae1r_cols = st.columns(2)
for i, (title, color, sub, desc) in enumerate(panels):
    buckets = bucket_mae_before_1r(sub)
    with mae1r_cols[i]:
        with st.container(border=True):
            st.markdown(f"### {title}")
            st.caption(desc)
            if len(sub) == 0:
                st.write("No trades.")
                continue
            st.markdown(f"**Total trades:** {len(sub)}")
            rows_html = []
            for label, cnt, pct in buckets:
                bar_w = min(100, pct)
                rows_html.append(
                    f"""
                    <div style='margin:6px 0'>
                      <div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:2px'>
                        <span>{label}</span>
                        <span><b>{cnt}</b> trades · {pct:.1f}%</span>
                      </div>
                      <div style='background:#1a1a1a;border-radius:3px;height:8px;overflow:hidden'>
                        <div style='background:{color};width:{bar_w}%;height:100%'></div>
                      </div>
                    </div>
                    """
                )
            st.markdown("\n".join(rows_html), unsafe_allow_html=True)
            # Also show median + p75 + max for quick reference
            med = sub["mae_before_1r_r"].median()
            p75 = sub["mae_before_1r_r"].quantile(0.75)
            mx  = sub["mae_before_1r_r"].max()
            st.markdown(
                f"<div style='color:#888;font-size:12px;margin-top:8px'>"
                f"Median: <b>{med:.2f}R</b> · P75: <b>{p75:.2f}R</b> · Max: <b>{mx:.2f}R</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ============================================================
# 6. RR METRICS
# ============================================================
st.markdown("## Risk-Reward metrics")
avg_rr = df[TP_COL].mean()
max_rr = df[TP_COL].max()
ideal_avg_rr = df["max_favorable_r"].mean()
max_ideal_rr = df["max_favorable_r"].max()
expectancy_usd = avg_rr * RISK_PER_TRADE
avg_win_usd = wins["pnl_usd"].mean() if len(wins) else 0
avg_loss_usd = losses["pnl_usd"].mean() if len(losses) else 0
pf = (wins["pnl_usd"].sum() / abs(losses["pnl_usd"].sum())) if len(losses) and losses["pnl_usd"].sum() != 0 else float("inf")

rr1, rr2, rr3, rr4 = st.columns(4)
rr1.metric("Average R", f"{avg_rr:.2f}")
rr2.metric("Max R", f"{max_rr:.2f}")
rr3.metric("Ideal Avg R (MFE)", f"{ideal_avg_rr:.2f}")
rr4.metric("Max Ideal R", f"{max_ideal_rr:.2f}")

rr5, rr6, rr7 = st.columns(3)
rr5.metric("Expectancy", f"${expectancy_usd:,.2f}",
            f"${avg_win_usd:,.0f} / ${avg_loss_usd:,.0f}")
rr6.metric("Profit Factor", f"{pf:.2f}" if pf != float("inf") else "∞")
rr7.metric("Profit per trade (avg)", f"${df['pnl_usd'].mean():.2f}")

# ============================================================
# 7. WINNERS / LOSERS PANELS
# ============================================================
st.markdown("## Winners and Losers")

def consecutive_streaks(series):
    max_streak = 0; cur = 0; streaks = []
    for v in series:
        if v:
            cur += 1
            if cur > max_streak: max_streak = cur
        else:
            if cur > 0: streaks.append(cur)
            cur = 0
    if cur > 0: streaks.append(cur)
    avg = sum(streaks) / len(streaks) if streaks else 0
    return max_streak, avg

def trade_duration_minutes(row):
    entry = row["mss_ts"]
    if pd.notna(row.get("managed_exit_ts")):
        exit_ts = row["managed_exit_ts"]
        if pd.isna(exit_ts) or pd.isna(entry): return None
        return (exit_ts - entry).total_seconds() / 60
    return None

df["_duration_min"] = df.apply(trade_duration_minutes, axis=1)

# Refresh wins/losses subsets so they include _duration_min
wins = df[df[TP_COL] > 0]
losses = df[df[TP_COL] <= 0]

is_win = (df[TP_COL] > 0).values
is_loss = (df[TP_COL] <= 0).values
max_win_streak, avg_win_streak = consecutive_streaks(is_win)
max_loss_streak, avg_loss_streak = consecutive_streaks(is_loss)

def fmt_dur(m):
    if m is None or pd.isna(m): return "—"
    h = int(m // 60); mm = int(m % 60)
    return f"{h}h {mm}m"

wl1, wl2 = st.columns(2)
with wl1:
    with st.container(border=True):
        st.markdown("### 🟢 Winners")
        st.markdown(f"**Total winners:** {len(wins)}")
        if len(wins):
            st.markdown(f"**Best win:** {wins[TP_COL].max():.2f}R (${wins['pnl_usd'].max():,.0f})")
            st.markdown(f"**Average win:** {wins[TP_COL].mean():.2f}R (${wins['pnl_usd'].mean():,.0f})")
            st.markdown(f"**Average duration:** {fmt_dur(wins['_duration_min'].mean())}")
        st.markdown(f"**Max consecutive wins:** {max_win_streak}")
        st.markdown(f"**Avg consecutive wins:** {avg_win_streak:.2f}")
with wl2:
    with st.container(border=True):
        st.markdown("### 🔴 Losers")
        st.markdown(f"**Total losers:** {len(losses)}")
        if len(losses):
            st.markdown(f"**Worst loss:** {losses[TP_COL].min():.2f}R (${losses['pnl_usd'].min():,.0f})")
            st.markdown(f"**Average loss:** {losses[TP_COL].mean():.2f}R (${losses['pnl_usd'].mean():,.0f})")
            st.markdown(f"**Average duration:** {fmt_dur(losses['_duration_min'].mean())}")
        st.markdown(f"**Max consecutive losses:** {max_loss_streak}")
        st.markdown(f"**Avg consecutive losses:** {avg_loss_streak:.2f}")


# ============================================================
# 8. (REMOVED — bullish vs bearish donuts)
# ============================================================

# ============================================================
# EXTRAS — time of day, day of week, level type, drawdown timeline, frequency
# ============================================================
st.divider()
st.markdown("## Extra analyses")

# ---- Time of day (hour of MSS entry) ----
st.markdown("### Time of day — performance by entry hour")
df["entry_hour"] = df["mss_ts"].dt.hour
hour_grp = df.groupby("entry_hour").agg(
    trades=(TP_COL, "size"),
    wins=(TP_COL, lambda x: (x > 0).sum()),
    mean_r=(TP_COL, "mean"),
    total_r=(TP_COL, "sum"),
)
hour_grp["win_rate_pct"] = hour_grp["wins"] / hour_grp["trades"] * 100
hour_grp = hour_grp.reset_index().sort_values("entry_hour")

fig_hour = go.Figure()
fig_hour.add_trace(go.Bar(
    x=hour_grp["entry_hour"], y=hour_grp["total_r"],
    marker=dict(color=["#26a69a" if v > 0 else "#d32f2f" for v in hour_grp["total_r"]]),
    text=[f"{int(t)} trades<br>{wr:.0f}% win" for t, wr in zip(hour_grp["trades"], hour_grp["win_rate_pct"])],
    textposition="outside",
    name="Total R",
))
fig_hour.update_layout(
    height=350, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    xaxis=dict(title="Entry hour (NY)", linecolor="#444",
                tickfont=dict(color="#ccc"), gridcolor="#222",
                tickmode="linear", dtick=1),
    yaxis=dict(title="Total R", linecolor="#444",
                tickfont=dict(color="#ccc"), gridcolor="#222"),
    font=dict(color="#ccc"),
    margin=dict(l=50, r=30, t=30, b=50),
    showlegend=False,
)
st.plotly_chart(fig_hour, use_container_width=True)

# ---- Day of week ----
st.markdown("### Day of week")
df["dow"] = df["mss_ts"].dt.day_name()
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_grp = df.groupby("dow").agg(
    trades=(TP_COL, "size"),
    wins=(TP_COL, lambda x: (x > 0).sum()),
    total_r=(TP_COL, "sum"),
    total_usd=("pnl_usd", "sum"),
)
dow_grp["win_rate_pct"] = dow_grp["wins"] / dow_grp["trades"] * 100
dow_grp = dow_grp.reindex(dow_order).dropna()
dow_grp = dow_grp.reset_index()

fig_dow = go.Figure()
fig_dow.add_trace(go.Bar(
    x=dow_grp["dow"], y=dow_grp["total_usd"],
    marker=dict(color=["#26a69a" if v > 0 else "#d32f2f" for v in dow_grp["total_usd"]]),
    text=[f"{int(t)} trades<br>{wr:.0f}% win" for t, wr in zip(dow_grp["trades"], dow_grp["win_rate_pct"])],
    textposition="outside",
))
fig_dow.update_layout(
    height=300, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    xaxis=dict(linecolor="#444", tickfont=dict(color="#ccc")),
    yaxis=dict(title="Total $", linecolor="#444",
                tickfont=dict(color="#ccc"), gridcolor="#222", tickprefix="$"),
    font=dict(color="#ccc"),
    margin=dict(l=50, r=30, t=30, b=40),
    showlegend=False,
)
st.plotly_chart(fig_dow, use_container_width=True)

# ---- Level type breakdown ----
st.markdown("### Level type contribution")
lt_grp = df.groupby("level_type").agg(
    trades=(TP_COL, "size"),
    wins=(TP_COL, lambda x: (x > 0).sum()),
    mean_r=(TP_COL, "mean"),
    total_r=(TP_COL, "sum"),
    total_usd=("pnl_usd", "sum"),
).sort_values("total_usd", ascending=False)
lt_grp["win_rate_pct"] = (lt_grp["wins"] / lt_grp["trades"] * 100).round(1)
lt_grp = lt_grp.reset_index()
st.dataframe(
    lt_grp[["level_type","trades","wins","win_rate_pct","mean_r","total_r","total_usd"]]
        .style.format({"mean_r": "{:.2f}", "total_r": "{:.2f}", "total_usd": "${:,.0f}"}),
    use_container_width=True,
    hide_index=True,
)

# ---- Drawdown timeline ----
st.markdown("### Drawdown timeline")
sorted_pnl = df.sort_values("mss_ts").reset_index(drop=True).copy()
sorted_pnl["cum"] = sorted_pnl["pnl_usd"].cumsum()
peak = sorted_pnl["cum"].cummax()
sorted_pnl["dd"] = sorted_pnl["cum"] - peak  # always <= 0

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=sorted_pnl["mss_ts"], y=sorted_pnl["dd"],
    mode="lines",
    line=dict(color="#d32f2f", width=1.5),
    fill="tozeroy", fillcolor="rgba(211,47,47,0.2)",
    name="Drawdown",
))
fig_dd.add_hline(y=-DAILY_DD_LIMIT, line_dash="dash", line_color="#ef5350",
                   annotation_text="−5% daily limit")
fig_dd.add_hline(y=-TRAILING_DD_LIM, line_dash="dash", line_color="#d32f2f",
                   annotation_text="−10% prop fail")
fig_dd.update_layout(
    height=300, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    xaxis=dict(linecolor="#444", tickfont=dict(color="#ccc"), gridcolor="#222"),
    yaxis=dict(title="DD from peak ($)", linecolor="#444",
                tickfont=dict(color="#ccc"), gridcolor="#222", tickprefix="$"),
    font=dict(color="#ccc"),
    margin=dict(l=50, r=30, t=20, b=40),
    showlegend=False,
)
st.plotly_chart(fig_dd, use_container_width=True)

max_dd = sorted_pnl["dd"].min()
max_dd_idx = sorted_pnl["dd"].idxmin()
max_dd_ts = sorted_pnl.iloc[max_dd_idx]["mss_ts"]
peak_before_max_dd = peak.iloc[max_dd_idx]
peak_idx = sorted_pnl[sorted_pnl["cum"] == peak_before_max_dd].index[0]
peak_ts = sorted_pnl.iloc[peak_idx]["mss_ts"]
recovery_idx = None
for i in range(max_dd_idx, len(sorted_pnl)):
    if sorted_pnl.iloc[i]["cum"] >= peak_before_max_dd:
        recovery_idx = i; break
recovery_ts = sorted_pnl.iloc[recovery_idx]["mss_ts"] if recovery_idx else None
recovery_str = recovery_ts.strftime("%Y-%m-%d") if recovery_ts else "(not yet recovered)"

dd1, dd2, dd3 = st.columns(3)
dd1.metric("Max DD ($)", f"${max_dd:,.2f}")
dd2.metric("DD trough date", max_dd_ts.strftime("%Y-%m-%d"))
dd3.metric("Recovery date", recovery_str)

# ---- Trade frequency ----
st.markdown("### Trade frequency")
total_days = (df["mss_ts"].max() - df["mss_ts"].min()).days
trading_days = df["trade_date"].nunique()
trades_per_day = len(df) / max(trading_days, 1)
trades_per_week = len(df) / max(total_days / 7, 1)
df_sorted = df.sort_values("mss_ts").reset_index(drop=True)
gaps = df_sorted["mss_ts"].diff().dropna()
longest_gap = gaps.max()
longest_gap_days = longest_gap.days if pd.notna(longest_gap) else 0

f1, f2, f3, f4 = st.columns(4)
f1.metric("Trades / week", f"{trades_per_week:.2f}")
f2.metric("Trades / trading day (avg)", f"{trades_per_day:.2f}")
f3.metric("Trading days used", f"{trading_days}")
f4.metric("Longest no-trade gap", f"{longest_gap_days}d")

# ============================================================
# 9. MONTHLY CALENDAR + DRILL DOWN
# ============================================================
st.divider()
st.markdown("## Calendar")

all_months = sorted(df["mss_ts"].dt.to_period("M").unique())
month_options = [str(p) for p in all_months]
if "cal_idx" not in st.session_state:
    st.session_state.cal_idx = len(month_options) - 1
st.session_state.cal_idx = max(0, min(st.session_state.cal_idx, len(month_options) - 1))

mc1, mc2, mc3 = st.columns([1,3,1])
with mc1:
    if st.button("←", key="prev_m", use_container_width=True):
        st.session_state.cal_idx = max(0, st.session_state.cal_idx - 1)
with mc2:
    st.session_state.cal_idx = month_options.index(
        st.selectbox("Month", month_options, index=st.session_state.cal_idx,
                       label_visibility="collapsed")
    )
with mc3:
    if st.button("→", key="next_m", use_container_width=True):
        st.session_state.cal_idx = min(len(month_options) - 1, st.session_state.cal_idx + 1)

selected_month = all_months[st.session_state.cal_idx]
month_start = pd.Timestamp(selected_month.to_timestamp())
month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
month_df = df[df["mss_ts"].dt.to_period("M") == selected_month]
by_day = month_df.groupby("trade_date").agg(
    pnl=("pnl_usd","sum"), trades=("pnl_usd","size"),
).to_dict("index")

m_total = len(month_df)
if m_total > 0:
    m_wins = month_df[month_df[TP_COL] > 0]
    m_losses = month_df[month_df[TP_COL] <= 0]
    m_pnl = month_df["pnl_usd"].sum()
    m_wr = len(m_wins) / m_total * 100
    m_total_r = month_df[TP_COL].sum()
    sorted_m = month_df.sort_values("mss_ts")["pnl_usd"].values
    cum = np.cumsum(sorted_m)
    peak = np.maximum.accumulate(np.concatenate(([0.0], cum)))
    cum0 = np.concatenate(([0.0], cum))
    dd = cum0 - peak
    m_max_dd = float(dd.min())
else:
    m_wins = m_losses = month_df
    m_pnl = m_wr = m_total_r = m_max_dd = 0.0

mm1, mm2, mm3, mm4, mm5 = st.columns(5)
mm1.metric("Month PnL", f"${m_pnl:,.2f}", f"{m_total_r:+.2f}R")
mm2.metric("Win Rate", f"{m_wr:.1f}%")
mm3.metric("Total Trades", f"{m_total}",
            f"{len(m_wins)}W / {len(m_losses)}L" if m_total else "—")
mm4.metric("Trading days", f"{len(by_day)}")
mm5.metric("Max DD", f"${m_max_dd:,.2f}", delta_color="inverse")

# Calendar grid
first_day = month_start
dow_of_first = first_day.weekday()
days_in_month = (month_end - month_start).days + 1
hdr = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
hdr_cols = st.columns(7)
for i, h in enumerate(hdr):
    hdr_cols[i].markdown(f"<div style='color:#888;font-size:12px;text-align:center'>{h}</div>",
                          unsafe_allow_html=True)
cal_rows = []
current_row = [None] * dow_of_first
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
                    f"<div style='text-align:right;font-size:12px'>{day_num}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                pnl = data["pnl"]; trades_n = data["trades"]
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
                    f"<span style='font-size:12px;color:#ddd'>{day_num}</span></div>"
                    f"<div style='text-align:center;font-weight:600;color:{txt_col};font-size:13px'>"
                    f"{sign}${abs(pnl):,.2f}</div></div>"
                )
                st.markdown(html, unsafe_allow_html=True)
                if st.button(" ", key=f"daybtn_{date}", help=f"{date} · {trades_n} trades"):
                    st.session_state.selected_day = date

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
            outcome_color = "🟢" if row[TP_COL] > 0 else "🔴"
            c5.write(f"{outcome_color} {row[TP_COL]:+.2f}R")
            c6.write(f"${row['pnl_usd']:+,.2f}")
            if st.button("📈 View chart", key=f"view_{sd}_{i}"):
                st.session_state.selected_trade = row.to_dict()


# ============================================================
# DOWNLOADS
# ============================================================
st.divider()
st.markdown("## Downloads")

trades_export = df.copy()
core_cols = [
    "smt_birth_ts","mss_ts","direction","level_type","sweeper","measured_asset",
    "session","entry_price","sl_price","r_points",
    TP_COL,"pnl_usd","cum_pnl","balance",
    "max_favorable_r","max_adverse_r",
    "outcome_group","mae_points","mae_ticks","mae_r","mae_usd",
    "hit_1r","hit_2r","hit_3r","hit_4r","hit_5r",
    "managed_exit_ts","managed_exit_reason",
    "_duration_min","entry_hour","dow",
]
core_cols = [c for c in core_cols if c in trades_export.columns]
trades_export = trades_export[core_cols + [c for c in trades_export.columns if c not in core_cols]]

# Summary
summary_data = {
    "strategy": "voligle_15m_after",
    "asset": ASSET, "tf": TF, "anchor": ANCHOR, "rule": RULE,
    "tp": f"TP{TP_LEVEL}", "sessions": "+".join(SESSIONS),
    "levels_excluded": ",".join(sorted(EXCLUDE_LEVELS)),
    "risk_per_trade_usd": RISK_PER_TRADE,
    "starting_balance_usd": STARTING_BAL,
    "total_trades": len(df),
    "wins": len(wins),
    "losses": len(losses),
    "win_rate_pct": round(win_rate, 2),
    "tp_hit_rate_pct": round(tp_hit_rate, 2),
    "drift_gap_pct": round(drift_gap, 2),
    "total_pnl_usd": round(total_pnl, 2),
    "final_balance_usd": round(final_balance, 2),
    "pct_return": round(pct_return, 2),
    "mean_r": round(avg_rr, 3),
    "max_r": round(max_rr, 2),
    "ideal_avg_r_mfe": round(ideal_avg_rr, 3) if not pd.isna(ideal_avg_rr) else 0.0,
    "expectancy_usd": round(expectancy_usd, 2),
    "avg_win_usd": round(avg_win_usd, 2),
    "avg_loss_usd": round(avg_loss_usd, 2),
    "profit_factor": round(pf, 2) if pf != float("inf") else 99.99,
    "max_consecutive_wins": int(max_win_streak),
    "max_consecutive_losses": int(max_loss_streak),
    "global_max_dd_usd": round(float(max_dd), 2),
    "trades_per_week": round(trades_per_week, 2),
    "trading_days": int(trading_days),
}
# Add MAE stats per group to summary
for g_key in ("tp2_hit","drift_win","loser","all"):
    s = stats_by_group[g_key]
    if s is None: continue
    summary_data[f"mae_{g_key}_n"] = s["n"]
    summary_data[f"mae_{g_key}_median_pts"] = round(s["median_pts"], 2)
    summary_data[f"mae_{g_key}_mean_pts"] = round(s["mean_pts"], 2)
    summary_data[f"mae_{g_key}_p75_pts"] = round(s["p75_pts"], 2)
    summary_data[f"mae_{g_key}_p90_pts"] = round(s["p90_pts"], 2)
    summary_data[f"mae_{g_key}_max_pts"] = round(s["max_pts"], 2)
summary_export = pd.DataFrame([summary_data])

# Monthly
monthly_rows = []
df_sorted_m = df.sort_values("mss_ts").copy()
df_sorted_m["_month"] = df_sorted_m["mss_ts"].dt.to_period("M").astype(str)
for month, m_df in df_sorted_m.groupby("_month"):
    if len(m_df) == 0: continue
    m_pnl_arr = m_df["pnl_usd"].values
    cum = np.cumsum(m_pnl_arr)
    peak = np.maximum.accumulate(np.concatenate(([0.0], cum)))
    cum0 = np.concatenate(([0.0], cum))
    dd = cum0 - peak
    monthly_rows.append({
        "month": month,
        "total_trades": len(m_df),
        "wins": int((m_df[TP_COL] > 0).sum()),
        "losses": int((m_df[TP_COL] <= 0).sum()),
        "win_rate_pct": round((m_df[TP_COL] > 0).sum() / len(m_df) * 100, 2),
        "total_pnl_usd": round(m_df["pnl_usd"].sum(), 2),
        "total_r": round(m_df[TP_COL].sum(), 2),
        "mean_r": round(m_df[TP_COL].mean(), 3),
        "trading_days": m_df["mss_ts"].dt.date.nunique(),
        "max_drawdown_usd": round(float(dd.min()), 2),
    })
monthly_export = pd.DataFrame(monthly_rows)

dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button(
        "📥 Trades CSV",
        data=trades_export.to_csv(index=False).encode("utf-8"),
        file_name="voligle_15m_after_trades.csv",
        mime="text/csv",
        use_container_width=True,
    )
with dl2:
    st.download_button(
        "📥 Summary CSV",
        data=summary_export.to_csv(index=False).encode("utf-8"),
        file_name="voligle_15m_after_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
with dl3:
    st.download_button(
        "📥 Monthly CSV",
        data=monthly_export.to_csv(index=False).encode("utf-8"),
        file_name="voligle_15m_after_monthly.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# TRADE CHART DIALOG (reuses dashboard chart logic, simplified)
# ============================================================
def session_bounds(date_str):
    start = pd.Timestamp(date_str, tz="America/New_York") - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
    end = pd.Timestamp(date_str, tz="America/New_York") + pd.Timedelta(hours=17)
    return start, end

def session_end_for(ts):
    if ts.hour >= 18:
        return ts.normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=17)
    return ts.normalize() + pd.Timedelta(hours=17)


@st.dialog("Trade chart")
def show_trade_chart(trade):
    smt_ts = trade["smt_birth_ts"]
    if pd.isna(smt_ts):
        st.error("Missing SMT birth timestamp"); return
    session_date_str = (smt_ts.normalize() if smt_ts.hour < 18 else (smt_ts + pd.Timedelta(days=1)).normalize()).strftime("%Y-%m-%d")
    s_start, s_end = session_bounds(session_date_str)
    mnq_bars = load_bars("MNQ", TF).loc[s_start:s_end]
    mes_bars = load_bars("MES", TF).loc[s_start:s_end]
    mnq_1m = load_bars("MNQ","1m"); mes_1m = load_bars("MES","1m")

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.04,
                          subplot_titles=(f"MNQ · {TF}", f"MES · {TF}"))
    def add_candles(d, col):
        fig.add_trace(go.Candlestick(
            x=d.index, open=d["open"], high=d["high"],
            low=d["low"], close=d["close"],
            increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
            decreasing=dict(line=dict(color="#000000"), fillcolor="#000000"),
            showlegend=False, hoverinfo="skip"), row=1, col=col)
    add_candles(mnq_bars, 1)
    add_candles(mes_bars, 2)

    # Position tool
    measured_asset = trade["measured_asset"]
    direction = trade["direction"]
    mss_ts = trade["mss_ts"]
    entry = trade["entry_price"]; sl = trade["sl_price"]
    r_pts = abs(entry - sl)
    bars_1m_a = mnq_1m if measured_asset == "MNQ" else mes_1m
    col = 1 if measured_asset == "MNQ" else 2
    end = session_end_for(mss_ts)

    # Use managed_exit_ts for visualization
    mgd_exit_ts = trade.get("managed_exit_ts")
    mgd_reason = trade.get("managed_exit_reason")
    if pd.notna(mgd_exit_ts):
        exit_ts = mgd_exit_ts
        try: exit_price = bars_1m_a.loc[mgd_exit_ts]["close"]
        except KeyError: exit_price = entry
        outcome = mgd_reason if mgd_reason else "session_end"
    else:
        exit_ts = end; exit_price = entry; outcome = "session_end"

    # R levels
    r_levels = [(n, entry + n * r_pts if direction == "bullish" else entry - n * r_pts) for n in range(1, 3)]
    if direction == "bullish":
        tp_top = r_levels[-1][1]; tp_bot = entry; sl_top = entry; sl_bot = sl
    else:
        tp_top = entry; tp_bot = r_levels[-1][1]; sl_top = sl; sl_bot = entry

    fig.add_shape(type="rect", x0=mss_ts, x1=exit_ts, y0=sl_bot, y1=sl_top,
                   fillcolor="rgba(239,83,80,0.25)", line=dict(width=0), row=1, col=col)
    fig.add_shape(type="rect", x0=mss_ts, x1=exit_ts, y0=tp_bot, y1=tp_top,
                   fillcolor="rgba(38,166,154,0.20)", line=dict(width=0), row=1, col=col)
    fig.add_shape(type="line", x0=mss_ts, x1=exit_ts, y0=entry, y1=entry,
                   line=dict(color="#1976d2", width=1.5), row=1, col=col)
    fig.add_shape(type="line", x0=mss_ts, x1=exit_ts, y0=sl, y1=sl,
                   line=dict(color="#d32f2f", width=1.5), row=1, col=col)
    for n, rv in r_levels:
        fig.add_shape(type="line", x0=mss_ts, x1=exit_ts, y0=rv, y1=rv,
                       line=dict(color="#388e3c", width=1, dash="dash"), row=1, col=col)
    label_offset = pd.Timedelta(minutes=2)
    fig.add_annotation(x=exit_ts + label_offset, y=entry,
                        text=f"Entry {entry:.2f}", showarrow=False,
                        font=dict(size=9, color="#1976d2"),
                        xanchor="left", yanchor="middle", row=1, col=col)
    fig.add_annotation(x=exit_ts + label_offset, y=sl,
                        text=f"SL {sl:.2f}", showarrow=False,
                        font=dict(size=9, color="#d32f2f"),
                        xanchor="left", yanchor="middle", row=1, col=col)
    for n, rv in r_levels:
        fig.add_annotation(x=exit_ts + label_offset, y=rv,
                            text=f"{n}R", showarrow=False,
                            font=dict(size=9, color="#388e3c"),
                            xanchor="left", yanchor="middle", row=1, col=col)
    exit_color = {"sl_hit":"#d32f2f","tp_hit":"#388e3c",
                   "session_end":"#757575","opp_smt_kill":"#9c27b0",
                   "opp_mss_kill":"#9c27b0","kill":"#9c27b0"}.get(outcome, "#757575")
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

    # Key level info
    level_type = trade.get("level_type", "—")
    sweeper_asset = trade.get("sweeper", "")
    level_id = trade.get("level_id", "")
    level_formed_str = "—"
    try:
        sweeper_levels = load_levels(sweeper_asset)
        sw_lvl = sweeper_levels[sweeper_levels["level_id"] == level_id]
        if len(sw_lvl):
            ts_formed = sw_lvl.iloc[0]["ts_formed"]
            if pd.notna(ts_formed):
                level_formed_str = ts_formed.strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass
    sweep_ts_str = pd.to_datetime(trade["smt_birth_ts"]).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M") \
        if pd.notna(trade.get("smt_birth_ts")) else "—"

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

    st.markdown("#### Trade details")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Direction", direction)
    t2.metric("Entry", f"{entry:.2f}")
    t3.metric("SL", f"{sl:.2f}")
    t4.metric("Outcome", outcome)
    t5, t6, t7, t8 = st.columns(4)
    t5.metric("R result", f"{trade[TP_COL]:+.2f}R")
    t6.metric("PnL", f"${trade['pnl_usd']:+,.2f}")
    t7.metric("MFE", f"{trade.get('max_favorable_r', 0):.2f}R")
    t8.metric("MAE", f"{trade.get('mae_r', 0):.2f}R ({trade.get('mae_ticks', 0):.0f}t)")


if "selected_trade" in st.session_state:
    show_trade_chart(st.session_state.selected_trade)
    del st.session_state.selected_trade