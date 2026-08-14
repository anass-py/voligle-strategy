"""
compare_strategies.py — side-by-side comparison of 3 candidate strategies.

Hardcoded candidates (you can swap by editing the CANDIDATES list below):
  A: 15m / after_smt  / close_break / TP1 / retr 0.50  — high-WR safe (DD ~$750)
  B: 15m / after_smt  / wick_break  / TP3 / retr 0.75  — high-PnL ($57k)
  C: 15m / before_smt / wick_break  / TP2 / retr 0.50  — broad workhorse ($48k)

Uses dashboard.py's logic exactly (auto_picked filter + realistic_trim +
simulate_retraced_entries) so the numbers match the dashboard.

Run:
  streamlit run compare_strategies.py --server.port 8509
"""
import os
import glob
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
RISK_PER_TRADE = 250.0
STARTING_BAL = 25000.0

LOCKED_SESSIONS = ["london", "ny_am"]
EXCLUDE_LEVELS = {"AsiaONHigh", "AsiaONLow", "PWH", "PWL"}

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


def hex_to_rgba(hex_str, alpha=1.0):
    """'#26a69a' -> 'rgba(38, 166, 154, 0.12)'."""
    h = hex_str.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

# ============================================================
# CANDIDATES (hardcoded)
# ============================================================
CANDIDATES = [
    {
        "name": "A · Safe (high WR, low DD)",
        "color": "#26a69a",
        "tf": "15m", "anchor": "after_smt", "rule": "close_break",
        "tp": "TP1", "tp_mult": 1.0, "retrace": 0.50,
    },
    {
        "name": "B · Aggressive (high PnL)",
        "color": "#fb8c00",
        "tf": "15m", "anchor": "after_smt", "rule": "wick_break",
        "tp": "TP3", "tp_mult": 3.0, "retrace": 0.75,
    },
    {
        "name": "C · Workhorse (broad sample)",
        "color": "#5e35b1",
        "tf": "15m", "anchor": "before_smt", "rule": "wick_break",
        "tp": "TP2", "tp_mult": 2.0, "retrace": 0.50,
    },
]

st.set_page_config(page_title="Strategy comparison", layout="wide")

# ============================================================
# LOADERS
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
    if os.path.getsize(path) == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    ts_cols = ["smt_birth_ts","smt_death_ts","swing_ts","mss_ts","sl_leg_extreme_ts",
                "hit_1r_ts","hit_2r_ts","hit_3r_ts","hit_4r_ts","hit_5r_ts",
                "sl_hit_ts","trade_exit_ts_if_held","managed_exit_ts"]
    for c in ts_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    return df


# ============================================================
# HELPERS (mirroring dashboard.py)
# ============================================================
def session_end_for(ts):
    if ts.hour >= 18:
        return ts.normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=17)
    return ts.normalize() + pd.Timedelta(hours=17)


def realistic_trim(trades, tp_col, single_position=False):
    """Prevent overlapping trades. If single_position=True, only ONE position at a time."""
    if len(trades) == 0:
        return trades
    t = trades.sort_values("mss_ts").reset_index(drop=True).copy()
    t["_exit_ts"] = t.get("managed_exit_ts")
    t["_exit_ts"] = t["_exit_ts"].fillna(
        t["mss_ts"].apply(lambda x: session_end_for(x) if pd.notna(x) else pd.NaT)
    )
    keep = []
    if single_position:
        last_exit = None
        for _, r in t.iterrows():
            entry = r["mss_ts"]
            if pd.isna(entry):
                keep.append(False); continue
            if last_exit is not None and entry < last_exit:
                keep.append(False); continue
            keep.append(True)
            last_exit = r["_exit_ts"]
    else:
        groups = {}
        for _, r in t.iterrows():
            key = (r["measured_asset"], r["direction"])
            entry = r["mss_ts"]
            if pd.isna(entry):
                keep.append(False); continue
            if key in groups and entry < groups[key]:
                keep.append(False); continue
            keep.append(True)
            groups[key] = r["_exit_ts"]
    t["_keep"] = keep
    return t[t["_keep"]].drop(columns=["_keep","_exit_ts"])


def simulate_retraced_entries(trades_df, retrace_r, tp_multiplier, tp_col_name, bars_1m_by_asset, strategy_tf):
    """Simulate retracement (same realism rules as dashboard.py + voligle_A.py).

    REALISM RULES:
      - Search starts at mss_ts + TF_minutes (= MSS bar close time)
      - Same-bar retracement-fill + SL collision: trade FILLS at retracement,
        immediately stops out at -1R (limit order would have triggered first)
      - Same-bar TP + SL during post-fill walk: SL wins (SL checked first)
    """
    TF_MIN = {"1m": 1, "5m": 5, "15m": 15}
    tf_minutes = TF_MIN.get(strategy_tf, 1)

    if retrace_r == 0:
        out = trades_df.copy()
        out["fill_ts"] = out["mss_ts"]
        out["entry_price_sim"] = out["entry_price"]
        out["r_points_sim"] = out["r_points"]
        out["outcome_r"] = out[tp_col_name]
        out["outcome_reason"] = out.get("managed_exit_reason", "session_end").fillna("session_end")
        out["exit_ts_sim"] = out.get("managed_exit_ts")
        return out

    rows = []
    for _, t in trades_df.iterrows():
        asset = t["measured_asset"]
        direction = t["direction"]
        orig_entry = float(t["entry_price"])
        sl = float(t["sl_price"])
        orig_r = abs(orig_entry - sl)
        if orig_r < TICK_SIZE - EPS: continue
        mss_ts = t["mss_ts"]
        if pd.isna(mss_ts): continue
        sess_end = session_end_for(mss_ts)

        kill_ts = None
        kill_reason = None
        mgd_reason = t.get("managed_exit_reason", "")
        mgd_exit_ts = t.get("managed_exit_ts")
        if mgd_reason in ("opp_smt_kill", "opp_mss_kill", "smt_invalidation_kill") and pd.notna(mgd_exit_ts):
            kill_ts = mgd_exit_ts
            kill_reason = mgd_reason

        if direction == "bullish":
            retrace_price = orig_entry - retrace_r * orig_r
        else:
            retrace_price = orig_entry + retrace_r * orig_r

        # FIX 1: Search starts at MSS bar close (mss_ts + TF_minutes)
        search_start = mss_ts + pd.Timedelta(minutes=tf_minutes)
        try:
            sub = bars_1m_by_asset[asset].loc[search_start:sess_end]
        except (KeyError, TypeError):
            continue
        if len(sub) == 0: continue

        fill_ts = None
        fill_then_stopped = False
        for ts, bar in sub.iterrows():
            if kill_ts is not None and ts >= kill_ts:
                fill_ts = None; break

            if direction == "bullish":
                hits_retrace = bar["low"] <= retrace_price + EPS
                hits_sl = bar["low"] <= sl + EPS
            else:
                hits_retrace = bar["high"] >= retrace_price - EPS
                hits_sl = bar["high"] >= sl - EPS

            if hits_retrace and hits_sl:
                # FIX 2: Same-bar fill + SL → fill then stopped out
                fill_ts = ts
                fill_then_stopped = True
                break
            if hits_sl and not hits_retrace:
                fill_ts = None; break
            if hits_retrace:
                fill_ts = ts; break

        if fill_ts is None: continue

        new_entry = retrace_price
        new_r = abs(new_entry - sl)
        if new_r < TICK_SIZE - EPS: continue

        # Same-bar fill + SL: record immediate stop-out
        if fill_then_stopped:
            new_row = t.to_dict()
            new_row["fill_ts"] = fill_ts
            new_row["entry_price_sim"] = new_entry
            new_row["r_points_sim"] = new_r
            new_row["outcome_r"] = -1.0
            new_row["outcome_reason"] = "sl_hit"
            new_row["exit_ts_sim"] = fill_ts
            rows.append(new_row)
            continue

        if direction == "bullish":
            new_tp = new_entry + tp_multiplier * new_r
        else:
            new_tp = new_entry - tp_multiplier * new_r

        try:
            sub2 = bars_1m_by_asset[asset].loc[fill_ts + pd.Timedelta(minutes=1):sess_end]
        except (KeyError, TypeError):
            sub2 = bars_1m_by_asset[asset].iloc[0:0]

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
                hit_tp = bar["high"] >= new_tp - EPS
                hit_sl = bar["low"] <= sl + EPS
            else:
                hit_tp = bar["low"] <= new_tp + EPS
                hit_sl = bar["high"] >= sl - EPS
            # FIX 3: SL checked first → same-bar TP+SL means SL wins
            if hit_sl:
                outcome_r = -1.0; outcome_reason = "sl_hit"; outcome_exit_ts = ts; break
            if hit_tp:
                outcome_r = float(tp_multiplier); outcome_reason = "tp_hit"; outcome_exit_ts = ts; break

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
        rows.append(new_row)

    return pd.DataFrame(rows)


def build_trades(meas, cand, bars_1m):
    """Reproduce dashboard.py's full filter pipeline for the candidate."""
    df = meas[
        (meas["mss_timeframe"] == cand["tf"])
        & (meas["anchor"] == cand["anchor"])
        & (meas["mss_rule"] == cand["rule"])
        & (meas["mss_found"] == True)
        & (meas["final_outcome"].isin(["sl_hit", "session_end"]))
        & (meas["session"].isin(LOCKED_SESSIONS))
        & (meas["level_type"].isin(LEVEL_TYPES_USED))
        & (meas["auto_picked"] == True)
    ].copy()

    tp_col_raw = f"r_tp{int(cand['tp_mult'])}"
    tp_col = tp_col_raw + "_mgd"
    if tp_col not in df.columns:
        tp_col = tp_col_raw
    df = df.dropna(subset=[tp_col])
    df = realistic_trim(df, tp_col, single_position=True)

    sim = simulate_retraced_entries(df, cand["retrace"], cand["tp_mult"], tp_col, bars_1m, cand["tf"])
    if len(sim) == 0:
        return pd.DataFrame(), tp_col

    sim["pnl_usd"] = sim["outcome_r"] * RISK_PER_TRADE
    sim = sim.sort_values("fill_ts").reset_index(drop=True)
    sim["cum_pnl"] = sim["pnl_usd"].cumsum()
    sim["balance"] = STARTING_BAL + sim["cum_pnl"]
    sim["peak"] = sim["cum_pnl"].cummax()
    sim["dd"] = sim["cum_pnl"] - sim["peak"]
    return sim, tp_col


def metrics(df, cand):
    n = len(df)
    if n == 0:
        return {}
    wins = int((df["outcome_r"] > 0).sum())
    losses = int((df["outcome_r"] <= 0).sum())
    tp_hits = int((df["outcome_r"] >= cand["tp_mult"] - 0.001).sum())
    total_pnl = float(df["pnl_usd"].sum())
    gross_win = float(df[df["pnl_usd"] > 0]["pnl_usd"].sum())
    gross_loss = float(abs(df[df["pnl_usd"] <= 0]["pnl_usd"].sum()))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    max_dd = float(df["dd"].min())

    # longest losing streak
    streak = 0; longest = 0
    for r in df["outcome_r"]:
        if r <= 0:
            streak += 1
            if streak > longest: longest = streak
        else:
            streak = 0

    # day stats
    df_day = df.copy()
    df_day["_day"] = df_day["fill_ts"].dt.date
    daily = df_day.groupby("_day")["pnl_usd"].sum()
    worst_day = float(daily.min()) if len(daily) else 0
    best_day = float(daily.max()) if len(daily) else 0

    return {
        "trades": n,
        "wins": wins,
        "losses": losses,
        "tp_hits": tp_hits,
        "wr": wins / n * 100,
        "tp_rate": tp_hits / n * 100,
        "total_pnl": total_pnl,
        "pct_return": total_pnl / STARTING_BAL * 100,
        "pf": pf,
        "exp": total_pnl / n,
        "max_dd": max_dd,
        "max_dd_pct": abs(max_dd) / STARTING_BAL * 100,
        "longest_loss_streak": longest,
        "worst_day": worst_day,
        "best_day": best_day,
        "trading_days": len(daily),
        "winning_days": int((daily > 0).sum()),
        "pnl_2023": float(df[df["fill_ts"].dt.year == 2023]["pnl_usd"].sum()),
        "pnl_2024": float(df[df["fill_ts"].dt.year == 2024]["pnl_usd"].sum()),
        "pnl_2025": float(df[df["fill_ts"].dt.year == 2025]["pnl_usd"].sum()),
    }


# ============================================================
# LOAD DATA
# ============================================================
measurement_files = sorted(glob.glob(os.path.join(OUT_FOLDER, "SMT_measurements_*.csv")))
if not measurement_files:
    st.error(f"No SMT_measurements_*.csv in {OUT_FOLDER}. Run phase4.py first.")
    st.stop()
csv_path = measurement_files[-1]
meas = load_measurements(csv_path, os.path.getmtime(csv_path))
if len(meas) == 0:
    st.error("Measurements file is empty.")
    st.stop()

bars_1m = {"MNQ": load_bars("MNQ", "1m"), "MES": load_bars("MES", "1m")}

# Build trades for each candidate (cached via session_state)
if "compare_results" not in st.session_state:
    with st.spinner("Building trades for all 3 strategies..."):
        results = []
        for cand in CANDIDATES:
            trades_df, tp_col = build_trades(meas, cand, bars_1m)
            m = metrics(trades_df, cand)
            results.append({"cand": cand, "trades": trades_df, "tp_col": tp_col, "metrics": m})
        st.session_state.compare_results = results
results = st.session_state.compare_results

# ============================================================
# UI
# ============================================================
st.markdown("# ⚖️ Strategy Comparison (A vs B vs C)")
st.caption(
    f"Dataset: **{os.path.basename(csv_path)}**  ·  Risk: **${int(RISK_PER_TRADE)}/trade**  "
    f"·  Start: **${int(STARTING_BAL):,}**  ·  Sessions: **london + ny_am**  "
    f"·  Levels excl: **AsiaONHigh, AsiaONLow, PWH, PWL**"
)

# ---- Candidate config row ----
cols = st.columns(3)
for i, (col, r) in enumerate(zip(cols, results)):
    cand = r["cand"]
    with col:
        st.markdown(
            f"<div style='padding:10px;border-radius:6px;background:{hex_to_rgba(cand['color'], 0.15)};border-left:4px solid {cand['color']}'>"
            f"<b style='font-size:15px'>{cand['name']}</b><br>"
            f"<span style='font-size:12px;opacity:0.85'>"
            f"TF <b>{cand['tf']}</b>  ·  Anchor <b>{cand['anchor']}</b>  ·  Rule <b>{cand['rule']}</b><br>"
            f"TP <b>{cand['tp']}</b>  ·  Retrace <b>{cand['retrace']:.2f}R</b>"
            f"</span></div>",
            unsafe_allow_html=True,
        )

st.divider()

# ---- Headline metrics ----
st.markdown("## Headline metrics")
cols = st.columns(3)
for col, r in zip(cols, results):
    m = r["metrics"]
    cand = r["cand"]
    with col:
        st.markdown(f"<div style='color:{cand['color']};font-weight:600'>{cand['name'].split(' · ')[0]}</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Total PnL", f"${m['total_pnl']:,.0f}", f"{m['pct_return']:+.2f}%")
        c2.metric("Win rate", f"{m['wr']:.2f}%", f"{m['wins']}W / {m['losses']}L")
        c1.metric("Trades", f"{m['trades']}")
        c2.metric("TP hit rate", f"{m['tp_rate']:.2f}%", f"{m['tp_hits']} hits")
        c1.metric("Max DD", f"${m['max_dd']:,.0f}", f"{m['max_dd_pct']:.2f}%")
        c2.metric("Profit factor", f"{m['pf']:.2f}" if m['pf'] != float("inf") else "∞")
        c1.metric("Longest loss streak", f"{m['longest_loss_streak']}")
        c2.metric("Worst day", f"${m['worst_day']:,.0f}")

st.divider()

# ---- Year split ----
st.markdown("## Year-by-year PnL")
year_rows = []
for r in results:
    m = r["metrics"]; cand = r["cand"]
    year_rows.append({
        "Strategy": cand["name"].split(" · ")[0],
        "2023": m["pnl_2023"],
        "2024": m["pnl_2024"],
        "2025": m["pnl_2025"],
        "Total": m["total_pnl"],
    })
year_df = pd.DataFrame(year_rows)
fig_yr = go.Figure()
for col_year in ["2023", "2024", "2025"]:
    fig_yr.add_trace(go.Bar(
        name=col_year, x=year_df["Strategy"], y=year_df[col_year],
        text=[f"${v:,.0f}" for v in year_df[col_year]],
        textposition="outside",
    ))
fig_yr.update_layout(
    barmode="group", height=350,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font=dict(color="#fafafa"),
    yaxis_title="PnL ($)", margin=dict(l=40, r=20, t=10, b=40),
)
fig_yr.update_xaxes(showgrid=False)
fig_yr.update_yaxes(gridcolor="#333", zerolinecolor="#666")
st.plotly_chart(fig_yr, use_container_width=True)

st.divider()

# ---- Equity curves overlaid ----
st.markdown("## Equity curves (overlaid)")
fig_eq = go.Figure()
for r in results:
    df_t = r["trades"]
    cand = r["cand"]
    if len(df_t) == 0: continue
    fig_eq.add_trace(go.Scatter(
        x=df_t["fill_ts"], y=df_t["balance"],
        mode="lines", name=cand["name"].split(" · ")[0],
        line=dict(color=cand["color"], width=2),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.0f}<extra></extra>",
    ))
fig_eq.add_hline(y=STARTING_BAL, line=dict(color="#666", width=1, dash="dash"), annotation_text="Start ($25k)")
fig_eq.update_layout(
    height=440,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font=dict(color="#fafafa"),
    yaxis_title="Account balance ($)",
    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    margin=dict(l=50, r=20, t=40, b=40),
)
fig_eq.update_xaxes(showgrid=False)
fig_eq.update_yaxes(gridcolor="#333")
st.plotly_chart(fig_eq, use_container_width=True)

st.divider()

# ---- Drawdown curves overlaid ----
st.markdown("## Drawdown curves (overlaid)")
fig_dd = go.Figure()
for r in results:
    df_t = r["trades"]
    cand = r["cand"]
    if len(df_t) == 0: continue
    fig_dd.add_trace(go.Scatter(
        x=df_t["fill_ts"], y=df_t["dd"],
        mode="lines", name=cand["name"].split(" · ")[0],
        line=dict(color=cand["color"], width=2),
        fill="tozeroy", fillcolor=hex_to_rgba(cand["color"], 0.12),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>DD: $%{y:,.0f}<extra></extra>",
    ))
fig_dd.update_layout(
    height=300,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font=dict(color="#fafafa"),
    yaxis_title="Drawdown ($)",
    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    margin=dict(l=50, r=20, t=40, b=40),
)
fig_dd.update_xaxes(showgrid=False)
fig_dd.update_yaxes(gridcolor="#333", zerolinecolor="#666")
st.plotly_chart(fig_dd, use_container_width=True)

st.divider()

# ============================================================
# MONTHLY CALENDAR (side by side)
# ============================================================
st.markdown("## Monthly calendar comparison")
st.caption("Each row is one month. Each column shows that month's PnL for each strategy.")

# Build month set
all_months = set()
for r in results:
    if len(r["trades"]):
        all_months |= set(r["trades"]["fill_ts"].dt.to_period("M").unique())
months_sorted = sorted(all_months)

# Per-strategy monthly PnL + WR
def build_month_summary(df_t):
    if len(df_t) == 0:
        return {}
    df_t = df_t.copy()
    df_t["_m"] = df_t["fill_ts"].dt.to_period("M")
    out = {}
    for m, g in df_t.groupby("_m"):
        wins_m = int((g["outcome_r"] > 0).sum())
        out[m] = {
            "pnl": float(g["pnl_usd"].sum()),
            "trades": len(g),
            "wr": wins_m / len(g) * 100 if len(g) else 0,
            "wins": wins_m,
            "losses": len(g) - wins_m,
        }
    return out

month_data = [build_month_summary(r["trades"]) for r in results]

# Render: month label + 3 cols
header = st.columns([1.4, 2, 2, 2])
header[0].markdown("**Month**")
for i, r in enumerate(results):
    header[i+1].markdown(f"<div style='color:{r['cand']['color']};font-weight:600'>{r['cand']['name'].split(' · ')[0]}</div>", unsafe_allow_html=True)

for m in months_sorted:
    row = st.columns([1.4, 2, 2, 2])
    row[0].markdown(f"**{m}**")
    for i, md in enumerate(month_data):
        d = md.get(m)
        with row[i+1]:
            if d is None:
                st.markdown("<span style='opacity:0.4'>—</span>", unsafe_allow_html=True)
            else:
                pnl_color = "#4caf50" if d["pnl"] > 0 else "#ef5350" if d["pnl"] < 0 else "#aaa"
                st.markdown(
                    f"<span style='color:{pnl_color};font-weight:600;font-size:14px'>${d['pnl']:,.0f}</span> "
                    f"<span style='opacity:0.75;font-size:11px'>"
                    f"({d['trades']} trd, {d['wr']:.0f}% WR)</span>",
                    unsafe_allow_html=True,
                )

st.divider()

# ============================================================
# PER-STRATEGY TRADE TABLES (with chart buttons)
# ============================================================
st.markdown("## Trade tables (click 📈 for trade chart)")

tabs = st.tabs([r["cand"]["name"] for r in results])
for tab_i, (tab, r) in enumerate(zip(tabs, results)):
    with tab:
        df_t = r["trades"]
        cand = r["cand"]
        if len(df_t) == 0:
            st.info("No trades for this strategy.")
            continue
        st.caption(f"{len(df_t)} trades. Click 📈 to view the trade chart (zoomed to MSS-2h → exit+30min).")

        # Compact table
        for i, t in df_t.iterrows():
            outcome_emoji = "🎯" if t["outcome_r"] >= cand["tp_mult"] - 0.001 else "🪂" if t["outcome_r"] > 0 else "❌"
            dir_emoji = "🟢" if t["direction"] == "bullish" else "🔴"
            lt_label = SHORT_LABEL.get(t["level_type"], t["level_type"])
            fill_str = pd.Timestamp(t["fill_ts"]).strftime("%Y-%m-%d %H:%M")
            mss_str = pd.Timestamp(t["mss_ts"]).strftime("%H:%M")
            r_str = f"{t['outcome_r']:+.2f}R"
            pnl_str = f"${t['pnl_usd']:+,.0f}"
            cum_str = f"${t['cum_pnl']:+,.0f}"

            cols = st.columns([10, 1])
            with cols[0]:
                st.markdown(
                    f"<div style='padding:5px 10px;border-left:3px solid {cand['color']};"
                    f"margin:2px 0;font-size:13px'>"
                    f"<b>{fill_str}</b> · MSS {mss_str} · "
                    f"{dir_emoji} <b>{lt_label}</b> on <b>{t['measured_asset']}</b> · "
                    f"{outcome_emoji} <b>{r_str}</b> ({pnl_str}) "
                    f"<span style='opacity:0.6'>cum {cum_str}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                if st.button("📈", key=f"chart_{tab_i}_{i}", help="View trade chart"):
                    st.session_state.compare_selected_trade = {
                        "trade": t.to_dict(),
                        "cand": cand,
                    }


# ============================================================
# TRADE CHART (zoomed to trade window)
# ============================================================
@st.dialog("Trade chart", width="large")
def show_trade_chart():
    payload = st.session_state.compare_selected_trade
    t = payload["trade"]
    cand = payload["cand"]
    asset = t["measured_asset"]
    direction = t["direction"]
    mss_ts = pd.Timestamp(t["mss_ts"])
    fill_ts = pd.Timestamp(t["fill_ts"])
    exit_ts = pd.Timestamp(t["exit_ts_sim"]) if pd.notna(t.get("exit_ts_sim")) else session_end_for(fill_ts)
    entry_price = float(t["entry_price_sim"]) if pd.notna(t.get("entry_price_sim")) else float(t["entry_price"])
    sl = float(t["sl_price"])
    r_pts = abs(entry_price - sl)
    tp_mult = cand["tp_mult"]
    if direction == "bullish":
        tp_price = entry_price + tp_mult * r_pts
    else:
        tp_price = entry_price - tp_mult * r_pts

    # Option A zoom: MSS - 2h to exit + 30min
    x_start = mss_ts - pd.Timedelta(hours=2)
    x_end = exit_ts + pd.Timedelta(minutes=30)

    bars = load_bars(asset, cand["tf"]).loc[x_start:x_end]
    if len(bars) == 0:
        st.error("No bars to plot.")
        return

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=bars.index, open=bars["open"], high=bars["high"],
        low=bars["low"], close=bars["close"],
        increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#000"), fillcolor="#000"),
        showlegend=False,
    ))

    # Position tool (Option X: from fill_ts to exit_ts — no extension)
    pos_left = fill_ts
    pos_right = exit_ts

    if direction == "bullish":
        tp_top = tp_price; tp_bot = entry_price
        sl_top = entry_price; sl_bot = sl
    else:
        tp_top = entry_price; tp_bot = tp_price
        sl_top = sl; sl_bot = entry_price

    # TP box
    fig.add_shape(type="rect", x0=pos_left, x1=pos_right, y0=tp_bot, y1=tp_top,
                  fillcolor="rgba(38,166,154,0.22)", line=dict(width=0))
    # SL box
    fig.add_shape(type="rect", x0=pos_left, x1=pos_right, y0=sl_bot, y1=sl_top,
                  fillcolor="rgba(239,83,80,0.25)", line=dict(width=0))
    # Entry / SL / TP lines
    fig.add_shape(type="line", x0=pos_left, x1=pos_right, y0=entry_price, y1=entry_price,
                  line=dict(color="#1976d2", width=1.5))
    fig.add_shape(type="line", x0=pos_left, x1=pos_right, y0=sl, y1=sl,
                  line=dict(color="#d32f2f", width=1.5))
    fig.add_shape(type="line", x0=pos_left, x1=pos_right, y0=tp_price, y1=tp_price,
                  line=dict(color="#388e3c", width=1.5, dash="dash"))

    # MSS marker (vertical line at MSS time)
    fig.add_shape(type="line", x0=mss_ts, x1=mss_ts, y0=0, y1=1,
                  xref="x", yref="paper",
                  line=dict(color="#fb8c00", width=1, dash="dot"))
    fig.add_annotation(x=mss_ts, y=1, xref="x", yref="paper",
                       text="MSS", showarrow=False, font=dict(size=10, color="#fb8c00"),
                       yshift=10)

    # Fill marker
    fig.add_shape(type="line", x0=fill_ts, x1=fill_ts, y0=0, y1=1,
                  xref="x", yref="paper",
                  line=dict(color="#1976d2", width=1, dash="dot"))
    fig.add_annotation(x=fill_ts, y=0, xref="x", yref="paper",
                       text="Fill", showarrow=False, font=dict(size=10, color="#1976d2"),
                       yshift=-10)

    # Price labels at right edge
    label_x = pos_right + pd.Timedelta(minutes=3)
    fig.add_annotation(x=label_x, y=entry_price, text=f"E {entry_price:.2f}",
                       showarrow=False, font=dict(size=10, color="#1976d2"),
                       xanchor="left", yanchor="middle")
    fig.add_annotation(x=label_x, y=sl, text=f"SL {sl:.2f}",
                       showarrow=False, font=dict(size=10, color="#d32f2f"),
                       xanchor="left", yanchor="middle")
    fig.add_annotation(x=label_x, y=tp_price, text=f"TP{int(tp_mult)} {tp_price:.2f}",
                       showarrow=False, font=dict(size=10, color="#388e3c"),
                       xanchor="left", yanchor="middle")

    fig.update_layout(
        height=520, paper_bgcolor="white", plot_bgcolor="white",
        xaxis_rangeslider_visible=False,
        hovermode=False, showlegend=False,
        margin=dict(l=50, r=110, t=30, b=40),
        font=dict(color="black"),
        title=f"{cand['name']} · {asset} {cand['tf']} · "
              f"{direction} {SHORT_LABEL.get(t['level_type'], t['level_type'])} · "
              f"{t['outcome_r']:+.2f}R (${t['outcome_r']*RISK_PER_TRADE:+,.0f})",
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor="black",
                     tickfont=dict(color="black", size=9),
                     rangebreaks=[dict(bounds=[17, 18], pattern="hour")],
                     range=[x_start, x_end])
    fig.update_yaxes(showgrid=False, showline=True, linecolor="black",
                     tickfont=dict(color="black", size=9))

    st.plotly_chart(fig, use_container_width=True)

    # Trade details below
    st.markdown("#### Trade detail")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Direction", direction)
    d2.metric("Asset", asset)
    d3.metric("Outcome", f"{t['outcome_r']:+.2f}R")
    d4.metric("PnL", f"${t['outcome_r']*RISK_PER_TRADE:+,.0f}")

    st.markdown(
        f"**MSS time:** {mss_ts.strftime('%Y-%m-%d %H:%M')} · "
        f"**Fill time:** {fill_ts.strftime('%H:%M')} · "
        f"**Exit time:** {exit_ts.strftime('%H:%M')} · "
        f"**Exit reason:** {t.get('outcome_reason','')}"
    )
    st.markdown(
        f"**Entry:** {entry_price:.2f}  ·  **SL:** {sl:.2f}  ·  **TP:** {tp_price:.2f}  ·  "
        f"**Risk distance:** {r_pts:.2f} pts  ·  "
        f"**Level type:** {t['level_type']}  ·  "
        f"**Sweeper:** {t.get('sweeper','')}"
    )


if "compare_selected_trade" in st.session_state:
    show_trade_chart()
    del st.session_state.compare_selected_trade