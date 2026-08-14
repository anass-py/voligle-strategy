"""
tp_simulator.py — what-if analysis on TP targets using Databento bars.

Compares 5 scenarios:
  A · Actual (baseline)        — your trades as recorded
  B · 2R TP + daily rules      — TP at 2R, walked via Databento 1m bars
  C · 3R TP + daily rules
  D · 4R TP + daily rules
  E · 5R TP + daily rules

DAILY RULES (apply to B / C / D / E):
  1. Stop trading for the day as soon as the chosen TP is HIT on any trade
     (the first TP-hit ends the day).
  2. Stop trading for the day after cumulative day R hits ≤ -2R
     (the trade that crosses -2R still counts; following trades are skipped).
  3. Session_end and SL outcomes update cumulative R but do NOT end the day
     by themselves — the day only ends from TP-hit OR -2R cumulative cap OR
     when the trader's last trade for the day has been processed.

PER-TRADE OUTCOME (Databento walk from entry to 17:00 NY):
  - SL hit first → -1R, day continues unless -2R rule fires
  - TP hit first → +(target)R, day ENDS (rule 1)
  - Same-bar TP + SL → SL wins (conservative)
  - Neither hit → mark-to-market at 17:00 close (fractional R)

PNL CALCULATION:
  PnL = realized_R × R_distance × amount    (from CSV columns)

Run:
  streamlit run tp_simulator.py --server.port 8512
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================
DATA_FOLDER = "/Users/voligle/workspace/Trading/Trading_code/clause Backtester"
OUT_FOLDER = os.path.join(DATA_FOLDER, "New/out")
ANALYTICS_CSV = "analytics.csv"
TICK_SIZE = 0.25
EPS = 1e-9
STARTING_BAL = 25000.0
DAILY_STOP_R = -2.0
TP_TARGETS = [2.0, 3.0, 4.0, 5.0]

st.set_page_config(page_title="TP simulator", layout="wide")


def hex_to_rgba(hex_str, alpha=1.0):
    h = hex_str.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# ============================================================
# LOADERS
# ============================================================
@st.cache_data
def load_trades(csv_path):
    df = pd.read_csv(csv_path)
    df["dateStart"] = pd.to_datetime(df["dateStart"])
    df["dateEnd"] = pd.to_datetime(df["dateEnd"])
    if df["dateStart"].dt.tz is None:
        df["dateStart"] = df["dateStart"].dt.tz_localize("America/New_York")
        df["dateEnd"] = df["dateEnd"].dt.tz_localize("America/New_York")
    df = df.sort_values("dateStart").reset_index(drop=True)
    return df


@st.cache_data
def load_databento_bars(asset):
    path = os.path.join(OUT_FOLDER, f"{asset}_1m.csv")
    d = pd.read_csv(path)
    d["ts_ny"] = pd.to_datetime(d["ts_ny"], utc=True).dt.tz_convert("America/New_York")
    return d.set_index("ts_ny").sort_index()[["open","high","low","close"]]


def map_pair_to_asset(pair):
    if "MNQ" in pair: return "MNQ"
    if "MES" in pair: return "MES"
    return None


def session_end_for(ts):
    """Returns the 17:00 NY close for the given timestamp's session.
    Futures session starts 18:00 prior day; closes 17:00 same calendar day
    if ts is between 00:00 and 17:00, otherwise 17:00 next day."""
    if ts.hour >= 18:
        return ts.normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=17)
    return ts.normalize() + pd.Timedelta(hours=17)


# ============================================================
# WALK ONE TRADE WITH ANY TP MULTIPLIER
# ============================================================
def simulate_tp_outcome(trade, tp_mult, bars_by_asset):
    """
    For one trade, walk 1m Databento bars from entry to session end.
    Return (outcome_r, outcome_reason, exit_ts).

    Conservative rules:
      - SL checked first → same-bar SL+TP → SL wins
      - Neither hit → mark to market at session close
    """
    asset = map_pair_to_asset(trade["pair"])
    bars = bars_by_asset.get(asset)
    if bars is None: return None, None, None

    entry_time = trade["dateStart"]
    side = trade["side"]
    raw_entry = float(trade["entryPrice"])
    raw_sl = float(trade["initalSL"])
    r_distance = abs(raw_entry - raw_sl)
    if r_distance < TICK_SIZE - EPS: return None, None, None

    # Find Databento entry: first 1m bar at or after entry_time
    sub = bars.loc[entry_time:]
    if len(sub) == 0: return None, None, None
    db_entry = float(sub.iloc[0]["open"])

    if side == "buy":
        db_sl = db_entry - r_distance
        db_tp = db_entry + tp_mult * r_distance
        direction = "long"
    else:
        db_sl = db_entry + r_distance
        db_tp = db_entry - tp_mult * r_distance
        direction = "short"

    sess_end = session_end_for(entry_time)
    walk = bars.loc[entry_time + pd.Timedelta(minutes=1):sess_end]

    for ts, bar in walk.iterrows():
        if direction == "long":
            hit_sl = bar["low"] <= db_sl + EPS
            hit_tp = bar["high"] >= db_tp - EPS
        else:
            hit_sl = bar["high"] >= db_sl - EPS
            hit_tp = bar["low"] <= db_tp + EPS

        # Same-bar SL+TP → SL wins
        if hit_sl:
            return -1.0, "sl_hit", ts
        if hit_tp:
            return float(tp_mult), "tp_hit", ts

    # Mark to market at session close
    if len(walk) > 0:
        last_close = float(walk.iloc[-1]["close"])
        if direction == "long":
            final_r = (last_close - db_entry) / r_distance
        else:
            final_r = (db_entry - last_close) / r_distance
        return final_r, "session_end", walk.index[-1]
    return 0.0, "session_end", entry_time


# ============================================================
# BUILD FULL SCENARIO (for one TP target)
# ============================================================
@st.cache_data
def build_scenario(tp_mult):
    """
    For each trade in chronological order:
      - Walk Databento bars to get (outcome_r, reason)
      - Apply daily rules:
          * If we've already had a tp_hit today → skip
          * If cumulative day R ≤ -2R already → skip
          * Otherwise: take the trade, update cum_R, possibly stop the day
    Returns DataFrame with kept trades + outcome columns + PnL.
    """
    trades = load_trades(ANALYTICS_CSV)
    bars_by_asset = {
        "MNQ": load_databento_bars("MNQ"),
        "MES": load_databento_bars("MES"),
    }

    rows = []
    # We must process trades day-by-day in chronological order
    trades["_date"] = trades["dateStart"].dt.date
    skipped_tp = 0
    skipped_dailystop = 0
    no_data = 0

    for date, day_df in trades.groupby("_date", sort=False):
        cum_r = 0.0
        day_stopped = False
        stop_reason = None
        for _, t in day_df.iterrows():
            if day_stopped:
                if stop_reason == "tp_hit":
                    skipped_tp += 1
                else:
                    skipped_dailystop += 1
                continue

            outcome_r, reason, exit_ts = simulate_tp_outcome(t, tp_mult, bars_by_asset)
            if outcome_r is None:
                no_data += 1
                continue

            r_distance = abs(float(t["entryPrice"]) - float(t["initalSL"]))
            amount = float(t["amount"])
            pnl = outcome_r * r_distance * amount

            cum_r += outcome_r

            row = t.to_dict()
            row["sim_outcome_r"] = outcome_r
            row["sim_outcome_reason"] = reason
            row["sim_exit_ts"] = exit_ts
            row["sim_pnl"] = pnl
            row["cum_day_r"] = cum_r
            rows.append(row)

            # Rule 1: TP hit → end day
            if reason == "tp_hit":
                day_stopped = True
                stop_reason = "tp_hit"
            # Rule 2: cum_R ≤ -2R → end day
            elif cum_r <= DAILY_STOP_R + EPS:
                day_stopped = True
                stop_reason = "daily_stop"

    df = pd.DataFrame(rows)
    return df, skipped_tp, skipped_dailystop, no_data


# ============================================================
# METRICS
# ============================================================
def compute_metrics(df, pnl_col):
    if len(df) == 0:
        return {"trades":0,"wins":0,"losses":0,"wr":0,"total_pnl":0,"max_dd":0,
                "max_dd_pct":0,"pf":0,"expectancy":0,"avg_win":0,"avg_loss":0,
                "longest_loss_streak":0,"longest_win_streak":0,"worst_day":0,
                "best_day":0,"trading_days":0,"winning_days":0,
                "pnl_2023":0,"pnl_2024":0,"pnl_2025":0,
                "tp_hits":0,"sl_hits":0,"session_ends":0}
    df = df.sort_values("dateStart").reset_index(drop=True).copy()
    n = len(df)
    wins = int((df[pnl_col] > 0).sum())
    losses = int((df[pnl_col] <= 0).sum())
    wr = wins / n * 100 if n else 0

    cum = df[pnl_col].cumsum().values
    peak = np.maximum.accumulate(np.concatenate(([0.0], cum)))
    cum0 = np.concatenate(([0.0], cum))
    dd = cum0 - peak
    max_dd = float(dd.min())

    gross_win = float(df[df[pnl_col] > 0][pnl_col].sum())
    gross_loss = float(abs(df[df[pnl_col] <= 0][pnl_col].sum()))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    total_pnl = float(df[pnl_col].sum())
    expectancy = total_pnl / n if n else 0

    cur = 0; longest_loss = 0
    for v in df[pnl_col].values:
        if v <= 0:
            cur += 1
            longest_loss = max(longest_loss, cur)
        else: cur = 0
    cur = 0; longest_win = 0
    for v in df[pnl_col].values:
        if v > 0:
            cur += 1
            longest_win = max(longest_win, cur)
        else: cur = 0

    df["_date"] = df["dateStart"].dt.date
    daily = df.groupby("_date")[pnl_col].sum()

    # Per-year
    df["_year"] = df["dateStart"].dt.year
    yearly = df.groupby("_year")[pnl_col].sum()

    # Outcome reasons
    tp_hits = int((df.get("sim_outcome_reason", "") == "tp_hit").sum()) if "sim_outcome_reason" in df.columns else None
    sl_hits = int((df.get("sim_outcome_reason", "") == "sl_hit").sum()) if "sim_outcome_reason" in df.columns else None
    session_ends = int((df.get("sim_outcome_reason", "") == "session_end").sum()) if "sim_outcome_reason" in df.columns else None

    return {
        "trades": n, "wins": wins, "losses": losses, "wr": wr,
        "total_pnl": total_pnl, "max_dd": max_dd,
        "max_dd_pct": abs(max_dd) / STARTING_BAL * 100,
        "pf": pf, "expectancy": expectancy,
        "avg_win": float(df[df[pnl_col] > 0][pnl_col].mean()) if wins else 0,
        "avg_loss": float(df[df[pnl_col] <= 0][pnl_col].mean()) if losses else 0,
        "longest_loss_streak": longest_loss,
        "longest_win_streak": longest_win,
        "worst_day": float(daily.min()) if len(daily) else 0,
        "best_day": float(daily.max()) if len(daily) else 0,
        "trading_days": len(daily),
        "winning_days": int((daily > 0).sum()),
        "pnl_2023": float(yearly.get(2023, 0)),
        "pnl_2024": float(yearly.get(2024, 0)),
        "pnl_2025": float(yearly.get(2025, 0)),
        "tp_hits": tp_hits, "sl_hits": sl_hits, "session_ends": session_ends,
    }


def equity_curve(df, pnl_col):
    df = df.sort_values("dateStart").reset_index(drop=True).copy()
    df["cum"] = df[pnl_col].cumsum()
    df["balance"] = STARTING_BAL + df["cum"]
    df["peak"] = df["cum"].cummax()
    df["dd"] = df["cum"] - df["peak"]
    return df[["dateStart","balance","cum","dd",pnl_col]].rename(columns={pnl_col: "pnl"})


def monthly_pnl(df, pnl_col):
    df = df.copy()
    df["_m"] = df["dateStart"].dt.to_period("M")
    out = {}
    for m, g in df.groupby("_m"):
        wins = int((g[pnl_col] > 0).sum())
        out[str(m)] = {
            "pnl": float(g[pnl_col].sum()),
            "trades": len(g),
            "wr": wins / len(g) * 100 if len(g) else 0,
        }
    return out


# ============================================================
# LOAD ALL SCENARIOS
# ============================================================
trades = load_trades(ANALYTICS_CSV)

st.markdown("# 🎯 TP Target Simulator")
st.caption(
    f"Source: **{ANALYTICS_CSV}** · {len(trades)} trades · "
    f"{trades['dateStart'].min():%Y-%m-%d} → {trades['dateEnd'].max():%Y-%m-%d}  "
    f"·  Daily rules: TP-hit ends day · Daily cum ≤ -2R ends day"
)

# Baseline: actual PnL
df_a = trades.copy()
df_a["pnl_a"] = df_a["rPnL"]

# Compute scenarios for each TP target
scenario_results = {}
for tp in TP_TARGETS:
    with st.spinner(f"Computing {tp:.0f}R scenario..."):
        df_x, skipped_tp, skipped_ds, no_data = build_scenario(tp)
        scenario_results[tp] = (df_x, skipped_tp, skipped_ds, no_data)

# Build SCENARIOS list
COLORS = {2.0: "#fb8c00", 3.0: "#26a69a", 4.0: "#5e35b1", 5.0: "#e91e63"}

SCENARIOS = [
    {"name": "A · Actual", "color": "#9e9e9e", "df": df_a, "pnl_col": "pnl_a", "skipped_tp": 0, "skipped_ds": 0},
]
for tp in TP_TARGETS:
    df_x, skipped_tp, skipped_ds, _ = scenario_results[tp]
    df_x = df_x.rename(columns={"sim_pnl": f"pnl_{int(tp)}r"})
    SCENARIOS.append({
        "name": f"{['B','C','D','E'][TP_TARGETS.index(tp)]} · {tp:.0f}R TP",
        "color": COLORS[tp],
        "df": df_x,
        "pnl_col": f"pnl_{int(tp)}r",
        "skipped_tp": skipped_tp,
        "skipped_ds": skipped_ds,
    })

# Metrics
for s in SCENARIOS:
    s["metrics"] = compute_metrics(s["df"], s["pnl_col"])
    s["eq"] = equity_curve(s["df"], s["pnl_col"])
    s["mo"] = monthly_pnl(s["df"], s["pnl_col"])

# ============================================================
# UI — HEADER CARDS
# ============================================================
cols = st.columns(5)
for col, s in zip(cols, SCENARIOS):
    with col:
        meta = ""
        if "TP" in s["name"]:
            meta = (
                f"<br><span style='font-size:11px;opacity:0.8'>"
                f"{s['skipped_tp']} skipped after TP · {s['skipped_ds']} after -2R"
                f"</span>"
            )
        st.markdown(
            f"<div style='padding:10px;border-radius:6px;background:{hex_to_rgba(s['color'], 0.15)};"
            f"border-left:4px solid {s['color']}'>"
            f"<b style='font-size:14px'>{s['name']}</b>{meta}"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ============================================================
# HEADLINE METRICS
# ============================================================
st.markdown("## Headline metrics")
cols = st.columns(5)
for col, s in zip(cols, SCENARIOS):
    m = s["metrics"]
    with col:
        st.markdown(
            f"<div style='color:{s['color']};font-weight:600;font-size:14px'>"
            f"{s['name'].split(' · ')[0]}</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("Total PnL", f"${m['total_pnl']:,.0f}")
        c2.metric("Win rate", f"{m['wr']:.2f}%", f"{m['wins']}W/{m['losses']}L")
        c1.metric("Max DD", f"${m['max_dd']:,.0f}", f"{m['max_dd_pct']:.1f}%")
        c2.metric("PF", f"{m['pf']:.2f}" if m['pf'] != float('inf') else "∞")
        c1.metric("Trades", f"{m['trades']}")
        c2.metric("Expect.", f"${m['expectancy']:.0f}/trd")
        c1.metric("Avg win", f"${m['avg_win']:.0f}")
        c2.metric("Avg loss", f"${m['avg_loss']:.0f}")
        c1.metric("Worst day", f"${m['worst_day']:,.0f}")
        c2.metric("Best day", f"${m['best_day']:,.0f}")
        c1.metric("Win days", f"{m['winning_days']}/{m['trading_days']}")
        c2.metric("Loss streak", f"{m['longest_loss_streak']}")
        if m.get("tp_hits") is not None:
            c1.metric("TP hits", f"{m['tp_hits']}")
            c2.metric("SL hits", f"{m['sl_hits']}")

st.divider()

# ============================================================
# EQUITY CURVES OVERLAID
# ============================================================
st.markdown("## Equity curves")
fig_eq = go.Figure()
for s in SCENARIOS:
    eq = s["eq"]
    if len(eq) == 0: continue
    fig_eq.add_trace(go.Scatter(
        x=eq["dateStart"], y=eq["balance"],
        mode="lines", name=s["name"],
        line=dict(color=s["color"], width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
    ))
fig_eq.add_hline(y=STARTING_BAL, line=dict(color="#666", width=1, dash="dash"))
fig_eq.update_layout(
    height=450, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font=dict(color="#fafafa"), yaxis_title="Account balance ($)",
    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    margin=dict(l=50, r=20, t=40, b=40),
)
fig_eq.update_xaxes(showgrid=False)
fig_eq.update_yaxes(gridcolor="#333")
st.plotly_chart(fig_eq, use_container_width=True)

# ============================================================
# DRAWDOWN CURVES OVERLAID
# ============================================================
st.markdown("## Drawdown curves")
fig_dd = go.Figure()
for s in SCENARIOS:
    eq = s["eq"]
    if len(eq) == 0: continue
    fig_dd.add_trace(go.Scatter(
        x=eq["dateStart"], y=eq["dd"],
        mode="lines", name=s["name"],
        line=dict(color=s["color"], width=2),
        fill="tozeroy", fillcolor=hex_to_rgba(s["color"], 0.12),
        hovertemplate="%{x|%Y-%m-%d}<br>DD: $%{y:,.0f}<extra></extra>",
    ))
fig_dd.update_layout(
    height=300, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font=dict(color="#fafafa"), yaxis_title="Drawdown ($)",
    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    margin=dict(l=50, r=20, t=40, b=40),
)
fig_dd.update_xaxes(showgrid=False)
fig_dd.update_yaxes(gridcolor="#333", zerolinecolor="#666")
st.plotly_chart(fig_dd, use_container_width=True)

st.divider()

# ============================================================
# YEAR-BY-YEAR
# ============================================================
st.markdown("## Year-by-year PnL")
year_data = []
for s in SCENARIOS:
    year_data.append({
        "Strategy": s["name"].split(" · ")[0],
        "2023": s["metrics"]["pnl_2023"],
        "2024": s["metrics"]["pnl_2024"],
        "2025": s["metrics"]["pnl_2025"],
        "Total": s["metrics"]["total_pnl"],
    })
year_df = pd.DataFrame(year_data)
fig_yr = go.Figure()
for y in ["2023", "2024", "2025"]:
    fig_yr.add_trace(go.Bar(
        name=y, x=year_df["Strategy"], y=year_df[y],
        text=[f"${v:,.0f}" for v in year_df[y]],
        textposition="outside",
    ))
fig_yr.update_layout(
    barmode="group", height=380,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font=dict(color="#fafafa"), yaxis_title="PnL ($)",
    margin=dict(l=40, r=20, t=20, b=40),
)
fig_yr.update_xaxes(showgrid=False)
fig_yr.update_yaxes(gridcolor="#333", zerolinecolor="#666")
st.plotly_chart(fig_yr, use_container_width=True)

st.divider()

# ============================================================
# MONTHLY COMPARISON
# ============================================================
st.markdown("## Monthly comparison")
st.caption("Each row is one month · PnL · Trades · WR per scenario.")

all_months = sorted(set().union(*[s["mo"].keys() for s in SCENARIOS]))

header = st.columns([1.0] + [1.8] * 5)
header[0].markdown("**Month**")
for i, s in enumerate(SCENARIOS):
    header[i+1].markdown(
        f"<div style='color:{s['color']};font-weight:600;font-size:13px'>"
        f"{s['name'].split(' · ')[0]}</div>",
        unsafe_allow_html=True,
    )

for m in all_months:
    row = st.columns([1.0] + [1.8] * 5)
    row[0].markdown(f"**{m}**")
    for i, s in enumerate(SCENARIOS):
        d = s["mo"].get(m)
        with row[i+1]:
            if d is None:
                st.markdown("<span style='opacity:0.4'>—</span>", unsafe_allow_html=True)
            else:
                color = "#4caf50" if d["pnl"] > 0 else "#ef5350" if d["pnl"] < 0 else "#aaa"
                st.markdown(
                    f"<span style='color:{color};font-weight:600;font-size:13px'>"
                    f"${d['pnl']:,.0f}</span> "
                    f"<span style='opacity:0.75;font-size:10px'>"
                    f"({d['trades']} trd, {d['wr']:.0f}%)</span>",
                    unsafe_allow_html=True,
                )

st.divider()

# ============================================================
# DELTAS vs BASELINE
# ============================================================
st.markdown("## Δ vs Actual (A)")
base = SCENARIOS[0]["metrics"]
delta_rows = []
for s in SCENARIOS[1:]:
    m = s["metrics"]
    delta_rows.append({
        "Scenario": s["name"],
        "Δ PnL": f"${m['total_pnl']-base['total_pnl']:+,.0f}",
        "Δ WR": f"{m['wr']-base['wr']:+.2f}%",
        "Δ Max DD": f"${abs(m['max_dd'])-abs(base['max_dd']):+,.0f}" + " (worse)" if abs(m['max_dd']) > abs(base['max_dd']) else f"${abs(m['max_dd'])-abs(base['max_dd']):+,.0f}" + " (better)",
        "Trades": f"{m['trades']} (Δ {m['trades']-base['trades']:+d})",
        "TP hits": str(m.get('tp_hits','—')),
        "Worst day": f"${m['worst_day']:,.0f}",
    })
st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)

# ============================================================
# DOWNLOAD
# ============================================================
st.divider()
st.markdown("### Download")
keys = list(SCENARIOS[0]["metrics"].keys())
data = {"metric": keys}
for s in SCENARIOS:
    data[s["name"]] = [s["metrics"][k] for k in keys]
st.download_button(
    "📥 Download metrics CSV",
    data=pd.DataFrame(data).to_csv(index=False).encode("utf-8"),
    file_name="tp_simulator_metrics.csv",
    mime="text/csv",
)