"""
monthly_audit.py — per-day SMT audit for the locked strategy.

For a chosen month, shows EVERY SMT detected (in MNQ or MES, ANY level type,
ANY session) and labels it with one of these verdicts:

  🟢 TRADED               — strategy entered, here's the R outcome
  🟡 AUTO_OTHER_ASSET     — same SMT was traded on the OTHER asset
  🟠 ENTRY_SKIPPED        — MSS formed but managed mode blocked entry
  🔵 NO_MSS               — no MSS produced in the session
  🔻 INVALIDATED          — non-confirmer also crossed the level (SMT broke)
  ⚫ EXCLUDED_LEVEL       — level type is in your exclusion list
  ⚫ EXCLUDED_SESSION     — MSS time outside london + ny_am

Use this to compare your manual backtest with what the code did/didn't do.

LOCKED STRATEGY (same as voligle_tp2.py):
  Asset       = Auto (first MSS, fallback if skipped)
  TF          = 5m
  Anchor      = before_smt
  Rule        = wick_break
  TP          = TP2 (managed mode ON)
  Sessions    = london + ny_am
  Levels excl = AsiaONHigh, AsiaONLow, PWH, PWL
  Risk        = $250 / trade

Run:
  streamlit run monthly_audit.py --server.port 8507
"""

import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# LOCKED STRATEGY (same as voligle_tp2.py)
# ============================================================
DATA_FOLDER = "/Users/voligle/workspace/Trading/Trading_code/clause Backtester"
OUT_FOLDER = os.path.join(DATA_FOLDER, "New/out")
TICK_SIZE = 0.25
EPS = 1e-9

TF               = "5m"
ANCHOR           = "before_smt"
RULE             = "wick_break"
TP_LEVEL         = 2
TP_COL_MGD       = "r_tp2_mgd"
HIT_COL          = "hit_2r"
HIT_TS_COL       = "hit_2r_ts"
SESSIONS         = ["london", "ny_am"]
EXCLUDE_LEVELS   = {"AsiaONHigh", "AsiaONLow", "PWH", "PWL"}
RISK_PER_TRADE   = 250.0

HIGH_LEVELS = {"PDH","PWH","LondonHigh","AsiaKZHigh","AsiaONHigh","4H_SwingHigh"}
LOW_LEVELS  = {"PDL","PWL","LondonLow","AsiaKZLow","AsiaONLow","4H_SwingLow"}
ALL_LEVELS  = HIGH_LEVELS | LOW_LEVELS
PERSISTENT_TYPES = {"PDH","PDL","PWH","PWL","4H_SwingHigh","4H_SwingLow"}
LEVEL_TYPES_USED = sorted(ALL_LEVELS - EXCLUDE_LEVELS)

SHORT_LABEL = {
    "PDH": "PDH", "PDL": "PDL", "PWH": "PWH", "PWL": "PWL",
    "LondonHigh": "LH", "LondonLow": "LL",
    "AsiaKZHigh": "AKH", "AsiaKZLow": "AKL",
    "AsiaONHigh": "AOH", "AsiaONLow": "AOL",
    "4H_SwingHigh": "SWH", "4H_SwingLow": "SWL",
}

st.set_page_config(page_title="Monthly Audit", layout="wide")

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
def load_smt_events(mtime):
    """Load ALL SMT events (the full universe of SMTs detected by Phase 3)."""
    files = glob.glob(os.path.join(OUT_FOLDER, "SMT_events_*.csv"))
    if not files:
        return pd.DataFrame()
    all_df = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if len(d) == 0: continue
            for c in ("birth_ts","death_ts"):
                if c in d.columns:
                    d[c] = pd.to_datetime(d[c], utc=True, errors="coerce").dt.tz_convert("America/New_York")
            all_df.append(d)
        except Exception:
            continue
    if not all_df:
        return pd.DataFrame()
    df = pd.concat(all_df, ignore_index=True)
    # Keep both 'session_end' and 'invalidated_strict' so audit can explain invalidations
    df = df[df["status"].isin(["session_end", "invalidated_strict"])]
    df = df.drop_duplicates(
        subset=["birth_ts","sweeper","level_type","level_price_sweeper"]
    ).reset_index(drop=True)
    df = df.sort_values("birth_ts").reset_index(drop=True)
    return df


@st.cache_data
def load_measurements(path, mtime):
    """Load the full measurements (24 rows per SMT)."""
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
    return df


# ============================================================
# AUDIT — for each SMT, attach a verdict
# ============================================================
def session_day_of(ts):
    if ts.hour >= 18:
        return (ts.normalize() + pd.Timedelta(days=1)).date()
    return ts.normalize().date()


def session_bounds(date):
    start = pd.Timestamp(date, tz="America/New_York") - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
    end = pd.Timestamp(date, tz="America/New_York") + pd.Timedelta(hours=17)
    return start, end


def session_end_for(ts):
    if ts.hour >= 18:
        return ts.normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=17)
    return ts.normalize() + pd.Timedelta(hours=17)


def build_audit(smts, meas):
    """
    For each (smt_birth_ts, sweeper, level_type, level_price_sweeper) in smts,
    look up the four measurement rows in `meas` for the locked strategy:
      - 2 assets × (TF=5m, anchor=before_smt, rule=wick_break)

    Return a DataFrame with one row per SMT with these columns:
      smt_id, smt_birth_ts, session_day, level_id, level_type, direction,
      sweeper, non_confirmer, level_price_sweeper, level_price_nonconf,
      separation_ticks,
      verdict, verdict_reason,
      auto_picked_asset, mss_ts, entry_price, sl_price,
      r_outcome, pnl_usd, hit_2r, managed_exit_reason
    """
    rows_out = []

    # Index measurements by (smt_birth_ts, measured_asset) for fast lookup
    meas_strategy = meas[
        (meas["mss_timeframe"] == TF)
        & (meas["anchor"] == ANCHOR)
        & (meas["mss_rule"] == RULE)
    ].copy()
    meas_by_key = {}
    for _, r in meas_strategy.iterrows():
        key = (r["smt_birth_ts"], r["measured_asset"])
        meas_by_key[key] = r

    for _, s in smts.iterrows():
        smt_ts = s["birth_ts"]
        session_day = session_day_of(smt_ts)
        level_type = s["level_type"]
        direction = s["direction"]
        smt_status = s.get("status", "session_end")
        smt_death_ts = s.get("death_ts")

        # Look up measurements for both assets
        m_mnq = meas_by_key.get((smt_ts, "MNQ"))
        m_mes = meas_by_key.get((smt_ts, "MES"))

        # Determine verdict
        verdict = "UNKNOWN"
        verdict_reason = ""
        auto_picked_asset = None
        chosen_row = None

        # Check 1: excluded level type
        if level_type in EXCLUDE_LEVELS:
            verdict = "EXCLUDED_LEVEL"
            verdict_reason = f"Level type '{level_type}' is in exclusion list"
        else:
            # Find auto-picked row (if any)
            picked_rows = []
            if m_mnq is not None and m_mnq.get("auto_picked") == True:
                picked_rows.append(("MNQ", m_mnq))
            if m_mes is not None and m_mes.get("auto_picked") == True:
                picked_rows.append(("MES", m_mes))

            if picked_rows:
                # Was traded — confirm session is london/ny_am
                pa, pr = picked_rows[0]
                pr_session = pr.get("session", "")
                if pr_session not in SESSIONS:
                    verdict = "EXCLUDED_SESSION"
                    verdict_reason = f"MSS happened in '{pr_session}' (not london/ny_am)"
                    auto_picked_asset = pa
                    chosen_row = pr
                else:
                    verdict = "TRADED"
                    auto_picked_asset = pa
                    chosen_row = pr
                    verdict_reason = f"Auto-picked {pa} (reason: {pr.get('auto_pick_reason','')})"
            else:
                # Not auto-picked. Why?
                # Phase 4 only generates measurements for status=session_end SMTs.
                # If the SMT was invalidated, neither m_mnq nor m_mes will exist
                # (or they'll exist but won't be auto-picked).
                if smt_status == "invalidated_strict":
                    # SMT was invalidated by the non-confirmer crossing too.
                    # Did either asset produce an MSS BEFORE invalidation?
                    has_mss_before_invalidation = []
                    for asset, m in [("MNQ", m_mnq), ("MES", m_mes)]:
                        if m is not None and m.get("mss_found") == True:
                            mss_ts = m.get("mss_ts")
                            if pd.notna(mss_ts) and pd.notna(smt_death_ts) and mss_ts < smt_death_ts:
                                has_mss_before_invalidation.append((asset, m))

                    if has_mss_before_invalidation:
                        # MSS formed BEFORE invalidation but auto-pick still rejected
                        # (probably because Phase 3/4 dropped the SMT for status reasons).
                        # Show as INVALIDATED with note.
                        a, m = has_mss_before_invalidation[0]
                        chosen_row = m
                        auto_picked_asset = a
                        verdict = "INVALIDATED"
                        death_str = pd.Timestamp(smt_death_ts).strftime("%H:%M") if pd.notna(smt_death_ts) else "?"
                        verdict_reason = (f"Non-confirmer ({s.get('non_confirmer','')}) also crossed at {death_str}. "
                                          f"MSS formed before invalidation but SMT was discarded.")
                    else:
                        verdict = "INVALIDATED"
                        death_str = pd.Timestamp(smt_death_ts).strftime("%H:%M") if pd.notna(smt_death_ts) else "?"
                        verdict_reason = (f"Non-confirmer ({s.get('non_confirmer','')}) also crossed at {death_str} "
                                          f"before any MSS could form.")
                else:
                    # status = session_end (SMT survived) but not auto-picked.
                    # Check if either asset had MSS at all
                    has_mss = []
                    for asset, m in [("MNQ", m_mnq), ("MES", m_mes)]:
                        if m is not None and m.get("mss_found") == True:
                            has_mss.append((asset, m))

                    if not has_mss:
                        verdict = "NO_MSS"
                        verdict_reason = "Neither asset produced an MSS in the session"
                    else:
                        # MSS exists but auto-pick rejected it. Check sessions and entry_skipped
                        skipped = [(a, m) for a, m in has_mss if m.get("entry_skipped") == True]
                        wrong_sess = [(a, m) for a, m in has_mss if m.get("session", "") not in SESSIONS]

                        if len(skipped) == len(has_mss):
                            a, m = skipped[0]
                            verdict = "ENTRY_SKIPPED"
                            verdict_reason = f"Entry skipped: {m.get('entry_skip_reason','?')}"
                            chosen_row = m
                            auto_picked_asset = a
                        elif len(wrong_sess) == len(has_mss):
                            a, m = wrong_sess[0]
                            verdict = "EXCLUDED_SESSION"
                            verdict_reason = f"MSS in '{m.get('session','')}' (not london/ny_am)"
                            chosen_row = m
                            auto_picked_asset = a
                        else:
                            a, m = has_mss[0]
                            if m.get("session","") not in SESSIONS:
                                verdict = "EXCLUDED_SESSION"
                                verdict_reason = f"MSS in '{m.get('session','')}' (not london/ny_am)"
                            else:
                                verdict = "ENTRY_SKIPPED"
                                verdict_reason = f"Skipped: {m.get('entry_skip_reason','?')}"
                            chosen_row = m
                            auto_picked_asset = a

        # Build the row
        row = {
            "smt_birth_ts": smt_ts,
            "session_day": session_day,
            "level_id": s.get("level_id", ""),
            "level_type": level_type,
            "direction": direction,
            "sweeper": s.get("sweeper", ""),
            "non_confirmer": s.get("non_confirmer", ""),
            "level_price_sweeper": s.get("level_price_sweeper"),
            "level_price_nonconf": s.get("level_price_nonconf"),
            "separation_ticks": s.get("separation_ticks"),
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "auto_picked_asset": auto_picked_asset,
            "mss_ts": chosen_row["mss_ts"] if chosen_row is not None else pd.NaT,
            "session": chosen_row.get("session", "") if chosen_row is not None else "",
            "entry_price": chosen_row.get("entry_price") if chosen_row is not None else None,
            "sl_price": chosen_row.get("sl_price") if chosen_row is not None else None,
            "r_outcome": chosen_row.get(TP_COL_MGD) if (chosen_row is not None and verdict == "TRADED") else None,
            "hit_2r": bool(chosen_row.get(HIT_COL, False)) if (chosen_row is not None and verdict == "TRADED") else None,
            "managed_exit_reason": chosen_row.get("managed_exit_reason", "") if chosen_row is not None else "",
        }
        if row["r_outcome"] is not None and pd.notna(row["r_outcome"]):
            row["pnl_usd"] = float(row["r_outcome"]) * RISK_PER_TRADE
        else:
            row["pnl_usd"] = None
        rows_out.append(row)

    audit = pd.DataFrame(rows_out)
    audit = audit.sort_values("smt_birth_ts").reset_index(drop=True)
    return audit


# ============================================================
# UI
# ============================================================
st.markdown("# 🔎 Monthly Audit — Locked Strategy")
st.caption(
    f"Asset=**Auto** · TF=**{TF}** · Anchor=**{ANCHOR}** · Rule=**{RULE}** · "
    f"TP=**TP{TP_LEVEL} (managed)** · Sessions=**{', '.join(SESSIONS)}** · "
    f"Excl levels=**{', '.join(sorted(EXCLUDE_LEVELS))}** · Risk=**${int(RISK_PER_TRADE)}**"
)

# Load data
measurement_files = sorted(glob.glob(os.path.join(OUT_FOLDER, "SMT_measurements_*.csv")))
if not measurement_files:
    st.error(f"No SMT_measurements_*.csv found in {OUT_FOLDER}. Run phase4.py first.")
    st.stop()
csv_path = measurement_files[-1]
meas = load_measurements(csv_path, os.path.getmtime(csv_path))

smt_files = sorted(glob.glob(os.path.join(OUT_FOLDER, "SMT_events_*.csv")))
if not smt_files:
    st.error(f"No SMT_events_*.csv found in {OUT_FOLDER}. Run phase3.py first.")
    st.stop()
mtimes = sum(os.path.getmtime(f) for f in smt_files)
smts = load_smt_events(mtimes)

if len(meas) == 0 or len(smts) == 0:
    st.error("Empty SMT or measurement data.")
    st.stop()

# Build audit (all SMTs across all dates)
with st.spinner("Building audit table..."):
    audit = build_audit(smts, meas)

# Filter to match the locked strategy:
# - exclude SMTs of excluded level types
# - exclude SMTs whose MSS happened outside london/ny_am
# This is identical to what voligle_tp2 considers the strategy's universe.
audit = audit[audit["verdict"] != "EXCLUDED_LEVEL"]
audit = audit[audit["verdict"] != "EXCLUDED_SESSION"]
audit = audit.reset_index(drop=True)

# ============================================================
# MONTH PICKER
# ============================================================
audit["session_day"] = pd.to_datetime(audit["session_day"])
all_months = sorted(audit["session_day"].dt.to_period("M").unique())
month_options = [str(p) for p in all_months]

if "audit_month_idx" not in st.session_state:
    st.session_state.audit_month_idx = len(month_options) - 1
st.session_state.audit_month_idx = max(0, min(st.session_state.audit_month_idx, len(month_options) - 1))

mc1, mc2, mc3 = st.columns([1, 4, 1])
with mc1:
    if st.button("← Prev month", use_container_width=True):
        st.session_state.audit_month_idx = max(0, st.session_state.audit_month_idx - 1)
with mc2:
    sel_idx = month_options.index(
        st.selectbox("Month", month_options, index=st.session_state.audit_month_idx,
                       label_visibility="collapsed")
    )
    st.session_state.audit_month_idx = sel_idx
with mc3:
    if st.button("Next month →", use_container_width=True):
        st.session_state.audit_month_idx = min(len(month_options) - 1, st.session_state.audit_month_idx + 1)

selected_month = all_months[st.session_state.audit_month_idx]
month_audit = audit[audit["session_day"].dt.to_period("M") == selected_month].copy()

# ============================================================
# MONTH SUMMARY
# ============================================================
st.markdown(f"## {selected_month}")

verdict_counts = month_audit["verdict"].value_counts().to_dict()
total_smts = len(month_audit)
n_traded = verdict_counts.get("TRADED", 0)
n_other = verdict_counts.get("AUTO_OTHER_ASSET", 0)
n_skipped = verdict_counts.get("ENTRY_SKIPPED", 0)
n_no_mss = verdict_counts.get("NO_MSS", 0)
n_invalidated = verdict_counts.get("INVALIDATED", 0)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total SMTs", f"{total_smts}")
c2.metric("🟢 TRADED", f"{n_traded}")
c3.metric("🟠 ENTRY SKIPPED", f"{n_skipped}")
c4.metric("🔵 NO MSS", f"{n_no_mss}")
c5.metric("🔻 INVALIDATED", f"{n_invalidated}")

# Trade summary
traded = month_audit[month_audit["verdict"] == "TRADED"]
if len(traded) > 0:
    wins = traded[traded["r_outcome"] > 0]
    losses = traded[traded["r_outcome"] <= 0]
    pnl = traded["pnl_usd"].sum()
    wr = len(wins) / len(traded) * 100
    tp2_hits = int(traded["hit_2r"].sum()) if "hit_2r" in traded.columns else 0
    tp2_rate = tp2_hits / len(traded) * 100 if len(traded) else 0

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Trades", f"{len(traded)}", f"{len(wins)}W / {len(losses)}L")
    s2.metric("Win rate", f"{wr:.1f}%")
    s3.metric("TP2 hit rate", f"{tp2_rate:.1f}%")
    s4.metric("Month PnL", f"${pnl:,.2f}")
    s5.metric("Total R", f"{traded['r_outcome'].sum():.2f}R")

st.divider()

# ============================================================
# DAY-BY-DAY BREAKDOWN
# ============================================================

VERDICT_STYLES = {
    "TRADED":              {"emoji": "🟢", "color": "#26a69a", "bg": "#0a2818"},
    "AUTO_OTHER_ASSET":    {"emoji": "🟡", "color": "#d4a017", "bg": "#2a2410"},
    "ENTRY_SKIPPED":       {"emoji": "🟠", "color": "#ff9800", "bg": "#2b1f0a"},
    "NO_MSS":              {"emoji": "🔵", "color": "#2196f3", "bg": "#0a1828"},
    "INVALIDATED":         {"emoji": "🔻", "color": "#9c27b0", "bg": "#1a0a1f"},
    "UNKNOWN":             {"emoji": "❓", "color": "#888",    "bg": "#1a1a1a"},
}

# Group by session_day
days_with_smts = sorted(month_audit["session_day"].dt.date.unique())

if len(days_with_smts) == 0:
    st.info("No SMTs detected this month.")
    st.stop()

# Optional verdict filter
verdict_filter = st.multiselect(
    "Show only these verdicts",
    list(VERDICT_STYLES.keys()),
    default=list(VERDICT_STYLES.keys()),
    help="Filter the day-by-day list to focus on specific verdicts.",
)

st.divider()

for day in days_with_smts:
    day_audit = month_audit[month_audit["session_day"].dt.date == day]
    day_audit_filtered = day_audit[day_audit["verdict"].isin(verdict_filter)]
    if len(day_audit_filtered) == 0:
        continue

    # Day header summary
    day_traded = day_audit[day_audit["verdict"] == "TRADED"]
    day_pnl = day_traded["pnl_usd"].sum() if len(day_traded) else 0
    day_n_total = len(day_audit)
    day_n_traded = len(day_traded)

    if day_pnl > 0:
        pnl_str = f"<span style='color:#4caf50;font-weight:600'>+${day_pnl:.2f}</span>"
    elif day_pnl < 0:
        pnl_str = f"<span style='color:#ef5350;font-weight:600'>${day_pnl:.2f}</span>"
    elif day_n_traded > 0:
        pnl_str = "<span style='color:#d4a017'>$0.00</span>"
    else:
        pnl_str = "<span style='color:#666'>(no trades)</span>"

    st.markdown(
        f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
                    padding:8px 12px;background:#1a1a1a;border-radius:6px;margin-top:12px'>
          <div style='font-size:16px;font-weight:600'>{day} ({day.strftime('%A')})</div>
          <div style='font-size:14px'>
            {day_n_total} SMT{'s' if day_n_total>1 else ''}, 
            {day_n_traded} traded · {pnl_str}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for i, smt in day_audit_filtered.iterrows():
        v = smt["verdict"]
        st_v = VERDICT_STYLES.get(v, VERDICT_STYLES["UNKNOWN"])
        emoji = st_v["emoji"]
        color = st_v["color"]
        bg = st_v["bg"]

        # Build summary line
        time_str = pd.Timestamp(smt["smt_birth_ts"]).strftime("%H:%M")
        direction_emoji = "🟢" if smt["direction"] == "bullish" else "🔴"
        lvl_short = SHORT_LABEL.get(smt["level_type"], smt["level_type"])
        sweeper = smt["sweeper"]
        sep = smt.get("separation_ticks")
        sep_str = f"{sep:.0f}t" if pd.notna(sep) else ""

        # Trade outcome (if any)
        outcome_str = ""
        if v == "TRADED":
            r = smt.get("r_outcome", 0)
            outcome_emoji = "🎯" if smt.get("hit_2r") else ("🪂" if r > 0 else "❌")
            entry = smt.get("entry_price")
            sl = smt.get("sl_price")
            mss_t = pd.Timestamp(smt["mss_ts"]).strftime("%H:%M") if pd.notna(smt.get("mss_ts")) else "—"
            outcome_str = (
                f" · entered {smt['auto_picked_asset']} @ {entry:.2f} (SL {sl:.2f}) "
                f"at {mss_t} → {outcome_emoji} <b>{r:+.2f}R</b> "
                f"(${r * RISK_PER_TRADE:+,.0f}) · {smt.get('managed_exit_reason','')}"
            )

        # Action button
        cols = st.columns([10, 1])
        with cols[0]:
            st.markdown(
                f"""
                <div style='background:{bg};border-left:4px solid {color};
                            border-radius:4px;padding:8px 12px;margin:4px 0;font-size:13px'>
                  <span style='font-size:14px'>{emoji}</span>
                  &nbsp;<b>{time_str}</b>
                  &nbsp;{direction_emoji} <b>{lvl_short}</b>
                  &nbsp;<span style='color:#888'>swept by {sweeper}{(' · sep ' + sep_str) if sep_str else ''}</span>
                  &nbsp;·&nbsp;<span style='color:{color};font-weight:600'>{v}</span>
                  &nbsp;<span style='color:#aaa'>{smt['verdict_reason']}</span>
                  {outcome_str}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cols[1]:
            if st.button("📈", key=f"chart_{i}_{smt['smt_birth_ts']}", help="View chart"):
                st.session_state.audit_selected_smt = smt.to_dict()


# ============================================================
# DOWNLOAD
# ============================================================
st.divider()
st.markdown("## Export")

# Export columns (rounded for readability)
exp = month_audit.copy()
for c in ("level_price_sweeper","level_price_nonconf","entry_price","sl_price",
          "r_outcome","pnl_usd","separation_ticks"):
    if c in exp.columns:
        exp[c] = pd.to_numeric(exp[c], errors="coerce").round(2)

# Pretty column order
core = [
    "session_day","smt_birth_ts","direction","level_type","sweeper","non_confirmer",
    "level_price_sweeper","level_price_nonconf","separation_ticks",
    "verdict","verdict_reason","auto_picked_asset",
    "mss_ts","session","entry_price","sl_price",
    "r_outcome","hit_2r","pnl_usd","managed_exit_reason","level_id",
]
core = [c for c in core if c in exp.columns]
exp = exp[core + [c for c in exp.columns if c not in core]]

st.download_button(
    f"📥 Download audit for {selected_month} (CSV)",
    data=exp.to_csv(index=False).encode("utf-8"),
    file_name=f"audit_{selected_month}.csv",
    mime="text/csv",
    use_container_width=True,
)


# ============================================================
# CHART DIALOG
# ============================================================
def find_extreme(bars_1m, start, end, find_high):
    sub = bars_1m.loc[start:end]
    if len(sub) == 0:
        return None
    return sub["high"].idxmax() if find_high else sub["low"].idxmin()


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


@st.dialog("SMT chart")
def show_smt_chart(smt):
    smt_ts = pd.Timestamp(smt["smt_birth_ts"])
    if pd.isna(smt_ts):
        st.error("Missing SMT timestamp"); return
    session_date_str = (smt_ts.normalize() if smt_ts.hour < 18 else (smt_ts + pd.Timedelta(days=1)).normalize()).strftime("%Y-%m-%d")
    s_start, s_end = session_bounds(session_date_str)
    mnq_bars = load_bars("MNQ", TF).loc[s_start:s_end]
    mes_bars = load_bars("MES", TF).loc[s_start:s_end]
    mnq_lv = load_levels("MNQ"); mes_lv = load_levels("MES")
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

    # Draw the SMT level lines on both panels
    level_id = smt.get("level_id", "")
    lt = smt.get("level_type", "")
    is_high = lt in HIGH_LEVELS
    sweeper = smt.get("sweeper", "")
    sweep_ts = smt_ts

    mnq_lvl_row = mnq_lv[mnq_lv["level_id"] == level_id]
    mes_lvl_row = mes_lv[mes_lv["level_id"] == level_id]

    if len(mnq_lvl_row) and len(mes_lvl_row):
        mnq_r = mnq_lvl_row.iloc[0]
        mes_r = mes_lvl_row.iloc[0]
        for asset, r, col in [("MNQ", mnq_r, 1), ("MES", mes_r, 2)]:
            ts_formed = r["ts_formed"]
            vf = r["valid_from"]
            win = formation_window(lt, ts_formed, vf)
            bars_1m = mnq_1m if asset == "MNQ" else mes_1m
            t_anchor = find_extreme(bars_1m, win[0], win[1], is_high)
            if t_anchor is None: continue
            t_start = max(t_anchor, s_start)
            yshift = 10 if is_high else -10
            yanchor = "bottom" if is_high else "top"
            fig.add_shape(type="line", x0=t_start, x1=sweep_ts, y0=r["price"], y1=r["price"],
                           line=dict(color="black", width=1), row=1, col=col)
            fig.add_annotation(x=sweep_ts, y=r["price"],
                                text=f" {SHORT_LABEL.get(lt, lt)} ",
                                showarrow=False, font=dict(size=9, color="black"),
                                xanchor="left", yanchor=yanchor, yshift=yshift,
                                row=1, col=col)

    # Draw position tool if traded
    if smt.get("verdict") == "TRADED" and pd.notna(smt.get("mss_ts")):
        asset = smt.get("auto_picked_asset", "")
        col = 1 if asset == "MNQ" else 2
        bars_1m_a = mnq_1m if asset == "MNQ" else mes_1m
        mss_ts = pd.Timestamp(smt["mss_ts"])
        entry = float(smt["entry_price"])
        sl = float(smt["sl_price"])
        r_pts = abs(entry - sl)
        direction = smt["direction"]

        # Find exit ts (from measurement managed_exit_ts not in audit; reload)
        end_session = pd.Timestamp(smt["session_day"], tz="America/New_York") + pd.Timedelta(hours=17)
        # Default to session end; if SL or TP2 hit in 1m bars, that becomes exit
        exit_ts = end_session
        sub_walk = bars_1m_a.loc[mss_ts + pd.Timedelta(minutes=1):end_session]
        for ts, bar in sub_walk.iterrows():
            tp2 = entry + 2 * r_pts if direction == "bullish" else entry - 2 * r_pts
            if direction == "bullish":
                if bar["low"] <= sl + EPS:
                    exit_ts = ts; break
                if bar["high"] >= tp2 - EPS:
                    exit_ts = ts; break
            else:
                if bar["high"] >= sl - EPS:
                    exit_ts = ts; break
                if bar["low"] <= tp2 + EPS:
                    exit_ts = ts; break

        r_levels = [(n, entry + n * r_pts if direction == "bullish" else entry - n * r_pts) for n in (1, 2)]
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
        label_off = pd.Timedelta(minutes=2)
        fig.add_annotation(x=exit_ts + label_off, y=entry, text=f"E {entry:.2f}",
                            showarrow=False, font=dict(size=9, color="#1976d2"),
                            xanchor="left", yanchor="middle", row=1, col=col)
        fig.add_annotation(x=exit_ts + label_off, y=sl, text=f"SL {sl:.2f}",
                            showarrow=False, font=dict(size=9, color="#d32f2f"),
                            xanchor="left", yanchor="middle", row=1, col=col)
        for n, rv in r_levels:
            fig.add_annotation(x=exit_ts + label_off, y=rv, text=f"{n}R",
                                showarrow=False, font=dict(size=9, color="#388e3c"),
                                xanchor="left", yanchor="middle", row=1, col=col)

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

    # Detail panel
    st.markdown("#### Audit details")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Verdict", smt.get("verdict",""))
    d2.metric("Level type", smt.get("level_type",""))
    d3.metric("Direction", smt.get("direction",""))
    d4.metric("Sweeper", smt.get("sweeper",""))

    st.markdown(f"**Reason:** {smt.get('verdict_reason','')}")
    if smt.get("verdict") == "TRADED":
        e = smt.get("entry_price"); sl = smt.get("sl_price")
        r = smt.get("r_outcome"); pnl = smt.get("pnl_usd")
        st.markdown(
            f"**Entry:** {e:.2f} · **SL:** {sl:.2f} · **R:** {r:+.2f}R · **PnL:** ${pnl:+,.2f} "
            f"· **Exit reason:** {smt.get('managed_exit_reason','')}"
        )


if "audit_selected_smt" in st.session_state:
    show_smt_chart(st.session_state.audit_selected_smt)
    del st.session_state.audit_selected_smt