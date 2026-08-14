"""
sweep_strategies.py — exhaustive strategy sweep.

Locked parameters:
  Asset       = Auto (first MSS)
  Sessions    = london + ny_am
  Levels excl = AsiaONHigh, AsiaONLow, PWH, PWL
  Managed     = ON
  Risk/trade  = $250
  Dataset     = latest SMT_measurements_*.csv in New/out

Free parameters:
  TF          : 1m, 5m, 15m
  Anchor      : before_smt, after_smt
  Rule        : wick_break, close_break
  TP          : TP1, TP2, TP3, TP4, TP5
  Retracement : 0.00, 0.25, 0.50, 0.75

Total = 3 * 2 * 2 * 5 * 4 = 240 combinations.

For each combo computes:
  - Total trades, win rate, TP hit rate, mean R
  - Total PnL, profit factor, expectancy ($)
  - Max drawdown (in $ and % of account)
  - Longest losing streak (count and $)
  - Worst trading day ($)
  - 2023 / 2024 / 2025 PnL split (consistency)
  - Train (2023) vs Test (2024-25) WR (overfit check)
  - Daily Sharpe, Sortino
  - Composite score (0-100)

Output: sweep_results.csv with one row per combo, ranked by composite score.
Also prints the TOP 10 to console.

Run:
  python3 sweep_strategies.py
"""

import os
import sys
import glob
import time
import itertools
import pandas as pd
import numpy as np

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

TFS         = ["1m", "5m", "15m"]
ANCHORS     = ["before_smt", "after_smt"]
RULES       = ["wick_break", "close_break"]
TPS         = ["TP1", "TP2", "TP3", "TP4", "TP5"]
RETRACES    = [0.00, 0.25, 0.50, 0.75]

TP_TO_MULT = {"TP1": 1.0, "TP2": 2.0, "TP3": 3.0, "TP4": 4.0, "TP5": 5.0}
TP_TO_RAW = {"TP1": "r_tp1", "TP2": "r_tp2", "TP3": "r_tp3", "TP4": "r_tp4", "TP5": "r_tp5"}


# ============================================================
# LOADERS
# ============================================================
def load_measurements():
    """Load the latest SMT_measurements CSV."""
    files = sorted(glob.glob(os.path.join(OUT_FOLDER, "SMT_measurements_*.csv")))
    if not files:
        raise SystemExit(f"No SMT_measurements_*.csv found in {OUT_FOLDER}. Run phase4.py first.")
    path = files[-1]
    print(f"Loading measurements: {os.path.basename(path)}")
    df = pd.read_csv(path)
    if len(df) == 0:
        raise SystemExit("Empty measurements file.")
    ts_cols = ["smt_birth_ts","smt_death_ts","swing_ts","mss_ts","sl_leg_extreme_ts",
                "hit_1r_ts","hit_2r_ts","hit_3r_ts","hit_4r_ts","hit_5r_ts",
                "sl_hit_ts","trade_exit_ts_if_held","managed_exit_ts"]
    for c in ts_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    return df, os.path.basename(path)


def load_1m_bars():
    """Load 1m bars for both assets (used for retracement simulation)."""
    out = {}
    for asset in ("MNQ", "MES"):
        path = os.path.join(OUT_FOLDER, f"{asset}_1m.csv")
        d = pd.read_csv(path)
        d["ts_ny"] = pd.to_datetime(d["ts_ny"], utc=True).dt.tz_convert("America/New_York")
        d = d.set_index("ts_ny").sort_index()[["open","high","low","close"]]
        out[asset] = d
    return out


# ============================================================
# CORE TRADE PROCESSING
# ============================================================
def session_end_for(ts):
    if ts.hour >= 18:
        return (ts.normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=17))
    return ts.normalize() + pd.Timedelta(hours=17)


def realistic_trim_auto(trades, tp_col):
    """
    For Auto mode: for each SMT (level_id + smt_birth_ts), the FIRST MSS wins.
    If that asset's entry is skipped, fall through to the OTHER asset's row.
    Then prevent overlapping trades (sequential entry).
    """
    if len(trades) == 0:
        return trades
    t = trades.copy()
    # Auto-pick: first MSS per SMT, fallback if entry skipped
    picks = []
    for (level_id, smt_birth_ts), grp in t.groupby(["level_id", "smt_birth_ts"], sort=False):
        grp = grp.sort_values("mss_ts")
        chosen = None
        for _, r in grp.iterrows():
            if not r.get("entry_skipped", False):
                chosen = r
                break
        if chosen is None and len(grp) > 0:
            chosen = grp.iloc[0]
        if chosen is not None:
            picks.append(chosen)
    if not picks:
        return t.iloc[0:0]
    t = pd.DataFrame(picks).reset_index(drop=True)
    # Sequential entry: prevent overlapping trades
    t = t.sort_values("mss_ts").reset_index(drop=True)
    keep = []
    last_exit = pd.Timestamp.min.tz_localize("America/New_York")
    for _, r in t.iterrows():
        entry = r["mss_ts"]
        if pd.isna(entry):
            keep.append(False); continue
        if entry < last_exit:
            keep.append(False); continue
        # Determine this trade's exit time
        if pd.notna(r.get("managed_exit_ts")):
            exit_ts = r["managed_exit_ts"]
        else:
            exit_ts = session_end_for(entry)
        keep.append(True)
        last_exit = exit_ts
    t["_keep"] = keep
    return t[t["_keep"]].drop(columns=["_keep"])


def simulate_retracement(trades, retrace_r, tp_mult, tp_col, bars_1m, strategy_tf):
    """Apply retracement simulation. Same realism rules as dashboard.py / voligle_A.py.

    REALISM RULES:
      - Search starts at mss_ts + TF_minutes (= MSS bar close time)
      - Same-bar retracement-fill + SL collision: trade FILLS at retracement,
        immediately stops out at -1R
      - Post-fill same-bar TP+SL: SL wins (SL checked first in inner loop)
    """
    TF_MIN = {"1m": 1, "5m": 5, "15m": 15}
    tf_minutes = TF_MIN.get(strategy_tf, 1)

    if retrace_r == 0:
        out = trades.copy()
        out["outcome_r"] = out[tp_col]
        out["fill_ts"] = out["mss_ts"]
        return out

    rows = []
    for _, t in trades.iterrows():
        asset = t["measured_asset"]
        direction = t["direction"]
        orig_entry = float(t["entry_price"])
        sl = float(t["sl_price"])
        orig_r = abs(orig_entry - sl)
        if orig_r < TICK_SIZE - EPS: continue
        mss_ts = t["mss_ts"]
        if pd.isna(mss_ts): continue
        sess_end = session_end_for(mss_ts)

        # Preserve Phase 4 kill timestamp
        kill_ts = None
        mgd_reason = t.get("managed_exit_reason", "")
        mgd_exit_ts = t.get("managed_exit_ts")
        if mgd_reason in ("opp_smt_kill", "opp_mss_kill", "smt_invalidation_kill") and pd.notna(mgd_exit_ts):
            kill_ts = mgd_exit_ts

        if direction == "bullish":
            retrace_price = orig_entry - retrace_r * orig_r
        else:
            retrace_price = orig_entry + retrace_r * orig_r

        # FIX 1: Search starts at MSS bar close
        search_start = mss_ts + pd.Timedelta(minutes=tf_minutes)
        try:
            sub = bars_1m[asset].loc[search_start:sess_end]
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
                # FIX 2: Same-bar fill + SL → fill then immediate stop-out
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
            new_row["outcome_r"] = -1.0
            new_row["fill_ts"] = fill_ts
            rows.append(new_row)
            continue

        if direction == "bullish":
            new_tp = new_entry + tp_mult * new_r
        else:
            new_tp = new_entry - tp_mult * new_r

        try:
            sub2 = bars_1m[asset].loc[fill_ts + pd.Timedelta(minutes=1):sess_end]
        except (KeyError, TypeError):
            sub2 = bars_1m[asset].iloc[0:0]

        outcome_r = None
        for ts, bar in sub2.iterrows():
            if kill_ts is not None and ts >= kill_ts:
                if direction == "bullish":
                    outcome_r = (bar["close"] - new_entry) / new_r
                else:
                    outcome_r = (new_entry - bar["close"]) / new_r
                break
            if direction == "bullish":
                hit_tp = bar["high"] >= new_tp - EPS
                hit_sl = bar["low"] <= sl + EPS
            else:
                hit_tp = bar["low"] <= new_tp + EPS
                hit_sl = bar["high"] >= sl - EPS
            # FIX 3: SL checked first → same-bar TP+SL means SL wins
            if hit_sl:
                outcome_r = -1.0; break
            if hit_tp:
                outcome_r = float(tp_mult); break

        if outcome_r is None:
            if len(sub2) > 0:
                last_close = sub2.iloc[-1]["close"]
                if direction == "bullish":
                    outcome_r = (last_close - new_entry) / new_r
                else:
                    outcome_r = (new_entry - last_close) / new_r
            else:
                outcome_r = 0.0

        new_row = t.to_dict()
        new_row["outcome_r"] = round(outcome_r, 3)
        new_row["fill_ts"] = fill_ts
        rows.append(new_row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ============================================================
# METRICS FOR A SINGLE STRATEGY
# ============================================================
def compute_metrics(trades_df, tp_mult, label):
    """Compute the full metrics dict for one strategy."""
    if len(trades_df) == 0:
        return None

    df = trades_df.sort_values("fill_ts").reset_index(drop=True).copy()
    df["pnl"] = df["outcome_r"] * RISK_PER_TRADE
    df["cum"] = df["pnl"].cumsum()
    df["balance"] = STARTING_BAL + df["cum"]
    df["peak"] = df["cum"].cummax()
    df["dd"] = df["cum"] - df["peak"]

    n = len(df)
    wins = int((df["outcome_r"] > 0).sum())
    losses = int((df["outcome_r"] <= 0).sum())
    wr = wins / n * 100 if n else 0
    tp_hits = int((df["outcome_r"] >= tp_mult - 0.001).sum())
    tp_hit_rate = tp_hits / n * 100 if n else 0
    total_r = df["outcome_r"].sum()
    mean_r = df["outcome_r"].mean()
    total_pnl = df["pnl"].sum()
    pct_return = total_pnl / STARTING_BAL * 100

    # Profit factor
    gross_win = df[df["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(df[df["pnl"] <= 0]["pnl"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy ($ per trade)
    expectancy = total_pnl / n if n else 0

    # Max DD
    max_dd_usd = float(df["dd"].min())
    max_dd_pct_of_account = abs(max_dd_usd) / STARTING_BAL * 100

    # Longest losing streak
    longest_loss_streak = 0
    cur_streak = 0
    longest_loss_streak_usd = 0
    cur_streak_usd = 0
    for r in df["outcome_r"]:
        if r <= 0:
            cur_streak += 1
            cur_streak_usd += r * RISK_PER_TRADE
            if cur_streak > longest_loss_streak:
                longest_loss_streak = cur_streak
                longest_loss_streak_usd = cur_streak_usd
        else:
            cur_streak = 0
            cur_streak_usd = 0

    # Worst day
    df["day"] = df["fill_ts"].dt.date
    daily_pnl = df.groupby("day")["pnl"].sum()
    worst_day = float(daily_pnl.min()) if len(daily_pnl) else 0
    best_day = float(daily_pnl.max()) if len(daily_pnl) else 0
    n_trading_days = len(daily_pnl)
    n_winning_days = int((daily_pnl > 0).sum())

    # Year split
    df["year"] = df["fill_ts"].dt.year
    pnl_2023 = float(df[df["year"] == 2023]["pnl"].sum())
    pnl_2024 = float(df[df["year"] == 2024]["pnl"].sum())
    pnl_2025 = float(df[df["year"] == 2025]["pnl"].sum())
    n_2023 = int((df["year"] == 2023).sum())
    n_2024 = int((df["year"] == 2024).sum())
    n_2025 = int((df["year"] == 2025).sum())

    # Train (2023) / Test (2024-25) WR
    train = df[df["year"] == 2023]
    test = df[df["year"].isin([2024, 2025])]
    train_wr = (train["outcome_r"] > 0).mean() * 100 if len(train) else 0
    test_wr = (test["outcome_r"] > 0).mean() * 100 if len(test) else 0
    overfit_gap = train_wr - test_wr  # positive = overfit signal

    # Sharpe / Sortino on DAILY returns
    daily_returns = daily_pnl / STARTING_BAL
    if len(daily_returns) >= 2:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        neg = daily_returns[daily_returns < 0]
        downside = neg.std() if len(neg) >= 2 else 0
        sortino = (daily_returns.mean() / downside) * np.sqrt(252) if downside > 0 else 0
    else:
        sharpe = 0
        sortino = 0

    # Composite score: blend the things that matter
    # We want: high PnL, low DD, stable across years, low overfit gap, good PF
    # Normalize each component to 0-1 then weight.
    pnl_score = min(max(total_pnl / 50000, 0), 1)            # cap at $50k = full score
    dd_score  = min(max(1 - abs(max_dd_usd) / 5000, 0), 1)   # max acceptable DD = $5k
    pf_score  = min(max((pf - 1) / 3, 0), 1) if pf != float("inf") else 1.0  # PF 1 = 0, PF 4+ = 1
    wr_score  = min(max((wr - 40) / 30, 0), 1)               # 40% = 0, 70% = 1
    consistency_score = 0
    if pnl_2023 > 0 and pnl_2024 > 0 and pnl_2025 > 0:
        consistency_score = 1.0
    elif sum(p > 0 for p in [pnl_2023, pnl_2024, pnl_2025]) == 2:
        consistency_score = 0.5
    overfit_score = min(max(1 - abs(overfit_gap) / 20, 0), 1)  # 0% gap = 1.0, 20%+ gap = 0
    sample_score = min(n / 200, 1)  # need at least 200 trades for reliable stats

    composite = (
        0.25 * pnl_score
        + 0.20 * dd_score
        + 0.15 * pf_score
        + 0.10 * wr_score
        + 0.15 * consistency_score
        + 0.10 * overfit_score
        + 0.05 * sample_score
    ) * 100

    return {
        "label": label,
        "trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wr, 2),
        "tp_hit_rate_pct": round(tp_hit_rate, 2),
        "total_pnl_usd": round(total_pnl, 2),
        "pct_return": round(pct_return, 2),
        "mean_r": round(mean_r, 3),
        "total_r": round(total_r, 2),
        "expectancy_usd": round(expectancy, 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else 99.99,
        "max_dd_usd": round(max_dd_usd, 2),
        "max_dd_pct": round(max_dd_pct_of_account, 2),
        "longest_loss_streak": longest_loss_streak,
        "longest_loss_streak_usd": round(longest_loss_streak_usd, 2),
        "worst_day_usd": round(worst_day, 2),
        "best_day_usd": round(best_day, 2),
        "trading_days": n_trading_days,
        "winning_days": n_winning_days,
        "pct_winning_days": round(n_winning_days / n_trading_days * 100, 2) if n_trading_days else 0,
        "pnl_2023_usd": round(pnl_2023, 2),
        "pnl_2024_usd": round(pnl_2024, 2),
        "pnl_2025_usd": round(pnl_2025, 2),
        "n_trades_2023": n_2023,
        "n_trades_2024": n_2024,
        "n_trades_2025": n_2025,
        "train_wr_pct": round(train_wr, 2),
        "test_wr_pct": round(test_wr, 2),
        "overfit_gap_pct": round(overfit_gap, 2),
        "sharpe_daily": round(sharpe, 2),
        "sortino_daily": round(sortino, 2),
        "composite_score": round(composite, 2),
    }


# ============================================================
# MAIN SWEEP
# ============================================================
def main():
    t_start = time.time()

    meas, ds_name = load_measurements()
    print(f"  Loaded {len(meas):,} rows")

    print("\nLoading 1m bars (for retracement)...")
    bars_1m = load_1m_bars()
    print(f"  MNQ: {len(bars_1m['MNQ']):,} bars")
    print(f"  MES: {len(bars_1m['MES']):,} bars")

    # Apply locked filters once
    print("\nApplying locked filters (sessions, levels, managed)...")
    meas = meas[meas["session"].isin(LOCKED_SESSIONS)]
    meas = meas[~meas["level_type"].isin(EXCLUDE_LEVELS)]
    print(f"  After locked filters: {len(meas):,} rows")

    combos = list(itertools.product(TFS, ANCHORS, RULES, TPS, RETRACES))
    print(f"\nTotal combinations: {len(combos)}")
    print(f"Estimated time: 15-60 minutes\n")

    results = []
    for i, (tf, anchor, rule, tp, retrace) in enumerate(combos, 1):
        label = f"{tf}_{anchor}_{rule}_{tp}_retr{retrace}"
        t0 = time.time()

        # Filter to this strategy
        sub = meas[
            (meas["mss_timeframe"] == tf)
            & (meas["anchor"] == anchor)
            & (meas["mss_rule"] == rule)
            & (meas["mss_found"] == True)
            & (meas["final_outcome"].isin(["sl_hit", "session_end"]))
        ].copy()

        if len(sub) == 0:
            print(f"  [{i:3}/{len(combos)}] {label}: SKIP (no rows)")
            continue

        # Managed-mode entry filter (must remove entry_skipped)
        sub = sub[sub.get("entry_skipped", pd.Series(False, index=sub.index)).fillna(False) == False]
        if len(sub) == 0:
            print(f"  [{i:3}/{len(combos)}] {label}: SKIP (all skipped at entry)")
            continue

        # Pick TP column (managed)
        tp_raw = TP_TO_RAW[tp]
        tp_col = tp_raw + "_mgd"
        if tp_col not in sub.columns:
            tp_col = tp_raw
        sub = sub.dropna(subset=[tp_col])
        if len(sub) == 0:
            continue

        # Auto-pick + sequential trim
        trimmed = realistic_trim_auto(sub, tp_col)
        if len(trimmed) == 0:
            print(f"  [{i:3}/{len(combos)}] {label}: SKIP (no trades after trim)")
            continue

        # Retracement simulation
        tp_mult = TP_TO_MULT[tp]
        sim = simulate_retracement(trimmed, retrace, tp_mult, tp_col, bars_1m, tf)
        if len(sim) == 0:
            print(f"  [{i:3}/{len(combos)}] {label}: SKIP (no trades filled)")
            continue

        # Metrics
        m = compute_metrics(sim, tp_mult, label)
        if m is None:
            continue
        m["tf"] = tf
        m["anchor"] = anchor
        m["rule"] = rule
        m["tp"] = tp
        m["retrace_r"] = retrace
        m["dataset"] = ds_name
        results.append(m)

        elapsed = time.time() - t0
        print(f"  [{i:3}/{len(combos)}] {label}: "
              f"{m['trades']:4} trd · {m['win_rate_pct']:5.2f}% WR · "
              f"${m['total_pnl_usd']:>8,.0f} · DD ${abs(m['max_dd_usd']):>6,.0f} · "
              f"score {m['composite_score']:5.2f}  ({elapsed:.1f}s)")

    if not results:
        print("\nNo valid combinations.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    # Reorder columns for readability
    cols_front = ["tf", "anchor", "rule", "tp", "retrace_r",
                  "composite_score", "trades", "win_rate_pct", "tp_hit_rate_pct",
                  "total_pnl_usd", "max_dd_usd", "max_dd_pct",
                  "profit_factor", "expectancy_usd", "mean_r",
                  "longest_loss_streak", "longest_loss_streak_usd", "worst_day_usd",
                  "pnl_2023_usd", "pnl_2024_usd", "pnl_2025_usd",
                  "train_wr_pct", "test_wr_pct", "overfit_gap_pct",
                  "sharpe_daily", "sortino_daily",
                  "trading_days", "winning_days", "pct_winning_days",
                  "n_trades_2023", "n_trades_2024", "n_trades_2025",
                  "wins", "losses", "best_day_usd", "label", "dataset"]
    df = df[[c for c in cols_front if c in df.columns]]

    out_path = "sweep_results.csv"
    df.to_csv(out_path, index=False)

    elapsed = time.time() - t_start
    print(f"\n{'='*100}")
    print(f"DONE in {elapsed/60:.1f} minutes. Wrote {out_path} ({len(df)} rows).")
    print('='*100)

    # ---- Print top 10 ----
    print("\nTOP 10 BY COMPOSITE SCORE:")
    print('-'*100)
    top = df.head(10)
    for i, r in top.iterrows():
        print(f"\n  #{i+1}: {r['tf']} / {r['anchor']} / {r['rule']} / {r['tp']} / retr {r['retrace_r']:.2f}R")
        print(f"      Trades: {r['trades']}  ·  WR: {r['win_rate_pct']:.2f}%  ·  TP-hit: {r['tp_hit_rate_pct']:.2f}%")
        print(f"      PnL: ${r['total_pnl_usd']:,.2f}  ·  Max DD: ${r['max_dd_usd']:,.2f} ({r['max_dd_pct']:.2f}%)")
        print(f"      PF: {r['profit_factor']:.2f}  ·  Exp: ${r['expectancy_usd']:.2f}/trade  ·  Mean R: {r['mean_r']:.3f}")
        print(f"      Worst day: ${r['worst_day_usd']:,.2f}  ·  Longest loss streak: {r['longest_loss_streak']} (${r['longest_loss_streak_usd']:,.2f})")
        print(f"      Year split: 2023 ${r['pnl_2023_usd']:,.0f}  |  2024 ${r['pnl_2024_usd']:,.0f}  |  2025 ${r['pnl_2025_usd']:,.0f}")
        print(f"      Train (2023) WR: {r['train_wr_pct']:.2f}%  ·  Test (2024-25) WR: {r['test_wr_pct']:.2f}%  ·  Gap: {r['overfit_gap_pct']:+.2f}%")
        print(f"      Sharpe: {r['sharpe_daily']:.2f}  ·  Sortino: {r['sortino_daily']:.2f}  ·  Composite: {r['composite_score']:.2f}")

    # ---- Bonus rankings ----
    print(f"\n{'='*100}")
    print("BONUS RANKINGS")
    print('='*100)

    print("\nTop 5 by TOTAL PnL:")
    for _, r in df.nlargest(5, "total_pnl_usd").iterrows():
        print(f"  {r['tf']:>3} / {r['anchor']:>10} / {r['rule']:>11} / {r['tp']} / r{r['retrace_r']:.2f} : "
              f"${r['total_pnl_usd']:>9,.0f}  DD ${r['max_dd_usd']:>7,.0f}  PF {r['profit_factor']:.2f}")

    print("\nTop 5 by SMALLEST DRAWDOWN (with positive PnL only):")
    pos = df[df["total_pnl_usd"] > 0]
    for _, r in pos.nsmallest(5, "max_dd_usd").iterrows():
        print(f"  {r['tf']:>3} / {r['anchor']:>10} / {r['rule']:>11} / {r['tp']} / r{r['retrace_r']:.2f} : "
              f"DD ${r['max_dd_usd']:>7,.0f}  PnL ${r['total_pnl_usd']:>8,.0f}  WR {r['win_rate_pct']:.1f}%")

    print("\nTop 5 by PROFIT FACTOR (≥100 trades):")
    big = df[df["trades"] >= 100]
    for _, r in big.nlargest(5, "profit_factor").iterrows():
        print(f"  {r['tf']:>3} / {r['anchor']:>10} / {r['rule']:>11} / {r['tp']} / r{r['retrace_r']:.2f} : "
              f"PF {r['profit_factor']:.2f}  PnL ${r['total_pnl_usd']:>8,.0f}  Trd {r['trades']}")

    print("\nMost CONSISTENT (positive in all 3 years, sorted by smallest year-to-year variance):")
    cons = df[(df["pnl_2023_usd"] > 0) & (df["pnl_2024_usd"] > 0) & (df["pnl_2025_usd"] > 0)].copy()
    cons["variance"] = cons[["pnl_2023_usd","pnl_2024_usd","pnl_2025_usd"]].std(axis=1)
    for _, r in cons.nsmallest(5, "variance").iterrows():
        print(f"  {r['tf']:>3} / {r['anchor']:>10} / {r['rule']:>11} / {r['tp']} / r{r['retrace_r']:.2f} : "
              f"2023 ${r['pnl_2023_usd']:>6,.0f} | 2024 ${r['pnl_2024_usd']:>6,.0f} | 2025 ${r['pnl_2025_usd']:>6,.0f}  "
              f"Total ${r['total_pnl_usd']:,.0f}")

    print("\nLEAST OVERFIT (smallest train/test WR gap, with positive PnL):")
    pos["overfit_abs"] = pos["overfit_gap_pct"].abs()
    for _, r in pos.nsmallest(5, "overfit_abs").iterrows():
        print(f"  {r['tf']:>3} / {r['anchor']:>10} / {r['rule']:>11} / {r['tp']} / r{r['retrace_r']:.2f} : "
              f"train {r['train_wr_pct']:.2f}% test {r['test_wr_pct']:.2f}%  gap {r['overfit_gap_pct']:+.2f}%  "
              f"PnL ${r['total_pnl_usd']:,.0f}")

    print(f"\n→ Full results in {out_path}")
    print("→ Open the CSV to sort/filter by any column you care about.")


if __name__ == "__main__":
    main()