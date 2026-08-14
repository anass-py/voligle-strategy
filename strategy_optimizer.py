"""
strategy_optimizer.py

Brute-force + intelligent ranking optimizer for Voligle SMT/MSS strategies.

Locked rules from request:
- Exclude AsiaONHigh/AsiaONLow and PWH/PWL
- Remove ny_pm session
- Always auto_first / auto_picked only

What it searches:
- MSS timeframe: values found in measurements, or --timeframes
- Anchor: values found in measurements, or --anchors
- MSS rule: values found in measurements, or --rules
- Session combinations from london, ny_am, outside
- Level-type combinations from allowed levels
- TP targets: TP1..TP5 by default
- Entry retracements: 0R, 0.25R, 0.50R, 0.75R by default

Outputs in out/optimizer_results by default:
- optimizer_full_results.csv
- optimizer_top_goats.csv
- optimizer_prop_firm_best.csv
- optimizer_recommendation.md

Run examples:
  python strategy_optimizer.py
  python strategy_optimizer.py --retrace-values 0,0.25,0.5,0.75 --level-combo-mode smart
  python strategy_optimizer.py --level-combo-mode all --min-trades 80
  python strategy_optimizer.py --no-retrace-simulation     # very fast, uses existing phase4 R columns only
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TICK_SIZE = 0.25
EPS = 1e-9
TZ = "America/New_York"

EXCLUDE_LEVELS = {"AsiaONHigh", "AsiaONLow", "PWH", "PWL"}
EXCLUDE_SESSIONS = {"ny_pm"}
DEFAULT_RETRACE_VALUES = [0.0, 0.25, 0.50, 0.75]
DEFAULT_TP_LEVELS = [1, 2, 3, 4, 5]

HIGH_LEVELS = {"PDH", "PWH", "LondonHigh", "AsiaKZHigh", "AsiaONHigh", "4H_SwingHigh"}
LOW_LEVELS = {"PDL", "PWL", "LondonLow", "AsiaKZLow", "AsiaONLow", "4H_SwingLow"}
ALL_LEVELS = HIGH_LEVELS | LOW_LEVELS
ALLOWED_LEVELS = sorted(ALL_LEVELS - EXCLUDE_LEVELS)


@dataclass(frozen=True)
class PropRules:
    starting_balance: float = 25_000.0
    daily_dd_pct: float = 0.05
    total_dd_pct: float = 0.10
    safety_buffer: float = 0.75

    @property
    def daily_limit_usd(self) -> float:
        return self.starting_balance * self.daily_dd_pct

    @property
    def total_limit_usd(self) -> float:
        return self.starting_balance * self.total_dd_pct


def parse_csv_list(value: str, cast=str):
    if value is None or str(value).strip() == "":
        return []
    return [cast(x.strip()) for x in str(value).split(",") if x.strip() != ""]


def to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def project_root() -> Path:
    return Path(__file__).resolve().parent


def latest_measurements(out_dir: Path) -> Path:
    files = sorted(out_dir.glob("SMT_measurements_*.csv"))
    if not files:
        raise FileNotFoundError(f"No SMT_measurements_*.csv found in {out_dir}")
    return files[-1]


def load_measurements(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_cols = [
        "smt_birth_ts", "smt_death_ts", "swing_ts", "mss_ts", "sl_leg_extreme_ts",
        "hit_1r_ts", "hit_2r_ts", "hit_3r_ts", "hit_4r_ts", "hit_5r_ts",
        "sl_hit_ts", "trade_exit_ts_if_held", "managed_exit_ts",
    ]
    for c in ts_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_convert(TZ)
    if "session_day" in df.columns:
        df["session_day"] = pd.to_datetime(df["session_day"], errors="coerce")
    return df


def load_bars(out_dir: Path, asset: str, tf: str = "1m") -> pd.DataFrame:
    path = out_dir / f"{asset}_{tf}.csv"
    df = pd.read_csv(path)
    df["ts_ny"] = pd.to_datetime(df["ts_ny"], utc=True, errors="coerce").dt.tz_convert(TZ)
    return df.set_index("ts_ny").sort_index()[["open", "high", "low", "close"]]


def session_end_of(ts: pd.Timestamp) -> pd.Timestamp:
    """Return 17:00 NY of the trading session day for a timestamp."""
    if ts.hour >= 18:
        return ts.normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=17)
    return ts.normalize() + pd.Timedelta(hours=17)


def base_locked_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the user's locked constraints before optimization."""
    mask = (
        to_bool(df["auto_picked"])
        & to_bool(df["mss_found"])
        & df["final_outcome"].isin(["sl_hit", "session_end"])
        & ~df["session"].isin(EXCLUDE_SESSIONS)
        & ~df["level_type"].isin(EXCLUDE_LEVELS)
        & df["entry_price"].notna()
        & df["sl_price"].notna()
        & df["mss_ts"].notna()
    )
    return df.loc[mask].copy()


def exit_timestamp(row: pd.Series, tp_level: int, managed: bool) -> pd.Timestamp:
    if managed and pd.notna(row.get("managed_exit_ts")):
        return row["managed_exit_ts"]
    sl_ts = row.get("sl_hit_ts")
    tp_ts = row.get(f"hit_{tp_level}r_ts")
    end_ts = row.get("trade_exit_ts_if_held")
    candidates = [x for x in [tp_ts, sl_ts, end_ts] if pd.notna(x)]
    return min(candidates) if candidates else row["mss_ts"]


def realistic_trim_single_position(trades: pd.DataFrame, tp_level: int, managed: bool, entry_col="mss_ts") -> pd.DataFrame:
    """Keep one open position at a time across both assets."""
    if trades.empty:
        return trades
    t = trades.copy()
    t["_exit_ts"] = [exit_timestamp(r, tp_level, managed) for _, r in t.iterrows()]
    t = t.sort_values(entry_col).reset_index(drop=True)
    keep = []
    last_exit = None
    for _, row in t.iterrows():
        entry = row[entry_col]
        if last_exit is not None and entry < last_exit:
            keep.append(False)
            continue
        keep.append(True)
        last_exit = row["_exit_ts"]
    return t.loc[keep].drop(columns=["_exit_ts"])


def simulate_retraced_entries(
    trades_df: pd.DataFrame,
    retrace_r: float,
    tp_level: int,
    bars_1m: dict[str, pd.DataFrame],
    use_managed_kills: bool = True,
) -> pd.DataFrame:
    """
    Simulate waiting for price to retrace X*original_R before entry.
    SL remains the original SL. TP is recomputed from the new entry/new R.
    If price never retraces before SL/session/kill, the trade is skipped.
    """
    if retrace_r == 0:
        col = f"r_tp{tp_level}_mgd" if f"r_tp{tp_level}_mgd" in trades_df.columns else f"r_tp{tp_level}"
        out = trades_df.dropna(subset=[col]).copy()
        out["entry_ts_sim"] = out["mss_ts"]
        out["exit_ts_sim"] = [exit_timestamp(r, tp_level, col.endswith("_mgd")) for _, r in out.iterrows()]
        out["outcome_r"] = out[col].astype(float)
        out["outcome_reason"] = out.get("managed_exit_reason", pd.Series(index=out.index, dtype=object)).fillna("phase4")
        out["filled_retrace"] = True
        return out

    rows_out = []
    for _, t in trades_df.iterrows():
        asset = t["measured_asset"]
        if asset not in bars_1m:
            continue
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

        kill_ts = None
        kill_reason = None
        if use_managed_kills:
            reason = t.get("managed_exit_reason", "")
            exit_ts = t.get("managed_exit_ts")
            if reason in ("opp_smt_kill", "opp_mss_kill") and pd.notna(exit_ts):
                kill_ts = exit_ts
                kill_reason = reason

        retrace_price = orig_entry - retrace_r * orig_r if direction == "bullish" else orig_entry + retrace_r * orig_r
        bars = bars_1m[asset]
        sub = bars.loc[mss_ts + pd.Timedelta(minutes=1):sess_end]
        if sub.empty:
            continue

        fill_ts = None
        for ts, bar in sub.iterrows():
            if kill_ts is not None and ts >= kill_ts:
                fill_ts = None
                break
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

        new_entry = retrace_price
        new_r = abs(new_entry - sl)
        if new_r < TICK_SIZE - EPS:
            continue
        tp_price = new_entry + tp_level * new_r if direction == "bullish" else new_entry - tp_level * new_r

        sub2 = bars.loc[fill_ts + pd.Timedelta(minutes=1):sess_end]
        outcome_r = None
        outcome_reason = None
        outcome_exit_ts = None

        for ts, bar in sub2.iterrows():
            # Kill checked first because phase4 managed exits are hard exits at that timestamp.
            if kill_ts is not None and ts >= kill_ts:
                outcome_r = (bar["close"] - new_entry) / new_r if direction == "bullish" else (new_entry - bar["close"]) / new_r
                outcome_reason = kill_reason or "kill"
                outcome_exit_ts = ts
                break
            if direction == "bullish":
                hit_tp = bar["high"] >= tp_price - EPS
                hit_sl = bar["low"] <= sl + EPS
            else:
                hit_tp = bar["low"] <= tp_price + EPS
                hit_sl = bar["high"] >= sl - EPS
            if hit_tp and hit_sl:
                outcome_r, outcome_reason, outcome_exit_ts = -1.0, "sl_hit_same_bar_conservative", ts
                break
            if hit_sl:
                outcome_r, outcome_reason, outcome_exit_ts = -1.0, "sl_hit", ts
                break
            if hit_tp:
                outcome_r, outcome_reason, outcome_exit_ts = float(tp_level), f"tp{tp_level}_hit", ts
                break

        if outcome_r is None:
            if not sub2.empty:
                last_close = sub2.iloc[-1]["close"]
                outcome_r = (last_close - new_entry) / new_r if direction == "bullish" else (new_entry - last_close) / new_r
                outcome_exit_ts = sub2.index[-1]
            else:
                outcome_r = 0.0
                outcome_exit_ts = fill_ts
            outcome_reason = "session_end"

        row = t.to_dict()
        row["entry_price_sim"] = new_entry
        row["r_points_sim"] = new_r
        row["entry_ts_sim"] = fill_ts
        row["exit_ts_sim"] = outcome_exit_ts
        row["outcome_r"] = round(float(outcome_r), 4)
        row["outcome_reason"] = outcome_reason
        row["filled_retrace"] = True
        rows_out.append(row)

    return pd.DataFrame(rows_out)


def non_empty_combinations(items: Sequence[str], max_size: int | None = None) -> list[tuple[str, ...]]:
    max_size = len(items) if max_size in (None, 0) else min(max_size, len(items))
    out = []
    for r in range(1, max_size + 1):
        out.extend(itertools.combinations(items, r))
    return out


def level_combinations(levels: list[str], mode: str, max_size: int | None = None) -> list[tuple[str, ...]]:
    if mode == "all":
        return non_empty_combinations(levels, max_size=max_size)
    if mode == "single":
        return [(x,) for x in levels]
    # smart: useful clusters + singles + all, much faster than all 255 subsets.
    clusters = set()
    clusters.add(tuple(sorted(levels)))
    clusters.add(tuple(sorted([x for x in levels if x in HIGH_LEVELS])))
    clusters.add(tuple(sorted([x for x in levels if x in LOW_LEVELS])))
    for x in levels:
        clusters.add((x,))
    paired_names = [
        ("PDH", "PDL"),
        ("LondonHigh", "LondonLow"),
        ("AsiaKZHigh", "AsiaKZLow"),
        ("4H_SwingHigh", "4H_SwingLow"),
    ]
    for pair in paired_names:
        p = tuple(sorted([x for x in pair if x in levels]))
        if p:
            clusters.add(p)
    return sorted(clusters, key=lambda x: (len(x), x))


def max_drawdown_r(r_values: np.ndarray) -> float:
    if len(r_values) == 0:
        return 0.0
    equity = np.cumsum(r_values)
    peak = np.maximum.accumulate(np.r_[0.0, equity])[:-1]
    dd = equity - peak
    return abs(float(dd.min())) if len(dd) else 0.0


def max_consecutive_losses(r_values: np.ndarray) -> int:
    best = cur = 0
    for r in r_values:
        if r < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def calc_metrics(trades: pd.DataFrame, prop: PropRules, risk_per_trade: float) -> dict:
    r = trades["outcome_r"].astype(float).to_numpy()
    n = len(r)
    wins = r[r > 0]
    losses = r[r < 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else (np.inf if gross_win > 0 else 0.0)
    expectancy = float(np.mean(r)) if n else 0.0
    win_rate = float((r > 0).mean() * 100) if n else 0.0
    total_r = float(r.sum()) if n else 0.0
    dd_r = max_drawdown_r(r)
    calmar_r = total_r / dd_r if dd_r > 0 else np.inf

    dates = pd.to_datetime(trades["entry_ts_sim"]).dt.date if "entry_ts_sim" in trades.columns else pd.to_datetime(trades["mss_ts"]).dt.date
    daily_r = pd.Series(r).groupby(dates).sum()
    worst_day_r = abs(float(daily_r.min())) if len(daily_r) else 0.0
    avg_trades_per_day = float(pd.Series(r).groupby(dates).size().mean()) if len(daily_r) else 0.0

    max_daily_loss_usd = worst_day_r * risk_per_trade
    max_dd_usd = dd_r * risk_per_trade
    prop_daily_ok = max_daily_loss_usd <= prop.daily_limit_usd + EPS
    prop_total_ok = max_dd_usd <= prop.total_limit_usd + EPS
    prop_ok = bool(prop_daily_ok and prop_total_ok and total_r > 0)

    risk_by_daily = prop.daily_limit_usd / worst_day_r if worst_day_r > EPS else np.inf
    risk_by_total = prop.total_limit_usd / dd_r if dd_r > EPS else np.inf
    risk_cap = min(risk_by_daily, risk_by_total)
    recommended_risk = risk_cap * prop.safety_buffer if np.isfinite(risk_cap) else risk_per_trade

    # If the strategy is statistically healthy, allow a larger aggressive value; otherwise keep it conservative.
    quality_ok = (n >= 100 and win_rate >= 55 and expectancy > 0 and pf >= 1.25)
    high_wr_ok = (n >= 100 and win_rate >= 65 and expectancy > 0 and pf >= 1.50)
    aggressive_risk = risk_cap * (0.90 if high_wr_ok else 0.75 if quality_ok else 0.50) if np.isfinite(risk_cap) else risk_per_trade

    # Balanced GOAT score: not only total profit. Penalizes drawdown and small samples.
    sample_factor = min(1.0, n / 150.0)
    pf_capped = min(pf, 5.0) if np.isfinite(pf) else 5.0
    goat_score = (
        expectancy * 100.0
        + pf_capped * 12.0
        + win_rate * 0.35
        + sample_factor * 20.0
        + calmar_r * 2.0 if np.isfinite(calmar_r) else 0.0
    )
    goat_score -= dd_r * 2.0
    goat_score -= max_consecutive_losses(r) * 1.5
    if prop_ok:
        goat_score += 15.0
    if high_wr_ok:
        goat_score += 8.0

    return {
        "trades": n,
        "wins": int((r > 0).sum()),
        "losses": int((r < 0).sum()),
        "win_rate_pct": round(win_rate, 2),
        "expectancy_r": round(expectancy, 4),
        "profit_factor": round(float(pf), 3) if np.isfinite(pf) else 999.0,
        "total_r": round(total_r, 2),
        "max_drawdown_r": round(dd_r, 2),
        "worst_day_r": round(worst_day_r, 2),
        "avg_trades_per_day": round(avg_trades_per_day, 2),
        "max_consecutive_losses": max_consecutive_losses(r),
        "calmar_r": round(float(calmar_r), 3) if np.isfinite(calmar_r) else 999.0,
        "fixed_risk_usd": round(risk_per_trade, 2),
        "max_daily_loss_usd_at_fixed_risk": round(max_daily_loss_usd, 2),
        "max_dd_usd_at_fixed_risk": round(max_dd_usd, 2),
        "prop_daily_ok_at_fixed_risk": prop_daily_ok,
        "prop_total_ok_at_fixed_risk": prop_total_ok,
        "prop_ok_at_fixed_risk": prop_ok,
        "recommended_risk_usd": round(max(0.0, recommended_risk), 2),
        "aggressive_risk_usd": round(max(0.0, aggressive_risk), 2),
        "goat_score": round(float(goat_score), 3),
    }


def evaluate_grid(
    base: pd.DataFrame,
    out_dir: Path,
    timeframes: list[str],
    anchors: list[str],
    rules: list[str],
    sessions: list[str],
    levels: list[str],
    tps: list[int],
    retraces: list[float],
    level_mode: str,
    max_level_combo_size: int | None,
    min_trades: int,
    prop: PropRules,
    risk_per_trade: float,
    simulate_retrace: bool,
) -> tuple[pd.DataFrame, dict[tuple[int, float], pd.DataFrame]]:
    session_combos = non_empty_combinations(sessions)
    lvl_combos = level_combinations(levels, level_mode, max_size=max_level_combo_size)

    bars_1m = None
    if simulate_retrace and any(r > 0 for r in retraces):
        print("Loading 1m bars for retracement simulation...")
        bars_1m = {asset: load_bars(out_dir, asset, "1m") for asset in sorted(base["measured_asset"].dropna().unique())}

    simulated_cache: dict[tuple[int, float], pd.DataFrame] = {}
    rows = []
    total_batches = len(tps) * len(retraces)
    batch_no = 0

    for tp in tps:
        for retrace in retraces:
            batch_no += 1
            print(f"[{batch_no}/{total_batches}] Preparing outcomes: TP{tp}, retrace={retrace:.2f}R")
            if retrace > 0 and not simulate_retrace:
                continue
            if retrace > 0:
                sim = simulate_retraced_entries(base, retrace, tp, bars_1m or {})
            else:
                sim = simulate_retraced_entries(base, 0.0, tp, {})
            if sim.empty:
                continue
            sim = sim.dropna(subset=["outcome_r", "entry_ts_sim"])
            simulated_cache[(tp, retrace)] = sim

            for tf in timeframes:
                sim_tf = sim[sim["mss_timeframe"] == tf]
                if len(sim_tf) < min_trades:
                    continue
                for anchor in anchors:
                    sim_anchor = sim_tf[sim_tf["anchor"] == anchor]
                    if len(sim_anchor) < min_trades:
                        continue
                    for rule in rules:
                        sim_rule = sim_anchor[sim_anchor["mss_rule"] == rule]
                        if len(sim_rule) < min_trades:
                            continue
                        for sess_combo in session_combos:
                            sim_sess = sim_rule[sim_rule["session"].isin(sess_combo)]
                            if len(sim_sess) < min_trades:
                                continue
                            for lvl_combo in lvl_combos:
                                trades = sim_sess[sim_sess["level_type"].isin(lvl_combo)]
                                if len(trades) < min_trades:
                                    continue
                                # Realistic one-position-at-a-time after exact strategy filtering.
                                trades = realistic_trim_single_position(trades, tp, managed=(retrace == 0), entry_col="entry_ts_sim")
                                if len(trades) < min_trades:
                                    continue
                                metrics = calc_metrics(trades.sort_values("entry_ts_sim"), prop, risk_per_trade)
                                rows.append({
                                    "mss_timeframe": tf,
                                    "anchor": anchor,
                                    "mss_rule": rule,
                                    "sessions": "+".join(sess_combo),
                                    "level_types": "+".join(lvl_combo),
                                    "level_count": len(lvl_combo),
                                    "tp": tp,
                                    "entry_retrace_r": retrace,
                                    **metrics,
                                })
    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(["goat_score", "expectancy_r", "profit_factor", "max_drawdown_r"], ascending=[False, False, False, True])
    return results, simulated_cache


def write_recommendation(results: pd.DataFrame, out_path: Path, prop: PropRules):
    if results.empty:
        out_path.write_text("# Optimizer Recommendation\n\nNo strategy passed the minimum trade filter.\n")
        return
    top = results.iloc[0]
    prop_best = results[results["prop_ok_at_fixed_risk"] == True].head(1)
    prop_row = prop_best.iloc[0] if len(prop_best) else top

    text = f"""# Optimizer Recommendation

## Overall GOAT strategy

- **Score:** {top['goat_score']}
- **TF:** {top['mss_timeframe']}
- **Anchor:** {top['anchor']}
- **Rule:** {top['mss_rule']}
- **Sessions:** {top['sessions']}
- **Levels:** {top['level_types']}
- **TP:** TP{int(top['tp'])}
- **Entry retracement:** {float(top['entry_retrace_r']):.2f}R
- **Trades:** {int(top['trades'])}
- **Win rate:** {top['win_rate_pct']}%
- **Expectancy:** {top['expectancy_r']}R/trade
- **Profit factor:** {top['profit_factor']}
- **Max drawdown:** {top['max_drawdown_r']}R
- **Worst day:** {top['worst_day_r']}R
- **Recommended risk:** ${top['recommended_risk_usd']}
- **Aggressive risk:** ${top['aggressive_risk_usd']}

## Best prop-firm candidate at fixed risk

- **TF:** {prop_row['mss_timeframe']}
- **Anchor:** {prop_row['anchor']}
- **Rule:** {prop_row['mss_rule']}
- **Sessions:** {prop_row['sessions']}
- **Levels:** {prop_row['level_types']}
- **TP:** TP{int(prop_row['tp'])}
- **Entry retracement:** {float(prop_row['entry_retrace_r']):.2f}R
- **Trades:** {int(prop_row['trades'])}
- **Win rate:** {prop_row['win_rate_pct']}%
- **Expectancy:** {prop_row['expectancy_r']}R/trade
- **Profit factor:** {prop_row['profit_factor']}
- **Max daily loss at fixed risk:** ${prop_row['max_daily_loss_usd_at_fixed_risk']}
- **Max drawdown at fixed risk:** ${prop_row['max_dd_usd_at_fixed_risk']}
- **Prop OK at fixed risk:** {prop_row['prop_ok_at_fixed_risk']}

## Risk logic

The script sizes risk from actual historical damage:

- Daily risk cap = ${prop.daily_limit_usd:,.0f} / worst losing day in R
- Total drawdown cap = ${prop.total_limit_usd:,.0f} / max drawdown in R
- Recommended risk = smaller cap × {prop.safety_buffer:.0%} safety buffer
- Aggressive risk increases only when sample size, win rate, expectancy, and profit factor are all strong.

A high win rate alone is **not** enough. The strategy must also have positive expectancy, enough trades, good profit factor, and acceptable drawdown.
"""
    out_path.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(project_root() / "out"), help="Folder containing SMT_measurements and bar CSVs.")
    ap.add_argument("--measurements", default="", help="Specific SMT_measurements CSV. Default: latest in out-dir.")
    ap.add_argument("--results-dir", default="", help="Output folder. Default: out/optimizer_results.")
    ap.add_argument("--timeframes", default="", help="Comma list, e.g. 5m,15m. Default: all found.")
    ap.add_argument("--anchors", default="", help="Comma list. Default: all found.")
    ap.add_argument("--rules", default="", help="Comma list. Default: all found.")
    ap.add_argument("--sessions", default="london,ny_am,outside", help="Sessions to combine; ny_pm is always removed.")
    ap.add_argument("--tp-levels", default="1,2,3,4,5", help="Comma list of TP levels.")
    ap.add_argument("--retrace-values", default="0,0.25,0.5,0.75", help="Comma list of entry retracements in R.")
    ap.add_argument("--level-combo-mode", choices=["smart", "single", "all"], default="all", help="all = every subset; smart = clusters/singles; single = one level at a time.")
    ap.add_argument("--max-level-combo-size", type=int, default=0, help="0 means no limit. Useful with --level-combo-mode all.")
    ap.add_argument("--min-trades", type=int, default=60)
    ap.add_argument("--risk-per-trade", type=float, default=250.0)
    ap.add_argument("--starting-balance", type=float, default=25_000.0)
    ap.add_argument("--daily-dd-pct", type=float, default=0.05)
    ap.add_argument("--total-dd-pct", type=float, default=0.10)
    ap.add_argument("--safety-buffer", type=float, default=0.75)
    ap.add_argument("--no-retrace-simulation", action="store_true", help="Skip retrace > 0 simulations; very fast.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    measurements = Path(args.measurements).expanduser().resolve() if args.measurements else latest_measurements(out_dir)
    results_dir = Path(args.results_dir).expanduser().resolve() if args.results_dir else out_dir / "optimizer_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading measurements: {measurements}")
    df = load_measurements(measurements)
    base = base_locked_filter(df)
    print(f"Locked-filter candidate trades: {len(base):,}")
    if base.empty:
        raise SystemExit("No rows remain after locked filters.")

    timeframes = parse_csv_list(args.timeframes) or sorted(base["mss_timeframe"].dropna().unique().tolist())
    anchors = parse_csv_list(args.anchors) or sorted(base["anchor"].dropna().unique().tolist())
    rules = parse_csv_list(args.rules) or sorted(base["mss_rule"].dropna().unique().tolist())
    sessions = [s for s in parse_csv_list(args.sessions) if s not in EXCLUDE_SESSIONS]
    tps = parse_csv_list(args.tp_levels, int) or DEFAULT_TP_LEVELS
    retraces = parse_csv_list(args.retrace_values, float) or DEFAULT_RETRACE_VALUES
    if args.no_retrace_simulation:
        retraces = [r for r in retraces if r == 0.0]

    levels = sorted([x for x in base["level_type"].dropna().unique().tolist() if x not in EXCLUDE_LEVELS])
    prop = PropRules(args.starting_balance, args.daily_dd_pct, args.total_dd_pct, args.safety_buffer)

    print("Search space:")
    print(f"  timeframes={timeframes}")
    print(f"  anchors={anchors}")
    print(f"  rules={rules}")
    print(f"  sessions={sessions} combinations={2 ** len(sessions) - 1}")
    print(f"  levels={levels} mode={args.level_combo_mode}")
    print(f"  tps={tps}")
    print(f"  retraces={retraces}")

    results, _ = evaluate_grid(
        base=base,
        out_dir=out_dir,
        timeframes=timeframes,
        anchors=anchors,
        rules=rules,
        sessions=sessions,
        levels=levels,
        tps=tps,
        retraces=retraces,
        level_mode=args.level_combo_mode,
        max_level_combo_size=args.max_level_combo_size,
        min_trades=args.min_trades,
        prop=prop,
        risk_per_trade=args.risk_per_trade,
        simulate_retrace=not args.no_retrace_simulation,
    )

    full_path = results_dir / "optimizer_full_results.csv"
    top_path = results_dir / "optimizer_top_goats.csv"
    prop_path = results_dir / "optimizer_prop_firm_best.csv"
    md_path = results_dir / "optimizer_recommendation.md"

    results.to_csv(full_path, index=False)
    results.head(100).to_csv(top_path, index=False)
    if not results.empty:
        prop_best = results[results["prop_ok_at_fixed_risk"] == True].sort_values(
            ["goat_score", "expectancy_r", "max_drawdown_r"], ascending=[False, False, True]
        )
        prop_best.head(100).to_csv(prop_path, index=False)
    else:
        pd.DataFrame().to_csv(prop_path, index=False)
    write_recommendation(results, md_path, prop)

    print("\nDone.")
    print(f"Full results: {full_path}")
    print(f"Top GOATs:    {top_path}")
    print(f"Prop best:    {prop_path}")
    print(f"Report:       {md_path}")
    if not results.empty:
        print("\nTop 10:")
        cols = [
            "goat_score", "mss_timeframe", "anchor", "mss_rule", "sessions", "level_types",
            "tp", "entry_retrace_r", "trades", "win_rate_pct", "expectancy_r", "profit_factor",
            "max_drawdown_r", "worst_day_r", "recommended_risk_usd", "prop_ok_at_fixed_risk",
        ]
        print(results[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
