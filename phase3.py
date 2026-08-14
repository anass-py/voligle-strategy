"""
Phase 3: Strict SMT detection using paired levels and first-touch filtering.

NEW IN THIS VERSION:
  opposing_live is now computed using a TIMEFRAME HIERARCHY:

  Rank (higher number = higher timeframe):
    4 : PWH / PWL          (weekly)
    3 : PDH / PDL          (daily)
    2 : 4H_SwingHigh / Low (4-hour)
    1 : LondonHigh/Low, AsiaKZ, AsiaON  (session, conceptually 1H)

  opposing_live = True only if an opposing-direction strict SMT is alive
  AT the moment of the current SMT's birth AND its level rank >= current
  SMT's level rank.

  Example: a 4H_SwingHigh (rank 2) bullish SMT with an opposing LondonLow
  (rank 1) bearish SMT alive -> opposing_live = False (session timeframe
  is below 4H).

Usage: python3 phase3.py
"""

import os
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

HIGH_LEVELS = ["PDH", "PWH", "LondonHigh", "AsiaKZHigh", "AsiaONHigh", "4H_SwingHigh"]
LOW_LEVELS  = ["PDL", "PWL", "LondonLow",  "AsiaKZLow",  "AsiaONLow",  "4H_SwingLow"]
ALL_LEVELS  = HIGH_LEVELS + LOW_LEVELS
PERSISTENT_TYPES = {"PDH","PDL","PWH","PWL","4H_SwingHigh","4H_SwingLow"}

# Level-type -> timeframe rank. Higher = higher timeframe.
LEVEL_RANK = {
    "PWH": 4, "PWL": 4,
    "PDH": 3, "PDL": 3,
    "4H_SwingHigh": 2, "4H_SwingLow": 2,
    "LondonHigh": 1, "LondonLow": 1,
    "AsiaKZHigh": 1, "AsiaKZLow": 1,
    "AsiaONHigh": 1, "AsiaONLow": 1,
}

LOAD_START = pd.Timestamp(START_DATE, tz="America/New_York") - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
LOAD_END   = pd.Timestamp(END_DATE,   tz="America/New_York") + pd.Timedelta(hours=17)

SMT_COLS = [
    "level_id", "birth_ts", "death_ts", "status", "lifespan_minutes",
    "level_type", "direction", "sweeper", "non_confirmer",
    "level_price_sweeper", "level_price_nonconf",
    "sweep_price", "nonconf_price_at_birth", "separation_ticks",
    "opposing_live", "opposing_count",
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


def load_1m(asset):
    path = os.path.join(OUT_FOLDER, f"{asset}_1m.csv")
    df = pd.read_csv(path)
    df["ts_ny"] = pd.to_datetime(df["ts_ny"], utc=True).dt.tz_convert("America/New_York")
    return df.set_index("ts_ny").sort_index().loc[LOAD_START:LOAD_END][["open","high","low","close"]]


def load_levels(asset):
    path = os.path.join(OUT_FOLDER, f"{asset}_levels.csv")
    df = pd.read_csv(path)
    df["ts_formed"]  = pd.to_datetime(df["ts_formed"],  utc=True).dt.tz_convert("America/New_York")
    df["valid_from"] = pd.to_datetime(df["valid_from"], utc=True).dt.tz_convert("America/New_York")
    df["valid_until"]= pd.to_datetime(df["valid_until"],utc=True).dt.tz_convert("America/New_York")
    if "first_touch_ts" in df.columns:
        df["first_touch_ts"] = pd.to_datetime(df["first_touch_ts"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    df = df[df["level_type"].isin(ALL_LEVELS)]
    df = df[(df["valid_from"] < LOAD_END) & (df["valid_until"] > LOAD_START)]
    return df.reset_index(drop=True)


def session_start_for(ts):
    if ts.hour >= 18:
        return ts.normalize() + pd.Timedelta(hours=18)
    return ts.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=18)


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


def detect_smts(mnq_1m, mes_1m, mnq_lv, mes_lv):
    smts = []
    mnq_by_id = {r["level_id"]: r for _, r in mnq_lv.iterrows()}
    mes_by_id = {r["level_id"]: r for _, r in mes_lv.iterrows()}
    common_ids = set(mnq_by_id.keys()) & set(mes_by_id.keys())

    for level_id in common_ids:
        mnq_row = mnq_by_id[level_id]
        mes_row = mes_by_id[level_id]
        lt = mnq_row["level_type"]
        is_high = lt in HIGH_LEVELS

        vf = max(mnq_row["valid_from"], mes_row["valid_from"])
        vu = min(mnq_row["valid_until"], mes_row["valid_until"], LOAD_END)
        if vf >= vu:
            continue

        if lt in PERSISTENT_TYPES:
            ft = mnq_row.get("first_touch_ts")
            if pd.notna(ft) and ft < LOAD_START:
                continue

        lp_mnq = mnq_row["price"]
        lp_mes = mes_row["price"]

        mnq_cross = first_strict_cross_ts(mnq_1m, vf, vu, lp_mnq, is_high)
        mes_cross = first_strict_cross_ts(mes_1m, vf, vu, lp_mes, is_high)
        if mnq_cross is None and mes_cross is None:
            continue

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

        if is_sunday_evening(sweep_ts):
            continue
        if sweep_ts not in nonconf_bars.index:
            continue

        nc_bar = nonconf_bars.loc[sweep_ts]
        if is_high:
            nc_extreme = nc_bar["high"]
            if nc_extreme >= lp_nonconf + TICK_SIZE - EPS:
                continue
        else:
            nc_extreme = nc_bar["low"]
            if nc_extreme <= lp_nonconf - TICK_SIZE + EPS:
                continue

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
            "level_id": level_id,
            "birth_ts": sweep_ts, "death_ts": death_ts, "status": death_reason,
            "lifespan_minutes": (death_ts - sweep_ts).total_seconds() / 60.0,
            "level_type": lt, "direction": "bearish" if is_high else "bullish",
            "sweeper": sweeper, "non_confirmer": nonconf,
            "level_price_sweeper": lp_sweeper, "level_price_nonconf": lp_nonconf,
            "sweep_price": sweep_price, "nonconf_price_at_birth": nc_extreme,
            "separation_ticks": sep_ticks,
        })

    df = pd.DataFrame(smts)
    if len(df):
        df = df.sort_values("birth_ts").reset_index(drop=True)
    return df


def flag_opposing_live(df):
    """
    opposing_live = True iff an opposing-direction SMT is alive at the
    current SMT's birth AND its level_type has a rank >= the current
    SMT's level_type rank.

    opposing_count = count of such qualifying opposing SMTs.
    """
    if len(df) == 0:
        df["opposing_live"] = pd.Series(dtype="bool")
        df["opposing_count"] = pd.Series(dtype="int64")
        return df
    df = df.copy().sort_values("birth_ts").reset_index(drop=True)

    births = df["birth_ts"].values
    deaths = df["death_ts"].values
    dirs = df["direction"].values
    ranks = df["level_type"].map(LEVEL_RANK).fillna(0).astype(int).values

    opp_live = []
    opp_count = []
    for i in range(len(df)):
        cur_birth = births[i]
        cur_dir = dirs[i]
        cur_rank = ranks[i]
        # Opposing = different direction, alive at my birth, and rank >= mine
        alive = (births <= cur_birth) & (deaths > cur_birth) & (dirs != cur_dir) & (ranks >= cur_rank)
        alive[i] = False
        c = int(alive.sum())
        opp_count.append(c)
        opp_live.append(c > 0)

    df["opposing_live"] = opp_live
    df["opposing_count"] = opp_count
    return df


def main():
    print(f"Phase 3: {START_DATE} -> {END_DATE}")
    print(f"Window: {LOAD_START} -> {LOAD_END}\n")
    mnq_1m = load_1m("MNQ"); mes_1m = load_1m("MES")
    mnq_lv = load_levels("MNQ"); mes_lv = load_levels("MES")
    print(f"MNQ: {len(mnq_1m):,} bars, {len(mnq_lv):,} levels")
    print(f"MES: {len(mes_1m):,} bars, {len(mes_lv):,} levels")

    smts = detect_smts(mnq_1m, mes_1m, mnq_lv, mes_lv)
    smts = flag_opposing_live(smts)

    print(f"\nTotal strict SMTs: {len(smts):,}")
    if len(smts):
        print("By status:");     print(smts["status"].value_counts().to_string())
        print("\nBy level type:"); print(smts["level_type"].value_counts().to_string())
        print(f"\nopposing_live True (>= rank): {smts['opposing_live'].sum()}")

    out = os.path.join(OUT_FOLDER, f"SMT_events_{START_DATE}_{END_DATE}.csv")
    save_csv(smts, out, SMT_COLS)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()