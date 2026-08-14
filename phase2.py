"""
Phase 2: Level detection with paired cross-asset IDs and first_touch_ts.

FIXES IN THIS VERSION:
  1. DST-AWARE 4H BUCKETING.
     The previous bucketing used `bucket_date + pd.Timedelta(hours=18)` which
     stays at a fixed UTC-offset relative to bucket_date. On days when DST
     starts (spring forward) or ends (fall back), this produces phantom bars
     at 19:00 / 23:00 (or 17:00 / 21:00) instead of 18:00 / 22:00.
     The new logic constructs bucket_start as a timezone-AWARE NY wall-clock
     timestamp using `pd.Timestamp(year, month, day, hour, tz="America/New_York")`
     which correctly handles DST transitions.

  2. DUPLICATE-PAIRING FIX.
     The previous pair_swings_symmetric() allowed a swing on one asset to be
     paired in TWO different rows: once as a shift_paired partner, once as a
     solo_parallel target. The fix is to also check `(other_asset, ts) not in
     paired` before emitting a solo_parallel pair.

ALGORITHM (single-pass chronological):
  1. Collect ALL swing events (from both assets, both High and Low kinds)
     into a time-ordered sequence.
  2. Walk the sequence. For each swing:
     a. If this swing was already paired by a previous event -> skip.
     b. Same-candle check: does the OTHER asset have a same-kind swing at
        the same 4H timestamp AND it isn't already paired? If yes -> same_candle pair.
     c. T-1 lookup: does the OTHER asset have an EARLIER same-kind swing at
        ts-4h that's still unpaired? If yes -> shift_paired (other earlier).
     d. T+1 lookup: does the OTHER asset have a LATER same-kind swing at ts+4h
        that's still unpaired? If yes -> shift_paired (this earlier, other later).
     e. Otherwise -> solo_parallel using the other asset's 4H BAR at this ts,
        ONLY IF that other bar isn't already paired.

HARD EXPIRY:
  PDH/PDL  = 14 days
  PWH/PWL  = 7 days
  4H swings= 90 days
  London/Asia = same session day

Usage: python3 phase2.py
"""

import os
import pandas as pd
import numpy as np

# ============================================================
# USER CONFIG
# ============================================================
DATA_FOLDER = "/Users/voligle/workspace/Trading/Trading_code/clause Backtester"
SWING_STRENGTH = 1
TICK_SIZE = 0.25
EPS = 1e-9
# ============================================================

OUT_FOLDER = os.path.join(DATA_FOLDER, "New/out")
os.makedirs(OUT_FOLDER, exist_ok=True)

HIGH_LEVEL_TYPES = {"PDH","PWH","LondonHigh","AsiaKZHigh","AsiaONHigh","4H_SwingHigh"}
LOW_LEVEL_TYPES  = {"PDL","PWL","LondonLow","AsiaKZLow","AsiaONLow","4H_SwingLow"}
PERSISTENT_TYPES = {"PDH","PDL","PWH","PWL","4H_SwingHigh","4H_SwingLow"}

LEVEL_COLS = [
    "level_id", "ts_formed", "level_type", "price",
    "valid_from", "valid_until", "asset", "pair_kind",
    "first_touch_ts",
]


def save_levels(df, path):
    if len(df) == 0:
        for c in LEVEL_COLS:
            if c not in df.columns:
                df[c] = pd.Series(dtype="object")
        df = df[LEVEL_COLS]
    else:
        for c in LEVEL_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[LEVEL_COLS + [c for c in df.columns if c not in LEVEL_COLS]]
    df.to_csv(path, index=False)


def load_1m(asset):
    path = os.path.join(OUT_FOLDER, f"{asset}_1m.csv")
    df = pd.read_csv(path)
    df["ts_ny"] = pd.to_datetime(df["ts_ny"], utc=True).dt.tz_convert("America/New_York")
    return df.set_index("ts_ny").sort_index()[["open","high","low","close"]]


def build_4h_anchored(df_1m):
    """
    Build 4H bars anchored to the futures session: 18/22/02/06/10/14 NY.
    DST-safe: for each 1m bar, find the LATEST anchored timestamp ≤ bar_ts
    that has hour ∈ {2, 6, 10, 14, 18, 22}. We do this by truncating the
    bar's existing tz-aware timestamp, never constructing new ones from
    scratch (which would be ambiguous on fall-DST days).
    """
    df = df_1m.copy()
    h = df.index.hour

    # Determine bucket NY hour for each 1m bar
    bucket_hour = pd.Series(index=df.index, dtype="int64")
    bucket_hour[(h >= 18) & (h < 22)] = 18
    bucket_hour[(h >= 22) & (h < 24)] = 22
    bucket_hour[(h >= 0) & (h < 2)]   = 22  # belongs to 22:00 of PREVIOUS day
    bucket_hour[(h >= 2) & (h < 6)]   = 2
    bucket_hour[(h >= 6) & (h < 10)]  = 6
    bucket_hour[(h >= 10) & (h < 14)] = 10
    bucket_hour[(h >= 14) & (h < 18)] = 14

    # Hours 0-1 belong to PREVIOUS NY day's 22:00 bucket
    needs_shift = (h >= 0) & (h < 2)

    # For each 1m bar at NY time `ts`, the bucket_start is:
    #   1. ts truncated to its calendar day (midnight)
    #   2. minus 1 day if needs_shift
    #   3. plus bucket_hour wall-clock hours
    #
    # CRITICAL: we replace HOUR/MINUTE/SECOND directly on the existing
    # timestamp (which is already DST-resolved) instead of constructing a
    # new tz-aware timestamp. This sidesteps DST ambiguity.

    # Step 1: truncate to midnight (date only, but keep tz)
    truncated = df.index.normalize()  # midnight NY of the same day, tz-aware

    # Step 2: optionally shift back 1 day
    bucket_dates = truncated.where(~needs_shift, truncated - pd.Timedelta(days=1))

    # Step 3: build bucket_start by replacing hour on each truncated date.
    # Since truncated is at midnight, replacing hour with bucket_hour gives
    # the correct wall-clock hour. We do this row-by-row to be safe.
    bucket_start_list = []
    for i in range(len(df)):
        bd = bucket_dates[i]  # tz-aware midnight on the bucket date
        bh = int(bucket_hour.iloc[i])
        # Replace hour using pandas: bd.replace(hour=bh) preserves tz info
        # and resolves DST correctly because bh ∈ {2, 6, 10, 14, 18, 22}
        # are all valid hours in NY (no skipped or ambiguous times).
        bucket_start_list.append(bd.replace(hour=bh))

    df["bucket"] = bucket_start_list
    ohlc = df.groupby("bucket").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),     close=("close","last"),
    )
    ohlc.index.name = "ts_ny"
    return ohlc


def find_swings(df_4h, n=1):
    highs = df_4h["high"].values
    lows = df_4h["low"].values
    times = df_4h.index
    high_swings = {}
    low_swings = {}
    for i in range(n, len(df_4h) - n):
        h = highs[i]; l = lows[i]
        is_high = all(h > highs[i - k] for k in range(1, n + 1)) and \
                  all(h > highs[i + k] for k in range(1, n + 1))
        is_low = all(l < lows[i - k] for k in range(1, n + 1)) and \
                 all(l < lows[i + k] for k in range(1, n + 1))
        ts = times[i]
        if is_high:
            high_swings[ts] = h
        if is_low:
            low_swings[ts] = l
    return high_swings, low_swings


# ============================================================
# SYMMETRIC PAIR 4H SWINGS (single chronological pass)
# ============================================================
def pair_swings_symmetric(mnq_swings, mes_swings, mnq_4h, mes_4h, swing_kind):
    """
    Walk all swing events chronologically and pair them.

    For each (asset, ts) in time order:
      if already paired -> skip
      elif same_candle pair possible -> pair
      elif T-1 shift pair possible (other asset has earlier swing at ts-4h, unpaired) -> pair
      elif T+1 shift pair possible (other asset has later swing at ts+4h, unpaired) -> pair
      else if other asset's 4H bar at ts is not paired yet -> solo_parallel
      else -> swing is left unpaired (no level emitted from it)
    """
    rows = []
    level_type = f"4H_Swing{swing_kind}"

    # Build chronological event list (sort by ts, then asset alphabetical)
    events = []
    for ts, price in mnq_swings.items():
        events.append((ts, "MNQ", price))
    for ts, price in mes_swings.items():
        events.append((ts, "MES", price))
    events.sort(key=lambda x: (x[0], x[1]))

    paired = set()  # set of (asset, ts) tuples already paired

    def emit_pair(primary_asset, primary_ts, primary_price,
                  other_asset, other_ts, other_price, pair_kind):
        anchor_ts = min(primary_ts, other_ts)
        level_id = f"{level_type}_{anchor_ts.strftime('%Y%m%d_%H%M')}"
        valid_from = anchor_ts + pd.Timedelta(hours=4)
        valid_until = valid_from + pd.Timedelta(days=90)
        rows.append({"level_id": level_id, "ts_formed": primary_ts, "level_type": level_type,
                      "price": primary_price, "valid_from": valid_from, "valid_until": valid_until,
                      "asset": primary_asset, "pair_kind": pair_kind})
        rows.append({"level_id": level_id, "ts_formed": other_ts, "level_type": level_type,
                      "price": other_price, "valid_from": valid_from, "valid_until": valid_until,
                      "asset": other_asset, "pair_kind": pair_kind})
        paired.add((primary_asset, primary_ts))
        paired.add((other_asset, other_ts))

    for ts, asset, price in events:
        if (asset, ts) in paired:
            continue
        other_asset = "MES" if asset == "MNQ" else "MNQ"
        other_swings = mes_swings if other_asset == "MES" else mnq_swings
        other_4h = mes_4h if other_asset == "MES" else mnq_4h

        # 1. same-candle pair
        if ts in other_swings and (other_asset, ts) not in paired:
            emit_pair(asset, ts, price, other_asset, ts, other_swings[ts], "same_candle")
            continue

        # 2. T-1 shift pair: other asset has EARLIER swing at ts-4h
        prev_ts = ts - pd.Timedelta(hours=4)
        if prev_ts in other_swings and (other_asset, prev_ts) not in paired:
            emit_pair(asset, ts, price, other_asset, prev_ts, other_swings[prev_ts], "shift_paired")
            continue

        # 3. T+1 shift pair: other asset has LATER swing at ts+4h
        next_ts = ts + pd.Timedelta(hours=4)
        if next_ts in other_swings and (other_asset, next_ts) not in paired:
            emit_pair(asset, ts, price, other_asset, next_ts, other_swings[next_ts], "shift_paired")
            continue

        # 4. Solo parallel using other asset's 4H bar at ts
        # FIX: only if the other asset's bar at ts is not already paired.
        if ts in other_4h.index and (other_asset, ts) not in paired:
            other_bar = other_4h.loc[ts]
            other_price = other_bar["high"] if swing_kind == "High" else other_bar["low"]
            emit_pair(asset, ts, price, other_asset, ts, other_price, "solo_parallel")
        # else: swing is left unpaired (no level emitted from it).

    return rows


# ============================================================
# OTHER LEVELS PER ASSET
# ============================================================
def detect_daily_for_asset(df_1m):
    df = df_1m.copy()
    hours = df.index.hour
    dates = df.index.normalize()
    session_day = dates.where(hours >= 18, dates - pd.Timedelta(days=1))
    df["session_day"] = session_day
    daily = df.groupby("session_day").agg(high=("high","max"), low=("low","min")).sort_index()
    out = []
    for i in range(1, len(daily)):
        prev_day = daily.index[i - 1]
        cur_day = daily.index[i]
        pdh = daily["high"].iloc[i - 1]
        pdl = daily["low"].iloc[i - 1]
        valid_from = cur_day + pd.Timedelta(hours=18)
        valid_until = valid_from + pd.Timedelta(days=14)
        ts_formed = prev_day + pd.Timedelta(hours=18)
        out.append((ts_formed, pdh, "PDH", valid_from, valid_until))
        out.append((ts_formed, pdl, "PDL", valid_from, valid_until))
    return out


def detect_weekly_for_asset(df_1m):
    df = df_1m.copy()
    idx = df.index
    days_since_sun = (idx.dayofweek + 1) % 7
    sunday = idx.normalize() - pd.to_timedelta(days_since_sun, unit="D")
    sunday_18 = sunday + pd.Timedelta(hours=18)
    needs_back = idx < sunday_18
    sunday_18 = sunday_18.where(~needs_back, sunday_18 - pd.Timedelta(days=7))
    df["week_start"] = sunday_18
    weekly = df.groupby("week_start").agg(high=("high","max"), low=("low","min")).sort_index()
    out = []
    for i in range(1, len(weekly)):
        prev_week_start = weekly.index[i - 1]
        cur_week_start = weekly.index[i]
        pwh = weekly["high"].iloc[i - 1]
        pwl = weekly["low"].iloc[i - 1]
        valid_from = cur_week_start
        valid_until = valid_from + pd.Timedelta(days=7)
        out.append((prev_week_start, pwh, "PWH", valid_from, valid_until))
        out.append((prev_week_start, pwl, "PWL", valid_from, valid_until))
    return out


def detect_london_for_asset(df_1m):
    df = df_1m.copy()
    df["date"] = df.index.normalize()
    h = df.index.hour
    bars = df[(h >= 2) & (h < 5)]
    daily = bars.groupby("date").agg(high=("high","max"), low=("low","min"))
    out = []
    for date, row in daily.iterrows():
        valid_from = date + pd.Timedelta(hours=5)
        valid_until = date + pd.Timedelta(hours=18)
        out.append((valid_from, row["high"], "LondonHigh", valid_from, valid_until))
        out.append((valid_from, row["low"], "LondonLow", valid_from, valid_until))
    return out


def detect_asia_kz_for_asset(df_1m):
    df = df_1m.copy()
    df["date"] = df.index.normalize()
    h = df.index.hour
    bars = df[(h >= 19) & (h < 22)]
    daily = bars.groupby("date").agg(high=("high","max"), low=("low","min"))
    out = []
    for date, row in daily.iterrows():
        valid_from = date + pd.Timedelta(hours=22)
        valid_until = date + pd.Timedelta(days=1) + pd.Timedelta(hours=18)
        out.append((valid_from, row["high"], "AsiaKZHigh", valid_from, valid_until))
        out.append((valid_from, row["low"], "AsiaKZLow", valid_from, valid_until))
    return out


def detect_asia_on_for_asset(df_1m):
    df = df_1m.copy()
    h = df.index.hour
    dates = df.index.normalize()
    overnight_date = dates.where(h >= 18, dates - pd.Timedelta(days=1))
    in_window = (h >= 18) | (h < 2)
    df = df[in_window].copy()
    df["overnight_date"] = overnight_date[in_window]
    daily = df.groupby("overnight_date").agg(high=("high","max"), low=("low","min"))
    out = []
    for date, row in daily.iterrows():
        valid_from = date + pd.Timedelta(days=1) + pd.Timedelta(hours=2)
        valid_until = date + pd.Timedelta(days=1) + pd.Timedelta(hours=18)
        out.append((valid_from, row["high"], "AsiaONHigh", valid_from, valid_until))
        out.append((valid_from, row["low"], "AsiaONLow", valid_from, valid_until))
    return out


# ============================================================
# FIRST-TOUCH COMPUTATION
# ============================================================
def compute_first_touch_for_paired_levels(paired_rows, mnq_1m, mes_1m):
    by_id = {}
    for r in paired_rows:
        by_id.setdefault(r["level_id"], []).append(r)

    result = {}
    for level_id, rs in by_id.items():
        if len(rs) != 2:
            result[level_id] = None
            continue
        mnq_row = next((x for x in rs if x["asset"] == "MNQ"), None)
        mes_row = next((x for x in rs if x["asset"] == "MES"), None)
        if mnq_row is None or mes_row is None:
            result[level_id] = None
            continue

        lt = mnq_row["level_type"]
        is_high = lt in HIGH_LEVEL_TYPES
        vf = max(mnq_row["valid_from"], mes_row["valid_from"])
        vu = min(mnq_row["valid_until"], mes_row["valid_until"])
        if vf >= vu:
            result[level_id] = None
            continue

        lp_mnq = mnq_row["price"]
        lp_mes = mes_row["price"]

        sub_mnq = mnq_1m.loc[vf:vu]
        sub_mes = mes_1m.loc[vf:vu]

        mnq_ts = None; mes_ts = None
        if len(sub_mnq):
            m = sub_mnq["high"] >= lp_mnq + TICK_SIZE - EPS if is_high else sub_mnq["low"] <= lp_mnq - TICK_SIZE + EPS
            if m.any():
                mnq_ts = sub_mnq.index[m.argmax()]
        if len(sub_mes):
            m = sub_mes["high"] >= lp_mes + TICK_SIZE - EPS if is_high else sub_mes["low"] <= lp_mes - TICK_SIZE + EPS
            if m.any():
                mes_ts = sub_mes.index[m.argmax()]

        if mnq_ts is None and mes_ts is None:
            result[level_id] = None
        elif mnq_ts is None:
            result[level_id] = mes_ts
        elif mes_ts is None:
            result[level_id] = mnq_ts
        else:
            result[level_id] = min(mnq_ts, mes_ts)

    return result


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading 1m data...")
    mnq_1m = load_1m("MNQ")
    mes_1m = load_1m("MES")
    print(f"MNQ: {len(mnq_1m):,} bars | MES: {len(mes_1m):,} bars")

    print("\nBuilding 4H anchored candles (DST-aware)...")
    mnq_4h = build_4h_anchored(mnq_1m)
    mes_4h = build_4h_anchored(mes_1m)
    mnq_4h.to_csv(os.path.join(OUT_FOLDER, "MNQ_4h_anchored.csv"))
    mes_4h.to_csv(os.path.join(OUT_FOLDER, "MES_4h_anchored.csv"))

    # Sanity: 4H bar starts should ONLY be at 02/06/10/14/18/22 NY
    print("\n4H bar start hours (should only show 2, 6, 10, 14, 18, 22):")
    for name, df in [("MNQ", mnq_4h), ("MES", mes_4h)]:
        h_counts = df.groupby(df.index.hour).size()
        out = ", ".join(f"H{h:02d}: {c:,}" for h, c in h_counts.items())
        print(f"  {name}: {out}")

    print("\nDetecting 4H swings (per asset)...")
    mnq_highs, mnq_lows = find_swings(mnq_4h, n=SWING_STRENGTH)
    mes_highs, mes_lows = find_swings(mes_4h, n=SWING_STRENGTH)
    print(f"MNQ: {len(mnq_highs)} highs, {len(mnq_lows)} lows")
    print(f"MES: {len(mes_highs)} highs, {len(mes_lows)} lows")

    print("\nPairing 4H swings across assets (symmetric single-pass)...")
    high_rows = pair_swings_symmetric(mnq_highs, mes_highs, mnq_4h, mes_4h, "High")
    low_rows  = pair_swings_symmetric(mnq_lows, mes_lows, mnq_4h, mes_4h, "Low")
    print(f"Paired swing rows: {len(high_rows) + len(low_rows)}")

    # Pair-kind summary
    for name, rows in [("High", high_rows), ("Low", low_rows)]:
        counts = {}
        for r in rows:
            counts[r["pair_kind"]] = counts.get(r["pair_kind"], 0) + 1
        print(f"  {name} pair kinds: {counts}")

    # Sanity: report any swings that ended up unpaired
    n_mnq_high_paired = sum(1 for r in high_rows if r["asset"] == "MNQ")
    n_mes_high_paired = sum(1 for r in high_rows if r["asset"] == "MES")
    n_mnq_low_paired  = sum(1 for r in low_rows if r["asset"] == "MNQ")
    n_mes_low_paired  = sum(1 for r in low_rows if r["asset"] == "MES")
    print(f"  Unpaired highs - MNQ: {len(mnq_highs) - n_mnq_high_paired}, "
          f"MES: {len(mes_highs) - n_mes_high_paired}")
    print(f"  Unpaired lows  - MNQ: {len(mnq_lows) - n_mnq_low_paired}, "
          f"MES: {len(mes_lows) - n_mes_low_paired}")

    print("\nBuilding per-asset daily/weekly/session level rows with IDs...")
    day_rows = []
    for asset, df_1m in [("MNQ", mnq_1m), ("MES", mes_1m)]:
        for ts_formed, price, lt, vf, vu in detect_daily_for_asset(df_1m):
            day_key = ts_formed.strftime("%Y%m%d")
            level_id = f"{lt}_{day_key}"
            day_rows.append({"level_id": level_id, "ts_formed": ts_formed, "level_type": lt,
                              "price": price, "valid_from": vf, "valid_until": vu,
                              "asset": asset, "pair_kind": "day"})
        for ts_formed, price, lt, vf, vu in detect_weekly_for_asset(df_1m):
            week_key = ts_formed.strftime("%Y%m%d")
            level_id = f"{lt}_{week_key}"
            day_rows.append({"level_id": level_id, "ts_formed": ts_formed, "level_type": lt,
                              "price": price, "valid_from": vf, "valid_until": vu,
                              "asset": asset, "pair_kind": "day"})
        for ts_formed, price, lt, vf, vu in detect_london_for_asset(df_1m):
            day_key = ts_formed.strftime("%Y%m%d")
            level_id = f"{lt}_{day_key}"
            day_rows.append({"level_id": level_id, "ts_formed": ts_formed, "level_type": lt,
                              "price": price, "valid_from": vf, "valid_until": vu,
                              "asset": asset, "pair_kind": "day"})
        for ts_formed, price, lt, vf, vu in detect_asia_kz_for_asset(df_1m):
            day_key = ts_formed.strftime("%Y%m%d")
            level_id = f"{lt}_{day_key}"
            day_rows.append({"level_id": level_id, "ts_formed": ts_formed, "level_type": lt,
                              "price": price, "valid_from": vf, "valid_until": vu,
                              "asset": asset, "pair_kind": "day"})
        for ts_formed, price, lt, vf, vu in detect_asia_on_for_asset(df_1m):
            day_key = ts_formed.strftime("%Y%m%d")
            level_id = f"{lt}_{day_key}"
            day_rows.append({"level_id": level_id, "ts_formed": ts_formed, "level_type": lt,
                              "price": price, "valid_from": vf, "valid_until": vu,
                              "asset": asset, "pair_kind": "day"})

    all_rows = high_rows + low_rows + day_rows

    print("\nComputing first-touch timestamps for persistent levels (may take a minute)...")
    persistent_rows = [r for r in all_rows if r["level_type"] in PERSISTENT_TYPES]
    first_touch_map = compute_first_touch_for_paired_levels(persistent_rows, mnq_1m, mes_1m)
    print(f"Computed first-touch for {len(first_touch_map)} paired levels")
    touched_count = sum(1 for v in first_touch_map.values() if v is not None)
    print(f"  Of which {touched_count} have been touched within their window")

    for r in all_rows:
        r["first_touch_ts"] = first_touch_map.get(r["level_id"], None)

    for asset in ["MNQ", "MES"]:
        asset_rows = [r for r in all_rows if r["asset"] == asset]
        df = pd.DataFrame(asset_rows)
        if len(df):
            df = df.sort_values("ts_formed").reset_index(drop=True)
        save_levels(df, os.path.join(OUT_FOLDER, f"{asset}_levels.csv"))
        print(f"\n{asset}: {len(df):,} levels saved")
        if len(df):
            print(df["level_type"].value_counts().to_string())


if __name__ == "__main__":
    main()