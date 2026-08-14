"""
Phase 1: Data pipeline for SMT+MSS backtest.

What it does:
  1. Loads Databento 1m CSVs (MNQ + MES outright contracts)
  2. Builds back-adjusted continuous price series for each (volume-based rollover)
  3. Converts UTC -> NY time
  4. Resamples to 5m, 15m, 1h, 4h
     - 4H bars are anchored to the futures session: 18/22/02/06/10/14 NY
       (same anchoring used in Phase 2 for swing detection)
  5. Saves all timeframes to ./out/ as CSV
  6. Prints sanity checks

Usage:
  - Put this script + your Databento CSVs in the same folder (or /New subfolder)
  - Edit DATA_FOLDER below if your CSVs are elsewhere
  - Run: python3 phase1.py
"""

import os
import re
import pandas as pd

# ============================================================
# USER CONFIG -- only thing you should ever need to change
# ============================================================
DATA_FOLDER = "/Users/voligle/workspace/Trading/Trading_code/clause Backtester/"

CSV_FILES = [
    "New/glbx-mdp3-20230301-20231230.ohlcv-1m.csv",
    "New/glbx-mdp3-20240301-20251230.ohlcv-1m.csv",
]
# ============================================================

OUT_FOLDER = os.path.join(DATA_FOLDER, "New/out")
os.makedirs(OUT_FOLDER, exist_ok=True)

# CME quarterly month codes
MONTH_CODE = {"H": 3, "M": 6, "U": 9, "Z": 12}


def parse_symbol(sym):
    """'MNQH3' or 'MNQH23' -> ('MNQ', 2023, 3). Returns None for spreads/other."""
    if not isinstance(sym, str) or "-" in sym:
        return None
    m = re.match(r"^(MNQ|MES)([HMUZ])(\d{1,2})$", sym)
    if not m:
        return None
    root = m.group(1)
    month = MONTH_CODE[m.group(2)]
    year_part = m.group(3)
    if len(year_part) == 1:
        year = 2020 + int(year_part)
    else:
        # 2-digit year. Assume 2000s if < 50, else 1900s.
        year_val = int(year_part)
        year = 2000 + year_val if year_val < 50 else 1900 + year_val
    return (root, year, month)


def load_file(filename):
    path = os.path.join(DATA_FOLDER, filename)
    print(f"Loading {filename}...")
    df = pd.read_csv(
        path,
        usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol"],
    )
    parsed = df["symbol"].apply(parse_symbol)
    df = df[parsed.notna()].copy()
    df[["root", "exp_year", "exp_month"]] = pd.DataFrame(
        parsed[parsed.notna()].tolist(), index=df.index
    )
    df["ts"] = pd.to_datetime(df["ts_event"], utc=True)
    return df[["ts", "open", "high", "low", "close", "volume", "root", "exp_year", "exp_month"]]


def detect_rollovers_by_volume(root_df, confirm_days=2):
    """
    Detect rollover dates using daily volume crossover with confirmation.

    Algorithm:
    1. For each date, compute total volume per contract.
    2. Walk forward day by day. Current "front month" = the contract we're using.
    3. If a DIFFERENT contract has higher daily volume than the front month
       for `confirm_days` consecutive days, roll on the FIRST of those days.
    4. Roll happens at 18:00 NY (start of futures session) of the first crossover day.

    Returns: list of (year, month, roll_ts_utc) tuples, in chronological order,
    where roll_ts_utc is when the new contract takes over.
    """
    df = root_df.copy()
    df["date"] = df["ts"].dt.tz_convert("America/New_York").dt.date
    df["contract_key"] = list(zip(df["exp_year"], df["exp_month"]))

    # Daily volume per contract
    daily_vol = df.groupby(["date", "contract_key"])["volume"].sum().unstack(fill_value=0)
    daily_vol = daily_vol.sort_index()

    # Walk forward to identify front month and rollover dates
    rollovers = []  # list of (yr, mo, roll_ts_utc)
    contracts_sorted = sorted(daily_vol.columns)  # ascending by (yr, mo)

    if len(contracts_sorted) == 0:
        return rollovers

    # Initial front month = contract with highest volume on first day
    front = daily_vol.iloc[0].idxmax()
    rollovers.append((front[0], front[1], None))  # None = "starts at beginning of data"

    # Track consecutive-day signal for each candidate contract
    pending_streak = {c: 0 for c in contracts_sorted}

    dates = list(daily_vol.index)
    for i, d in enumerate(dates):
        # On this date, find which contract has the highest volume
        row = daily_vol.loc[d]
        if row.sum() == 0:
            # No data day, skip without resetting streaks (could be a holiday)
            continue
        leader = row.idxmax()

        # If leader is the current front, no rollover; reset all streaks
        if leader == front:
            pending_streak = {c: 0 for c in contracts_sorted}
            continue

        # Leader is NOT current front. Increment streak for leader, reset others
        for c in contracts_sorted:
            if c == leader:
                pending_streak[c] += 1
            else:
                pending_streak[c] = 0

        # Confirm rollover after `confirm_days` consecutive days
        if pending_streak[leader] >= confirm_days:
            # Roll happens at 18:00 NY of (i - confirm_days + 1)-th day,
            # i.e. the FIRST day of the streak
            roll_date = dates[i - confirm_days + 1]
            roll_ts_ny = pd.Timestamp(roll_date, tz="America/New_York") - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
            roll_ts_utc = roll_ts_ny.tz_convert("UTC")
            rollovers.append((leader[0], leader[1], roll_ts_utc))
            front = leader
            pending_streak = {c: 0 for c in contracts_sorted}

    return rollovers


def build_continuous(root_df):
    """
    Back-adjusted continuous series, using VOLUME-BASED rollover detection.
    Older bars are shifted by cumulative price differences at each rollover.
    """
    rollovers = detect_rollovers_by_volume(root_df, confirm_days=2)

    if not rollovers:
        # No data at all
        return root_df[["ts", "open", "high", "low", "close", "volume"]].copy()

    # Build segments: each contract used from its rollover_in to next rollover
    segments = []
    for i, (yr, mo, roll_in) in enumerate(rollovers):
        if i + 1 < len(rollovers):
            roll_out = rollovers[i + 1][2]
        else:
            roll_out = pd.Timestamp("2099-01-01", tz="UTC")

        if roll_in is None:
            # First segment uses contract from beginning of data
            seg_mask = (
                (root_df["exp_year"] == yr)
                & (root_df["exp_month"] == mo)
                & (root_df["ts"] < roll_out)
            )
        else:
            seg_mask = (
                (root_df["exp_year"] == yr)
                & (root_df["exp_month"] == mo)
                & (root_df["ts"] >= roll_in)
                & (root_df["ts"] < roll_out)
            )
        seg = root_df[seg_mask].copy()
        if len(seg) > 0:
            segments.append((yr, mo, roll_out, seg))

    if not segments:
        return root_df[["ts", "open", "high", "low", "close", "volume"]].copy()

    # Walk newest -> oldest, accumulate price offset, apply to older segments.
    # Offset = (new contract first close - old contract last close) at the rollover boundary.
    cum_offset = 0.0
    adjusted = []
    for i in range(len(segments) - 1, -1, -1):
        yr, mo, roll_out, seg = segments[i]
        seg = seg.copy()
        seg[["open", "high", "low", "close"]] += cum_offset
        adjusted.insert(0, seg)
        if i > 0:
            prev_yr, prev_mo, prev_roll_out, _ = segments[i - 1]
            # rollover boundary is between prev segment and this one
            # (the time when this segment took over)
            # Find first bar of new contract at/after roll_in
            roll_in = rollovers[i][2]
            new_first = root_df[
                (root_df["exp_year"] == yr)
                & (root_df["exp_month"] == mo)
                & (root_df["ts"] >= roll_in)
            ].head(1)
            # Find last bar of old contract before roll_in (its close at boundary)
            old_last = root_df[
                (root_df["exp_year"] == prev_yr)
                & (root_df["exp_month"] == prev_mo)
                & (root_df["ts"] < roll_in)
            ].tail(1)
            if len(new_first) and len(old_last):
                diff = new_first["close"].iloc[0] - old_last["close"].iloc[0]
                cum_offset += diff

    cont = pd.concat(adjusted, ignore_index=True).sort_values("ts").reset_index(drop=True)

    # Diagnostics print
    print(f"  Rollovers detected: {len(rollovers)}")
    for yr, mo, roll_ts in rollovers:
        ts_ny = roll_ts.tz_convert("America/New_York") if roll_ts else "(start of data)"
        print(f"    -> contract {yr}-{mo:02d} starts: {ts_ny}")

    return cont[["ts", "open", "high", "low", "close", "volume"]]


def resample_4h_session_anchored(df_ny):
    """
    Aggregate 1m NY bars into 4H buckets anchored to the futures session.
    Bucket starts: 18/22/02/06/10/14 NY  (NOT calendar-aligned).

    For each 1m bar, pick its bucket_start by hour:
       18-22 -> 18:00 of that calendar day
       22-02 -> 22:00 (the 02-06 hours belong to NEXT calendar day, but
                       the bucket_start uses the day BEFORE)
       02-06 -> 02:00 of that calendar day
       06-10 -> 06:00
       10-14 -> 10:00
       14-18 -> 14:00
    """
    df = df_ny.copy()
    h = df.index.hour
    bucket_hour = pd.Series(index=df.index, dtype="int64")
    bucket_hour[(h >= 18) & (h < 22)] = 18
    bucket_hour[(h >= 22) | (h < 2)] = 22
    bucket_hour[(h >= 2) & (h < 6)] = 2
    bucket_hour[(h >= 6) & (h < 10)] = 6
    bucket_hour[(h >= 10) & (h < 14)] = 10
    bucket_hour[(h >= 14) & (h < 18)] = 14

    bucket_date = df.index.normalize()
    # Hours 0-1 belong to the 22:00 bucket of the PREVIOUS calendar day
    needs_shift = (h >= 0) & (h < 2)
    bucket_date = bucket_date.where(~needs_shift, bucket_date - pd.Timedelta(days=1))

    bucket_start = bucket_date + pd.to_timedelta(bucket_hour, unit="h")
    df = df.copy()
    df["bucket"] = bucket_start

    ohlc = df.groupby("bucket").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    ohlc.index.name = "ts_ny"
    return ohlc


def resample_to_ny(df_1m):
    """Convert UTC -> NY time and resample to multiple timeframes."""
    df = df_1m.copy()
    df["ts_ny"] = df["ts"].dt.tz_convert("America/New_York")
    df = df.set_index("ts_ny")[["open", "high", "low", "close", "volume"]]

    # 1m, 5m, 15m, 1h: standard left-anchored resampling
    timeframes = {"1m": None, "5m": "5min", "15m": "15min", "1h": "1h"}
    out = {}
    for tf, rule in timeframes.items():
        if rule is None:
            out[tf] = df
        else:
            out[tf] = df.resample(rule, label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"}
            ).dropna()

    # 4h: futures-session anchored at 18/22/02/06/10/14 NY
    out["4h"] = resample_4h_session_anchored(df).dropna()

    return out


def main():
    # 1) Load all files
    all_data = pd.concat([load_file(f) for f in CSV_FILES], ignore_index=True)
    print(f"Total outright bars loaded: {len(all_data):,}")

    # 2) Build continuous + resample for each root
    for root in ["MNQ", "MES"]:
        print(f"\nProcessing {root}...")
        rdf = all_data[all_data["root"] == root]
        cont = build_continuous(rdf)
        print(f"  Continuous bars: {len(cont):,}")
        print(f"  Price range: {cont['close'].min():.2f} - {cont['close'].max():.2f}")

        timeframes = resample_to_ny(cont)
        for tf, df in timeframes.items():
            out_path = os.path.join(OUT_FOLDER, f"{root}_{tf}.csv")
            df.to_csv(out_path)
            print(f"  Saved {root}_{tf}.csv ({len(df):,} bars)")

    # 3) Sanity checks
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    mnq_1m = pd.read_csv(os.path.join(OUT_FOLDER, "MNQ_1m.csv"))
    mnq_1m["ts_ny"] = pd.to_datetime(mnq_1m["ts_ny"], utc=True)
    mnq_1m = mnq_1m.set_index("ts_ny")

    daily_counts = mnq_1m.groupby(mnq_1m.index.date).size()
    print(f"\nMNQ 1m bars per day: avg={daily_counts.mean():.0f}, "
          f"min={daily_counts.min()}, max={daily_counts.max()}")
    print("(Expect avg ~1100-1200 for full futures session minus 17:00 break)")

    print("\nBars by NY hour (hour 17 should be empty - daily futures break):")
    hour_counts = mnq_1m.groupby(mnq_1m.index.hour).size()
    for h in range(24):
        n = hour_counts.get(h, 0)
        bar = "#" * int(n / 5000)
        print(f"  H{h:02d}: {n:>7,} {bar}")

    # 4H bar anchoring sanity check
    print("\n4H bar start hours (should only show 2, 6, 10, 14, 18, 22):")
    mnq_4h = pd.read_csv(os.path.join(OUT_FOLDER, "MNQ_4h.csv"))
    mnq_4h["ts_ny"] = pd.to_datetime(mnq_4h["ts_ny"], utc=True)
    mnq_4h = mnq_4h.set_index("ts_ny")
    h4_counts = mnq_4h.groupby(mnq_4h.index.hour).size()
    for h, c in h4_counts.items():
        print(f"  H{h:02d}: {c:>5,} bars")

    print(f"\nDone. Output files in: {OUT_FOLDER}")


if __name__ == "__main__":
    main()