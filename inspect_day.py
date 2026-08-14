import pandas as pd, glob, os
OUT = "/Users/voligle/workspace/Trading/Trading_code/clause Backtester/New/out"
TICK = 0.25
TZ = "America/New_York"

mnq_1m = pd.read_csv(os.path.join(OUT, "MNQ_1m.csv"))
mnq_1m["ts_ny"] = pd.to_datetime(mnq_1m["ts_ny"], utc=True).dt.tz_convert(TZ)
mnq_1m = mnq_1m.set_index("ts_ny")

start = pd.Timestamp("2023-03-08 07:40", tz=TZ)
end   = pd.Timestamp("2023-03-08 17:00", tz=TZ)
LP_NC = 14720.25

window = mnq_1m.loc[start:end]
mask = window["low"] <= LP_NC - TICK + 1e-9
if mask.any():
    first_cross = window.index[mask.argmax()]
    print(f"MNQ first crossed asia low at: {first_cross}")
    print(f"  Bar: low={window.loc[first_cross]['low']:.2f}, threshold={LP_NC-TICK:.2f}")
else:
    print("MNQ never crossed asia low during the session.")

# Also confirm: SMT's death_ts from the events file
all_smt = []
for f in glob.glob(os.path.join(OUT, "SMT_events_*.csv")):
    d = pd.read_csv(f)
    if len(d) == 0: continue
    d["birth_ts"] = pd.to_datetime(d["birth_ts"], utc=True).dt.tz_convert(TZ)
    d["death_ts"] = pd.to_datetime(d["death_ts"], utc=True).dt.tz_convert(TZ)
    all_smt.append(d)
smts = pd.concat(all_smt, ignore_index=True)
win_start = pd.Timestamp("2023-03-08", tz=TZ) - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
win_end   = pd.Timestamp("2023-03-08", tz=TZ) + pd.Timedelta(hours=17)
day = smts[
    (smts["birth_ts"] >= win_start)
    & (smts["birth_ts"] <= win_end)
    & (smts["level_type"].isin(["AsiaKZLow","AsiaONLow"]))
    & (smts["direction"] == "bullish")
]
print("\nSMT row (from SMT_events):")
print(day[["level_id","birth_ts","death_ts","status"]].to_string(index=False))