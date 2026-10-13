"""
Fetch continuous back-adjusted MES 1-minute bars from Interactive Brokers.

Contract chain is bounded by what IBKR's paper API can actually resolve:
expired-contract lookups only reach back to ~Sept 2024, so the continuous
series here covers ~mid-2024 onward, not the full 2019-2026 range originally
scoped in exp1_taxonomy.py.

Roll rule: each contract is "front month" from 8 calendar days before the
PRIOR contract's expiry, until 8 calendar days before its OWN expiry.
Back-adjustment: additive splice, most-recent contract un-shifted, each
older contract shifted so its raw close lines up with the newer contract's
raw close at the roll boundary (accumulated going backward through the chain).
"""

import asyncio
import time
from pathlib import Path

import pandas as pd

asyncio.set_event_loop(asyncio.new_event_loop())
from ib_insync import IB, Future

DATA_DIR = Path.home() / "mes_research" / "data"
RAW_DIR = DATA_DIR / "raw"
ROLL_DAYS_BEFORE_EXPIRY = 8
CHUNK_DAYS = 6            # 1-min bars: IBKR max ~1 week per request
PACE_SLEEP_SEC = 8        # delay between historical data requests
RETRY_SLEEPS = [10, 30, 60]

CONTRACTS = [
    # (localSymbol, expiry YYYYMMDD, conId)
    ("MESU4", "20240920", 637533398),
    ("MESZ4", "20241220", 654503314),
    ("MESH5", "20250321", 672387462),
    ("MESM5", "20250620", 691171673),
    ("MESU5", "20250919", 711280067),
    ("MESZ5", "20251219", 730283085),
    ("MESH6", "20260320", 750150186),
    ("MESM6", "20260618", 770561194),
    ("MESU6", "20260918", 793356217),
]


def roll_out_date(expiry_str: str) -> pd.Timestamp:
    return pd.Timestamp(expiry_str) - pd.Timedelta(days=ROLL_DAYS_BEFORE_EXPIRY)


def build_windows():
    """Assign each contract an active [start, end) window based on roll dates."""
    windows = []
    for i, (sym, expiry, conid) in enumerate(CONTRACTS):
        end = roll_out_date(expiry)
        if i == 0:
            # No earlier contract in our resolvable chain; assume ~1 quarter.
            start = end - pd.Timedelta(days=90)
        else:
            start = roll_out_date(CONTRACTS[i - 1][1])
        windows.append((sym, expiry, conid, start, end))
    return windows


def fetch_contract_bars(ib: IB, sym: str, expiry: str, conid: int,
                         window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    raw_path = RAW_DIR / f"{sym}.parquet"
    if raw_path.exists():
        print(f"[{sym}] already fetched, loading from disk", flush=True)
        return pd.read_parquet(raw_path)

    contract = Future(symbol="MES", lastTradeDateOrContractMonth=expiry,
                       exchange="CME", currency="USD", conId=conid)

    now = pd.Timestamp.now()
    fetch_end = min(window_end, now)

    all_bars = []
    cursor = fetch_end
    empty_streak = 0

    while cursor > window_start:
        chunk_start = max(cursor - pd.Timedelta(days=CHUNK_DAYS), window_start)
        end_dt_str = cursor.strftime("%Y%m%d-%H:%M:%S")

        bars = None
        for attempt, retry_sleep in enumerate([0] + RETRY_SLEEPS):
            if retry_sleep:
                print(f"[{sym}] retry {attempt} after pacing/error, sleeping {retry_sleep}s", flush=True)
                time.sleep(retry_sleep)
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_dt_str,
                    durationStr=f"{CHUNK_DAYS} D",
                    barSizeSetting="1 min",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=1,
                )
                break
            except Exception as e:
                print(f"[{sym}] error at {end_dt_str}: {e}", flush=True)
                bars = None

        if bars:
            all_bars.extend(bars)
            empty_streak = 0
            earliest = min(pd.Timestamp(b.date).tz_localize(None) for b in bars)
            print(f"[{sym}] {end_dt_str} -> {len(bars)} bars (earliest {earliest})", flush=True)
        else:
            empty_streak += 1
            print(f"[{sym}] {end_dt_str} -> 0 bars (empty streak {empty_streak})", flush=True)
            if empty_streak >= 4:
                print(f"[{sym}] stopping early: contract likely not trading before this point", flush=True)
                break

        cursor = chunk_start
        time.sleep(PACE_SLEEP_SEC)

    if not all_bars:
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    else:
        df = pd.DataFrame([{
            "timestamp": pd.Timestamp(b.date).tz_localize(None),
            "open": b.open, "high": b.high, "low": b.low, "close": b.close,
            "volume": b.volume,
        } for b in all_bars])
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        df = df[(df["timestamp"] >= window_start) & (df["timestamp"] < min(window_end, now))]

    df.to_parquet(raw_path, index=False)
    print(f"[{sym}] saved {len(df)} bars to {raw_path}", flush=True)
    return df


def back_adjust_and_stitch(windows, raw_frames):
    """Splice contracts newest-first, additively shifting each older contract
    so its raw close matches the newer contract's raw close at the roll boundary."""
    windows_desc = list(reversed(windows))
    offset = 0.0
    stitched = []
    roll_log = []

    prev_sym = None
    prev_df = None
    for sym, expiry, conid, start, end in windows_desc:
        df = raw_frames[sym].copy()
        if prev_df is not None and not df.empty and not prev_df.empty:
            splice_time = start  # this contract's window start = older/newer boundary
            older_at_splice = df[df["timestamp"] <= splice_time]
            newer_at_splice = prev_df[prev_df["timestamp"] >= splice_time]
            if not older_at_splice.empty and not newer_at_splice.empty:
                older_raw = older_at_splice.iloc[-1]["close"]
                newer_raw = newer_at_splice.iloc[0]["close"]
                offset = offset + (newer_raw - older_raw)
                roll_log.append({
                    "from": prev_sym, "to": sym, "splice_time": splice_time,
                    "older_raw_close": older_raw, "newer_raw_close": newer_raw,
                    "cumulative_offset": offset,
                })

        df["open"] += offset
        df["high"] += offset
        df["low"] += offset
        df["close"] += offset
        df["contract"] = sym
        stitched.append(df)

        prev_sym = sym
        prev_df = raw_frames[sym]

    continuous = pd.concat(reversed(stitched), ignore_index=True)
    continuous = continuous.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return continuous, pd.DataFrame(roll_log)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    windows = build_windows()
    print("Contract windows:", flush=True)
    for sym, expiry, conid, start, end in windows:
        print(f"  {sym}: [{start.date()}, {end.date()})", flush=True)

    ib = IB()
    ib.connect("127.0.0.1", 7497, clientId=21, timeout=15)

    raw_frames = {}
    try:
        for sym, expiry, conid, start, end in windows:
            raw_frames[sym] = fetch_contract_bars(ib, sym, expiry, conid, start, end)
    finally:
        ib.disconnect()

    print("\nStitching continuous back-adjusted series...", flush=True)
    continuous, roll_log = back_adjust_and_stitch(windows, raw_frames)

    out_path = DATA_DIR / "mes_continuous_1min.parquet"
    continuous.to_parquet(out_path, index=False)
    roll_log.to_csv(DATA_DIR / "roll_log.csv", index=False)

    print(f"\nDone. {len(continuous)} bars -> {out_path}", flush=True)
    print(f"Date range: {continuous['timestamp'].min()} to {continuous['timestamp'].max()}", flush=True)
    print("\nRoll log:", flush=True)
    print(roll_log.to_string(index=False), flush=True)
    print("\nBars per contract:", flush=True)
    print(continuous["contract"].value_counts().sort_index().to_string(), flush=True)


if __name__ == "__main__":
    main()
