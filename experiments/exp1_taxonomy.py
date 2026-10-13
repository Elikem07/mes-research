"""
Experiment 1: The Reaction Taxonomy

This is the core of the research. Before any strategy, measure the five-outcome
classification on every level touch, conditioned on 16 regime variables.

Output: P(outcome | regime) — the answer to the central research question,
independent of any strategy design.

Start here AFTER Experiment 0 passes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Optional
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mes_research_project import ATRCompute, VWAPCompute  # noqa: E402

# ============================================================================
# 1. FIVE-OUTCOME TAXONOMY
# ============================================================================

class ReactionOutcome(Enum):
    """Five mutually exclusive outcomes for a level breach."""
    CONTINUATION = "continuation"      # Breakout and follow-through
    RETEST = "retest"                  # Breakout, pullback, then continuation
    REJECTION = "rejection"            # False breakout and reversal
    MEAN_REVERSION = "mean_reversion"  # Return to reference, no opposite-side extreme
    NO_REACTION = "no_reaction"        # Neither threshold met within horizon

@dataclass
class LevelTouchEvent:
    """A touch of a reference level, classified into one of five outcomes."""
    timestamp: datetime
    level_name: str  # "PDH", "PDL", "ONH", "ONL", "VWAP", etc.
    level_price: float
    
    breach_time: datetime
    breach_extreme: float  # furthest point reached after breach

    # Regime variables at the time of breach (all point-in-time)
    volreg: str  # "low", "mid", "high"
    gap: float  # signed, in ATR_D units
    open_loc: float  # 0-1 in prior range
    onr: float  # overnight range in ATR_D
    slope_vwap: float  # VWAP slope in ATR units
    align: bool  # aligned with daily + overnight trend
    rvol_cum: float  # relative cumulative volume
    t_bucket: str  # "09:30-11:00", "11:00-14:00", "14:00-16:00"
    side: str  # "long" (upside breach) or "short" (downside)
    n_tests: int  # first vs repeated
    rv_pct: str  # "low", "mid", "high" intraday volatility
    news_session: bool
    roll_week: bool

    outcome: Optional[ReactionOutcome] = None
    outcome_details: Dict[str, any] = None

    def __repr__(self):
        return (
            f"LevelTouch({self.level_name} {self.outcome.value if self.outcome else '?'} "
            f"at {self.breach_time.strftime('%H:%M')})"
        )


class TaxonomyClassifier:
    """
    Classify each level breach into one of five outcomes.
    
    Thresholds (from catalog):
    - a = 0.75 * ATR_5: favorable excursion to consider "continuation"
    - b = 0.35 * ATR_5: return threshold
    - H = 30 min: horizon
    """
    
    def __init__(self, atr5: float, vwap_ref: float, horizon_minutes: int = 30):
        self.atr5 = atr5
        self.vwap_ref = vwap_ref
        self.horizon_minutes = horizon_minutes
        
        self.a = 0.75 * atr5  # favorable excursion threshold
        self.b = 0.35 * atr5  # pullback threshold
    
    def classify(self, 
                 level_price: float,
                 bars: List[Dict],  # OHLCV bars after breach, in order
                 side: str,  # "long" (breach above) or "short" (breach below)
                 ) -> ReactionOutcome:
        """
        Classify using first-hit logic (mutually exclusive).
        
        bars: list of dicts with keys: timestamp, high, low, close
        side: "long" = breach above level, "short" = breach below
        
        Returns the first-matched outcome.
        """
        
        if side == "long":
            return self._classify_upside(level_price, bars)
        else:
            return self._classify_downside(level_price, bars)
    
    def _classify_upside(self, level_price: float, bars: List[Dict]) -> ReactionOutcome:
        """Upside breach (break above level_price)."""
        
        max_above = None  # furthest point above level
        pullback_to_level = None  # did it pull back to within b?
        max_below = None  # furthest point below level (for rejection)

        if not bars:
            return ReactionOutcome.NO_REACTION
        horizon_end = pd.Timestamp(bars[0]['timestamp']) + timedelta(minutes=self.horizon_minutes)

        for i, bar in enumerate(bars):
            if pd.Timestamp(bar['timestamp']) > horizon_end:
                break  # outside horizon
            
            h, l, c = bar['high'], bar['low'], bar['close']
            
            # Track furthest point above
            if h > level_price:
                if max_above is None:
                    max_above = h
                else:
                    max_above = max(max_above, h)
            
            # Track pullback to within b of level
            if pullback_to_level is None and c <= level_price + self.b and c >= level_price:
                pullback_to_level = i
            
            # Track furthest point below (for rejection)
            if l < level_price:
                if max_below is None:
                    max_below = l
                else:
                    max_below = min(max_below, l)
        
        # First-hit logic (ordered as in catalog)
        if max_above is not None and (max_above - level_price) >= self.a:
            if pullback_to_level is None or pullback_to_level > 10:  # no pullback within ~10 bars
                # 1. CONTINUATION: reached favorable excursion without pullback
                return ReactionOutcome.CONTINUATION
        
        if pullback_to_level is not None:
            # Look for re-extension after pullback
            for i in range(pullback_to_level, len(bars)):
                if pd.Timestamp(bars[i]['timestamp']) > horizon_end:
                    break
                if bars[i]['high'] >= level_price + self.a:
                    # 2. RETEST: pullback then re-extension
                    return ReactionOutcome.RETEST
            # Pullback occurred but no re-extension -> not retest
        
        # Check for rejection: close back inside, opposite side exceeds threshold
        if max_below is not None and (level_price - max_below) >= self.a:
            if pullback_to_level is not None:  # must have come back inside
                # 3. REJECTION: back inside, opposite side exceeds
                return ReactionOutcome.REJECTION
        
        # Check for mean reversion to VWAP (without opposite-side extreme)
        if pullback_to_level is not None and max_below is None:
            # 4. MEAN REVERSION: returned inside, no opposite extreme
            return ReactionOutcome.MEAN_REVERSION
        
        # 5. NO MEANINGFUL REACTION
        return ReactionOutcome.NO_REACTION
    
    def _classify_downside(self, level_price: float, bars: List[Dict]) -> ReactionOutcome:
        """Downside breach (break below level_price). Mirror of upside."""
        
        max_below = None
        pullback_to_level = None
        max_above = None

        if not bars:
            return ReactionOutcome.NO_REACTION
        horizon_end = pd.Timestamp(bars[0]['timestamp']) + timedelta(minutes=self.horizon_minutes)

        for i, bar in enumerate(bars):
            if pd.Timestamp(bar['timestamp']) > horizon_end:
                break
            
            h, l, c = bar['high'], bar['low'], bar['close']
            
            if l < level_price:
                if max_below is None:
                    max_below = l
                else:
                    max_below = min(max_below, l)
            
            if pullback_to_level is None and c >= level_price - self.b and c <= level_price:
                pullback_to_level = i
            
            if h > level_price:
                if max_above is None:
                    max_above = h
                else:
                    max_above = max(max_above, h)
        
        if max_below is not None and (level_price - max_below) >= self.a:
            if pullback_to_level is None or pullback_to_level > 10:
                return ReactionOutcome.CONTINUATION
        
        if pullback_to_level is not None:
            for i in range(pullback_to_level, len(bars)):
                if pd.Timestamp(bars[i]['timestamp']) > horizon_end:
                    break
                if bars[i]['low'] <= level_price - self.a:
                    return ReactionOutcome.RETEST
        
        if max_above is not None and (max_above - level_price) >= self.a:
            if pullback_to_level is not None:
                return ReactionOutcome.REJECTION
        
        if pullback_to_level is not None and max_above is None:
            return ReactionOutcome.MEAN_REVERSION
        
        return ReactionOutcome.NO_REACTION


# ============================================================================
# 2. REGIME BUCKETING
# ============================================================================

def bucket_volreg(atr_d_value: float, atr_d_series: pd.Series) -> str:
    """Tercile bucketing of ATR_D."""
    terciles = atr_d_series.quantile([0.33, 0.67])
    if atr_d_value <= terciles[0.33]:
        return "low"
    elif atr_d_value <= terciles[0.67]:
        return "mid"
    else:
        return "high"

def bucket_rv_pct(rv_30: float, rv_30_series: pd.Series) -> str:
    """Tercile bucketing of intraday realized volatility."""
    terciles = rv_30_series.quantile([0.33, 0.67])
    if rv_30 <= terciles[0.33]:
        return "low"
    elif rv_30 <= terciles[0.67]:
        return "mid"
    else:
        return "high"

def bucket_rvol_cum(rvol_cum: float) -> str:
    """Bucketing of relative cumulative volume."""
    if rvol_cum <= 0.80:
        return "low"
    elif rvol_cum <= 1.20:
        return "normal"
    else:
        return "high"

def bucket_time(timestamp: datetime) -> str:
    """Time-of-day bucket (ET, RTH)."""
    hour = timestamp.hour
    minute = timestamp.minute
    total_min = hour * 60 + minute
    
    # 09:30 ET = 570 minutes after midnight
    mins_since_open = total_min - 570
    
    if 0 <= mins_since_open < 90:  # 09:30-11:00
        return "morning"
    elif 90 <= mins_since_open < 270:  # 11:00-14:00 (midday chop)
        return "midday"
    else:  # 14:00-16:00
        return "afternoon"


# ============================================================================
# 3. TAXONOMY STUDY RUNNER
# ============================================================================

class TaxonomyStudy:
    """Run the full taxonomy classification on a dataset.

    Modeling choices made to fill gaps left open by the taxonomy spec
    (documented here since they're judgment calls, not given facts):

    - RTH session = 09:30-16:00 ET. Only RTH bars are tested for level
      touches; overnight bars are used only to derive ONH/ONL and
      overnight-drift context for the following RTH day.
    - Input timestamps are assumed to be IBKR's native US/Central wall
      clock (what TWS returns for CME futures); they're converted to ET
      internally since the taxonomy's time buckets are ET-based.
    - ATR_5 (used by TaxonomyClassifier's a/b thresholds) is a 14-period
      ATR on 5-minute RTH bars, looked up point-in-time (last fully closed
      5-minute bar strictly before the current 1-minute bar) - a 5-period
      ATR on 1-minute bars would be too small relative to a 30-minute
      classification horizon to mean anything.
    - ATR_D is a 14-period ATR on daily RTH bars (PDH/PDL/gap/open_loc all
      derive from this).
    - VWAP is session-anchored (resets each RTH day); slope_vwap compares
      the current VWAP to its value 15 bars earlier, in ATR_D units.
    - volreg/rv_pct bucket thresholds, and the rvol_cum baseline, are all
      computed from the full sample. This is fine for a *descriptive*
      study of where a day sits in its own historical distribution, but
      would be lookahead if reused as a live trading signal.
    - align = True when the breach side agrees with the sign of both the
      day's gap and the overnight drift; False (including when either is
      undefined) otherwise.
    - news_session is always False - no news calendar is wired up. This is
      a known gap, not a modeled "no news" fact.
    - roll_week uses the actual contract-roll dates recorded during the
      IBKR fetch (data/roll_log.csv), flagging RTH days within 3 calendar
      days of a splice.
    """

    RTH_START_MIN = 9 * 60 + 30   # 09:30 ET
    RTH_END_MIN = 16 * 60         # 16:00 ET
    HORIZON_MINUTES = 30
    ATR_D_WARMUP = 14

    def __init__(self, bars_1m: pd.DataFrame, roll_dates: Optional[List] = None,
                 source_tz: str = "US/Central"):
        """
        bars_1m: DataFrame with columns [timestamp, open, high, low, close, volume].
        roll_dates: known contract-roll dates (e.g. from roll_log.csv's splice_time),
                    used to flag roll_week.
        source_tz: timezone the input timestamps are actually in.
        """
        df = bars_1m.copy()
        ts_local = pd.DatetimeIndex(pd.to_datetime(df["timestamp"])).tz_localize(
            source_tz, ambiguous="NaT", nonexistent="shift_forward"
        )
        df["timestamp"] = ts_local.tz_convert("US/Eastern").tz_localize(None)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        mins_of_day = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
        df["date"] = df["timestamp"].dt.date
        df["is_rth"] = (mins_of_day >= self.RTH_START_MIN) & (mins_of_day < self.RTH_END_MIN)

        self.bars = df
        self.roll_dates = roll_dates or []
        self.touches: List[LevelTouchEvent] = []
        self.touches_df: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def _build_daily_context(self) -> pd.DataFrame:
        """One row per RTH trading day: OHLC, ATR_D, PDH/PDL, gap, open_loc."""
        rth = self.bars[self.bars["is_rth"]]
        daily = rth.groupby("date").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
        ).reset_index().sort_values("date").reset_index(drop=True)

        daily["atr_d"] = ATRCompute.compute(
            daily["high"].values, daily["low"].values, daily["close"].values,
            period=self.ATR_D_WARMUP,
        )
        daily["pdh"] = daily["high"].shift(1)
        daily["pdl"] = daily["low"].shift(1)
        daily["prior_close"] = daily["close"].shift(1)
        daily["gap"] = (daily["open"] - daily["prior_close"]) / daily["atr_d"]
        daily["open_loc"] = (daily["open"] - daily["pdl"]) / (daily["pdh"] - daily["pdl"])
        return daily

    def _add_overnight_context(self, daily: pd.DataFrame) -> pd.DataFrame:
        """ONH/ONL/overnight-drift from bars between the prior RTH close and
        this day's RTH open."""
        rth = self.bars[self.bars["is_rth"]]
        rth_open_time = rth.groupby("date")["timestamp"].min()
        rth_close_time = rth.groupby("date")["timestamp"].max()

        dates = daily["date"].tolist()
        records = []
        for i, d in enumerate(dates):
            if i == 0 or d not in rth_open_time.index or dates[i - 1] not in rth_close_time.index:
                records.append({"date": d, "onh": np.nan, "onl": np.nan, "overnight_change": np.nan})
                continue
            window_start = rth_close_time[dates[i - 1]]
            window_end = rth_open_time[d]
            on_bars = self.bars[(self.bars["timestamp"] > window_start) & (self.bars["timestamp"] < window_end)]
            if on_bars.empty:
                records.append({"date": d, "onh": np.nan, "onl": np.nan, "overnight_change": np.nan})
            else:
                records.append({
                    "date": d, "onh": on_bars["high"].max(), "onl": on_bars["low"].min(),
                    "overnight_change": on_bars.iloc[-1]["close"] - on_bars.iloc[0]["open"],
                })
        merged = daily.merge(pd.DataFrame(records), on="date", how="left")
        merged["onr"] = (merged["onh"] - merged["onl"]) / merged["atr_d"]
        return merged

    def _compute_atr5_lookup(self) -> pd.DataFrame:
        """14-period ATR on 5-minute RTH bars, exposed for point-in-time lookup."""
        rth = self.bars[self.bars["is_rth"]].set_index("timestamp")
        bars5 = rth.resample("5min").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
        ).dropna().reset_index()
        bars5["atr5"] = ATRCompute.compute(
            bars5["high"].values, bars5["low"].values, bars5["close"].values, period=14
        )
        # A 5-min bar starting at t is only "known" once it closes, at t+5min.
        bars5["available_at"] = bars5["timestamp"] + pd.Timedelta(minutes=5)
        return bars5[["available_at", "atr5"]].dropna()

    def _compute_vwap(self) -> pd.DataFrame:
        rth = self.bars[self.bars["is_rth"]].copy()
        out = []
        for _, g in rth.groupby("date"):
            g = g.sort_values("timestamp")
            vwap = VWAPCompute.compute(g["high"].values, g["low"].values, g["close"].values, g["volume"].values)
            g = g.assign(vwap=vwap)
            g["vwap_lag15"] = g["vwap"].shift(15)
            out.append(g[["timestamp", "vwap", "vwap_lag15"]])
        return pd.concat(out, ignore_index=True)

    def _compute_rv30(self) -> pd.DataFrame:
        """Trailing ~30-minute realized vol of 1-min log returns, reset each RTH day."""
        rth = self.bars[self.bars["is_rth"]].copy()
        out = []
        for _, g in rth.groupby("date"):
            g = g.sort_values("timestamp")
            log_ret = np.log(g["close"] / g["close"].shift(1))
            rv30 = log_ret.rolling(30, min_periods=5).std()
            out.append(pd.DataFrame({"timestamp": g["timestamp"], "rv30": rv30.values}))
        return pd.concat(out, ignore_index=True)

    def _compute_rvol_cum(self) -> pd.DataFrame:
        """Cumulative RTH volume so far today vs. the historical median
        cumulative volume at the same minute-of-session."""
        rth = self.bars[self.bars["is_rth"]].copy()
        out = []
        for _, g in rth.groupby("date"):
            g = g.sort_values("timestamp").reset_index(drop=True)
            g["minute_of_session"] = np.arange(len(g))
            g["cum_vol"] = g["volume"].cumsum()
            out.append(g[["timestamp", "minute_of_session", "cum_vol"]])
        cum_df = pd.concat(out, ignore_index=True)
        baseline = cum_df.groupby("minute_of_session")["cum_vol"].median().rename("baseline_cum_vol")
        cum_df = cum_df.merge(baseline, on="minute_of_session", how="left")
        cum_df["rvol_cum"] = cum_df["cum_vol"] / cum_df["baseline_cum_vol"].replace(0, np.nan)
        return cum_df[["timestamp", "rvol_cum"]]

    def _roll_week_flags(self, dates: pd.Series) -> pd.Series:
        if not self.roll_dates:
            return pd.Series(False, index=dates.index)
        roll_ts = [pd.Timestamp(r).normalize() for r in self.roll_dates]
        return dates.apply(lambda d: any(abs((pd.Timestamp(d) - r).days) <= 3 for r in roll_ts))

    # ------------------------------------------------------------------
    # Touch detection + classification
    # ------------------------------------------------------------------

    def _register_touch(self, name, level_price, side, bar, ctx, day_bars, bar_pos, bar_dicts,
                         atr5, n_tests, rv30_all):
        classifier = TaxonomyClassifier(atr5=atr5, vwap_ref=bar["vwap"], horizon_minutes=self.HORIZON_MINUTES)
        future_bars = bar_dicts[bar_pos:]  # breach bar onward; classifier's own horizon check bounds it
        outcome = classifier.classify(level_price=level_price, bars=future_bars, side=side)

        rv30_val = bar["rv30"]
        event = LevelTouchEvent(
            timestamp=bar["timestamp"], level_name=name, level_price=level_price,
            breach_time=bar["timestamp"], breach_extreme=bar["high"] if side == "long" else bar["low"],
            volreg=ctx["volreg"] if pd.notna(ctx["volreg"]) else "unknown",
            gap=ctx["gap"], open_loc=ctx["open_loc"], onr=ctx["onr"],
            slope_vwap=((bar["vwap"] - bar["vwap_lag15"]) / ctx["atr_d"]
                        if pd.notna(bar["vwap"]) and pd.notna(bar["vwap_lag15"]) else np.nan),
            align=self._is_aligned(side, ctx["gap"], ctx["overnight_change"]),
            rvol_cum=bar["rvol_cum"], t_bucket=bucket_time(bar["timestamp"]), side=side,
            n_tests=n_tests, rv_pct=bucket_rv_pct(rv30_val, rv30_all) if pd.notna(rv30_val) else "unknown",
            news_session=False, roll_week=bool(ctx["roll_week"]),
            outcome=outcome, outcome_details={},
        )
        self.touches.append(event)

    @staticmethod
    def _is_aligned(side: str, gap: float, overnight_change: float) -> bool:
        if pd.isna(gap) or pd.isna(overnight_change) or gap == 0 or overnight_change == 0:
            return False
        want_up = side == "long"
        return (gap > 0) == want_up and (overnight_change > 0) == want_up

    def run(self, test_levels: List[str] = None) -> pd.DataFrame:
        """
        test_levels: which levels to test, e.g. ["PDH", "PDL", "VWAP"]

        Returns: DataFrame of all classified touches with regime variables.
        """
        if test_levels is None:
            test_levels = ["PDH", "PDL", "ONH", "ONL", "VWAP"]

        daily = self._build_daily_context()
        daily = self._add_overnight_context(daily)
        atr_d_valid = daily["atr_d"].dropna()
        daily["volreg"] = daily["atr_d"].apply(lambda v: bucket_volreg(v, atr_d_valid) if pd.notna(v) else None)
        daily["roll_week"] = self._roll_week_flags(pd.to_datetime(daily["date"]))
        daily_idx = daily.set_index("date")

        atr5_lookup = self._compute_atr5_lookup()
        vwap_df = self._compute_vwap()
        rv30_df = self._compute_rv30()
        rvol_df = self._compute_rvol_cum()

        rth = self.bars[self.bars["is_rth"]].sort_values("timestamp").reset_index(drop=True)
        rth = pd.merge_asof(rth, atr5_lookup, left_on="timestamp", right_on="available_at", direction="backward")
        rth = rth.merge(vwap_df, on="timestamp", how="left")
        rth = rth.merge(rv30_df, on="timestamp", how="left")
        rth = rth.merge(rvol_df, on="timestamp", how="left")
        rv30_all = rth["rv30"].dropna()

        self.touches = []
        n_days_used = 0
        for d, day_bars in rth.groupby("date"):
            if d not in daily_idx.index:
                continue
            ctx = daily_idx.loc[d]
            if pd.isna(ctx["atr_d"]) or pd.isna(ctx["pdh"]) or pd.isna(ctx["pdl"]):
                continue  # ATR/prior-day warmup not yet available
            n_days_used += 1

            day_bars = day_bars.sort_values("timestamp").reset_index(drop=True)
            atr5_series = day_bars["atr5"].values
            bar_dicts = day_bars.to_dict("records")

            levels = {}
            if "PDH" in test_levels and pd.notna(ctx["pdh"]):
                levels["PDH"] = (float(ctx["pdh"]), "long")
            if "PDL" in test_levels and pd.notna(ctx["pdl"]):
                levels["PDL"] = (float(ctx["pdl"]), "short")
            if "ONH" in test_levels and pd.notna(ctx["onh"]):
                levels["ONH"] = (float(ctx["onh"]), "long")
            if "ONL" in test_levels and pd.notna(ctx["onl"]):
                levels["ONL"] = (float(ctx["onl"]), "short")

            first_close = day_bars.iloc[0]["close"]
            armed = {name: (first_close < lvl if side == "long" else first_close > lvl)
                     for name, (lvl, side) in levels.items()}
            n_tests_counter = {name: 0 for name in levels}
            n_tests_counter["VWAP"] = 0

            for i in range(len(day_bars)):
                bar = day_bars.iloc[i]
                atr5 = atr5_series[i]
                if pd.isna(atr5) or atr5 <= 0:
                    continue
                b_margin = 0.35 * atr5

                for name, (lvl, side) in levels.items():
                    if side == "long":
                        if armed[name] and bar["high"] > lvl:
                            n_tests_counter[name] += 1
                            self._register_touch(name, lvl, side, bar, ctx, day_bars, i, bar_dicts,
                                                  atr5, n_tests_counter[name], rv30_all)
                            armed[name] = False
                        elif not armed[name] and bar["close"] < lvl - b_margin:
                            armed[name] = True
                    else:
                        if armed[name] and bar["low"] < lvl:
                            n_tests_counter[name] += 1
                            self._register_touch(name, lvl, side, bar, ctx, day_bars, i, bar_dicts,
                                                  atr5, n_tests_counter[name], rv30_all)
                            armed[name] = False
                        elif not armed[name] and bar["close"] > lvl + b_margin:
                            armed[name] = True

                if "VWAP" in test_levels and i > 0 and pd.notna(bar["vwap"]):
                    prev = day_bars.iloc[i - 1]
                    if pd.notna(prev["vwap"]):
                        if prev["close"] < prev["vwap"] and bar["close"] >= bar["vwap"]:
                            n_tests_counter["VWAP"] += 1
                            self._register_touch("VWAP", float(bar["vwap"]), "long", bar, ctx, day_bars, i,
                                                  bar_dicts, atr5, n_tests_counter["VWAP"], rv30_all)
                        elif prev["close"] > prev["vwap"] and bar["close"] <= bar["vwap"]:
                            n_tests_counter["VWAP"] += 1
                            self._register_touch("VWAP", float(bar["vwap"]), "short", bar, ctx, day_bars, i,
                                                  bar_dicts, atr5, n_tests_counter["VWAP"], rv30_all)

        self.touches_df = self._touches_to_dataframe()
        print(f"[Exp1] Classified {len(self.touches_df)} touches across {n_days_used} RTH days, "
              f"testing {test_levels}")
        return self.touches_df

    def _touches_to_dataframe(self) -> pd.DataFrame:
        rows = []
        for t in self.touches:
            rows.append({
                "timestamp": t.timestamp, "level_name": t.level_name, "level_price": t.level_price,
                "outcome": t.outcome.value if t.outcome else None,
                "volreg": t.volreg, "gap": t.gap, "open_loc": t.open_loc, "onr": t.onr,
                "slope_vwap": t.slope_vwap, "align": t.align, "rvol_cum": t.rvol_cum,
                "t_bucket": t.t_bucket, "side": t.side, "n_tests": t.n_tests, "rv_pct": t.rv_pct,
                "news_session": t.news_session, "roll_week": t.roll_week,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _wilson_ci(hits: int, n: int, z: float = 1.96) -> Tuple[float, float]:
        if n == 0:
            return (np.nan, np.nan)
        p = hits / n
        denom = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denom
        half_width = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
        return (max(0.0, center - half_width), min(1.0, center + half_width))

    @staticmethod
    def _two_sided_p_value(hits: int, n: int, p0: float) -> float:
        if n == 0 or p0 <= 0 or p0 >= 1:
            return np.nan
        p_hat = hits / n
        se = math.sqrt(p0 * (1 - p0) / n)
        if se == 0:
            return np.nan
        z = (p_hat - p0) / se
        # two-sided normal p-value via erf
        return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    @staticmethod
    def _benjamini_hochberg(p_values: pd.Series, q: float = 0.05) -> pd.Series:
        valid = p_values.dropna().sort_values()
        m = len(valid)
        if m == 0:
            return pd.Series(False, index=p_values.index)
        thresholds = (np.arange(1, m + 1) / m) * q
        passing = valid.values <= thresholds
        significant = pd.Series(False, index=p_values.index)
        if passing.any():
            cutoff_rank = np.max(np.where(passing)[0])
            cutoff_p = valid.values[cutoff_rank]
            significant = p_values <= cutoff_p
            significant = significant.fillna(False)
        return significant

    def report_by_regime(self, outcome: ReactionOutcome, min_n: int = 30, fdr_q: float = 0.05) -> pd.DataFrame:
        """
        Crosstab: P(outcome | regime variable), across every regime variable.

        Includes n per cell, Wilson-score 95% CIs, and Benjamini-Hochberg
        FDR-corrected significance (vs. the overall base rate of `outcome`)
        pooled across all cells from all regime variables. Cells with
        n < min_n are dropped before FDR correction (per-spec: "report only
        cells with n >= 30").
        """
        if self.touches_df.empty:
            print("[Exp1] No touches to report on - call run() first.")
            return pd.DataFrame()

        df = self.touches_df.copy()
        df["hit"] = (df["outcome"] == outcome.value).astype(int)
        base_rate = df["hit"].mean()

        continuous_vars = ["gap", "open_loc", "onr", "slope_vwap", "rvol_cum"]
        bucketed_vars = []
        for col in continuous_vars:
            try:
                df[f"{col}_bucket"] = pd.qcut(df[col], q=3, labels=["low", "mid", "high"], duplicates="drop")
                bucketed_vars.append(f"{col}_bucket")
            except (ValueError, IndexError):
                pass  # too many duplicate values to form 3 distinct buckets; skip this variable

        categorical_vars = ["level_name", "volreg", "t_bucket", "side", "rv_pct", "align",
                             "roll_week", "news_session"] + bucketed_vars
        categorical_vars.append("n_tests_bucket")
        df["n_tests_bucket"] = np.where(df["n_tests"] <= 1, "first", "repeat")

        rows = []
        for var in categorical_vars:
            for bucket, g in df.groupby(var, observed=True):
                n = len(g)
                if n < min_n:
                    continue
                hits = g["hit"].sum()
                p_hat = hits / n
                ci_low, ci_high = self._wilson_ci(hits, n)
                p_value = self._two_sided_p_value(hits, n, base_rate)
                rows.append({
                    "regime_var": var, "bucket": bucket, "n": n, "hits": hits,
                    "p_hat": p_hat, "base_rate": base_rate, "ci_low": ci_low, "ci_high": ci_high,
                    "p_value": p_value,
                })

        report = pd.DataFrame(rows)
        if report.empty:
            print(f"[Exp1] No regime cells with n >= {min_n} for {outcome.value}.")
            return report

        report["significant_fdr"] = self._benjamini_hochberg(report["p_value"], q=fdr_q)
        report = report.sort_values("p_value").reset_index(drop=True)
        return report


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Experiment 1: Reaction Taxonomy Study")
    print("=" * 70)
    print("""
This study classifies every level touch into one of five outcomes:
    1. CONTINUATION (break and follow-through)
    2. RETEST (break, pullback, then re-extension)
    3. REJECTION (false break and reversal)
    4. MEAN_REVERSION (pullback to reference, no opposite extreme)
    5. NO_REACTION (neither threshold met)

Output: P(outcome | regime) with confidence intervals.
""")

    data_dir = Path.home() / "mes_research" / "data"
    bars_path = data_dir / "mes_continuous_1min.parquet"
    roll_log_path = data_dir / "roll_log.csv"

    bars = pd.read_parquet(bars_path)
    print(f"Loaded {len(bars)} bars from {bars_path}")
    print(f"Range: {bars['timestamp'].min()} to {bars['timestamp'].max()}\n")

    roll_dates = []
    if roll_log_path.exists():
        roll_log = pd.read_csv(roll_log_path, parse_dates=["splice_time"])
        roll_dates = roll_log["splice_time"].tolist()

    study = TaxonomyStudy(bars, roll_dates=roll_dates)
    touches = study.run()

    print("\nOutcome distribution (all levels, all regimes):")
    print(touches["outcome"].value_counts().to_string())

    print("\nOutcome distribution by level:")
    print(pd.crosstab(touches["level_name"], touches["outcome"]).to_string())

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)
    touches.to_csv(out_dir / "exp1_touches.csv", index=False)
    print(f"\nSaved {len(touches)} touches to {out_dir / 'exp1_touches.csv'}")

    print("\n" + "=" * 70)
    print("Regime reports (n >= 30, sorted by significance, FDR q=0.05)")
    print("=" * 70)
    for outcome in ReactionOutcome:
        report = study.report_by_regime(outcome)
        if report.empty:
            continue
        print(f"\n--- P({outcome.value} | regime) ---")
        display_cols = ["regime_var", "bucket", "n", "p_hat", "base_rate", "ci_low", "ci_high",
                         "p_value", "significant_fdr"]
        print(report[display_cols].head(15).to_string(index=False))
        report.to_csv(out_dir / f"exp1_regime_report_{outcome.value}.csv", index=False)

    print("\n" + "=" * 70)
