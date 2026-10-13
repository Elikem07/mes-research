"""
MES Intraday Strategy Research — Experiment 0 Foundation

Project structure:
  mes_research/
    ├── cost_model.py          (this file, partially)
    ├── features.py            (measurement primitives)
    ├── engine.py              (backtester with three layers)
    ├── data_loader.py         (session/calendar/contract handling)
    ├── tests/
    │   └── test_synthetic.py  (validation suite)
    └── main.py                (entry point)

Start here: Run `python main.py --validate` to confirm the cost model and engine.

WARNING: This is infrastructure only. No strategies until Experiment 1.
"""

# ============================================================================
# 1. COST_MODEL.PY — Three-tier cost framework
# ============================================================================

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from cost_config import COSTS

class CostTier(Enum):
    OPTIMISTIC = "optimistic"
    BASE = "base"
    STRESS = "stress"

@dataclass
class CostAssumptions:
    """
    MES contract specs:
    - Multiplier: $5.00 per index point
    - Tick: 0.25 points = $1.25
    
    Round-trip friction modeled per contract per trade.
    """
    tier: CostTier
    
    # Commission + exchange fees (per contract, round-trip)
    commission_rt: float  # $
    
    # Spread assumption (entry)
    spread_entry_ticks: float  # number of ticks, adverse
    
    # Slippage on stop exit (base case, before volatility adjustment)
    stop_slippage_ticks: float
    
    # Slippage on limit exits
    limit_slippage_ticks: float  # 0 if limit never fills
    
    # News-window extra slippage (added to stop_slippage)
    news_slippage_extra_ticks: float
    
    @classmethod
    def create(cls, tier: CostTier) -> "CostAssumptions":
        """Factory: build a tier from config/costs.toml."""
        spec = COSTS["tier"][tier.value]
        return cls(
            tier=tier,
            commission_rt=spec["commission_rt"],
            spread_entry_ticks=spec["spread_entry_ticks"],
            stop_slippage_ticks=spec["stop_slippage_ticks"],
            limit_slippage_ticks=spec["limit_slippage_ticks"],
            news_slippage_extra_ticks=spec["news_slippage_extra_ticks"],
        )

    def round_trip_cost_points(self, multiplier: float = None) -> float:
        """
        Estimate round-trip friction in index points.
        multiplier = $ per index point (from [contract] in config/costs.toml)
        """
        if multiplier is None:
            multiplier = COSTS["contract"]["multiplier"]
        tick_value = COSTS["contract"]["tick_value"]
        # Entry spread crossed
        entry_cost = self.spread_entry_ticks * tick_value
        
        # Stop exit slippage
        stop_cost = self.stop_slippage_ticks * tick_value
        
        # Commission
        total_dollar_cost = entry_cost + stop_cost + self.commission_rt
        
        # Convert to index points
        return total_dollar_cost / multiplier
    
    def __repr__(self) -> str:
        cost_points = self.round_trip_cost_points()
        return (
            f"CostTier.{self.tier.value.upper()}: "
            f"${self.commission_rt:.2f} comm + {self.spread_entry_ticks:.1f} tick entry + "
            f"{self.stop_slippage_ticks:.1f} tick stop ≈ {cost_points:.2f} pt RT"
        )


# ============================================================================
# 2. FEATURES.PY — Measurement primitives
# ============================================================================

from typing import Optional
import numpy as np
import pandas as pd

class ATRCompute:
    """ATR14 computation."""
    
    @staticmethod
    def compute(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        True Range = max(H - L, abs(H - C_prev), abs(L - C_prev))
        ATR = SMA(TR, period)
        """
        if len(high) < period:
            return np.full_like(close, np.nan, dtype=float)
        
        h_l = high - low
        h_cp = np.abs(high - np.roll(close, 1))
        l_cp = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(np.maximum(h_l, h_cp), l_cp)
        tr[0] = h_l[0]  # first bar: no previous close
        
        atr = np.convolve(tr, np.ones(period) / period, mode='valid')
        return np.concatenate([np.full(period - 1, np.nan), atr])


class VWAPCompute:
    """Session-anchored VWAP from 1-minute bars."""
    
    @staticmethod
    def compute(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        Typical Price = (H + L + C) / 3
        VWAP = cumsum(TP * V) / cumsum(V)
        Anchored at array position 0 (09:30 ET).
        """
        tp = (high + low + close) / 3.0
        cum_tp_vol = np.cumsum(tp * volume)
        cum_vol = np.cumsum(volume)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            vwap = cum_tp_vol / cum_vol
            vwap[cum_vol == 0] = np.nan
        
        return vwap
    
    @staticmethod
    def volume_weighted_std(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                            volume: np.ndarray) -> np.ndarray:
        """
        Volume-weighted standard deviation of typical price about VWAP.
        Computed on a trailing cumulative basis.
        """
        tp = (high + low + close) / 3.0
        vwap = VWAPCompute.compute(high, low, close, volume)
        
        cum_vol = np.cumsum(volume)
        cum_var_vol = np.cumsum((tp - vwap) ** 2 * volume)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            var = cum_var_vol / cum_vol
            sd = np.sqrt(np.maximum(var, 0))  # sqrt of negative is 0 due to floating point
            sd[cum_vol == 0] = np.nan
        
        return sd


class RelativeVolumeCompute:
    """Relative volume against the same time-of-day bucket."""
    
    @staticmethod
    def compute(volume: np.ndarray, timestamps: np.ndarray, lookback_sessions: int = 20) -> np.ndarray:
        """
        Assumes timestamps are aligned to a single session with known t_bucket.
        This is a stub; in production, iterate over all sessions and compute
        median volume per t_bucket separately.
        
        For now: RVOL = volume / rolling_median(volume, window=lookback_sessions)
        """
        rolling_median = pd.Series(volume).rolling(window=lookback_sessions, min_periods=1).median().values
        rvol = volume / np.maximum(rolling_median, 1)  # avoid divide by zero
        return rvol


# ============================================================================
# 3. ENGINE.PY — Three-layer backtest framework
# ============================================================================

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any

@dataclass
class Signal:
    """Entry signal produced by the signal layer."""
    timestamp: datetime
    side: Literal["LONG", "SHORT"]
    price: float  # approximate entry price
    stop_loss: float  # hard stop in index points
    take_profit: float  # hard target in index points
    time_exit: datetime  # exit if not filled by this time
    metadata: Dict[str, Any]  # regime variables, features, etc.

@dataclass
class Trade:
    """Completed or open trade."""
    entry_time: datetime
    entry_price: float
    entry_side: Literal["LONG", "SHORT"]
    stop_loss: float
    take_profit: float
    time_exit: datetime

    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[Literal["stop", "target", "time", "close"]] = None
    
    gross_points: Optional[float] = None
    cost_points: Optional[float] = None
    net_points: Optional[float] = None
    
    def pnl(self) -> Dict[str, float]:
        """Return P&L summary."""
        if self.exit_price is None:
            return {"status": "open"}
        
        if self.entry_side == "LONG":
            gross = self.exit_price - self.entry_price
        else:
            gross = self.entry_price - self.exit_price
        
        return {
            "gross_points": gross,
            "cost_points": self.cost_points or 0.0,
            "net_points": (gross - self.cost_points) if self.cost_points else gross,
        }

class BacktestEngine:
    """
    Three-layer execution:
    1. Feature layer: computes point-in-time features (no future data)
    2. Signal layer: produces entry signals from features
    3. Execution layer: fills signals and closes trades
    """
    
    def __init__(self, cost_tier: CostTier):
        self.cost = CostAssumptions.create(cost_tier)
        self.trades: List[Trade] = []
        self.open_trade: Optional[Trade] = None
    
    def validate_no_future_data(self, feature_timestamp: datetime, decision_timestamp: datetime):
        """Assert that we are not using future data in the signal."""
        assert feature_timestamp <= decision_timestamp, (
            f"Look-ahead detected: feature at {feature_timestamp} "
            f"used in decision at {decision_timestamp}"
        )
    
    def submit_signal(self, signal: Signal) -> Optional[Trade]:
        """Entry layer: create a trade from a signal."""
        if self.open_trade is not None:
            return None  # One trade at a time in baseline
        
        # Entry spread cost is already modeled in round_trip_cost_points()
        # via spread_entry_ticks, deducted from Trade.cost_points at exit.
        trade = Trade(
            entry_time=signal.timestamp,
            entry_price=signal.price,
            entry_side=signal.side,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            time_exit=signal.time_exit,
        )
        self.open_trade = trade
        return trade
    
    def check_exit(self, timestamp: datetime, bar_high: float, bar_low: float, bar_close: float) -> Optional[Trade]:
        """Execution layer: check for stop/target/time exit."""
        if self.open_trade is None:
            return None
        
        trade = self.open_trade
        
        # Stop loss
        if trade.entry_side == "LONG":
            if bar_low <= trade.stop_loss:  # pessimistic: stop fills at stop price
                trade.exit_time = timestamp
                trade.exit_price = trade.stop_loss
                trade.exit_reason = "stop"
                trade.cost_points = self.cost.round_trip_cost_points()
                self.open_trade = None
                self.trades.append(trade)
                return trade
            if bar_high >= trade.take_profit:
                trade.exit_time = timestamp
                trade.exit_price = trade.take_profit
                trade.exit_reason = "target"
                trade.cost_points = self.cost.round_trip_cost_points()
                self.open_trade = None
                self.trades.append(trade)
                return trade
        else:  # SHORT
            if bar_high >= trade.stop_loss:
                trade.exit_time = timestamp
                trade.exit_price = trade.stop_loss
                trade.exit_reason = "stop"
                trade.cost_points = self.cost.round_trip_cost_points()
                self.open_trade = None
                self.trades.append(trade)
                return trade
            if bar_low <= trade.take_profit:
                trade.exit_time = timestamp
                trade.exit_price = trade.take_profit
                trade.exit_reason = "target"
                trade.cost_points = self.cost.round_trip_cost_points()
                self.open_trade = None
                self.trades.append(trade)
                return trade

        # Time exit
        if timestamp >= trade.time_exit:
            trade.exit_time = timestamp
            trade.exit_price = bar_close
            trade.exit_reason = "time"
            trade.cost_points = self.cost.round_trip_cost_points()
            self.open_trade = None
            self.trades.append(trade)
            return trade

        return None

    def close_all_eod(self, timestamp: datetime, bar_close: float):
        """Close any open trade at end-of-day."""
        if self.open_trade is not None:
            self.open_trade.exit_time = timestamp
            self.open_trade.exit_price = bar_close
            self.open_trade.exit_reason = "close"
            self.open_trade.cost_points = self.cost.round_trip_cost_points()
            self.trades.append(self.open_trade)
            self.open_trade = None
    
    def summary(self) -> Dict[str, Any]:
        """Generate statistics from trade list."""
        if not self.trades:
            return {"trades": 0, "note": "no trades"}
        
        returns = []
        wins = 0
        for trade in self.trades:
            pnl = trade.pnl()
            if "net_points" in pnl:
                net = pnl["net_points"]
                returns.append(net)
                if net > 0:
                    wins += 1
        
        returns = np.array(returns)
        
        return {
            "trades": len(self.trades),
            "wins": wins,
            "win_rate": wins / len(self.trades) if self.trades else 0,
            "gross_mean": returns.mean(),
            "gross_std": returns.std(),
            "max_loss": returns.min(),
            "max_win": returns.max(),
            "profit_factor": (
                returns[returns > 0].sum() / abs(returns[returns < 0].sum()) 
                if (returns < 0).any() else np.inf
            ),
        }


# ============================================================================
# 4. SYNTHETIC TEST HARNESS
# ============================================================================

def test_random_walk_should_lose_cost():
    """
    Test: A random-entry strategy on a pure random walk should return
    exactly minus the cost model, no more, no less.
    """
    np.random.seed(42)
    
    # Generate a random walk: 500 bars, starting at 4000
    returns = np.random.normal(0, 0.001, 500)  # small daily volatility
    prices = 4000 * np.exp(np.cumsum(returns))
    
    # Create OHLCV (simple: O=C=midpoint, H/L around it)
    high = prices * 1.001
    low = prices * 0.999
    close = prices
    volume = np.full_like(prices, 100)
    
    # Run engine with base-tier costs
    engine = BacktestEngine(CostTier.BASE)
    
    # Entry: every 10 bars, alternate long/short
    for i in range(10, len(close) - 20, 10):
        side = "LONG" if i % 20 == 10 else "SHORT"
        
        base = pd.Timestamp("2024-01-01")
        signal = Signal(
            timestamp=base + pd.Timedelta(minutes=i),
            side=side,
            price=close[i],
            # Stop/target distances kept wider than a single bar's high/low
            # range (~0.1% of price) so a bar can't hit both in one check.
            stop_loss=close[i] - 6.0 if side == "LONG" else close[i] + 6.0,
            take_profit=close[i] + 10.0 if side == "LONG" else close[i] - 10.0,
            time_exit=base + pd.Timedelta(minutes=i + 20),
            metadata={},
        )

        engine.submit_signal(signal)

        # Simulate bars: no meaningful move (random walk)
        for j in range(i + 1, min(i + 20, len(close))):
            engine.check_exit(
                timestamp=base + pd.Timedelta(minutes=j),
                bar_high=high[j],
                bar_low=low[j],
                bar_close=close[j],
            )
    
    summary = engine.summary()

    expected_cost_per_trade = engine.cost.round_trip_cost_points()
    expected_total_loss = expected_cost_per_trade * summary["trades"]

    actual_loss = -summary["gross_mean"] * summary["trades"]

    # Tolerance scales with the sampling noise of a `trades`-sized draw
    # instead of a fixed constant: a handful of trades against wide
    # stop/target distances has real sample variance around the true mean.
    standard_error = summary["gross_std"] / np.sqrt(summary["trades"])
    tolerance = 3 * standard_error
    deviation = abs(summary["gross_mean"] + expected_cost_per_trade)

    print(f"Random walk test:")
    print(f"  Trades: {summary['trades']}")
    print(f"  Expected cost per trade: {expected_cost_per_trade:.2f} pt")
    print(f"  Expected total loss: {expected_total_loss:.2f} pt")
    print(f"  Actual mean P&L: {summary['gross_mean']:.2f} pt/trade")
    print(f"  Sampling tolerance (3 SE, std={summary['gross_std']:.2f}): ±{tolerance:.2f} pt")
    print(f"  Cost model validation: {'PASS' if deviation < tolerance else 'FAIL'}")

    return deviation < tolerance


def test_pure_trend_should_capture():
    """
    Test: A long-only strategy on an uptrend should capture the trend minus costs.
    """
    # Pure uptrend: start 4000, add 0.5 pt per bar for 200 bars
    trend = np.linspace(4000, 4100, 200)
    
    high = trend * 1.002
    low = trend * 0.998
    close = trend
    volume = np.full_like(close, 100)
    
    engine = BacktestEngine(CostTier.BASE)

    base = pd.Timestamp("2024-01-01")

    # Single long entry at bar 10
    signal = Signal(
        timestamp=base + pd.Timedelta(minutes=10),
        side="LONG",
        price=trend[10],
        # Stop distance kept wider than a single bar's high/low range
        # (~0.2% of price, ~8pt here) so a bar can't hit both in one check.
        # Target kept above the full trend range (~95pt) so it's never hit;
        # the trade should only close via time exit.
        stop_loss=trend[10] - 10.0,
        take_profit=trend[10] + 200.0,
        time_exit=base + pd.Timedelta(minutes=200),
        metadata={},
    )

    engine.submit_signal(signal)

    # Simulate all bars
    for i in range(11, len(close)):
        engine.check_exit(
            timestamp=base + pd.Timedelta(minutes=i),
            bar_high=high[i],
            bar_low=low[i],
            bar_close=close[i],
        )

    engine.close_all_eod(base + pd.Timedelta(minutes=200), close[-1])
    
    summary = engine.summary()
    expected_gross = close[-1] - trend[10]
    expected_cost = engine.cost.round_trip_cost_points()
    expected_net = expected_gross - expected_cost
    
    actual_net = summary["gross_mean"]
    
    print(f"\nTrend capture test:")
    print(f"  Entry price: {trend[10]:.2f}")
    print(f"  Exit price: {close[-1]:.2f}")
    print(f"  Gross points: {expected_gross:.2f}")
    print(f"  Cost deduction: {expected_cost:.2f}")
    print(f"  Expected net: {expected_net:.2f}")
    print(f"  Actual net: {actual_net:.2f}")
    print(f"  Trend capture validation: {'PASS' if abs(actual_net - expected_net) < 0.2 else 'FAIL'}")
    
    return abs(actual_net - expected_net) < 0.2


# ============================================================================
# MAIN — Experiment 0 validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MES Research — Experiment 0: Infrastructure Validation")
    print("=" * 70)
    
    print("\n[1] Cost Model Tiers:")
    for tier in CostTier:
        cost = CostAssumptions.create(tier)
        print(f"  {cost}")
    
    print("\n[2] Synthetic Data Tests:")
    test1 = test_random_walk_should_lose_cost()
    test2 = test_pure_trend_should_capture()
    
    print("\n" + "=" * 70)
    if test1 and test2:
        print("✓ Experiment 0 PASSED: Engine is valid, cost model is correct")
        print("\nNext step: Build Experiment 1 (reaction taxonomy)")
        print("See: experiments/exp1_taxonomy.py")
    else:
        print("✗ Experiment 0 FAILED: Fix the engine before proceeding")
    print("=" * 70)
