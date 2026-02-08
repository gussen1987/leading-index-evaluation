"""Regime-based backtesting module for SPY."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import numpy as np

from risk_index.core.logger import get_logger
from risk_index.core.types import Regime
from risk_index.core.constants import (
    CACHE_DIR,
    COL_REGIME_FAST,
    COL_REGIME_MEDIUM,
    COL_REGIME_SLOW,
    WEEK_ANCHOR,
)
from risk_index.core.utils_io import read_parquet

logger = get_logger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    regime_at_entry: str

    @property
    def return_pct(self) -> float:
        """Return percentage."""
        return (self.exit_price / self.entry_price - 1) * 100

    @property
    def duration_weeks(self) -> int:
        """Trade duration in weeks."""
        return int((self.exit_date - self.entry_date).days / 7)

    @property
    def is_winner(self) -> bool:
        """True if trade was profitable."""
        return self.exit_price > self.entry_price


def load_spy_prices(align_to_weekly: bool = True) -> pd.Series:
    """Load SPY price data from cache.

    Args:
        align_to_weekly: If True, resample to weekly (Friday close)

    Returns:
        Series of SPY adjusted close prices
    """
    spy_path = CACHE_DIR / "yahoo_SPY.parquet"
    if not spy_path.exists():
        raise FileNotFoundError(f"SPY data not found at {spy_path}")

    df = read_parquet(spy_path)

    # Handle different column formats
    if "Adj Close" in df.columns:
        prices = df["Adj Close"]
    elif "Close" in df.columns:
        prices = df["Close"]
    elif "SPY" in df.columns:
        prices = df["SPY"]
    elif len(df.columns) == 1:
        # Single column DataFrame - use that column
        prices = df.iloc[:, 0]
    else:
        raise ValueError(f"No price column found in SPY data. Columns: {list(df.columns)}")

    prices.name = "SPY"

    if align_to_weekly:
        # Resample to weekly Friday close
        prices = prices.resample(WEEK_ANCHOR).last()

    return prices.dropna()


def backtest_spy_regime(
    regimes_df: pd.DataFrame,
    spy_prices: pd.Series | None = None,
    regime_column: str = COL_REGIME_MEDIUM,
    strategy: Literal["long_risk_on", "avoid_risk_off"] = "long_risk_on",
    neutral_action: Literal["hold", "exit"] = "hold",
) -> dict:
    """Backtest SPY based on regime signals.

    Strategies:
    - long_risk_on: Hold SPY only during Risk-On (green) periods
    - avoid_risk_off: Hold SPY except during Risk-Off (red) periods

    Args:
        regimes_df: DataFrame with regime columns (DatetimeIndex)
        spy_prices: SPY price series (if None, will load from cache)
        regime_column: Which regime column to use (regime_fast, regime_medium, regime_slow)
        strategy: Trading strategy
        neutral_action: What to do during Neutral periods ("hold" or "exit")

    Returns:
        Dict containing:
            - total_return: Total cumulative return (%)
            - cagr: Compound annual growth rate (%)
            - num_trades: Number of round-trip trades
            - win_rate: Percentage of winning trades
            - avg_trade_return: Average return per trade (%)
            - max_drawdown: Maximum drawdown (%)
            - sharpe_ratio: Annualized Sharpe ratio
            - trades: DataFrame of individual trades
            - equity_curve: Series of portfolio value over time
    """
    if spy_prices is None:
        spy_prices = load_spy_prices(align_to_weekly=True)

    if regime_column not in regimes_df.columns:
        raise ValueError(f"Regime column '{regime_column}' not found in DataFrame")

    # Align data
    common_idx = regimes_df.index.intersection(spy_prices.index)
    if len(common_idx) < 10:
        raise ValueError("Insufficient overlapping data between regimes and SPY prices")

    regimes = regimes_df.loc[common_idx, regime_column]
    prices = spy_prices.loc[common_idx]

    logger.info(f"Backtesting {regime_column} with {len(common_idx)} weeks of data")

    # Determine position signals based on strategy
    if strategy == "long_risk_on":
        # Long only during Risk-On
        if neutral_action == "hold":
            position_signal = regimes == Regime.RISK_ON.value
        else:
            position_signal = regimes == Regime.RISK_ON.value
    else:  # avoid_risk_off
        # Long except during Risk-Off
        if neutral_action == "hold":
            position_signal = regimes != Regime.RISK_OFF.value
        else:
            position_signal = regimes == Regime.RISK_ON.value

    # Calculate returns and equity curve
    weekly_returns = prices.pct_change()
    strategy_returns = weekly_returns * position_signal.shift(1)  # Shift to avoid lookahead
    strategy_returns = strategy_returns.infer_objects(copy=False).fillna(0)

    # Build equity curve (starting at 100)
    equity_curve = (1 + strategy_returns).cumprod() * 100
    equity_curve.name = f"Strategy ({regime_column})"

    # Calculate buy-and-hold equity
    bh_returns = weekly_returns.fillna(0)
    bh_equity = (1 + bh_returns).cumprod() * 100

    # Extract individual trades
    trades = _extract_trades(position_signal, prices, regimes)

    # Calculate metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

    # CAGR
    years = (common_idx[-1] - common_idx[0]).days / 365.25
    cagr = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Max drawdown
    rolling_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - rolling_max) / rolling_max * 100
    max_drawdown = drawdowns.min()

    # Trade statistics
    num_trades = len(trades)
    if num_trades > 0:
        win_rate = sum(1 for t in trades if t.is_winner) / num_trades * 100
        avg_trade_return = np.mean([t.return_pct for t in trades])
    else:
        win_rate = 0
        avg_trade_return = 0

    # Sharpe ratio (annualized, assuming 52 weeks per year)
    if strategy_returns.std() > 0:
        sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(52)
    else:
        sharpe_ratio = 0

    # Buy-and-hold metrics for comparison
    bh_total_return = (bh_equity.iloc[-1] / bh_equity.iloc[0] - 1) * 100
    bh_cagr = ((bh_equity.iloc[-1] / bh_equity.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0
    bh_rolling_max = bh_equity.expanding().max()
    bh_drawdowns = (bh_equity - bh_rolling_max) / bh_rolling_max * 100
    bh_max_drawdown = bh_drawdowns.min()

    # Create trades DataFrame
    trades_df = pd.DataFrame([
        {
            "Entry Date": t.entry_date,
            "Exit Date": t.exit_date,
            "Entry Price": t.entry_price,
            "Exit Price": t.exit_price,
            "Return (%)": t.return_pct,
            "Duration (weeks)": t.duration_weeks,
            "Regime": t.regime_at_entry,
        }
        for t in trades
    ])

    result = {
        "total_return": total_return,
        "cagr": cagr,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "equity_curve": equity_curve,
        "buy_hold_equity": bh_equity,
        "buy_hold_return": bh_total_return,
        "buy_hold_cagr": bh_cagr,
        "buy_hold_max_dd": bh_max_drawdown,
        "trades": trades_df,
        "position_signal": position_signal,
        "start_date": common_idx[0],
        "end_date": common_idx[-1],
    }

    logger.info(
        f"Backtest complete: Total Return={total_return:.1f}%, "
        f"CAGR={cagr:.1f}%, Trades={num_trades}, Win Rate={win_rate:.1f}%"
    )

    return result


def _extract_trades(
    position_signal: pd.Series,
    prices: pd.Series,
    regimes: pd.Series,
) -> list[Trade]:
    """Extract individual trades from position signals.

    Args:
        position_signal: Boolean series indicating when to be long
        prices: Price series
        regimes: Regime series

    Returns:
        List of Trade objects
    """
    trades = []
    in_trade = False
    entry_date = None
    entry_price = None
    entry_regime = None

    for i, (date, is_long) in enumerate(position_signal.items()):
        if is_long and not in_trade:
            # Enter trade
            in_trade = True
            entry_date = date
            entry_price = prices.loc[date]
            entry_regime = regimes.loc[date]
        elif not is_long and in_trade:
            # Exit trade
            exit_date = date
            exit_price = prices.loc[date]
            trades.append(Trade(
                entry_date=entry_date,
                exit_date=exit_date,
                entry_price=entry_price,
                exit_price=exit_price,
                regime_at_entry=entry_regime,
            ))
            in_trade = False

    # Close any open trade at end
    if in_trade:
        exit_date = position_signal.index[-1]
        exit_price = prices.iloc[-1]
        trades.append(Trade(
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            regime_at_entry=entry_regime,
        ))

    return trades


def run_all_regime_backtests(
    regimes_df: pd.DataFrame,
    spy_prices: pd.Series | None = None,
    strategy: Literal["long_risk_on", "avoid_risk_off"] = "long_risk_on",
) -> dict[str, dict]:
    """Run backtests for all regime speeds.

    Args:
        regimes_df: DataFrame with regime columns
        spy_prices: SPY price series (if None, will load from cache)
        strategy: Trading strategy

    Returns:
        Dict mapping speed name to backtest results
    """
    if spy_prices is None:
        spy_prices = load_spy_prices(align_to_weekly=True)

    results = {}

    for name, col in [
        ("Fast", COL_REGIME_FAST),
        ("Medium", COL_REGIME_MEDIUM),
        ("Slow", COL_REGIME_SLOW),
    ]:
        if col in regimes_df.columns:
            try:
                results[name] = backtest_spy_regime(
                    regimes_df,
                    spy_prices,
                    regime_column=col,
                    strategy=strategy,
                )
            except Exception as e:
                logger.warning(f"Backtest failed for {name}: {e}")
                results[name] = None

    return results


def create_backtest_summary(results: dict[str, dict]) -> pd.DataFrame:
    """Create summary DataFrame from backtest results.

    Args:
        results: Dict from run_all_regime_backtests

    Returns:
        DataFrame with metrics as rows and speeds as columns
    """
    metrics = [
        ("Total Return (%)", "total_return"),
        ("CAGR (%)", "cagr"),
        ("Max Drawdown (%)", "max_drawdown"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("# Trades", "num_trades"),
        ("Win Rate (%)", "win_rate"),
        ("Avg Trade Return (%)", "avg_trade_return"),
    ]

    data = {}

    for speed, res in results.items():
        if res is not None:
            data[speed] = {label: res.get(key, 0) for label, key in metrics}

    # Add Buy & Hold column
    if results:
        first_valid = next((r for r in results.values() if r is not None), None)
        if first_valid:
            data["Buy & Hold"] = {
                "Total Return (%)": first_valid["buy_hold_return"],
                "CAGR (%)": first_valid["buy_hold_cagr"],
                "Max Drawdown (%)": first_valid["buy_hold_max_dd"],
                "Sharpe Ratio": "-",
                "# Trades": 1,
                "Win Rate (%)": "-",
                "Avg Trade Return (%)": "-",
            }

    df = pd.DataFrame(data)
    return df
