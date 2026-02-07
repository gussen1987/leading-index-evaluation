"""Mathematical utilities and transforms for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Callable

from risk_index.core.constants import WINSORIZE_LOWER, WINSORIZE_UPPER


def zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score normalization.


    Args:
        s: Input series
        window: Rolling window size

    Returns:
        Z-score normalized series
    """
    rolling_mean = s.rolling(window=window, min_periods=max(1, window // 2)).mean()
    rolling_std = s.rolling(window=window, min_periods=max(1, window // 2)).std()
    return (s - rolling_mean) / rolling_std.replace(0, np.nan)


def percentile(s: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank [0, 1].

    Args:
        s: Input series
        window: Rolling window size

    Returns:
        Percentile rank series
    """

    def pct_rank(x):
        if len(x.dropna()) < 2:
            return np.nan
        return stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100.0

    return s.rolling(window=window, min_periods=max(1, window // 2)).apply(
        pct_rank, raw=False
    )


def roc(s: pd.Series, window: int) -> pd.Series:
    """Rate of change (percentage).

    Args:
        s: Input series
        window: Lookback period

    Returns:
        Rate of change series
    """
    return s.pct_change(periods=window)


def slope(s: pd.Series, window: int) -> pd.Series:
    """Standardized OLS slope (trend strength).

    Args:
        s: Input series
        window: Rolling window size

    Returns:
        Standardized slope series
    """

    def calc_slope(y):
        # y is a numpy array when raw=True
        mask = ~np.isnan(y)
        valid_count = mask.sum()
        if valid_count < max(2, window // 2):
            return np.nan
        x = np.arange(len(y))
        try:
            slope_val, _, _, _, _ = stats.linregress(x[mask], y[mask])
            # Standardize by series std
            std = np.nanstd(y)
            if std > 0:
                return slope_val * window / std
            return np.nan
        except Exception:
            return np.nan

    return s.rolling(window=window, min_periods=max(2, window // 2)).apply(
        calc_slope, raw=True
    )


def drawdown_from_high(s: pd.Series, window: int) -> pd.Series:
    """Drawdown from rolling high (always <= 0).

    Args:
        s: Input series (prices)
        window: Rolling window for high

    Returns:
        Drawdown series
    """
    rolling_max = s.rolling(window=window, min_periods=1).max()
    return (s - rolling_max) / rolling_max


def realized_vol(s: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    """Realized volatility (annualized by default).

    Args:
        s: Input series (prices)
        window: Rolling window in trading days
        annualize: Whether to annualize (252 trading days)

    Returns:
        Volatility series
    """
    log_returns = np.log(s / s.shift(1))
    vol = log_returns.rolling(window=window, min_periods=max(1, window // 2)).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index.

    Args:
        s: Input series (prices)
        window: RSI period

    Returns:
        RSI series [0, 100]
    """
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def moving_average(s: pd.Series, window: int) -> pd.Series:
    """Simple moving average.

    Args:
        s: Input series
        window: MA period

    Returns:
        Moving average series
    """
    return s.rolling(window=window, min_periods=max(1, window // 2)).mean()


def ema(s: pd.Series, window: int) -> pd.Series:
    """Exponential moving average.

    Args:
        s: Input series
        window: EMA span

    Returns:
        EMA series
    """
    return s.ewm(span=window, min_periods=max(1, window // 2)).mean()


def above_ma(s: pd.Series, window: int) -> pd.Series:
    """Binary indicator: series above its moving average.

    Args:
        s: Input series
        window: MA period

    Returns:
        Binary series (1.0 if above, 0.0 if below/at)
    """
    ma = moving_average(s, window)
    return (s > ma).astype(float)


def ma_slope(s: pd.Series, window: int, slope_window: int = 13) -> pd.Series:
    """Slope of moving average.

    Args:
        s: Input series
        window: MA period
        slope_window: Window for slope calculation

    Returns:
        MA slope series
    """
    ma = moving_average(s, window)
    return slope(ma, slope_window)


def winsorize(s: pd.Series, lower: float = WINSORIZE_LOWER, upper: float = WINSORIZE_UPPER) -> pd.Series:
    """Winsorize series at specified percentiles.

    Args:
        s: Input series
        lower: Lower percentile (e.g., 0.005 for 0.5th)
        upper: Upper percentile (e.g., 0.995 for 99.5th)

    Returns:
        Winsorized series
    """
    lower_val = s.quantile(lower)
    upper_val = s.quantile(upper)
    return s.clip(lower=lower_val, upper=upper_val)


def replace_inf(s: pd.Series) -> pd.Series:
    """Replace infinite values with NaN.

    Args:
        s: Input series

    Returns:
        Series with inf replaced by NaN
    """
    return s.replace([np.inf, -np.inf], np.nan)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safe division handling zeros and infinities.

    Args:
        numerator: Numerator series
        denominator: Denominator series

    Returns:
        Division result with inf/nan handled
    """
    result = numerator / denominator.replace(0, np.nan)
    return replace_inf(result)


def compute_ratio(s1: pd.Series, s2: pd.Series, invert: bool = False) -> pd.Series:
    """Compute ratio of two series.

    Args:
        s1: Numerator series
        s2: Denominator series
        invert: If True, compute s2/s1 instead

    Returns:
        Ratio series
    """
    if invert:
        return safe_divide(s2, s1)
    return safe_divide(s1, s2)


def cross_sectional_mean(df: pd.DataFrame) -> pd.Series:
    """Compute cross-sectional mean of DataFrame columns.

    Args:
        df: Input DataFrame

    Returns:
        Mean across columns for each row
    """
    return df.mean(axis=1)


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize each row to mean=0, std=1.

    Args:
        df: Input DataFrame

    Returns:
        Cross-sectionally standardized DataFrame
    """
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1)
    return df.sub(row_mean, axis=0).div(row_std.replace(0, np.nan), axis=0)


def weighted_mean(values: pd.Series | pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Compute weighted mean.

    Args:
        values: Series or DataFrame with named columns
        weights: Dict mapping names to weights

    Returns:
        Weighted mean series
    """
    if isinstance(values, pd.Series):
        values = values.to_frame().T

    result = pd.Series(0.0, index=values.index)
    total_weight = 0.0

    for col, weight in weights.items():
        if col in values.columns:
            result += values[col].fillna(0) * weight
            total_weight += weight

    if total_weight > 0:
        result /= total_weight

    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise correlation matrix.

    Args:
        df: Input DataFrame

    Returns:
        Correlation matrix
    """
    return df.corr()


def max_pairwise_correlation(df: pd.DataFrame) -> tuple[float, str, str]:
    """Find maximum pairwise correlation.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (max_corr, col1, col2)
    """
    corr = df.corr().abs()
    # Mask diagonal
    np.fill_diagonal(corr.values, 0)
    max_idx = corr.values.argmax()
    i, j = divmod(max_idx, corr.shape[1])
    return corr.iloc[i, j], corr.index[i], corr.columns[j]


def forward_return(s: pd.Series, horizon: int, log: bool = True) -> pd.Series:
    """Compute forward return.

    Args:
        s: Price series
        horizon: Forward horizon (periods)
        log: If True, compute log return

    Returns:
        Forward return series
    """
    future_price = s.shift(-horizon)
    if log:
        return np.log(future_price / s)
    return (future_price - s) / s


def max_drawdown_forward(s: pd.Series, horizon: int) -> pd.Series:
    """Compute maximum drawdown over forward horizon.

    Args:
        s: Price series
        horizon: Forward horizon (periods)

    Returns:
        Maximum drawdown in forward window (negative values)
    """
    result = pd.Series(np.nan, index=s.index)

    for i in range(len(s) - horizon):
        window = s.iloc[i : i + horizon + 1]
        cummax = window.cummax()
        dd = (window - cummax) / cummax
        result.iloc[i] = dd.min()

    return result


def information_coefficient(
    signal: pd.Series, forward_ret: pd.Series, method: str = "spearman"
) -> float:
    """Compute information coefficient (rank correlation with forward returns).

    Args:
        signal: Signal series
        forward_ret: Forward return series
        method: Correlation method ('spearman' or 'pearson')

    Returns:
        IC value
    """
    # Align series
    aligned = pd.concat([signal, forward_ret], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan

    if method == "spearman":
        return aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
