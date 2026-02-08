"""Walk-forward validation module for time series cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import numpy as np

from risk_index.core.logger import get_logger
from risk_index.core.utils_math import forward_return, information_coefficient

logger = get_logger(__name__)


@dataclass
class WalkForwardSplit:
    """Single walk-forward train/test split."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    split_id: int

    @property
    def train_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (self.train_start, self.train_end)

    @property
    def test_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (self.test_start, self.test_end)


def walk_forward_splits(
    df: pd.DataFrame,
    train_years: int = 8,
    test_years: int = 2,
    step_years: int = 1,
    purge_weeks: int = 2,
    embargo_weeks: int = 1,
) -> list[WalkForwardSplit]:
    """Generate train/test splits with proper purge gaps.

    The purge gap is a period between train and test that is excluded
    to prevent lookahead bias from overlapping target horizons.
    The embargo is a period after test that's also excluded.

    Args:
        df: DataFrame with DatetimeIndex
        train_years: Length of training period in years
        test_years: Length of test period in years
        step_years: How far to advance between splits
        purge_weeks: Gap between train end and test start (weeks)
        embargo_weeks: Gap after test end (weeks)

    Returns:
        List of WalkForwardSplit objects
    """
    if df.empty:
        return []

    # Convert to timestamps
    data_start = df.index.min()
    data_end = df.index.max()

    splits = []
    split_id = 0

    # Start first training period at data start
    train_start = data_start

    while True:
        # Calculate training end
        train_end = train_start + pd.DateOffset(years=train_years)

        # Calculate test period with purge gap
        purge_end = train_end + pd.DateOffset(weeks=purge_weeks)
        test_start = purge_end
        test_end = test_start + pd.DateOffset(years=test_years)

        # Check if we have enough data for this split
        if test_end > data_end:
            # Try to fit a partial test period
            if test_start < data_end:
                test_end = data_end
            else:
                break

        # Create split
        split = WalkForwardSplit(
            train_start=pd.Timestamp(train_start),
            train_end=pd.Timestamp(train_end),
            test_start=pd.Timestamp(test_start),
            test_end=pd.Timestamp(test_end),
            split_id=split_id,
        )
        splits.append(split)
        split_id += 1

        # Advance to next split
        train_start = train_start + pd.DateOffset(years=step_years)

        # Check if we can fit another full train+test
        if train_start + pd.DateOffset(years=train_years + test_years) > data_end + pd.DateOffset(years=1):
            break

    logger.info(f"Generated {len(splits)} walk-forward splits")
    for s in splits:
        logger.debug(
            f"Split {s.split_id}: train [{s.train_start.date()} - {s.train_end.date()}], "
            f"test [{s.test_start.date()} - {s.test_end.date()}]"
        )

    return splits


def get_split_data(
    df: pd.DataFrame,
    split: WalkForwardSplit,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get train and test DataFrames for a split.

    Args:
        df: Full DataFrame
        split: WalkForwardSplit

    Returns:
        Tuple of (train_df, test_df)
    """
    train_mask = (df.index >= split.train_start) & (df.index <= split.train_end)
    test_mask = (df.index >= split.test_start) & (df.index <= split.test_end)

    return df[train_mask].copy(), df[test_mask].copy()


def evaluate_feature_single_split(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_target: pd.Series,
    test_target: pd.Series,
    feature_name: str,
    method: Literal["spearman", "pearson"] = "spearman",
) -> dict:
    """Evaluate a single feature on one walk-forward split.

    Args:
        train_features: Training features DataFrame
        test_features: Test features DataFrame
        train_target: Training forward returns
        test_target: Test forward returns
        feature_name: Name of feature to evaluate
        method: Correlation method

    Returns:
        Dict with train_ic, test_ic, sign_match, etc.
    """
    # Train IC
    if feature_name in train_features.columns:
        train_ic = information_coefficient(
            train_features[feature_name], train_target, method=method
        )
    else:
        train_ic = np.nan

    # Test IC
    if feature_name in test_features.columns:
        test_ic = information_coefficient(
            test_features[feature_name], test_target, method=method
        )
    else:
        test_ic = np.nan

    # Sign consistency
    sign_match = (
        np.sign(train_ic) == np.sign(test_ic)
        if not (np.isnan(train_ic) or np.isnan(test_ic))
        else np.nan
    )

    return {
        "train_ic": train_ic,
        "test_ic": test_ic,
        "sign_match": sign_match,
        "ic_decay": train_ic - test_ic if not (np.isnan(train_ic) or np.isnan(test_ic)) else np.nan,
    }


def evaluate_features_walk_forward(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    horizons: list[int],
    splits: list[WalkForwardSplit] | None = None,
    train_years: int = 8,
    test_years: int = 2,
    step_years: int = 1,
    purge_weeks: int = 2,
    embargo_weeks: int = 1,
    method: Literal["spearman", "pearson"] = "spearman",
) -> pd.DataFrame:
    """Evaluate each feature across all walk-forward windows.

    Args:
        features_df: DataFrame with features
        target_series: Price series for forward returns
        horizons: List of forward horizons (weeks)
        splits: Pre-computed splits (if None, will be generated)
        train_years: Training period length
        test_years: Test period length
        step_years: Step between splits
        purge_weeks: Purge gap
        embargo_weeks: Embargo period
        method: IC method

    Returns:
        DataFrame with multi-level columns:
            (feature, horizon) -> metrics across splits
    """
    logger.info(f"Evaluating {len(features_df.columns)} features with walk-forward validation")

    # Generate splits if not provided
    if splits is None:
        splits = walk_forward_splits(
            features_df,
            train_years=train_years,
            test_years=test_years,
            step_years=step_years,
            purge_weeks=purge_weeks,
            embargo_weeks=embargo_weeks,
        )

    if not splits:
        logger.warning("No valid walk-forward splits could be generated")
        return pd.DataFrame()

    # Compute forward returns for all horizons
    forward_returns = {h: forward_return(target_series, horizon=h, log=True) for h in horizons}

    # Results storage
    results = []

    for split in splits:
        logger.debug(f"Processing split {split.split_id}")

        # Get train/test indices
        train_mask = (features_df.index >= split.train_start) & (features_df.index <= split.train_end)
        test_mask = (features_df.index >= split.test_start) & (features_df.index <= split.test_end)

        train_features = features_df[train_mask]
        test_features = features_df[test_mask]

        for horizon in horizons:
            fwd_ret = forward_returns[horizon]
            train_target = fwd_ret[train_mask]
            test_target = fwd_ret[test_mask]

            for feature in features_df.columns:
                eval_result = evaluate_feature_single_split(
                    train_features,
                    test_features,
                    train_target,
                    test_target,
                    feature,
                    method=method,
                )

                results.append({
                    "split_id": split.split_id,
                    "horizon": horizon,
                    "feature": feature,
                    "train_start": split.train_start,
                    "train_end": split.train_end,
                    "test_start": split.test_start,
                    "test_end": split.test_end,
                    **eval_result,
                })

    results_df = pd.DataFrame(results)
    logger.info(f"Walk-forward results: {len(results_df)} evaluations")
    return results_df


def aggregate_walk_forward_results(
    wf_results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate walk-forward results by feature and horizon.

    Args:
        wf_results: Raw walk-forward results from evaluate_features_walk_forward

    Returns:
        DataFrame with aggregated metrics per feature × horizon:
            - mean_train_ic: Average IC in training
            - mean_test_ic: Average IC in test (out-of-sample)
            - std_test_ic: Std of test IC
            - sign_consistency: Fraction of splits with consistent IC sign
            - pass_rate: Fraction of splits with positive test IC (if train IC positive)
            - n_splits: Number of valid splits
    """
    if wf_results.empty:
        return pd.DataFrame()

    agg_results = []

    for (feature, horizon), group in wf_results.groupby(["feature", "horizon"]):
        valid = group.dropna(subset=["train_ic", "test_ic"])

        if len(valid) == 0:
            continue

        # Sign consistency: test IC has same sign as train IC
        sign_match = valid["sign_match"].dropna()
        sign_consistency = sign_match.mean() if len(sign_match) > 0 else np.nan

        # Pass rate: test IC > 0 when train IC > 0 (or test IC < 0 when train IC < 0)
        same_sign_nonzero = (valid["train_ic"] * valid["test_ic"] > 0).sum()
        pass_rate = same_sign_nonzero / len(valid) if len(valid) > 0 else np.nan

        agg_results.append({
            "feature": feature,
            "horizon": horizon,
            "mean_train_ic": valid["train_ic"].mean(),
            "mean_test_ic": valid["test_ic"].mean(),
            "std_test_ic": valid["test_ic"].std(),
            "median_test_ic": valid["test_ic"].median(),
            "sign_consistency": sign_consistency,
            "pass_rate": pass_rate,
            "n_splits": len(valid),
        })

    result = pd.DataFrame(agg_results)

    if not result.empty:
        result = result.set_index(["feature", "horizon"])

    logger.info(f"Aggregated {len(result)} feature × horizon combinations")
    return result


def compute_ic_stability_score(
    wf_results: pd.DataFrame,
    feature: str,
    horizons: list[int] | None = None,
) -> float:
    """Compute IC stability score for a feature across walk-forward splits.

    Stability = mean(test_ic) / std(test_ic) if std > 0 else 0

    Args:
        wf_results: Walk-forward results DataFrame
        feature: Feature name
        horizons: Optional list of horizons to include

    Returns:
        Stability score (higher is better)
    """
    subset = wf_results[wf_results["feature"] == feature]

    if horizons is not None:
        subset = subset[subset["horizon"].isin(horizons)]

    test_ics = subset["test_ic"].dropna()

    if len(test_ics) < 2:
        return 0.0

    mean_ic = test_ics.mean()
    std_ic = test_ics.std()

    if std_ic == 0 or np.isnan(std_ic):
        return 0.0

    return mean_ic / std_ic
