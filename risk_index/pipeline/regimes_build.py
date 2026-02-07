"""Regime classification module with hysteresis and confidence scoring."""

import pandas as pd
import numpy as np

from risk_index.core.config_schema import RegimesConfig
from risk_index.core.types import Regime
from risk_index.core.constants import (
    COL_COMPOSITE_FAST,
    COL_COMPOSITE_MEDIUM,
    COL_COMPOSITE_SLOW,
    COL_REGIME_FAST,
    COL_REGIME_MEDIUM,
    COL_REGIME_SLOW,
    COL_CONFIDENCE_FAST,
    COL_CONFIDENCE_MEDIUM,
    COL_CONFIDENCE_SLOW,
    FEATURE_SEPARATOR,
)
from risk_index.core.utils_math import zscore
from risk_index.core.logger import get_logger

logger = get_logger(__name__)


def build_regimes(
    composites_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    features_df: pd.DataFrame,
    regimes_cfg: RegimesConfig,
) -> pd.DataFrame:
    """Build regime classifications and confidence scores.

    Args:
        composites_df: DataFrame with composite scores
        blocks_df: DataFrame with block scores
        features_df: DataFrame with features (for VIX, NFCI)
        regimes_cfg: Regimes configuration

    Returns:
        DataFrame with regimes and confidence scores
    """
    logger.info("Building regime classifications")

    result = composites_df.copy()

    # Classify regimes for each composite
    for composite_col, regime_col in [
        (COL_COMPOSITE_FAST, COL_REGIME_FAST),
        (COL_COMPOSITE_MEDIUM, COL_REGIME_MEDIUM),
        (COL_COMPOSITE_SLOW, COL_REGIME_SLOW),
    ]:
        if composite_col in result.columns:
            regime = classify_regime_with_hysteresis(
                result[composite_col],
                regimes_cfg.thresholds,
            )
            result[regime_col] = regime

    # Compute confidence scores
    for composite_col, confidence_col in [
        (COL_COMPOSITE_FAST, COL_CONFIDENCE_FAST),
        (COL_COMPOSITE_MEDIUM, COL_CONFIDENCE_MEDIUM),
        (COL_COMPOSITE_SLOW, COL_CONFIDENCE_SLOW),
    ]:
        if composite_col in result.columns:
            confidence = compute_confidence(
                result[composite_col],
                blocks_df,
                features_df,
                regimes_cfg,
            )
            result[confidence_col] = confidence

    # Validate regimes
    validation = validate_regimes(result)
    logger.info(f"Regime validation: {validation}")

    return result


def classify_regime_with_hysteresis(
    composite: pd.Series,
    thresholds,
) -> pd.Series:
    """Classify regimes with hysteresis to reduce whipsaw.

    Args:
        composite: Composite score series
        thresholds: RegimeThresholds

    Returns:
        Series of regime labels
    """
    regime = pd.Series(Regime.NEUTRAL.value, index=composite.index)

    if composite.empty:
        return regime

    # Initial classification without hysteresis
    regime[composite > thresholds.risk_on] = Regime.RISK_ON.value
    regime[composite < thresholds.risk_off] = Regime.RISK_OFF.value

    # Apply hysteresis
    hysteresis = thresholds.hysteresis_buffer
    min_weeks = thresholds.min_weeks_in_regime

    current_regime = Regime.NEUTRAL.value
    weeks_in_regime = 0

    for i in range(len(composite)):
        if pd.isna(composite.iloc[i]):
            regime.iloc[i] = current_regime
            weeks_in_regime += 1
            continue

        val = composite.iloc[i]

        # Determine what regime the current value suggests
        if val > thresholds.risk_on:
            suggested = Regime.RISK_ON.value
        elif val < thresholds.risk_off:
            suggested = Regime.RISK_OFF.value
        else:
            suggested = Regime.NEUTRAL.value

        if suggested == current_regime:
            # Stay in current regime
            weeks_in_regime += 1
            regime.iloc[i] = current_regime

        elif weeks_in_regime < min_weeks:
            # Must stay in current regime for minimum period
            weeks_in_regime += 1
            regime.iloc[i] = current_regime

        else:
            # Check if we should switch
            should_switch = False

            if current_regime == Regime.RISK_ON.value:
                # Need to drop below threshold - buffer to exit
                if val < thresholds.risk_on - hysteresis:
                    should_switch = True

            elif current_regime == Regime.RISK_OFF.value:
                # Need to rise above threshold + buffer to exit
                if val > thresholds.risk_off + hysteresis:
                    should_switch = True

            else:  # NEUTRAL
                # Need to cross threshold + buffer to exit
                if val > thresholds.risk_on + hysteresis:
                    should_switch = True
                elif val < thresholds.risk_off - hysteresis:
                    should_switch = True

            if should_switch:
                current_regime = suggested
                weeks_in_regime = 1
                regime.iloc[i] = current_regime
            else:
                weeks_in_regime += 1
                regime.iloc[i] = current_regime

    return regime


def compute_confidence(
    composite: pd.Series,
    blocks_df: pd.DataFrame,
    features_df: pd.DataFrame,
    regimes_cfg: RegimesConfig,
) -> pd.Series:
    """Compute confidence score for regime classification.

    Confidence = dispersion * 0.60 + (1 - tail_penalty) * 0.20 + (1 - liquidity_penalty) * 0.20

    Args:
        composite: Composite score series
        blocks_df: DataFrame with block scores
        features_df: DataFrame with features
        regimes_cfg: Regimes configuration

    Returns:
        Confidence score series [0, 1]
    """
    weights = regimes_cfg.confidence_weights
    thresholds = regimes_cfg.tail_liquidity

    confidence = pd.Series(0.5, index=composite.index)

    # Dispersion: fraction of blocks agreeing with composite sign
    dispersion = compute_block_dispersion(composite, blocks_df)

    # Tail penalty: VIX or VVIX elevated
    tail_penalty = compute_tail_penalty(features_df, thresholds)

    # Liquidity penalty: NFCI > 0
    liquidity_penalty = compute_liquidity_penalty(features_df, thresholds)

    # Combine components
    confidence = (
        dispersion * weights.dispersion
        + (1 - tail_penalty) * weights.tail_penalty
        + (1 - liquidity_penalty) * weights.liquidity_penalty
    )

    # Clip to [0, 1]
    confidence = confidence.clip(0, 1)

    return confidence


def compute_block_dispersion(
    composite: pd.Series,
    blocks_df: pd.DataFrame,
) -> pd.Series:
    """Compute fraction of blocks agreeing with composite sign.

    Args:
        composite: Composite score series
        blocks_df: DataFrame with block scores

    Returns:
        Dispersion score [0, 1]
    """
    if blocks_df.empty:
        return pd.Series(0.5, index=composite.index)

    # Align indices
    aligned_blocks = blocks_df.reindex(composite.index)

    # Get sign of composite
    composite_sign = np.sign(composite)

    # Get sign of each block
    block_signs = np.sign(aligned_blocks)

    # Count agreeing blocks
    agreement = (block_signs.T == composite_sign).T

    # Fraction agreeing (ignoring NaN)
    dispersion = agreement.mean(axis=1)

    return dispersion.fillna(0.5)


def compute_tail_penalty(
    features_df: pd.DataFrame,
    thresholds,
) -> pd.Series:
    """Compute tail penalty based on VIX/VVIX elevation.

    Args:
        features_df: DataFrame with features
        thresholds: TailLiquidityThresholds

    Returns:
        Penalty score [0, 1]
    """
    penalty = pd.Series(0.0, index=features_df.index)

    # Check VIX z-score
    vix_z_col = f"VIX{FEATURE_SEPARATOR}z_52w"
    if vix_z_col in features_df.columns:
        vix_z = features_df[vix_z_col]
        vix_elevated = (vix_z > thresholds.vix_z_threshold).astype(float)
        penalty = pd.concat([penalty, vix_elevated], axis=1).max(axis=1)

    # Check VVIX z-score
    vvix_z_col = f"VVIX{FEATURE_SEPARATOR}z_52w"
    if vvix_z_col in features_df.columns:
        vvix_z = features_df[vvix_z_col]
        vvix_elevated = (vvix_z > thresholds.vvix_z_threshold).astype(float)
        penalty = pd.concat([penalty, vvix_elevated], axis=1).max(axis=1)

    return penalty.fillna(0.0)


def compute_liquidity_penalty(
    features_df: pd.DataFrame,
    thresholds,
) -> pd.Series:
    """Compute liquidity penalty based on NFCI.

    Args:
        features_df: DataFrame with features
        thresholds: TailLiquidityThresholds

    Returns:
        Penalty score [0, 1]
    """
    penalty = pd.Series(0.0, index=features_df.index)

    # Check NFCI
    if "NFCI" in features_df.columns:
        nfci = features_df["NFCI"]
        nfci_stressed = (nfci > thresholds.nfci_threshold).astype(float)
        penalty = nfci_stressed

    return penalty.fillna(0.0)


def validate_regimes(regimes_df: pd.DataFrame) -> dict:
    """Validate regime classifications.

    Args:
        regimes_df: DataFrame with regime columns

    Returns:
        Validation metrics
    """
    validation = {}

    for regime_col in [COL_REGIME_FAST, COL_REGIME_MEDIUM, COL_REGIME_SLOW]:
        if regime_col not in regimes_df.columns:
            continue

        regime = regimes_df[regime_col]

        # Count transitions
        transitions = (regime != regime.shift(1)).sum() - 1
        years = len(regime) / 52

        transitions_per_year = transitions / years if years > 0 else 0

        # Count time in each regime
        regime_counts = regime.value_counts(normalize=True)

        validation[regime_col] = {
            "transitions_per_year": round(transitions_per_year, 1),
            "regime_distribution": regime_counts.to_dict(),
        }

        # Warn if too many transitions
        if "medium" in regime_col and transitions_per_year > 12:
            logger.warning(f"{regime_col} has {transitions_per_year:.1f} transitions/year (> 12)")

    return validation


def get_regime_history(regimes_df: pd.DataFrame) -> pd.DataFrame:
    """Get regime transition history.

    Args:
        regimes_df: DataFrame with regime columns

    Returns:
        DataFrame with regime transitions
    """
    transitions = []

    for regime_col in [COL_REGIME_FAST, COL_REGIME_MEDIUM, COL_REGIME_SLOW]:
        if regime_col not in regimes_df.columns:
            continue

        regime = regimes_df[regime_col]
        speed = regime_col.replace("regime_", "")

        # Find transitions
        changed = regime != regime.shift(1)
        transition_dates = regime.index[changed]

        for date in transition_dates:
            prev_idx = regime.index.get_loc(date) - 1
            if prev_idx >= 0:
                prev_regime = regime.iloc[prev_idx]
                new_regime = regime.loc[date]
                transitions.append(
                    {
                        "date": date,
                        "speed": speed,
                        "from_regime": prev_regime,
                        "to_regime": new_regime,
                    }
                )

    if not transitions:
        return pd.DataFrame()

    return pd.DataFrame(transitions).sort_values("date")


def get_current_regime_summary(regimes_df: pd.DataFrame) -> dict:
    """Get summary of current regime state.

    Args:
        regimes_df: DataFrame with regime columns

    Returns:
        Summary dict
    """
    if regimes_df.empty:
        return {}

    latest = regimes_df.iloc[-1]

    summary = {"date": regimes_df.index[-1].strftime("%Y-%m-%d")}

    for speed in ["fast", "medium", "slow"]:
        regime_col = f"regime_{speed}"
        confidence_col = f"confidence_{speed}"
        composite_col = f"composite_{speed}"

        if regime_col in latest.index:
            summary[f"{speed}_regime"] = latest[regime_col]

        if confidence_col in latest.index:
            summary[f"{speed}_confidence"] = round(latest[confidence_col], 2)

        if composite_col in latest.index:
            summary[f"{speed}_composite"] = round(latest[composite_col], 2)

    return summary
