"""Pydantic models for configuration validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union, List

from pydantic import BaseModel, Field, field_validator, model_validator

from risk_index.core.constants import CONFIG_DIR
from risk_index.core.utils_io import read_yaml
from risk_index.core.exceptions import ConfigError


# ============================================================================
# sources.yml
# ============================================================================


class SourceConfig(BaseModel):
    """Configuration for a data source."""


    name: str
    enabled: bool = True
    rate_limit_per_minute: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


class SourcesConfig(BaseModel):
    """Configuration for all data sources."""

    yahoo: SourceConfig
    fred: SourceConfig


# ============================================================================
# universe.yml
# ============================================================================


class FallbackConfig(BaseModel):
    """Fallback ticker configuration."""

    source: Literal["yahoo", "fred"]
    ticker: str


class SeriesConfig(BaseModel):
    """Configuration for a single data series."""

    id: str
    source: Literal["yahoo", "fred"]
    ticker: str
    kind: Literal["price", "fx", "vol", "macro_daily", "macro_weekly", "macro_monthly"] = "price"
    fallbacks: list[FallbackConfig] = Field(default_factory=list)
    description: str = ""


class RatioConfig(BaseModel):
    """Configuration for a ratio series."""

    id: str
    numerator: str
    denominator: str
    invert: bool = False
    description: str = ""


class ComputedSeriesConfig(BaseModel):
    """Configuration for computed series (derived from base series)."""

    id: str
    type: Literal["realized_vol", "rsi", "above_ma", "ma_slope", "moving_average"]
    series: str
    window_days: int = 21
    description: str = ""


class BlockMemberConfig(BaseModel):
    """Configuration for a block member."""

    id: str
    use_features: list[str] = Field(default_factory=lambda: ["z_52w", "roc_8w"])
    invert: bool = False


class BlockConfig(BaseModel):
    """Configuration for a signal block."""

    name: str
    members: list[BlockMemberConfig]
    description: str = ""


class UniverseProfileConfig(BaseModel):
    """Configuration for universe profile (full_etf or long_history)."""

    description: str
    min_coverage: float = 0.85
    series: list[str] = Field(default_factory=list)


class UniverseConfig(BaseModel):
    """Configuration for the data universe."""

    profiles: dict[str, UniverseProfileConfig]
    series: list[SeriesConfig]
    ratios: list[RatioConfig]
    computed: list[ComputedSeriesConfig] = Field(default_factory=list)
    blocks: list[BlockConfig]

    @property
    def series_by_id(self) -> dict[str, SeriesConfig]:
        return {s.id: s for s in self.series}

    @property
    def ratios_by_id(self) -> dict[str, RatioConfig]:
        return {r.id: r for r in self.ratios}

    @property
    def blocks_by_name(self) -> dict[str, BlockConfig]:
        return {b.name: b for b in self.blocks}


# ============================================================================
# transforms.yml
# ============================================================================


class FfillLimitsConfig(BaseModel):
    """Forward-fill limits by series kind."""

    price: int = 5
    fx: int = 3
    vol: int = 2
    macro_daily: int = 10
    macro_weekly: int = 21
    macro_monthly: int = 60


class TransformDefinition(BaseModel):
    """Definition for a feature transform."""

    name: str
    function: Literal["zscore", "percentile", "roc", "slope", "drawdown", "realized_vol"]
    window: int
    description: str = ""


class TransformsConfig(BaseModel):
    """Configuration for feature transforms."""

    ffill_limits: FfillLimitsConfig
    transforms: list[TransformDefinition]

    @property
    def transforms_by_name(self) -> dict[str, TransformDefinition]:
        return {t.name: t for t in self.transforms}


# ============================================================================
# composites.yml
# ============================================================================


class BlockWeightConfig(BaseModel):
    """Weight for a block in a composite."""

    block: str
    weight: float = 1.0


class CorrelationPolicyConfig(BaseModel):
    """Policy for handling correlated blocks."""

    max_block_corr: float = 0.90
    action_if_exceeded: Literal["cap_combined_weight", "merge_blocks"] = "cap_combined_weight"
    max_combined_weight: float = 0.25


class CompositeDefinition(BaseModel):
    """Definition for a composite signal."""

    name: str
    speed: Literal["fast", "medium", "slow"]
    z_window: int
    blocks: list[BlockWeightConfig]
    target_horizons: list[int]
    objective: str = ""


class CompositesConfig(BaseModel):
    """Configuration for composite signals."""

    correlation_policy: CorrelationPolicyConfig
    composites: list[CompositeDefinition]

    @property
    def composites_by_speed(self) -> dict[str, CompositeDefinition]:
        return {c.speed: c for c in self.composites}


# ============================================================================
# regimes.yml
# ============================================================================


class RegimeThresholds(BaseModel):
    """Thresholds for regime classification."""

    risk_on: float = 0.50
    risk_off: float = -0.50
    hysteresis_buffer: float = 0.15
    min_weeks_in_regime: int = 2


class ConfidenceWeights(BaseModel):
    """Weights for confidence scoring."""

    dispersion: float = 0.60
    tail_penalty: float = 0.20
    liquidity_penalty: float = 0.20

    @model_validator(mode="after")
    def validate_weights_sum(self):
        total = self.dispersion + self.tail_penalty + self.liquidity_penalty
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Confidence weights must sum to 1.0, got {total}")
        return self


class TailLiquidityThresholds(BaseModel):
    """Thresholds for tail and liquidity penalties."""

    vix_z_threshold: float = 1.5
    vvix_z_threshold: float = 1.5
    nfci_threshold: float = 0.0


class RegimesConfig(BaseModel):
    """Configuration for regime classification."""

    thresholds: RegimeThresholds
    confidence_weights: ConfidenceWeights
    tail_liquidity: TailLiquidityThresholds


# ============================================================================
# checklist.yml
# ============================================================================


class ThresholdRuleConfig(BaseModel):
    """Configuration for a threshold-based rule."""

    type: Literal["threshold_rule"] = "threshold_rule"
    series: str
    feature: str = ""
    operator: Literal["gt", "lt", "gte", "lte"]
    bull_threshold: float
    bear_threshold: float


class TrendRuleConfig(BaseModel):
    """Configuration for a trend-based rule."""

    type: Literal["trend_rule"] = "trend_rule"
    series: str
    feature: str
    bull_threshold: float = 0.0
    bear_threshold: float = 0.0


class CompoundRuleConfig(BaseModel):
    """Configuration for a compound rule (multiple conditions)."""

    type: Literal["compound_rule"] = "compound_rule"
    conditions: List[Union[ThresholdRuleConfig, TrendRuleConfig]]
    logic: Literal["all", "any"] = "all"


class ChecklistItemConfig(BaseModel):
    """Configuration for a checklist item."""

    id: str
    name: str
    category: str
    weight: float = 1.0
    rule: Union[ThresholdRuleConfig, TrendRuleConfig, CompoundRuleConfig]
    description: str = ""


class ChecklistConfig(BaseModel):
    """Configuration for the checklist engine."""

    items: list[ChecklistItemConfig]
    score_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"confirmed_risk_on": 75, "on_watch": 50}
    )

    @property
    def items_by_id(self) -> dict[str, ChecklistItemConfig]:
        return {item.id: item for item in self.items}


# ============================================================================
# backtest.yml
# ============================================================================


class TargetConfig(BaseModel):
    """Configuration for a prediction target."""

    id: str
    series: str
    horizon_weeks: list[int]
    type: Literal["log_return", "max_drawdown_forward"]


class SelectionRulesConfig(BaseModel):
    """Rules for feature selection."""

    min_abs_ic: float = 0.03
    min_horizons: int = 2
    min_windows_pass: float = 0.60
    min_sign_consistency: float = 0.70
    max_pair_corr: float = 0.85


class WalkForwardConfig(BaseModel):
    """Walk-forward validation settings."""

    train_years: int = 8
    test_years: int = 2
    step_years: int = 1
    purge_gap_weeks: int = 2
    embargo_weeks: int = 1


class BacktestConfig(BaseModel):
    """Configuration for backtesting and research."""

    targets: list[TargetConfig]
    selection_rules: SelectionRulesConfig
    walk_forward: WalkForwardConfig


# ============================================================================
# Config Loading
# ============================================================================


def load_config(config_type: str, config_dir: Path = CONFIG_DIR) -> BaseModel:
    """Load and validate a configuration file.

    Args:
        config_type: Type of config ('sources', 'universe', 'transforms',
                     'composites', 'regimes', 'checklist', 'backtest')
        config_dir: Configuration directory

    Returns:
        Validated Pydantic model

    Raises:
        ConfigError: If config file is missing or invalid
    """
    config_map = {
        "sources": (SourcesConfig, "sources.yml"),
        "universe": (UniverseConfig, "universe.yml"),
        "transforms": (TransformsConfig, "transforms.yml"),
        "composites": (CompositesConfig, "composites.yml"),
        "regimes": (RegimesConfig, "regimes.yml"),
        "checklist": (ChecklistConfig, "checklist.yml"),
        "backtest": (BacktestConfig, "backtest.yml"),
    }

    if config_type not in config_map:
        raise ConfigError(f"Unknown config type: {config_type}")

    model_class, filename = config_map[config_type]
    config_path = config_dir / filename

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw_config = read_yaml(config_path)
        return model_class.model_validate(raw_config)
    except Exception as e:
        raise ConfigError(f"Failed to parse {filename}: {e}")


def load_all_configs(config_dir: Path = CONFIG_DIR) -> dict[str, BaseModel]:
    """Load all configuration files.

    Args:
        config_dir: Configuration directory

    Returns:
        Dict mapping config type to validated model
    """
    configs = {}
    for config_type in ["sources", "universe", "transforms", "composites", "regimes", "checklist", "backtest"]:
        try:
            configs[config_type] = load_config(config_type, config_dir)
        except ConfigError:
            # Some configs may be optional
            pass
    return configs
