"""Tests for configuration schema validation."""

import pytest
from pathlib import Path

from risk_index.core.config_schema import (
    load_config,
    load_all_configs,
    SourcesConfig,
    UniverseConfig,
    TransformsConfig,
    CompositesConfig,
    RegimesConfig,
    ChecklistConfig,
    BacktestConfig,
)
from risk_index.core.exceptions import ConfigError
from risk_index.core.constants import CONFIG_DIR


class TestSourcesConfig:
    """Tests for sources.yml validation."""

    def test_load_sources_config(self):
        """Test loading sources config."""
        config = load_config("sources")
        assert isinstance(config, SourcesConfig)
        assert config.yahoo.enabled is True
        assert config.fred.enabled is True
        assert config.yahoo.retry_attempts == 3

    def test_sources_has_rate_limits(self):
        """Test that rate limits are defined."""
        config = load_config("sources")
        assert config.yahoo.rate_limit_per_minute > 0
        assert config.fred.rate_limit_per_minute > 0


class TestUniverseConfig:
    """Tests for universe.yml validation."""

    def test_load_universe_config(self):
        """Test loading universe config."""
        config = load_config("universe")
        assert isinstance(config, UniverseConfig)
        assert len(config.series) > 0
        assert len(config.ratios) > 0
        assert len(config.blocks) > 0

    def test_universe_has_profiles(self):
        """Test that profiles are defined."""
        config = load_config("universe")
        assert "full_etf" in config.profiles
        assert "long_history" in config.profiles

    def test_series_by_id_lookup(self):
        """Test series lookup by ID."""
        config = load_config("universe")
        spy = config.series_by_id.get("SPY")
        assert spy is not None
        assert spy.source == "yahoo"
        assert spy.ticker == "SPY"

    def test_ratios_by_id_lookup(self):
        """Test ratio lookup by ID."""
        config = load_config("universe")
        ratio = config.ratios_by_id.get("RSP_SPY")
        assert ratio is not None
        assert ratio.numerator == "RSP"
        assert ratio.denominator == "SPY"

    def test_blocks_have_members(self):
        """Test that all blocks have at least one member."""
        config = load_config("universe")
        for block in config.blocks:
            assert len(block.members) > 0, f"Block {block.name} has no members"

    def test_computed_series_valid(self):
        """Test computed series configuration."""
        config = load_config("universe")
        assert len(config.computed) > 0
        for computed in config.computed:
            assert computed.type in ["realized_vol", "rsi", "above_ma", "ma_slope", "moving_average"]


class TestTransformsConfig:
    """Tests for transforms.yml validation."""

    def test_load_transforms_config(self):
        """Test loading transforms config."""
        config = load_config("transforms")
        assert isinstance(config, TransformsConfig)
        assert len(config.transforms) > 0

    def test_ffill_limits_defined(self):
        """Test forward-fill limits are defined."""
        config = load_config("transforms")
        assert config.ffill_limits.price == 5
        assert config.ffill_limits.fx == 3
        assert config.ffill_limits.vol == 2

    def test_transforms_by_name_lookup(self):
        """Test transform lookup by name."""
        config = load_config("transforms")
        z52 = config.transforms_by_name.get("z_52w")
        assert z52 is not None
        assert z52.function == "zscore"
        assert z52.window == 52


class TestCompositesConfig:
    """Tests for composites.yml validation."""

    def test_load_composites_config(self):
        """Test loading composites config."""
        config = load_config("composites")
        assert isinstance(config, CompositesConfig)
        assert len(config.composites) == 3  # fast, medium, slow

    def test_composites_have_all_speeds(self):
        """Test that all three speeds are defined."""
        config = load_config("composites")
        speeds = {c.speed for c in config.composites}
        assert speeds == {"fast", "medium", "slow"}

    def test_correlation_policy_defined(self):
        """Test correlation policy is defined."""
        config = load_config("composites")
        assert config.correlation_policy.max_block_corr == 0.90
        assert config.correlation_policy.max_combined_weight == 0.25

    def test_composites_have_blocks(self):
        """Test that all composites have blocks."""
        config = load_config("composites")
        for composite in config.composites:
            assert len(composite.blocks) > 0, f"Composite {composite.name} has no blocks"


class TestRegimesConfig:
    """Tests for regimes.yml validation."""

    def test_load_regimes_config(self):
        """Test loading regimes config."""
        config = load_config("regimes")
        assert isinstance(config, RegimesConfig)

    def test_thresholds_defined(self):
        """Test that thresholds are defined."""
        config = load_config("regimes")
        assert config.thresholds.risk_on == 0.50
        assert config.thresholds.risk_off == -0.50
        assert config.thresholds.hysteresis_buffer == 0.15

    def test_confidence_weights_sum_to_one(self):
        """Test that confidence weights sum to 1.0."""
        config = load_config("regimes")
        total = (
            config.confidence_weights.dispersion
            + config.confidence_weights.tail_penalty
            + config.confidence_weights.liquidity_penalty
        )
        assert abs(total - 1.0) < 0.01


class TestChecklistConfig:
    """Tests for checklist.yml validation."""

    def test_load_checklist_config(self):
        """Test loading checklist config."""
        config = load_config("checklist")
        assert isinstance(config, ChecklistConfig)
        assert len(config.items) == 14  # 14 items per spec

    def test_checklist_items_have_weights(self):
        """Test that all items have weights."""
        config = load_config("checklist")
        for item in config.items:
            assert item.weight > 0, f"Item {item.id} has invalid weight"

    def test_checklist_items_have_rules(self):
        """Test that all items have rules."""
        config = load_config("checklist")
        for item in config.items:
            assert item.rule is not None, f"Item {item.id} has no rule"

    def test_score_thresholds_defined(self):
        """Test score thresholds are defined."""
        config = load_config("checklist")
        assert config.score_thresholds["confirmed_risk_on"] == 75
        assert config.score_thresholds["on_watch"] == 50


class TestBacktestConfig:
    """Tests for backtest.yml validation."""

    def test_load_backtest_config(self):
        """Test loading backtest config."""
        config = load_config("backtest")
        assert isinstance(config, BacktestConfig)

    def test_targets_defined(self):
        """Test that targets are defined."""
        config = load_config("backtest")
        assert len(config.targets) > 0

    def test_selection_rules_defined(self):
        """Test selection rules are defined."""
        config = load_config("backtest")
        assert config.selection_rules.min_abs_ic == 0.03
        assert config.selection_rules.max_pair_corr == 0.85

    def test_walk_forward_defined(self):
        """Test walk-forward settings are defined."""
        config = load_config("backtest")
        assert config.walk_forward.train_years == 8
        assert config.walk_forward.purge_gap_weeks == 2


class TestLoadAllConfigs:
    """Tests for loading all configurations."""

    def test_load_all_configs(self):
        """Test loading all config files."""
        configs = load_all_configs()
        assert "sources" in configs
        assert "universe" in configs
        assert "transforms" in configs
        assert "composites" in configs
        assert "regimes" in configs
        assert "checklist" in configs
        assert "backtest" in configs


class TestInvalidConfigs:
    """Tests for invalid configuration handling."""

    def test_unknown_config_type_raises(self):
        """Test that unknown config type raises error."""
        with pytest.raises(ConfigError):
            load_config("unknown")

    def test_missing_file_raises(self):
        """Test that missing file raises error."""
        with pytest.raises(ConfigError):
            load_config("sources", config_dir=Path("/nonexistent"))
