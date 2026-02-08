"""Research and backtesting modules."""

from risk_index.research.lead_lag import (
    compute_lead_lag_matrix,
    compute_ic_with_stats,
    compute_rolling_ic,
    rank_features_by_ic,
    identify_horizon_specialists,
)
from risk_index.research.walk_forward import (
    WalkForwardSplit,
    walk_forward_splits,
    get_split_data,
    evaluate_features_walk_forward,
    aggregate_walk_forward_results,
    compute_ic_stability_score,
)
from risk_index.research.feature_selection import (
    SelectionRules,
    select_features,
    compute_feature_scores,
    get_selection_report,
    assign_features_to_blocks,
)
from risk_index.research.weight_optimization import (
    optimize_block_weights,
    optimize_composite_weights,
    generate_optimized_composites_config,
    compare_weights,
    validate_weights,
)
from risk_index.research.backtest_regimes import (
    Trade,
    load_spy_prices,
    backtest_spy_regime,
    run_all_regime_backtests,
    create_backtest_summary,
)

__all__ = [
    # lead_lag
    "compute_lead_lag_matrix",
    "compute_ic_with_stats",
    "compute_rolling_ic",
    "rank_features_by_ic",
    "identify_horizon_specialists",
    # walk_forward
    "WalkForwardSplit",
    "walk_forward_splits",
    "get_split_data",
    "evaluate_features_walk_forward",
    "aggregate_walk_forward_results",
    "compute_ic_stability_score",
    # feature_selection
    "SelectionRules",
    "select_features",
    "compute_feature_scores",
    "get_selection_report",
    "assign_features_to_blocks",
    # weight_optimization
    "optimize_block_weights",
    "optimize_composite_weights",
    "generate_optimized_composites_config",
    "compare_weights",
    "validate_weights",
    # backtest_regimes
    "Trade",
    "load_spy_prices",
    "backtest_spy_regime",
    "run_all_regime_backtests",
    "create_backtest_summary",
]
