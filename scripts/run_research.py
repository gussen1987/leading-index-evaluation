#!/usr/bin/env python
"""Run full research pipeline with optimization.

Usage:
    python scripts/run_research.py [--force-refresh] [--skip-fetch]

This script runs:
1. Data update pipeline (fetch, clean, build features)
2. Lead-lag analysis (IC computation)
3. Walk-forward validation
4. Feature selection
5. Weight optimization
6. Generate optimized composites config
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risk_index.core.config_schema import load_config
from risk_index.core.constants import (
    PROCESSED_DIR,
    ARTIFACTS_DIR,
    FAST_HORIZONS,
    MEDIUM_HORIZONS,
    SLOW_HORIZONS,
)
from risk_index.core.utils_io import (
    ensure_dir,
    write_parquet,
    write_json,
    write_yaml,
    read_parquet,
)
from risk_index.core.logger import setup_logger, log_step
from risk_index.pipeline.fetch import fetch_all
from risk_index.pipeline.clean_align import clean_and_align
from risk_index.pipeline.features_build import build_features
from risk_index.pipeline.blocks_build import build_blocks
from risk_index.pipeline.composite_build import build_composites
from risk_index.pipeline.regimes_build import build_regimes
from risk_index.pipeline.checklist_build import build_checklist
from risk_index.research import (
    compute_lead_lag_matrix,
    compute_ic_with_stats,
    rank_features_by_ic,
    walk_forward_splits,
    evaluate_features_walk_forward,
    aggregate_walk_forward_results,
    SelectionRules,
    select_features,
    compute_feature_scores,
    get_selection_report,
    optimize_block_weights,
    generate_optimized_composites_config,
    compare_weights,
    validate_weights,
)


def main():
    """Run the full research pipeline."""
    parser = argparse.ArgumentParser(description="Run research pipeline with optimization")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cache and re-fetch all data",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip data fetching, use existing processed data",
    )
    parser.add_argument(
        "--optimization-method",
        choices=["equal", "inverse_variance", "ic_weighted", "max_ic"],
        default="ic_weighted",
        help="Weight optimization method (default: ic_weighted)",
    )
    parser.add_argument(
        "--skip-ic-stats",
        action="store_true",
        help="Skip slow rolling IC stats computation",
    )
    args = parser.parse_args()

    logger = setup_logger("risk_index")
    log_step(logger, "research_pipeline", "start")

    try:
        # Load configurations
        sources_cfg = load_config("sources")
        universe_cfg = load_config("universe")
        transforms_cfg = load_config("transforms")
        composites_cfg = load_config("composites")
        regimes_cfg = load_config("regimes")
        checklist_cfg = load_config("checklist")
        backtest_cfg = load_config("backtest")

        ensure_dir(PROCESSED_DIR)
        ensure_dir(ARTIFACTS_DIR)

        # =====================================================================
        # Step 1: Fetch and process data
        # =====================================================================
        log_step(logger, "data_pipeline", "start")

        if args.skip_fetch:
            logger.info("Skipping data fetch, loading existing processed data")
            weekly_df = read_parquet(PROCESSED_DIR / "weekly_latest.parquet")
            features_df = read_parquet(PROCESSED_DIR / "features_latest.parquet")
            blocks_df = read_parquet(PROCESSED_DIR / "blocks_latest.parquet")
        else:
            raw_series = fetch_all(sources_cfg, universe_cfg, force_refresh=args.force_refresh)
            weekly_df = clean_and_align(raw_series, universe_cfg, transforms_cfg)
            features_df = build_features(weekly_df, universe_cfg, transforms_cfg)
            blocks_df = build_blocks(features_df, universe_cfg, composites_cfg)

            # Save processed data
            write_parquet(weekly_df, PROCESSED_DIR / "weekly_latest.parquet")
            write_parquet(features_df, PROCESSED_DIR / "features_latest.parquet")
            write_parquet(blocks_df, PROCESSED_DIR / "blocks_latest.parquet")

        composites_df = build_composites(blocks_df, composites_cfg)
        regimes_df = build_regimes(composites_df, blocks_df, features_df, regimes_cfg)
        checklist_df = build_checklist(features_df, composites_df, checklist_cfg)

        write_parquet(composites_df, PROCESSED_DIR / "composites_latest.parquet")
        write_parquet(regimes_df, PROCESSED_DIR / "regimes_latest.parquet")
        write_parquet(checklist_df, PROCESSED_DIR / "checklist_latest.parquet")

        log_step(logger, "data_pipeline", "complete")
        logger.info(f"Data as of: {weekly_df.index[-1].strftime('%Y-%m-%d')}")
        logger.info(f"Features: {features_df.shape[1]} columns")

        # Get target series (SPY)
        if "SPY" in weekly_df.columns:
            target_series = weekly_df["SPY"]
        else:
            logger.error("SPY not found in weekly data")
            sys.exit(1)

        # Get feature columns only (those with __)
        feature_cols = [c for c in features_df.columns if "__" in c]
        features_only_df = features_df[feature_cols]
        logger.info(f"Feature columns for analysis: {len(feature_cols)}")

        # =====================================================================
        # Step 2: Lead-lag analysis
        # =====================================================================
        log_step(logger, "lead_lag_analysis", "start")

        horizons = [4, 8, 13, 26, 52]

        # Compute full IC matrix
        ic_matrix = compute_lead_lag_matrix(
            features_only_df,
            target_series,
            horizons=horizons,
            method="spearman",
        )

        # Save IC matrix
        write_parquet(ic_matrix.reset_index(), ARTIFACTS_DIR / "lead_lag_matrix.parquet")

        # Compute IC with statistics (optional - slow for large feature sets)
        if not args.skip_ic_stats:
            ic_stats = compute_ic_with_stats(
                features_only_df,
                target_series,
                horizons=horizons,
                rolling_window=52,
            )

            # Save IC stats
            for name, df in ic_stats.items():
                write_parquet(df.reset_index(), ARTIFACTS_DIR / f"ic_{name}.parquet")
        else:
            logger.info("Skipping IC stats computation (--skip-ic-stats)")

        # Rank features
        feature_ranking = rank_features_by_ic(ic_matrix, min_abs_ic=0.02)
        write_parquet(feature_ranking.reset_index(), ARTIFACTS_DIR / "feature_ranking.parquet")

        logger.info(f"Lead-lag matrix shape: {ic_matrix.shape}")
        logger.info(f"Top 10 features by IC:")
        for idx, row in feature_ranking.head(10).iterrows():
            logger.info(f"  {idx}: avg_abs_ic={row['avg_abs_ic']:.3f}, best={row['best_horizon']}")

        log_step(logger, "lead_lag_analysis", "complete")

        # =====================================================================
        # Step 3: Walk-forward validation
        # =====================================================================
        log_step(logger, "walk_forward", "start")

        wf_config = backtest_cfg.walk_forward

        # Generate splits
        splits = walk_forward_splits(
            features_only_df,
            train_years=wf_config.train_years,
            test_years=wf_config.test_years,
            step_years=wf_config.step_years,
            purge_weeks=wf_config.purge_gap_weeks,
            embargo_weeks=wf_config.embargo_weeks,
        )

        logger.info(f"Generated {len(splits)} walk-forward splits")

        # Evaluate features
        wf_results = evaluate_features_walk_forward(
            features_only_df,
            target_series,
            horizons=horizons,
            splits=splits,
        )

        # Save raw results
        write_parquet(wf_results, ARTIFACTS_DIR / "walk_forward_results.parquet")

        # Aggregate results
        wf_aggregated = aggregate_walk_forward_results(wf_results)
        write_parquet(wf_aggregated.reset_index(), ARTIFACTS_DIR / "walk_forward_aggregated.parquet")

        logger.info(f"Walk-forward results: {len(wf_results)} evaluations")
        logger.info(f"Aggregated: {len(wf_aggregated)} feature × horizon combinations")

        log_step(logger, "walk_forward", "complete")

        # =====================================================================
        # Step 4: Feature selection
        # =====================================================================
        log_step(logger, "feature_selection", "start")

        selection_rules = SelectionRules(
            min_abs_ic=backtest_cfg.selection_rules.min_abs_ic,
            min_horizons=backtest_cfg.selection_rules.min_horizons,
            min_windows_pass=backtest_cfg.selection_rules.min_windows_pass,
            min_sign_consistency=backtest_cfg.selection_rules.min_sign_consistency,
            max_pair_corr=backtest_cfg.selection_rules.max_pair_corr,
        )

        selected_features = select_features(
            wf_aggregated,
            features_only_df,
            selection_rules=selection_rules,
            fast_horizons=FAST_HORIZONS,
            medium_horizons=MEDIUM_HORIZONS,
            slow_horizons=SLOW_HORIZONS,
        )

        # Save selected features
        for speed, features in selected_features.items():
            write_json(
                {"selected_features": features, "count": len(features)},
                ARTIFACTS_DIR / f"selected_features_{speed}.json",
            )
            logger.info(f"{speed}: {len(features)} features selected")

        # Compute feature scores
        feature_scores = {}
        for speed, speed_horizons in [
            ("fast", FAST_HORIZONS),
            ("medium", MEDIUM_HORIZONS),
            ("slow", SLOW_HORIZONS),
        ]:
            scores = compute_feature_scores(wf_aggregated, horizons=speed_horizons)
            feature_scores[speed] = scores
            if not scores.empty:
                write_parquet(scores, ARTIFACTS_DIR / f"feature_scores_{speed}.parquet")

        # Generate selection report
        selection_report = get_selection_report(wf_aggregated, selected_features, selection_rules)
        write_json(selection_report, ARTIFACTS_DIR / "selection_report.json")

        log_step(logger, "feature_selection", "complete")

        # =====================================================================
        # Step 5: Weight optimization
        # =====================================================================
        log_step(logger, "weight_optimization", "start")

        # Get old weights from current config
        old_weights = {}
        for composite_def in composites_cfg.composites:
            speed = composite_def.speed
            old_weights[speed] = {
                bw.block: bw.weight for bw in composite_def.blocks
            }

        # Optimize weights for each speed
        optimized_weights = {}
        validation_results = {}

        for speed, speed_horizons in [
            ("fast", FAST_HORIZONS),
            ("medium", MEDIUM_HORIZONS),
            ("slow", SLOW_HORIZONS),
        ]:
            # Get blocks for this speed
            block_cols = [c for c in blocks_df.columns if c in old_weights.get(speed, {})]

            if not block_cols:
                logger.warning(f"No blocks found for {speed} composite")
                continue

            blocks_subset = blocks_df[block_cols]

            # Optimize for primary horizon of this speed
            primary_horizon = speed_horizons[0]

            new_weights = optimize_block_weights(
                blocks_subset,
                target_series,
                horizon=primary_horizon,
                method=args.optimization_method,
            )

            optimized_weights[speed] = new_weights

            # Compare old vs new
            if speed in old_weights:
                comparison = compare_weights(
                    old_weights[speed],
                    new_weights,
                    blocks_subset,
                    target_series,
                    horizon=primary_horizon,
                )
                logger.info(
                    f"{speed}: Old IC={comparison['old_ic']:.3f}, "
                    f"New IC={comparison['new_ic']:.3f}, "
                    f"Improvement={comparison.get('pct_improvement', 0):.1f}%"
                )

            # Validate on all horizons
            validation = validate_weights(
                new_weights,
                blocks_subset,
                target_series,
                horizons=speed_horizons,
            )
            validation_results[speed] = validation
            logger.info(f"{speed} validation: avg_ic={validation['avg_ic']:.3f}")

        # Save optimized weights
        write_json(optimized_weights, ARTIFACTS_DIR / "optimized_weights.json")
        write_json(validation_results, ARTIFACTS_DIR / "weight_validation.json")

        # Generate new composites config
        optimized_composites = generate_optimized_composites_config(
            optimized_weights,
            base_config=composites_cfg.model_dump(),
        )

        write_yaml(optimized_composites, ARTIFACTS_DIR / "optimized_composites.yml")

        log_step(logger, "weight_optimization", "complete")

        # =====================================================================
        # Summary
        # =====================================================================
        log_step(logger, "research_pipeline", "complete")

        print("\n" + "=" * 60)
        print("RESEARCH PIPELINE COMPLETE")
        print("=" * 60)
        print(f"\nData as of: {weekly_df.index[-1].strftime('%Y-%m-%d')}")
        print(f"\nUniverse:")
        print(f"  Series: {len(universe_cfg.series)}")
        print(f"  Ratios: {len(universe_cfg.ratios)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"\nWalk-Forward Validation:")
        print(f"  Splits: {len(splits)}")
        print(f"  Train period: {wf_config.train_years} years")
        print(f"  Test period: {wf_config.test_years} years")
        print(f"\nSelected Features:")
        for speed, features in selected_features.items():
            print(f"  {speed}: {len(features)} features")
        print(f"\nWeight Optimization ({args.optimization_method}):")
        for speed, weights in optimized_weights.items():
            print(f"  {speed}: {len(weights)} blocks")
            if speed in validation_results:
                print(f"    Validated avg IC: {validation_results[speed]['avg_ic']:.3f}")
        print(f"\nOutputs saved to: {ARTIFACTS_DIR}")
        print("  - lead_lag_matrix.parquet")
        print("  - walk_forward_results.parquet")
        print("  - selected_features_*.json")
        print("  - optimized_weights.json")
        print("  - optimized_composites.yml")
        print("=" * 60)

    except Exception as e:
        log_step(logger, "research_pipeline", "error", error=str(e))
        logger.exception("Research pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
