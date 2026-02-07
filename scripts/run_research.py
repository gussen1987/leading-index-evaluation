#!/usr/bin/env python
"""Run full research pipeline with optimization.

Usage:
    python scripts/run_research.py [--force-refresh]

This script runs:
1. Data update pipeline
2. Lead-lag analysis
3. Walk-forward validation
4. Feature selection
5. Weight optimization
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risk_index.core.config_schema import load_config
from risk_index.core.constants import PROCESSED_DIR, ARTIFACTS_DIR
from risk_index.core.utils_io import ensure_dir, write_parquet, write_json
from risk_index.core.logger import setup_logger, log_step
from risk_index.pipeline.fetch import fetch_all
from risk_index.pipeline.clean_align import clean_and_align
from risk_index.pipeline.features_build import build_features
from risk_index.pipeline.blocks_build import build_blocks
from risk_index.pipeline.composite_build import build_composites
from risk_index.pipeline.regimes_build import build_regimes
from risk_index.pipeline.checklist_build import build_checklist


def main():
    """Run the full research pipeline."""
    parser = argparse.ArgumentParser(description="Run research pipeline with optimization")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cache and re-fetch all data",
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

        # Step 1: Fetch and process data
        log_step(logger, "data_pipeline", "start")

        raw_series = fetch_all(sources_cfg, universe_cfg, force_refresh=args.force_refresh)
        weekly_df = clean_and_align(raw_series, universe_cfg, transforms_cfg)
        features_df = build_features(weekly_df, universe_cfg, transforms_cfg)
        blocks_df = build_blocks(features_df, universe_cfg, composites_cfg)
        composites_df = build_composites(blocks_df, composites_cfg)
        regimes_df = build_regimes(composites_df, blocks_df, features_df, regimes_cfg)
        checklist_df = build_checklist(features_df, composites_df, checklist_cfg)

        log_step(logger, "data_pipeline", "complete")

        # Save processed data
        write_parquet(weekly_df, PROCESSED_DIR / "weekly_latest.parquet")
        write_parquet(features_df, PROCESSED_DIR / "features_latest.parquet")
        write_parquet(blocks_df, PROCESSED_DIR / "blocks_latest.parquet")
        write_parquet(composites_df, PROCESSED_DIR / "composites_latest.parquet")
        write_parquet(regimes_df, PROCESSED_DIR / "regimes_latest.parquet")
        write_parquet(checklist_df, PROCESSED_DIR / "checklist_latest.parquet")

        # Step 2: Lead-lag analysis
        log_step(logger, "lead_lag_analysis", "start")
        # TODO: Implement lead-lag analysis in research module
        logger.info("Lead-lag analysis not yet implemented")
        log_step(logger, "lead_lag_analysis", "complete")

        # Step 3: Walk-forward validation
        log_step(logger, "walk_forward", "start")
        # TODO: Implement walk-forward validation
        logger.info("Walk-forward validation not yet implemented")
        log_step(logger, "walk_forward", "complete")

        # Step 4: Feature selection
        log_step(logger, "feature_selection", "start")
        # TODO: Implement feature selection
        logger.info("Feature selection not yet implemented")
        log_step(logger, "feature_selection", "complete")

        # Step 5: Weight optimization
        log_step(logger, "weight_optimization", "start")
        # TODO: Implement weight optimization
        logger.info("Weight optimization not yet implemented")
        log_step(logger, "weight_optimization", "complete")

        log_step(logger, "research_pipeline", "complete")

        print("\n" + "=" * 60)
        print("RESEARCH PIPELINE COMPLETE")
        print("=" * 60)
        print("\nNote: Lead-lag, walk-forward, feature selection, and")
        print("weight optimization modules are planned for Phase 2.")
        print("\nBasic pipeline completed successfully.")
        print(f"Data as of: {weekly_df.index[-1].strftime('%Y-%m-%d')}")
        print("=" * 60)

    except Exception as e:
        log_step(logger, "research_pipeline", "error", error=str(e))
        logger.exception("Research pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
