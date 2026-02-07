#!/usr/bin/env python
"""Run weekly data update pipeline.

Usage:
    python scripts/run_update.py [--force-refresh] [--archive]

Options:
    --force-refresh    Ignore cache and re-fetch all data
    --archive          Save dated copies in addition to latest
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risk_index.core.config_schema import load_config
from risk_index.core.constants import PROCESSED_DIR, ARTIFACTS_DIR
from risk_index.core.utils_io import ensure_dir, write_parquet, save_run_manifest
from risk_index.core.logger import setup_logger, log_step
from risk_index.pipeline.fetch import fetch_all
from risk_index.pipeline.clean_align import clean_and_align
from risk_index.pipeline.features_build import build_features
from risk_index.pipeline.blocks_build import build_blocks
from risk_index.pipeline.composite_build import build_composites
from risk_index.pipeline.regimes_build import build_regimes
from risk_index.pipeline.checklist_build import build_checklist
from risk_index.reporting.excel_export import export_all_excel
from risk_index.reporting.report_html import generate_html_report
from risk_index.reporting.charts import save_all_charts


def main():
    """Run the full update pipeline."""
    parser = argparse.ArgumentParser(description="Run weekly data update pipeline")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cache and re-fetch all data",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Save dated copies in addition to latest",
    )
    args = parser.parse_args()

    logger = setup_logger("risk_index")
    log_step(logger, "pipeline", "start")

    try:
        # Load configurations
        log_step(logger, "load_config", "start")
        sources_cfg = load_config("sources")
        universe_cfg = load_config("universe")
        transforms_cfg = load_config("transforms")
        composites_cfg = load_config("composites")
        regimes_cfg = load_config("regimes")
        checklist_cfg = load_config("checklist")
        log_step(logger, "load_config", "complete")

        # Ensure directories exist
        ensure_dir(PROCESSED_DIR)
        ensure_dir(ARTIFACTS_DIR)

        # Step 1: Fetch data
        log_step(logger, "fetch_data", "start")
        raw_series = fetch_all(
            sources_cfg,
            universe_cfg,
            force_refresh=args.force_refresh,
        )
        log_step(logger, "fetch_data", "complete", series_count=len(raw_series))

        # Step 2: Clean and align
        log_step(logger, "clean_align", "start")
        weekly_df = clean_and_align(raw_series, universe_cfg, transforms_cfg)
        write_parquet(weekly_df, PROCESSED_DIR / "weekly_latest.parquet")
        log_step(logger, "clean_align", "complete", rows=len(weekly_df))

        # Step 3: Build features
        log_step(logger, "build_features", "start")
        features_df = build_features(weekly_df, universe_cfg, transforms_cfg)
        write_parquet(features_df, PROCESSED_DIR / "features_latest.parquet")
        log_step(logger, "build_features", "complete", features=len(features_df.columns))

        # Step 4: Build blocks
        log_step(logger, "build_blocks", "start")
        blocks_df = build_blocks(features_df, universe_cfg, composites_cfg)
        write_parquet(blocks_df, PROCESSED_DIR / "blocks_latest.parquet")
        log_step(logger, "build_blocks", "complete", blocks=len(blocks_df.columns))

        # Step 5: Build composites
        log_step(logger, "build_composites", "start")
        composites_df = build_composites(blocks_df, composites_cfg)
        write_parquet(composites_df, PROCESSED_DIR / "composites_latest.parquet")
        log_step(logger, "build_composites", "complete")

        # Step 6: Build regimes
        log_step(logger, "build_regimes", "start")
        regimes_df = build_regimes(composites_df, blocks_df, features_df, regimes_cfg)
        write_parquet(regimes_df, PROCESSED_DIR / "regimes_latest.parquet")
        log_step(logger, "build_regimes", "complete")

        # Step 7: Build checklist
        log_step(logger, "build_checklist", "start")
        checklist_df = build_checklist(features_df, composites_df, checklist_cfg)
        write_parquet(checklist_df, PROCESSED_DIR / "checklist_latest.parquet")
        log_step(logger, "build_checklist", "complete")

        # Step 8: Export reports
        log_step(logger, "export_reports", "start")

        # Excel exports
        export_all_excel(
            weekly_df,
            features_df,
            blocks_df,
            composites_df,
            regimes_df,
            checklist_df,
            universe_cfg,
            checklist_cfg,
        )

        # HTML report
        generate_html_report(
            composites_df,
            blocks_df,
            regimes_df,
            checklist_df,
            checklist_cfg,
        )

        # PNG charts
        save_all_charts(
            composites_df,
            blocks_df,
            regimes_df,
            checklist_df,
        )

        log_step(logger, "export_reports", "complete")

        # Save run manifest
        data_as_of = weekly_df.index[-1].strftime("%Y-%m-%d") if not weekly_df.empty else "unknown"
        save_run_manifest(
            data_as_of=data_as_of,
            universe_size={
                "series": len(universe_cfg.series),
                "ratios": len(universe_cfg.ratios),
                "features": len(features_df.columns),
            },
            selected_features={
                "fast": 0,  # Not doing feature selection in update mode
                "medium": 0,
                "slow": 0,
            },
        )

        log_step(logger, "pipeline", "complete")
        logger.info(f"Pipeline completed successfully. Data as of: {data_as_of}")

        # Print summary
        print("\n" + "=" * 60)
        print("RISK REGIME UPDATE COMPLETE")
        print("=" * 60)

        if not regimes_df.empty:
            latest = regimes_df.iloc[-1]
            print(f"\nDate: {regimes_df.index[-1].strftime('%Y-%m-%d')}")
            print(f"\nRegimes:")
            print(f"  Fast (4-8w):    {latest.get('regime_fast', 'N/A')}")
            print(f"  Medium (13-26w): {latest.get('regime_medium', 'N/A')}")
            print(f"  Slow (26-52w):  {latest.get('regime_slow', 'N/A')}")

        if not checklist_df.empty:
            latest_checklist = checklist_df.iloc[-1]
            print(f"\nChecklist Score: {latest_checklist.get('checklist_score', 'N/A'):.0f}/100")
            print(f"Checklist Label: {latest_checklist.get('checklist_label', 'N/A')}")

        print(f"\nOutputs:")
        print(f"  Processed data: {PROCESSED_DIR}")
        print(f"  Reports: {ARTIFACTS_DIR}")
        print(f"  View dashboard: streamlit run risk_index/reporting/dashboard.py")
        print("=" * 60)

    except Exception as e:
        log_step(logger, "pipeline", "error", error=str(e))
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
