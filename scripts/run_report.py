#!/usr/bin/env python
"""Regenerate reports from existing processed data.

Usage:
    python scripts/run_report.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risk_index.core.config_schema import load_config
from risk_index.core.constants import PROCESSED_DIR, ARTIFACTS_DIR
from risk_index.core.utils_io import read_parquet, ensure_dir
from risk_index.core.logger import setup_logger, log_step
from risk_index.reporting.excel_export import export_all_excel
from risk_index.reporting.report_html import generate_html_report
from risk_index.reporting.charts import save_all_charts


def main():
    """Regenerate all reports from existing data."""
    logger = setup_logger("risk_index")
    log_step(logger, "report_generation", "start")

    try:
        # Load configurations
        universe_cfg = load_config("universe")
        checklist_cfg = load_config("checklist")

        # Load processed data
        log_step(logger, "load_data", "start")

        weekly_df = read_parquet(PROCESSED_DIR / "weekly_latest.parquet")
        features_df = read_parquet(PROCESSED_DIR / "features_latest.parquet")
        blocks_df = read_parquet(PROCESSED_DIR / "blocks_latest.parquet")
        composites_df = read_parquet(PROCESSED_DIR / "composites_latest.parquet")
        regimes_df = read_parquet(PROCESSED_DIR / "regimes_latest.parquet")
        checklist_df = read_parquet(PROCESSED_DIR / "checklist_latest.parquet")

        log_step(logger, "load_data", "complete")

        # Generate reports
        ensure_dir(ARTIFACTS_DIR)

        log_step(logger, "generate_excel", "start")
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
        log_step(logger, "generate_excel", "complete")

        log_step(logger, "generate_html", "start")
        report_path = generate_html_report(
            composites_df,
            blocks_df,
            regimes_df,
            checklist_df,
            checklist_cfg,
        )
        log_step(logger, "generate_html", "complete")

        log_step(logger, "generate_charts", "start")
        chart_paths = save_all_charts(
            composites_df,
            blocks_df,
            regimes_df,
            checklist_df,
        )
        log_step(logger, "generate_charts", "complete", chart_count=len(chart_paths))

        log_step(logger, "report_generation", "complete")

        print(f"\nReports generated successfully!")
        print(f"HTML Report: {report_path}")
        print(f"Charts: {ARTIFACTS_DIR / 'charts'}")

    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        print("\nError: Processed data not found. Run the update pipeline first:")
        print("  python scripts/run_update.py")
        sys.exit(1)

    except Exception as e:
        log_step(logger, "report_generation", "error", error=str(e))
        logger.exception("Report generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
