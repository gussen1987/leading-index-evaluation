# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Risk-On/Risk-Off Regime Index - A quantitative system that classifies market regimes by aggregating 100+ intermarket signals into 10 thematic blocks, then combining them into Fast/Medium/Slow composite scores.

## Commands

### Development
```bash
pip install -e .              # Install package
pip install -e ".[dev]"       # Install with dev dependencies (pytest, ruff)
```

### Run Pipeline
```bash
python scripts/run_update.py [--force-refresh] [--archive]   # Full weekly pipeline + exports
python scripts/run_dashboard.py                               # Launch Streamlit dashboard
python scripts/run_report.py                                  # Regenerate reports from existing data (no refetch)
python scripts/run_research.py [--optimization-method {equal|inverse_variance|ic_weighted|max_ic}] [--skip-fetch] [--skip-ic-stats]
```

After `pip install -e .`, CLI entry points are also available: `risk-update`, `risk-dashboard`, `risk-report`, `risk-research`.

### Testing & Linting
```bash
pytest tests/                          # Run all tests
pytest tests/test_config_schema.py     # Single test file
pytest tests/test_config_schema.py::TestUniverseConfig  # Single test class
ruff check .                           # Lint
ruff format .                          # Format
```

## Architecture

### Data Flow
```
Yahoo Finance / FRED APIs
    ↓
[fetch] → Raw daily series (data/cache/)
    ↓
[clean_align] → Weekly Friday alignment (W-FRI)
    ↓
[features_build] → Z-scores, ROC, percentiles (100+ features)
    ↓
[blocks_build] → 10 thematic blocks
    ↓
[composite_build] → Fast/Medium/Slow weighted signals
    ↓
[regimes_build] → Risk-On/Neutral/Risk-Off labels
    ↓
[checklist_build] → 14-item bull market scoring
    ↓
Dashboard (Streamlit) / Excel / HTML / PNG exports
```

Two additional data sources run **lazily on-demand inside the dashboard** (not part of the weekly pipeline):
- `breadth_fetch.py` → Market breadth (S&P 500/400/600, NYSE, NASDAQ, Russell), cached to `breadth_latest.parquet` (12h TTL)
- `treasury_fetch.py` → Treasury tax flow indicators via fiscaldata.treasury.gov (no API key), cached to `treasury_tax_latest.parquet` (12h TTL)

### Module Layout
- `risk_index/core/` - Config, types, utilities (`constants.py`, `types.py`, `utils_io.py`, `exceptions.py`)
- `risk_index/sources/` - Data providers: `yahoo.py`, `fred.py`, `fiscal_data.py` (all extend `base.py` ABC)
- `risk_index/pipeline/` - ETL stages (fetch → clean_align → features_build → blocks_build → composite_build → regimes_build → checklist_build) plus dashboard-only breadth_fetch and treasury_fetch
- `risk_index/reporting/` - Output: `dashboard.py`, `charts.py`, `attribution.py`, `report_html.py`, `excel_export.py`
- `risk_index/research/` - Analysis: `lead_lag.py`, `walk_forward.py`, `feature_selection.py`, `weight_optimization.py`, `backtest_regimes.py`

### Configuration
All parameters are YAML-driven in `config/`:
- `universe.yml` - Data series, ratios, computed series, block definitions
- `transforms.yml` - Feature definitions (z-scores, ROC, slopes, drawdown, realized_vol)
- `composites.yml` - Block weights per speed (fast/medium/slow)
- `regimes.yml` - Thresholds, hysteresis buffer, confidence scoring
- `checklist.yml` - 14-item bull market scoring
- `sources.yml` - Data provider configurations (yahoo, fred, fiscal_data)
- `backtest.yml` - Walk-forward, selection rules, prediction targets
- `dashboard.yml` - Sector scorecard thresholds, breadth thresholds, block heatmap settings

Load configs via `load_config("name")` → returns Pydantic-validated objects. Dashboard-specific thresholds loaded via `load_dashboard_config()` from `dashboard.yml`.

### Key Conventions
- Feature naming: `TICKER__transform_window` (double underscore, e.g., `HYG_IEF__z_52w`)
- Weekly frequency: Friday close alignment (`W-FRI`)
- Enums in `risk_index/core/types.py`:
  - `Regime`: RISK_ON, NEUTRAL, RISK_OFF
  - `Speed`: FAST, MEDIUM, SLOW
  - `Signal`: BULL, WATCH, BEAR
  - `BlockName`: 10 blocks (equity_leadership, factor_prefs, credit, rates, fx, commodities, vol_liquidity, global, defensive_rotation, macro_slow)
  - `ChecklistLabel`: CONFIRMED_RISK_ON, ON_WATCH, RISK_OFF
- Data I/O: Use `read_parquet()`, `write_parquet()`, `read_yaml()` from `utils_io.py`
- Logging: Use `log_step()` for pipeline stages with metadata
- Config errors raise `ConfigError` from `risk_index.core.exceptions`

### Dashboard Patterns
- 10 tabs: Composites, Blocks, Checklist, Attribution, Composition, Backtest, Factor Leadership, Market Breadth, Sector Scorecard, Treasury Tax Flow
- Composites and Attribution tabs have nested sub-tabs per speed (Fast/Medium/Slow)
- Use `st.expander()` for explanation sections (collapsed by default)
- Color styling via `df.style.apply()` with RGBA colors
- Chart colors defined in `charts.py` `COLORS` dict (risk_on=#2ecc71, neutral=#f39c12, risk_off=#e74c3c)

## Environment

Requires Python >=3.9. Build system: hatchling. Optional `FRED_API_KEY` in `.env` for treasury/macro data. The `fiscal_data.py` source (Treasury tax flows) requires no API key.
