# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Risk-On/Risk-Off Regime Index - A quantitative system that classifies market regimes by aggregating 100+ intermarket signals into 10 thematic blocks, then combining them into Fast/Medium/Slow composite scores.

## Commands

### Development
```bash
pip install -e .              # Install package
pip install -e ".[dev]"       # Install with dev dependencies
```

### Run Pipeline
```bash
python scripts/run_update.py [--force-refresh] [--archive]   # Weekly data update
python scripts/run_dashboard.py                               # Launch Streamlit dashboard
python scripts/run_research.py [--optimization-method {equal|inverse_variance|ic_weighted|max_ic}]  # Optimization
```

### Testing & Linting
```bash
pytest tests/                  # Run all tests
pytest tests/test_config_schema.py  # Single test file
ruff check .                   # Lint
ruff format .                  # Format
```

## Architecture

### Data Flow
```
Yahoo Finance / FRED APIs
    ↓
[fetch] → Raw daily series (data/cache/)
    ↓
[clean_align] → Weekly Friday alignment
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
Dashboard (Streamlit) / Excel / HTML exports
```

### Module Layout
- `risk_index/core/` - Config, types, utilities (constants.py, types.py, utils_*.py)
- `risk_index/sources/` - Data providers (yahoo.py, fred.py)
- `risk_index/pipeline/` - ETL stages (fetch → clean_align → features_build → blocks_build → composite_build → regimes_build)
- `risk_index/reporting/` - Output (dashboard.py, charts.py, excel_export.py)
- `risk_index/research/` - Backtesting and optimization (lead_lag.py, walk_forward.py, weight_optimization.py)

### Configuration
All parameters are YAML-driven in `config/`:
- `universe.yml` - Data series, ratios, computed series, block definitions
- `transforms.yml` - Feature definitions (z-scores, ROC, slopes)
- `composites.yml` - Block weights per speed (fast/medium/slow)
- `regimes.yml` - Thresholds, hysteresis buffer, confidence scoring
- `checklist.yml` - 14-item bull market scoring

### Key Conventions
- Feature naming: `TICKER__transform_window` (e.g., `HYG_IEF__z_52w`)
- Weekly frequency: Friday close alignment (`W-FRI`)
- Block/regime enums: Use `BlockName`, `Regime`, `Speed` from `risk_index/core/types.py`
- Config loading: Use `load_config(name)` which returns Pydantic-validated objects
- Data I/O: Use `read_parquet()`, `write_parquet()`, `read_yaml()` from `utils_io.py`
- Logging: Use `log_step()` for pipeline stages with metadata

### Dashboard Patterns
- Use `st.expander()` for explanation sections (collapsed by default)
- Color styling via `df.style.apply()` with RGBA colors
- Dashboard has 9 tabs: Composites, Blocks, Checklist, Attribution, Composition, Backtest, Factor Leadership, Market Breadth, Sector Scorecard
- Breadth data cached to `breadth_latest.parquet` with 12-hour TTL

## Environment

Requires Python >=3.9. Optional FRED_API_KEY in `.env` for treasury/macro data.
