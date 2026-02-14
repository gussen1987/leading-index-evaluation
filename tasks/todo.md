# Risk-On/Risk-Off Regime Index Implementation

## Completed Tasks

- [x] Task 1: Project Skeleton + Config Schemas
  - Created full directory structure
  - pyproject.toml with all dependencies
  - .env.example and .gitignore
  - Core modules: config_schema.py, constants.py, types.py, exceptions.py, utils_dates.py, utils_io.py, utils_math.py, logger.py
  - All YAML configs: sources.yml, universe.yml, transforms.yml, composites.yml, regimes.yml, checklist.yml, backtest.yml

- [x] Task 2: Data Providers (Yahoo + FRED)
  - base.py: Abstract DataProvider class
  - yahoo.py: YahooProvider with batch fetching and retry logic
  - fred.py: FredProvider with series status checking
  - pipeline/fetch.py: fetch_all() with caching

- [x] Task 3: Weekly Alignment + Ratios + Computed Series
  - pipeline/clean_align.py: clean_and_align() function
  - Weekly Friday resampling
  - Forward-fill with series-specific limits
  - Ratio computation
  - Computed series (volatility, RSI, MA signals)

- [x] Task 4: Feature Engineering
  - pipeline/features_build.py: build_features() function
  - Transform functions: zscore, percentile, roc, slope, drawdown, realized_vol
  - Feature naming convention: {series_id}__{transform_name}
  - Winsorization and inf replacement

- [x] Task 5: Block Scoring
  - pipeline/blocks_build.py: build_blocks() function
  - 10 block scores computed
  - Inversion handling for risk-off signals
  - Correlation policy enforcement

- [x] Task 6: Composites + Regimes + Confidence
  - pipeline/composite_build.py: build_composites() function
  - pipeline/regimes_build.py: build_regimes() function
  - Fast/Medium/Slow composites with weighted blocks
  - Regime classification with hysteresis
  - Confidence scoring (dispersion + tail + liquidity penalties)

- [x] Task 7: Checklist Engine
  - pipeline/checklist_build.py: build_checklist() function
  - 14 checklist items with threshold/trend/compound rules
  - Weighted scoring (0-100 scale)
  - Labels: Confirmed Risk-On / On Watch / Risk-Off

- [x] Task 8: Reporting (HTML, Excel, Streamlit)
  - reporting/charts.py: Plotly chart generation
  - reporting/attribution.py: Block and feature attribution
  - reporting/excel_export.py: 5 Excel workbooks
  - reporting/report_html.py: Interactive HTML report
  - reporting/dashboard.py: Streamlit dashboard
  - Entry scripts: run_update.py, run_report.py, run_dashboard.py, run_research.py

## Phase 2: Empirical Signal Selection (COMPLETED)

- [x] Research Module Implementation
  - [x] research/lead_lag.py: Lead-lag IC analysis
    - compute_lead_lag_matrix() - IC at 5 horizons (4, 8, 13, 26, 52 weeks)
    - compute_ic_with_stats() - IC with t-stats, p-values, sign consistency
    - compute_rolling_ic() - Rolling IC over time
    - rank_features_by_ic() - Feature ranking by average IC
  - [x] research/walk_forward.py: Rolling OOS validation
    - walk_forward_splits() - Train/test splits with purge/embargo gaps
    - evaluate_features_walk_forward() - IC evaluation per split
    - aggregate_walk_forward_results() - Aggregate metrics per feature × horizon
  - [x] research/feature_selection.py: Robust selection with multiple comparison protection
    - SelectionRules dataclass with 5 gates
    - select_features() - Apply all gates per speed category
    - compute_feature_scores() - Composite ranking scores
  - [x] research/weight_optimization.py: Block weight tuning
    - optimize_block_weights() - 4 methods (equal, inverse_variance, ic_weighted, max_ic)
    - generate_optimized_composites_config() - Output new composites.yml
    - compare_weights() / validate_weights() - Before/after analysis

- [x] Universe Expansion
  - Added 27 new Yahoo ETFs (TIP, KRE, VNQ, SMH, XHB, IBB, JNK, EMB, etc.)
  - Added 20+ new FRED series (DGS5, DGS30, T10Y3M, DFII5, BAMLH0A1HYBB, CCSA, etc.)
  - Added 15+ new ratios (TIP/IEF, KRE/VNQ, JNK/TLT, EMB/AGG, FXA/FXY, etc.)
  - Added new blocks: inflation, housing

## Phase 3: Dashboard Enhancements (COMPLETED)

- [x] Normalized Weights
  - Converted all composite weights from relative (1.5, 1.2, etc.) to normalized (0-1, sum to 1.0)
  - Fast: 5 blocks (equity_leadership 25%, vol_liquidity 25%, factor_prefs 20%, etc.)
  - Medium: 9 blocks (credit 17.2%, rates 13.8%, global 13.8%, etc.)
  - Slow: 7 blocks (rates 19.2%, credit 19.2%, macro_slow 19.2%, etc.)

- [x] Backtest Module (research/backtest_regimes.py)
  - load_spy_prices() - Load and align SPY data
  - backtest_spy_regime() - Single composite backtest with metrics
  - run_all_regime_backtests() - Run all three composites
  - create_backtest_summary() - Summary DataFrame for display
  - Metrics: total return, CAGR, max drawdown, Sharpe, trades, win rate

- [x] Dashboard Composition Tab
  - Shows block weights per composite as table
  - Current block scores and weighted contributions
  - Pie chart visualization of weights

- [x] Dashboard Backtest Tab
  - Performance summary table (Fast/Medium/Slow vs Buy & Hold)
  - Equity curves chart comparing strategies
  - Expandable trade logs per composite
  - Key metrics: total return, win rate, max drawdown

## Phase 4: Dashboard Professional Enhancements (COMPLETED)

- [x] Fix Checklist Chart
  - Added proper empty data handling with "No data available" annotations
  - Used `.values` explicitly to avoid Plotly issues
  - Added Risk-Off threshold line at y=25

- [x] Enhanced Checklist Table
  - Color-coded rows: green (bull), red (bear), yellow (neutral)
  - Direction column: Improving/Flat/Declining based on previous period
  - Prominent regime label: "Confirmed Bull Market" / "On Watch" / "Risk-Off"
  - Score summary: "X of Y Bullish" with percentage

- [x] Composition Tab Enhancements
  - Expandable block details showing all member ratios
  - RATIO_DESCRIPTIONS mapping with 70+ entries
  - Shows: Ratio ID, Description, Tickers, Category, Inverted flag

- [x] Multi-Panel SPY Regime Chart
  - Daily Number-style visualization
  - Panel 1: SPY price with regime shading
  - Panel 2: Composite score oscillator with threshold lines
  - Helper function for subplot regime shading

- [x] Factor Leadership Tab (New)
  - 6 key factor metrics: Size, Style, Risk Appetite, Cyclical, Credit, Global
  - Color-coded leader display (green=risk-on, red=defensive)
  - Z-score and trend indicators
  - Integrated multi-panel SPY chart

## Phase 5: Dashboard Quality + Treasury Tab (COMPLETED)

### Dashboard Quality Fixes
- [x] Issue 1: Checklist Direction Logic
  - Fixed incomplete elif chain causing neutral->bull to show "Flat"
  - Now uses signal ordering (bear=0, neutral=1, bull=2) for comparison
  - File: dashboard.py lines 583-595

- [x] Issue 2: Attribution Tab Waterfall Chart
  - Added import for create_attribution_chart()
  - Tab 4 now displays waterfall chart with proper attribution data
  - Shows contribution table below each chart
  - File: dashboard.py tab4 section

- [x] Issue 3: Finviz Scraping Robustness
  - Added try-catch with explicit error messages per field
  - Returns status dict: {"status": "success/partial/failed", "fields_parsed": N}
  - Logs warnings when fields can't be parsed
  - File: breadth_fetch.py fetch_finviz_breadth()

- [x] Issue 4: Sector Grading Thresholds from Config
  - Created config/dashboard.yml with sector scorecard thresholds
  - Added load_dashboard_config() function
  - Sector scorecard now reads thresholds dynamically
  - File: dashboard.py, config/dashboard.yml

- [x] Issue 5: Block Heatmap Respects Date Range
  - Changed from hardcoded "52 Weeks" to calculated weeks from filtered_blocks
  - Title now shows actual number of weeks displayed
  - File: dashboard.py tab2 section

### Treasury Tax Flow Tab (Tab 10)
- [x] Created fiscal_data.py provider
  - FiscalDataProvider class following fred.py pattern
  - Handles schema change (Feb 14, 2023)
  - Fetches from fiscaldata.treasury.gov API
  - File: sources/fiscal_data.py

- [x] Created treasury_fetch.py pipeline
  - fetch_tax_deposits() with 12-hour cache
  - calculate_rolling_sums() for noise reduction
  - calculate_yoy_growth() with business day alignment
  - calculate_ytd_cumulative() for year-over-year comparison
  - prepare_treasury_indicators() main entry point
  - Gig economy, corporate profits, labor market indicators
  - File: pipeline/treasury_fetch.py

- [x] Added treasury chart functions to charts.py
  - create_tax_yoy_chart() - YoY growth time series
  - create_ytd_comparison_chart() - This year vs last year (Deluard style)
  - create_tax_vs_spy_chart() - Overlay with SPY
  - create_gig_economy_chart() - Non-withheld taxes as gig economy proxy
  - File: charts.py

- [x] Added Tab 10 - Treasury Tax Flow
  - Educational expander explaining Deluard methodology
  - Control panel: rolling window selector, category multiselect
  - Chart 1: YoY Growth by Tax Category
  - Chart 2: YTD Cumulative (This Year vs Last Year)
  - Chart 3: Gig Economy Indicator
  - Chart 4: Tax Receipts vs S&P 500
  - Data freshness indicator
  - Refresh button
  - File: dashboard.py tab10 section

- [x] Updated config/sources.yml with fiscal_data source

## Remaining

- [ ] Email Report
  - reporting/email_report.py: Weekly summary via Gmail

## Next Steps (Validation) - COMPLETED

- [x] Run: `python scripts/run_research.py --skip-ic-stats`
- [x] Verify data/artifacts/lead_lag_matrix.parquet (1204 features × 5 horizons)
- [x] Check selected_features_*.json - 0 features passed strict gates (expected behavior)
- [x] Compare old vs new composite IC - Mixed results, medium composite improved
- [x] Walk-forward validation: 28 splits, 168,560 evaluations

### Key Findings:
- 1066 features have |IC| > 0.05 at some horizon
- Top predictors: ACWX_SPY (global vs US), CADJPY (risk proxy), USO_GLD (commodities)
- Sign consistency < 70% for all features (strict gate working correctly)
- Selection gates filter unstable predictors - validates conservative approach
- IC-weighted optimization completed for all composites

## Usage

```bash
# Install dependencies
pip install -e .

# Run weekly update
python scripts/run_update.py

# Launch dashboard
python scripts/run_dashboard.py

# Regenerate reports only
python scripts/run_report.py
```

## Review

Implementation follows the specification with:
- Price-based regime model (not total return)
- Secrets from .env only
- Overwrite *_latest.* pattern for outputs
- Streamlit dashboard included
- FRED staleness handling with status checks
- Hysteresis for regime classification
- Attribution module for transparency
