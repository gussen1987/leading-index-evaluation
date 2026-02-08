# Lessons Learned

## Project: Risk-On/Risk-Off Regime Index

### Configuration Design

1. **Use Pydantic for config validation** - Catches errors at load time with clear messages
2. **Separate concerns in YAML files** - universe.yml for data, transforms.yml for features, composites.yml for aggregation
3. **Include sensible defaults** - ffill limits, window sizes, weights should have defaults

### Data Pipeline

1. **Cache aggressively** - Yahoo/FRED rate limits are real; cache at parquet level
2. **Handle FRED staleness explicitly** - Not all "missing" data is obsolete; some is just publication lag
3. **Forward-fill with limits** - Weekly data needs ffill but with sensible limits per series type

### Feature Engineering

1. **Standardize naming convention early** - `{series}__{transform}` makes column selection easy
2. **Always winsorize and replace inf** - Financial data has outliers; catch them systematically
3. **Z-scores need sufficient history** - 52-week z-score needs 52 weeks of data

### Regime Classification

1. **Hysteresis prevents whipsaw** - Buffer of 0.15 and min 2 weeks reduces noise
2. **Confidence scoring aids interpretation** - Users trust signals more when they understand confidence
3. **Block correlation matters** - Highly correlated blocks can dominate composites

### Reporting

1. **Attribution is critical for trust** - Show what's driving the signal, not just the signal
2. **Multiple export formats serve different users** - Excel for analysis, HTML for viewing, Streamlit for interaction
3. **Save both _latest and dated versions** - Makes reproducibility possible while keeping API simple

### Research & Optimization (Phase 2)

1. **Walk-forward validation is essential** - In-sample IC is meaningless; always use OOS validation
2. **Purge gaps prevent lookahead bias** - When forward return horizon is N weeks, need N+ week gap between train/test
3. **Multiple comparison protection is critical** - With 300+ features, many will show spurious IC by chance
4. **Sign consistency matters as much as IC magnitude** - A feature with 0.05 IC but 90% sign consistency beats 0.08 IC with 50% consistency
5. **Redundancy removal prevents overfitting** - Correlated features add noise, not information
6. **Equal weighting often beats complex schemes** - Only optimize when OOS improvement is demonstrated
7. **IC-weighted optimization is a good middle ground** - More stable than max_ic, more informed than equal
8. **Sign consistency is the hardest gate to pass** - Most features have unstable IC signs across train/test, indicating regime sensitivity
9. **Skip slow rolling IC stats for large feature sets** - Use `--skip-ic-stats` flag when testing; the static IC matrix is sufficient for selection
