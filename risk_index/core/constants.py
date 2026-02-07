"""Constants and naming conventions for the risk index system."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
EXPORTS_DIR = DATA_DIR / "exports"
LOGS_DIR = DATA_DIR / "logs"

# Data alignment
WEEK_ANCHOR = "W-FRI"  # Weekly frequency aligned to Friday close
MIN_COVERAGE_RATIO = 0.80  # Minimum data coverage for blocks

# Forward-fill limits (business days)
FFILL_LIMITS = {
    "price": 5,
    "fx": 3,
    "vol": 2,
    "macro_daily": 10,
    "macro_weekly": 21,
    "macro_monthly": 60,
}

# Feature naming convention
FEATURE_SEPARATOR = "__"  # e.g., HYG_IEF__z_52w

# Transform windows (weeks)
TRANSFORM_WINDOWS = {
    "z_52w": 52,
    "z_104w": 104,
    "z_156w": 156,
    "z_260w": 260,
    "pctile_52w": 52,
    "pctile_104w": 104,
    "roc_4w": 4,
    "roc_8w": 8,
    "roc_13w": 13,
    "roc_26w": 26,
    "slope_13w": 13,
    "slope_26w": 26,
    "slope_52w": 52,
}

# Regime thresholds
REGIME_THRESHOLD_RISK_ON = 0.50
REGIME_THRESHOLD_RISK_OFF = -0.50
REGIME_HYSTERESIS_BUFFER = 0.15
MIN_WEEKS_IN_REGIME = 2

# Checklist scoring
CHECKLIST_SCORE_BULL = 1.0
CHECKLIST_SCORE_WATCH = 0.5
CHECKLIST_SCORE_BEAR = 0.0

# Checklist labels
CHECKLIST_CONFIRMED_RISK_ON = 75
CHECKLIST_ON_WATCH = 50

# Confidence scoring weights
CONFIDENCE_WEIGHT_DISPERSION = 0.60
CONFIDENCE_WEIGHT_TAIL = 0.20
CONFIDENCE_WEIGHT_LIQUIDITY = 0.20

# Tail/liquidity thresholds
VIX_TAIL_THRESHOLD_Z = 1.5
VVIX_TAIL_THRESHOLD_Z = 1.5
NFCI_LIQUIDITY_THRESHOLD = 0.0

# Winsorization percentiles
WINSORIZE_LOWER = 0.005
WINSORIZE_UPPER = 0.995

# Cache settings
CACHE_MAX_AGE_DAYS = 1

# Walk-forward settings
PURGE_GAP_WEEKS = 2
EMBARGO_WEEKS = 1

# Block correlation policy
MAX_BLOCK_CORRELATION = 0.90
MAX_COMBINED_WEIGHT = 0.25

# Target horizons (weeks)
FAST_HORIZONS = [4, 8]
MEDIUM_HORIZONS = [13, 26]
SLOW_HORIZONS = [26, 52]

# Column name patterns
COL_COMPOSITE_FAST = "composite_fast"
COL_COMPOSITE_MEDIUM = "composite_medium"
COL_COMPOSITE_SLOW = "composite_slow"
COL_REGIME_FAST = "regime_fast"
COL_REGIME_MEDIUM = "regime_medium"
COL_REGIME_SLOW = "regime_slow"
COL_CONFIDENCE_FAST = "confidence_fast"
COL_CONFIDENCE_MEDIUM = "confidence_medium"
COL_CONFIDENCE_SLOW = "confidence_slow"
COL_CHECKLIST_SCORE = "checklist_score"
COL_CHECKLIST_LABEL = "checklist_label"
