"""Type definitions and enums for the risk index system."""

from __future__ import annotations

from enum import Enum
from typing import Dict

import pandas as pd


class Regime(str, Enum):
    """Market regime classification."""

    RISK_ON = "Risk-On"
    NEUTRAL = "Neutral"
    RISK_OFF = "Risk-Off"


class Signal(str, Enum):
    """Checklist signal status."""

    BULL = "bull"
    WATCH = "watch"
    BEAR = "bear"


class BlockName(str, Enum):
    """Block identifiers for composite construction."""

    EQUITY_LEADERSHIP = "equity_leadership"
    FACTOR_PREFS = "factor_prefs"
    CREDIT = "credit"
    RATES = "rates"
    FX = "fx"
    COMMODITIES = "commodities"
    VOL_LIQUIDITY = "vol_liquidity"
    GLOBAL = "global"
    DEFENSIVE_ROTATION = "defensive_rotation"
    MACRO_SLOW = "macro_slow"


class Speed(str, Enum):
    """Composite speed classification."""

    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"


class ChecklistLabel(str, Enum):
    """Checklist aggregate labels."""

    CONFIRMED_RISK_ON = "Confirmed Risk-On"
    ON_WATCH = "On Watch"
    RISK_OFF = "Risk-Off"


# Type aliases (for documentation purposes)
SeriesDict = Dict[str, pd.Series]
FeatureFrame = pd.DataFrame
BlockScores = pd.DataFrame
