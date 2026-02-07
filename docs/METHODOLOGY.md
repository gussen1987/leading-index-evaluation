# Signal Methodology

This document describes the composition of the Risk-On/Risk-Off Regime Index, including signal selection, block construction, and composite weighting.

---

## Overview

The regime model combines **10 thematic blocks** into **3 composite signals** at different time horizons:

| Composite | Horizon | Purpose |
|-----------|---------|---------|
| Fast | 4-8 weeks | Tactical risk positioning, tail drawdown reduction |
| Medium | 13-26 weeks | Regime classification, drawdown anticipation |
| Slow | 26-52 weeks | Macro cycle alignment, long-term risk assessment |

---

## Signal Blocks

### 1. Equity Leadership

Measures risk appetite through equity market rotation patterns.

| Member | Features | Description |
|--------|----------|-------------|
| RSP/SPY | z_52w, roc_8w | Equal weight vs cap weight (breadth proxy) |
| IWM/SPY | z_52w, roc_8w | Small cap vs large cap |
| XLY/XLP | z_52w, roc_8w | Discretionary vs staples |
| XLF/XLU | z_52w, roc_8w | Financials vs utilities |
| XLK/XLU | z_52w, roc_8w | Technology vs utilities |
| XLI/XLU | z_52w, roc_8w | Industrials vs utilities |

**Interpretation:** Rising ratios = risk-on leadership.

### 2. Factor Preferences

Measures risk appetite through factor rotation.

| Member | Features | Description |
|--------|----------|-------------|
| SPHB/SPLV | z_52w, roc_8w | High beta vs low volatility |
| MTUM/USMV | z_52w, roc_8w | Momentum vs minimum volatility |
| QUAL/USMV | z_52w, roc_8w | Quality vs minimum volatility |

**Interpretation:** Rising ratios = preference for aggressive factors.

### 3. Credit

Measures credit market conditions and spread dynamics.

| Member | Features | Invert? | Description |
|--------|----------|---------|-------------|
| HYG/IEF | z_52w, roc_8w | No | High yield vs treasuries |
| LQD/IEF | z_52w, roc_8w | No | Investment grade vs treasuries |
| BAMLH0A0HYM2 | z_52w, roc_8w | Yes | ICE BofA HY OAS |
| BAMLC0A0CM | z_52w | Yes | ICE BofA IG OAS |

**Interpretation:** Rising ETF ratios + falling spreads = risk-on credit.

### 4. Rates

Measures yield curve and inflation expectations.

| Member | Features | Description |
|--------|----------|-------------|
| T10Y2Y | z_104w, roc_13w | 10Y-2Y treasury spread |
| DFII10 | z_104w | 10-year TIPS real yield |
| T10YIE | z_52w, roc_8w | 10-year breakeven inflation |

**Interpretation:** Steepening curve + stable inflation = constructive.

### 5. FX

Measures risk appetite through currency pairs.

| Member | Features | Invert? | Description |
|--------|----------|---------|-------------|
| AUDJPY | z_52w, roc_8w | No | AUD/JPY cross (risk barometer) |
| CADJPY | z_52w, roc_8w | No | CAD/JPY cross |
| UUP | z_52w, roc_8w | Yes | US Dollar Index ETF |

**Interpretation:** Rising risk FX + falling dollar = risk-on.

### 6. Commodities

Measures growth expectations through commodity ratios.

| Member | Features | Description |
|--------|----------|-------------|
| COPX/GLD | z_52w, roc_8w | Copper miners vs gold |
| USO/GLD | z_52w, roc_8w | Oil vs gold |
| DBC/GLD | z_52w, roc_8w | Commodities vs gold |

**Interpretation:** Rising ratios = growth/inflation expectations.

### 7. Volatility & Liquidity

Measures market stress and financial conditions.

| Member | Features | Invert? | Description |
|--------|----------|---------|-------------|
| VIX | z_52w, roc_8w | Yes | CBOE Volatility Index |
| VVIX | z_52w | Yes | VIX of VIX |
| NFCI | z_104w | Yes | Chicago Fed Financial Conditions |
| RATES_VOL_PROXY_21D | z_52w, pctile_52w | Yes | TLT 21-day realized vol |

**Interpretation:** Low vol + easy conditions = risk-on.

**Note:** RATES_VOL_PROXY is TLT realized vol, NOT the MOVE index.

### 8. Global

Measures global risk appetite through relative performance.

| Member | Features | Description |
|--------|----------|-------------|
| EEM/SPY | z_52w, roc_13w | Emerging markets vs US |
| EFA/SPY | z_52w, roc_13w | Developed ex-US vs US |
| ACWX/SPY | z_52w, roc_13w | All world ex-US vs US |
| FXI/SPY | z_52w, roc_8w | China vs US |

**Interpretation:** Global outperformance = risk-on.

### 9. Defensive Rotation

Measures flight to safety (inverted for risk-on).

| Member | Features | Invert? | Description |
|--------|----------|---------|-------------|
| XLU/SPY | z_52w, roc_8w | Yes | Utilities vs market |
| TLT/SHY | z_52w, roc_8w | Yes | Long duration vs short |
| GLD/SPY | z_52w, roc_8w | Yes | Gold vs market |

**Interpretation:** Defensive underperformance = risk-on.

### 10. Macro Slow

Slow-moving macroeconomic indicators.

| Member | Features | Invert? | Description |
|--------|----------|---------|-------------|
| ICSA | z_104w, roc_26w | Yes | Initial jobless claims |
| PERMIT | z_104w, roc_26w | No | Building permits |
| UMCSENT | z_104w, roc_26w | No | U of Michigan consumer sentiment |

**Interpretation:** Low claims + rising permits/sentiment = risk-on.

---

## Composite Construction

### Fast Composite (4-8 week horizon)

Optimized for tactical positioning and tail risk reduction.

| Block | Weight | Rationale |
|-------|--------|-----------|
| equity_leadership | 1.5 | Direct equity risk appetite |
| vol_liquidity | 1.5 | Early stress detection |
| factor_prefs | 1.2 | Factor rotation signals |
| credit | 1.0 | Credit market confirmation |
| defensive_rotation | 0.8 | Flight to safety detection |

**Z-Window:** 104 weeks (2 years)

### Medium Composite (13-26 week horizon)

Optimized for regime classification and drawdown anticipation.

| Block | Weight | Rationale |
|-------|--------|-----------|
| credit | 1.5 | Credit leads equity |
| rates | 1.2 | Yield curve signal |
| global | 1.2 | Global confirmation |
| vol_liquidity | 1.0 | Stress conditions |
| fx | 1.0 | Currency risk appetite |
| equity_leadership | 0.8 | Equity confirmation |
| commodities | 0.6 | Growth expectations |

**Z-Window:** 156 weeks (3 years)

### Slow Composite (26-52 week horizon)

Optimized for macro cycle alignment and long-term positioning.

| Block | Weight | Rationale |
|-------|--------|-----------|
| rates | 1.5 | Macro leading indicator |
| credit | 1.5 | Credit cycle |
| macro_slow | 1.5 | Direct macro indicators |
| global | 1.0 | Global cycle |
| commodities | 0.8 | Commodity cycle |
| fx | 0.8 | Currency trends |

**Z-Window:** 260 weeks (5 years)

---

## Regime Classification

### Thresholds

| Regime | Condition |
|--------|-----------|
| Risk-On | Composite > 0.50 |
| Neutral | -0.50 ≤ Composite ≤ 0.50 |
| Risk-Off | Composite < -0.50 |

### Hysteresis

- **Buffer:** 0.15 (must cross threshold ± buffer to exit regime)
- **Minimum hold:** 2 weeks before regime switch

### Confidence Score

```
confidence = dispersion × 0.60 + (1 - tail_penalty) × 0.20 + (1 - liquidity_penalty) × 0.20
```

Where:
- **Dispersion:** Fraction of blocks agreeing with composite sign
- **Tail penalty:** Applied if VIX z > 1.5 or VVIX z > 1.5
- **Liquidity penalty:** Applied if NFCI > 0

---

## Bull Market Checklist

14-item scoring system across 7 categories:

### Trend (2 items)
- SPY Above 200-Day MA (weight: 1.5)
- SPY Above 50-Day MA (weight: 1.0)

### Momentum (2 items)
- SPY RSI Healthy (40-70) (weight: 1.0)
- Medium Composite Positive (weight: 1.2)

### Breadth (2 items) - Proxy Breadth v1
- Equal Weight vs Cap Weight (weight: 1.2)
- Small Cap vs Large Cap (weight: 1.0)

### Credit (2 items)
- High Yield vs Treasuries (weight: 1.5)
- HY Spreads Contained (weight: 1.2)

### Rates (2 items)
- Yield Curve Not Inverted (weight: 1.2)
- Breakeven Inflation Stable (weight: 0.8)

### FX (2 items)
- AUD/JPY Trend Positive (weight: 1.0)
- Dollar Not Surging (weight: 0.8)

### Volatility (2 items)
- VIX Contained (weight: 1.5)
- Financial Conditions Normal (weight: 1.0)

### Scoring

- Each item: Bull (1.0) / Watch (0.5) / Bear (0.0)
- Weighted sum scaled to 0-100
- Labels: ≥75 = "Confirmed Risk-On", ≥50 = "On Watch", <50 = "Risk-Off"

---

## Data Gaps & Limitations

### Not Currently Included

1. **FRED Data:** Requires API key configuration
   - Treasury rates (DGS10, DGS2, T10Y2Y)
   - Real rates and breakevens (DFII10, T10YIE)
   - Credit spreads (BAMLH0A0HYM2, BAMLC0A0CM)
   - Macro indicators (NFCI, ICSA, PERMIT, UMCSENT)

2. **MOVE Index:** Using TLT realized vol as proxy
   - Labeled honestly as `RATES_VOL_PROXY_21D`
   - Different behavior during rate shocks

3. **True Breadth:** Using proxy breadth (v1)
   - RSP/SPY ratio instead of constituent advance/decline
   - IWM/SPY instead of Russell 2000 breadth
   - No survivorship handling

### Future Enhancements (v2)

- FRED data integration with API key
- Constituent-level breadth with survivorship handling
- MOVE index direct integration
- Walk-forward feature selection
- Regime-specific weight optimization
