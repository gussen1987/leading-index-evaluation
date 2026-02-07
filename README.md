# Risk-On/Risk-Off Regime Index

A Python system for constructing intermarket risk signals and classifying market regimes.

## Features

- Fetches market data from Yahoo Finance and FRED
- Constructs intermarket risk signals using ratios and transforms
- Outputs regime labels (Risk-On / Neutral / Risk-Off) at 3 speeds
- Produces a Bull Market Checklist with weighted scoring
- Generates HTML reports, Streamlit dashboard, and Excel exports

## Installation

```bash
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and add your FRED API key:

```
FRED_API_KEY=your_key_here
```

## Usage

```bash
# Run weekly update
python scripts/run_update.py

# Launch dashboard
python scripts/run_dashboard.py

# Regenerate reports
python scripts/run_report.py
```

## License

Private use only.
