"""HTML report generation module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Template

from risk_index.core.config_schema import ChecklistConfig
from risk_index.core.constants import (
    ARTIFACTS_DIR,
    COL_COMPOSITE_FAST,
    COL_COMPOSITE_MEDIUM,
    COL_COMPOSITE_SLOW,
    COL_REGIME_FAST,
    COL_REGIME_MEDIUM,
    COL_REGIME_SLOW,
    COL_CONFIDENCE_FAST,
    COL_CONFIDENCE_MEDIUM,
    COL_CONFIDENCE_SLOW,
    COL_CHECKLIST_SCORE,
    COL_CHECKLIST_LABEL,
)
from risk_index.core.types import Regime, Signal
from risk_index.core.utils_io import ensure_dir
from risk_index.core.logger import get_logger
from risk_index.reporting.charts import (
    create_composite_timeseries,
    create_block_bars,
    create_block_heatmap,
    create_checklist_score_chart,
    create_regime_distribution,
)

logger = get_logger(__name__)


HTML_TEMPLATE = """

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Risk Regime Report - {{ date }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 20px;
            margin-bottom: 30px;
            border-radius: 10px;
        }
        header h1 { font-size: 2em; margin-bottom: 10px; }
        header .date { opacity: 0.9; font-size: 1.1em; }

        .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        .card h3 {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .card .value { font-size: 1.8em; font-weight: bold; }
        .card .label { font-size: 1.1em; margin-top: 5px; }
        .card .confidence { font-size: 0.9em; color: #888; margin-top: 5px; }

        .regime-risk-on { color: #27ae60; }
        .regime-neutral { color: #f39c12; }
        .regime-risk-off { color: #e74c3c; }

        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        .chart-container h2 {
            margin-bottom: 15px;
            color: #444;
        }

        .checklist-table {
            width: 100%;
            border-collapse: collapse;
        }
        .checklist-table th, .checklist-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .checklist-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #666;
        }
        .signal-bull { color: #27ae60; }
        .signal-watch { color: #f39c12; }
        .signal-bear { color: #e74c3c; }

        .emoji { font-size: 1.2em; margin-right: 5px; }

        footer {
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Risk Regime Report</h1>
            <div class="date">{{ date }} | Generated {{ generated_at }}</div>
        </header>

        <!-- Regime Cards -->
        <div class="cards">
            <div class="card">
                <h3>Fast (4-8 Week)</h3>
                <div class="value regime-{{ fast_regime_class }}">{{ fast_regime }}</div>
                <div class="label">Composite: {{ fast_composite }}</div>
                <div class="confidence">Confidence: {{ fast_confidence }}%</div>
            </div>
            <div class="card">
                <h3>Medium (13-26 Week)</h3>
                <div class="value regime-{{ medium_regime_class }}">{{ medium_regime }}</div>
                <div class="label">Composite: {{ medium_composite }}</div>
                <div class="confidence">Confidence: {{ medium_confidence }}%</div>
            </div>
            <div class="card">
                <h3>Slow (26-52 Week)</h3>
                <div class="value regime-{{ slow_regime_class }}">{{ slow_regime }}</div>
                <div class="label">Composite: {{ slow_composite }}</div>
                <div class="confidence">Confidence: {{ slow_confidence }}%</div>
            </div>
            <div class="card">
                <h3>Bull Market Checklist</h3>
                <div class="value">{{ checklist_score }}/100</div>
                <div class="label regime-{{ checklist_class }}">{{ checklist_label }}</div>
            </div>
        </div>

        <!-- Composite Chart -->
        <div class="chart-container">
            <h2>Composite Signals</h2>
            <div id="composite-chart"></div>
        </div>

        <!-- Block Scores -->
        <div class="chart-container">
            <h2>Current Block Scores</h2>
            <div id="block-chart"></div>
        </div>

        <!-- Block Heatmap -->
        <div class="chart-container">
            <h2>Block Scores (Last 52 Weeks)</h2>
            <div id="heatmap-chart"></div>
        </div>

        <!-- Checklist Table -->
        <div class="chart-container">
            <h2>Checklist Items</h2>
            <table class="checklist-table">
                <thead>
                    <tr>
                        <th>Item</th>
                        <th>Category</th>
                        <th>Signal</th>
                        <th>Weight</th>
                    </tr>
                </thead>
                <tbody>
                    {% for item in checklist_items %}
                    <tr>
                        <td>{{ item.name }}</td>
                        <td>{{ item.category }}</td>
                        <td class="signal-{{ item.signal_class }}">
                            <span class="emoji">{{ item.emoji }}</span>{{ item.signal }}
                        </td>
                        <td>{{ item.weight }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- Checklist Score Chart -->
        <div class="chart-container">
            <h2>Checklist Score History</h2>
            <div id="checklist-chart"></div>
        </div>

        <footer>
            Risk Regime Report | Data as of {{ date }}
        </footer>
    </div>

    <script>
        // Composite Chart
        var compositeData = {{ composite_chart_json | safe }};
        Plotly.newPlot('composite-chart', compositeData.data, compositeData.layout);

        // Block Chart
        var blockData = {{ block_chart_json | safe }};
        Plotly.newPlot('block-chart', blockData.data, blockData.layout);

        // Heatmap Chart
        var heatmapData = {{ heatmap_chart_json | safe }};
        Plotly.newPlot('heatmap-chart', heatmapData.data, heatmapData.layout);

        // Checklist Chart
        var checklistData = {{ checklist_chart_json | safe }};
        Plotly.newPlot('checklist-chart', checklistData.data, checklistData.layout);
    </script>
</body>
</html>
"""


def generate_html_report(
    composites_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    checklist_df: pd.DataFrame,
    checklist_cfg: ChecklistConfig,
    output_dir: Path = ARTIFACTS_DIR,
    date: pd.Timestamp | None = None,
) -> Path:
    """Generate interactive HTML report.

    Args:
        composites_df: Composites DataFrame
        blocks_df: Blocks DataFrame
        regimes_df: Regimes DataFrame
        checklist_df: Checklist DataFrame
        checklist_cfg: Checklist configuration
        output_dir: Output directory
        date: Report date (defaults to latest)

    Returns:
        Path to generated report
    """
    ensure_dir(output_dir)

    if date is None:
        date = composites_df.index[-1]

    # Get latest values
    latest_composites = composites_df.loc[date] if date in composites_df.index else pd.Series()
    latest_regimes = regimes_df.loc[date] if date in regimes_df.index else pd.Series()
    latest_checklist = checklist_df.loc[date] if date in checklist_df.index else pd.Series()

    # Helper to get regime class
    def regime_class(regime_str):
        if regime_str == Regime.RISK_ON.value:
            return "risk-on"
        elif regime_str == Regime.RISK_OFF.value:
            return "risk-off"
        return "neutral"

    # Create charts
    composite_chart = create_composite_timeseries(composites_df, regimes_df)
    block_chart = create_block_bars(blocks_df, date)
    heatmap_chart = create_block_heatmap(blocks_df)
    checklist_chart = create_checklist_score_chart(checklist_df)

    # Prepare checklist items
    checklist_items = []
    for item in checklist_cfg.items:
        signal_col = f"{item.id}_signal"
        signal = (
            latest_checklist.get(signal_col, Signal.WATCH.value)
            if not latest_checklist.empty
            else Signal.WATCH.value
        )

        signal_class = signal.lower() if isinstance(signal, str) else "watch"
        emoji = {"bull": "\u2705", "watch": "\u26a0\ufe0f", "bear": "\u274c"}.get(signal_class, "\u26a0\ufe0f")

        checklist_items.append({
            "name": item.name,
            "category": item.category,
            "signal": signal,
            "signal_class": signal_class,
            "emoji": emoji,
            "weight": item.weight,
        })

    # Render template
    template = Template(HTML_TEMPLATE)

    html_content = template.render(
        date=date.strftime("%Y-%m-%d"),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        # Fast
        fast_regime=latest_regimes.get(COL_REGIME_FAST, "N/A"),
        fast_regime_class=regime_class(latest_regimes.get(COL_REGIME_FAST, "")),
        fast_composite=f"{latest_composites.get(COL_COMPOSITE_FAST, 0):.2f}",
        fast_confidence=f"{latest_regimes.get(COL_CONFIDENCE_FAST, 0) * 100:.0f}",
        # Medium
        medium_regime=latest_regimes.get(COL_REGIME_MEDIUM, "N/A"),
        medium_regime_class=regime_class(latest_regimes.get(COL_REGIME_MEDIUM, "")),
        medium_composite=f"{latest_composites.get(COL_COMPOSITE_MEDIUM, 0):.2f}",
        medium_confidence=f"{latest_regimes.get(COL_CONFIDENCE_MEDIUM, 0) * 100:.0f}",
        # Slow
        slow_regime=latest_regimes.get(COL_REGIME_SLOW, "N/A"),
        slow_regime_class=regime_class(latest_regimes.get(COL_REGIME_SLOW, "")),
        slow_composite=f"{latest_composites.get(COL_COMPOSITE_SLOW, 0):.2f}",
        slow_confidence=f"{latest_regimes.get(COL_CONFIDENCE_SLOW, 0) * 100:.0f}",
        # Checklist
        checklist_score=f"{latest_checklist.get(COL_CHECKLIST_SCORE, 50):.0f}",
        checklist_label=latest_checklist.get(COL_CHECKLIST_LABEL, "N/A"),
        checklist_class=regime_class(
            Regime.RISK_ON.value
            if latest_checklist.get(COL_CHECKLIST_SCORE, 0) >= 75
            else (
                Regime.NEUTRAL.value
                if latest_checklist.get(COL_CHECKLIST_SCORE, 0) >= 50
                else Regime.RISK_OFF.value
            )
        ),
        checklist_items=checklist_items,
        # Charts
        composite_chart_json=composite_chart.to_json(),
        block_chart_json=block_chart.to_json(),
        heatmap_chart_json=heatmap_chart.to_json(),
        checklist_chart_json=checklist_chart.to_json(),
    )

    # Save report
    report_path = output_dir / f"report_{date.strftime('%Y%m%d')}.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Also save as latest
    latest_path = output_dir / "report_latest.html"
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Generated HTML report: {report_path}")

    return report_path
