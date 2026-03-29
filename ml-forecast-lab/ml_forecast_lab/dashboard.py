"""
Auto-generate ApexCharts dashboard YAML for ML Forecast Lab.

Creates a HA dashboard configuration file with forecast visualisation
cards that users can import into their Lovelace dashboards.
"""

import logging
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)


def _forecast_chart(exp_name: str, prefix: str, horizons: List[int], units: str) -> dict:
    """
    ApexCharts card: forecast curve with prediction intervals.

    Shows the point forecast, upper/lower 95% bounds, and actual values
    on a single time-series chart.
    """
    base = f"sensor.{prefix}{exp_name}"

    series = [
        {
            "entity": f"{base}_point",
            "name": "Forecast",
            "stroke_width": 2,
            "curve": "smooth",
            "color": "#2196F3",
            "data_generator": (
                "return entity.attributes.forecast?.timestamps?.map((t, i) => "
                "[new Date(t).getTime(), entity.attributes.forecast.values[i]]) || [];"
            ),
        },
        {
            "entity": f"{base}_upper_95",
            "name": "Upper 95%",
            "stroke_width": 1,
            "curve": "smooth",
            "color": "#90CAF9",
            "opacity": 0.3,
            "data_generator": (
                "return entity.attributes.forecast?.timestamps?.map((t, i) => "
                "[new Date(t).getTime(), entity.attributes.forecast.values[i]]) || [];"
            ),
        },
        {
            "entity": f"{base}_lower_95",
            "name": "Lower 95%",
            "stroke_width": 1,
            "curve": "smooth",
            "color": "#90CAF9",
            "opacity": 0.3,
            "data_generator": (
                "return entity.attributes.forecast?.timestamps?.map((t, i) => "
                "[new Date(t).getTime(), entity.attributes.forecast.values[i]]) || [];"
            ),
        },
    ]

    return {
        "type": "custom:apexcharts-card",
        "header": {
            "show": True,
            "title": f"{exp_name} — Forecast",
            "show_states": True,
            "colorize_states": True,
        },
        "graph_span": "48h",
        "span": {"start": "hour"},
        "yaxis": [{"id": "main", "decimals": 1}],
        "series": series,
    }


def _cumulative_chart(exp_name: str, prefix: str, units: str) -> dict:
    """
    ApexCharts card: daily cumulative forecast vs actual.
    """
    base = f"sensor.{prefix}{exp_name}"

    return {
        "type": "custom:apexcharts-card",
        "header": {
            "show": True,
            "title": f"{exp_name} — Daily Cumulative",
            "show_states": True,
            "colorize_states": True,
        },
        "graph_span": "24h",
        "span": {"start": "day"},
        "yaxis": [{"id": "main", "decimals": 1}],
        "series": [
            {
                "entity": f"{base}_daily_cumulative",
                "name": f"Cumulative ({units})",
                "stroke_width": 2,
                "curve": "stepline",
                "color": "#4CAF50",
                "data_generator": (
                    "return entity.attributes.cumulative?.timestamps?.map((t, i) => "
                    "[new Date(t).getTime(), entity.attributes.cumulative.values[i]]) || [];"
                ),
            },
        ],
    }


def _horizon_gauges(exp_name: str, prefix: str, horizons: List[int], units: str) -> dict:
    """
    Horizontal stack of entity cards showing scalar forecast at each horizon.
    """
    base = f"sensor.{prefix}{exp_name}"
    cards = []
    for h in horizons:
        if h < 60:
            label = f"+{h}m"
        else:
            label = f"+{h // 60}h"

        cards.append({
            "type": "entity",
            "entity": f"{base}_horizon_{h}m" if h < 60 else f"{base}_horizon_{h // 60}h",
            "name": label,
            "unit": units,
        })

    return {
        "type": "horizontal-stack",
        "cards": cards,
    }


def _benchmark_table(exp_name: str) -> dict:
    """
    Markdown card showing latest benchmark results from the web UI.

    Points users to the web UI for detailed results since benchmark
    data is stored in-memory, not as HA entities.
    """
    return {
        "type": "markdown",
        "title": f"{exp_name} — Benchmark",
        "content": (
            f"View detailed benchmark results and model comparison at:\n\n"
            f"**[ML Forecast Lab Web UI](http://homeassistant.local:5052/experiment/{exp_name})**\n\n"
            f"Includes cross-validation metrics, model rankings, and fold-level results."
        ),
    }


def _status_card() -> dict:
    """
    Entity card for the MLFL heartbeat sensor.
    """
    return {
        "type": "entities",
        "title": "ML Forecast Lab",
        "entities": [
            {
                "entity": "sensor.mlfl_last_run",
                "name": "Last Run",
                "icon": "mdi:clock-check",
            },
        ],
    }


def _prediction_curve_chart(exp_name: str, prefix: str, units: str) -> dict:
    """
    ApexCharts card: historical actual + forecast prediction curve.

    Shows the stitched curve of recent actuals flowing into future forecast.
    """
    base = f"sensor.{prefix}{exp_name}"

    return {
        "type": "custom:apexcharts-card",
        "header": {
            "show": True,
            "title": f"{exp_name} — Prediction Curve",
            "show_states": True,
            "colorize_states": True,
        },
        "graph_span": "48h",
        "span": {"start": "day", "offset": "-12h"},
        "yaxis": [{"id": "main", "decimals": 1}],
        "series": [
            {
                "entity": f"{base}_curve",
                "name": f"Actual + Forecast ({units})",
                "stroke_width": 2,
                "curve": "smooth",
                "color": "#FF9800",
                "data_generator": (
                    "return entity.attributes.curve?.timestamps?.map((t, i) => "
                    "[new Date(t).getTime(), entity.attributes.curve.values[i]]) || [];"
                ),
            },
        ],
    }


def generate_dashboard(experiments: list, output_path: Path) -> None:
    """
    Generate a complete HA dashboard YAML with ApexCharts cards.

    Creates one view per experiment plus a summary view.
    The output file can be imported into HA as a new dashboard.

    Parameters
    ----------
    experiments : list
        List of ExperimentCfg dataclass instances.
    output_path : Path
        Path to write the YAML file.
    """
    views = []

    # Summary view
    summary_cards = [_status_card()]
    for exp in experiments:
        summary_cards.append({
            "type": "markdown",
            "content": (
                f"### {exp.name}\n"
                f"**Target:** `{exp.target_entity}`\n"
                f"**Mode:** {exp.mode} | **Models:** {', '.join(exp.models_enabled)}\n"
                f"**Horizons:** {', '.join(str(h) + 'm' for h in exp.horizons_minutes)}"
            ),
        })

    views.append({
        "title": "ML Forecast Lab",
        "path": "mlfl-summary",
        "icon": "mdi:chart-timeline-variant-shimmer",
        "cards": summary_cards,
    })

    # Per-experiment views
    for exp in experiments:
        prefix = exp.publish_prefix if hasattr(exp, "publish_prefix") else "mlfl_"
        units = exp.units if hasattr(exp, "units") else ""
        horizons = exp.horizons_minutes if hasattr(exp, "horizons_minutes") else [120, 480]

        cards = [
            _forecast_chart(exp.name, prefix, horizons, units),
            _prediction_curve_chart(exp.name, prefix, units),
        ]

        if hasattr(exp, "publish_daily_cumulative") and exp.publish_daily_cumulative:
            cards.append(_cumulative_chart(exp.name, prefix, units))

        cards.append(_horizon_gauges(exp.name, prefix, horizons, units))
        cards.append(_benchmark_table(exp.name))

        views.append({
            "title": exp.name,
            "path": f"mlfl-{exp.name}",
            "icon": "mdi:chart-line",
            "cards": cards,
        })

    dashboard = {
        "title": "ML Forecast Lab",
        "views": views,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(dashboard, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"Dashboard YAML written to {output_path}")
