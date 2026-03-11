"""dashboard.components — Shared chart, sidebar, and metrics components."""
from dashboard.components.charts import (
    candlestick_chart,
    forecast_chart,
    efficient_frontier_chart,
    allocation_pie,
    drawdown_chart,
    monthly_returns_heatmap,
    sector_heatmap,
)
from dashboard.components.sidebar import render_sidebar
from dashboard.components.metrics_cards import metric_card, metric_row, index_banner

__all__ = [
    "candlestick_chart", "forecast_chart", "efficient_frontier_chart",
    "allocation_pie", "drawdown_chart", "monthly_returns_heatmap", "sector_heatmap",
    "render_sidebar",
    "metric_card", "metric_row", "index_banner",
]
