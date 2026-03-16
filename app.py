from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.express as px
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

BASE_DIR = Path(__file__).resolve().parent
ALL_BUS_CSV = BASE_DIR / "MTA_Bus_Speeds__Beginning_2015_20260309.csv"
CBD_BUS_CSV = BASE_DIR / "MTA_Central_Business_District_Bus_Speeds__Beginning_2023_20260309.csv"

DEFAULT_CUTOFF_DATE = date(2025, 1, 5)
DEFAULT_WINDOW_MONTHS = 3
LOCAL_TRIP_TYPE = "LCL/LTD"
EXPRESS_ROUTE_TYPE = "express"
LOCAL_ROUTE_TYPE = "local"

ALL_DAY_MAP = {"1": "Weekday", "2": "Weekend"}
DAY_CHOICES = {"All": "All days", "Weekday": "Weekday", "Weekend": "Weekend"}
ALL_PERIOD_CHOICES = {"All": "All all-bus periods", "Peak": "Peak", "Off-Peak": "Off-Peak"}
CBD_PERIOD_CHOICES = {"All": "All CBD periods", "Peak": "Peak", "Overnight": "Overnight"}
BOROUGH_CHOICES = {
    "All": "All boroughs",
    "Bronx": "Bronx",
    "Brooklyn": "Brooklyn",
    "Manhattan": "Manhattan",
    "Queens": "Queens",
    "Staten Island": "Staten Island",
}


def empty_speed_df(extra_cols: list[str] | None = None) -> pd.DataFrame:
    base = {
        "month": pd.Series(dtype="datetime64[ns]"),
        "avg_speed": pd.Series(dtype="float64"),
    }
    if extra_cols:
        for col in extra_cols:
            base[col] = pd.Series(dtype="object")
    return pd.DataFrame(base)


def normalize_col_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_col_name(col) for col in out.columns]
    return out


def to_numeric_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def normalize_route_query(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def apply_route_filter(df: pd.DataFrame, route_query: str) -> pd.DataFrame:
    if not route_query:
        return df
    return df[df["route_id_norm"].str.contains(route_query, regex=False, na=False)].copy()


def weighted_monthly_speed(
    df: pd.DataFrame,
    mileage_col: str,
    time_col: str,
    by_cols: list[str] | None = None,
) -> pd.DataFrame:
    by_cols = by_cols or []
    if df.empty:
        return empty_speed_df(extra_cols=by_cols)

    out = df.copy()
    out["month"] = pd.to_datetime(out["month"], errors="coerce")
    out[mileage_col] = to_numeric_clean(out[mileage_col])
    out[time_col] = to_numeric_clean(out[time_col])

    group_cols = ["month"] + by_cols
    grouped = (
        out.dropna(subset=group_cols + [mileage_col, time_col])
        .groupby(group_cols, as_index=False)[[mileage_col, time_col]]
        .sum()
    )
    grouped = grouped[grouped[time_col] > 0]
    if grouped.empty:
        return empty_speed_df(extra_cols=by_cols)

    grouped["avg_speed"] = grouped[mileage_col] / grouped[time_col]
    grouped["month"] = grouped["month"].dt.to_period("M").dt.to_timestamp()
    return grouped.sort_values("month").reset_index(drop=True)


def build_empty_figure(title: str, message: str):
    fig = px.line(template="plotly_white")
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="MPH",
        margin=dict(l=20, r=20, t=70, b=20),
        height=360,
    )
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    return fig


def before_after_window_summary(
    monthly_df: pd.DataFrame,
    cutoff_month_start: pd.Timestamp,
    window_months: int,
) -> dict | None:
    if monthly_df.empty:
        return None

    df = monthly_df.sort_values("month")
    before = df[df["month"] < cutoff_month_start].tail(window_months)
    after = df[df["month"] >= cutoff_month_start].head(window_months)
    if before.empty or after.empty:
        return None

    before_avg = float(before["avg_speed"].mean())
    after_avg = float(after["avg_speed"].mean())
    delta = after_avg - before_avg
    pct = (delta / before_avg * 100.0) if before_avg else 0.0

    summary = pd.DataFrame(
        {
            "period": [f"Before ({window_months} mo)", f"After ({window_months} mo)"],
            "avg_speed": [before_avg, after_avg],
        }
    )

    return {
        "summary": summary,
        "before_avg": before_avg,
        "after_avg": after_avg,
        "delta": delta,
        "pct": pct,
        "before_range": f"{before['month'].min():%b %Y} to {before['month'].max():%b %Y}",
        "after_range": f"{after['month'].min():%b %Y} to {after['month'].max():%b %Y}",
    }


def format_signed(value: float, decimals: int = 2) -> str:
    return f"{value:+.{decimals}f}"


def direction_phrase(delta: float, threshold: float = 0.02) -> str:
    if delta > threshold:
        return "increased"
    if delta < -threshold:
        return "decreased"
    return "was nearly flat"


def comparative_phrase(primary: float, reference: float) -> str:
    diff = primary - reference
    if diff > 0.02:
        return f"improved more than the reference by {diff:.2f} MPH"
    if diff < -0.02:
        return f"improved less than the reference by {abs(diff):.2f} MPH"
    return "moved almost the same as the reference"


def minutes_for_trip(speed_mph: float | None, miles: float = 5.0) -> float | None:
    if speed_mph is None or speed_mph <= 0:
        return None
    return miles / speed_mph * 60.0


def build_before_after_bar(
    monthly_df: pd.DataFrame,
    title: str,
    cutoff_month_start: pd.Timestamp,
    window_months: int,
    accent_color: str,
):
    stats = before_after_window_summary(
        monthly_df=monthly_df,
        cutoff_month_start=cutoff_month_start,
        window_months=window_months,
    )
    if stats is None:
        return build_empty_figure(title, "Not enough data to build before/after comparison.")

    before_label = f"Before ({window_months} mo)"
    after_label = f"After ({window_months} mo)"
    summary = stats["summary"]

    fig = px.bar(
        summary,
        x="period",
        y="avg_speed",
        color="period",
        text="avg_speed",
        template="plotly_white",
        color_discrete_map={before_label: "#a0aab4", after_label: accent_color},
    )
    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate="%{x}<br>%{y:.2f} MPH<extra></extra>",
    )
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Average MPH",
        showlegend=False,
        margin=dict(l=20, r=20, t=95, b=20),
        height=360,
    )

    min_speed = float(summary["avg_speed"].min())
    max_speed = float(summary["avg_speed"].max())
    padding = max(0.15, (max_speed - min_speed) * 1.2)
    fig.update_yaxes(range=[max(0, min_speed - padding), max_speed + padding])

    fig.add_annotation(
        text=(
            f"Change: {format_signed(stats['delta'])} MPH ({format_signed(stats['pct'], 1)}%)"
            f"<br>{stats['before_range']} vs {stats['after_range']}"
        ),
        x=0.5,
        y=1.18,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    return fig


def build_line_compare(
    df: pd.DataFrame,
    category_col: str,
    title: str,
    cutoff_month_start: pd.Timestamp,
    color_map: dict[str, str],
    category_orders: dict[str, list[str]] | None = None,
):
    if df.empty:
        return build_empty_figure(title, "No data available for the selected filters.")

    plot_df = df.sort_values("month").copy()
    plot_df["month_iso"] = plot_df["month"].dt.strftime("%Y-%m-%d")

    fig = px.line(
        plot_df,
        x="month_iso",
        y="avg_speed",
        color=category_col,
        markers=True,
        template="plotly_white",
        color_discrete_map=color_map,
        category_orders=category_orders,
    )
    fig.update_traces(hovertemplate="Month: %{x|%b %Y}<br>Speed: %{y:.2f} MPH<extra></extra>")
    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="MPH",
        margin=dict(l=20, r=20, t=70, b=20),
        height=360,
        legend_title_text="",
    )
    fig.update_xaxes(type="date", tickformat="%b %Y", dtick="M3")

    cutoff_str = cutoff_month_start.strftime("%Y-%m-%d")
    fig.add_vline(x=cutoff_str, line_dash="dash", line_color="#6c757d")
    fig.add_annotation(
        x=cutoff_str,
        y=1,
        xref="x",
        yref="paper",
        text="CRZ start",
        showarrow=False,
        yshift=8,
    )
    return fig


def make_kpi_card(title: str, stats: dict | None, accent_class: str):
    if stats is None:
        body = ui.div(
            ui.div("No data for selected filters", class_="kpi-empty"),
            class_="kpi-body",
        )
    else:
        body = ui.div(
            ui.div(
                ui.div(f"{format_signed(stats['delta'])} MPH", class_="kpi-delta"),
                ui.div(f"{format_signed(stats['pct'], 1)}%", class_="kpi-pct"),
                class_="kpi-topline",
            ),
            ui.div(
                ui.div(
                    ui.div("Before MPH", class_="kpi-metric-label"),
                    ui.div(f"{stats['before_avg']:.2f}", class_="kpi-metric-value"),
                    class_="kpi-metric",
                ),
                ui.div(
                    ui.div("After MPH", class_="kpi-metric-label"),
                    ui.div(f"{stats['after_avg']:.2f}", class_="kpi-metric-value"),
                    class_="kpi-metric",
                ),
                ui.div(
                    ui.div("Abs change", class_="kpi-metric-label"),
                    ui.div(format_signed(stats["delta"]), class_="kpi-metric-value"),
                    class_="kpi-metric",
                ),
                ui.div(
                    ui.div("Percent change", class_="kpi-metric-label"),
                    ui.div(f"{format_signed(stats['pct'], 1)}%", class_="kpi-metric-value"),
                    class_="kpi-metric",
                ),
                class_="kpi-grid",
            ),
            ui.div(
                f"{stats['before_range']} vs {stats['after_range']}",
                class_="kpi-range",
            ),
            class_="kpi-body",
        )

    return ui.column(
        3,
        ui.card(
            ui.div(title, class_="kpi-title"),
            body,
            class_=f"kpi-card {accent_class}",
        ),
    )


def make_summary_list(items: list[str]):
    return ui.tags.ul(*[ui.tags.li(item) for item in items], class_="summary-list")


@lru_cache(maxsize=1)
def load_all_raw() -> pd.DataFrame:
    if not ALL_BUS_CSV.exists():
        raise FileNotFoundError(f"Missing file: {ALL_BUS_CSV.name}")

    raw = pd.read_csv(ALL_BUS_CSV, dtype=str)
    df = normalize_columns(raw)
    required = {
        "month",
        "borough",
        "day_type",
        "trip_type",
        "route_id",
        "period",
        "total_operating_time",
        "total_mileage",
    }
    if not required.issubset(df.columns):
        raise ValueError("All-bus CSV is missing one or more required columns.")

    out = df.copy()
    out["month"] = pd.to_datetime(out["month"], errors="coerce")
    out["borough"] = out["borough"].astype(str).str.strip()
    out["route_id_norm"] = out["route_id"].astype(str).str.strip().str.upper()
    out["day_group"] = out["day_type"].astype(str).str.strip().map(ALL_DAY_MAP).fillna("Unknown")
    out["period_group"] = out["period"].astype(str).str.strip()
    return out


@lru_cache(maxsize=1)
def load_cbd_raw() -> pd.DataFrame:
    if not CBD_BUS_CSV.exists():
        raise FileNotFoundError(f"Missing file: {CBD_BUS_CSV.name}")

    raw = pd.read_csv(CBD_BUS_CSV, dtype=str)
    df = normalize_columns(raw)
    required = {"month", "day_type", "time_period", "route_type", "route_id", "cbd_relation", "sum_mileage", "sum_time"}
    if not required.issubset(df.columns):
        raise ValueError("CBD CSV is missing one or more required columns.")

    out = df.copy()
    out["month"] = pd.to_datetime(out["month"], errors="coerce")
    out["day_group"] = out["day_type"].astype(str).str.strip()
    out["time_group"] = out["time_period"].astype(str).str.strip()
    out["route_type_norm"] = out["route_type"].astype(str).str.strip().str.lower()
    out["route_id_norm"] = out["route_id"].astype(str).str.strip().str.upper()
    out["cbd_relation_norm"] = out["cbd_relation"].astype(str).str.strip().str.upper()
    return out


def filter_all_raw(
    df: pd.DataFrame,
    day_filter: str,
    period_filter: str,
    borough_filter: str,
    route_query: str,
) -> pd.DataFrame:
    out = df.copy()
    if day_filter != "All":
        out = out[out["day_group"] == day_filter]
    if period_filter != "All":
        out = out[out["period_group"] == period_filter]
    if borough_filter != "All":
        out = out[out["borough"] == borough_filter]
    out = apply_route_filter(out, route_query)
    return out


def filter_cbd_raw(
    df: pd.DataFrame,
    day_filter: str,
    cbd_period_filter: str,
    route_query: str,
) -> pd.DataFrame:
    out = df.copy()
    if day_filter != "All":
        out = out[out["day_group"] == day_filter]
    if cbd_period_filter != "All":
        out = out[out["time_group"] == cbd_period_filter]
    out = apply_route_filter(out, route_query)
    return out


def monthly_series(df: pd.DataFrame, mileage_col: str, time_col: str) -> pd.DataFrame:
    monthly = weighted_monthly_speed(df=df, mileage_col=mileage_col, time_col=time_col)
    if monthly.empty:
        return empty_speed_df()
    return monthly[["month", "avg_speed"]]


def monthly_compare_series(
    df: pd.DataFrame,
    mileage_col: str,
    time_col: str,
    group_col: str,
    label_map: dict[str, str],
    label_name: str,
) -> pd.DataFrame:
    monthly = weighted_monthly_speed(
        df=df,
        mileage_col=mileage_col,
        time_col=time_col,
        by_cols=[group_col],
    )
    if monthly.empty:
        return empty_speed_df(extra_cols=[label_name])
    monthly[label_name] = monthly[group_col].map(label_map)
    monthly = monthly.dropna(subset=[label_name])
    return monthly[["month", "avg_speed", label_name]]


def filter_context_sentence(
    day_filter: str,
    all_period_filter: str,
    cbd_period_filter: str,
    borough_filter: str,
    route_query: str,
) -> str:
    parts = [
        f"Day type: {day_filter}",
        f"All-bus period: {all_period_filter}",
        f"CBD period: {cbd_period_filter}",
        f"Borough: {borough_filter}",
        f"Route: {route_query if route_query else 'All routes'}",
    ]
    return " | ".join(parts)


app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style(
            """
            :root {
              --ink: #14213d;
              --muted: #5c677d;
              --panel: rgba(255, 255, 255, 0.92);
              --panel-border: rgba(20, 33, 61, 0.08);
              --accent-a: #2a9d8f;
              --accent-b: #f4a261;
              --accent-c: #e76f51;
              --accent-d: #457b9d;
              --bg-top: #f5f3ee;
              --bg-bottom: #e8eef5;
            }
            body {
              background:
                radial-gradient(circle at top left, rgba(42, 157, 143, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(244, 162, 97, 0.12), transparent 24%),
                linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
              color: var(--ink);
              font-family: "Aptos", "Trebuchet MS", "Segoe UI", sans-serif;
            }
            .dashboard-shell {
              max-width: 1480px;
              margin: 0 auto;
              padding: 24px 10px 36px;
            }
            .hero-panel {
              background: linear-gradient(135deg, rgba(20, 33, 61, 0.98), rgba(36, 74, 116, 0.92));
              color: #f8fafc;
              border-radius: 24px;
              padding: 28px 30px 24px;
              box-shadow: 0 20px 42px rgba(20, 33, 61, 0.18);
              margin-bottom: 18px;
            }
            .hero-kicker {
              font-size: 0.82rem;
              letter-spacing: 0.18em;
              text-transform: uppercase;
              opacity: 0.82;
              margin-bottom: 10px;
            }
            .hero-title {
              font-family: "Aptos Display", "Trebuchet MS", sans-serif;
              font-size: 2.15rem;
              font-weight: 700;
              line-height: 1.08;
              margin-bottom: 8px;
            }
            .hero-subtitle {
              font-size: 1rem;
              line-height: 1.55;
              max-width: 980px;
              opacity: 0.92;
              margin: 0;
            }
            .filter-card,
            .chart-card,
            .summary-card,
            .kpi-card {
              background: var(--panel);
              border: 1px solid var(--panel-border);
              border-radius: 22px;
              box-shadow: 0 14px 30px rgba(20, 33, 61, 0.08);
            }
            .filter-note {
              margin-top: 12px;
              color: var(--muted);
              font-size: 0.92rem;
              line-height: 1.45;
            }
            .context-note {
              background: rgba(255, 255, 255, 0.74);
              border: 1px solid rgba(20, 33, 61, 0.08);
              border-radius: 16px;
              padding: 12px 16px;
              margin: 12px 0 18px;
              color: var(--muted);
              font-size: 0.95rem;
              line-height: 1.45;
            }
            .kpi-row {
              margin-top: 4px;
            }
            .kpi-card {
              min-height: 244px;
              position: relative;
              overflow: hidden;
            }
            .kpi-card::before {
              content: "";
              position: absolute;
              left: 0;
              top: 0;
              width: 100%;
              height: 6px;
              background: var(--accent-a);
            }
            .kpi-b::before { background: var(--accent-b); }
            .kpi-c::before { background: var(--accent-c); }
            .kpi-d::before { background: var(--accent-d); }
            .kpi-title {
              font-size: 0.95rem;
              font-weight: 700;
              letter-spacing: 0.01em;
              color: var(--ink);
              margin-bottom: 14px;
            }
            .kpi-body {
              display: flex;
              flex-direction: column;
              gap: 12px;
            }
            .kpi-topline {
              display: flex;
              justify-content: space-between;
              align-items: baseline;
              gap: 10px;
            }
            .kpi-delta {
              font-size: 1.9rem;
              font-weight: 700;
              line-height: 1;
            }
            .kpi-pct {
              font-size: 1rem;
              font-weight: 700;
              color: var(--muted);
            }
            .kpi-grid {
              display: grid;
              grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 10px;
            }
            .kpi-metric {
              background: rgba(20, 33, 61, 0.04);
              border-radius: 14px;
              padding: 10px 12px;
            }
            .kpi-metric-label {
              font-size: 0.76rem;
              color: var(--muted);
              text-transform: uppercase;
              letter-spacing: 0.06em;
              margin-bottom: 4px;
            }
            .kpi-metric-value {
              font-size: 1.02rem;
              font-weight: 700;
            }
            .kpi-range {
              font-size: 0.88rem;
              color: var(--muted);
            }
            .kpi-empty {
              font-size: 0.98rem;
              color: var(--muted);
              padding: 12px 0;
            }
            .summary-card .card-header,
            .chart-card .card-header,
            .filter-card .card-header {
              font-weight: 700;
              font-size: 1rem;
              background: transparent;
              border-bottom: 1px solid rgba(20, 33, 61, 0.08);
            }
            .summary-list {
              margin: 0;
              padding-left: 18px;
              line-height: 1.55;
            }
            .summary-list li + li {
              margin-top: 8px;
            }
            .so-what-note {
              margin-top: 12px;
              color: var(--muted);
              font-size: 0.9rem;
              line-height: 1.45;
            }
            @media (max-width: 991px) {
              .hero-title {
                font-size: 1.8rem;
              }
              .kpi-card {
                min-height: auto;
              }
            }
            """
        )
    ),
    ui.div(
        ui.div(
            ui.div("MANHATTAN CONGESTION RELIEF ZONE", class_="hero-kicker"),
            ui.div("Bus Speed Impact Dashboard", class_="hero-title"),
            ui.p(
                "Local CSV analysis with KPI cards, filterable comparisons, and auto-generated manager summary. "
                "The focus stays on before/after change around CRZ start, while the line charts show how the selected slice behaves over time.",
                class_="hero-subtitle",
            ),
            class_="hero-panel",
        ),
        ui.card(
            ui.card_header("Filters"),
            ui.row(
                ui.column(
                    2,
                    ui.input_date(
                        "cutoff_date",
                        "CRZ start date",
                        value=DEFAULT_CUTOFF_DATE,
                        format="yyyy-mm-dd",
                    ),
                ),
                ui.column(
                    2,
                    ui.input_select(
                        "window_months",
                        "Before/After window",
                        choices={"3": "3 months", "6": "6 months", "12": "12 months"},
                        selected=str(DEFAULT_WINDOW_MONTHS),
                    ),
                ),
                ui.column(
                    2,
                    ui.input_select("day_filter", "Day type", choices=DAY_CHOICES, selected="All"),
                ),
                ui.column(
                    2,
                    ui.input_select(
                        "all_period_filter",
                        "All-bus period",
                        choices=ALL_PERIOD_CHOICES,
                        selected="All",
                    ),
                ),
                ui.column(
                    2,
                    ui.input_select(
                        "cbd_period_filter",
                        "CBD period",
                        choices=CBD_PERIOD_CHOICES,
                        selected="All",
                    ),
                ),
                ui.column(
                    2,
                    ui.input_select(
                        "borough_filter",
                        "Borough (all-bus only)",
                        choices=BOROUGH_CHOICES,
                        selected="All",
                    ),
                ),
            ),
            ui.row(
                ui.column(
                    6,
                    ui.input_text(
                        "route_filter",
                        "Route contains",
                        placeholder="e.g. M15, Bx12, Q44",
                    ),
                ),
            ),
            ui.div(
                "Direction is not available in these source files. "
                "The all-bus file supports Peak/Off-Peak; the CBD file supports Peak/Overnight. "
                "Borough only exists in the all-bus file, so CBD charts ignore that filter.",
                class_="filter-note",
            ),
            class_="filter-card",
        ),
        ui.output_ui("context_note"),
        ui.output_ui("kpi_cards"),
        ui.row(
            ui.column(
                7,
                ui.card(
                    ui.card_header("Executive Summary"),
                    ui.output_ui("executive_summary"),
                    class_="summary-card",
                ),
            ),
            ui.column(
                5,
                ui.card(
                    ui.card_header("So What"),
                    ui.output_ui("so_what"),
                    class_="summary-card",
                ),
            ),
        ),
        ui.row(
            ui.column(
                6,
                ui.card(
                    ui.card_header("All Local Bus: Before vs After"),
                    output_widget("all_before_after_plot"),
                    class_="chart-card",
                ),
            ),
            ui.column(
                6,
                ui.card(
                    ui.card_header("CBD Local Bus: Before vs After"),
                    output_widget("cbd_before_after_plot"),
                    class_="chart-card",
                ),
            ),
        ),
        ui.row(
            ui.column(
                6,
                ui.card(
                    ui.card_header("CBD: Express vs Local Speed"),
                    output_widget("cbd_exp_local_plot"),
                    class_="chart-card",
                ),
            ),
            ui.column(
                6,
                ui.card(
                    ui.card_header("Local Bus: Within CBD vs Without CBD"),
                    output_widget("within_without_plot"),
                    class_="chart-card",
                ),
            ),
        ),
        class_="dashboard-shell",
    ),
)


def server(input, output, session):
    @reactive.calc
    def cutoff_month_start() -> pd.Timestamp:
        return pd.Timestamp(input.cutoff_date()).to_period("M").to_timestamp()

    @reactive.calc
    def window_months() -> int:
        try:
            value = int(input.window_months())
            return value if value > 0 else DEFAULT_WINDOW_MONTHS
        except Exception:
            return DEFAULT_WINDOW_MONTHS

    @reactive.calc
    def route_query() -> str:
        return normalize_route_query(input.route_filter())

    @reactive.calc
    def data_bundle():
        errors: list[str] = []

        try:
            all_raw = filter_all_raw(
                df=load_all_raw(),
                day_filter=input.day_filter(),
                period_filter=input.all_period_filter(),
                borough_filter=input.borough_filter(),
                route_query=route_query(),
            )
            all_local = monthly_series(
                df=all_raw[all_raw["trip_type"].astype(str).str.strip().str.upper() == LOCAL_TRIP_TYPE].copy(),
                mileage_col="total_mileage",
                time_col="total_operating_time",
            )
        except Exception as exc:
            all_local = empty_speed_df()
            errors.append(f"All local load failed: {exc}")

        try:
            cbd_filtered = filter_cbd_raw(
                df=load_cbd_raw(),
                day_filter=input.day_filter(),
                cbd_period_filter=input.cbd_period_filter(),
                route_query=route_query(),
            )
            cbd_local = monthly_series(
                df=cbd_filtered[
                    (cbd_filtered["route_type_norm"] == LOCAL_ROUTE_TYPE)
                    & (cbd_filtered["cbd_relation_norm"] == "CBD")
                ].copy(),
                mileage_col="sum_mileage",
                time_col="sum_time",
            )
            non_cbd_local = monthly_series(
                df=cbd_filtered[
                    (cbd_filtered["route_type_norm"] == LOCAL_ROUTE_TYPE)
                    & (cbd_filtered["cbd_relation_norm"] == "NON-CBD")
                ].copy(),
                mileage_col="sum_mileage",
                time_col="sum_time",
            )
            express_control = monthly_series(
                df=cbd_filtered[cbd_filtered["route_type_norm"] == EXPRESS_ROUTE_TYPE].copy(),
                mileage_col="sum_mileage",
                time_col="sum_time",
            )
            exp_local = monthly_compare_series(
                df=cbd_filtered[
                    (cbd_filtered["cbd_relation_norm"] == "CBD")
                    & (cbd_filtered["route_type_norm"].isin([EXPRESS_ROUTE_TYPE, LOCAL_ROUTE_TYPE]))
                ].copy(),
                mileage_col="sum_mileage",
                time_col="sum_time",
                group_col="route_type_norm",
                label_map={EXPRESS_ROUTE_TYPE: "Express", LOCAL_ROUTE_TYPE: "Local"},
                label_name="service_type",
            )
            within_without = monthly_compare_series(
                df=cbd_filtered[
                    (cbd_filtered["route_type_norm"] == LOCAL_ROUTE_TYPE)
                    & (cbd_filtered["cbd_relation_norm"].isin(["CBD", "NON-CBD"]))
                ].copy(),
                mileage_col="sum_mileage",
                time_col="sum_time",
                group_col="cbd_relation_norm",
                label_map={"CBD": "Within CBD", "NON-CBD": "Without CBD"},
                label_name="zone_type",
            )
        except Exception as exc:
            cbd_local = empty_speed_df()
            non_cbd_local = empty_speed_df()
            express_control = empty_speed_df()
            exp_local = empty_speed_df(extra_cols=["service_type"])
            within_without = empty_speed_df(extra_cols=["zone_type"])
            errors.append(f"CBD load failed: {exc}")

        return {
            "all_local": all_local,
            "cbd_local": cbd_local,
            "non_cbd_local": non_cbd_local,
            "express_control": express_control,
            "exp_local": exp_local,
            "within_without": within_without,
            "errors": errors,
        }

    @reactive.calc
    def kpi_stats():
        bundle = data_bundle()
        return {
            "all_local": before_after_window_summary(
                monthly_df=bundle["all_local"],
                cutoff_month_start=cutoff_month_start(),
                window_months=window_months(),
            ),
            "cbd_local": before_after_window_summary(
                monthly_df=bundle["cbd_local"],
                cutoff_month_start=cutoff_month_start(),
                window_months=window_months(),
            ),
            "non_cbd_local": before_after_window_summary(
                monthly_df=bundle["non_cbd_local"],
                cutoff_month_start=cutoff_month_start(),
                window_months=window_months(),
            ),
            "express_control": before_after_window_summary(
                monthly_df=bundle["express_control"],
                cutoff_month_start=cutoff_month_start(),
                window_months=window_months(),
            ),
        }

    @render.ui
    def context_note():
        bundle = data_bundle()
        if bundle["errors"]:
            return ui.div("Data warning: " + " | ".join(bundle["errors"]), class_="context-note")

        return ui.div(
            filter_context_sentence(
                day_filter=input.day_filter(),
                all_period_filter=input.all_period_filter(),
                cbd_period_filter=input.cbd_period_filter(),
                borough_filter=input.borough_filter(),
                route_query=route_query(),
            ),
            class_="context-note",
        )

    @render.ui
    def kpi_cards():
        stats = kpi_stats()
        return ui.div(
            ui.row(
                make_kpi_card("All Local Speed Change", stats["all_local"], "kpi-a"),
                make_kpi_card("CBD Local Speed Change", stats["cbd_local"], "kpi-b"),
                make_kpi_card("Non-CBD Local Speed Change", stats["non_cbd_local"], "kpi-c"),
                make_kpi_card("Express Control Change", stats["express_control"], "kpi-d"),
            ),
            class_="kpi-row",
        )

    @render.ui
    def executive_summary():
        stats = kpi_stats()
        if data_bundle()["errors"]:
            return ui.div("Summary unavailable because one or more datasets failed to load.")

        required = [stats["all_local"], stats["cbd_local"], stats["non_cbd_local"], stats["express_control"]]
        if any(item is None for item in required):
            return ui.div("Not enough data to generate the summary for the selected filter combination.")

        all_stats = stats["all_local"]
        cbd_stats = stats["cbd_local"]
        non_cbd_stats = stats["non_cbd_local"]
        express_stats = stats["express_control"]

        bullets = [
            (
                f"All local buses {direction_phrase(all_stats['delta'])} from "
                f"{all_stats['before_avg']:.2f} to {all_stats['after_avg']:.2f} MPH "
                f"in the selected {window_months()}-month window."
            ),
            (
                f"CBD local buses {direction_phrase(cbd_stats['delta'])} from "
                f"{cbd_stats['before_avg']:.2f} to {cbd_stats['after_avg']:.2f} MPH and "
                f"{comparative_phrase(cbd_stats['delta'], all_stats['delta'])}."
            ),
            (
                f"Compared with non-CBD local service, CBD local routes "
                f"{comparative_phrase(cbd_stats['delta'], non_cbd_stats['delta'])}; "
                f"this is the clearest indicator of whether CRZ effects are concentrated inside the zone."
            ),
        ]

        express_vs_local = abs(express_stats["delta"]) - abs(cbd_stats["delta"])
        if express_vs_local < -0.02:
            bullets.append(
                f"Express service changed less than CBD local service in this slice "
                f"({format_signed(express_stats['delta'])} vs {format_signed(cbd_stats['delta'])} MPH), "
                "which supports a more local-service-focused interpretation."
            )
        elif express_vs_local > 0.02:
            bullets.append(
                f"Express service also moved materially ({format_signed(express_stats['delta'])} MPH), "
                "so the selected slice is not a pure local-only story."
            )
        else:
            bullets.append(
                "Express and CBD local changes are very close in this slice, so the control signal is mixed."
            )

        bullets.append(
            "Raw speed levels outside CBD can still be higher than inside CBD. "
            "For policy interpretation, the before/after change matters more than the absolute level gap."
        )

        return make_summary_list(bullets)

    @render.ui
    def so_what():
        stats = kpi_stats()
        cbd_stats = stats["cbd_local"]
        all_stats = stats["all_local"]
        if cbd_stats is None or all_stats is None:
            return ui.div("Trip-time translation is unavailable for the selected filter combination.")

        cbd_before_min = minutes_for_trip(cbd_stats["before_avg"])
        cbd_after_min = minutes_for_trip(cbd_stats["after_avg"])
        all_before_min = minutes_for_trip(all_stats["before_avg"])
        all_after_min = minutes_for_trip(all_stats["after_avg"])
        if None in {cbd_before_min, cbd_after_min, all_before_min, all_after_min}:
            return ui.div("Trip-time translation is unavailable because one of the selected slices has invalid speed values.")

        cbd_saved = cbd_before_min - cbd_after_min
        all_saved = all_before_min - all_after_min
        hundred_trip_minutes = cbd_saved * 100.0

        bullets = [
            (
                f"A 5-mile CBD local trip moves from {cbd_before_min:.1f} to {cbd_after_min:.1f} minutes "
                f"in the selected window, a change of {cbd_saved:+.1f} minutes."
            ),
            (
                f"A 5-mile citywide local trip moves from {all_before_min:.1f} to {all_after_min:.1f} minutes, "
                f"a change of {all_saved:+.1f} minutes."
            ),
            (
                f"If the CBD local gain held across 100 similar bus trips, that would equal about "
                f"{hundred_trip_minutes:+.0f} bus-minutes saved."
            ),
            (
                "Ridership is not included in these source files, so this dashboard translates CRZ effects "
                "into trip-time rather than passenger-hours."
            ),
        ]

        return ui.div(
            make_summary_list(bullets),
            ui.div(
                "Use the KPI cards for headline results, and the line charts to see whether the selected slice stays directionally consistent over time.",
                class_="so-what-note",
            ),
        )

    @render_widget
    def all_before_after_plot():
        return build_before_after_bar(
            monthly_df=data_bundle()["all_local"],
            title="All Local Bus: Before vs After",
            cutoff_month_start=cutoff_month_start(),
            window_months=window_months(),
            accent_color="#2a9d8f",
        )

    @render_widget
    def cbd_before_after_plot():
        return build_before_after_bar(
            monthly_df=data_bundle()["cbd_local"],
            title="CBD Local Bus: Before vs After",
            cutoff_month_start=cutoff_month_start(),
            window_months=window_months(),
            accent_color="#f4a261",
        )

    @render_widget
    def cbd_exp_local_plot():
        return build_line_compare(
            df=data_bundle()["exp_local"],
            category_col="service_type",
            title="CBD: Express vs Local",
            cutoff_month_start=cutoff_month_start(),
            color_map={"Express": "#457b9d", "Local": "#e76f51"},
            category_orders={"service_type": ["Local", "Express"]},
        )

    @render_widget
    def within_without_plot():
        return build_line_compare(
            df=data_bundle()["within_without"],
            category_col="zone_type",
            title="Local Bus: Within CBD vs Without CBD",
            cutoff_month_start=cutoff_month_start(),
            color_map={"Within CBD": "#2a9d8f", "Without CBD": "#6d597a"},
            category_orders={"zone_type": ["Within CBD", "Without CBD"]},
        )


app = App(app_ui, server)
