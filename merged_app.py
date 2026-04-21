
from __future__ import annotations
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_DIR = BASE_DIR / "Dashboard Yixuan Wang&Sijin Li Dashboard"
ALL_BUS_CSV = ORIGINAL_DIR / "MTA_Bus_Speeds__Beginning_2015_20260309.csv"
CBD_BUS_CSV = ORIGINAL_DIR / "MTA_Central_Business_District_Bus_Speeds__Beginning_2023_20260309.csv"
TRAFFIC_CSV = ORIGINAL_DIR / "Daily_Traffic_on_MTA_Bridges_&_Tunnels_20260406.csv"
SUBWAY_CSV = BASE_DIR / "weekly_aggregated_mta.csv"
RESEARCH_DIR = BASE_DIR / "Dashboard Jiashuo Xu"
RESEARCH_METRICS_CSV = RESEARCH_DIR / "nyc_congestion_pricing_data.csv"
RESEARCH_REVENUE_CSV = RESEARCH_DIR / "toll_revenue_2025.csv"
RESEARCH_TRAFFIC_VOLUME_CSV = RESEARCH_DIR / "traffic_volume_2025.csv"

DEFAULT_CUTOFF_DATE = date(2025, 1, 5)
DEFAULT_WINDOW_MONTHS = 3
DEFAULT_TRAFFIC_START_DATE = date(2024, 1, 1)
DEFAULT_TRAFFIC_END_DATE = date(2025, 3, 31)
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
TRAFFIC_DIRECTION_CHOICES = {
    "I": "Inbound",
    "O": "Outbound",
    "All": "Both directions",
}
SUBWAY_PEAK_CHOICES = {
    "Morning Peak": "Morning Peak",
    "Evening Peak": "Evening Peak",
}
SUBWAY_BOROUGH_CHOICES = {
    "All": "All boroughs",
    "Bronx": "Bronx",
    "Brooklyn": "Brooklyn",
    "Manhattan": "Manhattan",
    "Queens": "Queens",
}
RESEARCH_COLORS = {
    "ink": "#14213d",
    "ink_light": "#5c677d",
    "paper": "#f5f3ee",
    "paper_dark": "#e8eef5",
    "accent": "#e76f51",
    "blue": "#457b9d",
    "green": "#2a9d8f",
    "gold": "#f4a261",
    "rule": "#a0aab4",
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


def empty_volume_df(extra_cols: list[str] | None = None) -> pd.DataFrame:
    base = {
        "month": pd.Series(dtype="datetime64[ns]"),
        "avg_daily_traffic": pd.Series(dtype="float64"),
        "total_traffic": pd.Series(dtype="float64"),
        "days_observed": pd.Series(dtype="int64"),
    }
    if extra_cols:
        for col in extra_cols:
            base[col] = pd.Series(dtype="object")
    return pd.DataFrame(base)


def build_empty_traffic_figure(title: str, message: str):
    fig = px.line(template="plotly_white")
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Vehicles per day",
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


def before_after_volume_summary(
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

    before_avg = float(before["avg_daily_traffic"].mean())
    after_avg = float(after["avg_daily_traffic"].mean())
    delta = after_avg - before_avg
    pct = (delta / before_avg * 100.0) if before_avg else 0.0

    summary = pd.DataFrame(
        {
            "period": [f"Before ({window_months} mo)", f"After ({window_months} mo)"],
            "avg_daily_traffic": [before_avg, after_avg],
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


def volume_change_phrase(delta: float, threshold_pct: float = 1.0, baseline: float | None = None) -> str:
    if baseline and baseline > 0:
        pct = abs(delta) / baseline * 100.0
        if pct < threshold_pct:
            return "was nearly flat"
    if delta < 0:
        return "decreased"
    if delta > 0:
        return "increased"
    return "was nearly flat"


def format_whole_number(value: float) -> str:
    return f"{value:,.0f}"


def make_volume_kpi_card(title: str, stats: dict | None, accent_class: str):
    if stats is None:
        body = ui.div(
            ui.div("No data for selected filters", class_="kpi-empty"),
            class_="kpi-body",
        )
    else:
        body = ui.div(
            ui.div(
                ui.div(f"{format_signed(stats['delta'], 0)}", class_="kpi-delta"),
                ui.div(f"{format_signed(stats['pct'], 1)}%", class_="kpi-pct"),
                class_="kpi-topline",
            ),
            ui.div(
                ui.div(
                    ui.div("Before vehicles/day", class_="kpi-metric-label"),
                    ui.div(format_whole_number(stats["before_avg"]), class_="kpi-metric-value"),
                    class_="kpi-metric",
                ),
                ui.div(
                    ui.div("After vehicles/day", class_="kpi-metric-label"),
                    ui.div(format_whole_number(stats["after_avg"]), class_="kpi-metric-value"),
                    class_="kpi-metric",
                ),
                ui.div(
                    ui.div("Abs change", class_="kpi-metric-label"),
                    ui.div(format_whole_number(stats["delta"]), class_="kpi-metric-value"),
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
        4,
        ui.card(
            ui.div(title, class_="kpi-title"),
            body,
            class_=f"kpi-card {accent_class}",
        ),
    )


def make_traffic_snapshot_card(
    title: str,
    value: float | None,
    detail: str,
    accent_class: str,
):
    if value is None:
        body = ui.div(
            ui.div("No data for selected filters", class_="kpi-empty"),
            class_="kpi-body",
        )
    else:
        body = ui.div(
            ui.div(format_whole_number(value), class_="kpi-delta"),
            ui.div(detail, class_="kpi-range"),
            class_="kpi-body",
        )

    return ui.column(
        4,
        ui.card(
            ui.div(title, class_="kpi-title"),
            body,
            class_=f"kpi-card {accent_class}",
        ),
    )


@lru_cache(maxsize=1)
def load_traffic_raw() -> pd.DataFrame:
    if not TRAFFIC_CSV.exists():
        raise FileNotFoundError(f"Missing file: {TRAFFIC_CSV.name}")

    raw = pd.read_csv(TRAFFIC_CSV, dtype=str)
    df = normalize_columns(raw)
    required = {"date", "plaza_id", "direction", "vehicles_e_zpass", "vehicles_vtoll"}
    if not required.issubset(df.columns):
        raise ValueError("Traffic CSV is missing one or more required columns.")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["plaza_id"] = out["plaza_id"].astype(str).str.strip()
    out["direction"] = out["direction"].astype(str).str.strip().str.upper()
    out["vehicles_e_zpass"] = to_numeric_clean(out["vehicles_e_zpass"]).fillna(0)
    out["vehicles_vtoll"] = to_numeric_clean(out["vehicles_vtoll"]).fillna(0)
    out["traffic_volume"] = out["vehicles_e_zpass"] + out["vehicles_vtoll"]
    out = out.dropna(subset=["date"]).copy()
    out["month"] = out["date"].dt.to_period("M").dt.to_timestamp()

    max_date = out["date"].max()
    latest_month_start = max_date.to_period("M").to_timestamp()
    latest_month_end = latest_month_start + pd.offsets.MonthEnd(1)
    if max_date.normalize() < latest_month_end.normalize():
        out = out[out["month"] < latest_month_start].copy()

    return out.sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=1)
def traffic_plaza_choices() -> dict[str, str]:
    choices = {"All": "All plazas"}
    try:
        ids = sorted(load_traffic_raw()["plaza_id"].dropna().unique().tolist(), key=lambda value: int(value))
        choices.update({plaza_id: f"Plaza {plaza_id}" for plaza_id in ids})
    except Exception:
        pass
    return choices


def filter_traffic_raw(
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
    direction_filter: str,
    plaza_filter: str,
) -> pd.DataFrame:
    out = df.copy()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    out = out[(out["date"] >= start_ts) & (out["date"] <= end_ts)]
    if direction_filter != "All":
        out = out[out["direction"] == direction_filter]
    if plaza_filter != "All":
        out = out[out["plaza_id"] == plaza_filter]
    return out


def monthly_traffic_series(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_volume_df()

    grouped = (
        df.groupby("month", as_index=False)
        .agg(
            total_traffic=("traffic_volume", "sum"),
            days_observed=("date", "nunique"),
        )
        .sort_values("month")
    )
    grouped = grouped[grouped["days_observed"] > 0].copy()
    if grouped.empty:
        return empty_volume_df()

    grouped["avg_daily_traffic"] = grouped["total_traffic"] / grouped["days_observed"]
    return grouped[["month", "avg_daily_traffic", "total_traffic", "days_observed"]].reset_index(drop=True)


def build_traffic_before_after_bar(
    monthly_df: pd.DataFrame,
    title: str,
    cutoff_month_start: pd.Timestamp,
    window_months: int,
    accent_color: str,
):
    stats = before_after_volume_summary(
        monthly_df=monthly_df,
        cutoff_month_start=cutoff_month_start,
        window_months=window_months,
    )
    if stats is None:
        return build_empty_traffic_figure(title, "Not enough data to build before/after comparison.")

    before_label = f"Before ({window_months} mo)"
    after_label = f"After ({window_months} mo)"
    summary = stats["summary"]

    fig = px.bar(
        summary,
        x="period",
        y="avg_daily_traffic",
        color="period",
        text="avg_daily_traffic",
        template="plotly_white",
        color_discrete_map={before_label: "#a0aab4", after_label: accent_color},
    )
    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        hovertemplate="%{x}<br>%{y:,.0f} vehicles/day<extra></extra>",
    )
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Average daily traffic",
        showlegend=False,
        margin=dict(l=20, r=20, t=95, b=20),
        height=360,
    )

    min_volume = float(summary["avg_daily_traffic"].min())
    max_volume = float(summary["avg_daily_traffic"].max())
    padding = max(250.0, (max_volume - min_volume) * 0.8)
    fig.update_yaxes(range=[max(0, min_volume - padding), max_volume + padding])
    fig.add_annotation(
        text=(
            f"Change: {format_whole_number(stats['delta'])} vehicles/day ({format_signed(stats['pct'], 1)}%)"
            f"<br>{stats['before_range']} vs {stats['after_range']}"
        ),
        x=0.5,
        y=1.18,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    return fig


def build_traffic_line(
    monthly_df: pd.DataFrame,
    title: str,
    cutoff_month_start: pd.Timestamp,
):
    if monthly_df.empty:
        return build_empty_traffic_figure(title, "No data available for the selected filters.")

    plot_df = monthly_df.sort_values("month").copy()
    plot_df["month_iso"] = plot_df["month"].dt.strftime("%Y-%m-%d")

    fig = px.line(
        plot_df,
        x="month_iso",
        y="avg_daily_traffic",
        markers=True,
        template="plotly_white",
    )
    fig.update_traces(
        line_color="#457b9d",
        marker_color="#2a9d8f",
        hovertemplate="Month: %{x|%b %Y}<br>Traffic: %{y:,.0f} vehicles/day<extra></extra>",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Average daily traffic",
        margin=dict(l=20, r=20, t=70, b=20),
        height=360,
        showlegend=False,
    )
    fig.update_xaxes(type="date", tickformat="%b %Y", dtick="M2")

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


def traffic_context_sentence(
    direction_filter: str,
    plaza_filter: str,
    start_date: date,
    end_date: date,
) -> str:
    plaza_text = "All plazas" if plaza_filter == "All" else f"Plaza {plaza_filter}"
    direction_text = TRAFFIC_DIRECTION_CHOICES.get(direction_filter, direction_filter)
    return (
        f"Direction: {direction_text} | Location: {plaza_text} | "
        f"Trend window: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d} | "
        "Monthly values are average daily traffic and the latest incomplete month is excluded."
    )


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


def empty_ridership_df(extra_cols: list[str] | None = None) -> pd.DataFrame:
    base = {
        "week_start": pd.Series(dtype="datetime64[ns]"),
        "avg_ridership": pd.Series(dtype="float64"),
        "total_ridership": pd.Series(dtype="float64"),
    }
    if extra_cols:
        for col in extra_cols:
            base[col] = pd.Series(dtype="object")
    return pd.DataFrame(base)


def build_empty_ridership_figure(title: str, message: str):
    fig = px.line(template="plotly_white")
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Average ridership",
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


@lru_cache(maxsize=1)
def load_subway_raw() -> pd.DataFrame:
    if not SUBWAY_CSV.exists():
        raise FileNotFoundError(f"Missing file: {SUBWAY_CSV.name}")

    raw = pd.read_csv(SUBWAY_CSV, dtype=str)
    df = normalize_columns(raw)
    required = {"week_start", "borough", "avg_ridership", "total_ridership", "peak_period", "period"}
    if not required.issubset(df.columns):
        raise ValueError("Subway CSV is missing one or more required columns.")

    out = df.copy()
    out["week_start"] = pd.to_datetime(out["week_start"], errors="coerce")
    out["borough"] = out["borough"].astype(str).str.strip()
    out["peak_period"] = out["peak_period"].astype(str).str.strip()
    out["period"] = out["period"].astype(str).str.strip()
    out["avg_ridership"] = to_numeric_clean(out["avg_ridership"])
    out["total_ridership"] = to_numeric_clean(out["total_ridership"])
    out = out.dropna(
        subset=["week_start", "borough", "peak_period", "period", "avg_ridership", "total_ridership"]
    ).copy()

    # Exclude the distorted 2020 baseline to keep the pre/post comparison consistent.
    out = out[out["week_start"] >= pd.Timestamp("2021-01-01")].copy()
    return out.sort_values(["week_start", "borough", "period"]).reset_index(drop=True)


def filter_subway_raw(df: pd.DataFrame, peak_choice: str, borough_choice: str) -> pd.DataFrame:
    out = df[df["peak_period"] == peak_choice].copy()
    if borough_choice != "All":
        out = out[out["borough"] == borough_choice].copy()
    return out


def subway_selection_stats(df: pd.DataFrame) -> dict | None:
    if df.empty:
        return None

    before = df[df["period"] == "Pre Policy"].copy()
    after = df[df["period"] == "Post Policy"].copy()
    if before.empty or after.empty:
        return None

    before_avg = float(before["avg_ridership"].mean())
    after_avg = float(after["avg_ridership"].mean())
    delta = after_avg - before_avg
    pct = (delta / before_avg * 100.0) if before_avg else 0.0

    return {
        "before_avg": before_avg,
        "after_avg": after_avg,
        "delta": delta,
        "pct": pct,
        "total_riders": float(df["total_ridership"].sum()),
        "weeks": int(df["week_start"].nunique()),
        "start_week": df["week_start"].min(),
        "end_week": df["week_start"].max(),
        "before_range": f"{before['week_start'].min():%b %Y} to {before['week_start'].max():%b %Y}",
        "after_range": f"{after['week_start'].min():%b %Y} to {after['week_start'].max():%b %Y}",
    }


def subway_change_phrase(pct: float, threshold_pct: float = 1.0) -> str:
    if pct > threshold_pct:
        return "increased"
    if pct < -threshold_pct:
        return "decreased"
    return "was nearly flat"


def subway_borough_change(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64")

    before = df[df["period"] == "Pre Policy"].groupby("borough")["avg_ridership"].mean()
    after = df[df["period"] == "Post Policy"].groupby("borough")["avg_ridership"].mean()
    return ((after - before) / before * 100.0).dropna().sort_values(ascending=False)


def subway_peak_change_summary(df: pd.DataFrame) -> dict[str, float]:
    changes: dict[str, float] = {}
    for peak_choice in SUBWAY_PEAK_CHOICES:
        stats = subway_selection_stats(df[df["peak_period"] == peak_choice].copy())
        if stats is not None:
            changes[peak_choice] = stats["pct"]
    return changes


def build_subway_trend_plot(df: pd.DataFrame, title: str):
    if df.empty:
        return build_empty_ridership_figure(title, "No subway ridership data is available for the selected filters.")

    weekly = (
        df.groupby(["week_start", "period"], as_index=False)["avg_ridership"]
        .mean()
        .sort_values(["week_start", "period"])
    )
    if weekly.empty:
        return build_empty_ridership_figure(title, "No weekly ridership trend can be built for the selected filters.")

    fig = px.line(
        weekly,
        x="week_start",
        y="avg_ridership",
        color="period",
        template="plotly_white",
        color_discrete_map={"Pre Policy": "#a0aab4", "Post Policy": "#2a9d8f"},
    )
    fig.update_traces(hovertemplate="Week of %{x|%b %d, %Y}<br>Average ridership: %{y:,.0f}<extra>%{fullData.name}</extra>")
    fig.update_layout(
        title=title,
        xaxis_title="Week",
        yaxis_title="Average ridership",
        margin=dict(l=20, r=20, t=70, b=20),
        height=360,
        legend_title_text="",
    )
    fig.update_xaxes(tickformat="%b %Y", dtick="M3")
    fig.add_vline(x=pd.Timestamp(DEFAULT_CUTOFF_DATE), line_dash="dash", line_color="#6c757d")
    fig.add_annotation(
        x=pd.Timestamp(DEFAULT_CUTOFF_DATE),
        y=1,
        xref="x",
        yref="paper",
        text="CRZ start",
        showarrow=False,
        yshift=8,
    )
    return fig


def build_subway_borough_bar(df: pd.DataFrame, title: str):
    if df.empty:
        return build_empty_ridership_figure(title, "No borough comparison is available for the selected peak period.")

    borough_avg = (
        df.groupby(["borough", "period"], as_index=False)["avg_ridership"]
        .mean()
        .sort_values(["borough", "period"])
    )
    if borough_avg.empty:
        return build_empty_ridership_figure(title, "No borough comparison is available for the selected peak period.")

    fig = px.bar(
        borough_avg,
        x="borough",
        y="avg_ridership",
        color="period",
        barmode="group",
        template="plotly_white",
        color_discrete_map={"Pre Policy": "#a0aab4", "Post Policy": "#2a9d8f"},
    )
    fig.update_traces(hovertemplate="Borough: %{x}<br>Average ridership: %{y:,.0f}<extra>%{fullData.name}</extra>")
    fig.update_layout(
        title=title,
        xaxis_title="Borough",
        yaxis_title="Average ridership",
        margin=dict(l=20, r=20, t=70, b=20),
        height=360,
        legend_title_text="",
    )
    return fig


def make_ridership_change_card(title: str, stats: dict | None, accent_class: str):
    if stats is None:
        body = ui.div(
            ui.div("No data for selected filters", class_="kpi-empty"),
            class_="kpi-body",
        )
    else:
        body = ui.div(
            ui.div(
                ui.div(format_signed(stats["delta"], 0), class_="kpi-delta"),
                ui.div(f"{format_signed(stats['pct'], 1)}%", class_="kpi-pct"),
                class_="kpi-topline",
            ),
            ui.div("Average riders per week", class_="kpi-range"),
            ui.div(f"{stats['before_range']} vs {stats['after_range']}", class_="kpi-range"),
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


def make_ridership_snapshot_card(title: str, value: float | None, detail: str, accent_class: str):
    if value is None:
        body = ui.div(
            ui.div("No data for selected filters", class_="kpi-empty"),
            class_="kpi-body",
        )
    else:
        body = ui.div(
            ui.div(format_whole_number(value), class_="kpi-delta"),
            ui.div(detail, class_="kpi-range"),
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


def subway_context_sentence(peak_choice: str, borough_choice: str) -> str:
    borough_text = "All boroughs" if borough_choice == "All" else borough_choice
    return (
        f"Peak period: {peak_choice} | Borough selection: {borough_text} | "
        "Source: weekly aggregated subway ridership from the merged course project CSV | "
        "2020 is excluded and the borough comparison chart keeps the selected peak period while showing all boroughs."
    )


@lru_cache(maxsize=1)
def load_research_metrics_raw() -> pd.DataFrame:
    if not RESEARCH_METRICS_CSV.exists():
        raise FileNotFoundError(f"Missing file: {RESEARCH_METRICS_CSV.name}")

    raw = pd.read_csv(RESEARCH_METRICS_CSV, dtype=str)
    df = normalize_columns(raw)
    required = {
        "category",
        "metric",
        "period",
        "value",
        "unit",
        "comparison",
        "comparison_value",
        "comparison_unit",
        "source",
        "notes",
    }
    if not required.issubset(df.columns):
        raise ValueError("Research metrics CSV is missing one or more required columns.")

    out = df.copy()
    for col in required:
        out[col] = out[col].fillna("").astype(str).str.strip()
    out["value_num"] = to_numeric_clean(out["value"])
    out["comparison_value_num"] = to_numeric_clean(out["comparison_value"])
    return out.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_research_traffic_volume_raw() -> pd.DataFrame:
    if not RESEARCH_TRAFFIC_VOLUME_CSV.exists():
        raise FileNotFoundError(f"Missing file: {RESEARCH_TRAFFIC_VOLUME_CSV.name}")

    raw = pd.read_csv(RESEARCH_TRAFFIC_VOLUME_CSV, dtype=str)
    df = normalize_columns(raw)
    required = {
        "month",
        "month_num",
        "baseline_2024_daily_thousands",
        "crz_2025_daily_thousands",
        "daily_reduction_thousands",
        "reduction_pct",
        "vehicles_kept_out_millions",
    }
    if not required.issubset(df.columns):
        raise ValueError("Research traffic CSV is missing one or more required columns.")

    out = df.copy()
    out["month_num"] = to_numeric_clean(out["month_num"])
    numeric_cols = [
        "baseline_2024_daily_thousands",
        "crz_2025_daily_thousands",
        "daily_reduction_thousands",
        "reduction_pct",
        "vehicles_kept_out_millions",
    ]
    for col in numeric_cols:
        out[col] = to_numeric_clean(out[col])
    out["month_short"] = out["month"].astype(str).str.slice(0, 3)
    return out.sort_values("month_num").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_research_revenue_raw() -> pd.DataFrame:
    if not RESEARCH_REVENUE_CSV.exists():
        raise FileNotFoundError(f"Missing file: {RESEARCH_REVENUE_CSV.name}")

    raw = pd.read_csv(RESEARCH_REVENUE_CSV, dtype=str)
    df = normalize_columns(raw)
    required = {
        "month",
        "month_num",
        "net_revenue_millions",
        "monthly_target_millions",
        "above_below_target_millions",
        "cumulative_revenue_millions",
        "cumulative_target_millions",
        "met_target",
        "notes",
    }
    if not required.issubset(df.columns):
        raise ValueError("Research revenue CSV is missing one or more required columns.")

    out = df.copy()
    out["month_num"] = to_numeric_clean(out["month_num"])
    numeric_cols = [
        "net_revenue_millions",
        "monthly_target_millions",
        "above_below_target_millions",
        "cumulative_revenue_millions",
        "cumulative_target_millions",
    ]
    for col in numeric_cols:
        out[col] = to_numeric_clean(out[col])
    out["month_short"] = out["month"].astype(str).str.slice(0, 3)
    return out.sort_values("month_num").reset_index(drop=True)


def research_metric_row(
    metrics_df: pd.DataFrame,
    category: str,
    metric: str,
    period: str | None = None,
) -> pd.Series | None:
    mask = (metrics_df["category"] == category) & (metrics_df["metric"] == metric)
    if period is not None:
        mask &= metrics_df["period"] == period
    matches = metrics_df[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def research_metric_text(row: pd.Series | None, fallback: str = "N/A") -> str:
    if row is None:
        return fallback
    value = str(row.get("value", "")).strip()
    return value or fallback


def research_metric_number(row: pd.Series | None) -> float | None:
    if row is None:
        return None
    value = row.get("value_num")
    if pd.isna(value):
        return None
    return float(value)


def research_metric_display(row: pd.Series | None, use_abs: bool = False) -> str:
    if row is None:
        return "N/A"

    value_num = research_metric_number(row)
    unit = str(row.get("unit", "")).strip()
    if value_num is not None:
        value_num = abs(value_num) if use_abs else value_num
        if unit == "%":
            return f"{value_num:.1f}%" if not float(value_num).is_integer() else f"{value_num:.0f}%"
        if unit == "million USD":
            return format_compact_millions(value_num)
        if unit:
            return f"{value_num:,.1f} {unit}" if not float(value_num).is_integer() else f"{value_num:,.0f} {unit}"
        return f"{value_num:,.1f}" if not float(value_num).is_integer() else f"{value_num:,.0f}"

    text = research_metric_text(row)
    if unit and unit not in text:
        spacer = "" if unit.startswith("/") or unit == "%" else " "
        return f"{text}{spacer}{unit}"
    return text


def format_compact_millions(value: float | None) -> str:
    if value is None:
        return "N/A"
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"${value / 1000:.0f}B"
    if float(value).is_integer():
        return f"${value:.0f}M"
    return f"${value:.1f}M"


def build_research_empty_figure(title: str, message: str):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title=title,
        margin=dict(l=20, r=20, t=70, b=20),
        height=320,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=RESEARCH_COLORS["ink_light"]),
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


def build_research_traffic_chart(monthly_df: pd.DataFrame):
    if monthly_df.empty:
        return build_research_empty_figure(
            "Average Daily Vehicle Entries to CRZ",
            "Traffic volume data is not available.",
        )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=monthly_df["month_short"],
            y=monthly_df["crz_2025_daily_thousands"],
            name="2025 Daily Entries (thousands)",
            marker_color="rgba(41, 128, 185, 0.82)",
            marker_line_color=RESEARCH_COLORS["blue"],
            marker_line_width=1,
            hovertemplate="%{x}<br>%{y:,.0f}k vehicles/day<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly_df["month_short"],
            y=monthly_df["baseline_2024_daily_thousands"],
            name="2024 Baseline (avg. daily vehicles, thousands)",
            mode="lines",
            line=dict(color=RESEARCH_COLORS["rule"], width=2),
            hovertemplate="%{x}<br>%{y:,.0f}k baseline<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Average Daily Vehicle Entries to CRZ",
        margin=dict(l=20, r=20, t=70, b=20),
        height=320,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=RESEARCH_COLORS["ink_light"]),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
    )
    fig.update_yaxes(range=[470, 600], title="", gridcolor=RESEARCH_COLORS["paper_dark"])
    fig.update_xaxes(title="", showgrid=False)
    return fig


def build_research_safety_chart(metrics_df: pd.DataFrame):
    rows = [
        ("CRZ Crashes", "Safety", "Crashes in CRZ", RESEARCH_COLORS["green"]),
        ("Traffic Injuries", "Safety", "Traffic injuries in CRZ", RESEARCH_COLORS["green"]),
        ("Citywide Traffic Deaths", "Safety", "Citywide traffic deaths", RESEARCH_COLORS["green"]),
        ("Noise Complaints", "Safety", "Noise complaints (NYC 311)", RESEARCH_COLORS["blue"]),
        ("Holland Tunnel Delays", "Speed", "Holland Tunnel rush hour delay reduction", RESEARCH_COLORS["accent"]),
    ]
    plot_rows: list[dict[str, object]] = []
    for label, category, metric, color in rows:
        row = research_metric_row(metrics_df, category, metric)
        value = research_metric_number(row)
        if value is None:
            continue
        plot_rows.append({"label": label, "value": value, "color": color})

    if not plot_rows:
        return build_research_empty_figure(
            "Safety & Quality-of-Life Metrics (% Change, 2025 vs. 2024)",
            "Safety metrics are not available.",
        )

    plot_df = pd.DataFrame(plot_rows)
    fig = go.Figure(
        go.Bar(
            x=plot_df["value"],
            y=plot_df["label"],
            orientation="h",
            marker_color=plot_df["color"],
            hovertemplate="%{y}<br>%{x:.0f}%<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Safety & Quality-of-Life Metrics (% Change, 2025 vs. 2024)",
        margin=dict(l=20, r=20, t=70, b=20),
        height=320,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=RESEARCH_COLORS["ink_light"]),
    )
    fig.update_xaxes(range=[-75, 0], ticksuffix="%", gridcolor=RESEARCH_COLORS["paper_dark"])
    fig.update_yaxes(title="", autorange="reversed", showgrid=False)
    return fig


def build_research_air_chart(metrics_df: pd.DataFrame):
    plot_rows = [
        (
            "NYC (Jan-Jun 2025)",
            abs(research_metric_number(research_metric_row(metrics_df, "Air Quality", "PM2.5 reduction in CRZ")) or 0),
            RESEARCH_COLORS["green"],
        ),
        (
            "Stockholm (2006-2010)",
            abs(research_metric_number(research_metric_row(metrics_df, "Air Quality", "PM2.5 reduction Stockholm benchmark")) or 0),
            RESEARCH_COLORS["blue"],
        ),
        (
            "London (2019-2022)",
            abs(research_metric_number(research_metric_row(metrics_df, "Air Quality", "PM2.5 reduction London benchmark")) or 0),
            RESEARCH_COLORS["gold"],
        ),
    ]
    plot_df = pd.DataFrame(plot_rows, columns=["label", "value", "color"])
    if plot_df["value"].eq(0).all():
        return build_research_empty_figure(
            "PM2.5 Reduction: NYC vs. Global Peers",
            "Air quality metrics are not available.",
        )

    fig = go.Figure(
        go.Bar(
            x=plot_df["label"],
            y=plot_df["value"],
            marker_color=plot_df["color"],
            hovertemplate="%{x}<br>%{y:.0f}% reduction<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="PM2.5 Reduction: NYC vs. Global Peers",
        margin=dict(l=20, r=20, t=70, b=20),
        height=320,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=RESEARCH_COLORS["ink_light"]),
        showlegend=False,
    )
    fig.update_yaxes(range=[0, 28], ticksuffix="%", gridcolor=RESEARCH_COLORS["paper_dark"])
    fig.update_xaxes(showgrid=False, title="")
    return fig


def build_research_revenue_chart(monthly_df: pd.DataFrame):
    if monthly_df.empty:
        return build_research_empty_figure(
            "Monthly Net Toll Revenue, 2025 (Millions USD)",
            "Revenue data is not available.",
        )

    plot_df = monthly_df.copy()
    plot_df.loc[plot_df["month_num"] == 12, "month_short"] = "Dec (est.)"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["month_short"],
            y=plot_df["net_revenue_millions"],
            name="Net Monthly Revenue ($M)",
            mode="lines+markers",
            line=dict(color=RESEARCH_COLORS["gold"], width=3),
            marker=dict(size=7, color=RESEARCH_COLORS["gold"]),
            fill="tozeroy",
            fillcolor="rgba(184, 134, 11, 0.15)",
            hovertemplate="%{x}<br>$%{y:.1f}M<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["month_short"],
            y=plot_df["monthly_target_millions"],
            name="Monthly Target ($41.6M)",
            mode="lines",
            line=dict(color=RESEARCH_COLORS["accent"], width=2, dash="dash"),
            hovertemplate="%{x}<br>$%{y:.1f}M target<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Monthly Net Toll Revenue, 2025 (Millions USD)",
        margin=dict(l=20, r=20, t=70, b=20),
        height=320,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=RESEARCH_COLORS["ink_light"]),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
    )
    fig.update_yaxes(range=[30, 70], tickprefix="$", ticksuffix="M", gridcolor=RESEARCH_COLORS["paper_dark"])
    fig.update_xaxes(showgrid=False, title="")
    return fig


def build_research_economy_chart(metrics_df: pd.DataFrame):
    private_emp_row = research_metric_row(metrics_df, "Economy", "Private sector employment NYC")
    plot_rows = [
        (
            "CRZ Visitor Traffic",
            research_metric_number(research_metric_row(metrics_df, "Economy", "CRZ visitor foot traffic")),
            RESEARCH_COLORS["blue"],
        ),
        (
            "Office Leasing (Q3)",
            research_metric_number(research_metric_row(metrics_df, "Economy", "Office leasing activity Q3")),
            RESEARCH_COLORS["blue"],
        ),
        (
            "Private Employment",
            research_metric_number(private_emp_row),
            RESEARCH_COLORS["green"],
        ),
        (
            "National Rate",
            None if private_emp_row is None or pd.isna(private_emp_row.get("comparison_value_num")) else float(private_emp_row["comparison_value_num"]),
            RESEARCH_COLORS["rule"],
        ),
    ]
    plot_df = pd.DataFrame(plot_rows, columns=["label", "value", "color"]).dropna(subset=["value"])
    if plot_df.empty:
        return build_research_empty_figure(
            "Selected Economic Indicators, Manhattan 2025",
            "Economic indicators are not available.",
        )

    fig = go.Figure(
        go.Bar(
            x=plot_df["label"],
            y=plot_df["value"],
            marker_color=plot_df["color"],
            hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Selected Economic Indicators, Manhattan 2025",
        margin=dict(l=20, r=20, t=70, b=20),
        height=320,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=RESEARCH_COLORS["ink_light"]),
        showlegend=False,
    )
    fig.update_yaxes(ticksuffix="%", gridcolor=RESEARCH_COLORS["paper_dark"])
    fig.update_xaxes(showgrid=False, title="")
    return fig


def build_research_global_chart():
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[22, 110, 11, 15, 19],
            theta=[
                "PM2.5 Reduction",
                "Revenue vs. Target",
                "Traffic Volume Reduction",
                "Speed Improvement",
                "Safety Improvement",
            ],
            fill="toself",
            name="NYC (2025)",
            line=dict(color=RESEARCH_COLORS["accent"], width=2),
            fillcolor="rgba(192, 57, 43, 0.18)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[7, 100, 18, 20, 14],
            theta=[
                "PM2.5 Reduction",
                "Revenue vs. Target",
                "Traffic Volume Reduction",
                "Speed Improvement",
                "Safety Improvement",
            ],
            fill="toself",
            name="London / Stockholm-era avg.",
            line=dict(color=RESEARCH_COLORS["blue"], width=2, dash="dash"),
            fillcolor="rgba(41, 128, 185, 0.08)",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Global Congestion Pricing: Comparative PM2.5 Reduction",
        margin=dict(l=20, r=20, t=70, b=20),
        height=360,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=RESEARCH_COLORS["ink_light"]),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        polar=dict(
            bgcolor="white",
            radialaxis=dict(visible=True, range=[0, 120], showticklabels=False, gridcolor=RESEARCH_COLORS["paper_dark"]),
            angularaxis=dict(gridcolor=RESEARCH_COLORS["paper_dark"]),
        ),
    )
    return fig


def make_research_stat_cell(value: str, label: str):
    return ui.div(
        ui.div(value, class_="research-stat-num"),
        ui.div(label, class_="research-stat-label"),
        class_="research-stat-cell",
    )


def make_research_metric_card(label: str, value: str, body: str, accent_class: str):
    return ui.div(
        ui.div(label, class_="research-card-label"),
        ui.div(value, class_="research-card-num"),
        ui.p(body, class_="research-card-copy"),
        class_=f"research-metric-card {accent_class}",
    )


def make_research_pullquote(quote: str, cite: str):
    return ui.div(
        ui.p(f'"{quote}"', class_="research-pullquote-text"),
        ui.div(cite, class_="research-pullquote-cite"),
        class_="research-pullquote",
    )


def make_research_bar_row(label: str, value_text: str, width_pct: float, fill_class: str = ""):
    fill_class_name = "research-bar-fill" if not fill_class else f"research-bar-fill {fill_class}"
    return ui.div(
        ui.div(
            ui.tags.span(label),
            ui.tags.span(value_text),
            class_="research-bar-row-label",
        ),
        ui.div(
            ui.div(
                style=f"width: {max(0.0, min(width_pct, 100.0)):.1f}%;",
                class_=fill_class_name,
            ),
            class_="research-bar-track",
        ),
    )


def make_overview_kpi_card(
    title: str,
    headline: str,
    subhead: str,
    metrics: list[tuple[str, str]],
    accent_class: str,
):
    metric_nodes = [
        ui.div(
            ui.div(label, class_="kpi-metric-label"),
            ui.div(value, class_="kpi-metric-value"),
            class_="kpi-metric",
        )
        for label, value in metrics
    ]
    return ui.column(
        3,
        ui.card(
            ui.div(title, class_="kpi-title"),
            ui.div(
                ui.div(
                    ui.div(headline, class_="kpi-delta"),
                    ui.div(subhead, class_="kpi-pct"),
                    class_="kpi-topline",
                ),
                ui.div(*metric_nodes, class_="kpi-grid"),
                class_="kpi-body",
            ),
            class_=f"kpi-card {accent_class}",
        ),
    )


def build_overview_speed_chart(metrics_df: pd.DataFrame):
    rows = [
        ("CBD Overall", research_metric_number(research_metric_row(metrics_df, "Speed", "CBD overall speed improvement")), RESEARCH_COLORS["accent"]),
        ("Highways", research_metric_number(research_metric_row(metrics_df, "Speed", "Highway speed improvement")), RESEARCH_COLORS["green"]),
        ("Arterial Roads", research_metric_number(research_metric_row(metrics_df, "Speed", "Arterial road speed improvement")), RESEARCH_COLORS["blue"]),
        ("Local Roads", research_metric_number(research_metric_row(metrics_df, "Speed", "Local road speed improvement")), RESEARCH_COLORS["gold"]),
        ("Weekend Evenings", research_metric_number(research_metric_row(metrics_df, "Speed", "Weekend evening speed improvement (3-9PM)")), RESEARCH_COLORS["accent"]),
    ]
    plot_df = pd.DataFrame(rows, columns=["label", "value", "color"]).dropna(subset=["value"])
    if plot_df.empty:
        return build_research_empty_figure("Speed Gains by Road Type", "Speed comparison data is not available.")

    fig = go.Figure(
        go.Bar(
            x=plot_df["value"],
            y=plot_df["label"],
            orientation="h",
            marker_color=plot_df["color"],
            text=[f"+{v:.0f}%" for v in plot_df["value"]],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:.0f}%<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Speed Gains by Road Type",
        margin=dict(l=20, r=20, t=70, b=20),
        height=320,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=RESEARCH_COLORS["ink_light"]),
        showlegend=False,
    )
    fig.update_xaxes(range=[0, 30], ticksuffix="%", gridcolor=RESEARCH_COLORS["paper_dark"])
    fig.update_yaxes(title="", autorange="reversed", showgrid=False)
    return fig


def build_hero_panel(title: str, subtitle: str, credit: str):
    return ui.div(
        ui.div(
            ui.div("MANHATTAN CONGESTION RELIEF ZONE", class_="hero-kicker"),
            ui.div(credit, class_="hero-credit"),
            class_="hero-toprow",
        ),
        ui.div(title, class_="hero-title"),
        ui.p(subtitle, class_="hero-subtitle"),
        class_="hero-panel",
    )


def build_shared_styles():
    return ui.tags.head(
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
            .hero-toprow {
              display: flex;
              justify-content: space-between;
              align-items: flex-start;
              gap: 16px;
              margin-bottom: 10px;
            }
            .hero-kicker {
              font-size: 0.82rem;
              letter-spacing: 0.18em;
              text-transform: uppercase;
              opacity: 0.82;
              margin-bottom: 0;
            }
            .hero-credit {
              font-size: 0.82rem;
              font-weight: 600;
              opacity: 0.88;
              text-align: right;
              white-space: nowrap;
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
            .nav-tabs {
              border-bottom: none;
              gap: 10px;
              margin: 0 auto;
              max-width: 1480px;
              padding: 18px 10px 0;
            }
            .nav-tabs .nav-link {
              border: 1px solid rgba(20, 33, 61, 0.10);
              border-radius: 999px;
              background: rgba(255, 255, 255, 0.78);
              color: var(--ink);
              font-weight: 700;
              padding: 10px 18px;
            }
            .nav-tabs .nav-link.active {
              background: var(--ink);
              color: #f8fafc;
              border-color: transparent;
              box-shadow: 0 10px 22px rgba(20, 33, 61, 0.16);
            }
            .research-shell {
              max-width: 1480px;
              margin: 0 auto;
              padding: 24px 10px 48px;
              color: #0f1923;
            }
            .research-shell p,
            .research-shell li,
            .research-shell td,
            .research-shell th {
              font-family: Georgia, "Times New Roman", serif;
            }
            .research-intro-note {
              margin-bottom: 16px;
              background: rgba(255, 255, 255, 0.76);
              border: 1px solid rgba(15, 25, 35, 0.08);
              border-radius: 16px;
              padding: 12px 16px;
              color: #536171;
              line-height: 1.5;
            }
            .research-hero {
              background: #101924;
              color: #f4f0e8;
              border-radius: 28px 28px 0 0;
              overflow: hidden;
              position: relative;
              border: 1px solid rgba(15, 25, 35, 0.12);
              border-bottom: none;
            }
            .research-hero::before {
              content: "";
              position: absolute;
              inset: 0;
              background: repeating-linear-gradient(
                0deg,
                transparent,
                transparent 47px,
                rgba(255, 255, 255, 0.035) 47px,
                rgba(255, 255, 255, 0.035) 48px
              );
              pointer-events: none;
            }
            .research-hero-inner {
              position: relative;
              max-width: 1120px;
              margin: 0 auto;
              padding: 72px 44px 54px;
            }
            .research-hero-label,
            .research-hero-meta-label,
            .research-stat-label,
            .research-section-label,
            .research-chart-source,
            .research-card-label,
            .research-pullquote-cite,
            .research-footer-note,
            .research-timeline-year,
            .research-compare-col h4,
            .research-table th {
              font-family: Consolas, "Courier New", monospace;
            }
            .research-hero-label {
              font-size: 0.72rem;
              letter-spacing: 0.18em;
              text-transform: uppercase;
              color: #c0392b;
              margin-bottom: 18px;
            }
            .research-hero-title {
              font-family: Georgia, "Times New Roman", serif;
              font-size: clamp(2.8rem, 5.4vw, 5rem);
              line-height: 0.98;
              font-weight: 700;
              letter-spacing: -0.04em;
              margin-bottom: 18px;
            }
            .research-hero-title em {
              color: #e8d5a3;
              font-style: italic;
            }
            .research-hero-subtitle {
              max-width: 760px;
              font-size: 1.16rem;
              line-height: 1.7;
              color: rgba(244, 240, 232, 0.72);
              margin: 0;
            }
            .research-hero-meta {
              display: grid;
              grid-template-columns: repeat(4, minmax(0, 1fr));
              gap: 18px 26px;
              margin-top: 34px;
              padding-top: 18px;
              border-top: 1px solid rgba(244, 240, 232, 0.16);
            }
            .research-hero-meta-label {
              font-size: 0.66rem;
              letter-spacing: 0.12em;
              text-transform: uppercase;
              color: rgba(244, 240, 232, 0.52);
              margin-bottom: 4px;
            }
            .research-hero-meta-value {
              color: rgba(244, 240, 232, 0.88);
              font-size: 0.96rem;
            }
            .research-stats-band {
              display: grid;
              grid-template-columns: repeat(4, minmax(0, 1fr));
              background: #c0392b;
              border: 1px solid rgba(15, 25, 35, 0.12);
              border-top: none;
              border-radius: 0 0 28px 28px;
              overflow: hidden;
              box-shadow: 0 22px 40px rgba(15, 25, 35, 0.14);
            }
            .research-stat-cell {
              padding: 26px 18px;
              text-align: center;
              border-right: 1px solid rgba(255, 255, 255, 0.16);
            }
            .research-stat-cell:last-child {
              border-right: none;
            }
            .research-stat-num {
              display: block;
              font-family: Georgia, "Times New Roman", serif;
              font-size: 2.8rem;
              font-weight: 700;
              line-height: 1;
              color: white;
              margin-bottom: 6px;
            }
            .research-stat-label {
              font-size: 0.68rem;
              letter-spacing: 0.12em;
              text-transform: uppercase;
              color: rgba(255, 255, 255, 0.86);
            }
            .research-section {
              background: rgba(244, 240, 232, 0.84);
              border: 1px solid rgba(15, 25, 35, 0.10);
              border-radius: 28px;
              padding: 38px 34px;
              margin-top: 26px;
              box-shadow: 0 14px 30px rgba(15, 25, 35, 0.08);
            }
            .research-section-label {
              font-size: 0.7rem;
              letter-spacing: 0.18em;
              text-transform: uppercase;
              color: #c0392b;
              margin-bottom: 10px;
            }
            .research-section-title {
              font-family: Georgia, "Times New Roman", serif;
              font-size: 2.1rem;
              font-weight: 700;
              line-height: 1.12;
              margin: 0 0 18px;
              color: #0f1923;
            }
            .research-lead {
              max-width: 760px;
              color: #2a3a4a;
              margin-bottom: 0;
            }
            .research-two-col {
              display: grid;
              grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
              gap: 30px;
              align-items: start;
            }
            .research-three-col {
              display: grid;
              grid-template-columns: repeat(3, minmax(0, 1fr));
              gap: 18px;
              margin-top: 22px;
            }
            .research-chart-card {
              background: white;
              border: 1px solid #c8bfa8;
              border-top: 4px solid #0f1923;
              border-radius: 20px;
              padding: 18px 18px 8px;
            }
            .research-chart-title {
              font-family: Georgia, "Times New Roman", serif;
              font-size: 1.35rem;
              font-weight: 700;
              margin-bottom: 4px;
            }
            .research-chart-source {
              font-size: 0.64rem;
              letter-spacing: 0.1em;
              text-transform: uppercase;
              color: #7c838b;
              margin-bottom: 10px;
            }
            .research-metric-card {
              background: white;
              border: 1px solid #c8bfa8;
              border-radius: 20px;
              padding: 22px 20px;
              min-height: 220px;
              position: relative;
            }
            .research-metric-card::before {
              content: "";
              position: absolute;
              left: 0;
              right: 0;
              top: 0;
              height: 4px;
              background: #c0392b;
              border-radius: 20px 20px 0 0;
            }
            .research-blue::before { background: #2980b9; }
            .research-green::before { background: #16a085; }
            .research-gold::before { background: #b8860b; }
            .research-card-label {
              font-size: 0.7rem;
              letter-spacing: 0.14em;
              text-transform: uppercase;
              color: #8a8f95;
              margin-bottom: 12px;
            }
            .research-card-num {
              font-family: Georgia, "Times New Roman", serif;
              font-size: 2.45rem;
              line-height: 1;
              font-weight: 700;
              color: #0f1923;
              margin-bottom: 12px;
            }
            .research-card-copy {
              margin: 0;
              color: #2a3a4a;
              line-height: 1.62;
            }
            .research-rule {
              text-align: center;
              color: #a39b8b;
              letter-spacing: 0.5em;
              margin: 24px 0;
            }
            .research-pullquote {
              margin-top: 20px;
              border-left: 4px solid #c0392b;
              background: #e8e2d5;
              padding: 16px 18px;
              border-radius: 0 18px 18px 0;
            }
            .research-pullquote-text {
              margin: 0 0 8px;
              font-size: 1.16rem;
              line-height: 1.55;
              color: #0f1923;
              font-style: italic;
            }
            .research-pullquote-cite {
              font-size: 0.68rem;
              letter-spacing: 0.08em;
              text-transform: uppercase;
              color: #c0392b;
            }
            .research-timeline {
              display: flex;
              flex-direction: column;
              gap: 12px;
              margin-top: 8px;
            }
            .research-timeline-item {
              display: grid;
              grid-template-columns: 110px 1fr;
              gap: 14px;
              padding-bottom: 12px;
              border-bottom: 1px solid rgba(15, 25, 35, 0.08);
            }
            .research-timeline-item:last-child {
              border-bottom: none;
              padding-bottom: 0;
            }
            .research-timeline-year {
              font-size: 0.72rem;
              letter-spacing: 0.12em;
              text-transform: uppercase;
              color: #c0392b;
              padding-top: 3px;
            }
            .research-timeline-text {
              color: #2a3a4a;
              line-height: 1.58;
            }
            .research-table {
              width: 100%;
              border-collapse: collapse;
              overflow: hidden;
              border-radius: 18px;
              margin-top: 18px;
              background: white;
            }
            .research-table th {
              background: #0f1923;
              color: #f4f0e8;
              font-size: 0.68rem;
              letter-spacing: 0.1em;
              text-transform: uppercase;
              padding: 14px 12px;
            }
            .research-table td {
              padding: 12px 12px;
              border-bottom: 1px solid #e8e2d5;
              color: #2a3a4a;
              vertical-align: top;
            }
            .research-table tr:last-child td {
              border-bottom: none;
            }
            .research-footer-note {
              margin-top: 10px;
              font-size: 0.66rem;
              letter-spacing: 0.08em;
              text-transform: uppercase;
              color: #7c838b;
            }
            .research-bar-stack {
              display: flex;
              flex-direction: column;
              gap: 14px;
            }
            .research-bar-row-label {
              display: flex;
              justify-content: space-between;
              gap: 16px;
              font-family: Consolas, "Courier New", monospace;
              font-size: 0.8rem;
              color: #0f1923;
              margin-bottom: 6px;
            }
            .research-bar-track {
              height: 10px;
              background: #e8e2d5;
              border-radius: 999px;
              overflow: hidden;
            }
            .research-bar-fill {
              height: 100%;
              background: #0f1923;
              border-radius: 999px;
            }
            .research-bar-fill.blue { background: #2980b9; }
            .research-bar-fill.green { background: #16a085; }
            .research-bar-fill.red { background: #c0392b; }
            .research-compare-grid {
              display: grid;
              grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 18px;
              margin-top: 24px;
            }
            .research-compare-col {
              background: white;
              border: 1px solid #c8bfa8;
              border-radius: 20px;
              padding: 20px;
            }
            .research-compare-col h4 {
              margin: 0 0 12px;
              font-size: 0.82rem;
              letter-spacing: 0.1em;
              text-transform: uppercase;
              color: #0f1923;
            }
            .research-compare-col ul {
              margin: 0;
              padding-left: 18px;
              color: #2a3a4a;
              line-height: 1.62;
            }
            .research-compare-col li + li {
              margin-top: 10px;
            }
            @media (max-width: 991px) {
              .hero-toprow {
                flex-direction: column;
                gap: 6px;
              }
              .hero-credit {
                text-align: left;
                white-space: normal;
              }
              .hero-title {
                font-size: 1.8rem;
              }
              .kpi-card {
                min-height: auto;
              }
              .research-hero-inner {
                padding: 48px 24px 36px;
              }
              .research-hero-meta,
              .research-stats-band {
                grid-template-columns: repeat(2, minmax(0, 1fr));
              }
              .research-two-col,
              .research-three-col,
              .research-compare-grid {
                grid-template-columns: 1fr;
              }
              .research-section {
                padding: 28px 22px;
              }
              .research-timeline-item {
                grid-template-columns: 1fr;
                gap: 6px;
              }
            }
            """
        )
    )


def bus_speed_page_ui():
    return ui.div(
        build_hero_panel(
            title="Bus Speed Impact Dashboard",
            subtitle=(
                "Local CSV analysis with KPI cards, filterable comparisons, and auto-generated manager summary. "
                "The focus stays on before/after change around CRZ start, while the line charts show how the selected slice behaves over time."
            ),
            credit="Designed by Yixuan Wang",
        ),
        ui.card(
            ui.card_header("Filters"),
            ui.row(
                ui.column(2, ui.input_date("cutoff_date", "CRZ start date", value=DEFAULT_CUTOFF_DATE, format="yyyy-mm-dd")),
                ui.column(
                    2,
                    ui.input_select(
                        "window_months",
                        "Before/After window",
                        choices={"3": "3 months", "6": "6 months", "12": "12 months"},
                        selected=str(DEFAULT_WINDOW_MONTHS),
                    ),
                ),
                ui.column(2, ui.input_select("day_filter", "Day type", choices=DAY_CHOICES, selected="All")),
                ui.column(2, ui.input_select("all_period_filter", "All-bus period", choices=ALL_PERIOD_CHOICES, selected="All")),
                ui.column(2, ui.input_select("cbd_period_filter", "CBD period", choices=CBD_PERIOD_CHOICES, selected="All")),
                ui.column(2, ui.input_select("borough_filter", "Borough (all-bus only)", choices=BOROUGH_CHOICES, selected="All")),
            ),
            ui.row(
                ui.column(6, ui.input_text("route_filter", "Route contains", placeholder="e.g. M15, Bx12, Q44")),
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
            ui.column(7, ui.card(ui.card_header("Executive Summary"), ui.output_ui("executive_summary"), class_="summary-card")),
            ui.column(5, ui.card(ui.card_header("So What"), ui.output_ui("so_what"), class_="summary-card")),
        ),
        ui.row(
            ui.column(6, ui.card(ui.card_header("All Local Bus: Before vs After"), output_widget("all_before_after_plot"), class_="chart-card")),
            ui.column(6, ui.card(ui.card_header("CBD Local Bus: Before vs After"), output_widget("cbd_before_after_plot"), class_="chart-card")),
        ),
        ui.row(
            ui.column(6, ui.card(ui.card_header("CBD: Express vs Local Speed"), output_widget("cbd_exp_local_plot"), class_="chart-card")),
            ui.column(6, ui.card(ui.card_header("Local Bus: Within CBD vs Without CBD"), output_widget("within_without_plot"), class_="chart-card")),
        ),
        class_="dashboard-shell",
    )


def traffic_volume_page_ui():
    return ui.div(
        build_hero_panel(
            title="Traffic Volume Impact Dashboard",
            subtitle=(
                "Daily bridge-and-tunnel counts are rolled into complete-month average daily traffic so the before/after comparison stays comparable around CRZ launch. "
                "The default view emphasizes inbound traffic, which is the slice most directly tied to Manhattan entry demand."
            ),
            credit="Designed by Sijin Li",
        ),
        ui.card(
            ui.card_header("Filters"),
            ui.row(
                ui.column(2, ui.input_date("traffic_cutoff_date", "CRZ start date", value=DEFAULT_CUTOFF_DATE, format="yyyy-mm-dd")),
                ui.column(
                    2,
                    ui.input_select(
                        "traffic_window_months",
                        "Before/After window",
                        choices={"3": "3 months", "6": "6 months", "12": "12 months"},
                        selected=str(DEFAULT_WINDOW_MONTHS),
                    ),
                ),
                ui.column(2, ui.input_select("traffic_direction_filter", "Direction", choices=TRAFFIC_DIRECTION_CHOICES, selected="I")),
                ui.column(2, ui.input_select("traffic_plaza_filter", "Location", choices=traffic_plaza_choices(), selected="All")),
                ui.column(2, ui.input_date("traffic_start_date", "Trend start", value=DEFAULT_TRAFFIC_START_DATE, format="yyyy-mm-dd")),
                ui.column(2, ui.input_date("traffic_end_date", "Trend end", value=DEFAULT_TRAFFIC_END_DATE, format="yyyy-mm-dd")),
            ),
            ui.div(
                "This page uses monthly average daily traffic instead of raw monthly totals so months with different day counts stay comparable. "
                "The latest incomplete month is automatically excluded to avoid understating post-CRZ traffic.",
                class_="filter-note",
            ),
            class_="filter-card",
        ),
        ui.output_ui("traffic_context_note"),
        ui.output_ui("traffic_kpi_cards"),
        ui.row(
            ui.column(7, ui.card(ui.card_header("Traffic Volume Summary"), ui.output_ui("traffic_summary"), class_="summary-card")),
            ui.column(5, ui.card(ui.card_header("Interpretation"), ui.output_ui("traffic_so_what"), class_="summary-card")),
        ),
        ui.row(
            ui.column(6, ui.card(ui.card_header("Traffic Volume: Before vs After"), output_widget("traffic_before_after_plot"), class_="chart-card")),
            ui.column(6, ui.card(ui.card_header("Monthly Traffic Trend"), output_widget("traffic_trend_plot"), class_="chart-card")),
        ),
        class_="dashboard-shell",
    )


def subway_ridership_page_ui():
    return ui.div(
        build_hero_panel(
            title="Subway Ridership Impact Dashboard",
            subtitle=(
                "This tab folds the cloned subway-ridership project into the same CRZ dashboard shell, "
                "so we can read transit demand alongside bus speed and traffic volume in one place."
            ),
            credit="Designed by Kegan Lin and Jack Zhou",
        ),
        ui.card(
            ui.card_header("Filters"),
            ui.row(
                ui.column(
                    6,
                    ui.input_select(
                        "subway_peak_choice",
                        "Peak hour",
                        choices=SUBWAY_PEAK_CHOICES,
                        selected="Morning Peak",
                    ),
                ),
                ui.column(
                    6,
                    ui.input_select(
                        "subway_borough_choice",
                        "Borough",
                        choices=SUBWAY_BOROUGH_CHOICES,
                        selected="All",
                    ),
                ),
            ),
            ui.div(
                "The weekly trend and KPI cards follow the selected borough filter. "
                "The borough comparison chart keeps the selected peak period but shows every borough for context. "
                "This tab uses the merged local CSV and excludes the 2020 COVID distortion period.",
                class_="filter-note",
            ),
            class_="filter-card",
        ),
        ui.output_ui("subway_context_note"),
        ui.output_ui("subway_kpi_cards"),
        ui.row(
            ui.column(
                7,
                ui.card(
                    ui.card_header("Subway Ridership Summary"),
                    ui.output_ui("subway_summary"),
                    class_="summary-card",
                ),
            ),
            ui.column(
                5,
                ui.card(
                    ui.card_header("Interpretation"),
                    ui.output_ui("subway_so_what"),
                    class_="summary-card",
                ),
            ),
        ),
        ui.row(
            ui.column(
                6,
                ui.card(
                    ui.card_header("Weekly Subway Ridership Trend"),
                    output_widget("subway_trend_plot"),
                    class_="chart-card",
                ),
            ),
            ui.column(
                6,
                ui.card(
                    ui.card_header("Average Ridership by Borough"),
                    output_widget("subway_borough_plot"),
                    class_="chart-card",
                ),
            ),
        ),
        class_="dashboard-shell",
    )


def research_overview_page_ui():
    return ui.div(
        build_hero_panel(
            title="Year-One Congestion Pricing Comparison",
            subtitle=(
                "This tab turns the imported Dashboard Jiashuo Xu source folder into the same dashboard format as the rest of the app. "
                "The emphasis stays on before-CRZ versus after-CRZ comparisons, with the pulled CSV files driving the headline cards and charts."
            ),
            credit="Designed by Jiashuo Xu",
        ),
        ui.card(
            ui.card_header("Data Scope"),
            ui.tags.ul(
                ui.tags.li("Traffic compares 2025 CRZ vehicle entries against the pulled 2024 baseline from the cloned CSV."),
                ui.tags.li("Speed, safety, air, and economy metrics come from the cloned overview CSV and preserve the source project's comparison windows."),
                ui.tags.li("Revenue compares monthly 2025 net revenue against the official target so this tab stays focused on measurable deltas."),
            ),
            ui.div(
                "Policy background from the HTML is kept only as supporting context; the main story here is data comparison.",
                class_="filter-note",
            ),
            class_="filter-card",
        ),
        ui.output_ui("overview_context_note"),
        ui.output_ui("overview_kpi_cards"),
        ui.row(
            ui.column(
                7,
                ui.card(
                    ui.card_header("Overview Summary"),
                    ui.output_ui("overview_summary"),
                    class_="summary-card",
                ),
            ),
            ui.column(
                5,
                ui.card(
                    ui.card_header("Interpretation"),
                    ui.output_ui("overview_so_what"),
                    class_="summary-card",
                ),
            ),
        ),
        ui.row(
            ui.column(
                6,
                ui.card(
                    ui.card_header("Traffic Volume: 2024 Baseline vs 2025 CRZ"),
                    output_widget("overview_traffic_chart"),
                    class_="chart-card",
                ),
            ),
            ui.column(
                6,
                ui.card(
                    ui.card_header("Revenue: Actual vs Monthly Target"),
                    output_widget("overview_revenue_chart"),
                    class_="chart-card",
                ),
            ),
        ),
        ui.row(
            ui.column(
                6,
                ui.card(
                    ui.card_header("Speed Gains by Road Type"),
                    output_widget("overview_speed_chart"),
                    class_="chart-card",
                ),
            ),
            ui.column(
                6,
                ui.card(
                    ui.card_header("Safety & Quality-of-Life Change"),
                    output_widget("overview_safety_chart"),
                    class_="chart-card",
                ),
            ),
        ),
        ui.row(
            ui.column(
                6,
                ui.card(
                    ui.card_header("Air Quality Comparison"),
                    output_widget("overview_air_chart"),
                    class_="chart-card",
                ),
            ),
            ui.column(
                6,
                ui.card(
                    ui.card_header("Economic Indicators"),
                    output_widget("overview_economy_chart"),
                    class_="chart-card",
                ),
            ),
        ),
        ui.row(
            ui.column(
                6,
                ui.card(
                    ui.card_header("Funding Notes from the Cloned Project"),
                    ui.output_ui("overview_funding_notes"),
                    class_="summary-card",
                ),
            ),
            ui.column(
                6,
                ui.card(
                    ui.card_header("Caveats"),
                    ui.output_ui("overview_caveats"),
                    class_="summary-card",
                ),
            ),
        ),
        class_="dashboard-shell",
    )


bus_speed_server = server


app_ui = ui.page_fluid(
    build_shared_styles(),
    ui.navset_tab(
        ui.nav_panel("Bus Speed", bus_speed_page_ui()),
        ui.nav_panel("Traffic Volume", traffic_volume_page_ui()),
        ui.nav_panel("Subway Ridership", subway_ridership_page_ui()),
        ui.nav_panel("CRZ Overview", research_overview_page_ui()),
        id="main_nav",
    ),
)


def server(input, output, session):
    bus_speed_server(input, output, session)

    @reactive.calc
    def overview_bundle():
        errors: list[str] = []

        try:
            metrics = load_research_metrics_raw()
        except Exception as exc:
            metrics = pd.DataFrame()
            errors.append(f"Research metrics load failed: {exc}")

        try:
            traffic = load_research_traffic_volume_raw()
        except Exception as exc:
            traffic = pd.DataFrame()
            errors.append(f"Research traffic CSV load failed: {exc}")

        try:
            revenue = load_research_revenue_raw()
        except Exception as exc:
            revenue = pd.DataFrame()
            errors.append(f"Research revenue CSV load failed: {exc}")

        return {"metrics": metrics, "traffic": traffic, "revenue": revenue, "errors": errors}

    @render.ui
    def overview_context_note():
        bundle = overview_bundle()
        if bundle["errors"]:
            return ui.div(
                "Data warning: " + " | ".join(bundle["errors"]),
                class_="context-note",
            )
        return ui.div(
            "Traffic uses the pulled 2024 baseline versus 2025 CRZ entries. Revenue uses the pulled monthly series versus the official target. "
            "Other supporting indicators keep the comparison windows stated in the cloned project's CSV.",
            class_="context-note",
        )

    @reactive.calc
    def overview_story_stats():
        metrics = overview_bundle()["metrics"]
        traffic = overview_bundle()["traffic"]
        revenue = overview_bundle()["revenue"]

        traffic_before = None if traffic.empty else float(traffic["baseline_2024_daily_thousands"].mean())
        traffic_after = None if traffic.empty else float(traffic["crz_2025_daily_thousands"].mean())
        traffic_delta = None if traffic_before is None or traffic_after is None else traffic_after - traffic_before
        traffic_pct = None if traffic_before in (None, 0) or traffic_delta is None else traffic_delta / traffic_before * 100.0

        speed_before = research_metric_number(research_metric_row(metrics, "Speed", "CBD average speed before"))
        speed_after = research_metric_number(research_metric_row(metrics, "Speed", "CBD average speed after"))
        speed_delta = None if speed_before is None or speed_after is None else speed_after - speed_before
        speed_pct = research_metric_number(research_metric_row(metrics, "Speed", "CBD overall speed improvement"))

        revenue_row = research_metric_row(metrics, "Revenue", "Annual net revenue (projected)")
        revenue_actual = research_metric_number(revenue_row)
        revenue_target = None if revenue_row is None or pd.isna(revenue_row.get("comparison_value_num")) else float(revenue_row["comparison_value_num"])
        revenue_delta = None if revenue_actual is None or revenue_target is None else revenue_actual - revenue_target
        revenue_months_met = 0 if revenue.empty else int(revenue["met_target"].astype(str).str.upper().eq("TRUE").sum())

        air_pct = research_metric_number(research_metric_row(metrics, "Air Quality", "PM2.5 reduction in CRZ"))
        air_abs = research_metric_number(research_metric_row(metrics, "Air Quality", "PM2.5 absolute reduction in CRZ"))
        emissions_row = research_metric_row(metrics, "Air Quality", "Vehicular emissions reduction")

        total_fewer = research_metric_number(research_metric_row(metrics, "Traffic", "Total vehicles kept out Year 1"))
        avg_daily_reduction = research_metric_number(research_metric_row(metrics, "Traffic", "Average daily reduction"))
        peak_window = research_metric_number(research_metric_row(metrics, "Traffic", "Morning rush hour speed improvement"))
        weekend_speed = research_metric_number(research_metric_row(metrics, "Speed", "Weekend evening speed improvement (3-9PM)"))
        office_attendance = research_metric_row(metrics, "Economy", "Manhattan office attendance")
        private_emp = research_metric_row(metrics, "Economy", "Private sector employment NYC")

        return {
            "traffic_before": traffic_before,
            "traffic_after": traffic_after,
            "traffic_delta": traffic_delta,
            "traffic_pct": traffic_pct,
            "speed_before": speed_before,
            "speed_after": speed_after,
            "speed_delta": speed_delta,
            "speed_pct": speed_pct,
            "revenue_actual": revenue_actual,
            "revenue_target": revenue_target,
            "revenue_delta": revenue_delta,
            "revenue_months_met": revenue_months_met,
            "air_pct": air_pct,
            "air_abs": air_abs,
            "emissions_text": research_metric_display(emissions_row),
            "total_fewer": total_fewer,
            "avg_daily_reduction": avg_daily_reduction,
            "peak_window": peak_window,
            "weekend_speed": weekend_speed,
            "office_attendance": office_attendance,
            "private_emp": private_emp,
        }

    @render.ui
    def overview_kpi_cards():
        stats = overview_story_stats()

        traffic_headline = "N/A" if stats["traffic_delta"] is None else f"{format_signed(stats['traffic_delta'], 0)}k/day"
        traffic_subhead = "N/A" if stats["traffic_pct"] is None else f"{format_signed(stats['traffic_pct'], 1)}% vs 2024"
        traffic_before = "N/A" if stats["traffic_before"] is None else f"{stats['traffic_before']:.0f}k/day"
        traffic_after = "N/A" if stats["traffic_after"] is None else f"{stats['traffic_after']:.0f}k/day"
        total_fewer = "N/A" if stats["total_fewer"] is None else f"{stats['total_fewer'] / 1_000_000:.0f}M"
        avg_daily = "N/A" if stats["avg_daily_reduction"] is None else f"{format_whole_number(stats['avg_daily_reduction'])}/day"

        speed_headline = "N/A" if stats["speed_delta"] is None else f"{format_signed(stats['speed_delta'], 1)} MPH"
        speed_subhead = "N/A" if stats["speed_pct"] is None else f"+{stats['speed_pct']:.0f}% in CBD"
        speed_before = "N/A" if stats["speed_before"] is None else f"{stats['speed_before']:.1f} MPH"
        speed_after = "N/A" if stats["speed_after"] is None else f"{stats['speed_after']:.1f} MPH"
        peak_window = "N/A" if stats["peak_window"] is None else f"+{stats['peak_window']:.0f}%"
        weekend_speed = "N/A" if stats["weekend_speed"] is None else f"+{stats['weekend_speed']:.0f}%"

        revenue_headline = "N/A" if stats["revenue_delta"] is None else format_compact_millions(stats["revenue_delta"])
        if stats["revenue_delta"] is not None and stats["revenue_delta"] > 0:
            revenue_headline = f"+{revenue_headline}"
        revenue_subhead = "N/A" if stats["revenue_actual"] is None or stats["revenue_target"] is None else f"{format_compact_millions(stats['revenue_actual'])} vs {format_compact_millions(stats['revenue_target'])}"
        revenue_actual = "N/A" if stats["revenue_actual"] is None else format_compact_millions(stats["revenue_actual"])
        revenue_target = "N/A" if stats["revenue_target"] is None else format_compact_millions(stats["revenue_target"])
        months_met = f"{stats['revenue_months_met']}/12"

        air_headline = "N/A" if stats["air_pct"] is None else f"-{abs(stats['air_pct']):.0f}%"
        air_subhead = "PM2.5 vs control period"
        air_abs = "N/A" if stats["air_abs"] is None else f"-{abs(stats['air_abs']):.2f} ug/m3"

        return ui.row(
            make_overview_kpi_card(
                "Traffic Volume",
                traffic_headline,
                traffic_subhead,
                [
                    ("Before", traffic_before),
                    ("After", traffic_after),
                    ("Vehicles kept out", total_fewer),
                    ("Avg daily reduction", avg_daily),
                ],
                "",
            ),
            make_overview_kpi_card(
                "CBD Speed",
                speed_headline,
                speed_subhead,
                [
                    ("Before", speed_before),
                    ("After", speed_after),
                    ("Morning rush", peak_window),
                    ("Weekend evenings", weekend_speed),
                ],
                "kpi-b",
            ),
            make_overview_kpi_card(
                "Revenue",
                revenue_headline,
                revenue_subhead,
                [
                    ("Year-one report", revenue_actual),
                    ("Target", revenue_target),
                    ("Months over target", months_met),
                    ("Monthly target", "$41.6M"),
                ],
                "kpi-c",
            ),
            make_overview_kpi_card(
                "Air Quality",
                air_headline,
                air_subhead,
                [
                    ("Absolute drop", air_abs),
                    ("Vehicular emissions", stats["emissions_text"]),
                    ("Noise complaints", "-45%"),
                    ("Comparison city range", "London 7%, Stockholm 10%"),
                ],
                "kpi-d",
            ),
            class_="kpi-row",
        )

    @render.ui
    def overview_summary():
        stats = overview_story_stats()
        office_row = stats["office_attendance"]
        office_text = "Office attendance still rose from 72% to 76% of pre-pandemic levels in March 2025." if office_row is not None else "Office attendance also increased during the same period in the cloned source material."
        traffic_text = (
            "Traffic fell from about "
            f"{stats['traffic_before']:.0f}k to {stats['traffic_after']:.0f}k average daily CRZ entries, "
            f"or roughly {abs(stats['traffic_delta']):.0f}k fewer vehicles per day."
            if stats["traffic_before"] is not None and stats["traffic_after"] is not None and stats["traffic_delta"] is not None
            else "Traffic comparisons could not be fully computed from the pulled CSV."
        )
        speed_text = (
            f"CBD speeds improved from {stats['speed_before']:.1f} to {stats['speed_after']:.1f} MPH, "
            f"with the cloned project citing a {stats['speed_pct']:.0f}% overall gain."
            if stats["speed_before"] is not None and stats["speed_after"] is not None and stats["speed_pct"] is not None
            else "Speed comparisons remain available only as summary metrics from the cloned project."
        )
        revenue_text = (
            f"The year-one revenue headline is {format_compact_millions(stats['revenue_actual'])} against a "
            f"{format_compact_millions(stats['revenue_target'])} target, and the monthly series clears target "
            f"in {stats['revenue_months_met']} of 12 months."
            if stats["revenue_actual"] is not None and stats["revenue_target"] is not None
            else "Revenue comparison is only partially available."
        )
        return make_summary_list(
            [
                traffic_text,
                speed_text,
                office_text,
                revenue_text,
            ]
        )

    @render.ui
    def overview_so_what():
        return ui.div(
            make_summary_list(
                [
                    "Use this tab as the top-level synthesis after drilling into the raw bus, traffic, and subway tabs.",
                    "The cleanest before/after evidence here is traffic volume, CBD speed, and revenue-versus-target.",
                    "Air quality and economy remain supporting comparisons from the cloned overview CSV, not raw local course files.",
                ]
            ),
            ui.div(
                "That keeps the fourth tab focused on the cloned project's strongest comparative claims instead of turning it into a separate policy microsite.",
                class_="so-what-note",
            ),
        )

    @render.ui
    def overview_funding_notes():
        metrics = overview_bundle()["metrics"]
        capital_row = research_metric_row(metrics, "Revenue", "Transit capital unlocked via bonds")
        projects_row = research_metric_row(metrics, "Revenue", "Active construction projects (Jan 2026)")
        return ui.div(
            make_summary_list(
                [
                    "The cloned project keeps the HTML funding section, but here it is compressed into the same card format as the other tabs.",
                    f"Year-one report headline: {research_metric_display(research_metric_row(metrics, 'Revenue', 'Annual net revenue (projected)'))} against a $500M target.",
                    f"Transit capital unlocked via bonds: {research_metric_display(capital_row)}.",
                    f"Projects already in construction as of Jan 2026: {research_metric_display(projects_row)}.",
                ]
            ),
            ui.div(
                "Revenue mix in the cloned CSV: 68% passenger vehicles, 22% taxis/FHV, 9% trucks, 1% buses and motorcycles.",
                class_="so-what-note",
            ),
        )

    @render.ui
    def overview_caveats():
        return ui.div(
            make_summary_list(
                [
                    "Not every metric uses the same comparison window: some are Jan-Feb, some Jan-Jun, and some full-year 2025 versus 2024.",
                    "The cloned repo's monthly revenue CSV sums above the HTML's $550M headline, so this tab keeps the report headline in KPIs and the pulled monthly values in the chart.",
                    "This is a synthesis tab; if you need the course project's raw local comparisons, the other three tabs remain the primary evidence base.",
                ]
            ),
        )

    @render_widget
    def overview_traffic_chart():
        return build_research_traffic_chart(overview_bundle()["traffic"])

    @render_widget
    def overview_speed_chart():
        return build_overview_speed_chart(overview_bundle()["metrics"])

    @render_widget
    def overview_safety_chart():
        return build_research_safety_chart(overview_bundle()["metrics"])

    @render_widget
    def overview_air_chart():
        return build_research_air_chart(overview_bundle()["metrics"])

    @render_widget
    def overview_revenue_chart():
        return build_research_revenue_chart(overview_bundle()["revenue"])

    @render_widget
    def overview_economy_chart():
        return build_research_economy_chart(overview_bundle()["metrics"])

    @reactive.calc
    def traffic_cutoff_month_start() -> pd.Timestamp:
        return pd.Timestamp(input.traffic_cutoff_date()).to_period("M").to_timestamp()

    @reactive.calc
    def traffic_window_months() -> int:
        try:
            value = int(input.traffic_window_months())
            return value if value > 0 else DEFAULT_WINDOW_MONTHS
        except Exception:
            return DEFAULT_WINDOW_MONTHS

    @reactive.calc
    def traffic_dates() -> tuple[date, date]:
        start_date = input.traffic_start_date()
        end_date = input.traffic_end_date()
        if start_date <= end_date:
            return start_date, end_date
        return end_date, start_date

    @reactive.calc
    def traffic_bundle():
        errors: list[str] = []
        try:
            start_date, end_date = traffic_dates()
            filtered = filter_traffic_raw(
                df=load_traffic_raw(),
                start_date=start_date,
                end_date=end_date,
                direction_filter=input.traffic_direction_filter(),
                plaza_filter=input.traffic_plaza_filter(),
            )
            monthly = monthly_traffic_series(filtered)
        except Exception as exc:
            monthly = empty_volume_df()
            errors.append(f"Traffic load failed: {exc}")
        return {"monthly": monthly, "errors": errors}

    @reactive.calc
    def traffic_stats():
        return before_after_volume_summary(
            monthly_df=traffic_bundle()["monthly"],
            cutoff_month_start=traffic_cutoff_month_start(),
            window_months=traffic_window_months(),
        )

    @render.ui
    def traffic_context_note():
        bundle = traffic_bundle()
        if bundle["errors"]:
            return ui.div("Data warning: " + " | ".join(bundle["errors"]), class_="context-note")
        start_date, end_date = traffic_dates()
        return ui.div(
            traffic_context_sentence(
                direction_filter=input.traffic_direction_filter(),
                plaza_filter=input.traffic_plaza_filter(),
                start_date=start_date,
                end_date=end_date,
            ),
            class_="context-note",
        )

    @render.ui
    def traffic_kpi_cards():
        stats = traffic_stats()
        return ui.div(
            ui.row(
                make_volume_kpi_card("Average Daily Traffic Change", stats, "kpi-a"),
                make_traffic_snapshot_card(
                    "Before Window",
                    None if stats is None else stats["before_avg"],
                    "Average daily traffic before CRZ",
                    "kpi-b",
                ),
                make_traffic_snapshot_card(
                    "After Window",
                    None if stats is None else stats["after_avg"],
                    "Average daily traffic after CRZ",
                    "kpi-d",
                ),
            ),
            class_="kpi-row",
        )

    @render.ui
    def traffic_summary():
        stats = traffic_stats()
        bundle = traffic_bundle()
        if bundle["errors"]:
            return ui.div("Summary unavailable because the traffic dataset failed to load.")
        if stats is None:
            return ui.div("Not enough complete monthly traffic data is available for the selected filters.")

        direction_label = TRAFFIC_DIRECTION_CHOICES.get(input.traffic_direction_filter(), input.traffic_direction_filter())
        plaza_text = "all plazas" if input.traffic_plaza_filter() == "All" else f"Plaza {input.traffic_plaza_filter()}"
        bullets = [
            (
                f"For {direction_label.lower()} traffic at {plaza_text}, average daily volume "
                f"{volume_change_phrase(stats['delta'], baseline=stats['before_avg'])} from "
                f"{format_whole_number(stats['before_avg'])} to {format_whole_number(stats['after_avg'])} vehicles/day "
                f"across the selected {traffic_window_months()}-month windows."
            ),
            (
                f"The net shift is {format_whole_number(stats['delta'])} vehicles/day "
                f"({format_signed(stats['pct'], 1)}%), comparing {stats['before_range']} against {stats['after_range']}."
            ),
            (
                "The calculation uses complete months only and averages by observed days, which avoids overstating changes driven by shorter months or the partial April 2025 tail."
            ),
            (
                "Inbound traffic is the default because it is the most direct proxy for vehicles entering Manhattan, which makes the CRZ comparison more policy-relevant than a pooled two-way count."
            ),
        ]
        return make_summary_list(bullets)

    @render.ui
    def traffic_so_what():
        stats = traffic_stats()
        if stats is None:
            return ui.div("Interpretation is unavailable for the selected traffic slice.")

        delta = stats["after_avg"] - stats["before_avg"]
        direction_text = "fewer" if delta < 0 else "more"
        bullets = [
            f"The selected slice translates to roughly {format_whole_number(abs(delta))} {direction_text} vehicles per day after CRZ in the comparison window.",
            "That makes the traffic page a useful complement to the bus-speed page: one view tracks whether road demand softened, and the other shows whether bus movement improved.",
            "If you switch to a single plaza, treat the result as a corridor-level signal; if you keep all plazas selected, treat it as a broader Manhattan access pattern.",
        ]
        return ui.div(
            make_summary_list(bullets),
            ui.div(
                "Use the bar chart for the headline before/after shift and the line chart to check whether the change is sustained month over month.",
                class_="so-what-note",
            ),
        )

    @render_widget
    def traffic_before_after_plot():
        return build_traffic_before_after_bar(
            monthly_df=traffic_bundle()["monthly"],
            title="Traffic Volume: Before vs After",
            cutoff_month_start=traffic_cutoff_month_start(),
            window_months=traffic_window_months(),
            accent_color="#2a9d8f",
        )

    @render_widget
    def traffic_trend_plot():
        return build_traffic_line(
            monthly_df=traffic_bundle()["monthly"],
            title="Monthly Average Daily Traffic",
            cutoff_month_start=traffic_cutoff_month_start(),
        )

    @reactive.calc
    def subway_bundle():
        errors: list[str] = []
        try:
            all_data = load_subway_raw()
            peak_all_boroughs = filter_subway_raw(
                df=all_data,
                peak_choice=input.subway_peak_choice(),
                borough_choice="All",
            )
            selected = filter_subway_raw(
                df=all_data,
                peak_choice=input.subway_peak_choice(),
                borough_choice=input.subway_borough_choice(),
            )
        except Exception as exc:
            all_data = empty_ridership_df(extra_cols=["borough", "peak_period", "period"])
            peak_all_boroughs = empty_ridership_df(extra_cols=["borough", "peak_period", "period"])
            selected = empty_ridership_df(extra_cols=["borough", "peak_period", "period"])
            errors.append(f"Subway load failed: {exc}")

        return {
            "all": all_data,
            "peak_all_boroughs": peak_all_boroughs,
            "selected": selected,
            "errors": errors,
        }

    @reactive.calc
    def subway_stats():
        return subway_selection_stats(subway_bundle()["selected"])

    @render.ui
    def subway_context_note():
        bundle = subway_bundle()
        if bundle["errors"]:
            return ui.div("Data warning: " + " | ".join(bundle["errors"]), class_="context-note")

        return ui.div(
            subway_context_sentence(
                peak_choice=input.subway_peak_choice(),
                borough_choice=input.subway_borough_choice(),
            ),
            class_="context-note",
        )

    @render.ui
    def subway_kpi_cards():
        stats = subway_stats()
        return ui.div(
            ui.row(
                make_ridership_change_card("Ridership Change", stats, "kpi-a"),
                make_ridership_snapshot_card(
                    "Pre-Policy Average",
                    None if stats is None else stats["before_avg"],
                    "Average ridership across pre-policy weeks",
                    "kpi-b",
                ),
                make_ridership_snapshot_card(
                    "Post-Policy Average",
                    None if stats is None else stats["after_avg"],
                    "Average ridership across post-policy weeks",
                    "kpi-c",
                ),
                make_ridership_snapshot_card(
                    "Total Riders in Selection",
                    None if stats is None else stats["total_riders"],
                    "Summed across the filtered weekly records",
                    "kpi-d",
                ),
            ),
            class_="kpi-row",
        )

    @render.ui
    def subway_summary():
        bundle = subway_bundle()
        stats = subway_stats()
        if bundle["errors"]:
            return ui.div("Summary unavailable because the subway ridership dataset failed to load.")
        if stats is None:
            return ui.div("Not enough pre-policy and post-policy subway data is available for the selected slice.")

        peak_text = input.subway_peak_choice().lower()
        borough_text = "all boroughs" if input.subway_borough_choice() == "All" else input.subway_borough_choice()
        peak_borough_change = subway_borough_change(bundle["peak_all_boroughs"])
        overall_peak_changes = subway_peak_change_summary(bundle["all"])

        bullets = [
            (
                f"For {peak_text} in {borough_text}, average subway ridership "
                f"{subway_change_phrase(stats['pct'])} by {abs(stats['pct']):.1f}% after CRZ, "
                f"moving from {format_whole_number(stats['before_avg'])} to {format_whole_number(stats['after_avg'])} riders."
            ),
            (
                f"The filtered selection contains {format_whole_number(stats['total_riders'])} total riders across "
                f"{stats['weeks']} observed weeks, spanning {stats['start_week']:%b %Y} to {stats['end_week']:%b %Y}."
            ),
        ]

        if not peak_borough_change.empty:
            top_borough = str(peak_borough_change.index[0])
            top_pct = float(peak_borough_change.iloc[0])
            bullets.append(
                f"For the selected peak period across all boroughs, {top_borough} shows the strongest post-policy shift at {format_signed(top_pct, 1)}%."
            )

        morning_change = overall_peak_changes.get("Morning Peak")
        evening_change = overall_peak_changes.get("Evening Peak")
        if morning_change is not None and evening_change is not None:
            bullets.append(
                f"Across the full subway dataset, morning peak changed {format_signed(morning_change, 1)}% and evening peak changed {format_signed(evening_change, 1)}%."
            )

        return make_summary_list(bullets)

    @render.ui
    def subway_so_what():
        bundle = subway_bundle()
        stats = subway_stats()
        if bundle["errors"] or stats is None:
            return ui.div("Interpretation is unavailable for the selected subway slice.")

        peak_all_stats = subway_selection_stats(bundle["peak_all_boroughs"])
        selected_borough = input.subway_borough_choice()
        comparison_line = "The selected borough is closely tracking the same-peak system average."

        if selected_borough != "All" and peak_all_stats is not None:
            pct_gap = stats["pct"] - peak_all_stats["pct"]
            if pct_gap > 1.0:
                comparison_line = (
                    f"{selected_borough} is running {abs(pct_gap):.1f} percentage points above the same-peak all-borough average."
                )
            elif pct_gap < -1.0:
                comparison_line = (
                    f"{selected_borough} is running {abs(pct_gap):.1f} percentage points below the same-peak all-borough average."
                )

        bullets = [
            "This tab is the passenger-side signal in the merged dashboard: rising subway demand alongside softer traffic volume or faster bus speeds is more consistent with mode shift than with a road-only story.",
            comparison_line,
            "Use the weekly line to judge whether the ridership change is sustained over time, and use the borough chart to see whether the gain is concentrated or broad-based.",
        ]
        return ui.div(
            make_summary_list(bullets),
            ui.div(
                "This keeps the cloned subway project aligned with the same before/after storytelling format used by the bus-speed and traffic tabs.",
                class_="so-what-note",
            ),
        )

    @render_widget
    def subway_trend_plot():
        borough_text = "All Boroughs" if input.subway_borough_choice() == "All" else input.subway_borough_choice()
        return build_subway_trend_plot(
            df=subway_bundle()["selected"],
            title=f"{input.subway_peak_choice()}: Weekly Ridership Trend ({borough_text})",
        )

    @render_widget
    def subway_borough_plot():
        return build_subway_borough_bar(
            df=subway_bundle()["peak_all_boroughs"],
            title=f"{input.subway_peak_choice()}: Average Ridership by Borough",
        )


app = App(app_ui, server)


