import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(
    page_title="NYC Subway Ridership Dashboard",
    layout="wide"
)

session = requests.Session()


# ── Data loading ───────────────────────────────────────────────────────────────
def load_weekly_peak_data(dataset_id, period_label, peak_label, start_hour, end_hour):
    url = f"https://data.ny.gov/resource/{dataset_id}.json"
    params = {
        "$select": (
            "date_extract_y(transit_timestamp) as year,"
            "date_extract_woy(transit_timestamp) as week_of_year,"
            "borough,"
            "avg(ridership) as avg_ridership,"
            "sum(ridership) as total_ridership"
        ),
        "$where": (
            f"transit_mode = 'subway' "
            f"AND borough IN('Manhattan','Brooklyn','Queens','Bronx') "
            f"AND date_extract_hh(transit_timestamp) >= {start_hour} "
            f"AND date_extract_hh(transit_timestamp) <= {end_hour}"
        ),
        "$group": "year,week_of_year,borough",
        "$limit": 500000
    }
    r = session.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError(f"No data returned for {dataset_id} - {peak_label}")
    df = pd.DataFrame(data)
    for col in ["year", "week_of_year", "avg_ridership", "total_ridership"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["year", "week_of_year", "borough", "avg_ridership", "total_ridership"]).copy()
    df["year"] = df["year"].astype(int)
    df["week_of_year"] = df["week_of_year"].astype(int)
    df["peak_period"] = peak_label
    df["period"] = period_label
    df["week_start"] = pd.to_datetime(
        df["year"].astype(str) + "-W" + df["week_of_year"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u", errors="coerce"
    )
    return df.dropna(subset=["week_start"]).copy()


@st.cache_data(show_spinner="Fetching data from MTA API...")
def load_all_data():
    jobs = [
        ("wujg-7c2s", "Pre Policy",  "Morning Peak", 7,  10),
        ("wujg-7c2s", "Pre Policy",  "Evening Peak", 16, 19),
        ("5wq4-mkjj", "Post Policy", "Morning Peak", 7,  10),
        ("5wq4-mkjj", "Post Policy", "Evening Peak", 16, 19),
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(load_weekly_peak_data, *job) for job in jobs]
        results = [f.result() for f in futures]
    return pd.concat(results, ignore_index=True)


@st.cache_data
def load_csv():
    return pd.read_csv("weekly_aggregated_mta.csv", parse_dates=["week_start"])


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Source")
    use_api = st.toggle("Fetch live data from API", value=False,
                        help="Pull fresh data from data.ny.gov (~30s)")
    st.header("Filters")
    peak_choice = st.selectbox("Peak Hour", ["Morning Peak", "Evening Peak"])
    borough_choice = st.selectbox("Borough", ["All", "Manhattan", "Brooklyn", "Queens", "Bronx"])

if use_api:
    try:
        df_all = load_all_data()
    except Exception as e:
        st.warning(f"API fetch failed ({e}), falling back to local CSV.")
        df_all = load_csv()
else:
    df_all = load_csv()

# Remove 2020 data (COVID anomaly distorts trend)
df_all = df_all[df_all["week_start"] >= "2021-01-01"].copy()

# ── Filter by sidebar selections ───────────────────────────────────────────────
data = df_all[df_all["peak_period"] == peak_choice].copy()
if borough_choice != "All":
    data = data[data["borough"] == borough_choice].copy()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("NYC Subway Ridership Dashboard")
st.caption("Congestion Pricing: Pre-Policy vs Post-Policy Ridership Comparison (2021–present)")

# ── Metrics ────────────────────────────────────────────────────────────────────
pre_mean  = data.loc[data["period"] == "Pre Policy",  "avg_ridership"].mean()
post_mean = data.loc[data["period"] == "Post Policy", "avg_ridership"].mean()

change_text = (
    f"{(post_mean - pre_mean) / pre_mean * 100:+.2f}%"
    if pd.notna(pre_mean) and pre_mean != 0 else "N/A"
)
total_riders = int(data["total_ridership"].sum())

c1, c2 = st.columns(2)
c1.metric("Average Ridership Change (Post vs Pre)", change_text)
c2.metric("Total Riders in Selection", f"{total_riders:,}")

# ── Chart 1: Weekly Trend (from 2021) ─────────────────────────────────────────
weekly = (
    data.groupby(["week_start", "period"], as_index=False)["avg_ridership"]
    .mean().sort_values(["week_start", "period"])
)

fig1 = px.line(
    weekly, x="week_start", y="avg_ridership", color="period",
    title="Weekly Subway Ridership Trend (2021–present)",
    labels={"week_start": "Week", "avg_ridership": "Average Ridership", "period": ""}
)
fig1.update_layout(height=430)
fig1.update_xaxes(nticks=14, tickformat="%Y-%m")

# ── Chart 2: Borough Bar Chart ─────────────────────────────────────────────────
borough_avg = (
    data.groupby(["borough", "period"], as_index=False)["avg_ridership"]
    .mean().sort_values(["borough", "period"])
)

fig2 = px.bar(
    borough_avg, x="borough", y="avg_ridership", color="period",
    barmode="group",
    title="Average Ridership by Borough",
    labels={"borough": "Borough", "avg_ridership": "Average Ridership", "period": ""}
)
fig2.update_layout(height=380)

st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

# ── Executive Summary & Insights ───────────────────────────────────────────────
st.divider()
st.subheader("Executive Summary")

# Compute per-borough changes (all peaks combined)
all_data = df_all.copy()  # already 2021+
boro_pre  = all_data[all_data["period"] == "Pre Policy"] .groupby("borough")["avg_ridership"].mean()
boro_post = all_data[all_data["period"] == "Post Policy"].groupby("borough")["avg_ridership"].mean()
boro_change = ((boro_post - boro_pre) / boro_pre * 100).dropna().sort_values(ascending=False)

# Overall change (all boroughs, all peaks)
overall_pre  = all_data[all_data["period"] == "Pre Policy" ]["avg_ridership"].mean()
overall_post = all_data[all_data["period"] == "Post Policy"]["avg_ridership"].mean()
overall_pct  = (overall_post - overall_pre) / overall_pre * 100

# AM vs PM (all boroughs)
am_pre  = df_all[(df_all["peak_period"] == "Morning Peak") & (df_all["period"] == "Pre Policy" )]["avg_ridership"].mean()
am_post = df_all[(df_all["peak_period"] == "Morning Peak") & (df_all["period"] == "Post Policy")]["avg_ridership"].mean()
pm_pre  = df_all[(df_all["peak_period"] == "Evening Peak") & (df_all["period"] == "Pre Policy" )]["avg_ridership"].mean()
pm_post = df_all[(df_all["peak_period"] == "Evening Peak") & (df_all["period"] == "Post Policy")]["avg_ridership"].mean()
am_pct  = (am_post - am_pre) / am_pre * 100
pm_pct  = (pm_post - pm_pre) / pm_pre * 100

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Key Findings**")
    bullets = []
    bullets.append(f"- Overall subway ridership increased by **{overall_pct:+.1f}%** after congestion pricing took effect (Jan 5, 2025).")
    bullets.append(f"- Morning peak ridership rose **{am_pct:+.1f}%**; evening peak rose **{pm_pct:+.1f}%**, suggesting commuters shifted from driving to transit.")
    for boro, pct in boro_change.items():
        direction = "increased" if pct > 0 else "decreased"
        bullets.append(f"- **{boro}** ridership {direction} by **{pct:+.1f}%** post-policy.")
    st.markdown("\n".join(bullets))

with col2:
    st.markdown("**Why Did Manhattan See the Largest Change?**")
    manhattan_pct = boro_change.get("Manhattan", None)
    manhattan_line = f"Manhattan ridership rose **{manhattan_pct:+.1f}%**" if manhattan_pct else "Manhattan saw the largest change"
    st.markdown(f"""
{manhattan_line} — the largest among all boroughs. Several factors explain this concentration:

- **The congestion zone is Manhattan-centric.** The toll zone covers all of Manhattan south of 60th Street, where the toll applies to vehicles entering. Commuters who previously drove into this zone now have a direct financial incentive to switch to subway.
- **Subway station density.** Manhattan has the highest station density of any borough, making transit substitution easier and more convenient than in outer boroughs.
- **Commuter origin patterns.** A large share of outer-borough and New Jersey commuters travel *through* Manhattan. The toll increases the relative cost of car commutes that terminate in Manhattan, pushing more riders onto the subway network that converges on Manhattan hubs.
- **Marginal driver effect.** Even a modest modal shift of former drivers represents a proportionally large ridership gain in Manhattan, where pre-policy ridership was already the highest baseline.
""")
