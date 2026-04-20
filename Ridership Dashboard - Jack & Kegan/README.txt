NYC Subway Ridership Dashboard
================================

Overview
--------
This Streamlit dashboard analyzes NYC subway ridership before and after the
congestion pricing policy took effect on January 5, 2025. It compares peak-hour
ridership across boroughs (Manhattan, Brooklyn, Queens, Bronx) using MTA data.


Requirements
------------
- Python 3.8 or higher
- Install dependencies:

    pip install -r requirements.txt

Dependencies: pandas, requests, plotly, streamlit


Files
-----
- app.py                     Main dashboard application
- requirements.txt           Python dependencies
- weekly_aggregated_mta.csv  Pre-fetched local data (used by default)
- Dashboard.ipynb            Data exploration and preprocessing notebook


How to Run
----------
    python -m streamlit run app.py


Data Sources
------------
Two modes are available via the sidebar toggle "Fetch live data from API":

- OFF (default): Loads data from the local file weekly_aggregated_mta.csv.
  Fast, no internet required.

- ON: Pulls live data directly from the NY Open Data API (data.ny.gov).
  Takes ~30 seconds. Requires internet access.
  Datasets used:
    - Pre-policy:  https://data.ny.gov/resource/wujg-7c2s.json
    - Post-policy: https://data.ny.gov/resource/5wq4-mkjj.json


Dashboard Features
------------------
- Sidebar filters: Peak Hour (Morning 7-10am / Evening 4-7pm), Borough
- Metrics: Average ridership change (post vs pre policy), total riders
- Chart 1: Weekly ridership trend line chart (2021-present)
- Chart 2: Average ridership by borough (grouped bar chart)
- Executive Summary: Key findings and analysis by borough and peak period


Notes
-----
- 2020 data is excluded due to COVID-19 anomalies.
- Morning peak = 7:00-10:00; Evening peak = 16:00-19:00.
