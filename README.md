# Manhattan CRZ Dashboard

This repository now keeps the merged dashboard and the preserved source dashboards separate.

## What lives where

- `app.py`
  A small launcher that imports `app` from `merged_app.py`, so the existing run command still works.
- `merged_app.py`
  The main merged Shiny dashboard with four tabs:
  - Bus Speed
  - Traffic Volume
  - Subway Ridership
  - CRZ Overview
- `Dashboard Yixuan Wang&Sijin Li Dashboard/`
  A preserved snapshot of the original two-tab local project, including its source files and original local CSV inputs.
- `Dashboard Jiashuo Xu/`
  The imported Jiashuo Xu dashboard source folder, including the HTML overview and the CSV files used by the fourth tab.
- `weekly_aggregated_mta.csv`
  The subway ridership dataset used by the merged dashboard.

## Data layout

The merged dashboard reads:

- bus speed data from `Dashboard Yixuan Wang&Sijin Li Dashboard/`
- traffic volume data from `Dashboard Yixuan Wang&Sijin Li Dashboard/`
- CRZ overview data from `Dashboard Jiashuo Xu/`
- subway ridership data from the repository root

This keeps the preserved source folders intact while letting the merged dashboard read each team's data from a single location.

## Run the merged dashboard

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
shiny run --reload --port 8000 app.py
```

Open `http://127.0.0.1:8000`.

## Run the preserved original version

Change into `Dashboard Yixuan Wang&Sijin Li Dashboard/` and run its own `app.py` there if you need the pre-merge snapshot.
