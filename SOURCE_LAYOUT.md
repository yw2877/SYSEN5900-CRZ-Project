# Source Layout

This workspace now separates the source dashboards from the merged four-tab version.

- `merged_app.py`
  The merged four-tab dashboard that combines:
  - the original bus speed tab
  - the original traffic volume tab
  - the imported subway ridership tab
  - the imported CRZ overview tab
- `app.py`
  A minimal launcher that keeps the existing `shiny run app.py` command working by importing `app` from `merged_app.py`.
- `Dashboard Yixuan Wang&Sijin Li Dashboard/`
  A preserved copy of the original two-tab local project before the subway-ridership merge.
  This folder now also holds the original local bus-speed and traffic CSV inputs, so those large files are no longer duplicated at the repository root.
- `Dashboard Jiashuo Xu/`
  The imported Jiashuo Xu source folder, including the HTML overview and the CSV files now used by the CRZ Overview tab.

Inside `Dashboard Yixuan Wang&Sijin Li Dashboard/`:

- `app.py`
  The pre-merge dashboard implementation.
- `requirements.txt`
  The original dependency list.
- `README.md`
  The original local project README copy.
- `MTA_Bus_Speeds__Beginning_2015_20260309.csv`
- `MTA_Central_Business_District_Bus_Speeds__Beginning_2023_20260309.csv`
- `Daily_Traffic_on_MTA_Bridges_&_Tunnels_20260406.csv`

Inside `Dashboard Jiashuo Xu/`:

- `app.py`
  The pulled Python chart generator from Jiashuo Xu's repository.
- `price index.html.html`
  The pulled HTML overview page source.
- `nyc_congestion_pricing_data.csv`
- `traffic_volume_2025.csv`
- `toll_revenue_2025.csv`

This layout keeps the merged deliverable as a standalone Python file while preserving each source dashboard in its own folder.
