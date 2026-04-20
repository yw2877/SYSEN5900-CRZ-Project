# Source Layout

This workspace now separates the original local project from the merged version.

- `merged_app.py`
  The merged three-tab dashboard that combines:
  - the original bus speed tab
  - the original traffic volume tab
  - the imported subway ridership tab
- `app.py`
  A minimal launcher that keeps the existing `shiny run app.py` command working by importing `app` from `merged_app.py`.
- `our_version/`
  A preserved copy of the original two-tab local project before the subway-ridership merge.

Inside `our_version/`:

- `app.py`
  The pre-merge dashboard implementation.
- `requirements.txt`
  The original dependency list.
- `README.md`
  The original local project README copy.
- `MTA_Bus_Speeds__Beginning_2015_20260309.csv`
- `MTA_Central_Business_District_Bus_Speeds__Beginning_2023_20260309.csv`
- `Daily_Traffic_on_MTA_Bridges_&_Tunnels_20260406.csv`

This layout keeps the merged deliverable as a standalone Python file while preserving the original local work in its own folder.

