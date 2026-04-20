# NYC Bus Speed Dashboard (Local CSV, Shiny for Python)

This dashboard uses local CSV files only and now includes:
- 4 KPI summary cards with before MPH, after MPH, absolute change, and percent change
- Filterable views for day type, period, borough, and route text search
- An Executive Summary module with auto-filled conclusion bullets
- A So What module that translates MPH changes into trip-time changes
- 4 charts:
  - All Local bus: before vs after congestion pricing
  - CBD Local bus: before vs after congestion pricing
  - CBD: Express vs Local speed
  - Local bus: Within CBD vs Without CBD speed

## Required files
- `MTA_Bus_Speeds__Beginning_2015_20260309.csv`
- `MTA_Central_Business_District_Bus_Speeds__Beginning_2023_20260309.csv`

Place both files in the same folder as `app.py`.

## Run
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
shiny run --reload --port 8000 app.py
```

Open `http://127.0.0.1:8000` in your browser.

## Notes
- Default congestion-pricing start date is `2025-01-05`.
- Default before/after window is `3 months` on each side.
- The all-bus file supports `Peak` / `Off-Peak`.
- The CBD file supports `Peak` / `Overnight`.
- Direction is not available in these source files, so there is no direction filter.
