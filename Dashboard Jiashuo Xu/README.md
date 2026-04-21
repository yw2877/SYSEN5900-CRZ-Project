# NYC Congestion Pricing — Academic Research Project

A data analysis and visualization project examining Year-1 outcomes of New York City's **Central Business District Tolling Program (CBDTP)** — the first cordon-based urban congestion pricing system in the United States, launched January 5, 2025.



## Python Analysis (`app.py`)

`app.py` generates 5 publication-ready charts and a console summary from Year-1 program data.

### Setup

```bash
pip install -r requirements.txt
python app.py
```

### Output

Charts are saved to the `output/` directory:

| File | Description |
|------|-------------|
| `fig1_traffic_volume.png` | Monthly CRZ vehicle entries vs. 2024 baseline |
| `fig2_revenue.png` | Monthly & cumulative MTA toll revenue vs. targets |
| `fig3_speed_safety.png` | Speed gains by road type & safety metric changes |
| `fig4_air_quality.png` | PM2.5 reduction: NYC vs. London & Stockholm |
| `fig5_revenue_breakdown.png` | Revenue share by vehicle type |

---

## Key Findings

| Metric | Result |
|--------|--------|
| Vehicles kept out of CRZ (Year 1) | ~27 million |
| Avg. daily traffic reduction | 73,000 vehicles (−11%) |
| CBD average speed improvement | +15% (8.2 → 9.7 mph) |
| Holland Tunnel rush-hour delays | −65% |
| PM2.5 reduction in CRZ (6 months) | −22% |
| MTA net revenue (2025 est.) | ~$550M (target: $500M) |
| Transit capital unlocked | $15 billion |
| Citywide traffic deaths | −19% (historic low) |

---

## Data Sources

- MTA Monthly Reports (2025)
- Governor Hochul Year-1 Report (January 2026)
- NBER Working Paper 33584 — Cook, Kreidieh, Vasserman et al. (2025)
- Cornell University / *npj Clean Air* — Gao et al. (December 2025)
- Regional Plan Association / Waze Report (June 2025)
- NYC Department of Health, NYC EDC, NYS Department of Labor

---

## Project Structure

```
nyc-congestion-pricing/
├── index.html          # Interactive website
├── app.py              # Python data analysis & chart generation
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── output/             # Generated charts (created by app.py)
```

---

*Academic Project · Spring 2026*
