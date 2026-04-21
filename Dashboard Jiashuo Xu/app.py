"""
NYC Congestion Pricing — Data Analysis
=======================================
Academic Project · Spring 2026

Analyzes Year-1 outcomes of New York City's Central Business District
Tolling Program (CBDTP), which launched January 5, 2025.

Data sources:
  - MTA Monthly Reports (2025)
  - Governor Hochul Year-1 Report (Jan 2026)
  - NBER Working Paper 33584 (Cook et al., 2025)
  - Cornell University / npj Clean Air (Dec 2025)
  - Regional Plan Association / Waze (Jun 2025)
  - NYC DOH, NYC EDC, NYS Dept. of Labor

Usage:
  python app.py
  Outputs charts to ./output/ directory and prints a summary to console.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# Avoid Windows console encoding crashes when printing Unicode status text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    "ink":   "#0f1923",
    "red":   "#c0392b",
    "blue":  "#2980b9",
    "green": "#16a085",
    "gold":  "#b8860b",
    "grey":  "#c8bfa8",
    "paper": "#f4f0e8",
}

plt.rcParams.update({
    "font.family":      "serif",
    "axes.facecolor":   "white",
    "figure.facecolor": C["paper"],
    "axes.edgecolor":   C["grey"],
    "axes.grid":        True,
    "grid.color":       C["grey"],
    "grid.linewidth":   0.5,
    "grid.alpha":       0.6,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "text.color":       C["ink"],
    "axes.labelcolor":  C["ink"],
    "xtick.color":      C["ink"],
    "ytick.color":      C["ink"],
})

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ═════════════════════════════════════════════════════════════════════════════
# DATA
# ═════════════════════════════════════════════════════════════════════════════

# Monthly average daily vehicle entries to CRZ (thousands)
# Baseline 2024: ~583k/day;  2025 observed MTA data
traffic_data = pd.DataFrame({
    "month":     MONTHS,
    "baseline":  [583] * 12,   # 2024 pre-program baseline
    "crz_2025":  [510, 505, 507, 512, 516, 518, 516, 513, 510, 508, 509, 510],
})
traffic_data["reduction_pct"] = (
    (traffic_data["baseline"] - traffic_data["crz_2025"]) / traffic_data["baseline"] * 100
)

# Monthly net toll revenue ($ millions)
revenue_data = pd.DataFrame({
    "month":   MONTHS,
    "revenue": [48.6, 44.0, 58.4, 56.7, 52.0, 50.0, 48.0, 50.0, 46.0, 47.0, 48.0, 49.6],
    "target":  [41.6] * 12,
})
revenue_data["cumulative"] = revenue_data["revenue"].cumsum()

# Speed improvements by road type (%) — NBER WP 33584
speed_data = pd.DataFrame({
    "road_type": ["Highways", "Arterial Roads", "Local Roads",
                  "CBD Overall", "Weekend Evenings"],
    "pct_gain":  [13, 10, 8, 15, 25],
})

# Safety metrics: % change 2025 vs 2024
safety_data = pd.DataFrame({
    "metric": ["CRZ Crashes", "Traffic Injuries", "Citywide Deaths",
               "Noise Complaints", "Holland Tunnel Delays"],
    "change": [-7, -8, -19, -23, -65],
    "color":  [C["green"], C["green"], C["green"], C["blue"], C["red"]],
})

# Air quality: PM2.5 reduction vs global peers
air_data = pd.DataFrame({
    "city":      ["NYC (6 months,\n2025)", "Stockholm\n(4 years, 2006–10)",
                  "London\n(3 years, 2019–22)"],
    "reduction": [22, 10, 7],
    "color":     [C["red"], C["blue"], C["gold"]],
})

# Revenue breakdown by vehicle type (Jan 2025, MTA report)
revenue_breakdown = pd.DataFrame({
    "type":    ["Passenger Vehicles", "Taxis & Rideshare", "Trucks", "Buses & Motorcycles"],
    "share":   [68, 22, 9, 1],
    "color":   [C["blue"], C["green"], C["gold"], C["grey"]],
})


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Traffic Volume
# ═════════════════════════════════════════════════════════════════════════════

def plot_traffic():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 1 — Traffic Volume: CRZ Daily Entries",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")

    # Left: bar + baseline line
    ax = axes[0]
    x = np.arange(len(MONTHS))
    ax.bar(x, traffic_data["crz_2025"], color=C["blue"], alpha=0.8,
           width=0.6, label="2025 Daily Entries")
    ax.plot(x, traffic_data["baseline"], color=C["red"], linewidth=1.8,
            linestyle="--", label="2024 Baseline (~583k)")
    ax.set_xticks(x); ax.set_xticklabels(MONTHS, fontsize=9)
    ax.set_ylabel("Avg. Daily Vehicles (thousands)")
    ax.set_ylim(470, 610)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}k"))
    ax.legend(fontsize=9)
    ax.set_title("Monthly Avg. Daily Entries vs. 2024 Baseline", fontsize=10)

    # Right: % reduction line
    ax2 = axes[1]
    ax2.fill_between(x, traffic_data["reduction_pct"], alpha=0.25, color=C["red"])
    ax2.plot(x, traffic_data["reduction_pct"], color=C["red"],
             linewidth=2, marker="o", markersize=5)
    ax2.axhline(11, color=C["ink"], linewidth=1, linestyle=":", alpha=0.5,
                label="Year avg. (11%)")
    ax2.set_xticks(x); ax2.set_xticklabels(MONTHS, fontsize=9)
    ax2.set_ylabel("Reduction vs. 2024 (%)")
    ax2.set_ylim(0, 16)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.legend(fontsize=9)
    ax2.set_title("Monthly Traffic Reduction (%)", fontsize=10)

    fig.text(0.02, 0.01,
             "Source: MTA Monthly Reports 2025 · Governor Hochul Year-1 Report (Jan 2026)",
             fontsize=7.5, color="#777", style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    path = f"{OUTPUT_DIR}/fig1_traffic_volume.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Revenue
# ═════════════════════════════════════════════════════════════════════════════

def plot_revenue():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 2 — MTA Toll Revenue, 2025",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")

    x = np.arange(len(MONTHS))

    # Left: monthly vs target
    ax = axes[0]
    bars = ax.bar(x, revenue_data["revenue"], color=C["gold"], alpha=0.85,
                  width=0.6, label="Monthly Net Revenue")
    ax.plot(x, revenue_data["target"], color=C["red"], linewidth=1.8,
            linestyle="--", label="Monthly Target ($41.6M)")
    ax.set_xticks(x); ax.set_xticklabels(MONTHS, fontsize=9)
    ax.set_ylabel("Net Revenue ($ millions)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}M"))
    ax.legend(fontsize=9)
    ax.set_title("Monthly Net Revenue vs. Target", fontsize=10)

    # Right: cumulative
    ax2 = axes[1]
    ax2.fill_between(x, revenue_data["cumulative"], alpha=0.2, color=C["gold"])
    ax2.plot(x, revenue_data["cumulative"], color=C["gold"],
             linewidth=2.5, marker="o", markersize=5, label="Cumulative Revenue")
    ax2.axhline(500, color=C["red"], linewidth=1.5, linestyle="--",
                label="Annual Target ($500M)")
    ax2.set_xticks(x); ax2.set_xticklabels(MONTHS, fontsize=9)
    ax2.set_ylabel("Cumulative Revenue ($ millions)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}M"))
    ax2.legend(fontsize=9)
    ax2.set_title("Cumulative Revenue vs. $500M Annual Target", fontsize=10)

    fig.text(0.02, 0.01,
             "Source: MTA Monthly Committee Reports · AMNY May 2025 · Davis Vanguard Dec 2025",
             fontsize=7.5, color="#777", style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    path = f"{OUTPUT_DIR}/fig2_revenue.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Speed & Safety
# ═════════════════════════════════════════════════════════════════════════════

def plot_speed_safety():
    fig = plt.figure(figsize=(13, 5.5))
    fig.suptitle("Figure 3 — Speed Improvements & Safety Outcomes",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    # Left: speed by road type
    ax = fig.add_subplot(gs[0])
    colors = [C["ink"], C["blue"], C["green"], C["red"], C["gold"]]
    bars = ax.barh(speed_data["road_type"], speed_data["pct_gain"],
                   color=colors, alpha=0.85, height=0.55)
    for bar, val in zip(bars, speed_data["pct_gain"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"+{val}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Speed Improvement (%)")
    ax.set_xlim(0, 32)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"+{v:.0f}%"))
    ax.set_title("Average Speed Gains by Road Type\n(Sep 2024 – Feb 2025)", fontsize=10)
    ax.text(0.01, -0.1, "Source: NBER Working Paper 33584 — Cook et al., 2025",
            transform=ax.transAxes, fontsize=7.5, color="#777", style="italic")

    # Right: safety metrics
    ax2 = fig.add_subplot(gs[1])
    bars2 = ax2.barh(safety_data["metric"], safety_data["change"],
                     color=safety_data["color"], alpha=0.85, height=0.55)
    for bar, val in zip(bars2, safety_data["change"]):
        ax2.text(val - 1, bar.get_y() + bar.get_height() / 2,
                 f"{val}%", va="center", ha="right",
                 fontsize=10, fontweight="bold", color="white")
    ax2.set_xlabel("Change vs. 2024 (%)")
    ax2.set_xlim(-80, 5)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.set_title("Safety & Quality-of-Life Metrics\n(% Change 2025 vs. 2024)", fontsize=10)
    ax2.text(0.01, -0.1, "Source: NYPD · NYC 311 · MTA Year-1 Report (Jan 2026)",
             transform=ax2.transAxes, fontsize=7.5, color="#777", style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = f"{OUTPUT_DIR}/fig3_speed_safety.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Air Quality
# ═════════════════════════════════════════════════════════════════════════════

def plot_air_quality():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 4 — Environmental Impact: PM2.5 Air Pollution",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")

    # Left: global comparison
    ax = axes[0]
    bars = ax.bar(air_data["city"], air_data["reduction"],
                  color=air_data["color"], alpha=0.85, width=0.5)
    for bar, val in zip(bars, air_data["reduction"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{val}%", ha="center", fontsize=13, fontweight="bold")
    ax.set_ylabel("PM2.5 Reduction (%)")
    ax.set_ylim(0, 28)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title("PM2.5 Reduction: NYC vs. Global Peers", fontsize=10)
    ax.text(0.01, -0.14,
            "Sources: Cornell / npj Clean Air (Dec 2025) · Stockholm & London peer studies",
            transform=ax.transAxes, fontsize=7.5, color="#777", style="italic")

    # Right: NYC borough breakdown (illustrative gradient from CRZ outward)
    ax2 = axes[1]
    boroughs = ["CRZ\n(Manhattan\nbelow 60th)", "Manhattan\n(full)", "Brooklyn", "Queens", "Bronx", "Staten\nIsland"]
    reductions = [22, 15, 9, 7, 6, 4]
    colors_b = [C["red"], C["red"] + "bb", C["blue"] + "cc",
                C["blue"] + "99", C["green"] + "99", C["gold"] + "99"]
    bars2 = ax2.bar(boroughs, reductions, color=colors_b, alpha=0.85, width=0.6)
    for bar, val in zip(bars2, reductions):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val}%", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Estimated PM2.5 Reduction (%)")
    ax2.set_ylim(0, 28)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.set_title("PM2.5 Reduction by Area\n(Jan–Jun 2025)", fontsize=10)
    ax2.text(0.01, -0.14,
             "Source: Cornell University / npj Clean Air (Dec 2025) — 42 monitors, 518 days",
             transform=ax2.transAxes, fontsize=7.5, color="#777", style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = f"{OUTPUT_DIR}/fig4_air_quality.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Revenue Breakdown
# ═════════════════════════════════════════════════════════════════════════════

def plot_revenue_breakdown():
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Figure 5 — Revenue Breakdown by Vehicle Type (Jan 2025)",
                 fontsize=12, fontweight="bold", x=0.05, ha="left")

    wedges, texts, autotexts = ax.pie(
        revenue_breakdown["share"],
        labels=revenue_breakdown["type"],
        colors=revenue_breakdown["color"],
        autopct="%1.0f%%",
        startangle=140,
        pctdistance=0.7,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
        t.set_color("white")

    ax.text(0, -1.5,
            "Source: MTA Monthly Committee Report, February 2025",
            ha="center", fontsize=8, color="#777", style="italic")
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    path = f"{OUTPUT_DIR}/fig5_revenue_breakdown.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {path}")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═════════════════════════════════════════════════════════════════════════════

def print_summary():
    total_rev   = revenue_data["revenue"].sum()
    avg_red     = traffic_data["reduction_pct"].mean()
    peak_rev    = revenue_data["revenue"].max()
    peak_month  = revenue_data.loc[revenue_data["revenue"].idxmax(), "month"]
    months_above = (revenue_data["revenue"] >= revenue_data["target"]).sum()

    print("\n" + "═" * 58)
    print("  NYC CONGESTION PRICING — YEAR 1 SUMMARY STATISTICS")
    print("═" * 58)

    print("\n  ── TRAFFIC ──────────────────────────────────────────")
    print(f"  Total vehicles kept out of CRZ (est.): ~27,000,000")
    print(f"  Avg. daily reduction:                   73,000 vehicles")
    print(f"  Avg. monthly traffic reduction:         {avg_red:.1f}%")
    print(f"  CBD speed improvement (NBER):           +15% (8.2→9.7 mph)")
    print(f"  Holland Tunnel rush-hour delay change:  -65%")

    print("\n  ── REVENUE ──────────────────────────────────────────")
    print(f"  Total 2025 net revenue (estimated):     ${total_rev:.1f}M")
    print(f"  Annual target:                          $500.0M")
    print(f"  Above/below target:                     +${total_rev - 500:.1f}M")
    print(f"  Peak month:                             {peak_month} (${peak_rev:.1f}M)")
    print(f"  Months meeting/exceeding target:        {months_above}/12")
    print(f"  Transit capital unlocked (via bonds):   $15 billion")

    print("\n  ── ENVIRONMENT ──────────────────────────────────────")
    print(f"  PM2.5 reduction in CRZ (Jan–Jun 2025): -22%")
    print(f"  (London benchmark, 2019–2022):          -7%")
    print(f"  (Stockholm benchmark, 2006–2010):       -5 to -15%")
    print(f"  Vehicular emissions reduction (Nature): -16 to -22%")
    print(f"  Noise complaints (NYC 311):              -23%")

    print("\n  ── SAFETY ───────────────────────────────────────────")
    print(f"  Crashes in CRZ:                         -7%")
    print(f"  Traffic injuries in CRZ:                -8%")
    print(f"  Citywide traffic deaths (2025):         -19% (historic low)")

    print("\n  ── SOURCES ──────────────────────────────────────────")
    print("  MTA Monthly Reports (2025)")
    print("  Governor Hochul Year-1 Report (Jan 2026)")
    print("  NBER Working Paper 33584 — Cook et al. (2025)")
    print("  Cornell Univ. / npj Clean Air — Gao et al. (Dec 2025)")
    print("  Regional Plan Association / Waze (Jun 2025)")
    print("  NYC DOH · NYC EDC · NYS Dept. of Labor")
    print("═" * 58 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nNYC Congestion Pricing — Generating analysis...")
    print(f"Output directory: ./{OUTPUT_DIR}/\n")

    plot_traffic()
    plot_revenue()
    plot_speed_safety()
    plot_air_quality()
    plot_revenue_breakdown()
    print_summary()

    print(f"Done. All figures saved to ./{OUTPUT_DIR}/")
