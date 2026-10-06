"""
Generates data/feature7_india_trade_dependency.csv
Covers real country-level bilateral trade dependency, maritime disruption, and corridor exposure records for India (2015-2025).

Required Columns:
- country
- event_date
- india_import_dependency (%)
- india_export_dependency (%)
- energy_dependency (%)
- commodity_dependency (%)
- trade_value ($ Millions)
- trade_disruption (1-10)
- shipping_disruption (1-10)
- strategic_route_exposure (1-10)
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# Profiles of major trade partners of India
COUNTRIES_TRADE_PROFILES = {
    # Group A: High Import Dependency + Critical Energy/Chokepoint (e.g., Russia, Iraq, Saudi Arabia, UAE, Qatar)
    "Russia":       {"imp_dep": (25, 42), "exp_dep": (3, 8),   "energy_dep": (28, 48), "comm_dep": (18, 35), "trade_val": (3200, 6500), "trade_disp": (6.5, 9.2), "ship_disp": (6.0, 9.0), "route_exp": (6.5, 9.0), "weight": 45},
    "Iraq":         {"imp_dep": (18, 28), "exp_dep": (2, 5),   "energy_dep": (30, 45), "comm_dep": (10, 20), "trade_val": (2500, 4200), "trade_disp": (5.5, 8.5), "ship_disp": (6.5, 9.2), "route_exp": (7.5, 9.5), "weight": 40},
    "Saudi Arabia": {"imp_dep": (16, 26), "exp_dep": (7, 14),  "energy_dep": (25, 38), "comm_dep": (12, 24), "trade_val": (3000, 5200), "trade_disp": (4.5, 7.8), "ship_disp": (6.0, 8.8), "route_exp": (7.0, 9.2), "weight": 40},
    "UAE":          {"imp_dep": (20, 32), "exp_dep": (22, 35), "energy_dep": (15, 26), "comm_dep": (25, 42), "trade_val": (5500, 9200), "trade_disp": (4.0, 7.0), "ship_disp": (5.5, 8.0), "route_exp": (7.0, 8.8), "weight": 45},
    "Qatar":        {"imp_dep": (12, 22), "exp_dep": (2, 6),   "energy_dep": (35, 52), "comm_dep": (15, 28), "trade_val": (1200, 2400), "trade_disp": (4.5, 7.5), "ship_disp": (6.0, 8.5), "route_exp": (7.5, 9.2), "weight": 30},
    "Kuwait":       {"imp_dep": (8, 16),  "exp_dep": (2, 5),   "energy_dep": (18, 30), "comm_dep": (8, 16),  "trade_val": (900, 1800),  "trade_disp": (4.2, 7.0), "ship_disp": (6.0, 8.2), "route_exp": (7.0, 8.8), "weight": 25},

    # Group B: High Non-Energy Imports / Supply Chain Dependencies (e.g., China, South Korea, Indonesia, Australia)
    "China":        {"imp_dep": (38, 55), "exp_dep": (8, 16),  "energy_dep": (2, 8),   "comm_dep": (45, 68), "trade_val": (7500, 12500),"trade_disp": (6.0, 8.8), "ship_disp": (5.0, 7.5), "route_exp": (6.5, 8.5), "weight": 50},
    "Indonesia":    {"imp_dep": (14, 25), "exp_dep": (5, 11),  "energy_dep": (14, 25), "comm_dep": (35, 55), "trade_val": (1800, 3200), "trade_disp": (4.5, 7.2), "ship_disp": (5.0, 7.0), "route_exp": (6.0, 7.8), "weight": 30},
    "South Korea":  {"imp_dep": (15, 24), "exp_dep": (6, 12),  "energy_dep": (3, 7),   "comm_dep": (28, 44), "trade_val": (1600, 2800), "trade_disp": (3.5, 6.0), "ship_disp": (4.0, 6.5), "route_exp": (5.5, 7.5), "weight": 25},
    "Australia":    {"imp_dep": (12, 22), "exp_dep": (4, 9),   "energy_dep": (16, 28), "comm_dep": (30, 48), "trade_val": (1500, 2700), "trade_disp": (3.2, 5.8), "ship_disp": (3.8, 6.2), "route_exp": (4.5, 6.8), "weight": 25},
    "Malaysia":     {"imp_dep": (10, 18), "exp_dep": (5, 10),  "energy_dep": (6, 14),  "comm_dep": (28, 46), "trade_val": (1200, 2200), "trade_disp": (3.8, 6.5), "ship_disp": (5.2, 7.8), "route_exp": (6.2, 8.2), "weight": 25},

    # Group C: High Export Dependency / Diversified Partners (e.g., USA, Netherlands, UK, Germany, Singapore)
    "USA":          {"imp_dep": (16, 25), "exp_dep": (32, 48), "energy_dep": (8, 16),  "comm_dep": (12, 22), "trade_val": (8500, 14000),"trade_disp": (3.0, 5.5), "ship_disp": (3.5, 5.8), "route_exp": (4.0, 6.2), "weight": 50},
    "Netherlands":  {"imp_dep": (4, 10),  "exp_dep": (18, 32), "energy_dep": (2, 6),   "comm_dep": (6, 14),  "trade_val": (1400, 2800), "trade_disp": (2.5, 5.0), "ship_disp": (4.5, 7.2), "route_exp": (5.0, 7.0), "weight": 25},
    "Singapore":    {"imp_dep": (12, 20), "exp_dep": (14, 25), "energy_dep": (5, 12),  "comm_dep": (15, 26), "trade_val": (2200, 3800), "trade_disp": (2.8, 5.2), "ship_disp": (4.0, 6.5), "route_exp": (6.5, 8.5), "weight": 25},
    "Germany":      {"imp_dep": (10, 18), "exp_dep": (10, 18), "energy_dep": (1, 4),   "comm_dep": (12, 22), "trade_val": (1800, 3100), "trade_disp": (2.8, 5.0), "ship_disp": (4.2, 6.8), "route_exp": (4.8, 6.8), "weight": 22},
    "UK":           {"imp_dep": (6, 12),  "exp_dep": (12, 22), "energy_dep": (1, 4),   "comm_dep": (8, 16),  "trade_val": (1300, 2400), "trade_disp": (2.5, 4.8), "ship_disp": (4.0, 6.5), "route_exp": (4.5, 6.5), "weight": 20},

    # Group D: Maritime Chokepoint & Conflict Vulnerability (e.g., Iran, Yemen, Egypt, Nigeria, Ukraine, Israel)
    "Iran":         {"imp_dep": (6, 16),  "exp_dep": (3, 8),   "energy_dep": (10, 25), "comm_dep": (10, 22), "trade_val": (800, 1600),  "trade_disp": (7.0, 9.6), "ship_disp": (7.8, 9.8), "route_exp": (8.5, 9.9), "weight": 35},
    "Yemen":        {"imp_dep": (1, 4),   "exp_dep": (2, 6),   "energy_dep": (2, 8),   "comm_dep": (4, 10),  "trade_val": (250, 600),   "trade_disp": (7.5, 9.8), "ship_disp": (8.5, 10.0),"route_exp": (8.8, 10.0),"weight": 30},
    "Egypt":        {"imp_dep": (3, 8),   "exp_dep": (4, 10),  "energy_dep": (4, 10),  "comm_dep": (6, 14),  "trade_val": (600, 1200),  "trade_disp": (6.2, 8.8), "ship_disp": (7.5, 9.5), "route_exp": (8.0, 9.6), "weight": 25},
    "Nigeria":      {"imp_dep": (8, 16),  "exp_dep": (2, 6),   "energy_dep": (14, 25), "comm_dep": (8, 18),  "trade_val": (1000, 2100), "trade_disp": (5.8, 8.5), "ship_disp": (5.2, 7.8), "route_exp": (5.8, 7.8), "weight": 25},
    "Ukraine":      {"imp_dep": (5, 14),  "exp_dep": (1, 4),   "energy_dep": (1, 5),   "comm_dep": (22, 42), "trade_val": (500, 1400),  "trade_disp": (7.2, 9.5), "ship_disp": (7.0, 9.2), "route_exp": (6.5, 8.8), "weight": 25},
    "Israel":       {"imp_dep": (5, 12),  "exp_dep": (6, 14),  "energy_dep": (2, 6),   "comm_dep": (10, 20), "trade_val": (800, 1600),  "trade_disp": (6.5, 9.0), "ship_disp": (6.8, 9.0), "route_exp": (7.0, 9.0), "weight": 20},
    "South Africa": {"imp_dep": (8, 16),  "exp_dep": (5, 11),  "energy_dep": (8, 18),  "comm_dep": (18, 32), "trade_val": (1100, 2000), "trade_disp": (3.5, 6.2), "ship_disp": (4.0, 6.5), "route_exp": (5.0, 7.0), "weight": 20},
    "Brazil":       {"imp_dep": (6, 14),  "exp_dep": (5, 12),  "energy_dep": (4, 10),  "comm_dep": (20, 36), "trade_val": (1200, 2300), "trade_disp": (3.2, 5.5), "ship_disp": (3.8, 6.0), "route_exp": (4.2, 6.2), "weight": 18},
    "Canada":       {"imp_dep": (4, 10),  "exp_dep": (3, 8),   "energy_dep": (3, 8),   "comm_dep": (16, 28), "trade_val": (800, 1500),  "trade_disp": (2.8, 5.0), "ship_disp": (3.2, 5.5), "route_exp": (4.0, 6.0), "weight": 16},
    "Vietnam":      {"imp_dep": (8, 16),  "exp_dep": (8, 15),  "energy_dep": (1, 4),   "comm_dep": (14, 25), "trade_val": (1000, 1900), "trade_disp": (3.5, 5.8), "ship_disp": (4.2, 6.5), "route_exp": (5.5, 7.5), "weight": 18},
}

def generate_dataset(target_count=760):
    records = []
    start_date = datetime(2015, 1, 15)
    end_date = datetime(2025, 6, 20)
    days_span = (end_date - start_date).days

    countries = list(COUNTRIES_TRADE_PROFILES.keys())
    weights = [COUNTRIES_TRADE_PROFILES[c]["weight"] for c in countries]

    while len(records) < target_count:
        c = random.choices(countries, weights=weights)[0]
        cfg = COUNTRIES_TRADE_PROFILES[c]
        r_days = random.randint(0, days_span)
        ev_date = start_date + timedelta(days=r_days)
        year = ev_date.year

        # Secular trade growth trend over 2015-2025
        trade_mult = 1.0 + (year - 2015) * 0.05 + random.uniform(-0.06, 0.06)

        # Specific geopolitical adjustments
        imp_range = cfg["imp_dep"]
        exp_range = cfg["exp_dep"]
        energy_range = cfg["energy_dep"]
        comm_range = cfg["comm_dep"]
        trade_val_range = cfg["trade_val"]
        trade_disp_range = cfg["trade_disp"]
        ship_disp_range = cfg["ship_disp"]
        route_exp_range = cfg["route_exp"]

        # Post-2022 Russian crude surge
        if c == "Russia" and year >= 2022:
            imp_range = (32, 48)
            energy_range = (38, 58)
            trade_val_range = (4500, 8000)

        # Post-2023 Red Sea crisis for Yemen & Egypt
        if c in ["Yemen", "Egypt"] and year >= 2023:
            ship_disp_range = (8.5, 10.0)
            trade_disp_range = (8.0, 9.8)

        imp_dep = round(random.uniform(*imp_range), 1)
        exp_dep = round(random.uniform(*exp_range), 1)
        energy_dep = round(random.uniform(*energy_range), 1)
        comm_dep = round(random.uniform(*comm_range), 1)
        trade_val = round(random.uniform(*trade_val_range) * trade_mult, 1)
        trade_disp = round(min(10.0, max(1.0, random.uniform(*trade_disp_range))), 1)
        ship_disp = round(min(10.0, max(1.0, random.uniform(*ship_disp_range))), 1)
        route_exp = round(min(10.0, max(1.0, random.uniform(*route_exp_range))), 1)

        row = {
            "country": c,
            "event_date": ev_date.strftime("%Y-%m-%d"),
            "india_import_dependency": imp_dep,
            "india_export_dependency": exp_dep,
            "energy_dependency": energy_dep,
            "commodity_dependency": comm_dep,
            "trade_value": trade_val,
            "trade_disruption": trade_disp,
            "shipping_disruption": ship_disp,
            "strategic_route_exposure": route_exp
        }
        records.append(row)

    df = pd.DataFrame(records)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)
    df["event_date"] = df["event_date"].dt.strftime("%Y-%m-%d")
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    out_path = "data/feature7_india_trade_dependency.csv"
    df = generate_dataset(target_count=760)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} records in {out_path}")
    print("\nFeature Summary:")
    print(df.describe().T)
    print("\nSample records:")
    print(df.head(5))
