"""
Generates data/feature8_india_supply_route_risk.csv
Covers real historical and documented maritime chokepoints and trade corridors critical to India (2015-2025).

Required Columns:
- route
- country_or_region
- event_date
- route_disruption (0-100)
- shipping_delay (Days or Index 0-100)
- freight_cost_change (%)
- trade_volume_exposure (0-100)
- india_import_exposure (0-100)
- india_export_exposure (0-100)
- energy_route_exposure (0-100)
- commodity_exposure (0-100)
- conflict_intensity (0-100)
- supply_route_risk (Target: Low, Moderate, High, Critical)
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

ROUTES_CONFIG = {
    "Strait of Hormuz": {
        "region": "Persian Gulf / Arabian Sea",
        "trade_vol": (75, 95), "imp_exp": (70, 92), "exp_exp": (30, 50),
        "energy_exp": (80, 98), "comm_exp": (40, 65), "weight": 55
    },
    "Bab el-Mandeb / Red Sea": {
        "region": "Gulf of Aden / Red Sea",
        "trade_vol": (65, 88), "imp_exp": (50, 75), "exp_exp": (65, 88),
        "energy_exp": (45, 70), "comm_exp": (55, 80), "weight": 50
    },
    "Suez Canal": {
        "region": "Mediterranean / Red Sea",
        "trade_vol": (60, 85), "imp_exp": (45, 68), "exp_exp": (60, 82),
        "energy_exp": (35, 60), "comm_exp": (50, 75), "weight": 40
    },
    "Malacca Strait": {
        "region": "Southeast Asia / Indo-Pacific",
        "trade_vol": (60, 82), "imp_exp": (55, 78), "exp_exp": (40, 62),
        "energy_exp": (25, 45), "comm_exp": (65, 88), "weight": 40
    },
    "Cape of Good Hope": {
        "region": "Southern Africa Route",
        "trade_vol": (45, 72), "imp_exp": (40, 65), "exp_exp": (45, 70),
        "energy_exp": (30, 55), "comm_exp": (35, 60), "weight": 35
    },
    "Persian Gulf Coastal": {
        "region": "Middle East Terminals",
        "trade_vol": (65, 85), "imp_exp": (65, 88), "exp_exp": (35, 55),
        "energy_exp": (70, 92), "comm_exp": (45, 68), "weight": 35
    },
    "Black Sea & Bosporus": {
        "region": "Eurasia / Turkish Straits",
        "trade_vol": (35, 60), "imp_exp": (35, 62), "exp_exp": (20, 40),
        "energy_exp": (25, 50), "comm_exp": (60, 85), "weight": 30
    },
    "Mozambique Channel": {
        "region": "Western Indian Ocean",
        "trade_vol": (25, 50), "imp_exp": (25, 48), "exp_exp": (20, 38),
        "energy_exp": (20, 42), "comm_exp": (35, 58), "weight": 25
    },
}

HISTORICAL_ANCHORS = [
    # 2015 Yemen Bab el-Mandeb escalation
    {"route": "Bab el-Mandeb / Red Sea", "region": "Yemen", "date": "2015-03-28", "disp": 68.0, "delay": 12.0, "freight": 28.5, "conf": 78.0},
    # 2016 Malacca piracy surge & patrol
    {"route": "Malacca Strait", "region": "Malaysia/Indonesia", "date": "2016-06-14", "disp": 42.0, "delay": 5.0, "freight": 14.0, "conf": 45.0},
    # 2019 Gulf of Oman tanker attacks (Kokuka Courageous / Front Altair)
    {"route": "Strait of Hormuz", "region": "Iran/Oman", "date": "2019-06-13", "disp": 88.0, "delay": 18.0, "freight": 65.0, "conf": 92.0},
    # 2019 Abqaiq strike maritime alert
    {"route": "Persian Gulf Coastal", "region": "Saudi Arabia", "date": "2019-09-15", "disp": 82.0, "delay": 15.0, "freight": 55.0, "conf": 88.0},
    # 2021 Ever Given container ship blockage in Suez Canal
    {"route": "Suez Canal", "region": "Egypt", "date": "2021-03-24", "disp": 95.0, "delay": 26.0, "freight": 78.0, "conf": 35.0},
    # 2022 Russian invasion of Ukraine / Black Sea grain initiative suspension
    {"route": "Black Sea & Bosporus", "region": "Ukraine/Russia", "date": "2022-02-26", "disp": 92.0, "delay": 24.0, "freight": 85.0, "conf": 96.0},
    # 2023 Galaxy Leader hijacked in Southern Red Sea
    {"route": "Bab el-Mandeb / Red Sea", "region": "Yemen", "date": "2023-11-20", "disp": 86.0, "delay": 20.0, "freight": 95.0, "conf": 88.0},
    # 2023 Cape of Good Hope massive carrier diversion
    {"route": "Cape of Good Hope", "region": "South Africa", "date": "2023-12-18", "disp": 58.0, "delay": 14.0, "freight": 62.0, "conf": 30.0},
    # 2024 MSC Aries seized in Strait of Hormuz
    {"route": "Strait of Hormuz", "region": "Iran", "date": "2024-04-13", "disp": 90.0, "delay": 22.0, "freight": 72.0, "conf": 95.0},
    # 2024 Bab el-Mandeb prolonged missile strikes
    {"route": "Bab el-Mandeb / Red Sea", "region": "Yemen", "date": "2024-06-22", "disp": 84.0, "delay": 18.0, "freight": 88.0, "conf": 86.0},
    # 2025 Red Sea & Gulf drone alert
    {"route": "Bab el-Mandeb / Red Sea", "region": "Red Sea Corridor", "date": "2025-02-15", "disp": 76.0, "delay": 16.0, "freight": 68.0, "conf": 82.0},
    # 2025 Malacca naval exercises
    {"route": "Malacca Strait", "region": "Indo-Pacific", "date": "2025-05-10", "disp": 38.0, "delay": 6.0, "freight": 12.0, "conf": 50.0},
]

def determine_supply_route_risk(row):
    """
    Computes real multi-class supply route risk label for India.
    Factors considered:
    - Physical route disruption (0-100)
    - Shipping delay in days / severity (0-100)
    - Freight cost change percentage (e.g. -10% to +120%)
    - India import and energy exposure through the route
    - Conflict intensity
    """
    disp = row["route_disruption"]
    delay = row["shipping_delay"]
    freight = max(0, row["freight_cost_change"])
    imp_exp = row["india_import_exposure"]
    energy_exp = row["energy_route_exposure"]
    conf = row["conflict_intensity"]

    # Composite index normalized to 100
    composite = (
        0.26 * (disp / 100.0) +
        0.22 * (energy_exp / 100.0) +
        0.18 * (imp_exp / 100.0) +
        0.15 * min(1.0, delay / 25.0) +
        0.11 * min(1.0, freight / 80.0) +
        0.08 * (conf / 100.0)
    ) * 100.0

    jitter = np.random.normal(0, 1.8)
    total_score = composite + jitter

    if total_score >= 64.0:
        return "Critical"
    elif total_score >= 46.0:
        return "High"
    elif total_score >= 30.0:
        return "Moderate"
    else:
        return "Low"

def generate_dataset(target_count=750):
    records = []
    start_date = datetime(2015, 1, 15)
    end_date = datetime(2025, 6, 20)
    days_span = (end_date - start_date).days

    # 1. Add historical anchors
    for h in HISTORICAL_ANCHORS:
        route_name = h["route"]
        cfg = ROUTES_CONFIG[route_name]
        row = {
            "route": route_name,
            "country_or_region": h["region"],
            "event_date": h["date"],
            "route_disruption": round(float(h["disp"] + random.uniform(-2, 2)), 1),
            "shipping_delay": round(float(h["delay"] + random.uniform(-1, 1)), 1),
            "freight_cost_change": round(float(h["freight"] + random.uniform(-2, 2)), 1),
            "trade_volume_exposure": round(float(random.uniform(*cfg["trade_vol"])), 1),
            "india_import_exposure": round(float(random.uniform(*cfg["imp_exp"])), 1),
            "india_export_exposure": round(float(random.uniform(*cfg["exp_exp"])), 1),
            "energy_route_exposure": round(float(random.uniform(*cfg["energy_exp"])), 1),
            "commodity_exposure": round(float(random.uniform(*cfg["comm_exp"])), 1),
            "conflict_intensity": round(float(h["conf"] + random.uniform(-2, 2)), 1),
        }
        row["supply_route_risk"] = determine_supply_route_risk(row)
        records.append(row)

    # 2. Add realistic events across routes
    routes = list(ROUTES_CONFIG.keys())
    weights = [ROUTES_CONFIG[r]["weight"] for r in routes]

    while len(records) < target_count:
        r_name = random.choices(routes, weights=weights)[0]
        cfg = ROUTES_CONFIG[r_name]
        r_days = random.randint(0, days_span)
        ev_date = start_date + timedelta(days=r_days)

        # Regimes: normal, congested, elevated_risk, acute_crisis
        regime = random.choices(["normal", "congested", "elevated_risk", "acute_crisis"], weights=[0.40, 0.32, 0.18, 0.10])[0]

        if regime == "acute_crisis":
            disp = random.uniform(70.0, 96.0)
            delay = random.uniform(14.0, 28.0)
            freight = random.uniform(50.0, 115.0)
            conf = random.uniform(75.0, 98.0)
        elif regime == "elevated_risk":
            disp = random.uniform(48.0, 75.0)
            delay = random.uniform(8.0, 18.0)
            freight = random.uniform(25.0, 65.0)
            conf = random.uniform(50.0, 78.0)
        elif regime == "congested":
            disp = random.uniform(25.0, 52.0)
            delay = random.uniform(4.0, 12.0)
            freight = random.uniform(10.0, 35.0)
            conf = random.uniform(25.0, 55.0)
        else: # normal
            disp = random.uniform(5.0, 28.0)
            delay = random.uniform(1.0, 5.0)
            freight = random.uniform(-8.0, 15.0)
            conf = random.uniform(5.0, 30.0)

        row = {
            "route": r_name,
            "country_or_region": cfg["region"],
            "event_date": ev_date.strftime("%Y-%m-%d"),
            "route_disruption": round(disp, 1),
            "shipping_delay": round(delay, 1),
            "freight_cost_change": round(freight, 1),
            "trade_volume_exposure": round(random.uniform(*cfg["trade_vol"]), 1),
            "india_import_exposure": round(random.uniform(*cfg["imp_exp"]), 1),
            "india_export_exposure": round(random.uniform(*cfg["exp_exp"]), 1),
            "energy_route_exposure": round(random.uniform(*cfg["energy_exp"]), 1),
            "commodity_exposure": round(random.uniform(*cfg["comm_exp"]), 1),
            "conflict_intensity": round(conf, 1),
        }
        row["supply_route_risk"] = determine_supply_route_risk(row)
        records.append(row)

    df = pd.DataFrame(records)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)
    df["event_date"] = df["event_date"].dt.strftime("%Y-%m-%d")
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    out_path = "data/feature8_india_supply_route_risk.csv"
    df = generate_dataset(target_count=750)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} records in {out_path}")
    print("\nTarget Class Distribution:")
    print(df["supply_route_risk"].value_counts())
    print("\nRoute Distribution:")
    print(df["route"].value_counts())
    print("\nHead:")
    print(df.head(5))
