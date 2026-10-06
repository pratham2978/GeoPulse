"""
Generates data/feature5_india_energy_risk.csv
Covers real historical geopolitical, chokepoint, and bilateral energy supply events affecting India (2015-2025).
Columns:
- country
- event_date
- oil_import_dependency (80-89%)
- oil_price_change (%)
- energy_supply_disruption (0-100)
- shipping_disruption (0-100)
- india_energy_exposure (%)
- strategic_route_exposure (0-100)
- commodity_price_change (%)
- energy_risk: Low, Moderate, High, Critical
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# Major global energy suppliers & transit corridors with their typical bilateral exposure to India
# India imports ~85% of crude oil and ~50% of gas (LNG).
COUNTRIES_CONFIG = {
    # Tier 1: Major crude & gas suppliers to India (Critical to High exposure)
    "Iraq":         {"base_exposure": 20.5, "route": 85.0, "region": "Middle East", "weight": 45},
    "Russia":       {"base_exposure": 28.0, "route": 65.0, "region": "Eurasia",     "weight": 50},
    "Saudi Arabia": {"base_exposure": 16.5, "route": 88.0, "region": "Middle East", "weight": 45},
    "UAE":          {"base_exposure": 9.2,  "route": 82.0, "region": "Middle East", "weight": 35},
    "Qatar":        {"base_exposure": 12.0, "route": 85.0, "region": "Middle East", "weight": 35}, # Leading LNG supplier
    "Kuwait":       {"base_exposure": 5.5,  "route": 84.0, "region": "Middle East", "weight": 25},
    "USA":          {"base_exposure": 6.8,  "route": 40.0, "region": "Americas",    "weight": 30},
    "Nigeria":      {"base_exposure": 4.5,  "route": 55.0, "region": "Africa",      "weight": 25},
    "Oman":         {"base_exposure": 3.8,  "route": 75.0, "region": "Middle East", "weight": 20},
    "Angola":       {"base_exposure": 2.5,  "route": 48.0, "region": "Africa",      "weight": 18},
    
    # Tier 2: Strategic Chokepoint & Corridor Nations (Severe shipping/route leverage)
    "Iran":         {"base_exposure": 2.0,  "route": 94.0, "region": "Strait of Hormuz", "weight": 35},
    "Yemen":        {"base_exposure": 0.5,  "route": 89.0, "region": "Bab el-Mandeb",    "weight": 35},
    "Egypt":        {"base_exposure": 1.2,  "route": 72.0, "region": "Suez Canal",       "weight": 28},
    "Malaysia":     {"base_exposure": 2.0,  "route": 65.0, "region": "Malacca Strait",   "weight": 20},
    "Indonesia":    {"base_exposure": 3.0,  "route": 60.0, "region": "Coal & LNG Route", "weight": 22},
    
    # Tier 3: Secondary Energy & Strategic Nations
    "Algeria":      {"base_exposure": 1.1,  "route": 45.0, "region": "Africa",      "weight": 15},
    "Australia":    {"base_exposure": 4.5,  "route": 50.0, "region": "Asia-Pacific","weight": 22}, # LNG & Coking coal
    "Brazil":       {"base_exposure": 1.5,  "route": 35.0, "region": "Americas",    "weight": 15},
    "Canada":       {"base_exposure": 1.2,  "route": 30.0, "region": "Americas",    "weight": 15},
    "Kazakhstan":   {"base_exposure": 1.8,  "route": 55.0, "region": "Central Asia","weight": 18},
    "Azerbaijan":   {"base_exposure": 1.0,  "route": 50.0, "region": "Caucasus",    "weight": 14},
    "Venezuela":    {"base_exposure": 2.5,  "route": 42.0, "region": "Americas",    "weight": 18},
    "Norway":       {"base_exposure": 0.8,  "route": 35.0, "region": "Europe",      "weight": 12},
    "Libya":        {"base_exposure": 0.9,  "route": 48.0, "region": "North Africa","weight": 15},
    "Singapore":    {"base_exposure": 1.5,  "route": 62.0, "region": "Bunkering Hub","weight": 16},
    "South Africa": {"base_exposure": 2.2,  "route": 45.0, "region": "Coal Corridor","weight": 16},
    "Colombia":     {"base_exposure": 0.7,  "route": 32.0, "region": "Americas",    "weight": 12},
    "Mexico":       {"base_exposure": 1.6,  "route": 38.0, "region": "Americas",    "weight": 14},
    "Mozambique":   {"base_exposure": 1.5,  "route": 40.0, "region": "East Africa", "weight": 14}, # LNG concession
    "Turkey":       {"base_exposure": 0.6,  "route": 55.0, "region": "Bosporus",    "weight": 14},
}

# Real historical anchors (dates, countries, nature of shock)
HISTORICAL_ANCHORS = [
    # 2015-2018: OPEC cuts, Houthis in Bab el-Mandeb, Iran Nuclear Sanctions re-imposition
    {"country": "Yemen", "date": "2015-03-26", "oil_chg": 8.5, "disp": 68, "ship": 74, "risk": "High"},
    {"country": "Iran",  "date": "2015-07-14", "oil_chg": -4.2, "disp": 22, "ship": 25, "risk": "Low"},
    {"country": "Iraq",  "date": "2016-04-17", "oil_chg": 5.8, "disp": 45, "ship": 40, "risk": "Moderate"},
    {"country": "Nigeria", "date": "2016-05-12", "oil_chg": 7.2, "disp": 62, "ship": 35, "risk": "Moderate"},
    {"country": "Saudi Arabia", "date": "2016-11-30", "oil_chg": 9.4, "disp": 48, "ship": 42, "risk": "Moderate"},
    {"country": "Qatar", "date": "2017-06-05", "oil_chg": 3.2, "disp": 52, "ship": 68, "risk": "Moderate"},
    {"country": "Iran",  "date": "2018-05-08", "oil_chg": 11.5, "disp": 72, "ship": 75, "risk": "High"},
    {"country": "Yemen", "date": "2018-07-25", "oil_chg": 6.8, "disp": 58, "ship": 80, "risk": "High"},
    # 2019: Hormuz tanker attacks & Abqaiq strike
    {"country": "Iran",  "date": "2019-06-13", "oil_chg": 14.2, "disp": 78, "ship": 92, "risk": "Critical"},
    {"country": "Saudi Arabia", "date": "2019-09-14", "oil_chg": 19.8, "disp": 92, "ship": 75, "risk": "Critical"},
    # 2020: Covid oil crash & recovery
    {"country": "Russia", "date": "2020-03-09", "oil_chg": -24.5, "disp": 30, "ship": 20, "risk": "Low"},
    {"country": "Saudi Arabia", "date": "2020-04-12", "oil_chg": 12.0, "disp": 40, "ship": 30, "risk": "Moderate"},
    # 2021: Suez canal blockage & energy crunch
    {"country": "Egypt", "date": "2021-03-23", "oil_chg": 7.5, "disp": 45, "ship": 88, "risk": "High"},
    {"country": "Australia", "date": "2021-10-15", "oil_chg": 8.1, "disp": 50, "ship": 40, "risk": "Moderate"},
    # 2022: Ukraine War & Russian Sanctions, European gas crisis
    {"country": "Russia", "date": "2022-02-24", "oil_chg": 28.5, "disp": 88, "ship": 82, "risk": "Critical"},
    {"country": "USA",    "date": "2022-03-08", "oil_chg": 14.0, "disp": 35, "ship": 25, "risk": "Moderate"},
    {"country": "Qatar",  "date": "2022-06-15", "oil_chg": 12.5, "disp": 60, "ship": 55, "risk": "High"},
    {"country": "Iraq",   "date": "2022-08-29", "oil_chg": 6.8, "disp": 65, "ship": 58, "risk": "High"},
    # 2023: Gaza conflict & Red Sea shipping crisis
    {"country": "Yemen",  "date": "2023-11-19", "oil_chg": 8.9, "disp": 70, "ship": 94, "risk": "Critical"},
    {"country": "Iran",   "date": "2023-12-23", "oil_chg": 7.4, "disp": 64, "ship": 85, "risk": "High"},
    # 2024: Direct Iran-Israel strikes, Bab el-Mandeb prolonged rerouting
    {"country": "Iran",   "date": "2024-04-13", "oil_chg": 12.8, "disp": 82, "ship": 90, "risk": "Critical"},
    {"country": "Russia", "date": "2024-06-20", "oil_chg": 4.5, "disp": 58, "ship": 60, "risk": "Moderate"},
    {"country": "Yemen",  "date": "2024-07-20", "oil_chg": 9.2, "disp": 75, "ship": 92, "risk": "Critical"},
    {"country": "Iran",   "date": "2024-10-01", "oil_chg": 16.4, "disp": 85, "ship": 95, "risk": "Critical"},
    # 2025: Strait of Hormuz tensions & maritime drone alerts
    {"country": "Oman",   "date": "2025-01-18", "oil_chg": 6.2, "disp": 52, "ship": 70, "risk": "High"},
    {"country": "Iraq",   "date": "2025-04-05", "oil_chg": 10.4, "disp": 68, "ship": 76, "risk": "High"},
]

def determine_energy_risk(features):
    """
    Computes real multi-class energy supply risk level for India:
    Features:
    - oil_import_dependency: ~80-88%
    - oil_price_change: %
    - energy_supply_disruption: 0-100
    - shipping_disruption: 0-100
    - india_energy_exposure: %
    - strategic_route_exposure: 0-100
    - commodity_price_change: %
    """
    # Weighted composite index of national energy stress
    # Higher weights on chokepoint transit (route), direct bilateral exposure, physical supply, and crude price spikes
    norm_dep = (features["oil_import_dependency"] - 78) / 12.0 # ~0 to 1
    norm_oil_price = max(0, features["oil_price_change"] + 15) / 50.0 # ~0 to 1
    norm_supply = features["energy_supply_disruption"] / 100.0
    norm_shipping = features["shipping_disruption"] / 100.0
    norm_exposure = min(features["india_energy_exposure"] / 30.0, 1.2)
    norm_route = features["strategic_route_exposure"] / 100.0
    norm_comm = max(0, features["commodity_price_change"] + 10) / 40.0

    score = (
        0.24 * norm_exposure +
        0.22 * norm_shipping +
        0.18 * norm_route +
        0.15 * norm_supply +
        0.11 * norm_oil_price +
        0.06 * norm_dep +
        0.04 * norm_comm
    ) * 100.0

    # Add slight realistic stochastic variability
    jitter = np.random.normal(0, 1.8)
    composite = score + jitter

    if composite >= 60.0:
        return "Critical"
    elif composite >= 44.0:
        return "High"
    elif composite >= 28.0:
        return "Moderate"
    else:
        return "Low"

def generate_records(target_count=720):
    records = []

    # First add historical anchor points with realistic variations
    start_date = datetime(2015, 1, 15)
    end_date = datetime(2025, 6, 20)

    for anchor in HISTORICAL_ANCHORS:
        c = anchor["country"]
        cfg = COUNTRIES_CONFIG[c]
        year = int(anchor["date"][:4])
        # India's oil import dependency grew from ~80% in 2015 to ~88% in 2024/2025
        base_dep = 80.0 + (year - 2015) * 0.75 + random.uniform(-0.5, 0.5)

        base_exp = cfg["base_exposure"]
        if c == "Russia":
            # Russia's share grew from ~2% pre-2022 to ~35% in 2023-2024
            base_exp = 2.5 if year < 2022 else 32.0 + random.uniform(-3, 3)

        row = {
            "country": c,
            "event_date": anchor["date"],
            "oil_import_dependency": round(base_dep, 2),
            "oil_price_change": round(anchor["oil_chg"] + random.uniform(-1.0, 1.0), 2),
            "energy_supply_disruption": round(float(anchor["disp"] + random.uniform(-3, 3)), 2),
            "shipping_disruption": round(float(anchor["ship"] + random.uniform(-3, 3)), 2),
            "india_energy_exposure": round(float(base_exp + random.uniform(-0.5, 0.5)), 2),
            "strategic_route_exposure": round(float(cfg["route"] + random.uniform(-2, 2)), 2),
            "commodity_price_change": round(float(anchor["oil_chg"] * 0.65 + random.uniform(-1.5, 1.5)), 2),
            "energy_risk": anchor["risk"]
        }
        records.append(row)

    # Now generate synthetic/historical events spanning all countries and time periods
    days_span = (end_date - start_date).days
    countries = list(COUNTRIES_CONFIG.keys())
    weights = [COUNTRIES_CONFIG[c]["weight"] for c in countries]

    while len(records) < target_count:
        c = random.choices(countries, weights=weights)[0]
        cfg = COUNTRIES_CONFIG[c]
        r_day = random.randint(0, days_span)
        ev_date = start_date + timedelta(days=r_day)
        year = ev_date.year

        # Real oil dependency trajectory
        dep = 79.5 + (year - 2015) * 0.8 + random.uniform(-1.0, 1.0)
        dep = min(max(dep, 78.0), 89.5)

        # Bilateral exposure
        exp = cfg["base_exposure"]
        if c == "Russia":
            exp = random.uniform(1.0, 3.5) if year < 2022 else random.uniform(22.0, 38.0)
        elif c in ["Iraq", "Saudi Arabia"] and year >= 2022:
            exp *= random.uniform(0.85, 0.98) # Slight drop due to Russian intake
        else:
            exp *= random.uniform(0.88, 1.12)
        exp = max(0.2, exp)

        # Route exposure
        route = cfg["route"] + random.uniform(-6, 6)
        route = min(max(route, 15.0), 98.0)

        # Event severity regime
        regime = random.choices(["mild", "moderate", "severe", "chokepoint_shock"], weights=[0.45, 0.32, 0.15, 0.08])[0]

        if regime == "chokepoint_shock":
            ship_disp = random.uniform(70.0, 98.0)
            sup_disp = random.uniform(50.0, 90.0)
            oil_chg = random.uniform(10.0, 35.0)
            comm_chg = oil_chg * random.uniform(0.5, 0.85) + random.uniform(-2, 3)
        elif regime == "severe":
            ship_disp = random.uniform(55.0, 85.0)
            sup_disp = random.uniform(55.0, 85.0)
            oil_chg = random.uniform(7.0, 22.0)
            comm_chg = oil_chg * random.uniform(0.4, 0.75) + random.uniform(-1, 2)
        elif regime == "moderate":
            ship_disp = random.uniform(30.0, 60.0)
            sup_disp = random.uniform(25.0, 55.0)
            oil_chg = random.uniform(2.0, 12.0)
            comm_chg = oil_chg * random.uniform(0.3, 0.7) + random.uniform(-1.5, 1.5)
        else: # mild
            ship_disp = random.uniform(8.0, 35.0)
            sup_disp = random.uniform(5.0, 30.0)
            oil_chg = random.uniform(-12.0, 6.0)
            comm_chg = random.uniform(-6.0, 5.0)

        row = {
            "country": c,
            "event_date": ev_date.strftime("%Y-%m-%d"),
            "oil_import_dependency": round(dep, 2),
            "oil_price_change": round(oil_chg, 2),
            "energy_supply_disruption": round(sup_disp, 2),
            "shipping_disruption": round(ship_disp, 2),
            "india_energy_exposure": round(exp, 2),
            "strategic_route_exposure": round(route, 2),
            "commodity_price_change": round(comm_chg, 2),
        }
        row["energy_risk"] = determine_energy_risk(row)
        records.append(row)

    df = pd.DataFrame(records)
    # Sort chronologically
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)
    df["event_date"] = df["event_date"].dt.strftime("%Y-%m-%d")
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    out_path = "data/feature5_india_energy_risk.csv"
    df = generate_records(target_count=750)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} records in {out_path}")
    print("\nClass distribution:")
    print(df["energy_risk"].value_counts())
    print("\nSample records:")
    print(df.head(5))
