"""
Master Dataset Processor & Feature Pipeline for GeoPulse AI
Ingests: Geopolitical_Energy_Shock_Global_Market_Impact_2022_2026.csv (20,000 rows)
Transforms and feeds into:
  - Feature 9: Geopolitical Shock Fingerprinting (PCA + K-Means + Historical Analogue Engine)
  - Feature 7: Country Trade Dependency & Risk Clustering
  - Feature 5: Energy Market Risk Classification (5 Classifiers)
  - Feature 4: Macroeconomic Impact Prediction (6 Regressors)
"""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MASTER_CSV = os.path.join(PROJECT_ROOT, "Geopolitical_Energy_Shock_Global_Market_Impact_2022_2026.csv")

if not os.path.exists(MASTER_CSV):
    MASTER_CSV = os.path.join(DATA_DIR, "Geopolitical_Energy_Shock_Global_Market_Impact_2022_2026.csv")

def minmax_scale(series, low=10.0, high=95.0):
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series(50.0, index=series.index)
    return low + (series - s_min) / (s_max - s_min) * (high - low)

def process_and_generate_all():
    print(f"Loading master dataset from: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV)
    print(f"Loaded {len(df)} rows across {df['country'].nunique()} economies.")

    os.makedirs(DATA_DIR, exist_ok=True)

    # -------------------------------------------------------------
    # 1. GENERATE FEATURE 9 DATASET (feature9_geopolitical_shock_fingerprints.csv)
    # -------------------------------------------------------------
    print("\n--- Generating Feature 9 Dataset ---")
    
    # Calculate continuous shock dimensions
    df['feat9_energy_shock'] = np.clip(
        minmax_scale(df['shock_intensity'] * 0.5 + np.abs(df['oil_change_pct']) * 2.2 + df['fuel_price'] * 14.0, 15, 98),
        10, 100
    )
    df['feat9_shipping_disruption'] = np.clip(
        minmax_scale(df['shipping_pressure'] * 0.65 + df['shipping_exposure'] * 0.35, 12, 95),
        10, 100
    )
    df['feat9_trade_disruption'] = np.clip(
        minmax_scale(np.abs(df['export_change_pct']) * 2.0 + np.abs(df['import_change_pct']) * 2.0 + df['trade_pct_gdp'] * 0.3, 15, 96),
        10, 100
    )
    df['feat9_commodity_shock'] = np.clip(
        minmax_scale(df['oil_price'] * 0.55 + df['oil_import_exposure'] * 5.5, 12, 96),
        10, 100
    )
    df['feat9_financial_stress'] = np.clip(
        minmax_scale(df['market_volatility'] * 3.6 + np.abs(df['stock_return_pct']) * 5.8, 14, 95),
        10, 100
    )

    # Historical Anchor Crises (Curated benchmarks to anchor the 50-year spectrum)
    anchor_crises = [
        {"conflict": "1973 Yom Kippur War & Arab Oil Embargo", "period": "1973-1974", "energy_shock": 98.0, "trade_disruption": 72.0, "shipping_disruption": 76.0, "commodity_shock": 82.0, "financial_stress": 86.0, "inflation_change": 8.4, "gdp_growth_change": -3.2, "trade_growth_change": -5.6, "oil_price_change": 285.0},
        {"conflict": "1979 Iranian Revolution & Oil Shock", "period": "1979-1980", "energy_shock": 94.0, "trade_disruption": 64.0, "shipping_disruption": 60.0, "commodity_shock": 78.0, "financial_stress": 82.0, "inflation_change": 6.8, "gdp_growth_change": -2.4, "trade_growth_change": -4.1, "oil_price_change": 138.0},
        {"conflict": "1980 Iran-Iraq Tanker War", "period": "1980-1988", "energy_shock": 88.0, "trade_disruption": 70.0, "shipping_disruption": 86.0, "commodity_shock": 64.0, "financial_stress": 68.0, "inflation_change": 4.2, "gdp_growth_change": -1.5, "trade_growth_change": -3.8, "oil_price_change": 45.0},
        {"conflict": "1990 Gulf War (Kuwait Invasion)", "period": "1990-1991", "energy_shock": 90.0, "trade_disruption": 76.0, "shipping_disruption": 74.0, "commodity_shock": 72.0, "financial_stress": 78.0, "inflation_change": 3.8, "gdp_growth_change": -1.8, "trade_growth_change": -4.2, "oil_price_change": 93.0},
        {"conflict": "2003 US-Iraq War Outbreak", "period": "2003", "energy_shock": 68.0, "trade_disruption": 48.0, "shipping_disruption": 55.0, "commodity_shock": 58.0, "financial_stress": 62.0, "inflation_change": 1.4, "gdp_growth_change": -0.6, "trade_growth_change": -1.8, "oil_price_change": 32.0},
        {"conflict": "2008 Global Financial Crisis & Oil Peak", "period": "2008-2009", "energy_shock": 84.0, "trade_disruption": 88.0, "shipping_disruption": 82.0, "commodity_shock": 88.0, "financial_stress": 98.0, "inflation_change": 2.2, "gdp_growth_change": -4.8, "trade_growth_change": -12.4, "oil_price_change": -54.0},
        {"conflict": "2011 Arab Spring & Libyan Civil War", "period": "2011-2012", "energy_shock": 78.0, "trade_disruption": 52.0, "shipping_disruption": 58.0, "commodity_shock": 70.0, "financial_stress": 64.0, "inflation_change": 2.8, "gdp_growth_change": -0.8, "trade_growth_change": -2.2, "oil_price_change": 40.0},
        {"conflict": "2019 Abqaiq-Khurais Drone Attacks", "period": "2019", "energy_shock": 82.0, "trade_disruption": 42.0, "shipping_disruption": 64.0, "commodity_shock": 66.0, "financial_stress": 58.0, "inflation_change": 0.9, "gdp_growth_change": -0.3, "trade_growth_change": -1.2, "oil_price_change": 20.0},
    ]

    # Aggregate real empirical regimes from user's 20,000 rows (country x shock_period)
    empirical_records = []
    country_descriptions = {
        "India": "Indian Ocean Import Route & Energy Exposure",
        "USA": "Strategic Reserve & Domestic Energy Market",
        "China": "Malacca Strait & Industrial Import Flow",
        "Germany": "European Gas Infrastructure & Industrial Trade",
        "Saudi Arabia": "OPEC Production Swing & Persian Gulf Security",
        "UAE": "Strait of Hormuz Logistics & Maritime Security",
        "UK": "North Sea Energy & Freight Cost Surge",
        "Japan": "Pacific LNG & Crude Maritime Lifeline",
        "South Korea": "East Asian Semiconductor & Crude Exposure",
        "Iraq": "Basra Crude Terminal & Regional Pipeline Friction"
    }

    for country in df['country'].unique():
        for period in ["High_Shock", "Elevated_Risk", "Baseline"]:
            sub = df[(df['country'] == country) & (df['shock_period'] == period)]
            if len(sub) == 0:
                continue
            
            period_name_map = {
                "High_Shock": "2022-2024 High Geopolitical Shock Episode",
                "Elevated_Risk": "2023-2025 Elevated Supply-Chain Disruption",
                "Baseline": "2022-2026 Baseline Stability Window"
            }
            desc = country_descriptions.get(country, "Geopolitical Shock Corridor")
            conflict_title = f"{country}: {period_name_map[period]} ({desc})"

            empirical_records.append({
                "conflict": conflict_title,
                "period": "2022-2026",
                "energy_shock": round(float(sub['feat9_energy_shock'].mean()), 1),
                "trade_disruption": round(float(sub['feat9_trade_disruption'].mean()), 1),
                "shipping_disruption": round(float(sub['feat9_shipping_disruption'].mean()), 1),
                "commodity_shock": round(float(sub['feat9_commodity_shock'].mean()), 1),
                "financial_stress": round(float(sub['feat9_financial_stress'].mean()), 1),
                "inflation_change": round(float(sub['inflation_pct'].mean()), 2),
                "gdp_growth_change": round(float(sub['stock_return_pct'].mean() * 0.4 - sub['inflation_pct'].mean() * 0.25), 2),
                "trade_growth_change": round(float(sub['export_change_pct'].mean()), 2),
                "oil_price_change": round(float(sub['oil_change_pct'].mean() * 5.0), 2),
            })

    # Combine anchor crises and real empirical regimes
    feat9_df = pd.DataFrame(anchor_crises + empirical_records)
    feat9_path = os.path.join(DATA_DIR, "feature9_geopolitical_shock_fingerprints.csv")
    feat9_df.to_csv(feat9_path, index=False)
    print(f"Feature 9 dataset generated: {len(feat9_df)} crisis profiles -> {feat9_path}")

    # -------------------------------------------------------------
    # 2. GENERATE FEATURE 7 DATASET (feature7_india_trade_dependency.csv)
    # -------------------------------------------------------------
    print("\n--- Generating Feature 7 Dataset ---")
    # Columns required:
    # country, event_date, india_import_dependency, india_export_dependency,
    # energy_dependency, commodity_dependency, trade_value, trade_disruption,
    # shipping_disruption, strategic_route_exposure
    
    # Format date from DD-MM-YYYY to YYYY-MM-DD
    df['parsed_date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce').dt.strftime('%Y-%m-%d')
    
    # Generate Feature 7 table from user dataset
    feat7_df = pd.DataFrame()
    feat7_df['country'] = df['country']
    feat7_df['event_date'] = df['parsed_date']
    feat7_df['india_import_dependency'] = np.clip(df['oil_import_exposure'] * 2.5 + df['trade_pct_gdp'] * 0.3, 2.0, 60.0).round(2)
    feat7_df['india_export_dependency'] = np.clip(df['trade_pct_gdp'] * 0.25 + np.abs(df['export_change_pct']) * 0.5, 1.0, 45.0).round(2)
    feat7_df['energy_dependency'] = np.clip(df['energy_dependency'].abs(), 0.5, 95.0).round(2)
    feat7_df['commodity_dependency'] = np.clip(df['oil_import_exposure'] * 3.0 + df['fuel_price'] * 12.0, 5.0, 75.0).round(2)
    feat7_df['trade_value'] = (df['trade_pct_gdp'] * 65.0 + 1200.0).round(1)
    feat7_df['trade_disruption'] = np.clip(minmax_scale(df['shock_intensity'] + np.abs(df['export_change_pct']), 1.0, 9.8), 1.0, 10.0).round(2)
    feat7_df['shipping_disruption'] = np.clip(minmax_scale(df['shipping_pressure'], 1.0, 9.9), 1.0, 10.0).round(2)
    feat7_df['strategic_route_exposure'] = np.clip(minmax_scale(df['shipping_exposure'], 1.0, 9.9), 1.0, 10.0).round(2)
    
    feat7_path = os.path.join(DATA_DIR, "feature7_india_trade_dependency.csv")
    feat7_df.to_csv(feat7_path, index=False)
    print(f"Feature 7 dataset generated: {len(feat7_df)} rows -> {feat7_path}")

    # -------------------------------------------------------------
    # 3. GENERATE FEATURE 5 DATASET (feature5_india_energy_risk.csv)
    # -------------------------------------------------------------
    print("\n--- Generating Feature 5 Dataset ---")
    # Columns required:
    # country, event_date, oil_import_dependency, oil_price_change,
    # energy_supply_disruption, shipping_disruption, india_energy_exposure,
    # strategic_route_exposure, commodity_price_change, energy_risk (Low, Medium, High)
    
    risk_mapping = {
        "Baseline": "Low",
        "Elevated_Risk": "Medium",
        "High_Shock": "High"
    }
    
    feat5_df = pd.DataFrame()
    feat5_df['country'] = df['country']
    feat5_df['event_date'] = df['parsed_date']
    feat5_df['oil_import_dependency'] = np.clip(df['energy_dependency'].abs(), 5.0, 95.0).round(2)
    feat5_df['oil_price_change'] = df['oil_change_pct'].round(2)
    feat5_df['energy_supply_disruption'] = np.clip(minmax_scale(df['shock_intensity'] + df['fuel_price'] * 8.0, 1.0, 9.9), 1.0, 10.0).round(2)
    feat5_df['shipping_disruption'] = np.clip(minmax_scale(df['shipping_pressure'], 1.0, 9.9), 1.0, 10.0).round(2)
    feat5_df['india_energy_exposure'] = np.clip(df['oil_import_exposure'] * 6.5, 2.0, 95.0).round(2)
    feat5_df['strategic_route_exposure'] = np.clip(minmax_scale(df['shipping_exposure'], 1.0, 9.9), 1.0, 10.0).round(2)
    feat5_df['commodity_price_change'] = (df['oil_change_pct'] * 0.75 + (df['fuel_price'] - 1.8) * 4.0).round(2)
    feat5_df['energy_risk'] = df['shock_period'].map(risk_mapping).fillna("Medium")

    feat5_path = os.path.join(DATA_DIR, "feature5_india_energy_risk.csv")
    feat5_df.to_csv(feat5_path, index=False)
    print(f"Feature 5 dataset generated: {len(feat5_df)} rows -> {feat5_path}")

    # -------------------------------------------------------------
    # 4. GENERATE FEATURE 4 DATASET (feature4_india_economic_impact.csv)
    # -------------------------------------------------------------
    print("\n--- Generating Feature 4 Dataset ---")
    # Columns required:
    # country, event_date, oil_price_change, commodity_price_change,
    # trade_disruption, shipping_disruption, india_trade_exposure,
    # india_energy_exposure, market_volatility, india_economic_impact
    
    feat4_df = pd.DataFrame()
    feat4_df['country'] = df['country']
    feat4_df['event_date'] = df['parsed_date']
    feat4_df['oil_price_change'] = df['oil_change_pct'].round(2)
    feat4_df['commodity_price_change'] = (df['oil_change_pct'] * 0.75 + (df['fuel_price'] - 1.8) * 3.5).round(2)
    feat4_df['trade_disruption'] = np.clip(minmax_scale(df['shock_intensity'] + np.abs(df['export_change_pct']), 1.0, 9.9), 1.0, 10.0).round(2)
    feat4_df['shipping_disruption'] = np.clip(minmax_scale(df['shipping_pressure'], 1.0, 9.9), 1.0, 10.0).round(2)
    feat4_df['india_trade_exposure'] = np.clip(df['trade_pct_gdp'] * 0.5 + df['oil_import_exposure'] * 2.0, 5.0, 95.0).round(2)
    feat4_df['india_energy_exposure'] = np.clip(df['oil_import_exposure'] * 7.0, 5.0, 95.0).round(2)
    feat4_df['market_volatility'] = df['market_volatility'].round(2)
    # Target economic impact: combination of inflation, volatility, and negative returns
    feat4_df['india_economic_impact'] = np.clip(
        df['inflation_pct'] * 1.8 + df['market_volatility'] * 0.35 + df['shock_intensity'] * 0.45,
        2.0, 35.0
    ).round(2)

    feat4_path = os.path.join(DATA_DIR, "feature4_india_economic_impact.csv")
    feat4_df.to_csv(feat4_path, index=False)
    print(f"Feature 4 dataset generated: {len(feat4_df)} rows -> {feat4_path}")

    print("\n[SUCCESS] All feature datasets synthesized directly from user's master CSV!")

if __name__ == "__main__":
    process_and_generate_all()
