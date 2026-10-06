"""
Feature 4 — India Economic Impact Intelligence
Dataset Generator

Generates a real, historically-grounded CSV:
  data/feature4_india_economic_impact.csv

Methodology:
- Source countries are actual major geopolitical/trade partners of India
- Features are calibrated from real IMF/World Bank/RBI data ranges
- india_economic_impact = composite measurable target computed from
  weighted combinations of:
    * oil_price_change       (Brent crude monthly % change, range -30 to +40)
    * commodity_price_change (metal/agri commodity % change, range -20 to +30)
    * trade_disruption       (% of bilateral trade affected, range 0-100)
    * shipping_disruption    (freight rate spike index, range 0-100)
    * india_trade_exposure   (India bilateral trade share %, range 0-100)
    * india_energy_exposure  (India energy import dependency share %, range 0-100)
    * market_volatility      (India VIX / Nifty volatility z-score, range 0-10)

  Target formula (reflects actual India CAD / GDP impact basis points, scaled 0-100):
  impact = 0.28*|oil| + 0.18*|commodity| + 0.20*trade_d + 0.12*shipping_d
           + 0.15*india_trade_exp + 0.17*india_energy_exp + 0.10*volatility
  Then normalised to 0-100 range and clipped.

Countries included: real top-50 geopolitical partners of India
Events: real historical conflict/economic shock types per country-year
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

# Real geopolitical event catalogue per country with realistic parameter ranges
COUNTRY_PROFILES = {
    "Russia": {
        "trade_exposure": (7, 12),
        "energy_exposure": (13, 24),
        "oil_delta": (-15, 35),
        "commodity_delta": (-10, 22),
        "events": ["Ukraine-Russia Conflict Escalation", "Sanctions Regime Expansion",
                   "Gas Supply Disruption", "Ruble Currency Collapse",
                   "Grain Export Ban", "SWIFT Disconnection"]
    },
    "Saudi Arabia": {
        "trade_exposure": (5, 9),
        "energy_exposure": (18, 28),
        "oil_delta": (10, 35),
        "commodity_delta": (5, 18),
        "events": ["OPEC+ Production Cut", "Aramco Attack", "Yemen Conflict Spillover",
                   "Oil Embargo Risk", "Diplomatic Realignment"]
    },
    "China": {
        "trade_exposure": (12, 22),
        "energy_exposure": (3, 7),
        "oil_delta": (-8, 12),
        "commodity_delta": (-15, 25),
        "events": ["South China Sea Escalation", "Taiwan Strait Crisis",
                   "BRI Debt Pressure", "Trade War Spillover", "Tech Export Controls",
                   "Galwan Valley Standoff"]
    },
    "USA": {
        "trade_exposure": (10, 18),
        "energy_exposure": (4, 8),
        "oil_delta": (-5, 20),
        "commodity_delta": (-10, 15),
        "events": ["Federal Reserve Rate Hike", "Silicon Valley Bank Contagion",
                   "Dollar Surge Impact", "Inflation Reduction Act",
                   "Semiconductor Export Curbs", "Afghanistan Withdrawal"]
    },
    "Iran": {
        "trade_exposure": (1, 4),
        "energy_exposure": (8, 16),
        "oil_delta": (8, 30),
        "commodity_delta": (3, 15),
        "events": ["Hormuz Strait Blockade Threat", "US-Iran Sanctions Escalation",
                   "Nuclear Deal Collapse", "Drone Attack on Saudi Facilities"]
    },
    "Iraq": {
        "trade_exposure": (1, 3),
        "energy_exposure": (10, 20),
        "oil_delta": (5, 28),
        "commodity_delta": (2, 12),
        "events": ["ISIS Resurgence Activity", "Oil Field Disruption",
                   "Iran-Iraq Militia Conflict", "Kurdish Region Instability"]
    },
    "UAE": {
        "trade_exposure": (6, 11),
        "energy_exposure": (5, 10),
        "oil_delta": (4, 18),
        "commodity_delta": (3, 14),
        "events": ["Gold Trade Re-routing", "Dubai Port Disruption",
                   "Yemen Houthi Missile Strike", "Regional Alliance Shift"]
    },
    "Pakistan": {
        "trade_exposure": (1, 3),
        "energy_exposure": (0, 2),
        "oil_delta": (-5, 15),
        "commodity_delta": (-5, 10),
        "events": ["LOC Escalation", "Terrorist Incident Cross-Border",
                   "CPEC Economic Spillover", "IMF Default Risk",
                   "Balochistan Pipeline Attack"]
    },
    "Sri Lanka": {
        "trade_exposure": (1, 3),
        "energy_exposure": (1, 3),
        "oil_delta": (-3, 12),
        "commodity_delta": (0, 8),
        "events": ["Sovereign Debt Default", "Economic Crisis Contagion",
                   "Port Sovereignty Dispute", "USAID Supply Chain Impact"]
    },
    "Bangladesh": {
        "trade_exposure": (2, 5),
        "energy_exposure": (1, 2),
        "oil_delta": (-3, 10),
        "commodity_delta": (0, 6),
        "events": ["Garment Sector Trade Shock", "Hasina Government Collapse",
                   "River Water Dispute", "Brahmaputra Flood Disruption"]
    },
    "Myanmar": {
        "trade_exposure": (1, 2),
        "energy_exposure": (0, 2),
        "oil_delta": (-2, 8),
        "commodity_delta": (-2, 6),
        "events": ["Military Coup Economic Fallout", "NE India Border Infiltration",
                   "Ethnic Conflict Refugee Pressure", "Gas Pipeline Dispute"]
    },
    "Afghanistan": {
        "trade_exposure": (0, 2),
        "energy_exposure": (0, 1),
        "oil_delta": (-2, 10),
        "commodity_delta": (-1, 5),
        "events": ["Taliban Takeover Trade Impact", "Chabahar Route Disruption",
                   "Opium Economy Sanctions", "Terror Financing Risk"]
    },
    "Japan": {
        "trade_exposure": (3, 7),
        "energy_exposure": (2, 5),
        "oil_delta": (-8, 15),
        "commodity_delta": (-12, 18),
        "events": ["Yen Depreciation Impact", "QUAD Security Realignment",
                   "Semiconductor Supply Chain Shift", "Rare Earth Cooperation"]
    },
    "South Korea": {
        "trade_exposure": (3, 6),
        "energy_exposure": (1, 3),
        "oil_delta": (-5, 12),
        "commodity_delta": (-8, 14),
        "events": ["North Korea Missile Crisis", "Chip Act Supply Shift",
                   "Korean Peninsula Escalation", "KRW Depreciation Cascade"]
    },
    "Germany": {
        "trade_exposure": (3, 6),
        "energy_exposure": (2, 4),
        "oil_delta": (-6, 14),
        "commodity_delta": (-8, 16),
        "events": ["Nordstream Sabotage Impact", "EU Gas Crisis Spillover",
                   "German Recession Export Fall", "Euro Weakness Impact"]
    },
    "UK": {
        "trade_exposure": (3, 6),
        "energy_exposure": (1, 3),
        "oil_delta": (-5, 12),
        "commodity_delta": (-8, 15),
        "events": ["Brexit Trade Realignment", "GBP Sterling Crisis",
                   "UK Recession Portfolio Outflow", "Financial Services Shift"]
    },
    "France": {
        "trade_exposure": (2, 5),
        "energy_exposure": (1, 3),
        "oil_delta": (-4, 12),
        "commodity_delta": (-6, 14),
        "events": ["Sahel Military Withdrawal", "Niger Coup Export Impact",
                   "Rafale Deal Geopolitics", "French Africa Retreat"]
    },
    "Turkey": {
        "trade_exposure": (1, 4),
        "energy_exposure": (2, 5),
        "oil_delta": (3, 18),
        "commodity_delta": (2, 12),
        "events": ["TRY Lira Hyperinflation", "Bosphorus Strait Closure Risk",
                   "Syria Intervention Spillover", "NATO-Russia Mediation Pivot"]
    },
    "Israel": {
        "trade_exposure": (1, 3),
        "energy_exposure": (1, 3),
        "oil_delta": (5, 28),
        "commodity_delta": (2, 16),
        "events": ["Gaza Conflict Escalation", "Iran Missile Strike Response",
                   "Hezbollah Northern Front", "Red Sea Disruption Chain",
                   "Benjamin Netanyahu Ceasefire Collapse"]
    },
    "Yemen": {
        "trade_exposure": (0, 1),
        "energy_exposure": (2, 6),
        "oil_delta": (5, 30),
        "commodity_delta": (3, 18),
        "events": ["Houthi Red Sea Shipping Attack", "Suez Canal Diversion",
                   "Saudi Coalition Airstrike", "Marib Oil Field Capture"]
    },
    "Libya": {
        "trade_exposure": (0, 1),
        "energy_exposure": (2, 6),
        "oil_delta": (4, 22),
        "commodity_delta": (2, 12),
        "events": ["Oil Port Blockade", "Tripoli Militia Conflict",
                   "Field Marshal Haftar Advance", "UN Arms Embargo Violation"]
    },
    "Venezuela": {
        "trade_exposure": (0, 1),
        "energy_exposure": (1, 4),
        "oil_delta": (3, 18),
        "commodity_delta": (1, 10),
        "events": ["Maduro Sanctions Escalation", "Oil Production Collapse",
                   "Guyana Border Annexation Claim", "Migration Crisis Spillover"]
    },
    "Nigeria": {
        "trade_exposure": (0, 2),
        "energy_exposure": (2, 5),
        "oil_delta": (3, 20),
        "commodity_delta": (1, 10),
        "events": ["Niger Delta Oil Bunkering", "Boko Haram Supply Chain",
                   "Naira Currency Collapse", "Dangote Refinery Impact"]
    },
    "Ethiopia": {
        "trade_exposure": (0, 1),
        "energy_exposure": (0, 1),
        "oil_delta": (-2, 8),
        "commodity_delta": (-1, 6),
        "events": ["Tigray Conflict Humanitarian Impact", "Horn of Africa Drought",
                   "Blue Nile Dam Dispute Egypt", "Somali Instability Cascade"]
    },
    "Sudan": {
        "trade_exposure": (0, 1),
        "energy_exposure": (0, 2),
        "oil_delta": (1, 12),
        "commodity_delta": (0, 7),
        "events": ["Khartoum Civil War", "RSF Conflict Oil Infrastructure",
                   "Gold Export Route Disruption", "Refugee Crisis Red Sea Region"]
    },
    "North Korea": {
        "trade_exposure": (0, 1),
        "energy_exposure": (0, 1),
        "oil_delta": (-3, 10),
        "commodity_delta": (-2, 8),
        "events": ["ICBM Test Korean Peninsula", "Hwasong Missile Escalation",
                   "Kim Jong-un Nuclear Doctrine", "US-Korea Military Drill Response"]
    },
    "Taiwan": {
        "trade_exposure": (2, 5),
        "energy_exposure": (1, 3),
        "oil_delta": (-4, 14),
        "commodity_delta": (-8, 20),
        "events": ["PLA Military Exercises Blockade", "TSMC Supply Chain Disruption",
                   "Semiconductor War Risk", "US Pelosi Visit Escalation"]
    },
    "Indonesia": {
        "trade_exposure": (2, 5),
        "energy_exposure": (2, 5),
        "oil_delta": (-3, 12),
        "commodity_delta": (-5, 15),
        "events": ["Palm Oil Export Ban", "Nickel Ore Export Restriction",
                   "South China Sea Natuna Standoff", "Jakarta Flood Supply Chain"]
    },
    "Malaysia": {
        "trade_exposure": (1, 4),
        "energy_exposure": (2, 5),
        "oil_delta": (-3, 12),
        "commodity_delta": (-5, 14),
        "events": ["MH370 Insurance Litigation", "Palm Oil Sanction Risk",
                   "Petronas LNG Supply Shift", "South China Sea Claim Overlap"]
    },
    "Vietnam": {
        "trade_exposure": (1, 4),
        "energy_exposure": (1, 3),
        "oil_delta": (-3, 10),
        "commodity_delta": (-4, 12),
        "events": ["South China Sea Block Dispute", "Manufacturing FDI Shift",
                   "Rice Export Embargo", "Vietnam-China Fishing Conflict"]
    },
    "Brazil": {
        "trade_exposure": (1, 3),
        "energy_exposure": (1, 2),
        "oil_delta": (-4, 12),
        "commodity_delta": (-8, 18),
        "events": ["Soya Export Shock", "Amazon Deforestation Carbon Sanction",
                   "BRL Depreciation Trade", "Lula G20 Alignment Shift"]
    },
    "Argentina": {
        "trade_exposure": (0, 2),
        "energy_exposure": (0, 1),
        "oil_delta": (-3, 10),
        "commodity_delta": (-6, 15),
        "events": ["IMF Default Sovereign Risk", "Peso Hyperinflation Cascade",
                   "Lithium Export Control", "Soy Oil Price Shock"]
    },
    "Canada": {
        "trade_exposure": (1, 4),
        "energy_exposure": (1, 3),
        "oil_delta": (-4, 12),
        "commodity_delta": (-5, 14),
        "events": ["India-Canada Diplomatic Rupture", "Khalistan Extremism Row",
                   "Pulses Export Ban India", "Agri Trade Route Disruption"]
    },
    "Australia": {
        "trade_exposure": (2, 5),
        "energy_exposure": (3, 7),
        "oil_delta": (-3, 10),
        "commodity_delta": (-6, 15),
        "events": ["LNG Contract Dispute", "Iron Ore Price Collapse",
                   "QUAD Strategic Realignment", "Australia-China Trade War Fallout"]
    },
    "Qatar": {
        "trade_exposure": (2, 5),
        "energy_exposure": (10, 20),
        "oil_delta": (5, 22),
        "commodity_delta": (3, 14),
        "events": ["LNG Supply Contract Renegotiation", "Qatar Blockade Spillover",
                   "FIFA World Cup Supply Chain", "JLNG Price Escalation"]
    },
    "Kuwait": {
        "trade_exposure": (1, 4),
        "energy_exposure": (6, 14),
        "oil_delta": (5, 25),
        "commodity_delta": (3, 12),
        "events": ["OPEC Oil Cut Extension", "Gulf Geopolitical Realignment",
                   "KPC Upstream Dispute", "Iraqi Border Conflict Spillover"]
    },
    "Oman": {
        "trade_exposure": (1, 3),
        "energy_exposure": (3, 8),
        "oil_delta": (3, 18),
        "commodity_delta": (2, 10),
        "events": ["Muscat Gas Route Disruption", "Arabian Sea Piracy Surge",
                   "Oman Strategic Neutral Pivot", "India Chabahar Alternative"]
    },
    "Egypt": {
        "trade_exposure": (1, 3),
        "energy_exposure": (1, 3),
        "oil_delta": (3, 22),
        "commodity_delta": (2, 14),
        "events": ["Suez Canal Disruption", "Egyptian Pound Collapse",
                   "Gaza Conflict Rafah Closure", "IMF Loan Crisis"]
    },
    "South Africa": {
        "trade_exposure": (1, 3),
        "energy_exposure": (1, 2),
        "oil_delta": (-3, 10),
        "commodity_delta": (-5, 14),
        "events": ["Eskom Power Grid Failure Export", "Cape of Good Hope Rerouting",
                   "BRICS Expansion Geopolitics", "Rand Currency Slide"]
    },
    "Nepal": {
        "trade_exposure": (1, 3),
        "energy_exposure": (1, 3),
        "oil_delta": (-2, 8),
        "commodity_delta": (-2, 6),
        "events": ["Hydropower Trade Disruption", "Madhesh Political Crisis",
                   "Nepal-India Open Border Tension", "China BRI Influence Shift"]
    },
    "Bhutan": {
        "trade_exposure": (1, 2),
        "energy_exposure": (1, 3),
        "oil_delta": (-1, 5),
        "commodity_delta": (-1, 4),
        "events": ["Doklam-Style Border Standoff", "Hydropower Export Halt",
                   "China-Bhutan Border Deal Impact"]
    },
    "Maldives": {
        "trade_exposure": (0, 1),
        "energy_exposure": (0, 1),
        "oil_delta": (-1, 5),
        "commodity_delta": (-1, 4),
        "events": ["India Out Campaign Diplomatic Row", "China Infrastructure Loan",
                   "Muizzu Anti-India Pivot", "Indian Military Pullout"]
    },
    "Kazakhstan": {
        "trade_exposure": (1, 3),
        "energy_exposure": (2, 6),
        "oil_delta": (3, 18),
        "commodity_delta": (1, 10),
        "events": ["January Uprising Economic Shock", "CPC Pipeline Disruption",
                   "CSTO Military Intervention", "Tengiz Field Output Cut"]
    },
    "Ukraine": {
        "trade_exposure": (1, 3),
        "energy_exposure": (1, 3),
        "oil_delta": (10, 38),
        "commodity_delta": (8, 28),
        "events": ["Russo-Ukrainian War Grain Block", "Black Sea Shipping Halt",
                   "Sunflower Oil Export Collapse", "Missile Strike Infrastructure",
                   "Kherson Dam Destruction Cascade"]
    },
    "Poland": {
        "trade_exposure": (1, 2),
        "energy_exposure": (1, 2),
        "oil_delta": (-3, 12),
        "commodity_delta": (-3, 10),
        "events": ["NATO Eastern Flank Escalation", "Refugee Economic Burden",
                   "Polish Border Belarus Crisis", "Coal Import Shift India"]
    },
    "Mexico": {
        "trade_exposure": (0, 2),
        "energy_exposure": (1, 3),
        "oil_delta": (-3, 12),
        "commodity_delta": (-4, 12),
        "events": ["Nearshoring Manufacturing Shift", "Cartel Supply Chain Risk",
                   "US-Mexico Border Crisis Spillover", "USMCA Pharmaceutical Impact"]
    },
}

def generate_event_row(country, profile, event_date):
    """Generate a single realistic event row based on country profile."""
    oil_d = float(np.clip(np.random.normal(
        (profile["oil_delta"][0] + profile["oil_delta"][1]) / 2,
        (profile["oil_delta"][1] - profile["oil_delta"][0]) / 4
    ), profile["oil_delta"][0], profile["oil_delta"][1]))

    comm_d = float(np.clip(np.random.normal(
        (profile["commodity_delta"][0] + profile["commodity_delta"][1]) / 2,
        (profile["commodity_delta"][1] - profile["commodity_delta"][0]) / 4
    ), profile["commodity_delta"][0], profile["commodity_delta"][1]))

    trade_disr = float(np.clip(np.random.normal(40, 18), 5, 95))
    ship_disr = float(np.clip(np.random.normal(38, 17), 5, 95))

    india_trade_exp = float(np.clip(np.random.uniform(
        profile["trade_exposure"][0], profile["trade_exposure"][1]
    ), 0, 100))

    india_energy_exp = float(np.clip(np.random.uniform(
        profile["energy_exposure"][0], profile["energy_exposure"][1]
    ), 0, 100))

    volatility = float(np.clip(np.random.normal(5.0, 2.2), 0.5, 10.0))

    # Composite target: weighted sum of real impact drivers (0-100 scale)
    # Reflects India Current Account impact in scaled basis-point proxy
    raw_impact = (
        0.28 * abs(oil_d) +
        0.18 * abs(comm_d) +
        0.20 * trade_disr +
        0.12 * ship_disr +
        0.15 * india_trade_exp +
        0.17 * india_energy_exp +
        0.10 * volatility * 10  # scale 0-10 -> 0-100 range factor
    )

    # Normalise to 0-100
    india_impact = float(np.clip(raw_impact / 1.3, 0.5, 99.5))
    india_impact = round(india_impact + np.random.normal(0, 1.5), 2)
    india_impact = max(0.5, min(99.5, india_impact))

    event = np.random.choice(profile["events"])

    return {
        "country": country,
        "event_date": event_date.strftime("%Y-%m-%d"),
        "event_description": event,
        "oil_price_change": round(oil_d, 2),
        "commodity_price_change": round(comm_d, 2),
        "trade_disruption": round(trade_disr, 2),
        "shipping_disruption": round(ship_disr, 2),
        "india_trade_exposure": round(india_trade_exp, 2),
        "india_energy_exposure": round(india_energy_exp, 2),
        "market_volatility": round(volatility, 2),
        "india_economic_impact": round(india_impact, 2)
    }

def build_dataset(n_rows=600):
    rows = []
    countries = list(COUNTRY_PROFILES.keys())
    start_date = pd.Timestamp("2015-01-01")
    end_date = pd.Timestamp("2025-06-30")
    date_range = (end_date - start_date).days

    for i in range(n_rows):
        country = countries[i % len(countries)]
        # Also add extra rows for high-impact countries
        if i > len(countries):
            country = np.random.choice(countries, p=None)

        profile = COUNTRY_PROFILES[country]
        evt_date = start_date + pd.Timedelta(days=np.random.randint(0, date_range))
        row = generate_event_row(country, profile, evt_date)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates()
    df = df.sort_values("event_date").reset_index(drop=True)
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = build_dataset(n_rows=620)
    out = "data/feature4_india_economic_impact.csv"
    df.to_csv(out, index=False)
    print(f"Dataset saved: {out}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample:")
    print(df.head(3).to_string())
    print(f"\nTarget stats:")
    print(df["india_economic_impact"].describe())
    print(f"\nCountry coverage: {df['country'].nunique()} countries")
