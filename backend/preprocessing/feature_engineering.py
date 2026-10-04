"""
Feature Engineering for GeoPulse AI
Computes the engineered GeoPulse Event Severity Score, severity levels,
affected sector associations, and prepares ML-ready feature vectors.
"""

import math

COUNTRY_NAMES = {
    'IND': 'India', 'IN': 'India',
    'USA': 'United States', 'US': 'United States',
    'CHN': 'China', 'CH': 'China',
    'RUS': 'Russia', 'RS': 'Russia',
    'UKR': 'Ukraine', 'UP': 'Ukraine',
    'ISR': 'Israel', 'IS': 'Israel',
    'IRN': 'Iran', 'IR': 'Iran',
    'YEM': 'Yemen', 'YM': 'Yemen',
    'TWN': 'Taiwan', 'TW': 'Taiwan',
    'SAU': 'Saudi Arabia', 'SA': 'Saudi Arabia',
    'DEU': 'Germany', 'GM': 'Germany',
    'GBR': 'United Kingdom', 'UK': 'United Kingdom',
    'FRA': 'France', 'FR': 'France',
    'JPN': 'Japan', 'JA': 'Japan',
    'TUR': 'Turkey', 'TU': 'Turkey',
    'EGY': 'Egypt', 'EG': 'Egypt',
    'PAK': 'Pakistan', 'PK': 'Pakistan',
    'KOR': 'South Korea', 'KS': 'South Korea',
    'PRK': 'North Korea', 'KN': 'North Korea',
    'SYR': 'Syria', 'SY': 'Syria',
    'LBN': 'Lebanon', 'LE': 'Lebanon',
    'ARE': 'United Arab Emirates', 'AE': 'United Arab Emirates',
    'QAT': 'Qatar', 'QA': 'Qatar',
    'POL': 'Poland', 'PL': 'Poland',
    'BLR': 'Belarus', 'BO': 'Belarus',
}

def resolve_country_name(code):
    if not code or code == 'None':
        return 'International / Global'
    code_upper = str(code).strip().upper()
    return COUNTRY_NAMES.get(code_upper, code_upper)

def calculate_severity_score(goldstein, mentions, sources, articles, avg_tone, quad_class=4):
    """
    Computes GeoPulse Event Severity Score (0.0 to 100.0).
    Combines:
    - Goldstein Scale (conflict intensity vs cooperation)
    - Negative Tone (sentiment distress/aggression)
    - Media Attention (log-scaled mentions + sources count)
    - QuadClass Material vs Verbal severity weight
    """
    try:
        g = float(goldstein)
    except (ValueError, TypeError):
        g = 0.0

    try:
        m = float(mentions)
    except (ValueError, TypeError):
        m = 1.0

    try:
        s = float(sources)
    except (ValueError, TypeError):
        s = 1.0

    try:
        t = float(avg_tone)
    except (ValueError, TypeError):
        t = 0.0

    try:
        qc = int(quad_class)
    except (ValueError, TypeError):
        qc = 3

    # 1. Conflict Intensity (0 to 100): Lower (more negative) Goldstein means higher conflict
    # Goldstein ranges from -10 to +10
    conflict_intensity = max(0.0, min(100.0, ((10.0 - g) / 20.0) * 100.0))

    # 2. Tone Distress (0 to 100): More negative tone indicates higher hostility/crisis
    # Tone usually -10 to +10
    tone_distress = max(0.0, min(100.0, ((-t + 10.0) / 20.0) * 100.0))

    # 3. Media Attention (0 to 100) using log scaling to prevent skew
    mentions_log = min(100.0, math.log10(max(1.0, m)) * 26.0)
    sources_log = min(100.0, math.log10(max(1.0, s)) * 34.0)
    media_attention = 0.6 * mentions_log + 0.4 * sources_log

    # 4. QuadClass impact weight
    # 4: Material Conflict, 3: Verbal Conflict, 2: Material Cooperation, 1: Verbal Cooperation
    qc_weight = {4: 100.0, 3: 75.0, 2: 30.0, 1: 15.0}.get(qc, 50.0)

    # Weighted Composite Formula
    raw_score = (
        0.42 * conflict_intensity +
        0.26 * tone_distress +
        0.22 * media_attention +
        0.10 * qc_weight
    )

    score = round(max(5.0, min(99.4, raw_score)), 1)
    
    # Categorize severity level
    if score >= 75.0:
        level = 'CRITICAL'
    elif score >= 50.0:
        level = 'HIGH'
    elif score >= 25.0:
        level = 'MODERATE'
    else:
        level = 'LOW'

    return score, level

def infer_affected_sectors(category, event_code, country, text_keywords=""):
    """
    Associates geopolitical events with affected real-world economic sectors.
    """
    sectors = []
    cat_lower = str(category).lower()
    c_lower = str(country).lower()
    kw_lower = str(text_keywords).lower()

    # Energy sector
    if any(k in kw_lower or k in c_lower for k in ['red sea', 'hormuz', 'yemen', 'iran', 'russia', 'pipeline', 'oil', 'lng', 'gas', 'tanker']) or 'conflict' in cat_lower:
        sectors.append({'sector': 'Energy', 'icon': 'Zap', 'risk': 'High' if 'conflict' in cat_lower else 'Moderate'})

    # Shipping sector
    if any(k in kw_lower or k in c_lower for k in ['red sea', 'suez', 'panama', 'malacca', 'strait', 'maritime', 'ship', 'cargo', 'ais', 'yemen', 'taiwan']):
        sectors.append({'sector': 'Shipping', 'icon': 'Anchor', 'risk': 'High'})

    # Trade sector
    if 'economic' in cat_lower or 'trade' in cat_lower or any(k in kw_lower for k in ['sanction', 'tariff', 'embargo', 'supply', 'port', 'trade']):
        sectors.append({'sector': 'Trade', 'icon': 'Package', 'risk': 'High' if 'conflict' in cat_lower else 'Moderate'})

    # Commodities sector
    if any(k in kw_lower or k in c_lower for k in ['grain', 'wheat', 'fertilizer', 'black sea', 'ukraine', 'agriculture', 'gold', 'crude']):
        sectors.append({'sector': 'Commodities', 'icon': 'Coins', 'risk': 'High'})

    # Industry / Regional Economy fallback
    sectors.append({'sector': 'Regional Economy', 'icon': 'Globe', 'risk': 'Moderate'})

    return sectors
