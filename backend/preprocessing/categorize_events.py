"""
CAMEO Event Category Mapping for GeoPulse AI
Maps standard GDELT CAMEO event root codes (01-20) and specific event codes to
human-understandable geopolitical categories.
"""

CAMEO_ROOT_MAP = {
    '01': 'Diplomatic',             # Make public statement
    '02': 'Diplomatic',             # Appeal
    '03': 'Security Cooperation',   # Express intent to cooperate
    '04': 'Diplomatic',             # Consult
    '05': 'Diplomatic',             # Engage in diplomatic cooperation
    '06': 'Economic / Trade',       # Engage in material cooperation
    '07': 'Humanitarian',           # Provide aid
    '08': 'Political',              # Yield
    '09': 'Political',              # Investigate
    '10': 'Political',              # Demand
    '11': 'Political',              # Disapprove
    '12': 'Political',              # Reject
    '13': 'Conflict / Military',    # Threaten
    '14': 'Protest / Civil Unrest', # Protest
    '15': 'Conflict / Military',    # Exhibit force posture
    '16': 'Diplomatic',             # Reduce relations
    '17': 'Conflict / Military',    # Coerce
    '18': 'Conflict / Military',    # Assault
    '19': 'Conflict / Military',    # Fight / Military Force
    '20': 'Conflict / Military',    # Unconventional Mass Violence
}

CAMEO_CODE_DESCRIPTIONS = {
    '190': 'Use conventional military force',
    '193': 'Fight with small arms and light weapons',
    '194': 'Fight with artillery / rockets / missiles',
    '195': 'Employ aerial weapons / airstrikes / drones',
    '196': 'Maritime military engagement / ship interdiction',
    '180': 'Use unconventional violence',
    '182': 'Physically assault / execute targeted strikes',
    '150': 'Demonstrate military / strategic force posture',
    '154': 'Mobilize or increase police/military alert',
    '141': 'Demonstrate or conduct civil protest',
    '145': 'Violent protest / riot / barricades',
    '130': 'Issue strategic threat or ultimatum',
    '138': 'Threaten military force or blockade',
    '111': 'Formally condemn or disapprove action',
    '120': 'Reject proposal or diplomatic treaty',
    '100': 'Demand compliance / territory / policy change',
    '060': 'Sign commercial or trade agreement',
    '061': 'Authorize bilateral investment or loans',
    '040': 'Host diplomatic summit or high-level talks',
    '030': 'Express intent to engage in security alliance',
    '020': 'Appeal to international community or UN',
    '010': 'Make official government press statement',
}

def get_event_category(event_code, root_code=None):
    code_str = str(event_code).zfill(3)
    if not root_code:
        root_code = code_str[:2]
    else:
        root_code = str(root_code).zfill(2)
        
    # Check direct code overrides
    if code_str.startswith('19') or code_str.startswith('20') or code_str.startswith('18'):
        return 'Conflict / Military'
    if code_str.startswith('14'):
        return 'Protest / Civil Unrest'
    if code_str in ['060', '061', '062', '063', '064']:
        return 'Economic / Trade'
    
    return CAMEO_ROOT_MAP.get(root_code, 'Other')

def get_event_description(event_code):
    code_str = str(event_code).zfill(3)
    return CAMEO_CODE_DESCRIPTIONS.get(code_str, f"CAMEO Event {code_str}")
