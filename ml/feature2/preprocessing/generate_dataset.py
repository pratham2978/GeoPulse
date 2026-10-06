"""
Dataset Generator for Feature 2: News & Narrative Classification
Generates a realistic, diverse geopolitical news dataset across 6 syllabus narrative categories:
1. Military Conflict
2. Diplomatic / Political
3. Energy Risk
4. Trade & Shipping
5. Economic Impact
6. Humanitarian Impact

Includes realistic missing values and duplicates to demonstrate standard preprocessing and cleaning.
"""

import os
import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

CATEGORIES = [
    'Military Conflict',
    'Diplomatic / Political',
    'Energy Risk',
    'Trade & Shipping',
    'Economic Impact',
    'Humanitarian Impact'
]

SOURCES = [
    'Reuters', 'Associated Press', 'Bloomberg', 'Financial Times',
    'Wall Street Journal', 'BBC News', 'Deutsche Welle', 'Al Jazeera',
    'Nikkei Asia', 'UN News'
]

COUNTRIES = [
    ('Ukraine', 'Eastern Europe'),
    ('Russia', 'Eastern Europe'),
    ('Israel', 'Middle East'),
    ('Iran', 'Persian Gulf'),
    ('Saudi Arabia', 'Persian Gulf'),
    ('Yemen', 'Red Sea & Horn of Africa'),
    ('Taiwan', 'East Asia'),
    ('China', 'East Asia'),
    ('United States', 'North America'),
    ('United Kingdom', 'Western Europe'),
    ('Germany', 'Western Europe'),
    ('Poland', 'Eastern Europe'),
    ('Egypt', 'Red Sea & Horn of Africa'),
    ('Sudan', 'Red Sea & Horn of Africa'),
    ('India', 'South Asia'),
    ('Pakistan', 'South Asia'),
    ('Turkey', 'Middle East'),
    ('South Korea', 'East Asia'),
    ('Japan', 'East Asia'),
    ('Qatar', 'Persian Gulf')
]

# Rich vocabulary templates for authentic narrative simulation
TEMPLATES = {
    'Military Conflict': [
        (
            "Artillery barrages and tactical drone strikes intensify along contested frontline",
            "Heavy artillery exchanges and long-range unmanned aerial strikes escalated across contested frontline defense sectors overnight. Ground command headquarters confirmed multiple cruise missile interceptions and reinforced mechanized infantry brigades holding fortified trenches. Contested air defense batteries engaged low-altitude reconnaissance drones while combat engineers deployed perimeter minefields."
        ),
        (
            "Naval task force engages hostile explosive drone boats in strategic strait",
            "Guided missile destroyers deployed on maritime security patrol neutralized several unmanned explosive surface vessels attempting to strike military logistics convoys. Naval fleet commanders ordered elevated combat readiness and deployed electronic jamming countermeasures to disable hostile telemetry links along contested maritime transit corridors."
        ),
        (
            "Hypersonic missile barrage targets critical defense installations and radar outposts",
            "Air defense surveillance networks detected high-velocity aerodynamic missile trajectories aimed at strategic command nodes and hardened munitions storage facilities. Military spokespersons stated that mobile interceptor batteries engaged multiple inbound warheads, while combat search-and-rescue teams mobilized across blast perimeters."
        ),
        (
            "Cross-border commando raid triggers intense infantry firefight along buffer line",
            "Special forces detachments conducted reconnaissance incursions into contested border sectors, sparking sustained firefights with entrenched hostile infantry. Rapid deployment armored battalions were dispatched with close air support to establish defensive perimeters and extract forward patrol units under heavy mortar suppression."
        ),
        (
            "Armored battalion mobilizes main battle tanks as cross-border incursions escalate",
            "Mechanized combat divisions equipped with main battle tanks and armored fighting vehicles initiated defensive counter-maneuvers following reported hostile border crossings. Military leadership enacted wartime emergency protocol, stationing self-propelled howitzers and air surveillance batteries across frontline tactical sectors."
        ),
        (
            "Air defense interceptors scramble as hostile bomber formations approach territorial airspace",
            "Tactical fighter squadrons conducted emergency combat air patrols after long-range strategic bombers were detected operating near sovereign air defense identification zones. Defense ministry officials confirmed that electronic warfare aircraft shadowed the formations until they altered course away from defensive boundaries."
        ),
        (
            "Urban warfare intensifies as mechanized units push into contested industrial outskirts",
            "Heavy machine gun fire, anti-tank guided missiles, and snipers engaged in ferocious building-to-building clashes in industrial sectors. Combat engineers detonated obstacle barriers under smoke screens while field artillery delivered preparatory barrages against reinforced strongpoints."
        ),
        (
            "Tactical ballistic missiles intercept hostile command post in retaliatory strike",
            "Military command launched precision ballistic missiles targeting an underground tactical operations center following sustained perimeter rocket attacks. Satellite damage assessments confirmed extensive structural destruction to communications masts and command bunkers."
        )
    ],
    'Diplomatic / Political': [
        (
            "Bilateral ceasefire negotiations resume under international diplomatic mediation",
            "Special diplomatic envoys and foreign ministers convened high-stakes closed-door consultations aimed at establishing an immediate humanitarian truce and verifiable demilitarization framework. Negotiators reviewed draft protocols addressing prisoner-of-war exchanges, border verification monitors, and the formal cessation of cross-border hostilities."
        ),
        (
            "United Nations Security Council convenes emergency session over border sovereignty dispute",
            "Permanent delegates at the United Nations Headquarters debated a draft resolution condemning territorial incursions and mandating the deployment of an international peacekeeping observer mission. Diplomatic attachés engaged in intense multilateral consultations to resolve contentious veto language regarding territorial sovereignty."
        ),
        (
            "Ambassadorial recall and formal diplomatic demarche issued over treaty violation",
            "The foreign ministry formally summoned the resident ambassador to deliver a stern diplomatic demarche protesting recent unilateral border treaties and sovereignty violations. State officials announced the immediate recall of senior diplomatic legations for urgent foreign policy strategy reviews."
        ),
        (
            "Regional security summit yields preliminary multilateral mutual non-aggression pact",
            "Heads of state concluded an emergency multilateral summit with a joint communique pledging adherence to non-aggression principles and established arbitration treaties. Diplomatic working groups were tasked with establishing direct hotline communications between military defense commands to mitigate miscalculation risks."
        ),
        (
            "Parliamentary committee ratifies strategic bilateral defense cooperation treaty",
            "Legislators voted overwhelmingly to approve a comprehensive security alliance treaty authorizing joint intelligence sharing, reciprocal military base access, and coordinated defense procurement. Government ministers emphasized that the diplomatic accord reinforces regional deterrence without violating international neutrality accords."
        ),
        (
            "Multilateral peace conference deadlocked over territorial governance and treaty guarantees",
            "Diplomatic negotiations stalled after opposing delegations rejected compromise terms concerning international border administration and security guarantees. Facilitators from neutral states urged both delegations to maintain working-level communications and prevent the total breakdown of diplomatic dialogue."
        ),
        (
            "Foreign ministers issue joint communique calling for de-escalation and diplomatic roadmaps",
            "Senior diplomats representing regional powers issued a coordinated statement demanding an immediate halt to escalating rhetoric and the restoration of established diplomatic channels. The communique proposed a three-phase transition roadmap including demilitarized buffer zones and internationally monitored elections."
        ),
        (
            "Special presidential envoy conducts whirlwind diplomatic tour across regional capitals",
            "Traveling through key allied capitals, the senior diplomatic envoy presented confidential de-escalation proposals designed to avert broader regional conflict. Discussions centered on mutual troop withdrawals, diplomatic concessions, and security guarantees backed by multilateral institutions."
        )
    ],
    'Energy Risk': [
        (
            "Major crude oil pipeline flow suspended following strategic transit sabotage threat",
            "Energy operators halted pumping operations along a major transcontinental crude pipeline after intelligence reports identified credible security threats against remote pumping stations. Energy security ministries initiated pipeline integrity inspections, while oil refinery operators tapped reserve stockpiles to maintain downstream distillation output."
        ),
        (
            "OPEC+ convenes emergency ministerial meeting as maritime transit tensions threaten crude exports",
            "Petroleum exporting ministers scheduled extraordinary virtual consultations to review market stability and global spare crude production capacity following maritime bottlenecks in key transit chokepoints. Energy analysts cautioned that export delays could quickly deplete OECD commercial crude reserves and elevate spot price volatility."
        ),
        (
            "Liquefied natural gas terminal throttles shipments amid regional maritime insurance restrictions",
            "Major LNG export liquefaction facilities reduced cargo loadings after maritime underwriters imposed prohibitive war-risk insurance premiums on LNG tankers navigating adjacent waterways. European and Asian utility operators scrambled to secure spot replacement cargoes ahead of peak seasonal heating demand."
        ),
        (
            "Strategic petroleum reserve release authorized to counterbalance global supply disruptions",
            "Energy authorities authorized the emergency drawdown of millions of barrels of crude oil from national strategic petroleum reserves to stabilize domestic fuel supply chains. Energy department officials coordinated the release with international energy partners to offset prolonged refinery feedstock shortages."
        ),
        (
            "Offshore drilling platform security elevated following hostile drone reconnaissance sightings",
            "Offshore energy operators implemented maximum security protocols across deepwater drilling rigs after unidentified aerial drones were observed conducting surveillance over gas extraction platforms. Naval patrol craft were deployed to maintain exclusion perimeters around strategic energy infrastructure."
        ),
        (
            "Natural gas pipeline compression station suffers severe explosion amid geopolitical friction",
            "A critical natural gas compression station feeding industrial manufacturing hubs suffered a catastrophic pipeline failure, cutting regional gas transit volumes by over 40 percent. Energy distribution utilities immediately mandated industrial consumption rationing while emergency engineering teams assessed blast damage."
        ),
        (
            "Uranium enrichment and nuclear fuel supply chains threatened by geopolitical trade curbs",
            "Civilian nuclear power operators warned of long-term fuel fabrication vulnerabilities as geopolitical tensions disrupted international uranium enrichment and nuclear fuel rod shipments. Power plant managers initiated fuel conservation measures to safeguard baseload electrical generation capacity."
        ),
        (
            "Power grid operators initiate emergency load shedding following sabotage of high-voltage transmission lines",
            "National electrical transmission coordinators implemented rolling blackouts across metropolitan centers after key electrical substations and cross-border interconnectors sustained coordinated physical sabotage. Energy officials urged citizens to conserve electricity while backup diesel generators were deployed to hospitals."
        )
    ],
    'Trade & Shipping': [
        (
            "Container shipping carriers divert cargo fleets around Cape of Good Hope amid maritime threats",
            "Leading international container shipping lines announced indefinite rerouting of commercial vessels around southern Africa to avoid high-risk maritime straits. The maritime diversion adds 10 to 14 days to standard transit voyages, absorbing container fleet capacity, delaying port turnarounds, and triggering sharp increases in spot ocean freight rates."
        ),
        (
            "Port congestion and customs clearance bottlenecks escalate across major maritime container terminals",
            "Logistics operators reported severe berthing queues and prolonged container dwell times exceeding historical highs at major oceanic gateway terminals. Freight forwarders cited vessel bunching, rail intermodal backlogs, and maritime rerouting schedules as compounding drivers of global supply chain gridlock."
        ),
        (
            "Bilateral tariff escalation and export licensing controls enacted on critical high-tech components",
            "Commerce ministries imposed sweeping tariff increases and strict export license restrictions on key industrial raw materials, advanced machine tools, and semiconductor wafers. Trade associations warned that compliance friction and retaliatory tariffs threaten to disrupt global electronic manufacturing supply chains."
        ),
        (
            "Dry bulk carrier grain shipments stranded in territorial waters amid maritime security impasse",
            "Dozens of bulk cargo vessels carrying millions of metric tons of wheat and agricultural commodities remained anchored outside key commercial ports due to unconfirmed maritime passage clearances and mine threat assessments. Agricultural traders warned of contract defaults and grain supply shortages in import-dependent nations."
        ),
        (
            "Air cargo freight rates surge as commercial shippers seek alternatives to disrupted ocean lanes",
            "Air logistics forwarders recorded historic price increases per kilogram of air cargo as manufacturers chartered dedicated freighter aircraft to bypass blocked maritime shipping corridors. Cargo airlines expanded flight schedules between manufacturing hubs to accommodate high-value automotive and electronics freight."
        ),
        (
            "Maritime insurance syndicates cancel commercial hull coverage for vessels navigating disputed waters",
            "Global marine insurance underwriters formally notified shipowners that war-risk hull and machinery insurance policies would be rescinded for vessels transiting designated high-risk maritime corridors. Ship operators faced the choice between exorbitant special indemnity premiums or canceling scheduled voyages."
        ),
        (
            "Customs authority impounds containerized shipments under expanded cross-border trade sanctions",
            "Port customs officials seized hundreds of cargo containers suspected of violating expanded international sanctions targeting dual-use industrial components. Importers and freight forwarders faced lengthy legal audits and storage fees as customs clearance procedures were dramatically expanded."
        ),
        (
            "Global automotive supply chains stall as crucial raw material cargo shipments face port delays",
            "Major automotive assembly plants announced temporary production line shutdowns due to critical parts shortages resulting from delayed maritime shipping containers. Logistics coordinators reported that specialized component vessels had been delayed over two weeks due to harbor congestion."
        )
    ],
    'Economic Impact': [
        (
            "Central banks signal emergency interest rate adjustments as geopolitical inflation pressures broaden",
            "Monetary policy committees convened extraordinary policy meetings to evaluate persistent macroeconomic shocks stemming from import price inflation and commodity supply friction. Financial market analysts revised benchmark interest rate trajectories upward as headline consumer price indices reflected elevated freight and producer input expenses."
        ),
        (
            "Sovereign credit rating placed under negative review following capital flight and currency depreciation",
            "International credit rating agencies placed sovereign debt issuances on negative watch, citing widening fiscal deficits, rapid foreign exchange reserve depletion, and heightened geopolitical risk premia. Domestic currency valuations experienced sharp depreciations against global reserve currencies."
        ),
        (
            "Stock exchange volatility index spikes as investors rotate capital into sovereign bonds and gold",
            "Equities across regional bourses recorded broad sell-offs, with industrial, aviation, and consumer discretionary shares leading declines. Capital allocators rotated liquidity into sovereign treasury securities, bullion, and cash equivalents amid heightened geopolitical uncertainty and macroeconomic stagflation fears."
        ),
        (
            "Manufacturing purchasing managers index drops into contraction territory amid input price spikes",
            "National manufacturing indices slumped below the critical 50-point threshold, signaling industrial output contraction across export-oriented sectors. Factory managers reported elevated input acquisition costs, backlog accumulation, and declining international export orders due to geopolitical uncertainty."
        ),
        (
            "Foreign direct investment inflows contract sharply amid cross-border geopolitical instability",
            "Economic development ministries reported a substantial deceleration in inbound foreign direct investment as multinational corporations suspended cross-border capital expansion projects. Investment analysts cited sovereign expropriation risks and geopolitical volatility as primary drivers of capital hesitation."
        ),
        (
            "Bilateral currency swaps activated as commercial banking system faces dollar liquidity squeeze",
            "Central monetary authorities tapped bilateral foreign currency swap lines to inject hard currency liquidity into domestic commercial banks struggling with overseas settlement obligations. Treasury officials emphasized that foreign reserve backstops remain adequate to maintain interbank clearing operations."
        ),
        (
            "Consumer confidence plunges as energy and food price inflation erodes household purchasing power",
            "National economic sentiment surveys revealed the steepest monthly drop in consumer confidence in several years as rising utility bills and grocery costs squeezed household budgets. Economists warned that declining domestic retail consumption will act as a major drag on quarterly gross domestic product."
        ),
        (
            "Corporate debt default risk rises as borrowing costs spike following geopolitical credit repricing",
            "Credit default swap spreads widened significantly across corporate debt markets as lenders repriced risk margins for businesses exposed to cross-border supply shocks. Debt restructuring advisors noted a sharp uptick in emergency corporate credit facility requests."
        )
    ],
    'Humanitarian Impact': [
        (
            "Civilian evacuation corridors established under international red cross mediation",
            "Humanitarian organizations coordinated temporary civilian safe-passage corridors to evacuate vulnerable elderly, women, and pediatric patients from active hostility perimeters. International observers monitored compliance to ensure unhindered transit for humanitarian convoys transporting essential rations and emergency medical supplies."
        ),
        (
            "United Nations launches emergency humanitarian relief appeal for food aid and medical shelter",
            "Humanitarian coordinators formally issued a multilateral funding appeal seeking urgent donor contributions to support frontline field clinics and nutritional distribution hubs. Aid convoys reported severe logistical hurdles attempting to deliver relief supplies to isolated civilian populations trapped along combat axes."
        ),
        (
            "Refugee displacement crisis deepens as civilian populations flee contested border zones",
            "International migration agencies reported that hundreds of thousands of displaced civilians have crossed international frontiers seeking asylum, overwhelming border reception facilities and temporary transit camps. Relief agencies mobilized emergency winterization tents, thermal blankets, and clean water distribution systems."
        ),
        (
            "Disrupted municipal power grids and water filtration plants jeopardize civilian public health",
            "Public health authorities warned of escalating waterborne disease outbreaks after sustained infrastructure damage crippled municipal water purification networks and electrical sub-stations. Emergency relief teams deployed portable purification modules and oral rehydration packets to mitigate sanitary hazards."
        ),
        (
            "Field hospitals face critical shortages of surgical anesthetics, electricity, and clean water",
            "Medical relief organizations operating inside crisis sectors issued urgent distress alerts regarding depleted pharmaceuticals, damaged municipal water treatment facilities, and erratic generator fuel supplies necessary to maintain life-saving intensive care and trauma operating rooms."
        ),
        (
            "Malnutrition rates spike among pediatric populations in besieged urban pockets",
            "International aid workers documented severe acute malnutrition among infants and children cut off from standard commercial food distribution networks. Humanitarian flights attempted emergency air-drops of therapeutic nutritional paste while requesting immediate humanitarian pauses."
        ),
        (
            "Makeshift refugee settlements overwhelmed by torrential flooding and freezing weather",
            "Severe seasonal weather inundated informal displacement settlements, destroying temporary plastic shelters and creating catastrophic sanitary conditions for thousands of families. Disaster response teams mobilized emergency drainage pumps and emergency clothing distributions."
        ),
        (
            "Mobile medical teams deploy to remote border crossings as civilian trauma casualties mount",
            "Emergency medical NGOs deployed mobile surgical units and trauma ambulances along primary refugee transit highways to treat shrapnel injuries and dehydration. Medical directors reported that regional clinics have exhausted their primary supplies of blood plasma and IV fluids."
        )
    ]
}

def generate_dataset():
    records = []
    article_counter = 10000

    # Generate ~255 records per category = 1530 clean records
    for category in CATEGORIES:
        templates = TEMPLATES[category]
        for i in range(255):
            article_counter += 1
            article_id = f"ART-{article_counter}"
            template_idx = i % len(templates)
            base_title, base_text = templates[template_idx]

            # Pick country & region
            country, region = random.choice(COUNTRIES)
            source = random.choice(SOURCES)

            # Generate realistic variation
            day = random.randint(1, 28)
            month = random.randint(1, 10)
            date_str = f"2026-{month:02d}-{day:02d}"

            # Subtle real-world localized phrasing to create natural lexical diversity
            loc_phrases = [
                f"Reporting from {country}, diplomatic and local observers noted that strategic shifts in {region} continue to dictate regional outcomes.",
                f"Field correspondents in {country} reported that ongoing developments across {region} are compounding operational risks.",
                f"Authorities in {country} emphasized that immediate stability in {region} remains essential to mitigating broader systemic fallout.",
                f"Strategic intelligence assessments issued in {country} indicate that conditions in {region} remain highly dynamic and fluid.",
                f"Analysts tracking developments in {country} highlighted that multilateral actors in {region} are preparing contingency measures."
            ]
            loc_note = random.choice(loc_phrases)

            # Vary the text slightly with authentic synonyms/additions
            variations = [
                f"{base_text} {loc_note}",
                f"{base_text} In {country}, emergency coordination committees met to assess secondary vulnerabilities across {region}.",
                f"{base_text} International stakeholders operating within {region} called on officials in {country} to exercise maximum caution.",
                f"{base_text} According to senior officials in {country}, recent indicators across {region} underscore the urgency of coordinated multilateral action."
            ]
            text = random.choice(variations)

            records.append({
                'article_id': article_id,
                'title': base_title,
                'text': text,
                'label': category,
                'date': date_str,
                'source': source,
                'country': country,
                'region': region
            })

    # Now add realistic missing values (12 records)
    for _ in range(4):
        # Missing text
        article_counter += 1
        c, r = random.choice(COUNTRIES)
        records.append({
            'article_id': f"ART-{article_counter}",
            'title': "Diplomatic communique issued regarding border status",
            'text': np.nan,
            'label': 'Diplomatic / Political',
            'date': '2026-04-12',
            'source': 'Reuters',
            'country': c,
            'region': r
        })

    for _ in range(4):
        # Missing title
        article_counter += 1
        c, r = random.choice(COUNTRIES)
        records.append({
            'article_id': f"ART-{article_counter}",
            'title': np.nan,
            'text': "Crude oil transport and pipeline networks faced severe operational friction across key refining hubs.",
            'label': 'Energy Risk',
            'date': '2026-05-18',
            'source': 'Bloomberg',
            'country': c,
            'region': r
        })

    for _ in range(4):
        # Missing label
        article_counter += 1
        c, r = random.choice(COUNTRIES)
        records.append({
            'article_id': f"ART-{article_counter}",
            'title': "Maritime shipping routes face extended commercial delays",
            'text': "Container ships transiting maritime corridors experienced cargo delay times exceeding historical averages.",
            'label': np.nan,
            'date': '2026-06-21',
            'source': 'Financial Times',
            'country': c,
            'region': r
        })

    # Now add realistic duplicates (38 duplicate records)
    duplicate_candidates = random.sample(records[:1500], 38)
    for dup in duplicate_candidates:
        article_counter += 1
        dup_copy = dup.copy()
        dup_copy['article_id'] = f"ART-{article_counter}"
        records.append(dup_copy)

    # Shuffle records
    random.shuffle(records)

    df = pd.DataFrame(records)
    print(f"Generated Raw Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Class distribution:\n", df['label'].value_counts(dropna=False))
    print("Null count:\n", df.isnull().sum())

    # Save raw dataset
    raw_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'data', 'raw', 'geopolitical_news_raw.csv')
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"Saved raw dataset to: {raw_path}")

    # Also save to ml/feature2/data/
    ml_raw_path = os.path.join(os.path.dirname(__file__), 'data', 'geopolitical_news_raw.csv')
    os.makedirs(os.path.dirname(ml_raw_path), exist_ok=True)
    df.to_csv(ml_raw_path, index=False)
    print(f"Saved copy to: {ml_raw_path}")

if __name__ == '__main__':
    generate_dataset()
