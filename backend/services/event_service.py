"""
Event Service for GeoPulse AI Feature 1
Provides querying, filtering, statistics, and country/cluster intelligence.
"""

import os
import json
import ast
import pandas as pd
from datetime import datetime, timedelta

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
CSV_PATH = os.path.join(PROCESSED_DIR, 'global_events.csv')
CLUSTERS_PATH = os.path.join(PROCESSED_DIR, 'event_clusters.json')
STATS_PATH = os.path.join(PROCESSED_DIR, 'summary_stats.json')

class EventService:
    def __init__(self):
        self._load_data()

    def _load_data(self):
        if os.path.exists(CSV_PATH):
            self.df = pd.read_csv(CSV_PATH)
            # Parse affected_sectors if string
            if 'affected_sectors' in self.df.columns and isinstance(self.df['affected_sectors'].iloc[0], str):
                self.df['affected_sectors'] = self.df['affected_sectors'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [x]
                )
        else:
            self.df = pd.DataFrame()

        if os.path.exists(CLUSTERS_PATH):
            with open(CLUSTERS_PATH, 'r', encoding='utf-8') as f:
                self.clusters = json.load(f)
        else:
            self.clusters = []

    def get_stats(self):
        if self.df.empty:
            return {}
        
        total_events = int(len(self.df))
        conflict_events = int(len(self.df[self.df['event_category'] == 'Conflict / Military']))
        political_events = int(len(self.df[self.df['event_category'] == 'Political']))
        diplomatic_events = int(len(self.df[self.df['event_category'] == 'Diplomatic']))
        economic_events = int(len(self.df[self.df['event_category'] == 'Economic / Trade']))
        protest_events = int(len(self.df[self.df['event_category'] == 'Protest / Civil Unrest']))
        
        high_risk_events = int(len(self.df[self.df['severity_level'].isin(['HIGH', 'CRITICAL'])]))
        critical_events = int(len(self.df[self.df['severity_level'] == 'CRITICAL']))
        countries_affected = int(self.df['event_country'].nunique())
        avg_severity = round(float(self.df['severity_score'].mean()), 1)

        # Sector frequency
        sector_counts = {}
        for sectors in self.df['affected_sectors']:
            for s in sectors:
                sector_counts[s] = sector_counts.get(s, 0) + 1

        top_sectors = sorted(
            [{'sector': k, 'count': v} for k, v in sector_counts.items()],
            key=lambda x: x['count'],
            reverse=True
        )

        return {
            'total_events': total_events,
            'conflict_events': conflict_events,
            'political_events': political_events,
            'diplomatic_events': diplomatic_events,
            'economic_events': economic_events,
            'protest_events': protest_events,
            'high_risk_events': high_risk_events,
            'critical_events': critical_events,
            'countries_affected': countries_affected,
            'average_severity': avg_severity,
            'top_affected_sectors': top_sectors[:5],
            'last_updated': datetime.now().isoformat()
        }

    def get_events(self, page=1, limit=20, country=None, category=None, severity=None, search=None, date_range=None):
        if self.df.empty:
            return {'events': [], 'total': 0, 'page': page, 'pages': 0}

        filtered = self.df.copy()

        if country and country.upper() != 'ALL':
            filtered = filtered[filtered['event_country'].str.upper() == country.upper()]

        if category and category.upper() != 'ALL':
            filtered = filtered[filtered['event_category'].str.lower() == category.lower()]

        if severity and severity.upper() != 'ALL':
            filtered = filtered[filtered['severity_level'].str.upper() == severity.upper()]

        if search:
            query = search.strip().lower()
            mask = (
                filtered['cluster_title'].str.lower().str.contains(query, na=False) |
                filtered['actor1'].str.lower().str.contains(query, na=False) |
                filtered['actor2'].str.lower().str.contains(query, na=False) |
                filtered['event_description'].str.lower().str.contains(query, na=False) |
                filtered['event_country_name'].str.lower().str.contains(query, na=False) |
                filtered['region'].str.lower().str.contains(query, na=False)
            )
            filtered = filtered[mask]

        if date_range:
            now = datetime(2026, 10, 4)
            if date_range == '7d':
                min_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
                filtered = filtered[filtered['date'] >= min_date]
            elif date_range == '30d':
                min_date = (now - timedelta(days=30)).strftime('%Y-%m-%d')
                filtered = filtered[filtered['date'] >= min_date]

        total = len(filtered)
        start = (page - 1) * limit
        end = start + limit
        
        # Sort descending by date and severity
        sorted_df = filtered.sort_values(by=['date', 'severity_score'], ascending=[False, False])
        paged_records = sorted_df.iloc[start:end].to_dict(orient='records')

        pages = (total + limit - 1) // limit if total > 0 else 1

        return {
            'events': paged_records,
            'total': total,
            'page': page,
            'limit': limit,
            'pages': pages,
            'filters_applied': {
                'country': country,
                'category': category,
                'severity': severity,
                'search': search,
                'date_range': date_range
            }
        }

    def get_event_by_id(self, event_id):
        if self.df.empty:
            return None
        match = self.df[self.df['event_id'] == int(event_id)]
        if not match.empty:
            record = match.iloc[0].to_dict()
            return record
        return None

    def get_clusters(self):
        return self.clusters

    def get_cluster_by_id(self, cluster_id):
        for c in self.clusters:
            if c['cluster_id'] == cluster_id:
                # Attach recent event samples
                samples = self.df[self.df['cluster_id'] == cluster_id].head(8).to_dict(orient='records')
                res = dict(c)
                res['recent_events'] = samples
                return res
        return None

    def get_timeline(self):
        if self.df.empty:
            return []
        
        # Aggregate daily counts by category
        timeline = (
            self.df.groupby(['date', 'event_category'])
            .size()
            .unstack(fill_value=0)
            .reset_index()
            .sort_values('date')
        )
        return timeline.to_dict(orient='records')

    def get_map_markers(self):
        markers = []
        for c in self.clusters:
            markers.append({
                'id': c['cluster_id'],
                'title': c['title'],
                'region': c['region'],
                'country': c['primary_country'],
                'country_name': c['country_name'],
                'latitude': c['latitude'],
                'longitude': c['longitude'],
                'severity_score': c['severity_score'],
                'severity_level': c['severity_level'],
                'total_events': c['total_events'],
                'conflict_events': c['conflict_events'],
                'trend': c['trend']['direction'],
                'affected_sectors': c['affected_sectors'],
            })
        return markers

    def get_country_intelligence(self, country_code):
        code = country_code.upper()
        country_df = self.df[self.df['event_country'] == code]
        
        if country_df.empty:
            country_df = self.df[self.df['actor1_country'] == code]

        if country_df.empty:
            return {
                'country_code': code,
                'country_name': code,
                'total_events': 0,
                'conflict_events': 0,
                'political_events': 0,
                'economic_events': 0,
                'diplomatic_events': 0,
                'average_severity': 0.0,
                'trend_pct': 0.0,
                'major_signals': ['Normal Baseline Activity']
            }

        country_name = country_df['event_country_name'].iloc[0]
        total = len(country_df)
        conflict_count = len(country_df[country_df['event_category'] == 'Conflict / Military'])
        political_count = len(country_df[country_df['event_category'] == 'Political'])
        economic_count = len(country_df[country_df['event_category'] == 'Economic / Trade'])
        diplomatic_count = len(country_df[country_df['event_category'] == 'Diplomatic'])
        avg_sev = round(float(country_df['severity_score'].mean()), 1)

        recent = len(country_df[country_df['date'] >= '2026-09-20'])
        earlier = len(country_df[country_df['date'] < '2026-09-20'])
        trend_pct = round(((recent - earlier) / max(1, earlier)) * 100.0, 1)

        signals = []
        if conflict_count > 5:
            signals.append('Elevated Regional Kinetic Tensions')
        if 'Energy' in [s for sublist in country_df['affected_sectors'] for s in sublist]:
            signals.append('Strategic Energy Flow Sensitivity')
        if 'Shipping' in [s for sublist in country_df['affected_sectors'] for s in sublist]:
            signals.append('Maritime Chokepoint Transit Exposure')
        if not signals:
            signals = ['Diplomatic Engagement', 'Bilateral Consultations']

        return {
            'country_code': code,
            'country_name': country_name,
            'total_events': total,
            'conflict_events': conflict_count,
            'political_events': political_count,
            'economic_events': economic_count,
            'diplomatic_events': diplomatic_count,
            'average_severity': avg_sev,
            'trend_pct': trend_pct,
            'major_signals': signals
        }

# Global singleton instance
event_service = EventService()
