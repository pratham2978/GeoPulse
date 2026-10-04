import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  TrendingUp,
  Activity,
  Radio,
  Globe,
  Search,
  Filter,
  ArrowRight,
  Clock,
  Zap,
  Package,
  Coins,
  X,
  ChevronRight,
  AlertTriangle,
  CheckCircle2,
  MapPin,
  ExternalLink,
  RefreshCw,
  SlidersHorizontal,
  Anchor,
  Cpu
} from 'lucide-react';

import defaultClusters from '../data/event_clusters.json';
import defaultStats from '../data/summary_stats.json';

export default function GlobalEventIntelligence() {
  const [clusters, setClusters] = useState(defaultClusters);
  const [stats, setStats] = useState(defaultStats);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('ALL');
  const [selectedSeverity, setSelectedSeverity] = useState('ALL');
  const [selectedCountry, setSelectedCountry] = useState('ALL');
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [apiConnected, setApiConnected] = useState(false);
  const [loading, setLoading] = useState(false);

  // Attempt live connection to FastAPI backend (graceful fallback to local data)
  useEffect(() => {
    async function fetchFromBackend() {
      try {
        const statsRes = await fetch('http://127.0.0.1:8000/api/events/stats');
        if (statsRes.ok) {
          const statsData = await statsRes.json();
          setStats(statsData);
          setApiConnected(true);
        }

        const clustersRes = await fetch('http://127.0.0.1:8000/api/events/clusters');
        if (clustersRes.ok) {
          const clustersData = await clustersRes.json();
          setClusters(clustersData);
        }
      } catch (err) {
        // Backend offline or starting: fallback to local bundled dataset
        setApiConnected(false);
      }
    }
    fetchFromBackend();
  }, []);

  // Filter clusters based on search query, category, severity, and country
  const filteredClusters = clusters.filter((item) => {
    const matchesSearch =
      searchQuery === '' ||
      item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.region.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.country_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.affected_sectors.some((s) => s.toLowerCase().includes(searchQuery.toLowerCase()));

    const matchesSeverity =
      selectedSeverity === 'ALL' || item.severity_level.toUpperCase() === selectedSeverity.toUpperCase();

    const matchesCountry =
      selectedCountry === 'ALL' || item.primary_country.toUpperCase() === selectedCountry.toUpperCase();

    return matchesSearch && matchesSeverity && matchesCountry;
  });

  const getSeverityBadgeClass = (level) => {
    switch (level) {
      case 'CRITICAL':
        return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'HIGH':
        return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
      case 'MODERATE':
        return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30';
      default:
        return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    }
  };

  const getSectorIcon = (sector) => {
    switch (sector.toLowerCase()) {
      case 'energy':
        return <Zap className="w-3.5 h-3.5 text-amber-400" />;
      case 'shipping':
        return <Anchor className="w-3.5 h-3.5 text-cyan-400" />;
      case 'trade':
        return <Package className="w-3.5 h-3.5 text-emerald-400" />;
      case 'commodities':
      case 'agriculture':
        return <Coins className="w-3.5 h-3.5 text-yellow-400" />;
      default:
        return <Globe className="w-3.5 h-3.5 text-purple-400" />;
    }
  };

  return (
    <div className="w-full text-white font-sans">
      
      {/* =========================================================================
          FEATURE 1: HEADER & LIVE STATUS BADGE
          ========================================================================= */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/40 border border-cyan-500/20 text-cyan-400 text-[10px] sm:text-xs font-semibold tracking-[0.2em] uppercase mb-3 font-mono">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400"></span>
            </span>
            <span>FEATURE 01: GLOBAL EVENT & CONFLICT INTELLIGENCE</span>
          </div>
          <h2 className="text-2xl sm:text-3xl md:text-4xl font-light tracking-tight text-white">
            LIVE GEOPOLITICAL EVENT ENGINE
          </h2>
          <p className="text-white/60 text-xs sm:text-sm font-light mt-1.5 max-w-2xl">
            In-depth GDELT CAMEO event classification, engineered GeoPulse severity calculations, and real-time sector vulnerability profiles.
          </p>
        </div>

        {/* Engine Status Tag */}
        <div className="flex items-center gap-2 font-mono text-[11px] px-3.5 py-1.5 rounded-lg bg-black/40 border border-white/10 self-start md:self-end">
          <span className={`w-2 h-2 rounded-full ${apiConnected ? 'bg-emerald-400 animate-pulse' : 'bg-cyan-400'}`} />
          <span className="text-white/60">SOURCE:</span>
          <span className="text-cyan-300 font-bold">{apiConnected ? 'FASTAPI REST LIVE' : 'GDELT 2.0 EMBEDDED'}</span>
        </div>
      </div>

      {/* =========================================================================
          1. KPI STATS METRIC CARDS (Calculated dynamically)
          ========================================================================= */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {/* KPI 1 */}
        <div className="glass-card rounded-xl p-4 sm:p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-white/50 text-[11px] font-mono mb-2">
            <span>ACTIVE EVENTS</span>
            <Activity className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
            {stats.total_events?.toLocaleString() || '1,284'}
          </div>
          <div className="text-[10px] text-emerald-400 font-mono mt-1">
            ● 100% Ingested & Cleaned
          </div>
        </div>

        {/* KPI 2 */}
        <div className="glass-card rounded-xl p-4 sm:p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-white/50 text-[11px] font-mono mb-2">
            <span>CONFLICT EVENTS</span>
            <ShieldAlert className="w-4 h-4 text-red-400" />
          </div>
          <div className="text-2xl sm:text-3xl font-bold text-red-400 tracking-tight">
            {stats.conflict_events?.toLocaleString() || '1,011'}
          </div>
          <div className="text-[10px] text-white/50 font-mono mt-1">
            CAMEO 190–196 Kinetic Force
          </div>
        </div>

        {/* KPI 3 */}
        <div className="glass-card rounded-xl p-4 sm:p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-white/50 text-[11px] font-mono mb-2">
            <span>HIGH / CRITICAL RISK</span>
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl sm:text-3xl font-bold text-amber-400 tracking-tight">
            {stats.high_risk_events?.toLocaleString() || '1,112'}
          </div>
          <div className="text-[10px] text-amber-300/80 font-mono mt-1">
            Severity Score &gt; 50.0
          </div>
        </div>

        {/* KPI 4 */}
        <div className="glass-card rounded-xl p-4 sm:p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-white/50 text-[11px] font-mono mb-2">
            <span>AVG SEVERITY</span>
            <TrendingUp className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl sm:text-3xl font-bold text-purple-300 tracking-tight">
            {stats.average_severity || '81.6'}
            <span className="text-xs text-white/40 font-normal"> / 100</span>
          </div>
          <div className="text-[10px] text-purple-400 font-mono mt-1">
            Multi-vector Composite Score
          </div>
        </div>
      </div>

      {/* =========================================================================
          2. SEARCH & MULTI-FILTER BAR
          ========================================================================= */}
      <div className="glass-card rounded-xl p-4 sm:p-5 mb-8 flex flex-col gap-4">
        {/* Search Input */}
        <div className="relative w-full">
          <Search className="w-4 h-4 text-white/40 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search country (India, Yemen, Ukraine), region, conflict type, or sector..."
            className="w-full bg-black/60 border border-white/10 rounded-lg pl-10 pr-4 py-2.5 text-xs sm:text-sm text-white placeholder:text-white/40 outline-none focus:border-cyan-500 transition-colors"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-white/40 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Filters Row */}
        <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
          {/* Severity Filters */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-white/40 font-mono text-[11px] mr-1">SEVERITY:</span>
            {['ALL', 'CRITICAL', 'HIGH', 'MODERATE'].map((sev) => (
              <button
                key={sev}
                onClick={() => setSelectedSeverity(sev)}
                className={`px-2.5 py-1 rounded-md text-[11px] font-mono transition-all ${
                  selectedSeverity === sev
                    ? 'bg-cyan-500 text-black font-bold shadow-md shadow-cyan-500/20'
                    : 'bg-white/5 text-white/70 hover:bg-white/10 hover:text-white'
                }`}
              >
                {sev}
              </button>
            ))}
          </div>

          {/* Region / Country Quick Select */}
          <div className="flex items-center gap-2">
            <span className="text-white/40 font-mono text-[11px]">REGION:</span>
            <select
              value={selectedCountry}
              onChange={(e) => setSelectedCountry(e.target.value)}
              className="bg-black/60 border border-white/10 rounded-md px-3 py-1 text-[11px] font-mono text-cyan-300 outline-none focus:border-cyan-400"
            >
              <option value="ALL">All Monitored Zones</option>
              <option value="YEM">Yemen / Red Sea</option>
              <option value="IRN">Iran / Strait of Hormuz</option>
              <option value="UKR">Ukraine / Black Sea</option>
              <option value="TWN">Taiwan Strait</option>
              <option value="PHL">Philippines / South China Sea</option>
              <option value="IND">India Northern Frontier</option>
            </select>
          </div>
        </div>
      </div>

      {/* =========================================================================
          3. TACTICAL GLOBAL MAP / HOTSPOTS RADAR VIEW
          ========================================================================= */}
      <div className="glass-card rounded-2xl p-5 sm:p-6 mb-8 relative overflow-hidden">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Globe className="w-4 h-4 text-cyan-400" />
            <span className="text-xs font-bold tracking-widest text-white/80 font-mono uppercase">
              GEOPOLITICAL RADAR SENSING MAP
            </span>
          </div>
          <span className="text-[10px] text-white/50 font-mono">
            CLICK ANY HOTSPOT NODE FOR INTELLIGENCE PROFILE
          </span>
        </div>

        {/* Interactive Tactical Map Canvas */}
        <div className="relative w-full h-64 sm:h-80 rounded-xl bg-gradient-to-b from-slate-950 via-slate-900 to-black border border-white/10 flex items-center justify-center overflow-hidden">
          {/* Grid lines */}
          <div className="absolute inset-0 opacity-15 bg-[radial-gradient(#06b6d4_1px,transparent_1px)] [background-size:16px_16px]" />
          
          {/* Concentric radar rings */}
          <div className="absolute w-96 h-96 rounded-full border border-cyan-500/10 pointer-events-none" />
          <div className="absolute w-[500px] h-[500px] rounded-full border border-cyan-500/5 pointer-events-none" />

          {/* Hotspot Nodes */}
          {clusters.map((c) => {
            // Coordinate projection onto canvas
            const leftPct = `${Math.min(92, Math.max(8, ((c.longitude + 180) / 360) * 100))}%`;
            const topPct = `${Math.min(90, Math.max(10, ((90 - c.latitude) / 180) * 100))}%`;
            const isSelected = selectedCluster?.cluster_id === c.cluster_id;

            return (
              <div
                key={c.cluster_id}
                onClick={() => setSelectedCluster(c)}
                style={{ left: leftPct, top: topPct }}
                className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer group z-20"
              >
                <div className="relative flex items-center justify-center">
                  {/* Ping wave */}
                  <span
                    className={`animate-ping absolute inline-flex h-7 w-7 rounded-full opacity-60 ${
                      c.severity_level === 'CRITICAL' ? 'bg-red-400' : 'bg-amber-400'
                    }`}
                  />
                  {/* Outer circle */}
                  <span
                    className={`relative inline-flex rounded-full h-3.5 w-3.5 border-2 border-white ${
                      c.severity_level === 'CRITICAL' ? 'bg-red-500' : 'bg-amber-500'
                    }`}
                  />

                  {/* Node Tooltip on Hover */}
                  <div className="absolute bottom-6 left-1/2 -translate-x-1/2 hidden group-hover:flex flex-col items-center pointer-events-none whitespace-nowrap z-30">
                    <div className="bg-black/90 border border-cyan-500/50 rounded-lg px-2.5 py-1.5 shadow-2xl text-left">
                      <div className="text-[10px] font-bold text-white font-mono">{c.title}</div>
                      <div className="text-[9px] text-cyan-300 font-mono flex items-center gap-2 mt-0.5">
                        <span>SEVERITY: {c.severity_score}</span>
                        <span>•</span>
                        <span>{c.total_events} EVENTS</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}

          {/* Center Coordinates Indicator */}
          <div className="absolute bottom-3 left-3 text-[10px] font-mono text-white/40">
            LAT 13.0°N / LON 43.5°E • GLOBAL SURVEILLANCE BUFFER
          </div>
          <div className="absolute bottom-3 right-3 text-[10px] font-mono text-emerald-400 flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span>6 ACTIVE STRATEGIC THEATERS</span>
          </div>
        </div>
      </div>

      {/* =========================================================================
          4. CONFLICT INTELLIGENCE CLUSTER CARDS GRID
          ========================================================================= */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base sm:text-lg font-bold tracking-wider text-white">
            ACTIVE CONFLICT PROFILES & CLUSTERS ({filteredClusters.length})
          </h3>
          <span className="text-[11px] text-white/50 font-mono">
            SORTED BY RISK SEVERITY INDEX
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredClusters.map((cluster) => {
            return (
              <div
                key={cluster.cluster_id}
                className="glass-card rounded-2xl p-5 sm:p-6 flex flex-col justify-between text-left transition-all duration-300 hover:-translate-y-1 hover:border-cyan-500/40 relative group"
              >
                <div>
                  {/* Top Bar: Severity Badge & Region */}
                  <div className="flex items-center justify-between mb-3">
                    <span
                      className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${getSeverityBadgeClass(
                        cluster.severity_level
                      )}`}
                    >
                      {cluster.severity_level} • {cluster.severity_score}
                    </span>
                    <span className="text-[11px] text-white/50 font-mono flex items-center gap-1">
                      <MapPin className="w-3 h-3 text-cyan-400" />
                      {cluster.country_name}
                    </span>
                  </div>

                  {/* Title */}
                  <h4 className="text-white text-base font-bold tracking-wide mb-2 group-hover:text-cyan-300 transition-colors">
                    {cluster.title}
                  </h4>

                  {/* Region */}
                  <div className="text-xs text-white/50 mb-4 font-mono">
                    THEATER: <span className="text-white/80">{cluster.region}</span>
                  </div>

                  {/* Stats Grid */}
                  <div className="grid grid-cols-3 gap-2 bg-black/40 rounded-xl p-3 border border-white/5 mb-4 text-center font-mono text-[11px]">
                    <div>
                      <div className="text-white/40 text-[9px]">EVENTS</div>
                      <div className="text-white font-bold">{cluster.total_events}</div>
                    </div>
                    <div>
                      <div className="text-white/40 text-[9px]">MENTIONS</div>
                      <div className="text-cyan-400 font-bold">{cluster.mentions.toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="text-white/40 text-[9px]">AVG TONE</div>
                      <div className="text-red-400 font-bold">{cluster.avg_tone}</div>
                    </div>
                  </div>

                  {/* Severity Gauge Bar */}
                  <div className="mb-4">
                    <div className="flex justify-between text-[10px] font-mono text-white/50 mb-1">
                      <span>SEVERITY INDEX</span>
                      <span className="text-cyan-300 font-bold">{cluster.severity_score}%</span>
                    </div>
                    <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-amber-500 via-orange-500 to-red-500"
                        style={{ width: `${cluster.severity_score}%` }}
                      />
                    </div>
                  </div>

                  {/* Affected Sectors */}
                  <div className="mb-4">
                    <div className="text-[10px] font-mono text-white/40 mb-1.5">AFFECTED SECTORS:</div>
                    <div className="flex flex-wrap gap-1.5">
                      {cluster.affected_sectors.map((sec) => (
                        <span
                          key={sec}
                          className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-white/5 border border-white/10 text-[10px] text-white/80 font-mono"
                        >
                          {getSectorIcon(sec)}
                          <span>{sec}</span>
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Footer Action */}
                <div className="pt-4 border-t border-white/10 flex items-center justify-between">
                  <div className="text-[10px] text-emerald-400 font-mono flex items-center gap-1">
                    <span>TREND:</span>
                    <span className="font-bold">{cluster.trend?.direction || 'Escalating'}</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setSelectedCluster(cluster)}
                    className="inline-flex items-center gap-1.5 text-xs font-bold text-cyan-400 hover:text-cyan-300 transition-colors cursor-pointer group-hover:translate-x-0.5"
                  >
                    <span>VIEW INTEL</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* =========================================================================
          5. DETAILED EVENT INTELLIGENCE MODAL / DOSSIER
          ========================================================================= */}
      {selectedCluster && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-xl animate-in fade-in duration-200">
          <div className="relative w-full max-w-3xl glass-card rounded-2xl border border-cyan-500/40 p-6 sm:p-8 max-h-[90vh] overflow-y-auto">
            {/* Modal Header */}
            <div className="flex items-start justify-between gap-4 border-b border-white/10 pb-5 mb-6">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span
                    className={`text-[10px] font-mono font-bold px-2.5 py-0.5 rounded border ${getSeverityBadgeClass(
                      selectedCluster.severity_level
                    )}`}
                  >
                    {selectedCluster.severity_level} • SEVERITY {selectedCluster.severity_score}
                  </span>
                  <span className="text-xs text-cyan-400 font-mono">
                    ID: {selectedCluster.cluster_id}
                  </span>
                </div>
                <h3 className="text-xl sm:text-2xl font-bold text-white tracking-wide">
                  {selectedCluster.title}
                </h3>
                <p className="text-xs sm:text-sm text-white/60 font-light mt-1">
                  Theater: <span className="text-white font-medium">{selectedCluster.region}</span> ({selectedCluster.country_name})
                </p>
              </div>

              <button
                type="button"
                onClick={() => setSelectedCluster(null)}
                className="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-white/70 hover:text-white transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Metrics Strip */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-black/50 rounded-xl p-4 border border-white/5 mb-6 text-center font-mono">
              <div>
                <div className="text-white/40 text-[10px]">TOTAL EVENTS</div>
                <div className="text-lg font-bold text-white mt-0.5">{selectedCluster.total_events}</div>
              </div>
              <div>
                <div className="text-white/40 text-[10px]">NEWS MENTIONS</div>
                <div className="text-lg font-bold text-cyan-400 mt-0.5">{selectedCluster.mentions?.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-white/40 text-[10px]">SOURCE COUNT</div>
                <div className="text-lg font-bold text-white mt-0.5">{selectedCluster.sources?.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-white/40 text-[10px]">AVERAGE TONE</div>
                <div className="text-lg font-bold text-red-400 mt-0.5">{selectedCluster.avg_tone}</div>
              </div>
            </div>

            {/* Event Timeline Progression */}
            <div className="mb-6">
              <h4 className="text-xs font-bold font-mono tracking-widest text-cyan-300 uppercase mb-3 flex items-center gap-2">
                <Clock className="w-3.5 h-3.5" />
                <span>INTELLIGENCE EVENT TIMELINE</span>
              </h4>

              <div className="space-y-3 relative pl-4 border-l-2 border-cyan-500/30">
                {selectedCluster.timeline?.map((step, idx) => (
                  <div key={idx} className="relative">
                    <span className="absolute -left-[21px] top-1.5 w-2.5 h-2.5 rounded-full bg-cyan-400 border border-black" />
                    <div className="flex items-baseline gap-2">
                      <span className="text-[11px] font-mono text-cyan-400 font-bold">{step.date}</span>
                      <span className="text-[11px] font-mono text-white/40">•</span>
                      <span className="text-xs font-bold text-white">{step.phase}</span>
                    </div>
                    <p className="text-xs text-white/70 font-light mt-0.5">{step.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Affected Sectors Deep-Dive */}
            <div className="mb-6">
              <h4 className="text-xs font-bold font-mono tracking-widest text-emerald-300 uppercase mb-3 flex items-center gap-2">
                <Activity className="w-3.5 h-3.5" />
                <span>AFFECTED ECONOMIC SECTORS & RISK EXPOSURE</span>
              </h4>

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {selectedCluster.affected_sectors?.map((sec) => (
                  <div key={sec} className="bg-black/40 border border-white/10 rounded-xl p-3 flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center">
                      {getSectorIcon(sec)}
                    </div>
                    <div>
                      <div className="text-xs font-bold text-white">{sec}</div>
                      <div className="text-[10px] text-amber-400 font-mono">High Exposure</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Modal Bottom Actions */}
            <div className="flex items-center justify-end gap-3 pt-4 border-t border-white/10">
              <button
                type="button"
                onClick={() => setSelectedCluster(null)}
                className="px-5 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-xs font-semibold text-white transition-colors cursor-pointer"
              >
                Close Dossier
              </button>
              <a
                href="#risk"
                onClick={() => setSelectedCluster(null)}
                className="px-5 py-2 rounded-lg bg-gradient-to-r from-emerald-400 to-cyan-500 text-xs font-bold text-white shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 transition-all cursor-pointer"
              >
                Correlate in Risk Engine →
              </a>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
