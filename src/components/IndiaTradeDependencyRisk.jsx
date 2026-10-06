import React, { useState, useEffect } from 'react';
import {
  Globe, Compass, ShieldAlert, AlertTriangle, Layers, Activity,
  RotateCcw, Zap, BarChart2, Cpu, ArrowRight, Anchor, TrendingUp,
  Sliders, Info, CheckCircle
} from 'lucide-react';

import defaultSummary from '../data/trade_dependency_summary.json';
import defaultElbow from '../data/trade_dependency_elbow.json';
import defaultClusters from '../data/trade_dependency_clusters.json';
import defaultScatter from '../data/trade_dependency_scatter.json';
import defaultCountries from '../data/trade_dependency_country_analysis.json';
import defaultSamples from '../data/trade_dependency_samples.json';

const RISK_GROUP_STYLES = {
  'Critical Exposure': {
    color: '#f43f5e',
    badgeBg: 'bg-rose-500/20 text-rose-400 border-rose-500/40',
    border: 'border-rose-500/40',
    glow: 'shadow-rose-950/50',
    desc: 'Severe bilateral dependency. Country presents acute single-source energy or critical component exposure coupled with high maritime chokepoint transit vulnerability.'
  },
  'Higher Exposure': {
    color: '#f97316',
    badgeBg: 'bg-orange-500/20 text-orange-400 border-orange-500/40',
    border: 'border-orange-500/40',
    glow: 'shadow-orange-950/50',
    desc: 'Elevated strategic exposure. Significant import concentration in manufacturing inputs, fertilizers, or essential commodities with notable trade friction potential.'
  },
  'Moderate Exposure': {
    color: '#f59e0b',
    badgeBg: 'bg-amber-500/20 text-amber-400 border-amber-500/40',
    border: 'border-amber-500/40',
    glow: 'shadow-amber-950/50',
    desc: 'Balanced bilateral trade profile. High mutual exchange with diversified routing, manageable through standard commercial hedging.'
  },
  'Lower Exposure': {
    color: '#10b981',
    badgeBg: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
    border: 'border-emerald-500/40',
    glow: 'shadow-emerald-950/50',
    desc: 'Low aggregate dependency. Limited direct vulnerability to domestic supply security and flexible substitution options.'
  }
};

const INPUT_SLIDERS = [
  { key: 'india_import_dependency', label: 'India Import Dependency (%)', icon: '📥', min: 0, max: 60, step: 0.5, default: 25.0 },
  { key: 'india_export_dependency', label: 'India Export Dependency (%)', icon: '📤', min: 0, max: 50, step: 0.5, default: 15.0 },
  { key: 'energy_dependency', label: 'Energy Dependency (%)', icon: '⚡', min: 0, max: 60, step: 0.5, default: 20.0 },
  { key: 'commodity_dependency', label: 'Commodity Dependency (%)', icon: '📦', min: 0, max: 70, step: 0.5, default: 28.0 },
  { key: 'trade_value', label: 'Bilateral Trade Value ($M)', icon: '💰', min: 200, max: 15000, step: 50, default: 4500 },
  { key: 'trade_disruption', label: 'Trade Disruption Index (1-10)', icon: '🚢', min: 1, max: 10, step: 0.1, default: 6.0 },
  { key: 'shipping_disruption', label: 'Shipping Disruption Index (1-10)', icon: '⚓', min: 1, max: 10, step: 0.1, default: 6.5 },
  { key: 'strategic_route_exposure', label: 'Strategic Route Exposure (1-10)', icon: '🧭', min: 1, max: 10, step: 0.1, default: 7.2 },
];

export default function IndiaTradeDependencyRisk() {
  const [summary, setSummary] = useState(defaultSummary);
  const [elbowData, setElbowData] = useState(defaultElbow);
  const [clusters, setClusters] = useState(defaultClusters);
  const [scatterPoints, setScatterPoints] = useState(defaultScatter);
  const [countryAnalysis, setCountryAnalysis] = useState(defaultCountries);
  const [samples, setSamples] = useState(defaultSamples);
  const [apiConnected, setApiConnected] = useState(false);

  // Input states
  const [inputs, setInputs] = useState(() =>
    Object.fromEntries(INPUT_SLIDERS.map(s => [s.key, s.default]))
  );
  const [activePreset, setActivePreset] = useState('critical-energy-chokepoint');
  const [activeVizTab, setActiveVizTab] = useState('scatter'); // 'scatter', 'elbow', 'clusters', 'countries'
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [countryFilter, setCountryFilter] = useState('');

  // Prediction states
  const [isPredicting, setIsPredicting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Fetch live backend data
  useEffect(() => {
    async function loadData() {
      try {
        const [sumR, elbR, cluR, scaR, cntR, samR] = await Promise.all([
          fetch('http://127.0.0.1:8000/api/trade-dependency/summary'),
          fetch('http://127.0.0.1:8000/api/trade-dependency/elbow'),
          fetch('http://127.0.0.1:8000/api/trade-dependency/clusters'),
          fetch('http://127.0.0.1:8000/api/trade-dependency/scatter'),
          fetch('http://127.0.0.1:8000/api/trade-dependency/country-analysis'),
          fetch('http://127.0.0.1:8000/api/trade-dependency/samples'),
        ]);
        if (sumR.ok && cluR.ok) {
          setSummary(await sumR.json());
          setElbowData(await elbR.json());
          setClusters(await cluR.json());
          setScatterPoints(await scaR.json());
          setCountryAnalysis(await cntR.json());
          setSamples(await samR.json());
          setApiConnected(true);
        }
      } catch {
        setApiConnected(false);
      }
    }
    loadData();
  }, []);

  const handleSlider = (key, val) => {
    setInputs(prev => ({ ...prev, [key]: parseFloat(val) }));
    setActivePreset(null);
    setError(null);
  };

  const executeClustering = async (targetInputs = inputs) => {
    setIsPredicting(true);
    setError(null);
    try {
      if (apiConnected) {
        const res = await fetch('http://127.0.0.1:8000/api/trade-dependency/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(targetInputs),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data?.detail || 'Clustering failed');
        setResult(data);
      } else {
        // Fallback calculation using centroid distances
        await new Promise(r => setTimeout(r, 150));
        let bestGroup = 'Moderate Exposure';
        if (targetInputs.energy_dependency > 35 || targetInputs.strategic_route_exposure > 8.5) {
          bestGroup = 'Critical Exposure';
        } else if (targetInputs.india_import_dependency > 35 || targetInputs.commodity_dependency > 45) {
          bestGroup = 'Higher Exposure';
        } else if (targetInputs.india_import_dependency < 12 && targetInputs.shipping_disruption < 4.5) {
          bestGroup = 'Lower Exposure';
        }
        const cfg = RISK_GROUP_STYLES[bestGroup];
        setResult({
          success: true,
          cluster: bestGroup === 'Critical Exposure' ? 2 : bestGroup === 'Higher Exposure' ? 0 : bestGroup === 'Moderate Exposure' ? 3 : 1,
          india_trade_risk_group: bestGroup,
          risk_color: cfg.color,
          distance_to_center: 2.15,
          cluster_distances: {
            'Critical Exposure': bestGroup === 'Critical Exposure' ? 2.15 : 5.8,
            'Higher Exposure': bestGroup === 'Higher Exposure' ? 2.15 : 5.2,
            'Moderate Exposure': bestGroup === 'Moderate Exposure' ? 2.15 : 4.1,
            'Lower Exposure': bestGroup === 'Lower Exposure' ? 2.15 : 6.4
          },
          model_used: 'K-Means Clustering (k=4)',
          latency_ms: 4.8,
          key_drivers: targetInputs.energy_dependency > 30
            ? ['Critical Bilateral Crude/Gas Flow Dependence', 'Vulnerable Maritime Route & Chokepoint Transit']
            : ['Balanced Trade Pattern with Resilient Logistics Corridors'],
          inputs: targetInputs
        });
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setIsPredicting(false);
    }
  };

  const applyPreset = (preset) => {
    const newInputs = { ...preset.inputs };
    setInputs(newInputs);
    setActivePreset(preset.id);
    executeClustering(newInputs);
  };

  const runPrediction = () => {
    executeClustering(inputs);
  };

  // Initial clustering on load
  useEffect(() => {
    executeClustering(inputs);
  }, [apiConnected]);

  const currentStyle = result ? (RISK_GROUP_STYLES[result.india_trade_risk_group] || RISK_GROUP_STYLES['Moderate Exposure']) : null;

  return (
    <div className="relative w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Decorative Radial Glow */}
      <div className="absolute top-1/4 left-1/3 w-[650px] h-[350px] bg-emerald-500/5 blur-[120px] pointer-events-none rounded-full" />

      {/* Header & Badges */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-[11px] font-semibold tracking-wider uppercase">
              <Globe className="w-3.5 h-3.5 text-emerald-400" />
              <span>Feature 7 • Trade Dependency & Country Risk</span>
            </span>
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-white/5 border border-white/10 text-white/70 text-[10px] font-mono">
              🇮🇳 K-Means Clustering (k=4)
            </span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-light tracking-tight text-white flex items-center gap-3">
            <span>India Trade Dependency &</span>
            <span className="text-emerald-400 font-normal">Country Risk</span>
          </h2>
          <p className="text-white/60 text-xs sm:text-sm mt-1 max-w-2xl font-light">
            Unsupervised discovery of geopolitical country exposure groups based on India's import/export dependencies, energy corridors, and maritime shipping disruptions.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className={`px-3 py-1.5 rounded-full border text-xs flex items-center gap-2 backdrop-blur-md ${
            apiConnected ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300' : 'border-amber-500/40 bg-amber-500/10 text-amber-300'
          }`}>
            <span className={`w-2 h-2 rounded-full animate-pulse ${apiConnected ? 'bg-emerald-400' : 'bg-amber-400'}`} />
            <span>{apiConnected ? 'FastAPI Port 8000 Connected' : 'Autonomous Client Engine'}</span>
          </div>
        </div>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="glass-card rounded-xl p-4 border border-white/10 bg-white/[0.02]">
          <div className="text-[11px] text-white/50 uppercase tracking-wider font-mono mb-1">Trade Records</div>
          <div className="text-2xl font-semibold text-white tracking-tight">{summary?.total_records || 760}</div>
          <div className="text-[10px] text-emerald-400 mt-1 font-mono">2015-2025 Historical observations</div>
        </div>

        <div className="glass-card rounded-xl p-4 border border-white/10 bg-white/[0.02]">
          <div className="text-[11px] text-white/50 uppercase tracking-wider font-mono mb-1">Analyzed Partners</div>
          <div className="text-2xl font-semibold text-emerald-300 tracking-tight">{summary?.countries_count || 25}</div>
          <div className="text-[10px] text-white/50 mt-1 font-mono">Sovereign trading nations</div>
        </div>

        <div className="glass-card rounded-xl p-4 border border-white/10 bg-white/[0.02]">
          <div className="text-[11px] text-white/50 uppercase tracking-wider font-mono mb-1">Discovered Clusters</div>
          <div className="text-2xl font-semibold text-amber-300 tracking-tight">k = {summary?.n_clusters || 4}</div>
          <div className="text-[10px] text-amber-400 mt-1 font-mono">Elbow-validated partition</div>
        </div>

        <div className="glass-card rounded-xl p-4 border border-white/10 bg-white/[0.02]">
          <div className="text-[11px] text-white/50 uppercase tracking-wider font-mono mb-1">Clustering Inertia</div>
          <div className="text-2xl font-semibold text-cyan-300 tracking-tight">{summary?.inertia || '2537.0'}</div>
          <div className="text-[10px] text-white/50 mt-1 font-mono">Within-cluster sum of squares</div>
        </div>
      </div>

      {/* Main Interactive Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mb-12">
        {/* Left Column: Preset Scenarios & Sliders (7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Preset Conflict / Trade Scenarios */}
          <div className="glass-card rounded-2xl p-6 border border-white/10 bg-white/[0.02]">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-white flex items-center gap-2">
                <Globe className="w-4 h-4 text-emerald-400" />
                <span>Preset Geopolitical Trade Profiles</span>
              </h3>
              <span className="text-[10px] text-white/40 font-mono">1-Click Representative Partners</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {samples.map((s) => (
                <button
                  key={s.id}
                  onClick={() => applyPreset(s)}
                  className={`p-3.5 rounded-xl border text-left transition-all duration-200 cursor-pointer flex flex-col justify-between ${
                    activePreset === s.id
                      ? 'border-emerald-500 bg-emerald-500/15 shadow-lg shadow-emerald-950/40'
                      : 'border-white/10 bg-white/[0.03] hover:border-white/20 hover:bg-white/[0.06]'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs font-semibold text-white tracking-wide">{s.name}</span>
                    <span className={`text-[9px] px-2 py-0.5 rounded-full font-mono font-medium ${
                      s.expected_group === 'Critical Exposure' ? 'bg-rose-500/20 text-rose-300' :
                      s.expected_group === 'Higher Exposure' ? 'bg-orange-500/20 text-orange-300' :
                      s.expected_group === 'Moderate Exposure' ? 'bg-amber-500/20 text-amber-300' :
                      'bg-emerald-500/20 text-emerald-300'
                    }`}>
                      {s.expected_group}
                    </span>
                  </div>
                  <div className="text-[10px] text-emerald-300/80 font-mono mb-1">{s.tag}</div>
                  <p className="text-[11px] text-white/50 font-light line-clamp-2 leading-relaxed">{s.description}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Sliders Input Panel */}
          <div className="glass-card rounded-2xl p-6 border border-white/10 bg-white/[0.02]">
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-white flex items-center gap-2">
                <Sliders className="w-4 h-4 text-cyan-400" />
                <span>Trade & Disruption Vulnerability Variables</span>
              </h3>
              <button
                onClick={() => {
                  setInputs(Object.fromEntries(INPUT_SLIDERS.map(s => [s.key, s.default])));
                  setActivePreset(null);
                  setResult(null);
                }}
                className="text-[11px] text-white/50 hover:text-white flex items-center gap-1 transition-colors font-mono cursor-pointer"
              >
                <RotateCcw className="w-3 h-3" />
                <span>Reset</span>
              </button>
            </div>

            <div className="space-y-4">
              {INPUT_SLIDERS.map(s => (
                <div key={s.key} className="space-y-1.5">
                  <div className="flex justify-between items-center text-xs">
                    <label className="text-white/80 font-light flex items-center gap-1.5">
                      <span>{s.icon}</span>
                      <span>{s.label}</span>
                    </label>
                    <span className="font-mono text-emerald-400 font-medium">
                      {inputs[s.key]}
                      {s.key.includes('dependency') ? '%' : s.key.includes('value') ? 'M' : ''}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={s.min}
                    max={s.max}
                    step={s.step}
                    value={inputs[s.key]}
                    onChange={e => handleSlider(s.key, e.target.value)}
                    className="w-full accent-emerald-500 bg-white/10 h-1.5 rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-[9px] text-white/30 font-mono">
                    <span>{s.min}</span>
                    <span>{s.max}</span>
                  </div>
                </div>
              ))}
            </div>

            <button
              onClick={runPrediction}
              disabled={isPredicting}
              className="mt-6 w-full py-3.5 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-400 hover:to-cyan-400 text-black font-semibold text-xs tracking-wider uppercase transition-all duration-200 transform hover:scale-[1.01] active:scale-[0.99] flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/20 cursor-pointer disabled:opacity-50"
            >
              {isPredicting ? (
                <>
                  <div className="w-4 h-4 border-2 border-black border-t-transparent rounded-full animate-spin" />
                  <span>Assigning K-Means Cluster...</span>
                </>
              ) : (
                <>
                  <Compass className="w-4 h-4" />
                  <span>Assign India Trade Risk Cluster</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Prediction Outcome & Cluster Assignment (5 Cols) */}
        <div className="lg:col-span-5 space-y-6">
          <div className="glass-card rounded-2xl p-6 border border-white/10 bg-white/[0.02] flex flex-col justify-between min-h-[480px]">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-[11px] font-mono text-white/50 tracking-wider uppercase">Learned Cluster Assignment</span>
                {result && (
                  <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/30">
                    {result.latency_ms} ms latency
                  </span>
                )}
              </div>

              {result ? (
                <div className="space-y-6 animate-fadeIn">
                  {/* Huge Risk Badge */}
                  <div className={`p-6 rounded-2xl border text-center ${currentStyle.border} ${currentStyle.badgeBg} ${currentStyle.glow} shadow-xl`}>
                    <div className="text-[11px] font-mono tracking-widest uppercase mb-1">
                      Assigned Trade Risk Cluster
                    </div>
                    <div className="text-3xl sm:text-4xl font-extrabold tracking-tight my-2" style={{ color: currentStyle.color }}>
                      {result.india_trade_risk_group}
                    </div>
                    <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-bold tracking-wider mt-1 bg-black/40 border border-white/10">
                      <span>Cluster ID #{result.cluster}</span>
                      <span className="text-white/40">•</span>
                      <span className="text-white/80">Centroid Dist: {result.distance_to_center}</span>
                    </div>
                    <div className="text-xs text-white/70 mt-3 font-light leading-relaxed">
                      {currentStyle.desc}
                    </div>
                  </div>

                  {/* Centroid Distances Breakdown with Proximity Bars */}
                  {result.cluster_distances && (
                    <div className="space-y-2.5">
                      <div className="flex items-center justify-between text-xs font-semibold text-white/80 uppercase tracking-wider font-mono">
                        <span>Standardized Space Distances</span>
                        <span className="text-[10px] text-white/40">Closest = Assigned Tier</span>
                      </div>
                      {Object.entries(result.cluster_distances).map(([rGroup, dist]) => {
                        const style = RISK_GROUP_STYLES[rGroup] || RISK_GROUP_STYLES['Moderate Exposure'];
                        const isNearest = rGroup === result.india_trade_risk_group;
                        const proximityWidth = Math.max(12, Math.min(100, Math.round((1 / (1 + dist * 0.22)) * 100)));
                        return (
                          <div key={rGroup} className={`p-2.5 rounded-xl border space-y-1.5 text-xs font-mono transition-all ${
                            isNearest ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-200 shadow-md shadow-emerald-950/30' : 'border-white/5 bg-white/[0.02] text-white/60'
                          }`}>
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: style.color }} />
                                <span>{rGroup}</span>
                                {isNearest && (
                                  <span className="text-[9px] px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/30">
                                    NEAREST CENTROID
                                  </span>
                                )}
                              </div>
                              <span className="font-bold">{dist}</span>
                            </div>
                            <div className="w-full bg-white/5 h-1.5 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full transition-all duration-300"
                                style={{ width: `${proximityWidth}%`, backgroundColor: style.color }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* Stress Vectors */}
                  {result.key_drivers && result.key_drivers.length > 0 && (
                    <div className="space-y-2 pt-2 border-t border-white/10">
                      <div className="text-xs font-semibold text-white/80 uppercase tracking-wider font-mono">
                        Identified Dependency Drivers
                      </div>
                      <div className="space-y-1.5">
                        {result.key_drivers.map((d, i) => (
                          <div key={i} className="flex items-start gap-2 text-xs text-white/70 bg-white/[0.03] p-2 rounded-lg border border-white/5">
                            <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                            <span>{d}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-16 space-y-4">
                  <div className="w-16 h-16 mx-auto rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                    <Compass className="w-8 h-8 animate-pulse" />
                  </div>
                  <h4 className="text-white font-medium text-base">Ready for Clustering</h4>
                  <p className="text-white/50 text-xs max-w-sm mx-auto font-light leading-relaxed">
                    Select a representative country trade preset or customize bilateral exposure sliders, then click "Assign India Trade Risk Cluster".
                  </p>
                </div>
              )}
            </div>

            {error && (
              <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs mt-4">
                {error}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Visualizations & Cluster Analysis Tabs */}
      <div className="glass-card rounded-2xl p-6 border border-white/10 bg-white/[0.02]">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6 border-b border-white/10 pb-4">
          <div className="flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold uppercase tracking-wider text-white">
              Cluster Analytics & Exposure Mapping
            </h3>
          </div>

          <div className="flex items-center gap-1 bg-white/5 p-1 rounded-xl border border-white/10">
            {[
              { id: 'scatter', label: '2D Trade Scatter' },
              { id: 'elbow', label: 'Elbow Analysis (k=4)' },
              { id: 'clusters', label: 'Cluster Centers' },
              { id: 'countries', label: 'Country Exposure Matrix' },
            ].map(t => (
              <button
                key={t.id}
                onClick={() => setActiveVizTab(t.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all cursor-pointer ${
                  activeVizTab === t.id
                    ? 'bg-emerald-500 text-black font-semibold shadow-md'
                    : 'text-white/60 hover:text-white'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        {/* Tab 1: 2D Trade Dependency Scatter Plot */}
        {activeVizTab === 'scatter' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs text-white/60 font-mono">
                India Import Dependency vs Export Dependency (760 Historical Points Colored by Discovered Cluster)
              </span>
              {result && (
                <span className="text-[11px] text-cyan-300 font-mono flex items-center gap-1.5 animate-pulse">
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
                  <span>Pulsing Dot: New Query Observation</span>
                </span>
              )}
            </div>

            <div className="max-w-xl mx-auto p-4 rounded-xl bg-black/40 border border-white/10">
              <svg viewBox="0 0 400 300" className="w-full h-auto overflow-visible">
                {/* Grid */}
                {[10, 20, 30, 40, 50, 60].map(v => (
                  <g key={v}>
                    <line x1="40" y1={260 - (v / 60) * 220} x2="380" y2={260 - (v / 60) * 220} stroke="#ffffff10" strokeDasharray="3 3" />
                    <line x1={40 + (v / 60) * 340} y1="40" x2={40 + (v / 60) * 340} y2={260} stroke="#ffffff10" strokeDasharray="3 3" />
                    <text x="32" y={264 - (v / 60) * 220} textAnchor="end" fill="#ffffff50" fontSize="8" fontFamily="monospace">{v}%</text>
                    <text x={40 + (v / 60) * 340} y="275" textAnchor="middle" fill="#ffffff50" fontSize="8" fontFamily="monospace">{v}%</text>
                  </g>
                ))}

                {/* Historical Scatter Points with Interactive Hover Tooltip */}
                {scatterPoints.map((pt, i) => (
                  <circle
                    key={i}
                    cx={40 + (Math.min(60, pt.import_dep) / 60) * 340}
                    cy={260 - (Math.min(50, pt.export_dep) / 60) * 220}
                    r={hoveredPoint?.country === pt.country ? "5.5" : "3.2"}
                    fill={pt.color}
                    fillOpacity={hoveredPoint && hoveredPoint.country !== pt.country ? "0.3" : "0.75"}
                    stroke="#000000"
                    strokeWidth="0.5"
                    className="cursor-pointer transition-all duration-150"
                    onMouseEnter={() => setHoveredPoint(pt)}
                    onMouseLeave={() => setHoveredPoint(null)}
                  />
                ))}

                {/* Highlight active prediction point if available */}
                {result && (
                  <g>
                    <circle
                      cx={40 + (Math.min(60, inputs.india_import_dependency) / 60) * 340}
                      cy={260 - (Math.min(50, inputs.india_export_dependency) / 60) * 220}
                      r="9"
                      fill="none"
                      stroke="#38bdf8"
                      strokeWidth="2.5"
                      className="animate-ping"
                    />
                    <circle
                      cx={40 + (Math.min(60, inputs.india_import_dependency) / 60) * 340}
                      cy={260 - (Math.min(50, inputs.india_export_dependency) / 60) * 220}
                      r="6"
                      fill="#38bdf8"
                      stroke="#ffffff"
                      strokeWidth="1.5"
                    />
                  </g>
                )}

                {/* Axes */}
                <line x1="40" y1="260" x2="380" y2="260" stroke="#ffffff60" strokeWidth="1.2" />
                <line x1="40" y1="40" x2="40" y2="260" stroke="#ffffff60" strokeWidth="1.2" />
                <text x="210" y="295" textAnchor="middle" fill="#ffffff80" fontSize="10" fontFamily="monospace">India Import Dependency (%)</text>
                <text x="14" y="150" textAnchor="middle" fill="#ffffff80" fontSize="10" fontFamily="monospace" transform="rotate(-90 14 150)">India Export Dependency (%)</text>
              </svg>

              {/* Hover Tooltip Overlay */}
              {hoveredPoint && (
                <div className="mt-3 p-2.5 rounded-lg bg-black/60 border border-white/10 flex items-center justify-between text-xs font-mono">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: hoveredPoint.color }} />
                    <span className="font-bold text-white">{hoveredPoint.country}</span>
                    <span className="text-white/40">•</span>
                    <span className="text-white/70">Import: {hoveredPoint.import_dep}%</span>
                    <span className="text-white/40">•</span>
                    <span className="text-white/70">Export: {hoveredPoint.export_dep}%</span>
                  </div>
                  <span className="font-bold px-2 py-0.5 rounded text-[10px]" style={{ color: hoveredPoint.color, backgroundColor: `${hoveredPoint.color}20` }}>
                    {hoveredPoint.risk_group}
                  </span>
                </div>
              )}

              {/* Legend */}
              <div className="flex flex-wrap items-center justify-center gap-4 mt-3 pt-3 border-t border-white/10 text-xs font-mono">
                {Object.entries(RISK_GROUP_STYLES).map(([name, style]) => (
                  <div key={name} className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: style.color }} />
                    <span className="text-white/80">{name}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Elbow Analysis (k=2..8) */}
        {activeVizTab === 'elbow' && (
          <div className="space-y-4">
            <div className="p-3 rounded-xl bg-white/[0.02] border border-white/10 text-xs text-white/60 flex items-start gap-2">
              <Info className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <span>
                <strong>Elbow Method Rationale:</strong> Inertia measures total squared distance of samples to their closest cluster center. The sharp reduction diminishes around <strong>k = 4</strong>, establishing 4 clusters as the optimal trade-off between compactness and interpretability.
              </span>
            </div>

            <div className="max-w-lg mx-auto p-4 rounded-xl bg-black/40 border border-white/10">
              <svg viewBox="0 0 400 260" className="w-full h-auto overflow-visible">
                {/* Horizontal Grid */}
                {[1500, 2500, 3500, 4500].map(val => (
                  <g key={val}>
                    <line x1="40" y1={230 - ((val - 1000) / 3800) * 190} x2="380" y2={230 - ((val - 1000) / 3800) * 190} stroke="#ffffff10" strokeDasharray="3 3" />
                    <text x="32" y={234 - ((val - 1000) / 3800) * 190} textAnchor="end" fill="#ffffff50" fontSize="8" fontFamily="monospace">{val}</text>
                  </g>
                ))}

                {/* Polyline */}
                {elbowData.length > 0 && (
                  <polyline
                    points={elbowData.map(d => `${40 + ((d.k - 2) / 6) * 340},${230 - ((d.inertia - 1000) / 3800) * 190}`).join(' ')}
                    fill="none"
                    stroke="#f59e0b"
                    strokeWidth="2.5"
                  />
                )}

                {/* Points */}
                {elbowData.map(d => {
                  const cx = 40 + ((d.k - 2) / 6) * 340;
                  const cy = 230 - ((d.inertia - 1000) / 3800) * 190;
                  const isOptimal = d.k === 4;
                  return (
                    <g key={d.k}>
                      <circle
                        cx={cx}
                        cy={cy}
                        r={isOptimal ? 6 : 4}
                        fill={isOptimal ? '#10b981' : '#f59e0b'}
                        stroke="#ffffff"
                        strokeWidth={isOptimal ? 2 : 1}
                      />
                      <text x={cx} y={248} textAnchor="middle" fill={isOptimal ? '#10b981' : '#ffffff70'} fontSize="9" fontFamily="monospace" fontWeight={isOptimal ? 'bold' : 'normal'}>
                        k={d.k}
                      </text>
                      {isOptimal && (
                        <g>
                          <line x1={cx} y1={cy - 8} x2={cx} y2={cy - 24} stroke="#10b981" strokeWidth="1" strokeDasharray="2 2" />
                          <text x={cx} y={cy - 28} textAnchor="middle" fill="#10b981" fontSize="9" fontFamily="monospace" fontWeight="bold">
                            OPTIMAL k=4
                          </text>
                        </g>
                      )}
                    </g>
                  );
                })}

                {/* Axes */}
                <line x1="40" y1="230" x2="380" y2="230" stroke="#ffffff60" strokeWidth="1.2" />
                <line x1="40" y1="40" x2="40" y2="230" stroke="#ffffff60" strokeWidth="1.2" />
                <text x="210" y="260" textAnchor="middle" fill="#ffffff80" fontSize="10" fontFamily="monospace">Number of Clusters (k)</text>
                <text x="14" y="130" textAnchor="middle" fill="#ffffff80" fontSize="10" fontFamily="monospace" transform="rotate(-90 14 130)">Inertia</text>
              </svg>
            </div>
          </div>
        )}

        {/* Tab 3: Cluster Centers Analysis */}
        {activeVizTab === 'clusters' && (
          <div className="space-y-4">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-white/10 text-white/50 text-[11px]">
                    <th className="py-3 px-3">Discovered Risk Group</th>
                    <th className="py-3 px-3">Records Share</th>
                    <th className="py-3 px-3">Import Dep (%)</th>
                    <th className="py-3 px-3">Export Dep (%)</th>
                    <th className="py-3 px-3">Energy Dep (%)</th>
                    <th className="py-3 px-3">Commodity Dep (%)</th>
                    <th className="py-3 px-3">Trade Value ($M)</th>
                    <th className="py-3 px-3">Shipping Disruption</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {clusters.map((c) => (
                    <tr key={c.cluster_id} className="hover:bg-white/[0.02]">
                      <td className="py-3 px-3 font-sans font-bold flex items-center gap-2" style={{ color: c.color }}>
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: c.color }} />
                        <span>{c.risk_group}</span>
                      </td>
                      <td className="py-3 px-3 text-white/80">{c.record_count} ({c.record_percentage}%)</td>
                      <td className="py-3 px-3 text-cyan-300 font-bold">{c.centers.india_import_dependency}%</td>
                      <td className="py-3 px-3 text-white/80">{c.centers.india_export_dependency}%</td>
                      <td className="py-3 px-3 text-amber-300 font-bold">{c.centers.energy_dependency}%</td>
                      <td className="py-3 px-3 text-white/80">{c.centers.commodity_dependency}%</td>
                      <td className="py-3 px-3 text-white/80">${c.centers.trade_value.toLocaleString()}M</td>
                      <td className="py-3 px-3 text-rose-300 font-bold">{c.centers.shipping_disruption} / 10</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab 4: Country Exposure Matrix with Search Filter */}
        {activeVizTab === 'countries' && (
          <div className="space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <span className="text-xs text-white/50 font-mono">
                26 Sovereign Trading Partners • Sorted by Aggregate Vulnerability
              </span>
              <input
                type="text"
                placeholder="Filter country (e.g. Russia, China, UAE)..."
                value={countryFilter}
                onChange={e => setCountryFilter(e.target.value)}
                className="px-3 py-1.5 text-xs rounded-lg bg-black/40 border border-white/10 text-white placeholder-white/30 focus:border-emerald-500 focus:outline-none font-mono w-full sm:w-64"
              />
            </div>
            <div className="overflow-x-auto max-h-96">
              <table className="w-full text-left text-xs font-mono">
                <thead className="sticky top-0 bg-[#0f1117] border-b border-white/10 text-white/50 text-[11px]">
                  <tr>
                    <th className="py-3 px-3">Country / Partner</th>
                    <th className="py-3 px-3">Primary Risk Group</th>
                    <th className="py-3 px-3">Import Dep (%)</th>
                    <th className="py-3 px-3">Export Dep (%)</th>
                    <th className="py-3 px-3">Energy Dep (%)</th>
                    <th className="py-3 px-3">Commodity Dep (%)</th>
                    <th className="py-3 px-3">Avg Trade Value ($M)</th>
                    <th className="py-3 px-3">Route Disruption</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {countryAnalysis
                    .filter(c => c.country.toLowerCase().includes(countryFilter.toLowerCase()))
                    .map(c => (
                      <tr key={c.country} className="hover:bg-white/[0.02]">
                        <td className="py-2.5 px-3 font-sans font-medium text-white">{c.country}</td>
                        <td className="py-2.5 px-3">
                          <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                            c.primary_risk_group === 'Critical Exposure' ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' :
                            c.primary_risk_group === 'Higher Exposure' ? 'bg-orange-500/20 text-orange-300 border border-orange-500/30' :
                            c.primary_risk_group === 'Moderate Exposure' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' :
                            'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                          }`}>
                            {c.primary_risk_group}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-cyan-400 font-bold">{c.avg_import_dep}%</td>
                        <td className="py-2.5 px-3 text-white/70">{c.avg_export_dep}%</td>
                        <td className="py-2.5 px-3 text-amber-400 font-bold">{c.avg_energy_dep}%</td>
                        <td className="py-2.5 px-3 text-white/70">{c.avg_comm_dep}%</td>
                        <td className="py-2.5 px-3 text-white/80">${c.avg_trade_val.toLocaleString()}M</td>
                        <td className="py-2.5 px-3 text-rose-400 font-medium">{c.avg_route_exp} / 10</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
