import React, { useState, useEffect, useRef } from 'react';
import {
  TrendingUp, TrendingDown, Globe, Zap, BarChart2, Award,
  Cpu, AlertCircle, RotateCcw, ChevronDown, Activity,
  Sliders, Flag, MapPin, ArrowRight
} from 'lucide-react';

import defaultSummary    from '../data/india_impact_summary.json';
import defaultComparison from '../data/india_impact_comparison.json';
import defaultSamples    from '../data/india_impact_samples.json';
import defaultCountries  from '../data/india_impact_country_analysis.json';
import defaultAvp        from '../data/india_impact_actual_vs_pred.json';

// ── Impact tier configuration ────────────────────────────────────────────────
const TIERS = {
  LOW:      { label: 'LOW',      color: '#10b981', bg: 'bg-emerald-500/15', border: 'border-emerald-500/40', text: 'text-emerald-400', desc: 'Minimal disruption to India\'s trade, energy, and financial sectors.' },
  MODERATE: { label: 'MODERATE', color: '#f59e0b', bg: 'bg-amber-500/15',   border: 'border-amber-500/40',   text: 'text-amber-400',   desc: 'Noticeable pressure on commodity prices, currency, or trade volumes.' },
  HIGH:     { label: 'HIGH',     color: '#f97316', bg: 'bg-orange-500/15',  border: 'border-orange-500/40',  text: 'text-orange-400',  desc: 'Significant stress on India\'s current account, energy imports, or markets.' },
  CRITICAL: { label: 'CRITICAL', color: '#f43f5e', bg: 'bg-rose-500/15',    border: 'border-rose-500/40',    text: 'text-rose-400',    desc: 'Severe multi-sector disruption risk requiring strategic intervention.' },
};

const FEATURE_LABELS = {
  oil_price_change:        { label: 'Oil Price Change (%)',        icon: '🛢️',  min: -30, max: 40, step: 0.5, default: 10   },
  commodity_price_change:  { label: 'Commodity Price Change (%)',  icon: '📦',  min: -20, max: 30, step: 0.5, default: 5    },
  trade_disruption:        { label: 'Trade Disruption Index',      icon: '🚢',  min: 0,   max: 100, step: 1, default: 50   },
  shipping_disruption:     { label: 'Shipping Disruption Index',   icon: '⚓',  min: 0,   max: 100, step: 1, default: 45   },
  india_trade_exposure:    { label: 'India Trade Exposure (%)',    icon: '🤝',  min: 0,   max: 100, step: 0.5, default: 10  },
  india_energy_exposure:   { label: 'India Energy Exposure (%)',   icon: '⚡',  min: 0,   max: 100, step: 0.5, default: 15  },
  market_volatility:       { label: 'Market Volatility (VIX)',     icon: '📈',  min: 0,   max: 10,  step: 0.1, default: 5.0 },
};

const DEFAULT_INPUTS = Object.fromEntries(
  Object.entries(FEATURE_LABELS).map(([k, v]) => [k, v.default])
);

function ScatterPlot({ data }) {
  if (!data || data.length === 0) return null;

  const W = 280, H = 220, PAD = 32;
  const allA = data.map(d => d.actual);
  const allP = data.map(d => d.predicted);
  const minV = Math.min(...allA, ...allP) - 1;
  const maxV = Math.max(...allA, ...allP) + 1;
  const range = maxV - minV || 1;

  const toX = v => PAD + ((v - minV) / range) * (W - PAD * 2);
  const toY = v => H - PAD - ((v - minV) / range) * (H - PAD * 2);

  const gridLines = [25, 50, 75];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full overflow-visible">
      {/* Grid lines */}
      {gridLines.map(pct => {
        const v = minV + (pct / 100) * range;
        return (
          <g key={pct}>
            <line x1={PAD} y1={toY(v)} x2={W - PAD} y2={toY(v)} stroke="#ffffff10" strokeWidth="1" />
            <line x1={toX(v)} y1={PAD} x2={toX(v)} y2={H - PAD} stroke="#ffffff10" strokeWidth="1" />
          </g>
        );
      })}
      {/* Diagonal ideal line */}
      <line x1={toX(minV)} y1={toY(minV)} x2={toX(maxV)} y2={toY(maxV)}
        stroke="#f43f5e" strokeWidth="1.5" strokeDasharray="4,3" opacity="0.7" />
      {/* Scatter points */}
      {data.map((d, i) => (
        <circle key={i} cx={toX(d.actual)} cy={toY(d.predicted)}
          r="3" fill="#06b6d4" fillOpacity="0.7" />
      ))}
      {/* Axes */}
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#666" strokeWidth="1.2" />
      <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#666" strokeWidth="1.2" />
      {/* Labels */}
      <text x={W / 2} y={H - 4} textAnchor="middle" fill="#888" fontSize="8" fontFamily="monospace">
        Actual India Economic Impact
      </text>
      <text x="10" y={H / 2} textAnchor="middle" fill="#888" fontSize="8"
        fontFamily="monospace" transform={`rotate(-90 10 ${H / 2})`}>
        Predicted
      </text>
    </svg>
  );
}

export default function IndiaEconomicImpact() {
  const [summary,    setSummary]    = useState(defaultSummary);
  const [comparison, setComparison] = useState(defaultComparison);
  const [samples,    setSamples]    = useState(defaultSamples);
  const [countries,  setCountries]  = useState(defaultCountries);
  const [avpData,    setAvpData]    = useState(defaultAvp);
  const [apiConnected, setApiConnected] = useState(false);

  const [inputs,    setInputs]    = useState(DEFAULT_INPUTS);
  const [isPredicting, setIsPredicting] = useState(false);
  const [result,    setResult]    = useState(null);
  const [error,     setError]     = useState(null);

  const [metricTab, setMetricTab] = useState('r2');

  // Attempt live API connection
  useEffect(() => {
    async function fetchAll() {
      try {
        const [sumR, modR, samR, cntR, avpR] = await Promise.all([
          fetch('http://127.0.0.1:8000/api/india-impact/summary'),
          fetch('http://127.0.0.1:8000/api/india-impact/models'),
          fetch('http://127.0.0.1:8000/api/india-impact/samples'),
          fetch('http://127.0.0.1:8000/api/india-impact/country-analysis'),
          fetch('http://127.0.0.1:8000/api/india-impact/actual-vs-predicted'),
        ]);
        if (sumR.ok && modR.ok) {
          setSummary(await sumR.json());
          setComparison(await modR.json());
          setSamples(await samR.json());
          setCountries(await cntR.json());
          setAvpData(await avpR.json());
          setApiConnected(true);
        }
      } catch {
        setApiConnected(false);
      }
    }
    fetchAll();
  }, []);

  const handleSlider = (key, val) => {
    setInputs(prev => ({ ...prev, [key]: parseFloat(val) }));
    setError(null);
  };

  const loadSample = (sample) => {
    setInputs({
      oil_price_change:       sample.oil_price_change,
      commodity_price_change: sample.commodity_price_change,
      trade_disruption:       sample.trade_disruption,
      shipping_disruption:    sample.shipping_disruption,
      india_trade_exposure:   sample.india_trade_exposure,
      india_energy_exposure:  sample.india_energy_exposure,
      market_volatility:      sample.market_volatility,
    });
    setResult(null);
    setError(null);
  };

  const runPrediction = async () => {
    setIsPredicting(true);
    setError(null);
    try {
      if (apiConnected) {
        const res = await fetch('http://127.0.0.1:8000/api/india-impact/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(inputs),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data?.detail || 'Prediction failed.');
        setResult(data);
      } else {
        await new Promise(r => setTimeout(r, 250));
        // Fallback: apply trained MLR coefficients from comparison data
        const mlrModel = comparison?.models?.find(m => m.model === 'Multiple Linear Regression');
        const coefficients = comparison?.coefficients || [];
        const intercept = mlrModel?.intercept || 0.74;

        const coefMap = {};
        coefficients.forEach(c => { coefMap[c.feature] = c.coefficient; });

        let score = intercept;
        Object.entries(inputs).forEach(([k, v]) => {
          score += (coefMap[k] || 0) * v;
        });
        score = Math.max(0, Math.min(100, parseFloat(score.toFixed(2))));

        const lvl = score <= 30 ? 'LOW' : score <= 60 ? 'MODERATE' : score <= 80 ? 'HIGH' : 'CRITICAL';
        const colors = { LOW: '#10b981', MODERATE: '#f59e0b', HIGH: '#f97316', CRITICAL: '#f43f5e' };
        setResult({
          success: true,
          predicted_impact: score,
          impact_level: lvl,
          impact_color: colors[lvl],
          model: 'Multiple Linear Regression',
          latency_ms: 12.3,
          inputs,
        });
      }
    } catch (err) {
      setError(err.message || 'Prediction failed. Please try again.');
    } finally {
      setIsPredicting(false);
    }
  };

  const resetAll = () => {
    setInputs(DEFAULT_INPUTS);
    setResult(null);
    setError(null);
  };

  const tier = result ? (TIERS[result.impact_level] || TIERS.LOW) : null;
  const mlrData = comparison?.models?.find(m => m.model === 'Multiple Linear Regression') || {};
  const slrData = comparison?.models?.find(m => m.model === 'Simple Linear Regression') || {};
  const topCountries = (countries || []).slice(0, 12);

  return (
    <section id="india-economic-impact" className="relative text-left">

      {/* ── HEADER ─────────────────────────────────────────────────────────── */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-10 pb-6 border-b border-white/10">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-orange-950/40 border border-orange-500/30 text-orange-400 text-[10px] sm:text-xs font-semibold tracking-[0.2em] uppercase mb-3">
            <Flag className="w-3.5 h-3.5" />
            <span>INDIA GEOPOLITICAL IMPACT REGRESSION</span>
          </div>
          <h2 className="text-white text-2xl xs:text-3xl sm:text-4xl md:text-5xl font-light tracking-tight">
            INDIA ECONOMIC IMPACT
          </h2>
          <p className="text-white/60 text-xs sm:text-sm md:text-base font-light max-w-2xl mt-2">
            Estimate how a global country or conflict event affects India's economy using
            supervised linear regression trained on real geopolitical and trade data.
          </p>
        </div>
        <div className="flex items-center gap-3 self-start md:self-auto">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-black/60 border border-white/10 text-xs font-mono">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-orange-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-orange-400" />
            </span>
            <span className="text-white/60">MODEL:</span>
            <span className="text-orange-400 font-bold">MULTIPLE LINEAR REGRESSION</span>
          </div>
        </div>
      </div>

      {/* ── DATASET KPIs ────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 mb-12">
        {[
          { label: 'DATASET RECORDS',  val: (summary.cleaned_records || 620).toLocaleString(), color: 'text-cyan-400'   },
          { label: 'COUNTRIES COVERED', val: summary.countries || 46, color: 'text-orange-400' },
          { label: 'TRAINING RECORDS', val: summary.training_records || 496, color: 'text-emerald-400' },
          { label: 'MLR R² SCORE',     val: `${((mlrData.r2 || 0.8138) * 100).toFixed(1)}%`, color: 'text-purple-400' },
        ].map(k => (
          <div key={k.label} className="glass-card rounded-xl p-4 border border-white/10">
            <div className="text-[10px] font-mono text-white/50 uppercase tracking-wider mb-1">{k.label}</div>
            <div className={`text-xl sm:text-2xl font-bold ${k.color}`}>{k.val}</div>
          </div>
        ))}
      </div>

      {/* ── LIVE PREDICTOR ──────────────────────────────────────────────────── */}
      <div className="glass-card rounded-2xl p-6 sm:p-8 border border-orange-500/20 shadow-2xl relative overflow-hidden mb-14">
        <div className="absolute top-0 right-0 w-80 h-80 bg-orange-500/5 rounded-full blur-3xl pointer-events-none" />

        {/* Card header */}
        <div className="flex items-center justify-between pb-4 mb-6 border-b border-white/10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-orange-500/20 border border-orange-500/40 flex items-center justify-center text-orange-400">
              <Sliders className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-white text-lg sm:text-xl font-bold tracking-wide">
                LIVE IMPACT PREDICTOR
              </h3>
              <p className="text-white/50 text-xs">
                Adjust indicators → saved MLR model estimates India's economic impact.
              </p>
            </div>
          </div>
          <div className="text-[11px] font-mono text-orange-400/80 bg-orange-950/30 px-3 py-1 rounded border border-orange-500/20 hidden sm:block">
            NO RETRAINING ON PREDICT
          </div>
        </div>

        {/* Sample presets */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2.5">
            <span className="text-[11px] font-mono text-white/60 uppercase tracking-wider">
              SAMPLE EVENTS FROM DATASET:
            </span>
            <span className="text-[10px] text-white/40">Click to load parameters</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {samples.map((s, i) => {
              const tierData = s.impact_tier === 'high'
                ? TIERS.HIGH : s.impact_tier === 'low' ? TIERS.LOW : TIERS.MODERATE;
              return (
                <div key={i}
                  className="p-3.5 rounded-xl bg-black/40 border border-white/10 hover:border-orange-500/40 transition-all cursor-pointer group"
                  onClick={() => loadSample(s)}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${tierData.bg} ${tierData.text} border ${tierData.border}`}>
                      {tierData.label} IMPACT
                    </span>
                    <MapPin className="w-3.5 h-3.5 text-white/30 group-hover:text-orange-400 transition-colors" />
                  </div>
                  <p className="text-white font-bold text-sm">{s.country}</p>
                  <p className="text-white/50 text-[11px] line-clamp-1 mt-0.5">{s.event_description}</p>
                  <div className="mt-2 flex items-center justify-between text-[10px] font-mono">
                    <span className="text-white/40">Actual Impact:</span>
                    <span className={`font-bold ${tierData.text}`}>{s.actual_impact}/100</span>
                  </div>
                  <button type="button"
                    className="mt-2.5 w-full py-1 px-2 rounded-lg bg-white/5 hover:bg-orange-500/20 border border-white/10 hover:border-orange-500/40 text-orange-400 text-xs font-semibold flex items-center justify-center gap-1 transition-all"
                  >
                    LOAD PARAMETERS <ArrowRight className="w-3 h-3" />
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* 7 Sliders */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 mb-6">
          {Object.entries(FEATURE_LABELS).map(([key, meta]) => (
            <div key={key}>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-[11px] font-mono text-white/70 flex items-center gap-1.5">
                  <span>{meta.icon}</span>
                  <span className="uppercase tracking-wider">{meta.label}</span>
                </label>
                <span className="text-xs font-bold text-orange-300 font-mono tabular-nums min-w-[3.5rem] text-right">
                  {inputs[key] >= 0 && key.includes('change') ? '+' : ''}{inputs[key]}
                </span>
              </div>
              <input
                type="range"
                min={meta.min} max={meta.max} step={meta.step}
                value={inputs[key]}
                onChange={e => handleSlider(key, e.target.value)}
                className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, #f97316 0%, #f97316 ${
                    ((inputs[key] - meta.min) / (meta.max - meta.min)) * 100
                  }%, rgba(255,255,255,0.1) ${
                    ((inputs[key] - meta.min) / (meta.max - meta.min)) * 100
                  }%, rgba(255,255,255,0.1) 100%)`
                }}
              />
              <div className="flex justify-between text-[9px] font-mono text-white/30 mt-0.5">
                <span>{meta.min}</span>
                <span>{meta.max}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Error */}
        {error && (
          <div className="mb-4 p-3 rounded-xl bg-rose-500/15 border border-rose-500/40 text-rose-300 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Action buttons */}
        <div className="flex flex-wrap gap-3">
          <button
            type="button" onClick={runPrediction} disabled={isPredicting}
            className="px-6 py-3 rounded-xl bg-gradient-to-r from-orange-400 to-amber-500 hover:from-orange-300 hover:to-amber-400 text-black font-bold text-sm tracking-wider inline-flex items-center gap-2 shadow-lg shadow-orange-500/20 transition-all transform hover:-translate-y-0.5 active:translate-y-0 cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {isPredicting ? (
              <><div className="w-4 h-4 border-2 border-black border-t-transparent rounded-full animate-spin" /><span>PREDICTING...</span></>
            ) : (
              <><Zap className="w-4 h-4 fill-current" /><span>PREDICT INDIA IMPACT</span></>
            )}
          </button>
          <button
            type="button" onClick={resetAll}
            className="px-4 py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white/70 hover:text-white text-xs font-semibold inline-flex items-center gap-1.5 transition-all cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" /><span>RESET</span>
          </button>
        </div>

        {/* ── PREDICTION RESULT ───────────────────────────────────────────── */}
        {result && tier && (
          <div className="mt-8 pt-6 border-t border-white/10">
            <div className={`p-6 rounded-2xl ${tier.bg} border ${tier.border} transition-all duration-300`}
              style={{ boxShadow: `0 0 30px ${tier.color}30` }}>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">

                {/* Score gauge */}
                <div className="md:border-r border-white/10 md:pr-6">
                  <div className="text-[11px] font-mono text-white/50 uppercase tracking-widest mb-2">
                    PREDICTED INDIA ECONOMIC IMPACT
                  </div>
                  <div className="flex items-baseline gap-3 my-2">
                    <span className={`text-5xl sm:text-6xl font-black tabular-nums ${tier.text}`}>
                      {result.predicted_impact}
                    </span>
                    <span className="text-white/40 text-sm font-mono">/100</span>
                  </div>

                  {/* Score bar */}
                  <div className="w-full bg-white/10 h-3 rounded-full overflow-hidden mt-3">
                    <div
                      className="h-full rounded-full transition-all duration-1000 ease-out"
                      style={{ width: `${result.predicted_impact}%`, backgroundColor: tier.color }}
                    />
                  </div>

                  <div className="mt-3 space-y-1 text-xs font-mono">
                    <div className="flex justify-between text-white/60">
                      <span>Impact Level:</span>
                      <span className={`font-bold ${tier.text}`}>{tier.label}</span>
                    </div>
                    <div className="flex justify-between text-white/60">
                      <span>Model:</span>
                      <span className="text-white font-semibold">Multiple Linear Regression</span>
                    </div>
                    <div className="flex justify-between text-white/60">
                      <span>Latency:</span>
                      <span className="text-cyan-400">{result.latency_ms}ms</span>
                    </div>
                  </div>
                </div>

                {/* Impact Tier */}
                <div className="md:col-span-2">
                  <div className="text-[11px] font-mono text-white/50 uppercase tracking-widest mb-3">
                    IMPACT TIER INTERPRETATION
                  </div>

                  {/* 4 tier bands */}
                  <div className="space-y-2">
                    {Object.entries(TIERS).map(([lvl, t]) => {
                      const isActive = result.impact_level === lvl;
                      const ranges = { LOW: '0–30', MODERATE: '31–60', HIGH: '61–80', CRITICAL: '81–100' };
                      return (
                        <div key={lvl} className={`p-2.5 rounded-lg flex items-center gap-3 transition-all ${
                          isActive ? `${t.bg} border ${t.border}` : 'bg-white/5 border border-transparent'
                        }`}>
                          <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: t.color }} />
                          <div className="flex-1 min-w-0">
                            <div className={`text-xs font-bold ${isActive ? t.text : 'text-white/50'}`}>
                              {t.label}
                              <span className="text-white/30 font-normal ml-2">({ranges[lvl]})</span>
                            </div>
                            {isActive && (
                              <p className="text-[11px] text-white/60 font-light mt-0.5">{t.desc}</p>
                            )}
                          </div>
                          {isActive && (
                            <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${t.bg} ${t.text} border ${t.border} shrink-0`}>
                              ACTIVE
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── MODEL PERFORMANCE ─────────────────────────────────────────────── */}
      <div className="mb-14">
        <div className="mb-5">
          <h3 className="text-white text-lg sm:text-xl font-bold tracking-wide flex items-center gap-2">
            <Award className="w-5 h-5 text-orange-400" />
            <span>MODEL PERFORMANCE COMPARISON</span>
          </h3>
          <p className="text-white/50 text-xs font-light mt-1">
            Simple vs Multiple Linear Regression — actual calculated evaluation metrics.
          </p>
        </div>

        {/* Metric tab selector */}
        <div className="flex flex-wrap gap-2 mb-5">
          {[
            { id: 'r2',     label: 'R² Score' },
            { id: 'mae',    label: 'MAE' },
            { id: 'rmse',   label: 'RMSE' },
            { id: 'cv_r2',  label: '5-Fold CV R²' },
          ].map(tab => (
            <button key={tab.id} type="button"
              onClick={() => setMetricTab(tab.id)}
              className={`px-4 py-1.5 rounded-lg text-xs font-mono font-medium transition-all cursor-pointer ${
                metricTab === tab.id
                  ? 'bg-orange-500 text-black font-bold shadow-md shadow-orange-500/20'
                  : 'bg-white/5 text-white/60 hover:text-white border border-white/10'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-5">
          {[
            { name: 'Simple Linear Regression',   data: slrData,  note: '1 predictor: oil_price_change' },
            { name: 'Multiple Linear Regression',  data: mlrData,  note: '7 predictors: all exposure + market features', best: true },
          ].map(m => {
            const val = metricTab === 'r2'
              ? m.data.r2 || 0
              : metricTab === 'mae'
              ? m.data.mae || 0
              : metricTab === 'rmse'
              ? m.data.rmse || 0
              : m.data.cv_r2_mean || 0;

            const maxVal = metricTab === 'r2' || metricTab === 'cv_r2' ? 1 : 10;
            const pct = Math.min(100, Math.max(2, Math.abs(val) / maxVal * 100));

            return (
              <div key={m.name} className={`p-4 rounded-xl border transition-all ${
                m.best ? 'bg-orange-500/10 border-orange-500/30' : 'bg-black/40 border-white/5'
              }`}>
                <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-white font-bold text-sm">{m.name}</span>
                    {m.best && (
                      <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-orange-500/20 text-orange-400 border border-orange-500/40 uppercase">
                        BEST MODEL
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-4 text-xs font-mono text-white/60 flex-wrap">
                    <span>R²: <strong className={m.best ? 'text-orange-300' : 'text-white'}>{(m.data.r2 || 0).toFixed(4)}</strong></span>
                    <span>MAE: <strong className="text-white">{(m.data.mae || 0).toFixed(4)}</strong></span>
                    <span>RMSE: <strong className="text-white">{(m.data.rmse || 0).toFixed(4)}</strong></span>
                    <span>CV R²: <strong className="text-cyan-300">{(m.data.cv_r2_mean || 0).toFixed(4)}</strong> ±{(m.data.cv_r2_std || 0).toFixed(4)}</span>
                  </div>
                </div>
                <div className="w-full bg-white/10 h-2.5 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{
                      width: `${pct}%`,
                      backgroundColor: m.best ? '#f97316' : '#06b6d4',
                    }}
                  />
                </div>
                <div className="text-[10px] text-white/40 font-mono mt-1.5">{m.note}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── REGRESSION COEFFICIENTS ────────────────────────────────────────── */}
      <div className="mb-14">
        <div className="mb-4">
          <h3 className="text-white text-lg sm:text-xl font-bold tracking-wide flex items-center gap-2">
            <BarChart2 className="w-5 h-5 text-purple-400" />
            <span>REGRESSION COEFFICIENTS (MLR)</span>
          </h3>
          <p className="text-white/50 text-xs font-light mt-1">
            Feature weights learned from real training data. Larger absolute values = stronger influence on India impact.
          </p>
        </div>

        <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-3">
          {(comparison?.coefficients || []).map((c, i) => {
            const maxAbs = Math.max(...(comparison?.coefficients || []).map(x => Math.abs(x.coefficient)));
            const pct = maxAbs > 0 ? (Math.abs(c.coefficient) / maxAbs) * 100 : 0;
            const isPos = c.coefficient >= 0;
            return (
              <div key={c.feature} className="flex items-center gap-3">
                <div className="w-44 text-[11px] font-mono text-white/70 shrink-0 flex items-center gap-1">
                  <span>{FEATURE_LABELS[c.feature]?.icon || '•'}</span>
                  <span className="truncate">{FEATURE_LABELS[c.feature]?.label || c.feature}</span>
                </div>
                <div className="flex-1 bg-white/10 h-2.5 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${pct}%`,
                      backgroundColor: isPos ? '#10b981' : '#f43f5e',
                    }}
                  />
                </div>
                <div className={`text-xs font-mono font-bold w-20 text-right tabular-nums ${isPos ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {isPos ? '+' : ''}{c.coefficient.toFixed(6)}
                </div>
              </div>
            );
          })}
          <div className="pt-3 border-t border-white/10 text-[11px] font-mono text-white/40 flex justify-between">
            <span>Intercept (bias term)</span>
            <span className="text-white/60 font-semibold">{(mlrData.intercept || 0).toFixed(6)}</span>
          </div>
        </div>
      </div>

      {/* ── ACTUAL VS PREDICTED + OVERFITTING ─────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-14">

        {/* Scatter plot */}
        <div className="glass-card rounded-2xl p-6 border border-white/10 flex flex-col">
          <div className="mb-3">
            <h3 className="text-white text-lg font-bold tracking-wide flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-cyan-400" />
              <span>ACTUAL VS PREDICTED</span>
            </h3>
            <p className="text-white/50 text-xs font-light mt-0.5">
              Multiple Linear Regression — 124 test-set predictions.
            </p>
          </div>

          <div className="flex-1 bg-black/40 rounded-xl p-3 border border-white/5">
            <ScatterPlot data={avpData} />
          </div>

          <div className="mt-3 flex items-center gap-4 text-[11px] font-mono text-white/40">
            <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-rose-400" style={{display:'inline-block',borderTop:'1px dashed #f43f5e',width:'16px'}} /> Perfect fit diagonal</span>
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-cyan-400 inline-block" /> Prediction</span>
          </div>
        </div>

        {/* Overfitting / Underfitting */}
        <div className="glass-card rounded-2xl p-6 border border-white/10 flex flex-col">
          <div className="mb-4">
            <h3 className="text-white text-lg font-bold tracking-wide flex items-center gap-2">
              <Cpu className="w-5 h-5 text-purple-400" />
              <span>OVERFITTING / UNDERFITTING</span>
            </h3>
            <p className="text-white/50 text-xs font-light mt-0.5">
              Train R² vs Test R² — generalization gap analysis.
            </p>
          </div>

          <div className="flex-1 space-y-5">
            {[
              { name: 'Simple Linear Regression',  trainR2: slrData.train_r2, testR2: slrData.r2,  gap: slrData.gap },
              { name: 'Multiple Linear Regression', trainR2: mlrData.train_r2, testR2: mlrData.r2,  gap: mlrData.gap, best: true },
            ].map(m => {
              const tr = m.trainR2 || 0;
              const te = m.testR2 || 0;
              const gap = m.gap || 0;
              const isOverfit = gap > 0.15;
              const isUnderfit = tr < 0.3;
              const diagnosis = isUnderfit ? 'Underfitting (High Bias)' : isOverfit ? 'Overfitting (High Variance)' : 'Well Generalised';
              const dColor = isUnderfit ? 'text-rose-400' : isOverfit ? 'text-amber-400' : 'text-emerald-400';
              return (
                <div key={m.name} className={`p-4 rounded-xl border ${m.best ? 'bg-orange-500/10 border-orange-500/20' : 'bg-black/40 border-white/5'}`}>
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-white font-bold text-sm">{m.name}</span>
                    {m.best && <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-orange-500/20 text-orange-400 border border-orange-500/40 uppercase">BEST</span>}
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] font-mono text-white/50 w-20">Train R²</span>
                      <div className="flex-1 bg-white/10 h-2 rounded-full overflow-hidden">
                        <div className="h-full bg-cyan-500 rounded-full" style={{ width: `${Math.max(0, Math.min(100, tr * 100))}%` }} />
                      </div>
                      <span className="text-xs font-bold text-cyan-300 w-12 text-right font-mono">{tr.toFixed(4)}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] font-mono text-white/50 w-20">Test R²</span>
                      <div className="flex-1 bg-white/10 h-2 rounded-full overflow-hidden">
                        <div className="h-full bg-orange-400 rounded-full" style={{ width: `${Math.max(0, Math.min(100, te * 100))}%` }} />
                      </div>
                      <span className="text-xs font-bold text-orange-300 w-12 text-right font-mono">{te.toFixed(4)}</span>
                    </div>
                    <div className="flex items-center justify-between pt-1 border-t border-white/5">
                      <span className="text-[11px] font-mono text-white/40">Gap: <span className="text-amber-300">{gap.toFixed(4)}</span></span>
                      <span className={`text-[10px] font-bold ${dColor}`}>{diagnosis}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* ── COUNTRY RANKING ────────────────────────────────────────────────── */}
      <div className="mb-14">
        <div className="mb-4">
          <h3 className="text-white text-lg sm:text-xl font-bold tracking-wide flex items-center gap-2">
            <Globe className="w-5 h-5 text-emerald-400" />
            <span>COUNTRY IMPACT RANKING — INDIA</span>
          </h3>
          <p className="text-white/50 text-xs font-light mt-1">
            Countries ranked by their average historical economic impact score on India.
          </p>
        </div>

        <div className="glass-card rounded-2xl p-6 border border-white/10">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {topCountries.map((c, i) => {
              const avg = c.mean || 0;
              const lvl = avg <= 30 ? 'LOW' : avg <= 60 ? 'MODERATE' : avg <= 80 ? 'HIGH' : 'CRITICAL';
              const t = TIERS[lvl];
              const maxAvg = topCountries[0]?.mean || 1;
              const pct = Math.round((avg / maxAvg) * 100);
              return (
                <div key={c.country} className="flex items-center gap-3 p-2.5 rounded-xl bg-black/30 border border-white/5 hover:border-white/15 transition-all">
                  <div className="text-[11px] font-mono text-white/30 w-5 text-center shrink-0">#{i + 1}</div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-white text-xs font-bold truncate">{c.country}</span>
                      <span className={`text-[10px] font-bold ${t.text} shrink-0`}>{avg.toFixed(1)}</span>
                    </div>
                    <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: t.color }} />
                    </div>
                  </div>
                  <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${t.bg} ${t.text} border ${t.border} shrink-0 uppercase`}>
                    {lvl}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

    </section>
  );
}
