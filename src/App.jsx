import React, { useState, useEffect } from 'react';
import GlobalEventIntelligence from './components/GlobalEventIntelligence';
import NewsNarrativeIntelligence from './components/NewsNarrativeIntelligence';
import SentimentIntelligence from './components/SentimentIntelligence';
import IndiaEconomicImpact from './components/IndiaEconomicImpact';
import IndiaEnergyRiskIntelligence from './components/IndiaEnergyRiskIntelligence';
import IndiaCommodityShockIntelligence from './components/IndiaCommodityShockIntelligence';
import IndiaTradeDependencyRisk from './components/IndiaTradeDependencyRisk';
import IndiaSupplyRouteDisruption from './components/IndiaSupplyRouteDisruption';
import {
  Menu,
  X,
  ArrowRight,
  ShieldAlert,
  TrendingUp,
  Activity,
  Radio,
  Globe,
  Sliders,
  Cpu,
  Layers,
  Zap,
  BarChart3,
  ExternalLink,
  ChevronRight,
  Droplet,
  Ship
} from 'lucide-react';

// Brand icons styled in the Feather/Lucide design language
const Facebook = ({ className = "w-4 h-4" }) => (
  <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z" />
  </svg>
);

const Twitter = ({ className = "w-4 h-4" }) => (
  <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M22 4s-.7 2.1-2 3.4c1.6 10-9.4 17.3-18 11.6 2.2.1 4.4-.6 6-2C3 15.5.5 9.6 3 5c2.2 2.6 5.6 4.1 9 4-.9-4.2 4-6.6 7-3.8 1.1 0 3-1.2 3-1.2z" />
  </svg>
);

const Dribbble = ({ className = "w-4 h-4" }) => (
  <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <circle cx="12" cy="12" r="10" />
    <path d="M19.13 5.09C15.22 9.14 10 10.44 2.25 10.94" />
    <path d="M21.75 12.84c-6.62-1.41-12.14 1-16.38 6.32" />
    <path d="M8.56 2.75c4.37 6 6 9.42 8 17.72" />
  </svg>
);

const Youtube = ({ className = "w-4 h-4" }) => (
  <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M2.5 17a24.12 24.12 0 0 1 0-10 2 2 0 0 1 1.4-1.4 49.56 49.56 0 0 1 16.2 0A2 2 0 0 1 21.5 7a24.12 24.12 0 0 1 0 10 2 2 0 0 1-1.4 1.4 49.55 49.55 0 0 1-16.2 0A2 2 0 0 1 2.5 17" />
    <polygon points="10 15 15 12 10 9 10 15" fill="currentColor" />
  </svg>
);

const Linkedin = ({ className = "w-4 h-4" }) => (
  <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z" />
    <rect x="2" y="9" width="4" height="12" />
    <circle cx="4" cy="4" r="2" />
  </svg>
);

const Instagram = ({ className = "w-4 h-4" }) => (
  <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <rect width="20" height="20" x="2" y="2" rx="5" ry="5" />
    <path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z" />
    <line x1="17.5" x2="17.51" y1="6.5" y2="6.5" />
  </svg>
);

const navLinks = [
  { name: 'Home', href: '#home' },
  { name: 'Global Events', href: '#events' },
  { name: 'News Intelligence', href: '#news-intelligence' },
  { name: 'Sentiment Analysis', href: '#sentiment-intelligence' },
  { name: 'India Impact 🇮🇳', href: '#india-impact' },
  { name: 'Energy Risk ⚡', href: '#india-energy-risk' },
  { name: 'Commodity Shock 🛢️', href: '#commodity-shock' },
  { name: 'Trade Risk 🌐', href: '#trade-dependency' },
  { name: 'Supply Routes 🚢', href: '#supply-route' },
  { name: 'Risk Analysis', href: '#risk' },
  { name: 'AI Intelligence', href: '#flow' },
];

const footerColumns = [
  {
    title: 'INTELLIGENCE',
    links: ['Conflict Radar', 'Maritime AIS', 'Energy Corridors', 'Currency Shocks', 'Sentiment Index'],
  },
  {
    title: 'PLATFORM',
    links: ['Global Heatmap', 'Signal Ingestion', 'Risk Matrices', 'Scenario Engine', 'API Feeds'],
  },
  {
    title: 'RESOURCES',
    links: ['Methodology', 'Case Studies', 'Threat Library', 'Research Papers', 'Documentation'],
  },
  {
    title: 'COMPANY',
    links: ['About GeoPulse', 'Geopolitical Analysts', 'Press Room', 'Careers', 'Security & Ethics'],
  },
];

export default function App() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [menuVisible, setMenuVisible] = useState(false);
  const [emailInput, setEmailInput] = useState('');
  const [subscribed, setSubscribed] = useState(false);

  // Manage mobile menu mounting and staggered animation transitions
  useEffect(() => {
    let timeoutId;
    if (mobileMenuOpen) {
      timeoutId = setTimeout(() => {
        setMenuVisible(true);
      }, 20);
    }
    return () => {
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [mobileMenuOpen]);

  const toggleMobileMenu = () => {
    if (!mobileMenuOpen) {
      setMobileMenuOpen(true);
    } else {
      setMenuVisible(false);
      setTimeout(() => {
        setMobileMenuOpen(false);
      }, 500);
    }
  };

  const closeMobileMenu = () => {
    if (mobileMenuOpen) {
      setMenuVisible(false);
      setTimeout(() => {
        setMobileMenuOpen(false);
      }, 500);
    }
  };

  const handleSubscribe = (e) => {
    e.preventDefault();
    if (emailInput.trim()) {
      setSubscribed(true);
      setTimeout(() => setSubscribed(false), 3500);
      setEmailInput('');
    }
  };

  return (
    <div
      className="relative min-h-screen flex flex-col text-white selection:bg-cyan-500 selection:text-white"
      style={{ fontFamily: '"Helvetica Now Var", Helvetica, Arial, sans-serif' }}
    >
      {/* Cinematic Looping Earth Background Video (fixed behind all content) */}
      <video
        autoPlay
        muted
        loop
        playsInline
        className="fixed inset-0 w-full h-full object-cover pointer-events-none z-0"
        src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260613_180732_a54afbf6-b30d-470e-861f-669871f09f67.mp4"
      />

      {/* Futuristic Scanline and Vignette Layer */}
      <div className="fixed inset-0 bg-gradient-to-b from-black/50 via-black/35 to-black/85 pointer-events-none z-0" />
      <div className="fixed inset-0 scanline-bg opacity-30 pointer-events-none z-0" />

      {/* Main Content Wrapper */}
      <div className="relative z-10 flex flex-col min-h-screen justify-between">

        {/* =========================================================================
            NAVIGATION BAR
            ========================================================================= */}
        <header className="sticky top-0 z-50 backdrop-blur-md bg-black/30 border-b border-white/5 transition-all duration-300">
          <nav className="flex items-center justify-between px-6 md:px-12 lg:px-16 py-4 sm:py-5">
            {/* Logo (left) */}
            <a href="#home" className="flex items-center gap-3 group focus:outline-none">
              <svg
                viewBox="0 0 480 480"
                className="w-8 h-8 fill-white transition-transform duration-300 group-hover:scale-105"
                xmlns="http://www.w3.org/2000/svg"
                aria-label="GeoPulse AI Logo"
              >
                <path d="M480 240a240 240 0 0 0-240 240 240 240 0 0 0 240-240Z M240 0A240 240 0 0 0 0 240 240 240 0 0 0 240 0Z M480 240A240 240 0 0 0 240 0a240 240 0 0 0 240 240Z M240 480A240 240 0 0 0 0 240a240 240 0 0 0 240 240Z" />
              </svg>
              <div className="flex items-center gap-1.5">
                <span className="text-white text-xl font-bold tracking-wider">
                  GeoPulse
                </span>
                <span className="text-xs font-semibold px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 tracking-widest uppercase">
                  AI
                </span>
              </div>
            </a>

            {/* Desktop Nav Links (center) */}
            <div className="hidden lg:flex items-center gap-8">
              {navLinks.map((link) => (
                <a
                  key={link.name}
                  href={link.href}
                  className="text-white/80 hover:text-white text-sm tracking-wide transition-colors duration-200 hover:drop-shadow-[0_0_8px_rgba(6,182,212,0.6)]"
                >
                  {link.name}
                </a>
              ))}
            </div>

            {/* Action CTA Button (right) */}
            <div className="hidden lg:flex items-center">
              <a
                href="#enter"
                className="bg-gradient-to-r from-emerald-400 to-cyan-500 text-white text-sm font-semibold px-6 py-2.5 rounded-full inline-flex items-center gap-2 hover:shadow-lg hover:shadow-cyan-500/30 transition-all duration-300 transform hover:-translate-y-0.5 active:translate-y-0 cursor-pointer"
              >
                <span>ENTER PLATFORM</span>
                <ArrowRight className="w-4 h-4" />
              </a>
            </div>

            {/* Mobile Hamburger Button */}
            <button
              type="button"
              onClick={toggleMobileMenu}
              aria-label={mobileMenuOpen ? "Close navigation menu" : "Open navigation menu"}
              className="lg:hidden relative z-[60] w-10 h-10 flex items-center justify-center text-white focus:outline-none cursor-pointer"
            >
              <Menu
                className={`w-6 h-6 absolute transition-all duration-300 ${mobileMenuOpen
                    ? 'opacity-0 rotate-90 scale-75 pointer-events-none'
                    : 'opacity-100 rotate-0 scale-100'
                  }`}
              />
              <X
                className={`w-6 h-6 absolute transition-all duration-300 ${mobileMenuOpen
                    ? 'opacity-100 rotate-0 scale-100'
                    : 'opacity-0 -rotate-90 scale-75 pointer-events-none'
                  }`}
              />
            </button>
          </nav>

          {/* Mobile Menu Overlay & Panel */}
          {mobileMenuOpen && (
            <>
              {/* Backdrop */}
              <div
                onClick={closeMobileMenu}
                className={`fixed inset-0 z-40 bg-black/75 backdrop-blur-md transition-opacity duration-400 ${menuVisible ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
                  }`}
              />

              {/* Menu Panel */}
              <div className="absolute left-0 right-0 top-[64px] z-50 overflow-hidden rounded-b-2xl">
                {/* Backdrop blur layer */}
                <div className="absolute inset-0 backdrop-blur-3xl bg-neutral-950/95 rounded-b-2xl border-b border-white/10" />

                {/* Content layer */}
                <div className="relative z-10 flex flex-col items-center py-10 px-6 gap-6">
                  {navLinks.map((link, index) => (
                    <a
                      key={link.name}
                      href={link.href}
                      onClick={closeMobileMenu}
                      style={{
                        transitionDelay: menuVisible ? `${350 + index * 50}ms` : '0ms',
                      }}
                      className={`text-lg sm:text-xl font-light tracking-[0.08em] text-white/80 hover:text-white transition-all duration-400 ease-out transform ${menuVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'
                        }`}
                    >
                      {link.name}
                    </a>
                  ))}

                  {/* Staggered Enter Platform button */}
                  <a
                    href="#enter"
                    onClick={closeMobileMenu}
                    style={{
                      transitionDelay: menuVisible ? `${350 + navLinks.length * 50}ms` : '0ms',
                    }}
                    className={`mt-3 bg-gradient-to-r from-emerald-400 to-cyan-500 text-white text-sm font-semibold px-8 py-3 rounded-full inline-flex items-center gap-2 shadow-lg shadow-cyan-500/25 transition-all duration-400 ease-out transform ${menuVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'
                      }`}
                  >
                    <span>ENTER PLATFORM</span>
                    <ArrowRight className="w-4 h-4" />
                  </a>
                </div>
              </div>
            </>
          )}
        </header>

        {/* =========================================================================
            HERO SECTION
            ========================================================================= */}
        <section id="home" className="relative flex-1 flex flex-col items-center justify-center text-center px-4 sm:px-6 pt-12 sm:pt-16 pb-16 sm:pb-24 overflow-hidden">

          {/* Small Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-cyan-500/30 bg-cyan-950/40 backdrop-blur-md mb-6 sm:mb-8 text-cyan-300 text-[10px] sm:text-xs font-semibold tracking-[0.2em] uppercase shadow-lg shadow-cyan-900/30">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400"></span>
            </span>
            <span>AI-POWERED GEOPOLITICAL INTELLIGENCE</span>
          </div>

          {/* Main Heading (two lines) */}
          <h1 className="text-white text-2xl xs:text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-light leading-tight sm:leading-tight tracking-tight max-w-5xl mb-2 sm:mb-3">
            UNDERSTAND GLOBAL EVENTS.
          </h1>
          <h1 className="text-transparent bg-clip-text bg-gradient-to-r from-white via-cyan-100 to-cyan-400 text-2xl xs:text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-light leading-tight sm:leading-tight tracking-tight max-w-5xl mb-6 sm:mb-8">
            MEASURE THEIR REAL-WORLD IMPACT.
          </h1>

          {/* Description */}
          <p className="text-white/75 text-sm sm:text-base md:text-lg font-light leading-relaxed max-w-2xl sm:max-w-3xl mb-8 sm:mb-10 px-2">
            GeoPulse AI analyzes global conflicts, economic signals, energy markets, trade routes, and public sentiment to reveal how geopolitical events impact the world.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-5 w-full sm:w-auto mb-12 sm:mb-16">
            <a
              href="#events"
              className="w-full sm:w-auto bg-gradient-to-r from-emerald-400 to-cyan-500 text-white font-semibold text-xs sm:text-sm tracking-wider px-8 py-3.5 rounded-full inline-flex items-center justify-center gap-2.5 shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 transition-all duration-300 transform hover:-translate-y-0.5 active:translate-y-0 cursor-pointer"
            >
              <span>EXPLORE INTELLIGENCE</span>
              <ArrowRight className="w-4 h-4" />
            </a>

            <a
              href="#events"
              className="w-full sm:w-auto liquid-glass text-white text-[11px] sm:text-xs tracking-[0.18em] font-medium px-8 py-3.5 rounded-full uppercase transition-all duration-300 hover:scale-105 active:scale-95 inline-flex items-center justify-center cursor-pointer"
            >
              VIEW GLOBAL EVENTS
            </a>

            <a
              href="#news-intelligence"
              className="w-full sm:w-auto px-8 py-3.5 rounded-full border border-cyan-500/30 bg-cyan-950/40 hover:bg-cyan-500/20 text-cyan-300 text-[11px] sm:text-xs tracking-[0.18em] font-medium uppercase transition-all duration-300 hover:scale-105 active:scale-95 inline-flex items-center justify-center cursor-pointer gap-2 shadow-lg shadow-cyan-950/50"
            >
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>NARRATIVE ML</span>
            </a>

            <a
              href="#india-impact"
              className="w-full sm:w-auto px-8 py-3.5 rounded-full border border-emerald-500/30 bg-emerald-950/40 hover:bg-emerald-500/20 text-emerald-300 text-[11px] sm:text-xs tracking-[0.18em] font-medium uppercase transition-all duration-300 hover:scale-105 active:scale-95 inline-flex items-center justify-center cursor-pointer gap-2 shadow-lg shadow-emerald-950/50"
            >
              <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
              <span>INDIA IMPACT 🇮🇳</span>
            </a>

            <a
              href="#india-energy-risk"
              className="w-full sm:w-auto px-8 py-3.5 rounded-full border border-amber-500/30 bg-amber-950/40 hover:bg-amber-500/20 text-amber-300 text-[11px] sm:text-xs tracking-[0.18em] font-medium uppercase transition-all duration-300 hover:scale-105 active:scale-95 inline-flex items-center justify-center cursor-pointer gap-2 shadow-lg shadow-amber-950/50"
            >
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span>ENERGY RISK ⚡</span>
            </a>

            <a
              href="#commodity-shock"
              className="w-full sm:w-auto px-8 py-3.5 rounded-full border border-cyan-500/30 bg-cyan-950/40 hover:bg-cyan-500/20 text-cyan-300 text-[11px] sm:text-xs tracking-[0.18em] font-medium uppercase transition-all duration-300 hover:scale-105 active:scale-95 inline-flex items-center justify-center cursor-pointer gap-2 shadow-lg shadow-cyan-950/50"
            >
              <Droplet className="w-3.5 h-3.5 text-cyan-400" />
              <span>COMMODITY SHOCK 🛢️</span>
            </a>

            <a
              href="#trade-dependency"
              className="w-full sm:w-auto px-8 py-3.5 rounded-full border border-emerald-500/30 bg-emerald-950/40 hover:bg-emerald-500/20 text-emerald-300 text-[11px] sm:text-xs tracking-[0.18em] font-medium uppercase transition-all duration-300 hover:scale-105 active:scale-95 inline-flex items-center justify-center cursor-pointer gap-2 shadow-lg shadow-emerald-950/50"
            >
              <Globe className="w-3.5 h-3.5 text-emerald-400" />
              <span>TRADE RISK 🌐</span>
            </a>

            <a
              href="#supply-route"
              className="w-full sm:w-auto px-8 py-3.5 rounded-full border border-cyan-500/30 bg-cyan-950/40 hover:bg-cyan-500/20 text-cyan-300 text-[11px] sm:text-xs tracking-[0.18em] font-medium uppercase transition-all duration-300 hover:scale-105 active:scale-95 inline-flex items-center justify-center cursor-pointer gap-2 shadow-lg shadow-cyan-950/50"
            >
              <Ship className="w-3.5 h-3.5 text-cyan-400" />
              <span>SUPPLY ROUTES 🚢</span>
            </a>
          </div>

          {/* =========================================================================
              HERO VISUAL: LIVE GEOPOLITICAL INTELLIGENCE GLOBE OVERLAY
              ========================================================================= */}
          <div className="relative w-full max-w-5xl mx-auto flex items-center justify-center min-h-[380px] sm:min-h-[460px] md:min-h-[520px] select-none my-2 sm:my-6">

            {/* SVG Geopolitical Overlay Canvas (routes, hot spots, connection arcs) */}
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none overflow-visible"
              viewBox="0 0 1000 500"
              preserveAspectRatio="xMidYMid meet"
            >
              <defs>
                {/* Radial gradient for glowing conflict nodes */}
                <radialGradient id="redGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity="0.9" />
                  <stop offset="40%" stopColor="#ef4444" stopOpacity="0.4" />
                  <stop offset="100%" stopColor="#ef4444" stopOpacity="0" />
                </radialGradient>
                <radialGradient id="cyanGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.9" />
                  <stop offset="40%" stopColor="#06b6d4" stopOpacity="0.4" />
                  <stop offset="100%" stopColor="#06b6d4" stopOpacity="0" />
                </radialGradient>
                <radialGradient id="amberGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.9" />
                  <stop offset="40%" stopColor="#f59e0b" stopOpacity="0.4" />
                  <stop offset="100%" stopColor="#f59e0b" stopOpacity="0" />
                </radialGradient>
                {/* Linear gradient for animated shipping route lines */}
                <linearGradient id="routeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.1" />
                  <stop offset="50%" stopColor="#10b981" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.1" />
                </linearGradient>
              </defs>

              {/* Geopolitical Shipping / Trade Connection Arcs */}
              <path
                d="M 230,220 Q 380,140 500,240 T 780,260"
                fill="none"
                stroke="url(#routeGradient)"
                strokeWidth="1.8"
                className="animate-dash-flow opacity-70"
              />
              <path
                d="M 280,310 Q 420,380 620,290 T 840,210"
                fill="none"
                stroke="#06b6d4"
                strokeWidth="1.5"
                className="animate-dash-flow opacity-60"
              />
              <path
                d="M 480,180 Q 560,110 680,190"
                fill="none"
                stroke="#10b981"
                strokeWidth="1.6"
                className="animate-dash-flow opacity-75"
              />

              {/* Pulsing Conflict Hotspot 1: Red Sea / Bab-el-Mandeb */}
              <g transform="translate(480, 250)">
                <circle cx="0" cy="0" r="32" fill="url(#redGlow)" className="animate-radar" />
                <circle cx="0" cy="0" r="6" fill="#ef4444" />
                <circle cx="0" cy="0" r="2.5" fill="#ffffff" />
                <line x1="0" y1="0" x2="-24" y2="-28" stroke="#ef4444" strokeWidth="1" strokeDasharray="2 2" opacity="0.8" />
                <rect x="-140" y="-46" width="112" height="18" rx="3" fill="rgba(15,23,42,0.85)" stroke="rgba(239,68,68,0.5)" strokeWidth="0.8" />
                <text x="-84" y="-33" fill="#fca5a5" fontSize="8" fontFamily="sans-serif" textAnchor="middle" fontWeight="bold" letterSpacing="0.05em">
                  HOTSPOT: RED SEA
                </text>
              </g>

              {/* Pulsing Hotspot 2: Strait of Hormuz (Energy Chokepoint) */}
              <g transform="translate(560, 210)">
                <circle cx="0" cy="0" r="30" fill="url(#amberGlow)" className="animate-radar" style={{ animationDelay: '1s' }} />
                <circle cx="0" cy="0" r="5" fill="#f59e0b" />
                <circle cx="0" cy="0" r="2" fill="#ffffff" />
                <line x1="0" y1="0" x2="28" y2="-24" stroke="#f59e0b" strokeWidth="1" strokeDasharray="2 2" opacity="0.8" />
                <rect x="30" y="-38" width="124" height="18" rx="3" fill="rgba(15,23,42,0.85)" stroke="rgba(245,158,11,0.5)" strokeWidth="0.8" />
                <text x="92" y="-25" fill="#fde68a" fontSize="8" fontFamily="sans-serif" textAnchor="middle" fontWeight="bold" letterSpacing="0.05em">
                  CHOKEPOINT: HORMUZ
                </text>
              </g>

              {/* Pulsing Hotspot 3: Strait of Malacca (Trade Route) */}
              <g transform="translate(740, 280)">
                <circle cx="0" cy="0" r="28" fill="url(#cyanGlow)" className="animate-radar" style={{ animationDelay: '1.8s' }} />
                <circle cx="0" cy="0" r="5" fill="#06b6d4" />
                <circle cx="0" cy="0" r="2" fill="#ffffff" />
                <line x1="0" y1="0" x2="24" y2="28" stroke="#06b6d4" strokeWidth="1" strokeDasharray="2 2" opacity="0.8" />
                <rect x="26" y="24" width="130" height="18" rx="3" fill="rgba(15,23,42,0.85)" stroke="rgba(6,182,212,0.5)" strokeWidth="0.8" />
                <text x="91" y="37" fill="#67e8f9" fontSize="8" fontFamily="sans-serif" textAnchor="middle" fontWeight="bold" letterSpacing="0.05em">
                  AIS TRADE FLOW: HIGH
                </text>
              </g>

              {/* Node 4: Black Sea Corridor */}
              <g transform="translate(470, 150)">
                <circle cx="0" cy="0" r="20" fill="url(#amberGlow)" className="animate-radar" style={{ animationDelay: '0.6s' }} />
                <circle cx="0" cy="0" r="4" fill="#fbbf24" />
                <line x1="0" y1="0" x2="-25" y2="-20" stroke="#fbbf24" strokeWidth="1" strokeDasharray="2 2" opacity="0.7" />
                <rect x="-132" y="-34" width="104" height="16" rx="3" fill="rgba(15,23,42,0.85)" stroke="rgba(251,191,36,0.4)" strokeWidth="0.8" />
                <text x="-80" y="-23" fill="#fef08a" fontSize="7.5" fontFamily="sans-serif" textAnchor="middle" fontWeight="bold" letterSpacing="0.05em">
                  GRAIN CORRIDOR
                </text>
              </g>
            </svg>

            {/* Central Globe Intelligence Radar Grid Ring */}
            <div className="relative w-64 h-64 sm:w-80 sm:h-80 md:w-96 md:h-96 rounded-full border border-cyan-500/20 flex items-center justify-center">
              <div className="absolute inset-0 rounded-full border border-cyan-400/10 animate-ping opacity-25" style={{ animationDuration: '6s' }} />
              <div className="absolute inset-4 rounded-full border border-emerald-400/15" />
              <div className="absolute inset-12 rounded-full border border-dashed border-cyan-500/20 animate-spin" style={{ animationDuration: '45s' }} />

              {/* Central live signal status circle */}
              <div className="text-center z-10 px-4 py-2 rounded-xl bg-black/40 backdrop-blur-md border border-white/10 shadow-2xl">
                <div className="flex items-center justify-center gap-1.5 mb-0.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-[10px] tracking-[0.2em] font-bold text-cyan-400 uppercase">
                    ACTIVE MONITORING
                  </span>
                </div>
                <div className="text-xs text-white/70 font-mono tracking-wider">
                  2,840+ SENSORS ONLINE
                </div>
              </div>
            </div>

            {/* =========================================================================
                4 FLOATING INTELLIGENCE CARDS (Positioned around the main visual)
                ========================================================================= */}

            {/* Card 1: ENERGY RISK (Top-Left) */}
            <div className="absolute top-1 sm:top-2 left-1 xs:left-2 sm:left-4 md:left-8 z-20 animate-float-slow">
              <div className="glass-card rounded-xl p-2.5 sm:p-4 text-left w-36 xs:w-44 sm:w-56 transition-all duration-300">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[9px] xs:text-[10px] sm:text-xs font-bold tracking-[0.15em] text-white/70 uppercase">
                    ENERGY RISK
                  </span>
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[8px] xs:text-[9px] sm:text-[10px] font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30">
                    Elevated
                  </span>
                </div>
                <div className="text-base sm:text-xl font-bold text-white tracking-tight flex items-baseline gap-1.5 sm:gap-2">
                  <span>+14.8%</span>
                  <span className="text-[9px] sm:text-[10px] text-amber-400/90 font-normal">Brent Volatility</span>
                </div>
                <div className="w-full bg-white/10 h-1 rounded-full mt-2 overflow-hidden">
                  <div className="bg-gradient-to-r from-amber-500 to-red-500 h-full w-[72%]" />
                </div>
                <div className="text-[8px] xs:text-[9px] sm:text-[10px] text-white/50 mt-1.5 flex items-center justify-between font-mono">
                  <span>Hormuz Transit</span>
                  <span className="text-amber-300">Alert L3</span>
                </div>
              </div>
            </div>

            {/* Card 2: TRADE IMPACT (Top-Right) */}
            <div className="absolute top-1 sm:top-2 right-1 xs:right-2 sm:right-4 md:right-8 z-20 animate-float-reverse">
              <div className="glass-card rounded-xl p-2.5 sm:p-4 text-left w-36 xs:w-44 sm:w-56 transition-all duration-300">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[9px] xs:text-[10px] sm:text-xs font-bold tracking-[0.15em] text-white/70 uppercase">
                    TRADE IMPACT
                  </span>
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[8px] xs:text-[9px] sm:text-[10px] font-bold bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
                    Moderate
                  </span>
                </div>
                <div className="text-base sm:text-xl font-bold text-white tracking-tight flex items-baseline gap-1.5 sm:gap-2">
                  <span>+4.2 Days</span>
                  <span className="text-[9px] sm:text-[10px] text-cyan-400/90 font-normal">Suez Latency</span>
                </div>
                <div className="w-full bg-white/10 h-1 rounded-full mt-2 overflow-hidden">
                  <div className="bg-gradient-to-r from-cyan-500 to-blue-500 h-full w-[48%]" />
                </div>
                <div className="text-[8px] xs:text-[9px] sm:text-[10px] text-white/50 mt-1.5 flex items-center justify-between font-mono">
                  <span>Shipping Re-routes</span>
                  <span className="text-cyan-300">18.4% Cape</span>
                </div>
              </div>
            </div>

            {/* Card 3: GLOBAL EVENTS (Bottom-Left) */}
            <div className="absolute bottom-1 sm:bottom-2 left-1 xs:left-2 sm:left-4 md:left-8 z-20 animate-float-reverse">
              <div className="glass-card rounded-xl p-2.5 sm:p-4 text-left w-36 xs:w-44 sm:w-56 transition-all duration-300">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[9px] xs:text-[10px] sm:text-xs font-bold tracking-[0.15em] text-white/70 uppercase">
                    GLOBAL EVENTS
                  </span>
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400"></span>
                  </span>
                </div>
                <div className="text-base sm:text-xl font-bold text-white tracking-tight">
                  24 Active Signals
                </div>
                <div className="text-[9px] sm:text-[10px] text-white/60 mt-1 line-clamp-1">
                  12 High-impact clusters
                </div>
                <div className="flex items-center gap-1.5 mt-2">
                  <span className="h-1.5 w-5 sm:w-6 rounded bg-red-500/80" />
                  <span className="h-1.5 w-7 sm:w-8 rounded bg-amber-500/80" />
                  <span className="h-1.5 w-10 sm:w-12 rounded bg-cyan-500/80" />
                </div>
              </div>
            </div>

            {/* Card 4: PUBLIC SENTIMENT (Bottom-Right) */}
            <div className="absolute bottom-1 sm:bottom-2 right-1 xs:right-2 sm:right-4 md:right-8 z-20 animate-float-slow">
              <div className="glass-card rounded-xl p-2.5 sm:p-4 text-left w-36 xs:w-44 sm:w-56 transition-all duration-300">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[9px] xs:text-[10px] sm:text-xs font-bold tracking-[0.15em] text-white/70 uppercase">
                    PUBLIC SENTIMENT
                  </span>
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[8px] xs:text-[9px] sm:text-[10px] font-bold bg-purple-500/20 text-purple-400 border border-purple-500/30">
                    Shifting
                  </span>
                </div>
                <div className="text-base sm:text-xl font-bold text-white tracking-tight">
                  -28.4 Index
                </div>
                <div className="text-[9px] sm:text-[10px] text-white/60 mt-1 line-clamp-1">
                  Narrative polarization high
                </div>
                <div className="w-full bg-white/10 h-1 rounded-full mt-2 overflow-hidden">
                  <div className="bg-gradient-to-r from-purple-500 to-pink-500 h-full w-[64%]" />
                </div>
              </div>
            </div>

          </div>
        </section>


        {/* =========================================================================
            FEATURE SECTION: GLOBAL EVENTS. CONNECTED INTELLIGENCE.
            ========================================================================= */}
        <section id="events" className="relative px-6 md:px-12 lg:px-16 py-16 sm:py-24 border-t border-white/10 bg-black/40 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto">

            {/* Header */}
            <div className="text-center mb-12 sm:mb-16">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/40 border border-cyan-500/20 text-cyan-400 text-[10px] sm:text-xs font-semibold tracking-[0.2em] uppercase mb-4">
                <Activity className="w-3.5 h-3.5 text-cyan-400" />
                <span>CROSS-DOMAIN SENSING ENGINE</span>
              </div>
              <h2 className="text-white text-2xl xs:text-3xl sm:text-4xl md:text-5xl font-light tracking-tight">
                GLOBAL EVENTS. CONNECTED INTELLIGENCE.
              </h2>
              <p className="text-white/60 text-xs sm:text-sm md:text-base font-light max-w-2xl mx-auto mt-3">
                Synthesizing fragmented conflict indicators, trade flow telemetry, and macroeconomic signals into a unified real-time impact matrix.
              </p>
            </div>

            {/* 4 Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">

              {/* Card 1: EVENT INTELLIGENCE */}
              <div className="glass-card rounded-2xl p-6 sm:p-7 flex flex-col justify-between text-left transition-all duration-300 hover:-translate-y-1">
                <div>
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-red-500/20 to-orange-500/10 border border-red-500/30 flex items-center justify-center text-red-400 mb-5">
                    <ShieldAlert className="w-6 h-6" />
                  </div>
                  <h3 className="text-white text-base sm:text-lg font-bold tracking-wider mb-2">
                    EVENT INTELLIGENCE
                  </h3>
                  <p className="text-white/65 text-xs sm:text-sm font-light leading-relaxed">
                    Track major global conflicts and geopolitical events in real time with multi-source verification and severity ranking.
                  </p>
                </div>
                <div className="pt-6 mt-6 border-t border-white/10 flex items-center justify-between text-[11px] text-white/50 font-mono">
                  <span>LIVE CLUSTERS</span>
                  <span className="text-red-400 font-bold">14 Active Zones</span>
                </div>
              </div>

              {/* Card 2: ECONOMIC IMPACT */}
              <div className="glass-card rounded-2xl p-6 sm:p-7 flex flex-col justify-between text-left transition-all duration-300 hover:-translate-y-1">
                <div>
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500/20 to-cyan-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 mb-5">
                    <TrendingUp className="w-6 h-6" />
                  </div>
                  <h3 className="text-white text-base sm:text-lg font-bold tracking-wider mb-2">
                    ECONOMIC IMPACT
                  </h3>
                  <p className="text-white/65 text-xs sm:text-sm font-light leading-relaxed">
                    Understand how events influence commodities, markets, and regional economies through quantitative correlation models.
                  </p>
                </div>
                <div className="pt-6 mt-6 border-t border-white/10 flex items-center justify-between text-[11px] text-white/50 font-mono">
                  <span>MARKET SENSITIVITY</span>
                  <span className="text-emerald-400 font-bold">98.4% Confidence</span>
                </div>
              </div>

              {/* Card 3: TRADE & ENERGY RISK */}
              <div className="glass-card rounded-2xl p-6 sm:p-7 flex flex-col justify-between text-left transition-all duration-300 hover:-translate-y-1">
                <div>
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 mb-5">
                    <Radio className="w-6 h-6" />
                  </div>
                  <h3 className="text-white text-base sm:text-lg font-bold tracking-wider mb-2">
                    TRADE & ENERGY RISK
                  </h3>
                  <p className="text-white/65 text-xs sm:text-sm font-light leading-relaxed">
                    Visualize potential disruptions to shipping routes, trade, and energy supply across major maritime corridors and pipelines.
                  </p>
                </div>
                <div className="pt-6 mt-6 border-t border-white/10 flex items-center justify-between text-[11px] text-white/50 font-mono">
                  <span>AIS CHOKEPOINTS</span>
                  <span className="text-cyan-400 font-bold">8 Critical Passages</span>
                </div>
              </div>

              {/* Card 4: NARRATIVE VS REALITY */}
              <div className="glass-card rounded-2xl p-6 sm:p-7 flex flex-col justify-between text-left transition-all duration-300 hover:-translate-y-1">
                <div>
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500/20 to-pink-500/10 border border-purple-500/30 flex items-center justify-center text-purple-400 mb-5">
                    <Activity className="w-6 h-6" />
                  </div>
                  <h3 className="text-white text-base sm:text-lg font-bold tracking-wider mb-2">
                    NARRATIVE VS REALITY
                  </h3>
                  <p className="text-white/65 text-xs sm:text-sm font-light leading-relaxed">
                    Compare public narratives with measurable economic and geopolitical signals to detect perception anomalies and misinformation.
                  </p>
                </div>
                <div className="pt-6 mt-6 border-t border-white/10 flex items-center justify-between text-[11px] text-white/50 font-mono">
                  <span>DIVERGENCE INDEX</span>
                  <span className="text-purple-400 font-bold">3.2σ Anomaly</span>
                </div>
              </div>

            </div>

            {/* Operational Feature 1: Global Event & Conflict Intelligence Engine */}
            <div className="mt-20 pt-16 border-t border-white/10">
              <GlobalEventIntelligence />
            </div>

            {/* Operational Feature 2: News & Narrative Classification Engine (Syllabus ML) */}
            <div id="news-intelligence" className="mt-20 pt-16 border-t border-white/10">
              <NewsNarrativeIntelligence onNavigateToEvents={() => {
                const el = document.getElementById('events');
                if (el) el.scrollIntoView({ behavior: 'smooth' });
              }} />
            </div>

            {/* Operational Feature 3: News Sentiment Analysis Engine (Supervised ML) */}
            <div id="sentiment-intelligence" className="mt-20 pt-16 border-t border-white/10">
              <SentimentIntelligence />
            </div>

            {/* Operational Feature 4: India Economic Impact Intelligence (Supervised Regression ML) */}
            <div id="india-impact" className="mt-20 pt-16 border-t border-white/10">
              <IndiaEconomicImpact />
            </div>

            {/* Operational Feature 5: India Energy Supply Risk Intelligence (Supervised Multi-Class ML) */}
            <div id="india-energy-risk" className="mt-20 pt-16 border-t border-white/10">
              <IndiaEnergyRiskIntelligence />
            </div>

            {/* Operational Feature 6: India Oil & Commodity Shock Intelligence (Multiple Linear Regression) */}
            <div id="commodity-shock" className="mt-20 pt-16 border-t border-white/10">
              <IndiaCommodityShockIntelligence />
            </div>

            {/* Operational Feature 7: India Trade Dependency & Country Risk (K-Means Clustering) */}
            <div id="trade-dependency" className="mt-20 pt-16 border-t border-white/10">
              <IndiaTradeDependencyRisk />
            </div>

            {/* Operational Feature 8: India Supply-Route Disruption Intelligence (Supervised Multi-Class ML) */}
            <div id="supply-route" className="mt-20 pt-16 border-t border-white/10">
              <IndiaSupplyRouteDisruption />
            </div>
          </div>
        </section>


        {/* =========================================================================
            INTELLIGENCE FLOW SECTION:
            GLOBAL SIGNALS → AI ANALYSIS → IMPACT → RISK → INTELLIGENCE
            ========================================================================= */}
        <section id="flow" className="relative px-6 md:px-12 lg:px-16 py-16 sm:py-24 border-t border-white/10 bg-black/50 backdrop-blur-md">
          <div className="max-w-6xl mx-auto text-center">

            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/40 border border-cyan-500/20 text-cyan-400 text-[10px] sm:text-xs font-semibold tracking-[0.2em] uppercase mb-4">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>END-TO-END PIPELINE</span>
            </div>

            <h2 className="text-white text-2xl xs:text-3xl sm:text-4xl md:text-5xl font-light tracking-tight mb-4">
              THE INTELLIGENCE FLOW
            </h2>
            <p className="text-white/60 text-xs sm:text-sm md:text-base font-light max-w-2xl mx-auto mb-12 sm:mb-16">
              How GeoPulse AI converts raw global telemetry into verified strategic impact intelligence.
            </p>

            {/* Pipeline Steps Container */}
            <div className="flex flex-col lg:flex-row items-center justify-between gap-4 lg:gap-2">

              {/* Step 1 */}
              <div className="flex-1 w-full glass-card rounded-xl p-5 text-center relative group hover:border-cyan-400/40 transition-all duration-300">
                <div className="text-[10px] font-mono tracking-widest text-cyan-400/80 mb-2">STAGE 01</div>
                <div className="text-white text-sm sm:text-base font-bold tracking-wider mb-1">GLOBAL SIGNALS</div>
                <div className="text-[11px] text-white/55 font-light leading-relaxed">
                  Satellite imagery, AIS ship telemetry, news wires, sensor logs
                </div>
              </div>

              {/* Arrow */}
              <div className="text-cyan-400/60 lg:rotate-0 rotate-90 flex items-center justify-center p-1">
                <ChevronRight className="w-5 h-5 animate-pulse" />
              </div>

              {/* Step 2 */}
              <div className="flex-1 w-full glass-card rounded-xl p-5 text-center relative group hover:border-cyan-400/40 transition-all duration-300">
                <div className="text-[10px] font-mono tracking-widest text-emerald-400/80 mb-2">STAGE 02</div>
                <div className="text-white text-sm sm:text-base font-bold tracking-wider mb-1">AI ANALYSIS</div>
                <div className="text-[11px] text-white/55 font-light leading-relaxed">
                  Deep neural clustering, anomaly detection, sentiment modeling
                </div>
              </div>

              {/* Arrow */}
              <div className="text-cyan-400/60 lg:rotate-0 rotate-90 flex items-center justify-center p-1">
                <ChevronRight className="w-5 h-5 animate-pulse" />
              </div>

              {/* Step 3 */}
              <div className="flex-1 w-full glass-card rounded-xl p-5 text-center relative group hover:border-cyan-400/40 transition-all duration-300">
                <div className="text-[10px] font-mono tracking-widest text-cyan-400/80 mb-2">STAGE 03</div>
                <div className="text-white text-sm sm:text-base font-bold tracking-wider mb-1">IMPACT</div>
                <div className="text-[11px] text-white/55 font-light leading-relaxed">
                  Supply chain delay, commodity pricing, currency shocks
                </div>
              </div>

              {/* Arrow */}
              <div className="text-cyan-400/60 lg:rotate-0 rotate-90 flex items-center justify-center p-1">
                <ChevronRight className="w-5 h-5 animate-pulse" />
              </div>

              {/* Step 4 */}
              <div className="flex-1 w-full glass-card rounded-xl p-5 text-center relative group hover:border-cyan-400/40 transition-all duration-300">
                <div className="text-[10px] font-mono tracking-widest text-amber-400/80 mb-2">STAGE 04</div>
                <div className="text-white text-sm sm:text-base font-bold tracking-wider mb-1">RISK</div>
                <div className="text-[11px] text-white/55 font-light leading-relaxed">
                  Escalation probability matrices, critical chokepoint exposure
                </div>
              </div>

              {/* Arrow */}
              <div className="text-cyan-400/60 lg:rotate-0 rotate-90 flex items-center justify-center p-1">
                <ChevronRight className="w-5 h-5 animate-pulse" />
              </div>

              {/* Step 5 */}
              <div className="flex-1 w-full glass-card rounded-xl p-5 text-center relative group border-cyan-500/40 bg-cyan-950/30 transition-all duration-300">
                <div className="text-[10px] font-mono tracking-widest text-cyan-300 mb-2">STAGE 05</div>
                <div className="text-white text-sm sm:text-base font-bold tracking-wider mb-1">INTELLIGENCE</div>
                <div className="text-[11px] text-cyan-200/70 font-light leading-relaxed">
                  Decisive actionable briefings and predictive scenario models
                </div>
              </div>

            </div>

          </div>
        </section>


        {/* =========================================================================
            FINAL CTA SECTION
            ========================================================================= */}
        <section id="enter" className="relative px-6 md:px-12 lg:px-16 py-20 sm:py-28 border-t border-white/10 text-center overflow-hidden">
          <div className="max-w-4xl mx-auto relative z-10 flex flex-col items-center">

            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-emerald-500/30 bg-emerald-950/40 backdrop-blur-md mb-6 text-emerald-300 text-[10px] sm:text-xs font-semibold tracking-[0.2em] uppercase">
              <Zap className="w-3.5 h-3.5 text-emerald-400" />
              <span>GLOBAL RECONNAISSANCE READY</span>
            </div>

            <h2 className="text-white text-2xl xs:text-3xl sm:text-4xl md:text-5xl font-light tracking-tight max-w-3xl mb-4 sm:mb-6">
              TURN GLOBAL UNCERTAINTY INTO INTELLIGENCE.
            </h2>

            <p className="text-white/70 text-sm sm:text-base md:text-lg font-light leading-relaxed max-w-2xl mb-8 sm:mb-10">
              Explore the signals behind global events and understand their real-world impact.
            </p>

            <a
              href="#platform"
              className="bg-gradient-to-r from-emerald-400 to-cyan-500 text-white font-bold text-xs sm:text-sm tracking-wider px-10 py-4 rounded-full inline-flex items-center gap-3 shadow-xl shadow-cyan-500/30 hover:shadow-cyan-500/50 hover:scale-105 active:scale-95 transition-all duration-300 uppercase cursor-pointer"
            >
              <span>ENTER GEOPULSE AI</span>
              <ArrowRight className="w-4 h-4" />
            </a>

          </div>
        </section>


        {/* =========================================================================
            FOOTER SECTION (Exact cloned layout and styling)
            ========================================================================= */}
        <footer className="relative z-10 px-4 sm:px-6 md:px-12 lg:px-16 pb-8 sm:pb-10 pt-10 sm:pt-16 border-t border-white/10 bg-black/60 backdrop-blur-lg">
          <div className="max-w-7xl mx-auto">
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-6 sm:gap-8 lg:gap-6">

              {/* 4 Link Columns */}
              {footerColumns.map((col) => (
                <div key={col.title} className="flex flex-col text-left">
                  <h3 className="text-white text-[10px] sm:text-xs font-bold tracking-[0.15em] mb-3 sm:mb-4">
                    {col.title}
                  </h3>
                  <ul className="space-y-2 sm:space-y-2.5 list-none p-0 m-0">
                    {col.links.map((link) => (
                      <li key={link}>
                        <a
                          href={`#${link.toLowerCase().replace(/[^a-z0-9]/g, '-')}`}
                          className="text-white/50 hover:text-white/80 text-[10px] sm:text-xs transition-colors duration-200 block"
                        >
                          {link}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}

              {/* Newsletter + Social Column */}
              <div className="col-span-2 md:col-span-4 lg:col-span-2 flex flex-col text-left">
                <h3 className="text-white text-[10px] sm:text-xs font-bold tracking-[0.15em] mb-3 sm:mb-4">
                  JOIN FOR EXCLUSIVE BRIEFINGS
                </h3>

                <form onSubmit={handleSubscribe} className="flex flex-row max-w-sm w-full">
                  <input
                    type="email"
                    required
                    value={emailInput}
                    onChange={(e) => setEmailInput(e.target.value)}
                    placeholder="Type your email for intelligence updates"
                    className="bg-white text-neutral-900 placeholder:text-neutral-400 text-[11px] sm:text-xs px-3.5 py-2.5 rounded-l-md outline-none w-full min-w-0 font-normal transition-all focus:ring-1 focus:ring-cyan-500"
                  />
                  <button
                    type="submit"
                    className="bg-gradient-to-r from-emerald-400 to-cyan-500 hover:opacity-95 text-white font-bold tracking-wider text-[11px] sm:text-xs px-4 sm:px-5 py-2.5 rounded-r-md whitespace-nowrap transition-all duration-200 uppercase cursor-pointer"
                  >
                    SEND IT
                  </button>
                </form>

                {subscribed && (
                  <p className="text-emerald-400 text-[11px] mt-1.5 animate-pulse">
                    Confirmed. Strategic briefings will be delivered to your inbox.
                  </p>
                )}

                <h4 className="text-white text-[10px] sm:text-xs font-bold tracking-[0.15em] mt-5 sm:mt-6 mb-3">
                  CONNECT
                </h4>

                <div className="flex items-center gap-3 text-white/50">
                  <a
                    href="#facebook"
                    aria-label="Facebook"
                    className="hover:text-white transition-colors duration-200"
                  >
                    <Facebook className="w-4 h-4" />
                  </a>
                  <a
                    href="#twitter"
                    aria-label="Twitter"
                    className="hover:text-white transition-colors duration-200"
                  >
                    <Twitter className="w-4 h-4" />
                  </a>
                  <a
                    href="#dribbble"
                    aria-label="Dribbble"
                    className="hover:text-white transition-colors duration-200"
                  >
                    <Dribbble className="w-4 h-4" />
                  </a>
                  <a
                    href="#youtube"
                    aria-label="Youtube"
                    className="hover:text-white transition-colors duration-200"
                  >
                    <Youtube className="w-4 h-4" />
                  </a>
                  <a
                    href="#linkedin"
                    aria-label="LinkedIn"
                    className="hover:text-white transition-colors duration-200"
                  >
                    <Linkedin className="w-4 h-4" />
                  </a>
                  <a
                    href="#instagram"
                    aria-label="Instagram"
                    className="hover:text-white transition-colors duration-200"
                  >
                    <Instagram className="w-4 h-4" />
                  </a>
                </div>
              </div>
            </div>

            {/* Bottom Disclaimer & Copyright */}
            <div className="pt-8 sm:pt-10 mt-8 sm:mt-10 border-t border-white/5 flex flex-col sm:flex-row items-center justify-between text-[10px] sm:text-xs text-white/40 gap-3">
              <p>
                © 2026 GeoPulse AI Inc. All rights reserved. Global Conflict Impact Intelligence Platform.
              </p>
              <div className="flex items-center gap-4 text-white/40">
                <a href="#privacy" className="hover:text-white/70 transition-colors">Privacy Policy</a>
                <span>•</span>
                <a href="#terms" className="hover:text-white/70 transition-colors">Terms of Reconnaissance</a>
                <span>•</span>
                <a href="#status" className="hover:text-emerald-400 transition-colors flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  Systems Operational
                </a>
              </div>
            </div>
          </div>
        </footer>

      </div>
    </div>
  );
}
