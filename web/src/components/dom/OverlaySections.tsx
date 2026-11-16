"use client";

import React from "react";
import { Shield, Sparkles, Cpu, Layers } from "lucide-react";

export function NavigationHeader() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-8 py-5 backdrop-blur-md bg-black/40 border-b border-white/5">
      <div className="flex items-center space-x-3">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-cyan-500 to-purple-600 flex items-center justify-center shadow-[0_0_15px_rgba(0,240,255,0.4)]">
          <Shield className="w-5 h-5 text-white" />
        </div>
        <div>
          <span className="font-extrabold tracking-wider text-sm bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400">
            TRUTH MATRIX
          </span>
          <span className="text-[10px] block font-mono text-cyan-400/70 tracking-widest uppercase">
            3D News Intelligence
          </span>
        </div>
      </div>

      <nav className="hidden md:flex items-center space-x-8 text-xs font-mono tracking-widest text-gray-400 uppercase">
        <a href="#intelligence-world" className="hover:text-cyan-400 transition-colors">
          01 // Intelligence World
        </a>
        <a href="#news-investigation" className="hover:text-cyan-400 transition-colors">
          02 // Investigation
        </a>
        <a href="#ai-processing" className="hover:text-cyan-400 transition-colors">
          03 // AI Processing
        </a>
        <a href="#prediction-results" className="hover:text-cyan-400 transition-colors">
          04 // Predictions
        </a>
      </nav>

      <div className="flex items-center space-x-3">
        <div className="px-3 py-1 rounded-full border border-cyan-500/30 bg-cyan-500/10 text-cyan-400 text-xs font-mono flex items-center space-x-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span>CORE ONLINE</span>
        </div>
      </div>
    </header>
  );
}

export function OverlaySections() {
  const scrollToNext = (targetId: string) => {
    const el = document.getElementById(targetId);
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="relative z-10 w-full pointer-events-auto">
      {/* SECTION 1: Intelligence World (Landing Hero) */}
      <section
        id="intelligence-world"
        className="min-h-screen flex flex-col items-center justify-center text-center px-6 relative"
      >
        <div className="max-w-4xl space-y-6 pt-20">
          <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full border border-cyan-500/30 bg-cyan-950/40 backdrop-blur-md text-cyan-300 text-xs font-mono tracking-widest uppercase">
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            <span>Autonomous Linguistic Verification System</span>
          </div>

          <h1 className="text-5xl md:text-7xl lg:text-8xl font-black tracking-tight uppercase leading-none">
            <span className="block text-transparent bg-clip-text bg-gradient-to-b from-white via-gray-100 to-gray-400">
              Fake News
            </span>
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 shadow-[0_0_80px_rgba(0,240,255,0.3)]">
              Detection
            </span>
          </h1>

          <p className="text-lg md:text-xl text-gray-300 max-w-2xl mx-auto font-light leading-relaxed">
            AI-powered intelligence environment classifying disinformation vectors and synthetic media patterns in real-time 3D space.
          </p>

          <div className="pt-4 flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              onClick={() => scrollToNext("news-investigation")}
              className="px-8 py-3.5 rounded-xl font-mono text-sm tracking-widest uppercase font-bold text-black bg-gradient-to-r from-cyan-400 to-blue-500 hover:from-cyan-300 hover:to-blue-400 transition-all duration-300 shadow-[0_0_30px_rgba(0,240,255,0.5)] transform hover:scale-105 active:scale-95"
            >
              Enter Experience
            </button>

            <a
              href="http://localhost:8501"
              target="_blank"
              rel="noreferrer"
              className="px-7 py-3.5 rounded-xl font-mono text-sm tracking-widest uppercase font-medium text-gray-300 border border-white/10 hover:border-cyan-500/40 bg-black/40 backdrop-blur-md hover:bg-white/5 transition-all duration-300"
            >
              Streamlit Console →
            </a>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-10 left-1/2 transform -translate-x-1/2 flex flex-col items-center space-y-2 opacity-60">
          <span className="text-[11px] font-mono tracking-widest text-cyan-400 uppercase">
            Scroll to Navigate
          </span>
          <div className="w-5 h-8 rounded-full border-2 border-cyan-500/40 flex items-start justify-center p-1">
            <div className="w-1 h-2 rounded-full bg-cyan-400 animate-bounce" />
          </div>
        </div>
      </section>

      {/* SECTION 2: News Investigation */}
      <section
        id="news-investigation"
        className="min-h-screen flex items-center justify-start px-8 md:px-20 py-24"
      >
        <div className="max-w-xl backdrop-blur-xl bg-black/60 p-8 md:p-10 rounded-2xl border border-white/10 shadow-[0_0_50px_rgba(0,0,0,0.8)] space-y-5">
          <div className="inline-flex items-center space-x-2 text-cyan-400 font-mono text-xs tracking-widest uppercase">
            <Layers className="w-4 h-4" />
            <span>02 // Multi-Vector News Investigation</span>
          </div>

          <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-white">
            Forensic Text Ingestion & Parsing
          </h2>

          <p className="text-gray-300 leading-relaxed text-sm md:text-base font-light">
            Articles are scrutinized across 20,000 statistical n-gram dimensions. Stylistic divergence maps indicate political polarization, sensational hyperbole, and synthetic news templates.
          </p>

          <div className="grid grid-cols-2 gap-3 pt-2 text-xs font-mono">
            <div className="p-3 rounded-lg bg-white/5 border border-white/5">
              <span className="text-gray-400 block">VOCABULARY</span>
              <span className="text-cyan-400 font-bold text-base">20,000 Terms</span>
            </div>
            <div className="p-3 rounded-lg bg-white/5 border border-white/5">
              <span className="text-gray-400 block">N-GRAM RANGE</span>
              <span className="text-purple-400 font-bold text-base">Unigram + Bigram</span>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 3: AI Processing */}
      <section
        id="ai-processing"
        className="min-h-screen flex items-center justify-end px-8 md:px-20 py-24"
      >
        <div className="max-w-xl backdrop-blur-xl bg-black/60 p-8 md:p-10 rounded-2xl border border-white/10 shadow-[0_0_50px_rgba(0,0,0,0.8)] space-y-5">
          <div className="inline-flex items-center space-x-2 text-purple-400 font-mono text-xs tracking-widest uppercase">
            <Cpu className="w-4 h-4" />
            <span>03 // AI Machine Learning Engine</span>
          </div>

          <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-white">
            Logistic Regression Core
          </h2>

          <p className="text-gray-300 leading-relaxed text-sm md:text-base font-light">
            Balanced L2 regularization operates at microsecond inference latencies. Features are evaluated against trained linguistic signatures with zero validation data leakage.
          </p>

          <div className="space-y-2 pt-2 text-xs font-mono">
            <div className="flex justify-between text-gray-400">
              <span>MODEL ACCURACY</span>
              <span className="text-emerald-400 font-bold">99.96%</span>
            </div>
            <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
              <div className="bg-emerald-400 h-full w-[99.96%]" />
            </div>

            <div className="flex justify-between text-gray-400 pt-2">
              <span>ROC-AUC SCORE</span>
              <span className="text-cyan-400 font-bold">0.9998</span>
            </div>
            <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
              <div className="bg-cyan-400 h-full w-[99.98%]" />
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 4: Prediction Results */}
      <section
        id="prediction-results"
        className="min-h-screen flex flex-col items-center justify-center text-center px-6 py-24"
      >
        <div className="max-w-2xl backdrop-blur-xl bg-black/70 p-8 md:p-12 rounded-3xl border border-cyan-500/30 shadow-[0_0_80px_rgba(0,240,255,0.2)] space-y-6">
          <div className="inline-flex items-center space-x-2 text-cyan-400 font-mono text-xs tracking-widest uppercase">
            <span>04 // Final Diagnostic Output</span>
          </div>

          <h2 className="text-3xl md:text-5xl font-black tracking-tight text-white uppercase">
            Truth Classification
          </h2>

          <p className="text-gray-300 text-sm md:text-base font-light">
            Real-time classification produces calibrated probability confidence scores for Fake and Synthetic article streams.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-left font-mono">
            <div className="p-4 rounded-xl bg-red-950/30 border border-red-500/30">
              <span className="text-red-400 font-bold block text-sm">LABEL 0 // FAKE</span>
              <p className="text-[11px] text-gray-300 mt-1">
                Sensationalized political commentary, social conspiracy vectors.
              </p>
            </div>
            <div className="p-4 rounded-xl bg-emerald-950/30 border border-emerald-500/30">
              <span className="text-emerald-400 font-bold block text-sm">LABEL 1 // SYNTHETIC</span>
              <p className="text-[11px] text-gray-300 mt-1">
                AI-generated institutional updates and press-release style templates.
              </p>
            </div>
          </div>

          <div className="pt-4">
            <a
              href="http://localhost:8501"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center px-8 py-3.5 rounded-xl font-mono text-sm tracking-widest uppercase font-bold text-black bg-cyan-400 hover:bg-cyan-300 transition-all shadow-[0_0_25px_rgba(0,240,255,0.6)]"
            >
              Open Interactive Inference UI →
            </a>
          </div>
        </div>

        <footer className="mt-16 text-xs text-gray-400 font-mono tracking-widest max-w-xl">
          ⚠️ DISCLAIMER: The model detects stylistic dataset divergences between scraped fake news and synthetic AI-generated content. It is an experimental intelligence research prototype.
        </footer>
      </section>
    </div>
  );
}
