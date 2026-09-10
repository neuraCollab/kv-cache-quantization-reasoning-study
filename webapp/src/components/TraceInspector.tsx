import React, { useState, useMemo } from 'react';
import { SAMPLE_TRACES } from '../data/sampleTraces';
import { QUANT_METHODS, MODELS, CATEGORIES } from '../data/taxonomyData';
import { findFDP } from '../utils/fdpFinder';
import { GitCompare, AlertTriangle, CheckCircle, RefreshCw, Zap, ArrowRight, Info } from 'lucide-react';
import { CategoryLetter } from '../types';

export const TraceInspector: React.FC = () => {
  const [selectedTraceId, setSelectedTraceId] = useState<string>(SAMPLE_TRACES[0].id);
  const [customMode, setCustomMode] = useState<boolean>(false);

  const selectedSample = useMemo(() => {
    return SAMPLE_TRACES.find((t) => t.id === selectedTraceId) || SAMPLE_TRACES[0];
  }, [selectedTraceId]);

  const [baselineText, setBaselineText] = useState<string>(selectedSample.baselineText);
  const [quantText, setQuantText] = useState<string>(selectedSample.quantText);
  const [groundTruth, setGroundTruth] = useState<string>(selectedSample.groundTruth);
  const [selectedModel, setSelectedModel] = useState<string>(selectedSample.model);
  const [selectedQuant, setSelectedQuant] = useState<string>(selectedSample.quantMethod);

  // When sample trace changes, update inputs if not in custom mode
  const handleSelectSample = (id: string) => {
    setSelectedTraceId(id);
    const sample = SAMPLE_TRACES.find((t) => t.id === id);
    if (sample) {
      setBaselineText(sample.baselineText);
      setQuantText(sample.quantText);
      setGroundTruth(sample.groundTruth);
      setSelectedModel(sample.model);
      setSelectedQuant(sample.quantMethod);
      setCustomMode(false);
    }
  };

  // Compute live FDP
  const fdpResult = useMemo(() => {
    return findFDP(baselineText, quantText, groundTruth, {
      contextWindow: 20,
      resyncLookahead: 10,
      maxCosmeticSkips: 5,
    });
  }, [baselineText, quantText, groundTruth]);

  // Determine error category
  const activeCategory = useMemo(() => {
    if (!customMode && selectedSample) {
      return CATEGORIES.find((c) => c.letter === selectedSample.expectedCategory);
    }
    // Heuristic inference for custom traces
    if (fdpResult.quantTruncated || quantText.endsWith('...') || !quantText.includes('\\boxed')) {
      return CATEGORIES.find((c) => c.letter === 'E');
    }
    if (quantText.includes('repeat') || /(.{10,})\1{2,}/.test(quantText)) {
      return CATEGORIES.find((c) => c.letter === 'F');
    }
    if (quantText.includes('theorem') || quantText.includes('identity')) {
      return CATEGORIES.find((c) => c.letter === 'D');
    }
    if (quantText.includes('restart') || quantText.includes('drop') || quantText.includes('instead')) {
      return CATEGORIES.find((c) => c.letter === 'C');
    }
    return CATEGORIES.find((c) => c.letter === 'A') || CATEGORIES[0];
  }, [customMode, selectedSample, fdpResult, quantText]);

  const quantInfo = QUANT_METHODS.find((q) => q.id === selectedQuant) || QUANT_METHODS[1];
  const modelInfo = MODELS.find((m) => m.id === selectedModel) || MODELS[0];

  const getBoxedBadge = () => {
    switch (fdpResult.boxedMatch) {
      case 'both_correct':
        return <span className="px-2.5 py-1 text-xs rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">Both Correct</span>;
      case 'baseline_only':
        return <span className="px-2.5 py-1 text-xs rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/30">Baseline Only Correct</span>;
      case 'quant_only':
        return <span className="px-2.5 py-1 text-xs rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/30">Quant Only Correct</span>;
      case 'both_wrong':
        return <span className="px-2.5 py-1 text-xs rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/30">Both Wrong</span>;
      default:
        return <span className="px-2.5 py-1 text-xs rounded-full bg-slate-700 text-slate-300">No Boxed Answer</span>;
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Controls Bar */}
      <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-4 sm:p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
              <GitCompare className="w-5 h-5 text-indigo-400" />
              Trace-Level Divergence Inspector
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Pinpoint the exact token where quantized KV-cache reasoning diverges from the BF16 reference baseline.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              id="toggle-custom-trace-btn"
              onClick={() => setCustomMode(!customMode)}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
                customMode
                  ? 'bg-indigo-600 text-white border-indigo-500'
                  : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
              }`}
            >
              {customMode ? 'Custom Mode Active' : 'Edit / Custom Traces'}
            </button>
          </div>
        </div>

        {/* Preset Traces Selector */}
        {!customMode && (
          <div className="mt-4 pt-4 border-t border-slate-700/60">
            <label className="block text-xs font-medium text-slate-400 mb-2">
              Select Curated Empirical Trace Pair:
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
              {SAMPLE_TRACES.map((trace) => {
                const isSelected = trace.id === selectedTraceId;
                const cat = CATEGORIES.find((c) => c.letter === trace.expectedCategory);
                return (
                  <button
                    key={trace.id}
                    id={`select-trace-${trace.id}`}
                    onClick={() => handleSelectSample(trace.id)}
                    className={`text-left p-3 rounded-lg border transition-all text-xs ${
                      isSelected
                        ? 'bg-indigo-950/40 border-indigo-500 text-slate-100 shadow-sm'
                        : 'bg-slate-800/40 border-slate-700/70 text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-1 mb-1">
                      <span className="font-semibold text-slate-200 truncate">{trace.benchmark}</span>
                      {cat && (
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${cat.badgeBg}`}>
                          Cat {cat.letter}: {cat.name}
                        </span>
                      )}
                    </div>
                    <div className="line-clamp-2 text-slate-400">{trace.problem}</div>
                    <div className="mt-2 flex items-center justify-between text-[11px] text-slate-400 border-t border-slate-700/40 pt-1.5">
                      <span>Method: <strong className="text-slate-300">{trace.quantMethod}</strong></span>
                      <span>GT: <strong className="text-slate-300">{trace.groundTruth}</strong></span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Model and Configuration Badges */}
        <div className="mt-4 flex flex-wrap items-center gap-3 text-xs bg-slate-900/60 p-3 rounded-lg border border-slate-800">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400">Model:</span>
            <span className="font-medium text-slate-200 bg-slate-800 px-2 py-0.5 rounded border border-slate-700">
              {modelInfo.name} ({modelInfo.params})
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400">Quant Method:</span>
            <span className="font-medium px-2 py-0.5 rounded border" style={{ color: quantInfo.color, borderColor: `${quantInfo.color}40`, backgroundColor: `${quantInfo.color}15` }}>
              {quantInfo.name} ({quantInfo.bits}-bit)
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400">Engine:</span>
            <span className="font-mono text-slate-300 uppercase">{quantInfo.engine}</span>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <span className="text-slate-400">Boxed Status:</span>
            {getBoxedBadge()}
          </div>
        </div>
      </div>

      {/* FDP Metric Banner */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-slate-800/40 border border-slate-700 rounded-xl p-4 flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 shrink-0">
            <Zap className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs text-slate-400 font-medium">First Divergence (FDP)</div>
            <div className="text-lg font-bold text-slate-100">
              {fdpResult.fdpTokenIdx !== null ? `Token #${fdpResult.fdpTokenIdx}` : 'No Divergence'}
            </div>
          </div>
        </div>

        <div className="bg-slate-800/40 border border-slate-700 rounded-xl p-4 flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400 shrink-0">
            <RefreshCw className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs text-slate-400 font-medium">Cosmetic Skips</div>
            <div className="text-lg font-bold text-slate-100">
              {fdpResult.cosmeticSkipped} <span className="text-xs text-slate-400 font-normal">re-syncs</span>
            </div>
          </div>
        </div>

        <div className="bg-slate-800/40 border border-slate-700 rounded-xl p-4 flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400 shrink-0">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs text-slate-400 font-medium">Divergent Token</div>
            <div className="text-sm font-mono font-bold text-amber-300 truncate max-w-[140px]" title={fdpResult.divergentTokenQuant}>
              "{fdpResult.divergentTokenQuant || 'None'}"
            </div>
          </div>
        </div>

        <div className="bg-slate-800/40 border border-slate-700 rounded-xl p-4 flex items-center gap-3">
          <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 border`} style={{ backgroundColor: `${activeCategory?.color}15`, borderColor: `${activeCategory?.color}40`, color: activeCategory?.color }}>
            <span className="font-bold text-sm">{activeCategory?.letter}</span>
          </div>
          <div>
            <div className="text-xs text-slate-400 font-medium">Error Category</div>
            <div className="text-sm font-semibold text-slate-100">
              {activeCategory?.name}
            </div>
          </div>
        </div>
      </div>

      {/* Synchronized Side-by-Side Trace Views */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Baseline (BF16) */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
          <div className="bg-slate-800/90 px-4 py-3 border-b border-slate-700/80 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
              <span className="font-semibold text-xs text-slate-200">
                Baseline Trace: BF16 (Unquantized Reference)
              </span>
            </div>
            <span className="text-[11px] font-mono text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-800/40">
              Reference Truth
            </span>
          </div>

          <div className="p-4">
            {customMode ? (
              <textarea
                value={baselineText}
                onChange={(e) => setBaselineText(e.target.value)}
                rows={14}
                className="w-full bg-slate-950 text-slate-200 font-mono text-xs p-3 rounded-lg border border-slate-800 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                placeholder="Enter baseline reasoning trace..."
              />
            ) : (
              <div className="bg-slate-950 p-4 rounded-lg font-mono text-xs text-slate-300 leading-relaxed overflow-x-auto max-h-[420px] whitespace-pre-wrap select-text">
                {fdpResult.fdpTokenIdx !== null ? (
                  <div>
                    <span className="text-slate-400">{fdpResult.commonPrefix}</span>
                    <span className="bg-emerald-500/20 text-emerald-300 px-1 py-0.5 rounded border border-emerald-500/40 font-bold">
                      {fdpResult.divergentTokenBaseline}
                    </span>
                    <span className="text-slate-300">
                      {baselineText.slice(fdpResult.commonPrefix.length + (fdpResult.divergentTokenBaseline?.length || 0))}
                    </span>
                  </div>
                ) : (
                  baselineText
                )}
              </div>
            )}
          </div>
        </div>

        {/* Quantized Trace */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
          <div className="bg-slate-800/90 px-4 py-3 border-b border-slate-700/80 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: quantInfo.color }}></span>
              <span className="font-semibold text-xs text-slate-200">
                Quantized Trace: {quantInfo.name} ({quantInfo.bits}-bit KV Cache)
              </span>
            </div>
            <span className="text-[11px] font-mono text-rose-400 bg-rose-950/40 px-2 py-0.5 rounded border border-rose-800/40">
              Evaluated
            </span>
          </div>

          <div className="p-4">
            {customMode ? (
              <textarea
                value={quantText}
                onChange={(e) => setQuantText(e.target.value)}
                rows={14}
                className="w-full bg-slate-950 text-slate-200 font-mono text-xs p-3 rounded-lg border border-slate-800 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                placeholder="Enter quantized reasoning trace..."
              />
            ) : (
              <div className="bg-slate-950 p-4 rounded-lg font-mono text-xs text-slate-300 leading-relaxed overflow-x-auto max-h-[420px] whitespace-pre-wrap select-text">
                {fdpResult.fdpTokenIdx !== null ? (
                  <div>
                    <span className="text-slate-400">{fdpResult.commonPrefix}</span>
                    <span className="bg-rose-500/20 text-rose-300 px-1 py-0.5 rounded border border-rose-500/40 font-bold animate-pulse">
                      {fdpResult.divergentTokenQuant}
                    </span>
                    <span className="text-slate-200">
                      {quantText.slice(fdpResult.commonPrefix.length + (fdpResult.divergentTokenQuant?.length || 0))}
                    </span>
                  </div>
                ) : (
                  quantText
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Divergence Analysis Card */}
      <div className="bg-slate-800/40 border border-slate-700 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-slate-100 mb-2 flex items-center gap-2">
          <Info className="w-4 h-4 text-indigo-400" />
          Judge Diagnosis & Taxonomy Rationale
        </h3>
        <p className="text-xs text-slate-300 leading-relaxed">
          {selectedSample?.rationale || activeCategory?.description}
        </p>

        <div className="mt-4 pt-4 border-t border-slate-700/60 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div>
            <div className="text-slate-400 font-medium mb-1">Baseline Context Window (±15 tokens):</div>
            <div className="bg-slate-950 p-2.5 rounded border border-slate-800 font-mono text-emerald-300 text-[11px]">
              "{fdpResult.baselineContext || '(none)'}"
            </div>
          </div>
          <div>
            <div className="text-slate-400 font-medium mb-1">Quantized Context Window (±15 tokens):</div>
            <div className="bg-slate-950 p-2.5 rounded border border-slate-800 font-mono text-rose-300 text-[11px]">
              "{fdpResult.quantContext || '(none)'}"
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
