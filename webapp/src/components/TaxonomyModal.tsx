import React from 'react';
import { CATEGORIES } from '../data/taxonomyData';
import { X, BookOpen, CheckCircle, Code } from 'lucide-react';

interface TaxonomyModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const TaxonomyModal: React.FC<TaxonomyModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fade-in">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col shadow-2xl">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-800/80">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-indigo-400" />
            <h3 className="text-base font-semibold text-slate-100">
              Taxonomy Reference Guide (TAXONOMY_V1)
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto space-y-6">
          <p className="text-xs text-slate-300 leading-relaxed">
            The 6 mutually exclusive error categories used by the judge to classify the First Divergence Point (FDP) of quantized KV-cache reasoning traces relative to their unquantized BF16 baseline.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {CATEGORIES.map((cat) => (
              <div
                key={cat.letter}
                className="bg-slate-800/40 border border-slate-700/80 rounded-xl p-4 space-y-2"
              >
                <div className="flex items-center gap-2.5">
                  <span
                    className="w-7 h-7 rounded-lg flex items-center justify-center font-mono font-bold text-xs border shrink-0"
                    style={{
                      backgroundColor: `${cat.color}15`,
                      borderColor: `${cat.color}40`,
                      color: cat.color,
                    }}
                  >
                    {cat.letter}
                  </span>
                  <div>
                    <h4 className="text-sm font-semibold text-slate-100">{cat.name}</h4>
                    <span className="text-[10px] text-slate-400 font-mono">Category {cat.letter}</span>
                  </div>
                </div>

                <p className="text-xs text-slate-300 leading-normal">{cat.description}</p>

                <div className="pt-2 border-t border-slate-700/60">
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                    Canonical Examples:
                  </span>
                  <ul className="space-y-1">
                    {cat.examples.map((ex, i) => (
                      <li key={i} className="text-xs text-slate-400 flex items-start gap-1.5 font-mono">
                        <span className="text-indigo-400 shrink-0">•</span>
                        <span>{ex}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>

          {/* JSON Output Schema Reference */}
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 mb-2">
              <Code className="w-4 h-4 text-indigo-400" />
              Judge Output Schema Requirement
            </div>
            <pre className="text-xs font-mono text-indigo-300 overflow-x-auto leading-relaxed">
{`{
  "category": "A" | "B" | "C" | "D" | "E" | "F",
  "confidence": 0.0 to 1.0,
  "rationale": "At most 2 sentences describing the local error",
  "affected_span": "3-15 word quote from quantized trace"
}`}
            </pre>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-slate-800/40 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 text-xs font-medium rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-colors"
          >
            Close Guide
          </button>
        </div>
      </div>
    </div>
  );
};
