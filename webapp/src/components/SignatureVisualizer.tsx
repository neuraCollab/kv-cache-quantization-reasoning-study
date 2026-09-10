import React from 'react';
import { PRECOMPUTED_STUDY_DATA } from '../data/sampleTraces';
import { CATEGORIES, QUANT_METHODS } from '../data/taxonomyData';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts';
import { TrendingUp, Layers, CheckCircle2, Cpu } from 'lucide-react';

export const SignatureVisualizer: React.FC = () => {
  const methods = PRECOMPUTED_STUDY_DATA.methods;
  const counts = PRECOMPUTED_STUDY_DATA.counts;

  // Format data for Recharts
  const chartData = methods.map((method, idx) => {
    const row = counts[idx];
    const total = row.reduce((a, b) => a + b, 0);
    const quantInfo = QUANT_METHODS.find((q) => q.id === method);

    return {
      method: quantInfo?.name || method,
      A: Number(((row[0] / total) * 100).toFixed(1)),
      B: Number(((row[1] / total) * 100).toFixed(1)),
      C: Number(((row[2] / total) * 100).toFixed(1)),
      D: Number(((row[3] / total) * 100).toFixed(1)),
      E: Number(((row[4] / total) * 100).toFixed(1)),
      F: Number(((row[5] / total) * 100).toFixed(1)),
    };
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-5">
        <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
          <Layers className="w-5 h-5 text-indigo-400" />
          Failure-Signature Profiles per Method
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Visual comparison illustrating how each quantization algorithm breaks reasoning in a distinct, characteristic way.
        </p>
      </div>

      {/* Main Chart */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Relative Error Distribution (% of Divergences per Method)
          </span>
          <span className="text-xs text-slate-400 font-mono">Normalized percentages</span>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="method" stroke="#94a3b8" fontSize={11} tickLine={false} />
              <YAxis stroke="#94a3b8" fontSize={11} unit="%" tickLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '8px',
                  fontSize: '11px',
                  color: '#f8fafc',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              <Bar dataKey="A" name="A: Arithmetic" fill="#ef4444" stackId="stack" />
              <Bar dataKey="B" name="B: Logical" fill="#f97316" stackId="stack" />
              <Bar dataKey="C" name="C: Strategy-switch" fill="#eab308" stackId="stack" />
              <Bar dataKey="D" name="D: Hallucination" fill="#a855f7" stackId="stack" />
              <Bar dataKey="E" name="E: Premature-trunc" fill="#3b82f6" stackId="stack" />
              <Bar dataKey="F" name="F: Repetition-loop" fill="#06b6d4" stackId="stack" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Architectural & Empirical Findings Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-slate-800/40 border border-slate-700/80 rounded-xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <span className="w-3 h-3 rounded-full bg-indigo-500"></span>
            <h3 className="text-sm font-semibold text-slate-100">
              FP8 E5M2 vs FP8 E4M3 Mantissa Trade-off
            </h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            On Ada GPUs (RTX 4090), <strong>FP8 E5M2</strong> allocates 5 bits to exponent and only 2 bits to mantissa.
            This wider dynamic range avoids overflow, but numerical precision severely degrades, triggering a <strong>39% arithmetic error rate</strong>.
            Conversely, <strong>FP8 E4M3</strong> (3 mantissa bits) preserves calculations better, but suffers from exponent clipping, causing reasoning drifts and strategy-switches (Category C).
          </p>
        </div>

        <div className="bg-slate-800/40 border border-slate-700/80 rounded-xl p-5">
          <div className="flex items-center gap-2 mb-3">
            <span className="w-3 h-3 rounded-full bg-rose-500"></span>
            <h3 className="text-sm font-semibold text-slate-100">
              HQQ INT4 vs INT2 Phase Transition
            </h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            While <strong>HQQ INT4</strong> retains solid coherence with balanced error rates, <strong>HQQ INT2</strong> exhibits a catastrophic breakdown.
            At 2-bit KV precision, attention weights lose sparsity and contrast, leading to <strong>37% degenerate repetition loops</strong> (Category F) and <strong>21% theorem hallucinations</strong> (Category D).
          </p>
        </div>
      </div>

      {/* Category Signatures Reference */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {CATEGORIES.map((cat) => (
          <div key={cat.letter} className="bg-slate-900/60 border border-slate-800 p-3.5 rounded-xl text-center">
            <div
              className="w-7 h-7 rounded-lg mx-auto flex items-center justify-center font-mono font-bold text-xs mb-2 border"
              style={{
                backgroundColor: `${cat.color}15`,
                borderColor: `${cat.color}40`,
                color: cat.color,
              }}
            >
              {cat.letter}
            </div>
            <div className="font-semibold text-xs text-slate-200">{cat.name}</div>
            <div className="text-[10px] text-slate-400 mt-1 line-clamp-2">{cat.description}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
