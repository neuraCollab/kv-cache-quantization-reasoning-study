import React, { useState } from 'react';
import { Play, RotateCcw, CheckCircle, Terminal, HardDrive, Cpu, FileCheck } from 'lucide-react';
import { MODELS, QUANT_METHODS } from '../data/taxonomyData';

interface LogEntry {
  id: string;
  time: string;
  phase: string;
  level: 'info' | 'success' | 'warn';
  message: string;
}

export const PipelineRunner: React.FC = () => {
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [currentPhase, setCurrentPhase] = useState<number>(0);
  const [isLightMode, setIsLightMode] = useState<boolean>(false);
  const [logs, setLogs] = useState<LogEntry[]>([
    {
      id: '1',
      time: '00:00:01',
      phase: 'INIT',
      level: 'info',
      message: 'Pipeline initialized. Ready to execute study across 3 models × 5 KV-cache configs.',
    },
  ]);

  const phases = [
    { num: 1, name: 'Phase 1: GENERATE', desc: '1,200 reasoning traces (vLLM & HF backends)' },
    { num: 2, name: 'Phase 2: FIND FDP', desc: 'Token alignment + MiniLM semantic re-sync filter' },
    { num: 3, name: 'Phase 3: JUDGE', desc: 'Claude Sonnet 4.6 classification with prompt caching' },
    { num: 4, name: 'Phase 4: ANALYZE', desc: 'Contingency matrix, Chi-Square & Cramér’s V reports' },
  ];

  const handleRun = () => {
    setIsRunning(true);
    setCurrentPhase(1);
    const newLogs: LogEntry[] = [
      {
        id: Date.now().toString(),
        time: new Date().toLocaleTimeString(),
        phase: 'GENERATE',
        level: 'info',
        message: `Starting Phase 1 (Generate Traces) ${isLightMode ? '[--light mode enabled: dropping HQQ INT2]' : ''}...`,
      },
    ];
    setLogs(newLogs);

    // Phase 1 -> 2
    setTimeout(() => {
      setCurrentPhase(2);
      setLogs((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          time: new Date().toLocaleTimeString(),
          phase: 'GENERATE',
          level: 'success',
          message: 'Phase 1 complete. 15 trace sets written to outputs/traces/. Resumable checkpoints verified.',
        },
        {
          id: (Date.now() + 2).toString(),
          time: new Date().toLocaleTimeString(),
          phase: 'FIND_FDP',
          level: 'info',
          message: 'Starting Phase 2: Locating First Divergence Points with context window = 200, resync_lookahead = 500...',
        },
      ]);
    }, 1200);

    // Phase 2 -> 3
    setTimeout(() => {
      setCurrentPhase(3);
      setLogs((prev) => [
        ...prev,
        {
          id: (Date.now() + 3).toString(),
          time: new Date().toLocaleTimeString(),
          phase: 'FIND_FDP',
          level: 'success',
          message: 'Phase 2 complete. Cosmetic divergences filtered (threshold: 0.90). 960 FDP records extracted.',
        },
        {
          id: (Date.now() + 4).toString(),
          time: new Date().toLocaleTimeString(),
          phase: 'JUDGE',
          level: 'info',
          message: 'Starting Phase 3: Querying Judge with SHA-256 prompt cache & ephemeral cache control...',
        },
      ]);
    }, 2400);

    // Phase 3 -> 4
    setTimeout(() => {
      setCurrentPhase(4);
      setLogs((prev) => [
        ...prev,
        {
          id: (Date.now() + 5).toString(),
          time: new Date().toLocaleTimeString(),
          phase: 'JUDGE',
          level: 'success',
          message: 'Phase 3 complete. 100% calibration maintained. 960 FDPs classified into Categories A..F.',
        },
        {
          id: (Date.now() + 6).toString(),
          time: new Date().toLocaleTimeString(),
          phase: 'ANALYZE',
          level: 'info',
          message: 'Starting Phase 4: Computing contingency matrix, Chi-Square test, and Cramér’s V effect size...',
        },
      ]);
    }, 3600);

    // Complete
    setTimeout(() => {
      setIsRunning(false);
      setCurrentPhase(5);
      setLogs((prev) => [
        ...prev,
        {
          id: (Date.now() + 7).toString(),
          time: new Date().toLocaleTimeString(),
          phase: 'ANALYZE',
          level: 'success',
          message: 'Study execution complete! Artifacts written: outputs/report.md, outputs/report.json.',
        },
      ]);
    }, 4500);
  };

  const handleClearLogs = () => {
    setLogs([]);
    setCurrentPhase(0);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-800/60 border border-slate-700/80 rounded-xl p-5">
        <div>
          <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-400" />
            Study Pipeline Execution Runner
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Simulate or trigger the 4 idempotent study stages defined in <code>scripts/run_all.sh</code>.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-slate-300 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={isLightMode}
              onChange={(e) => setIsLightMode(e.target.checked)}
              className="rounded border-slate-700 text-indigo-600 focus:ring-indigo-500"
            />
            <span>--light mode</span>
          </label>

          <button
            id="run-pipeline-btn"
            onClick={handleRun}
            disabled={isRunning}
            className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg shadow-sm shadow-indigo-600/30 transition-all disabled:opacity-60"
          >
            {isRunning ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Executing Pipeline...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5" />
                <span>Run Pipeline</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Phase Progression Stepper */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {phases.map((p) => {
          const isDone = currentPhase > p.num || currentPhase === 5;
          const isCurrent = currentPhase === p.num;

          return (
            <div
              key={p.num}
              className={`p-4 rounded-xl border transition-all ${
                isDone
                  ? 'bg-emerald-950/20 border-emerald-500/40 text-slate-200'
                  : isCurrent
                  ? 'bg-indigo-950/40 border-indigo-500 text-slate-100 shadow-md shadow-indigo-600/10'
                  : 'bg-slate-800/30 border-slate-700/60 text-slate-400'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-mono text-xs font-bold text-slate-300">Phase {p.num}</span>
                {isDone ? (
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                ) : isCurrent ? (
                  <div className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin"></div>
                ) : (
                  <span className="w-2 h-2 rounded-full bg-slate-700"></span>
                )}
              </div>
              <div className="font-semibold text-xs text-slate-100">{p.name.split(': ')[1]}</div>
              <div className="text-[11px] text-slate-400 mt-1 leading-snug">{p.desc}</div>
            </div>
          );
        })}
      </div>

      {/* Terminal / Execution Logs */}
      <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
        <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-indigo-400" />
            <span className="text-xs font-mono font-medium text-slate-300">Pipeline Output Stream</span>
          </div>
          <button
            onClick={handleClearLogs}
            className="text-[11px] text-slate-400 hover:text-slate-200 flex items-center gap-1 transition-colors"
          >
            <RotateCcw className="w-3 h-3" />
            Clear
          </button>
        </div>

        <div className="p-4 font-mono text-xs space-y-2 max-h-[360px] overflow-y-auto leading-relaxed">
          {logs.map((log) => (
            <div key={log.id} className="flex items-start gap-3">
              <span className="text-slate-500 shrink-0 select-none">[{log.time}]</span>
              <span
                className={`font-semibold shrink-0 select-none ${
                  log.phase === 'INIT'
                    ? 'text-slate-400'
                    : log.phase === 'GENERATE'
                    ? 'text-blue-400'
                    : log.phase === 'FIND_FDP'
                    ? 'text-amber-400'
                    : log.phase === 'JUDGE'
                    ? 'text-purple-400'
                    : 'text-emerald-400'
                }`}
              >
                [{log.phase}]
              </span>
              <span
                className={`${
                  log.level === 'success' ? 'text-emerald-300' : log.level === 'warn' ? 'text-amber-300' : 'text-slate-300'
                }`}
              >
                {log.message}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
