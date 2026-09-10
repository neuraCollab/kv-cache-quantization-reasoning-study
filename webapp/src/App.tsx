import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { TraceInspector } from './components/TraceInspector';
import { ContingencyMatrix } from './components/ContingencyMatrix';
import { SignatureVisualizer } from './components/SignatureVisualizer';
import { CalibrationSuite } from './components/CalibrationSuite';
import { PipelineRunner } from './components/PipelineRunner';
import { TaxonomyModal } from './components/TaxonomyModal';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('inspector');
  const [isTaxonomyOpen, setIsTaxonomyOpen] = useState<boolean>(false);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenTaxonomy={() => setIsTaxonomyOpen(true)}
      />

      {/* Data-provenance disclosure */}
      <div className="bg-amber-500/10 border-b border-amber-500/20 text-amber-300 text-xs text-center px-4 py-2">
        Interactive walkthrough of the diagnostic pipeline — the traces below are
        illustrative examples, not a dump of the real study data. Real findings
        and reproduction commands live in{' '}
        <a
          href="https://github.com/neuraCollab/kv-cache-quantization-reasoning-study"
          className="underline hover:text-amber-200"
        >
          the repo's README and RESULTS.md
        </a>.
      </div>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {activeTab === 'inspector' && <TraceInspector />}
        {activeTab === 'matrix' && <ContingencyMatrix />}
        {activeTab === 'signatures' && <SignatureVisualizer />}
        {activeTab === 'calibration' && <CalibrationSuite />}
        {activeTab === 'pipeline' && <PipelineRunner />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-900/60 py-6 text-center text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>
            KV Cache Quantization Trace-Level Diagnostic Study • Reproducibility Appendix
          </span>
          <div className="flex items-center gap-4 text-slate-400">
            <span>3 Models (1.5B–7B)</span>
            <span>•</span>
            <span>5 KV Methods</span>
            <span>•</span>
            <span>Ada FP8 & HQQ INT</span>
          </div>
        </div>
      </footer>

      {/* Taxonomy Reference Modal */}
      <TaxonomyModal isOpen={isTaxonomyOpen} onClose={() => setIsTaxonomyOpen(false)} />
    </div>
  );
};

export default App;
