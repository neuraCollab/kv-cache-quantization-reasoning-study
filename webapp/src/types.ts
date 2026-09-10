export type CategoryLetter = 'A' | 'B' | 'C' | 'D' | 'E' | 'F';

export interface Category {
  letter: CategoryLetter;
  name: string;
  description: string;
  examples: string[];
  color: string;
  badgeBg: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  hfId: string;
  params: string;
  dtype: string;
  maxModelLen: number;
  thinkingMode?: boolean;
}

export interface QuantMethodInfo {
  id: string;
  name: string;
  engine: 'vllm' | 'hf';
  kvCacheDtype: string;
  bits: number | string;
  description: string;
  color: string;
}

export type BoxedMatch = 'both_correct' | 'baseline_only' | 'quant_only' | 'both_wrong' | 'no_boxed';

export interface FDPRecord {
  fdpTokenIdx: number | null;
  cosmeticSkipped: number;
  baselineContext: string;
  quantContext: string;
  commonPrefix: string;
  boxedMatch: BoxedMatch;
  baselineTruncated: boolean;
  quantTruncated: boolean;
  divergentTokenBaseline?: string;
  divergentTokenQuant?: string;
}

export interface Judgment {
  id: string;
  problemId: string;
  problem: string;
  groundTruth: string;
  model: string;
  quantMethod: string;
  fdpTokenIdx: number | null;
  category: CategoryLetter;
  confidence: number;
  rationale: string;
  affectedSpan: string;
  boxedMatch: BoxedMatch;
}

export interface GoldenExample {
  id: string;
  problem: string;
  groundTruth: string;
  commonPrefix: string;
  baselineContext: string;
  quantContext: string;
  goldCategory: CategoryLetter;
}

export interface TraceSample {
  id: string;
  benchmark: 'AIME-24' | 'MATH-500';
  problem: string;
  groundTruth: string;
  model: string;
  quantMethod: string;
  baselineText: string;
  quantText: string;
  baselineBoxed: string | null;
  quantBoxed: string | null;
  expectedCategory: CategoryLetter;
  fdpIdx: number;
  rationale: string;
}

export interface AnalysisReportData {
  methods: string[];
  categories: CategoryLetter[];
  counts: number[][];
  signatures: number[][];
  chiSquare: number;
  chiSquareP: number;
  chiSquareDof: number;
  cramersV: number;
  totalJudgments: number;
  markdown: string;
  jsonString: string;
}

export interface PipelineConfig {
  aimeCount: number;
  math500Count: number;
  seed: number;
  temperature: number;
  topP: number;
  maxTokens: number;
  repetitionPenalty: number;
  contextWindow: number;
  resyncLookahead: number;
  cosmeticThreshold: number;
  maxCosmeticSkips: number;
}
