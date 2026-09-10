import { BoxedMatch, FDPRecord } from '../types';

export interface FDPParams {
  contextWindow?: number;
  resyncLookahead?: number;
  maxCosmeticSkips?: number;
  cosmeticThreshold?: number;
}

// Simple tokenization that respects math notation, symbols and words
export function tokenize(text: string): string[] {
  const tokens: string[] = [];
  // Match LaTeX commands like \boxed, numbers, words, or individual punctuation
  const regex = /\\[a-zA-Z]+|\d+|[a-zA-Z]+|[^\s\w]/g;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(text)) !== null) {
    tokens.push(match[0]);
  }
  return tokens;
}

export function extractBoxed(text: string): string | null {
  const match = text.match(/\\boxed\{([^}]+)\}/);
  return match ? match[1].trim() : null;
}

export function evaluateBoxedMatch(
  baselineBoxed: string | null,
  quantBoxed: string | null,
  groundTruth: string | null
): BoxedMatch {
  if (!baselineBoxed && !quantBoxed) return 'no_boxed';
  const gt = (groundTruth || '').trim().toLowerCase();
  const b = (baselineBoxed || '').trim().toLowerCase();
  const q = (quantBoxed || '').trim().toLowerCase();

  const bOk = b.length > 0 && (gt.length === 0 || b === gt || b.includes(gt) || gt.includes(b));
  const qOk = q.length > 0 && (gt.length === 0 || q === gt || q.includes(gt) || gt.includes(q));

  if (bOk && qOk) return 'both_correct';
  if (bOk && !qOk) return 'baseline_only';
  if (qOk && !bOk) return 'quant_only';
  return 'both_wrong';
}

// Jaccard / token similarity to detect cosmetic vs semantic divergence
function isCosmeticDivergence(baselineWindow: string[], quantWindow: string[], threshold = 0.85): boolean {
  if (baselineWindow.length === 0 || quantWindow.length === 0) return false;
  const setB = new Set(baselineWindow.map(t => t.toLowerCase()));
  const setQ = new Set(quantWindow.map(t => t.toLowerCase()));

  let intersection = 0;
  for (const item of setB) {
    if (setQ.has(item)) intersection++;
  }
  const union = new Set([...setB, ...setQ]).size;
  if (union === 0) return false;
  const similarity = intersection / union;
  return similarity >= threshold;
}

export function findFDP(
  baselineText: string,
  quantText: string,
  groundTruth?: string,
  params?: FDPParams
): FDPRecord {
  const p: Required<FDPParams> = {
    contextWindow: params?.contextWindow ?? 15,
    resyncLookahead: params?.resyncLookahead ?? 10,
    maxCosmeticSkips: params?.maxCosmeticSkips ?? 5,
    cosmeticThreshold: params?.cosmeticThreshold ?? 0.85,
  };

  const baselineTokens = tokenize(baselineText);
  const quantTokens = tokenize(quantText);

  const baselineBoxed = extractBoxed(baselineText);
  const quantBoxed = extractBoxed(quantText);
  const boxedMatch = evaluateBoxedMatch(baselineBoxed, quantBoxed, groundTruth ?? null);

  const baselineTruncated = !baselineText.includes('\\boxed') && baselineText.length > 500;
  const quantTruncated = !quantText.includes('\\boxed') && quantText.length > 500;

  let offset = 0;
  let cosmeticSkipped = 0;
  const minLen = Math.min(baselineTokens.length, quantTokens.length);

  while (offset < minLen) {
    let mismatchIdx: number | null = null;
    for (let i = offset; i < minLen; i++) {
      if (baselineTokens[i] !== quantTokens[i]) {
        mismatchIdx = i;
        break;
      }
    }

    if (mismatchIdx === null) {
      if (baselineTokens.length !== quantTokens.length) {
        mismatchIdx = minLen;
      } else {
        // Exact match
        return {
          fdpTokenIdx: null,
          cosmeticSkipped,
          baselineContext: '',
          quantContext: '',
          commonPrefix: baselineText.slice(0, 300),
          boxedMatch,
          baselineTruncated,
          quantTruncated,
        };
      }
    }

    // Check if cosmetic
    if (cosmeticSkipped < p.maxCosmeticSkips) {
      const bWindow = baselineTokens.slice(mismatchIdx, mismatchIdx + p.resyncLookahead);
      const qWindow = quantTokens.slice(mismatchIdx, mismatchIdx + p.resyncLookahead);
      const isCosmetic = isCosmeticDivergence(bWindow, qWindow, p.cosmeticThreshold);

      if (isCosmetic) {
        cosmeticSkipped++;
        offset = mismatchIdx + p.resyncLookahead;
        continue;
      }
    }

    // Real FDP found
    const prefixStart = Math.max(0, mismatchIdx - 20);
    const commonPrefixTokens = baselineTokens.slice(prefixStart, mismatchIdx);
    const commonPrefix = commonPrefixTokens.join(' ');

    const bCtxStart = Math.max(0, mismatchIdx - 5);
    const bCtxEnd = Math.min(baselineTokens.length, mismatchIdx + p.contextWindow);
    const baselineContext = baselineTokens.slice(bCtxStart, bCtxEnd).join(' ');

    const qCtxStart = Math.max(0, mismatchIdx - 5);
    const qCtxEnd = Math.min(quantTokens.length, mismatchIdx + p.contextWindow);
    const quantContext = quantTokens.slice(qCtxStart, qCtxEnd).join(' ');

    return {
      fdpTokenIdx: mismatchIdx,
      cosmeticSkipped,
      baselineContext,
      quantContext,
      commonPrefix,
      boxedMatch,
      baselineTruncated,
      quantTruncated,
      divergentTokenBaseline: baselineTokens[mismatchIdx] || '(end of trace)',
      divergentTokenQuant: quantTokens[mismatchIdx] || '(end of trace)',
    };
  }

  return {
    fdpTokenIdx: null,
    cosmeticSkipped,
    baselineContext: '',
    quantContext: '',
    commonPrefix: baselineText.slice(0, 300),
    boxedMatch,
    baselineTruncated,
    quantTruncated,
  };
}
