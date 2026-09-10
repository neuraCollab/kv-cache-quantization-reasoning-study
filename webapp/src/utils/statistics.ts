import { CategoryLetter, AnalysisReportData } from '../types';

export const CATEGORY_ORDER: CategoryLetter[] = ['A', 'B', 'C', 'D', 'E', 'F'];

// Approximation of regularized upper incomplete gamma Q(a, x) for Chi-Square p-value
function gammpApprox(a: number, x: number): number {
  if (x <= 0) return 1.0;
  if (x < a + 1) {
    // Series expansion
    let sum = 1 / a;
    let term = 1 / a;
    for (let n = 1; n < 100; n++) {
      term *= x / (a + n);
      sum += term;
      if (Math.abs(term) < Math.abs(sum) * 1e-10) break;
    }
    const logGammaA = logGamma(a);
    return Math.max(0, Math.min(1, 1 - Math.exp(-x + a * Math.log(x) - logGammaA) * sum));
  } else {
    // Continued fraction
    let b = x + 1 - a;
    let c = 1 / 1e-30;
    let d = 1 / b;
    let h = d;
    for (let i = 1; i < 100; i++) {
      const an = -i * (i - a);
      b += 2;
      d = an * d + b;
      if (Math.abs(d) < 1e-30) d = 1e-30;
      c = b + an / c;
      if (Math.abs(c) < 1e-30) c = 1e-30;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 1e-10) break;
    }
    const logGammaA = logGamma(a);
    return Math.max(0, Math.min(1, Math.exp(-x + a * Math.log(x) - logGammaA) * h));
  }
}

function logGamma(z: number): number {
  const c = [
    57.1562356658629235,
    -59.5979603554754912,
    14.1360979747417471,
    -0.491913816097620199,
    0.339946499848118887e-4,
    0.465236289270485756e-4,
    -0.983744753048795646e-4,
    0.158088703224377949e-3,
    -0.210264441724104883e-3,
    0.217439618115212643e-3,
    -0.16431810653676389e-3,
    0.844182239838527433e-4,
    -0.261908384015814087e-4,
    0.368991826595316234e-5,
  ];
  let sum = 0.99999999999999709182;
  for (let j = 0; j < 14; j++) {
    sum += c[j] / (z + j + 1);
  }
  const t = z + 14 - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(sum);
}

export function chiSquarePValue(chi2: number, dof: number): number {
  if (dof <= 0 || chi2 <= 0) return 1.0;
  return gammpApprox(dof / 2, chi2 / 2);
}

export function rowNormalize(matrix: number[][]): number[][] {
  return matrix.map(row => {
    const sum = row.reduce((a, b) => a + b, 0);
    if (sum === 0) return new Array(row.length).fill(0);
    return row.map(val => Number((val / sum).toFixed(4)));
  });
}

export function chiSquareTest(matrix: number[][]): { chi2: number; p: number; dof: number } {
  const numRows = matrix.length;
  const numCols = matrix[0]?.length ?? 0;
  if (numRows < 2 || numCols < 2) {
    return { chi2: 0, p: 1, dof: 0 };
  }

  // Row sums and Column sums
  const rowSums = matrix.map(row => row.reduce((a, b) => a + b, 0));
  const colSums = new Array(numCols).fill(0);
  for (let j = 0; j < numCols; j++) {
    for (let i = 0; i < numRows; i++) {
      colSums[j] += matrix[i][j];
    }
  }
  const total = rowSums.reduce((a, b) => a + b, 0);
  if (total === 0) return { chi2: 0, p: 1, dof: 0 };

  let chi2 = 0;
  for (let i = 0; i < numRows; i++) {
    for (let j = 0; j < numCols; j++) {
      const expected = (rowSums[i] * colSums[j]) / total;
      if (expected > 0) {
        const diff = matrix[i][j] - expected;
        chi2 += (diff * diff) / expected;
      }
    }
  }

  const dof = (numRows - 1) * (numCols - 1);
  const p = chiSquarePValue(chi2, dof);
  return { chi2: Number(chi2.toFixed(3)), p: Number(p.toFixed(6)), dof };
}

export function calculateCramersV(matrix: number[][], chi2: number): number {
  const total = matrix.reduce((acc, row) => acc + row.reduce((a, b) => a + b, 0), 0);
  if (total === 0) return 0;
  const numRows = matrix.length;
  const numCols = matrix[0]?.length ?? 0;
  const k = Math.min(numRows, numCols);
  if (k <= 1) return 0;
  const v = Math.sqrt(chi2 / (total * (k - 1)));
  return Number(v.toFixed(3));
}

export function buildAnalysisReport(methods: string[], counts: number[][]): AnalysisReportData {
  const signatures = rowNormalize(counts);
  const { chi2, p, dof } = chiSquareTest(counts);
  const v = calculateCramersV(counts, chi2);
  const totalJudgments = counts.reduce((acc, row) => acc + row.reduce((a, b) => a + b, 0), 0);

  // Markdown Report matching report.py
  const mdLines: string[] = [
    '# KV Cache Quantization — Failure-Signature Report',
    '',
    `Total judgments: **${totalJudgments}**`,
    `Chi-square: chi2=${chi2.toFixed(2)}, dof=${dof}, **p=${p < 0.001 ? '<0.001' : p.toFixed(3)}**`,
    `Cramér's V: **${v.toFixed(3)}** (effect size: ${v > 0.35 ? 'Large' : v > 0.15 ? 'Moderate' : 'Small'})`,
    '',
    '## Raw counts (rows = quant method, cols = category A..F)',
    '',
    `| method | ${CATEGORY_ORDER.join(' | ')} | total |`,
    `|---|${CATEGORY_ORDER.map(() => '---').join('|')}|---|`,
  ];

  methods.forEach((m, i) => {
    const row = counts[i];
    const rowSum = row.reduce((a, b) => a + b, 0);
    mdLines.push(`| ${m} | ${row.join(' | ')} | ${rowSum} |`);
  });

  mdLines.push('', '## Normalized failure signatures (rows sum to 1)', '');
  mdLines.push(`| method | ${CATEGORY_ORDER.join(' | ')} |`);
  mdLines.push(`|---|${CATEGORY_ORDER.map(() => '---').join('|')}|`);

  methods.forEach((m, i) => {
    const row = signatures[i];
    mdLines.push(`| ${m} | ${row.map(x => x.toFixed(2)).join(' | ')} |`);
  });

  const jsonObject = {
    methods,
    categories: CATEGORY_ORDER,
    counts,
    signatures,
    chi_square: chi2,
    chi_square_p: p,
    chi_square_dof: dof,
    cramers_v: v,
    total_judgments: totalJudgments,
  };

  return {
    methods,
    categories: CATEGORY_ORDER,
    counts,
    signatures,
    chiSquare: chi2,
    chiSquareP: p,
    chiSquareDof: dof,
    cramersV: v,
    totalJudgments,
    markdown: mdLines.join('\n'),
    jsonString: JSON.stringify(jsonObject, null, 2),
  };
}
