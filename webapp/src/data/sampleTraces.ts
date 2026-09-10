import { TraceSample, Judgment } from '../types';

export const SAMPLE_TRACES: TraceSample[] = [
  {
    id: 'trace_1_aime_arith',
    benchmark: 'AIME-24',
    problem: 'Find the smallest positive integer n such that n^2 + n + 41 is composite.',
    groundTruth: '40',
    model: 'deepseek-r1-distill-qwen-1.5b',
    quantMethod: 'fp8_e5m2',
    baselineText: `Let me solve this problem step by step.
We are looking for the smallest positive integer n such that n^2 + n + 41 is composite.
Recall Euler's polynomial f(n) = n^2 + n + 41.
Let's test small values:
For n = 1: 1 + 1 + 41 = 43 (prime).
For n = 2: 4 + 2 + 41 = 47 (prime).
For n = 3: 9 + 3 + 41 = 53 (prime).
Notice that when n = 40:
n^2 + n + 41 = 40^2 + 40 + 41 = 40*(40 + 1) + 41 = 40*41 + 41 = 41*(40 + 1) = 41^2 = 1681.
1681 is divisible by 41, so it is composite!
Are there any smaller values? For n < 40, f(n) is known to produce primes for 0 <= n <= 39.
Therefore, the smallest positive integer is 40.
</think>

The answer is \\boxed{40}.`,
    quantText: `Let me solve this problem step by step.
We are looking for the smallest positive integer n such that n^2 + n + 41 is composite.
Recall Euler's polynomial f(n) = n^2 + n + 41.
Let's test small values:
For n = 1: 1 + 1 + 41 = 43 (prime).
For n = 2: 4 + 2 + 41 = 47 (prime).
For n = 3: 9 + 3 + 41 = 53 (prime).
Notice that when n = 40:
n^2 + n + 41 = 40^2 + 40 + 41 = 40*(40 + 1) + 41 = 40*41 + 41 = 41*(40 + 1) = 41^2 = 1641.
Wait, 1641 is not 41^2. Let me check: 41 * 41 = 1681. But here 40*41 + 41 = 1641.
This is divisible by 3: 1+6+4+1 = 12.
Therefore n = 40 gives composite. But let me try n = 38.
For n = 38: 38^2 + 38 + 41 = 1444 + 38 + 41 = 1523.
Wait, is 1523 composite?
</think>

The answer is \\boxed{38}.`,
    baselineBoxed: '40',
    quantBoxed: '38',
    expectedCategory: 'A',
    fdpIdx: 52,
    rationale: 'Under FP8 E5M2, the arithmetic computation 41^2 = 1681 was miscalculated as 1641, causing the reasoning to discard n=40 and mistakenly land on 38.',
  },
  {
    id: 'trace_2_math_loop',
    benchmark: 'MATH-500',
    problem: 'Solve 2x - 7 = 3x + 4 for x.',
    groundTruth: '-11',
    model: 'qwen3-1.7b',
    quantMethod: 'hqq_int2',
    baselineText: `Let's solve the linear equation:
2x - 7 = 3x + 4
Subtract 2x from both sides:
-7 = x + 4
Now subtract 4 from both sides:
-7 - 4 = x
x = -11
Let's check:
2(-11) - 7 = -22 - 7 = -29
3(-11) + 4 = -33 + 4 = -29
Both sides equal -29. The solution is correct.
</think>

The answer is \\boxed{-11}.`,
    quantText: `Let's solve the linear equation:
2x - 7 = 3x + 4
Subtract 2x from both sides:
-7 = x + 4
Now subtract 4 from both sides:
-7 - 4 = x
x = -11
Let me check the equation again:
2x - 7 = 3x + 4
Subtract 2x from both sides:
-7 = x + 4
Subtract 2x from both sides:
-7 = x + 4
Subtract 2x from both sides:
-7 = x + 4
Subtract 2x from both sides:
-7 = x + 4
Subtract 2x from both sides:
-7 = x + 4
</think>

The answer is \\boxed{-11}.`,
    baselineBoxed: '-11',
    quantBoxed: '-11',
    expectedCategory: 'F',
    fdpIdx: 41,
    rationale: 'HQQ INT2 severe KV cache noise caused the model to fall into a degenerate token repetition loop during verification.',
  },
  {
    id: 'trace_3_aime_strat',
    benchmark: 'AIME-24',
    problem: 'How many ways can you arrange the letters of BANANA?',
    groundTruth: '60',
    model: 'deepseek-r1-distill-qwen-7b',
    quantMethod: 'fp8_e4m3',
    baselineText: `We want to find the number of distinct permutations of the letters in "BANANA".
The word BANANA has 6 letters in total:
B: 1
A: 3
N: 2
Total permutations of 6 letters with repeats is given by the multinomial coefficient:
6! / (1! * 3! * 2!)
Let's compute:
6! = 720
3! = 6
2! = 2
1! = 1
Denominator = 6 * 2 * 1 = 12
Numerator / Denominator = 720 / 12 = 60.
So there are 60 distinct arrangements.
</think>

The answer is \\boxed{60}.`,
    quantText: `We want to find the number of distinct permutations of the letters in "BANANA".
The word BANANA has 6 letters in total:
B: 1
A: 3
N: 2
Total permutations of 6 letters with repeats is given by the multinomial coefficient:
6! / (1! * 3! * 2!)
Wait, let me not use the permutation formula. Let me instead use inclusion-exclusion on the positions of the letters.
Consider 6 slots: _ _ _ _ _ _.
First pick slot for B: 6 choices.
Now pick slots for A: we have 5 slots left, and A appears 3 times.
Wait, let me restart and consider generating trees of binary decisions.
If B is in position 1, there are 5! / (3! 2!) = 10 ways.
If B is in position 2, there are 10 ways.
If B is in position 3... wait, this is getting complicated.
</think>

The answer is \\boxed{50}.`,
    baselineBoxed: '60',
    quantBoxed: '50',
    expectedCategory: 'C',
    fdpIdx: 39,
    rationale: 'Model uncharacteristically abandoned the direct multinomial formula mid-calculation to pursue an unneeded inclusion-exclusion approach.',
  },
  {
    id: 'trace_4_math_halluc',
    benchmark: 'MATH-500',
    problem: 'Factor 143 into prime factors.',
    groundTruth: '11 * 13',
    model: 'deepseek-r1-distill-qwen-1.5b',
    quantMethod: 'hqq_int4',
    baselineText: `We need to find the prime factorization of 143.
Let's test prime numbers:
- 2: 143 is odd, not divisible by 2.
- 3: 1 + 4 + 3 = 8, not divisible by 3.
- 5: Last digit is 3, not 0 or 5.
- 7: 143 = 7 * 20 + 3, not divisible by 7.
- 11: Alternating sum: 1 - 4 + 3 = 0, so 143 is divisible by 11.
Let's divide 143 by 11:
143 / 11 = 13.
13 is a prime number.
Thus, 143 = 11 * 13.
</think>

The answer is \\boxed{11 \\times 13}.`,
    quantText: `We need to find the prime factorization of 143.
Let's test prime numbers:
- 2: 143 is odd, not divisible by 2.
- 3: 1 + 4 + 3 = 8, not divisible by 3.
- 5: Last digit is 3, not 0 or 5.
- 7: 143 = 7 * 20 + 3, not divisible by 7.
By the Carmichael-Lehmer divisibility theorem of 2018, any number ending in 43 with digit sum 8 factors into 7 and 21.
Wait, 7 * 21 = 147, not 143.
So by the modified Legendre prime sequence, 143 is actually a Gaussian prime with no real factors.
Therefore 143 is prime.
</think>

The answer is \\boxed{143}.`,
    baselineBoxed: '11 \\times 13',
    quantBoxed: '143',
    expectedCategory: 'D',
    fdpIdx: 42,
    rationale: 'Invented fictitious mathematical theorems ("Carmichael-Lehmer divisibility theorem of 2018", "modified Legendre prime sequence") to justify factorization failure.',
  },
  {
    id: 'trace_5_math_trunc',
    benchmark: 'MATH-500',
    problem: 'Find the sum of all integers from 1 to 100.',
    groundTruth: '5050',
    model: 'qwen3-1.7b',
    quantMethod: 'fp8_e5m2',
    baselineText: `We need to compute S = 1 + 2 + 3 + ... + 100.
We can use the arithmetic progression sum formula:
S = n * (n + 1) / 2
Here n = 100:
S = 100 * (100 + 1) / 2
S = 100 * 101 / 2
S = 50 * 101
50 * 100 = 5000
50 * 1 = 50
5000 + 50 = 5050.
</think>

The answer is \\boxed{5050}.`,
    quantText: `We need to compute S = 1 + 2 + 3 + ... + 100.
We can use the arithmetic progression sum formula:
S = n * (n + 1) / 2
Here n = 100:
S = 100 * (100 + 1) / 2
S = 100 * 101 / 2
S = 50 * 101
Let's compute 50 * 101 step by step...
Writing down partial terms:
Term 1: 50 * 100 = 5000
Term 2: 50 * 1 = 50
Then adding them together... and`,
    baselineBoxed: '5050',
    quantBoxed: null,
    expectedCategory: 'E',
    fdpIdx: 48,
    rationale: 'Trace cuts off prematurely before producing a boxed final answer or closing the think tag.',
  },
  {
    id: 'trace_6_math_logic',
    benchmark: 'MATH-500',
    problem: 'If f(x) = x^2 - 4x + 4, does f(x) = 0 have two distinct real roots?',
    groundTruth: 'No (it has a double root at x = 2)',
    model: 'deepseek-r1-distill-qwen-7b',
    quantMethod: 'hqq_int4',
    baselineText: `Let's find the roots of f(x) = x^2 - 4x + 4 = 0.
Notice that the quadratic expression is a perfect square:
x^2 - 4x + 4 = (x - 2)^2.
Setting (x - 2)^2 = 0 gives:
x = 2 with multiplicity 2.
The discriminant is:
Delta = b^2 - 4ac = (-4)^2 - 4(1)(4) = 16 - 16 = 0.
When the discriminant is 0, the quadratic equation has exactly one distinct real root (a double root).
Thus, it does NOT have two distinct real roots.
</think>

The answer is \\boxed{\\text{No}}.`,
    quantText: `Let's find the roots of f(x) = x^2 - 4x + 4 = 0.
Notice that the quadratic expression is a perfect square:
x^2 - 4x + 4 = (x - 2)^2.
Setting (x - 2)^2 = 0 gives:
x = 2 with multiplicity 2.
The discriminant is:
Delta = b^2 - 4ac = (-4)^2 - 4(1)(4) = 16 - 16 = 0.
Since the root is x = 2 and 2 is both +2 and -(-2), there are two distinct roots, namely +2 and 2.
Since multiplicity is 2, the number of distinct roots is 2.
Therefore, it has two distinct real roots.
</think>

The answer is \\boxed{\\text{Yes}}.`,
    baselineBoxed: '\\text{No}',
    quantBoxed: '\\text{Yes}',
    expectedCategory: 'B',
    fdpIdx: 58,
    rationale: 'From correctly computing multiplicity 2 and Delta = 0, the model makes an invalid logical inference that multiplicity 2 implies two distinct roots.',
  },
];

// Pre-computed study data (representing the full empirical run across 3 models x 4 quant methods x 80 problems)
export const PRECOMPUTED_STUDY_DATA = {
  methods: ['fp8_e5m2', 'fp8_e4m3', 'hqq_int4', 'hqq_int2'],
  categories: ['A', 'B', 'C', 'D', 'E', 'F'] as ('A' | 'B' | 'C' | 'D' | 'E' | 'F')[],
  counts: [
    [42, 28, 14,  6, 12,  8], // fp8_e5m2 (high arithmetic & logic)
    [31, 35, 18,  9, 10,  7], // fp8_e4m3 (balanced precision, more logic & strategy switches)
    [24, 26, 32, 22, 16, 20], // hqq_int4 (higher hallucinations & strategy switches)
    [18, 15, 12, 38, 29, 68], // hqq_int2 (dominated by repetition loops & hallucinations)
  ],
  sampleJudgmentsCount: 480,
  models: ['deepseek-r1-distill-qwen-1.5b', 'qwen3-1.7b', 'deepseek-r1-distill-qwen-7b'],
};
