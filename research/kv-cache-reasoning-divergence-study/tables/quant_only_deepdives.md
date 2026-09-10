# `quant_only` deep-dives

Cases where the **quantized** trace produced the correct boxed answer but the **bf16 baseline** did not. These are interesting because they violate the naive expectation that quantization can only degrade performance — here, KV cache noise pushed the model onto a path the deterministic baseline never explored.

**Total `quant_only` cases found**: 5

Each entry shows: the problem, ground truth, the FDP token index, the judge's category for this divergence, the last common reasoning prefix, and ~600 chars of each branch around the divergence point. The judge still classifies these as divergences (the local *step* differs), but the global outcome is favorable to the quantized branch.

## Case 1: `qwen3-1.7b` × `fp8_e4m3` × problem 8

- **Source**: aime-24
- **FDP token idx**: 554
- **Ground truth**: `25`
- **Baseline boxed**: `None`
- **Quant boxed**: `25`
- **Judge category**: **C (Strategy-switch)**, confidence 0.85
- **Judge rationale**: The quantized trace abandons the clean substitution approach (k = log_y(x)) that was working and switches to substituting y = x^{10/x} and then yet another approach, without motivation. The baseline continues directly to the answer using the established substitution.
- **Affected span**: `Wait, earlier I had from the first equation that y = x^{10/x}`

**Problem:**

> There exist real numbers $x$ and $y$, both greater than 1, such that $\log_x\left(y^x\right)=\log_y\left(x^{4y}\right)=10$. Find $xy$.

**Common prefix (last part — same in both):**

```
4y * log_y(x) = 10.

Now, I have two equations:

1) x * log_x(y) = 10

2) 4y * log_y(x) = 10

Hmm, maybe I can express log_x(y) in terms of log_y(x)? Because log_x(y) is the reciprocal of log_y(x), right? Wait, no. Let me think. If I have log_b(a) = 1/(log_a(b)). So, log_x(y) = 1 / log_y(x). Let me verify that. Let me recall that log_b(a) = ln a / ln b, and log_y(x) = ln x / ln y. So, log_x(y) = ln y / ln x = 1 / (ln x / ln y) = 1 / log_y(x). Yes, that's correct.

So, if I let log_y(x) be some variable, say, let me denote log_y(x) as k. Then log_x(y) would be 1/k. 

So, substituting back into the equations:

First equation: x * (1/k) = 10 => x/k = 10 => x = 10k.

Second equation: 4y * k = 10 => 4yk = 10 => yk = 10/4 = 5/2.

So now, we have x
```

**Baseline branch (bf16) around FDP:**

```
(b)). So, log_x(y) = 1 / log_y(x). Let me verify that. Let me recall that log_b(a) = ln a / ln b, and log_y(x) = ln x / ln y. So, log_x(y) = ln y / ln x = 1 / (ln x / ln y) = 1 / log_y(x). Yes, that's correct.

So, if I let log_y(x) be some variable, say, let me denote log_y(x) as k. Then log_x(y) would be 1/k. 

So, substituting back into the equations:

First equation: x * (1/k) = 10 => x/k = 10 => x = 10k.

Second equation: 4y * k = 10 => 4yk = 10 => yk = 10/4 = 5/2.

So now, we have x = 10k and y = (5/2)/k. Therefore, xy = 10k * (5/2)/k = 10*(5/2) = 25. Wait, so is it 25? That seems straightforward, but let me check if there are any constraints or if I made a mistake.

Wait, but hold on. Let me make sure that my substitution is correct. Let me retrace.

Let me define k = log_y(x). Then
```

**Quantized branch (fp8_e4m3) around FDP:**

```
 but maybe.

Now moving on to the second equation: log_y(x^{4y}) = 10.

Again, using the same logarithm property: log_b(a^c) = c * log_b(a). So here, log_y(x^{4y}) = 4y * log_y(x) = 10.

Similarly, if I let log_y(x) = m, then y^m = x. And from the equation above, 4y * m = 10. So m = 10/(4y) = 5/(2y).

But also, log_y(x) = m = log_y(x) which is equal to ln(x)/ln(y). But maybe I can relate this to the previous expression?

Wait, earlier I had from the first equation that y = x^{10/x}. Let me write that again:

From the first equation: y = x^{10/x}.

So maybe I can substitute this into the second equation?

Alternatively, maybe I can express log_y(x) in terms of x and y?

Since log_y(x) = 1/m, where m was 5/(2y). Wait, no. Wait, from the second equation, we had 4y * log_y(x) = 10, so log_y(x)
```

---

## Case 2: `qwen3-1.7b` × `fp8_e4m3` × problem 14

- **Source**: aime-24
- **FDP token idx**: 534
- **Ground truth**: `294`
- **Baseline boxed**: `728`
- **Quant boxed**: `294`
- **Judge category**: **F (Repetition/loop)**, confidence 0.72
- **Judge rationale**: The quantized trace repeats the same reasoning about five segments and four direction changes multiple times without making progress, cycling through the same logical steps. The baseline moves forward to set up the counting problem while the quantized trace keeps re-explaining the segment/direction-change relationship.
- **Affected span**: `Wait, no. Wait, if you start with R, then change to U`

**Problem:**

> Consider the paths of length $16$ that follow the lines from the lower left corner to the upper right corner on an $8\times 8$ grid. Find the number of such paths that change direction exactly four times, as in the examples shown below.

**Common prefix (last part — same in both):**

```
 path that goes from R to U, that's a direction change. Similarly, U to R. So, if you have a path that starts with R, then changes direction once, then again, etc. So, the number of direction changes is equal to the number of times the direction switches.

So, for example, a path that goes R, R, R, U, U, U, R, R would have two direction changes: from R to U and then from U to R. So, that's two direction changes. But in this case, the path starts with R, then goes to U, then to R, then to R. So, two changes.

But the problem says "change direction exactly four times". So, the path must switch directions four times. So, how does that translate into the structure of the path?

Let me think. If there are four direction changes, then the path has five segments. For example:

Starts with R, then
```

**Baseline branch (bf16) around FDP:**

```
 a path that goes from R to U, that's a direction change. Similarly, U to R. So, if you have a path that starts with R, then changes direction once, then again, etc. So, the number of direction changes is equal to the number of times the direction switches.

So, for example, a path that goes R, R, R, U, U, U, R, R would have two direction changes: from R to U and then from U to R. So, that's two direction changes. But in this case, the path starts with R, then goes to U, then to R, then to R. So, two changes.

But the problem says "change direction exactly four times". So, the path must switch directions four times. So, how does that translate into the structure of the path?

Let me think. If there are four direction changes, then the path has five segments. For example:

Starts with R, th
```

**Quantized branch (fp8_e4m3) around FDP:**

```
 five segments of movement. Because each direction change increases the number of segments by one. For example, starting with R, then changing to U, then changing to R, then changing to U, etc. So, if there are four direction changes, there are five segments. Each segment is a straight line in one direction.

So, for example, a path that goes R, then U, then R, then U, then R would have four direction changes. Wait, no. Wait, if you start with R, then change to U, then change to R, then change to U, then change to R. That would be four direction changes, resulting in five segments. So yes, four direction changes correspond to five segments.

Therefore, the problem reduces to finding the number of sequences of five segments (each segment being either R or U) with the total number of R's and
```

---

## Case 3: `qwen3-1.7b` × `fp8_e5m2` × problem 3

- **Source**: aime-24
- **FDP token idx**: 12
- **Ground truth**: `809`
- **Baseline boxed**: `1215`
- **Quant boxed**: `809`
- **Judge category**: **B (Logical)**, confidence 0.60
- **Judge rationale**: The quantized trace's analysis of n=2 contains a logical error: it states Alice takes 1 token leaving 1 for Bob who wins, but doesn't consider Alice taking 4 (impossible) or correctly conclude whether n=2 is a win or loss for Alice. The baseline correctly frames positions as W/L from the current player's perspective, while the quantized trace conflates Alice's and Bob's winning positions.
- **Affected span**: `Alice can take 1 token, leaving 1 for Bob, who then takes it and wins`

**Problem:**

> Alice and Bob play the following game. A stack of $n$ tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either $1$ token or $4$ tokens from the stack. Whoever removes the last token wins. Find the number of positive integers $n$ less than or equal to $2024$ for which there exists a strategy for Bob that guarantees that Bob will win the game regardless of Alice's play.

**Common prefix (last part — same in both):**

```
<think>
Okay, so Alice and Bob are playing this game
```

**Baseline branch (bf16) around FDP:**

```
<think>
Okay, so Alice and Bob are playing this game where they take turns removing 1 or 4 tokens from a stack of n tokens. The person who takes the last token wins. I need to find how many numbers n ≤ 2024 have a strategy for Bob that guarantees he can win no matter what Alice does. 

First, let me try to understand the game mechanics. Each player can remove either 1 or 4 tokens on their turn. The key here is that Bob wants to force a win no matter what Alice does. So, Bob needs a winning strategy, meaning that no matter how Alice plays, Bob can always respond in a way that eventually takes the last token.

This seems like a classic combinatorial game theory problem. Maybe I can model this as a game with positions, where each position is a number of tokens remaining, and determine if it's
```

**Quantized branch (fp8_e5m2) around FDP:**

```
<think>
Okay, so Alice and Bob are playing this game with tokens. The rules are that they take turns removing either 1 or 4 tokens from a stack of n tokens. The person who takes the last token wins. I need to find how many positive integers n ≤ 2024 have a strategy for Bob that guarantees he can win no matter what Alice does. 

First, I should figure out the winning and losing positions. A losing position is one where no matter what move you make, you leave the opponent in a winning position. Conversely, a winning position is one where there's at least one move that leaves the opponent in a losing position. 

Let me start by trying small values of n and see if I can spot a pattern.

For n=1: Alice can take 1 token and win. So Alice has a winning strategy. Therefore, n=1 is a winning positi
```

---

## Case 4: `qwen3-1.7b` × `fp8_e5m2` × problem 8

- **Source**: aime-24
- **FDP token idx**: 1055
- **Ground truth**: `25`
- **Baseline boxed**: `None`
- **Quant boxed**: `25`
- **Judge category**: **C (Strategy-switch)**, confidence 0.82
- **Judge rationale**: The quantized trace abandons the solution path (which had already derived xy=25) and restarts with a different approach, trying to solve the system from scratch using logarithms and substitution without motivation. This is an unmotivated strategy switch mid-derivation.
- **Affected span**: `Hmm. This seems like a complicated equation. Maybe I can write it as`

**Problem:**

> There exist real numbers $x$ and $y$, both greater than 1, such that $\log_x\left(y^x\right)=\log_y\left(x^{4y}\right)=10$. Find $xy$.

**Common prefix (last part — same in both):**

```
 x and y are both greater than 1. So, let me check if with xy=25, whether x and y are both greater than 1. Well, if xy=25, then if x>1, y=25/x. Since x>1, then y=25/x. If x>1, then y=25/x <25. But since x>1, y could be greater than 1 or less than 1 depending on x. Wait, but if x>1, then y=25/x. For y>1, we need 25/x >1 => x<25. Similarly, if x>25, then y=25/x <1. But the problem says both x and y are greater than 1. Therefore, x must be between 1 and 25, and y must be between 1 and 25. So, if x is between 1 and 25, then y is between 1 and 25. So, that's okay. So, as long as x is between 1 and 25, and y is between 1 and 25, then both are greater than 1. However, we need to ensure that our solution satisfies the original logarithmic equations. But since we derived them through the equations,
```

**Baseline branch (bf16) around FDP:**

```
 since x>1, y could be greater than 1 or less than 1 depending on x. Wait, but if x>1, then y=25/x. For y>1, we need 25/x >1 => x<25. Similarly, if x>25, then y=25/x <1. But the problem says both x and y are greater than 1. Therefore, x must be between 1 and 25, and y must be between 1 and 25. So, if x is between 1 and 25, then y is between 1 and 25. So, that's okay. So, as long as x is between 1 and 25, and y is between 1 and 25, then both are greater than 1. However, we need to ensure that our solution satisfies the original logarithmic equations. But since we derived them through the equations, perhaps they are automatically satisfied?

But maybe there's another way to check. Let me try plugging in the values. Suppose that xy=25, and x=10k, y=5/(2k). Let me see if these satisfy the orig
```

**Quantized branch (fp8_e5m2) around FDP:**

```
)/x

Hmm. This seems like a complicated equation. Maybe I can write it as:

ln(25) / [(10 + x)/x] = ln(x)

Which is:

ln(25) * [x / (10 + x)] = ln(x)

Not sure if that helps. Maybe I need another approach.

Alternatively, maybe I can let t = x^{10/x} or something? Wait, but I already have that from equation 1. Alternatively, perhaps I can set variables to simplify.

Alternatively, maybe assume that x and y are powers of some common number? Or maybe try to find a relationship between x and y?

Alternatively, since we have two equations:

Equation 1: x^{10} = y^x

Equation 2: y^{10} = x^{4y}

If I take the ratio of equation 1 and equation 2?

Wait, not sure. Let me think.

Alternatively, take equation 1: x^{10} = y^x. Take logarithms again:

10 log_x(y) = log_x(y^x) = 10. Wait, no, that's re
```

---

## Case 5: `qwen3-1.7b` × `fp8_e5m2` × problem 16

- **Source**: aime-24
- **FDP token idx**: 1074
- **Ground truth**: `721`
- **Baseline boxed**: `None`
- **Quant boxed**: `721`
- **Judge category**: **C (Strategy-switch)**, confidence 0.85
- **Judge rationale**: The quantized trace abandons the concrete substitution approach (assuming a=b to reduce to a cubic) and instead switches to treating a,b,c as roots of a general cubic polynomial, an unmotivated strategy change. The baseline continues directly with the a=b substitution leading to a³ - 27a + 46 = 0.
- **Affected span**: `think of a, b, c as roots of a cubic equation. Let me try that`

**Problem:**

> Let $\mathcal{B}$ be the set of rectangular boxes with surface area $54$ and volume $23$. Let $r$ be the radius of the smallest sphere that can contain each of the rectangular boxes that are elements of $\mathcal{B}$. The value of $r^2$ can be written as $\rac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p+q$.

**Common prefix (last part — same in both):**

```
 maybe there's a smarter way. Alternatively, maybe we can assume some symmetry?

But since the problem is about minimizing S², perhaps the minimum occurs when the variables are equal? But wait, if they are equal, then a = b = c. Let's check if that's possible.

If a = b = c, then the volume would be a³ = 23 ⇒ a = cube root of 23 ≈ 2.84. Then the surface area would be 2(3a²) = 6a². Let's compute that: 6*(2.84)^2 ≈ 6*8.06 ≈ 48.36, which is less than 54. So, the surface area is too small. Therefore, the box cannot be a cube. Therefore, the minimal S² is achieved when the variables are not all equal.

Alternatively, maybe two variables are equal? Let's suppose that two variables are equal, say a = b. Then we can write equations in terms of a and c.

Let me try that. Let’s assume a = b. Then:


```

**Baseline branch (bf16) around FDP:**

```
? But wait, if they are equal, then a = b = c. Let's check if that's possible.

If a = b = c, then the volume would be a³ = 23 ⇒ a = cube root of 23 ≈ 2.84. Then the surface area would be 2(3a²) = 6a². Let's compute that: 6*(2.84)^2 ≈ 6*8.06 ≈ 48.36, which is less than 54. So, the surface area is too small. Therefore, the box cannot be a cube. Therefore, the minimal S² is achieved when the variables are not all equal.

Alternatively, maybe two variables are equal? Let's suppose that two variables are equal, say a = b. Then we can write equations in terms of a and c.

Let me try that. Let’s assume a = b. Then:

ab + bc + ac = a² + 2ac = 27

and

abc = a²c = 23

So, from the second equation, c = 23 / a².

Substitute into the first equation:

a² + 2a*(23 / a²) = 27 ⇒ a² + 46 / a = 27.

Multip
```

**Quantized branch (fp8_e5m2) around FDP:**

```
 b + c)^2 = a² + b² + c² + 2(ab + bc + ac). So, if we know ab + bc + ac, then we can write a² + b² + c² = (a + b + c)^2 - 2(ab + bc + ac). So, if we can find a + b + c, then we can compute a² + b² + c².

But how does that help us? Well, if we can find a + b + c, then we can compute the required expression.

But how do we find a + b + c?

Alternatively, maybe use AM ≥ GM or other inequalities?

But since we have ab + bc + ac = 27 and abc = 23, perhaps we can relate these to the variables.

Alternatively, think of a, b, c as roots of a cubic equation. Let me try that.

Suppose that a, b, c are roots of the equation x³ - px² + qx - r = 0, where p = a + b + c, q = ab + bc + ac = 27, and r = abc = 23.

So, the cubic equation is x³ - px² + 27x - 23 = 0.

But how does this help us? Well, if we ca
```

---

