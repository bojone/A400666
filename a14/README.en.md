# a(14): Reducing the Equation to a Linear Congruence via Exponential Periods

**[中文](README.md) | English**

## Result and evidence

$$\boxed{a(14)=675924569493048966}$$

Verification modulo the full composite modulus gives

$$c(675924569493048966)\bmod 1234567891011121314=0.$$

The search covered $`14 < k < 10^{18}`$. There is no solution with at most 17 digits, and the 18-digit interval produced only this solution. Its orbit parameters are

$$x=1484703107343807,\qquad t=177,$$

with $`x+Qt=675924569493048966`$ and $`k\bmod6=0`$.

Principal evidence:

- [Host verification of the unique 18-digit candidate](results/d18_x0.candidates).
- [Original manager log](manager.log): task queues and completion status for eight GPUs.
- [Shard states and logs](results/): configurations, checkpoints, and `DONE` records for 36 shards.
- [Verification of the first 13 terms](oeis_results/verification.txt): divisibility and exhaustive results below every published value.

## 1. Modulus factorization and initial filtering

$$c(14)=1234567891011121314=2\cdot3\cdot p,$$

$$p=205761315168520219.$$

Here $`p`$ is prime. The last decimal digit of $`c(k)`$ has the same parity as $`k`$. Modulo 3, the digit sum gives

$$c(k)\equiv\sum_{i=1}^{k}i=\frac{k(k+1)}2\pmod3.$$

It therefore suffices to retain $`k\bmod6\in\\{0,2\\}`$ and require $`c(k)\equiv0\pmod p`$. Every final candidate is still verified modulo the full $`c(14)`$; divisibility by the prime factor alone is not treated as sufficient.

## 2. A closed form within a fixed digit-length interval

Let $`r_k=c(k)\bmod p`$. If $`k`$ has exactly $`d`$ decimal digits, put $`q=10^d\bmod p`$. Then

$$r_k=qr_{k-1}+k\pmod p.$$

Define

$$A=\frac1{1-q}\pmod p,\qquad B=-\frac q{(1-q)^2}\pmod p.$$

These definitions require $`q\ne1\pmod p`$. Every production interval satisfies this condition, and every coefficient inverted below is nonzero, so the formulas are well-defined.

Then

$$r_k=Uq^k+Ak+B\pmod p,$$

where $`U`$ is determined by the interval start $`s`$:

$$U=\left(r_{s-1}-A(s-1)-B\right)q^{-(s-1)}\pmod p.$$

This avoids concatenating every integer from $`1`$ onward. Nevertheless, directly enumerating the 18-digit interval would still leave about $`9\times10^{17}`$ candidates. Period reduction is the key next step.

## 3. The 18-digit interval

Factor

$$p-1=54Q,\qquad Q=3810394725342967,$$

where $`Q`$ is also prime and

$$\mathrm{ord}_p(10^{18})=Q.$$

Write $`k=x+Qt`$ uniquely, with $`0\le x < Q`$. Since $`q^{x+Qt}=q^x`$, the zero-remainder equation becomes

$$Uq^x+Ax+B+AQt\equiv0\pmod p.$$

For fixed $`x`$, the variable $`t`$ occurs only linearly:

$$t\equiv-\left(Uq^x+Ax+B\right)(AQ)^{-1}\pmod p.$$

Within the 18-digit interval, one need only test $`0\le t\le236`$ and the actual interval boundaries. The work consequently falls from about $`9\times10^{17}`$ integers to $`Q`$ orbit points, a reduction by a factor of approximately 236.

Rewrite the expression as

$$t(x)=\alpha q^x+\beta x+\gamma\pmod p,$$

$$\alpha=-U(AQ)^{-1},\qquad \beta=-Q^{-1},\qquad \gamma=-B(AQ)^{-1}\pmod p.$$

Maintaining $`z_x=\alpha q^x`$ and $`w_x=\beta x+\gamma`$ gives

$$z_{x+1}=qz_x,\qquad w_{x+1}=w_x+\beta\pmod p.$$

Each orbit point therefore needs just one fixed-modulus multiplication and a few modular additions.

## 4. The 17-digit and lower intervals

For 17-digit candidates, the initial filters give

$$k=a+6j,\qquad a\in\\{0,2\\}.$$

Here $`g=10^{17\cdot6}\bmod p`$ still has order $`Q`$. Writing $`j=x+Qt`$ again gives a linear congruence in $`t`$ for each fixed $`x`$:

$$t\equiv-\left(Uq^a g^x+Aa+B+Ahx\right)(AhQ)^{-1}\pmod p,\qquad h=6.$$

Each of the two residue classes scans $`Q`$ points, for $`2Q`$ points in total, covering approximately $`9\times10^{16}`$ 17-digit integers.

At 16 digits and below, a direct scan with step 6 is more economical. The relation $`r_k-Ak-B=Uq^k`$ removes the inhomogeneous exponential term: the power is updated with one modular multiplication per step and the linear term with additions. All stages together process

$$14,764,517,509,362,229$$

orbit points. The exact task boundaries come from `make_tasks()` in [src/run_all.py](src/run_all.py), not from a separate estimate in this document.

## 5. GPU implementation and sharding

The search used eight NVIDIA GPUs:

- GPUs 0–1: the complete 18-digit interval, with $`Q/2`$ points per GPU.
- GPUs 2–5: the two modulo-6 residue classes in the 17-digit interval, with each class split into two shards.
- GPUs 6–7: the intervals with at most 16 digits, comprising 30 tasks.
- Total: 36 independent shards.

Each thread initializes at an arbitrary orbit position using fast exponentiation, then processes 65,536 consecutive points. Measured throughput was approximately $`3.9\times10^{11}`$ to $`4.4\times10^{11}`$ orbit points per second per GPU, and the complete search took about 80 minutes.

For the period-reduced paths, the core loop is:

```text
z = alpha * g^x mod p
w = beta * x + gamma mod p

repeat:
    t = (z + w) mod p
    if t is in the permitted small range:
        reconstruct k
        check interval boundaries and k mod 6
        independently verify every reported hit

    z = Montgomery_multiply(z, gR)
    w = (w + beta) mod p
    x = x + 1
```

The arithmetic core is 64-bit Montgomery multiplication with $`R=2^{64}`$. Keep the orbit value in ordinary representation and store the fixed multiplier as $`gR\bmod p`$:

$$\mathrm{Mont}(z,gR)=zg\pmod p.$$

The output remains in ordinary representation and can enter the next iteration directly, eliminating per-point representation conversions.

After every CUDA kernel completes, the host atomically writes the configuration, `next_x`, hit count, and related fields to a `.state` file. One kernel launch takes about one second in this configuration, so an interruption requires recomputing at most the most recent kernel. Hits are first written to `.candidates` and then independently checked. A saved configuration must match the current task; checkpoints from different intervals cannot be mixed. Arbitrary-range splitting, resumption, and migration are possible because each thread can initialize independently at any $`x`$.

## 6. Independent remainder verification

The block-sum method in [src/verify.py](src/verify.py) works directly modulo the full composite modulus. It does not use the modular inverses or period properties required by the search.

For a block of integers with a fixed digit length, let $`s`$ be its first term, $`n`$ its length, and $`q=10^d`$. Define

$$S_0(n)=\sum_{i=0}^{n-1}q^i,\qquad S_1(n)=\sum_{i=1}^{n}iq^{n-i}.$$

The recurrence for the entire block is

$$r_{\mathrm{end}}=r_{\mathrm{start}}q^n+(s-1)S_0(n)+S_1(n)\pmod{c(14)}.$$

Adjacent blocks of lengths $`a`$ and $`b`$ can be combined by binary doubling:

$$S_0(a+b)=q^bS_0(a)+S_0(b),$$

$$S_1(a+b)=q^bS_1(a)+aS_0(b)+S_1(b).$$

The verifier therefore needs only additions and multiplications, even when the modulus is composite. The search for $`a(15)`$ reused the same general function; the `M14` self-test constant in that file does not restrict `concat_mod()` to $`a(14)`$.

Checks performed for this implementation included:

- Literal concatenation through 1000 and 110 large-$`k`$ comparisons against independent fast formulas.
- 100,000 randomized GPU Montgomery products checked against host-side 128-bit arithmetic.
- Full composite-modulus verification of `675924569493048966`, plus configuration, endpoint, and completion-log audits for all 36 shards.
- Divisibility and exhaustive minimality checks for the first 13 published OEIS values, detailed in the next section.

## 7. Verifying the first 13 terms

For every $`n=2,\ldots,13`$, [src/verify_oeis.cu](src/verify_oeis.cu) exhaustively checks $`n < k < a(n)`$, split across eight GPUs. For $`n=1`$, minimality is immediate because $`2`$ is the smallest integer greater than $`1`$. The scanned modulus is the odd part obtained by removing factors of 2, while last-digit parity is handled during the scan; the published value is separately checked modulo the full modulus.

| $`n`$ | $`a(n)`$ | $`n`$ | $`a(n)`$ |
|---:|---:|---:|---:|
| 1 | 2 | 8 | 795420 |
| 2 | 6 | 9 | 37366496 |
| 3 | 23 | 10 | 8803255110 |
| 4 | 1838 | 11 | 1462927131789 |
| 5 | 15000 | 12 | 36810773632272 |
| 6 | 6992 | 13 | 2879846318370304 |
| 7 | 2888377 |  |  |

The audit confirms that every published value has remainder zero and that no smaller $`k`$ was reported. It does more than check that the OEIS values happen to be divisible by their respective moduli.

## 8. Audit, build, and reproduce

From the repository root:

```bash
# No GPU required: check block sums, self-tests, 36 shards, and every saved candidate
make -C a14 audit

# No GPU required: read the complete coverage evidence for the first 13 terms
make -C a14 verify-first13

# Build; NVCC and GPU_ARCH can be overridden
make -C a14 NVCC=/usr/local/cuda-12.9/bin/nvcc GPU_ARCH=native

# Small CUDA numerical self-tests, not a full interval rescan
make -C a14 selftest GPU_ID=0
```

The full search entry point is `cd a14/src && python3 run_all.py`, followed by `python3 audit_results.py`. It uses eight GPUs and resumes any checkpoints already present. For a genuinely independent run, copy the source and create an empty `results/` directory in a new working directory; do not mix incomplete new results with the completed evidence in this project.

To rescan the first 13 terms on GPUs, run `cd a14/src && python3 verify_first13.py`. The `--audit` option only reads archived results, while `verify_oeis selftest 0` runs numerical self-tests alone.
