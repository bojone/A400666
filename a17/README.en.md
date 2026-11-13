# a(17): Small-Factor Filtering, Periodic Reduction, and an Eight-GPU Search

**[中文](README.md) | English**

## Final result

$$\boxed{a(17)=553943104744829340555260}$$

There is no complete solution in the 2–23-digit intervals, and the complete 24-digit interval contains exactly this one solution. The candidate is therefore also the minimum supported by the coverage evidence.

Its main-orbit parameters are

$$a=8,\qquad x=68524741962902104,\qquad t=631772,\qquad L=R=97423066131127417,$$

so the value can be reconstructed directly:

$$k=a+9(x+Lt)=553943104744829340555260.$$

The remainder modulo the full composite modulus is

$$c(553943104744829340555260)\bmod 1234567891011121314151617=0.$$

Principal evidence:

- The [answer](results/answer.txt) and [final independent audit](results/final_independent_audit.txt), including `FINAL_A17_AUDIT_OK`.
- The [24-digit audit](results/main_d24.audit.txt), [candidate coordinates](results/d24_a8_x6.candidates), and [candidate verification record](results/candidate_553943104744829340555260.audit.txt).
- [Block-sum and matrix verification](results/matrix_verification.txt).
- The [hardware and timing summary](results/campaign_summary.txt), [24-digit campaign log](campaign17_main.log), and [final audit log](finalize17.log).

## 1. Factorization and the necessary condition modulo 9

$$c(17)=1234567891011121314151617=9\cdot47\cdot4993\cdot P,$$

where

$$P=584538396786764503,\qquad R=97423066131127417,\qquad P-1=6R.$$

The decimal digit sum of the concatenation gives

$$c(k)\equiv\frac{k(k+1)}2\pmod 9.$$

Thus only $k\bmod9\in\{0,8\}$ is possible. Write

$$k=a+9j,\qquad a\in\{0,8\}.$$

The two residue classes are searched independently, with four shards per class on eight GPUs.

## 2. Digits 2–20: small-factor residue classes and CPU filtering

For a fixed digit count $d$ and modulus $m$, let $q=10^d\bmod m$. When $q\ne1$, the closed form is

$$c(k)\equiv Uq^k+Ak+B\pmod m,$$

$$A=(1-q)^{-1},\qquad B=-q(1-q)^{-2}\pmod m.$$

The generator finds the zero residue classes over a complete period for $m=47$ and $m=4993$, combines them with the Chinese remainder theorem, and then includes the modulo-9 condition. It handles the degenerate case $q=1$ separately rather than applying these inverses.

- Digits 2–6 are checked by direct enumeration.
- Digits 7–20 scan only the merged residue classes, not every integer in the interval.
- The filter uses eight 64-bit AVX-512 lanes to advance the large-prime remainder, with 272 OpenMP workers processing independent groups.
- Every completed group updates a bitmap, the searched-point count, and the full-candidate count. A restart checks that the bitmap and count agree and does not repeat completed groups.

The 20-digit filter checked **85,225,698,957,274** candidate points and found no complete solution; digits 2–19 also produced none. See [filters/](filters/) for the residue data and [results/filter_d*.audit.txt](results/) for per-digit coverage evidence.

## 3. Digits 21–24: reducing the large-prime condition to one congruence

The GPU search handles $P$ first and then rechecks 47 and 4993 for complete candidates. For fixed $d$, again let $q=10^d\bmod P$, and define

$$g=q^9,\qquad L=\mathrm{ord}_P(g),\qquad D=9AL.$$

For every digit count used here,

$$L=\begin{cases}R,&d\text{ is even},\\ 2R,&d\text{ is odd}.\end{cases}$$

These orders are recorded independently in [orders17.audit.txt](results/orders17.audit.txt). Write

$$j=x+Lt,\qquad 0\le x < L.$$

Since $g^L=1$, the term $g^j$ no longer depends on $t$. The zero-remainder condition becomes

$$t_*=\alpha g^x+\beta x+\gamma\pmod P,$$

$$\alpha=-Uq^aD^{-1},\qquad \beta=-L^{-1},\qquad \gamma=-(Aa+B)D^{-1}\pmod P.$$

Every solution must have $t=t_*+mP$ for an integer $m\ge0$. An orbit point therefore requires mainly one linear-congruence evaluation rather than division of an enormous concatenated integer.

The exact interval boundaries are

$$j_{\min}=\left\lceil\frac{10^{d-1}-a}{9}\right\rceil,\qquad j_{\max}=\left\lfloor\frac{10^d-1-a}{9}\right\rfloor.$$

For digits 21–24, $\lfloor j_{\max}/L\rfloor$ is smaller than $P$, so $t_*+P$ is necessarily out of range and only $t_*$ itself needs checking. This is the proof behind the specialized `SINGLE_T` branch; it does not omit a valid second boundary point.

The orbit counts and audited results are:

| Digits | Orbit length per $a$ class | Total points for both classes | Pre-hits | Complete solutions |
|---|---:|---:|---:|---:|
| 21 | $2R$ | $4R$ | 346 | 0 |
| 22 | $R$ | $2R$ | 3,446 | 0 |
| 23 | $2R$ | $4R$ | 34,413 | 0 |
| 24 | $R$ | $2R$ | 343,046 | 1 |

Here $R=97423066131127417$. A pre-hit only passes the initial $P$ test; it is not necessarily a complete solution.

## 4. The CUDA inner loop

The kernel maintains

$$z_x=\alpha g^x,\qquad w_x=\beta x+\gamma\pmod P.$$

For each orbit point it:

1. updates $z_{x+1}=z_xg$ with a Montgomery multiplication;
2. updates $w_{x+1}$ with a negative-step modular addition;
3. computes $t_*=z_x+w_x\bmod P$;
4. immediately advances when $t_*$ is out of range, avoiding 128-bit coordinate reconstruction;
5. reconstructs $j$ and $k$ only for the rare in-range points, checks 47, 4993, and the modulo-9 condition on the host, and then independently verifies the full composite modulus in Python.

The fixed Montgomery multiplier is odd. A nonzero orbit value times an odd multiplier has a nonzero low product, so the carry in the Montgomery reduction is always one and a conditional operation can be removed. The inner-loop counter is 32-bit, while all hit coordinates remain 64-bit.

The production configuration unrolls 16 iterations, uses 128 threads per block, 262,144 consecutive points per thread, and 32,768 blocks per launch. Each shard atomically replaces its `.state` file with its exact configuration, endpoint, pre-hit count, and full-candidate count.

## 5. Measured optimization

The hardware was **eight NVIDIA GPUs** with CUDA 12.9, plus the AVX-512 CPU filter. The production binary SHA-256 was

```text
e8b5545c5dc0d409fb4c9e01334a6f0e0dd1ebbd81af691b28925090e15a8f8c
```

The main measured stages are listed below. Rates are orbit points per second per GPU:

| Change | Rate | Decision |
|---|---:|---|
| Initial periodic kernel | $3.390\times10^{11}$ | Baseline |
| Early rejection of out-of-range $t$ | $4.278\times10^{11}$ | Adopted |
| Single boundary point, odd multiplier, and negative step | $4.722\times10^{11}$ | Adopted |
| Unrolling and block tuning | $5.215\times10^{11}$ | Adopted |
| 128 threads and chunk=262144 | $5.23675\times10^{11}$ | Adopted |
| 32-bit inner-loop counter | approximately $5.3388\times10^{11}$ | Adopted |

The last counter change improved alternating real-production benchmarks by **1.966867%**, with approximately **1.9437%** observed in production. Before deployment it passed numerical self-tests and equivalent-range checks on a 22-digit interval containing a pre-hit and a partial 24-digit tail, then resumed from validated checkpoints. See the [optimization audit](results/counter32_optimization.audit.txt) and [speed_history.log](results/speed_history.log).

Final throughput was approximately $5.3388\times10^{11}$ points per second per GPU, about 1.57 times the initial single-GPU rate, or $4.2711\times10^{12}$ points per second across all eight GPUs. Unrolling 32 iterations, splitting Montgomery arithmetic into 32-bit pieces, alternative modular additions, and experiments with lower warp occupancy were slower and were not deployed. Their sources are in [experiments/](experiments/); the production implementation is [src/search17.cu](src/search17.cu).

## 6. Elapsed time, recovery, and the final low-digit gap

The overall exploration ran from the first production-source timestamp, **2026-11-08 09:58:42 UTC**, to the final audit at **2026-11-11 17:30:35 UTC**, a span of approximately **79.53 hours**. This includes algorithm development, optimization experiments, controlled stops, checkpoint resumes, the low-digit CPU recovery, and audits; it is not the running time of one uninterrupted batch job.

- The first complete candidate was saved around **2026-11-11 14:59 UTC**, when the 24-digit search was approximately 81.43% complete.
- The search continued through every 24-digit shard to exclude a smaller solution in the same interval.
- The first final audit found that only `2043/2392` groups of the 20-digit CPU filter had completed, so minimality could not yet be claimed.
- The validated bitmap checkpoint was resumed to `2392/2392`, with no complete solution. A separate 120-second command timeout was also resumed from its checkpoint.
- The answer-writing logic was corrected: an audit failure can leave only `PENDING`, while the completion state is written only after a successful audit. An injected-failure test covers this behavior.
- `FINAL_A17_AUDIT_OK` was generated at **2026-11-11 17:30:35 UTC**.

Historical logs retain tracebacks from controlled stops and the low-digit gap described above. They are not substitutes for the successful final audit. Completion should be judged from the exact shard endpoints, corresponding `DONE` records, and the [final independent audit](results/final_independent_audit.txt). The final dedicated snapshot was `20261111T173141Z`, containing 146 files with a verified manifest; this release does not duplicate every historical snapshot.

## 7. Independent verification and the scope of the audit

[src/verify17.py](src/verify17.py) uses binary block-sum recurrences, while [src/verify17_matrix.py](src/verify17_matrix.py) independently uses $3\times3$ matrix exponentiation. Appending an integer $i$ transforms the state $(r,i,1)^T$ by

$$T=\begin{pmatrix}q&1&0\\ 0&1&1\\ 0&0&1\end{pmatrix},\qquad q=10^d.$$

The matrix implementation was checked against 200 small literal concatenations and another 103 block-sum results. Both methods return remainder zero modulo the full $c(17)$, not merely modulo $P$.

Materializing $c(k)$ for this answer would require approximately $1.31835\times10^{25}$ decimal digits. The verifiers instead use equivalent modular recurrences and never attempt to store that decimal string.

`make audit` checks every shard configuration, exact endpoint, completion log, CPU bitmap, theoretical coverage count, candidate coordinate, and full-modulus remainder, and confirms the absence of lower-digit solutions. It does not rescan the entire space in a few seconds; minimality relies on the archived coverage evidence, the algorithmic proof, and independent candidate verification.

## 8. Build, audit, and reproduce

From the repository root:

```bash
# Coverage audit and two independent remainder checks; no GPU or SymPy required
make -C a17 audit

# Numerical and matrix self-tests
make -C a17 validate

# Compile the CUDA search and CPU filter without starting a search
make -C a17 NVCC=/usr/local/cuda-12.9/bin/nvcc GPU_ARCH=native

# GPU numerical tests and the AVX-512 CPU test
make -C a17 selftest GPU_ID=0

# Inspect the archived checkpoints
make -C a17 status
```

Auditing and verification use only the Python standard library. SymPy is needed only to regenerate small-factor residue classes with `residues17.py`; the environment used version 1.14.0, installable with `python3 -m pip install -r a17/requirements.txt`.

The complete reproduction entry point is:

```bash
make -C a17 run
```

It runs the 2–20-digit CPU filter, the 21–24-digit GPU search, and the final audit in order. Because the archived results are complete, this command rechecks or skips completed shards instead of rescanning them. For an independent run, copy `src/` and use fresh empty `results/` and `filters/` directories rather than overwriting the archived evidence; install the dependency above before regenerating filters. The production CPU program requires AVX-512 in the compilation target. The default GPU architecture is `native`; other architectures require fresh numerical and performance validation.
