# a(15): Period Reduction, Four-Coset Sharing, and CPU/GPU Cooperation

**[中文](README.md) | English**

## Final result

$$\boxed{a(15)=70685893246361615705}$$

Verification modulo the full composite modulus gives

$$c(70685893246361615705)\bmod 123456789101112131415=0.$$

The 20-digit interval contains three solutions. With $`Q=205761315168520219`$, each one can be reconstructed directly from its saved orbit parameters:

$$k=x+Qt.$$

| $`k`$ | $`x`$ | $`t`$ | $`k\bmod15`$ |
|---|---:|---:|---:|
| [`70685893246361615705`](results15/d20_x4.candidates) | `109762143559180588` | `343` | `5` |
| [`73394079098572402415`](results15/d20_x5.candidates) | `143050898579204451` | `356` | `5` |
| [`74341537919751115530`](results15/d20_x2.candidates) | `61703143915316471` | `361` | `0` |

There are no solutions with 18 or 19 digits, and the verification of intervals with at most 17 digits also found none. The smallest 20-digit value is therefore $`a(15)`$. All three candidates were checked modulo the full modulus using both block sums and matrices.

- [Final independent audit log](results15/final_independent_audit.log)
- [Final audit marker](results15/final_independent_audit.txt), containing `FINAL_INDEPENDENT_AUDIT_OK`.
- [Audit of the 20-digit candidates](results15/d20_complete.audit.txt)
- [Audit of 19-digit coverage](results15/d19_complete.audit.txt)
- [Measured optimization timeline](results15/speed_history.log)
- [Rejected experiments and engineering trade-offs](EXPERIMENTS.md) (Chinese)

Approximately 69 hours elapsed from the first GPU task to the final audit, including multiple optimization rounds and necessary resumed runs. The hardware consisted of eight NVIDIA GPUs and 272 AVX-512 CPU worker threads, on a machine with 288 hardware threads.

## 1. Factorization and the common closed form

The correct factorization for this problem is

$$c(15)=123456789101112131415=3\cdot5\cdot p=15p,$$

$$p=8230452606740808761.$$

Here $`p`$ is prime and

$$p-1=40Q,\qquad Q=205761315168520219.$$

This $`Q`$ is exactly the large prime factor called $`p`$ in the search for $`a(14)`$. The last decimal digit gives a condition modulo 5 and the digit sum gives one modulo 3, yielding the candidate filter

$$k\bmod15\in\\{0,5\\}.$$

Within an interval of fixed digit length $`d`$, define

$$q=10^d\bmod p,\qquad A=(1-q)^{-1}\pmod p,\qquad B=-q(1-q)^{-2}\pmod p.$$

As for $`a(14)`$,

$$c(k)\equiv Uq^k+Ak+B\pmod p,$$

$$U=\left(c(s-1)-A(s-1)-B\right)q^{-(s-1)}\pmod p,$$

where $`s=10^{d-1}`$ and $`c(s-1)`$ above is computed modulo $`p`$.

If $`k=hj`$, $`g=q^h`$, and $`\mathrm{ord}_p(g)=L`$, write $`j=x+Lt`$ with $`0\le x < L`$. Then

$$c(hj)\equiv Uq^{hj}+Ahj+B\equiv D(t-t_*)\pmod p,$$

$$D=AhL,\qquad t_*=\alpha g^x+\beta x+\gamma\pmod p,$$

$$\alpha=-UD^{-1},\qquad \beta=-L^{-1},\qquad \gamma=-BD^{-1}\pmod p.$$

The inverse formulas require $`q\ne1\pmod p`$ and $`D=AhL\ne0\pmod p`$. All production jobs satisfy both conditions.

Thus each $`x`$ produces one small integer $`t_*`$, rather than requiring every $`k`$ in an arithmetic progression to be tested individually. [src/validate15.py](src/validate15.py) independently cross-checks the exact formulas and interval boundaries.

## 2. Strategies by interval

| Interval | Parameterization and principal method | Result |
|---|---|---|
| 2–17 digits | $`k=a+15j`$, $`a\in\\{0,5\\}`$; direct zero tests with CPU AVX-512 | `13333333333333331` orbit points checked, no solution |
| 18 digits | Exhaust both classes $`k=a+15j`$; $`g=q^{15}`$ and direct zero tests | No solution |
| 19 digits | $`k=5j`$, $`L=4Q`$; $`t\in\\{0,1,2\\}`$; four cosets share modular multiplications | `823045260674080876` points covered or pruned, no solution |
| 20 digits | $`k=j`$, $`L=Q`$; $`0\le t\le486`$ | Three solutions, the smallest being the answer |
| 21-digit fallback | $`k=5j`$, $`L=4Q`$; $`0\le t\le243`$ | Not needed to completion because a 20-digit solution exists |

The low-digit search starts at $`k=16`$ because the definition requires $`k > 15`$. Each CPU value of $`j`$ represents one candidate after modulo-15 filtering, so “orbit points” in the table are not counts of original integers.

### 2.1 Digits 2–17: AVX-512 with 272 threads

[src/cpu15_avx.cpp](src/cpu15_avx.cpp) processes eight 64-bit orbit values at once. A 64×64→128-bit product is assembled from 32-bit products and carries, and Montgomery multiplication likewise retains eight-way vector form. Each thread owns an independent, resumable interval. Work proceeds in chunks of one million points with periodic checkpoints, rather than placing every task in one sequential queue.

The numerical self-test compares 1,000,000 scalar Montgomery products with host-side 128-bit arithmetic, followed by 10,000 eight-lane batches (80,000 values) against the scalar implementation. It also checks direct concatenation through 1000, 100 larger inputs, and the intentional known hit at $`k=15`$. The `selftest-cpu` target invokes these checks through the shared wrapper, which removes the intentional test candidate afterward.

The coverage audit checks the exact configuration of 272 endpoints for every `d` and every `a`. In total it accounts for

$$13,333,333,333,333,331$$

orbit points, with no earlier candidate. See [cpu_results/audit.txt](cpu_results/audit.txt) and the final audit.

### 2.2 Eighteen digits: direct remainders and a seamless CPU/GPU handoff

Put $`k=a+15j`$. The production CUDA program maintains

$$z_j=Uq^a g^j,\qquad w_j=A(a+15j)+B,\qquad g=q^{15},$$

and directly tests whether $`z_j+w_j\bmod p`$ is zero at each point. It does not use a small $`t`$ to discard values of $`j`$ that have not been enumerated, so both residue classes are scanned over their complete legal intervals. The sign structure $`g^Q=-1`$ is also an important algebraic property of this task, but the production zero test does not depend on an unchecked bound on $`t`$.

While the GPUs were busy with higher intervals, the CPU preprocessed prefixes of the 18-digit interval. The shared queue records a continuous prefix of completed blocks, not the maximum allocated position, so an interruption cannot leave holes. If `next_work` is the next unallocated position and `active_i` is thread $`i`$'s current block start, the safe checkpoint is

$$x_{\mathrm{safe}}=\min(next\\_work,active_0,\ldots,active_{271}).$$

Idle threads store an infinite sentinel in `active_i`. On a stop request, no new blocks are taken, all active blocks finish, and only then is the final `clean_checkpoint=1` state written. A GPU takes over the suffix only at such a clean boundary.

Both classes use the half-open range $`6666666666666667\le j < 66666666666666667`$, with these exact handoffs:

| Class | CPU-completed prefix | GPU suffix start | CPU points |
|---|---|---:|---:|
| $`a=0`$ ([state](cpu_results/d18_a0_shared.state)) | $`[6666666666666667,10852848839666667)`$ | `10852848839666667` | `4186182173000000` |
| $`a=5`$ ([state](cpu_results/d18_a5_shared.state)) | $`[6666666666666667,7700737376666667)`$ | `7700737376666667` | `1034070710000000` |

Each suffix was divided among eight GPU shards. The boundary audits confirm zero overlap and zero gaps; see also the [second-class CPU/GPU handoff evidence](results15/cpu18_a5_handoff_complete.audit.txt).

### 2.3 Nineteen digits: four residue classes share an exponential orbit

Here $`k=5j`$, and $`g=q^5`$ has order

$$L=4Q=823045260674080876.$$

Within the interval, $`j=x+Lt`$ only requires $`t\in\\{0,1,2\\}`$. Decompose further:

$$x=uQ+r,\qquad u\in\\{0,1,2,3\\},\qquad 0\le r < Q.$$

Since $`\mathrm{ord}_p(g^Q)=4`$,

$$g^x=g^r(g^Q)^u.$$

For each $`r`$, initialize $`z=\alpha g^r`$ and $`w=\beta r+\gamma`$ once. All four values of $`u`$ share the pointwise update of $`z`$; each class then applies its own order-four factor and updates its linear term. The average cost is approximately half a Montgomery multiplication per orbit point, instead of one full multiplication per point.

The parameterization $`k=5j`$ is a superset chosen to make the reduction convenient. The host still checks $`k\bmod15\in\\{0,5\\}`$, so the additional modulo-3 cases cannot be reported as false solutions. The complete search found no 19-digit solution.

Two coverage-preserving migrations took place during production:

1. Serial checkpoints were converted to four-coset checkpoints; see the [migration record](results15/d19_coset4_migration.txt).
2. The remaining work was redistributed into 15 jobs so that eight GPUs would finish as closely together as possible; see the [balance manifest](results15/d19_balance_jobs.txt) and [migration hashes](results15/d19_balance_migration.txt).

#### Exact serial-to-coset checkpoint mapping

For a new shard $`v\in\\{0,\ldots,7\\}`$ and coset $`u\in\\{0,1,2,3\\}`$, define the $`r`$-interval and originating serial GPU by

$$r_{\mathrm{lo}}=\left\lfloor\frac{Qv}{8}\right\rfloor,\qquad r_{\mathrm{hi}}=\left\lfloor\frac{Q(v+1)}8\right\rfloor,\qquad e=2u+\left\lfloor\frac v4\right\rfloor.$$

Let $`N_e`$ be the old serial `next_x` on GPU $`e`$. The converted checkpoint is

$$n_{v,u}=r_{\mathrm{lo}}+\min\left(r_{\mathrm{hi}}-r_{\mathrm{lo}},\max\left(0,N_e-uQ-r_{\mathrm{lo}}\right)\right).$$

For $`x=uQ+r`$ in this shard, the old and new representations describe exactly the same completed prefix:

$$r < n_{v,u}\quad\Longleftrightarrow\quad x < N_e.$$

The clipping at both ends is therefore essential: it preserves already completed work without introducing overlap or gaps. [src/validate15_coset4.py](src/validate15_coset4.py) checks this equivalence, total point counts, directed boundaries, and 100 randomized checkpoint mappings.

#### Coverage and pruning accounting

For a balanced job $`[B,E)`$, let $`\ell_u`$ be the checkpoint's stored stopping point in coset $`u`$, clipped to the job, and let $`n_u`$ be the saved `next_r[u]`. With no known 19-digit candidate, the global exponent limit is $`x_{\lim}=4Q`$; after finding $`k_{\mathrm{best}}`$, it becomes $`\min(4Q,k_{\mathrm{best}}/5)`$. Every $`x < x_{\lim}`$ must be covered, while $`x\ge x_{\lim}`$ can safely be credited as pruned.

The checkpoint's completed-or-pruned count and displayed `next_x` are

$$C=\sum_{u=0}^{3}\left((n_u-B)+(E-\ell_u)\right),\qquad N=4(E-B),$$

$$\texttt{next\\_x}=B+(E-B)\left\lfloor\frac CN\right\rfloor.$$

Here $`n_u-B`$ is the scanned prefix and $`E-\ell_u`$ is the pruned suffix. Thus $`C=N`$ and `next_x=E` do **not** imply that all $`4(E-B)`$ points were physically scanned. The final audit sorts and merges the old prefixes and new segments, adds only the safely pruned tails, and checks that the union has no overlap or omission.

Balancing reduced the estimated makespan from 15.707 hours to 12.704 hours, an estimated saving of about 3 hours of tail waiting. This was an estimate based on measured rates at the time, not an independently measured speedup. The final coverage accounting confirms

$$\text{covered\\_or\\_pruned}=823045260674080876,$$

without double counting or omissions. If a smaller $`k`$ is found, the corresponding search bound can be lowered to $`k/5`$; only the region $`x\ge x_{\lim}`$ is pruned.

### 2.4 Twenty digits: find an upper bound, then rule out smaller solutions

For the 20-digit interval, take $`h=1`$ and $`L=Q`$. Since $`\mathrm{ord}_p(10^{20})=Q`$, each $`x`$ directly recovers $`0\le t\le486`$. Eight GPUs each process $`Q/8`$ orbit points.

The scheduler did not wait for every high-digit interval to finish before considering lower intervals. It first obtained a 20-digit candidate, then completed the 19-digit, 18-digit, and lower-digit checks, returning to finish any prematurely stopped high-digit suffix when necessary. This utilized CPUs and GPUs concurrently and reduced subsequent search bounds as early as possible. The final complete 20-digit search contains three solutions, while no solution exists with at most 19 digits.

The 21-digit fallback uses $`h=5`$ and $`L=4Q`$, giving $`t\le243`$. It did not need a full run because a 20-digit answer had already been found. When building the fallback, `T_LIMIT=243` must be defined; retaining the default value 2 for the 19-digit source would be incorrect.

## 3. Making the GPU hot loop cheaper

Every production version uses Montgomery multiplication with $`R=2^{64}`$. The fixed multiplier is stored as $`gR\bmod p`$, and orbit values stay in ordinary representation:

$$\mathrm{Mont}(z,gR)=zg\pmod p.$$

Threads initialize at arbitrary positions using fast exponentiation, allowing work to be split, resumed, and migrated without recomputing from the interval start.

The principal measured improvements are listed below. Different rows have different baselines, intervals, and measurement stages, so the percentages **cannot simply be multiplied together**:

| Improvement | Measured effect |
|---|---:|
| Direct modular subtraction instead of more complicated reduction | +6.809% |
| Signed-bit tests and residue-class alignment | +15.804% |
| Negative stepping of the 19-digit linear term | +1.517% |
| Lazy normalization for small remainders | +3.055% |
| Combined negative stepping and lazy testing, 20 digits / 18 digits | +4.821% / +9.176% |
| Odd fixed multiplier for 18 digits, idle benchmark | +6.728% |

The small-remainder test first checks whether an unnormalized sum lies near $`0`$ or a small positive value. Only a possible hit undergoes the final modular reduction, avoiding the same cost at every point for an event that almost never occurs. Negative stepping, signed tests, and the other optimizations were accompanied by boundary tests, randomized comparisons, and checkpoint-resume tests. The complete figures are in [speed_history.log](results15/speed_history.log).

### Exact 64-bit reduction invariants

The signed arithmetic is valid because $`p < 2^{63}`$ and $`2p < R=2^{64}`$. For $`0\le a,b < p`$, the sum $`s=a+b`$ satisfies $`0\le s < 2p < R`$, so the addition itself cannot overflow. In unsigned arithmetic, $`s-p\bmod R`$ has its high bit set exactly when $`s < p`$; interpreting it as a signed 64-bit value therefore selects the correct conditional subtraction. For negative stepping, `a-negative_step` lies strictly between $`-p`$ and $`p`$, so a signed test followed by one addition of $`p`$ is likewise exact.

For $`s=z+w`$, the lazy small-candidate predicate is equivalent to

$$\mathrm{small}_T(s)=(s\le T)\ \lor\ \big(((s-p)\bmod R)\le T\big),\qquad 0\le s < 2p < R.$$

The second subtraction is unsigned. It is small exactly in the branch $`s\ge p`$ and $`s-p\le T`$; in the underflow branch it is too large to pass. Here $`T`$ is the kernel's small candidate bound, such as 0, 2, or 486. Canonical modular normalization is only needed when this predicate succeeds. The deployed direct kernels checked this predicate on 100,000 random inputs and 6,000 directed boundary cases; the 18-digit self-test also checked 134 independently computed actual remainders.

The Montgomery implementation uses $`m\equiv-abp^{-1}\pmod R`$. With $`lo=ab\bmod R`$, $`hi=\lfloor ab/R\rfloor`$, and $`mhi=\lfloor mp/R\rfloor`$, cancellation modulo $`R`$ gives the exact quotient expression

$$r=\frac{ab+mp}{R}=hi+mhi+(lo\ne0).$$

The final term is 0 when `lo==0` and 1 otherwise. This identity is the basis for the following odd-multiplier specialization.

### Odd fixed multiplier: eliminating the `(lo != 0)` test

This was the final optimization deployed for the 18-digit search. Let an orbit value satisfy $`0 < a < p`$, and let the fixed multiplier be $`b`$. If $`b`$ is even, use the congruent representative $`b+p`$ instead. Since $`p`$ is odd, the new multiplier is always odd.

In Montgomery reduction, $`R=2^{64}`$ and the multiplier is odd. If the low product were $`lo=(ab)\bmod R=0`$, then $`R\mid ab`$, which would imply $`R\mid a`$, contradicting $`0 < a < p < R`$. The low product therefore cannot be zero on this orbit, the Montgomery carry is always 1, and the code can change from

```cuda
r = hi + mhi + (lo != 0);
```

to

```cuda
r = hi + mhi + 1;
```

Initialization still uses general Montgomery multiplication. Only fixed stepping that satisfies the nonzero-orbit premise uses this special case. Before deployment, the transformation passed 100,000 general products, 100,000 odd-fixed products, independent traces, and actual 18-digit boundary checks.

Average throughput over eight repeated benchmarks increased from `531481500000` to `567241000000` points per second, a gain of 6.728%. A production checkpoint measured `571588252417.3169` points per second. See the [numerical verification](results15/direct_odd_verification.txt), [benchmark](results15/direct_odd_benchmark.txt), and [deployment record](results15/direct_odd_deployment.txt).

These verification files are chronological field records. For example, `production_changed=NO` or `benchmark_result_pending=YES` in an early file describes the pre-deployment state and must be read together with later benchmark and deployment logs; it does not supersede those later records.

## 4. Two independent remainder verifiers

### Block sums

For a block of integers with the same digit length, [src/verify.py](src/verify.py) computes

$$S_0(n)=\sum_{i=0}^{n-1}q^i,\qquad S_1(n)=\sum_{i=1}^{n}iq^{n-i},$$

and combines blocks by doubling:

$$S_0(a+b)=q^bS_0(a)+S_0(b),$$

$$S_1(a+b)=q^bS_1(a)+aS_0(b)+S_1(b).$$

The remainder of the whole block is $`r_{\mathrm{start}}q^n+(s-1)S_0(n)+S_1(n)`$. It does not depend on inverses modulo a prime and works directly modulo the full $`c(15)`$.

### Matrices

[src/verify15_matrix.py](src/verify15_matrix.py) uses a different recurrence:

$$\begin{pmatrix}r'\\\\k'\\\\1\end{pmatrix}=
\begin{pmatrix}q&1&0\\\\0&1&1\\\\0&0&1\end{pmatrix}
\begin{pmatrix}r\\\\k\\\\1\end{pmatrix}\pmod M.$$

Matrix exponentiation over each digit-length interval yields the same remainder modulo the full modulus. Its self-test also cross-checks direct concatenation for small integers, the block-sum method, and 100 pseudorandom inputs with a fixed seed.

Neither verifier uses the GPU search's $`t_*`$ formula. The final audit additionally checks every search phase's configuration, endpoints, completion logs, and candidates. The single assertion `concat_mod(answer, M15)==0` is not, by itself, a proof of minimality.

## 5. Engineering reliability and records

- State files are written to `.tmp` and atomically renamed. A configuration mismatch prevents resumption.
- Candidates are written immediately and independently checked on the host.
- Four-coset and balance migrations retain old checkpoints, exact manifests, and SHA-256 hashes, rather than just a progress percentage.
- CPU handoff uses a clean continuous prefix. Stop markers, active per-thread blocks, and final checkpoints jointly prevent gaps.
- `status15.py` once subtracted the first CPU prefix twice and could produce a negative ETA. This affected only the display. The fix sums only unfinished GPU suffixes and does not change search states or candidates. See the [status-fix audit](results15/status15_d18_remaining.audit.txt).
- This curated release excludes approximately 537 MB of duplicated historical snapshots while retaining about 68 MB of active `results15/` evidence. The final audit does not require the excluded temporary dry-run files. To create a new local snapshot in this reorganized layout, run `cd a15/src && python3 snapshot15.py --once`; newly generated snapshots are ignored by `.gitignore`.

SHA-256 hashes of the original production binaries, provided only to identify the binaries in historical logs:

| Component | SHA-256 |
|---|---|
| 20-digit `search15_b1024_signed` | `77b301c9ff588a7727523e57478e61ee06482cc3cfde5a1097722bff34def751` |
| 19-digit balanced `search15_balanced` | `67e7b1fc23a73d66c1a0af5921742d122cc692596839ef824765362cd1c47e72` |
| 18-digit odd-multiplier `search15_direct_signed` | `f6e02bd31e729e4f9581e966244f8274ec12f8b75ef93d12a5e2edd8cd34e073` |

The SHA-256 of the corresponding 18-digit **source**, [search15_direct_odd.cu](src/search15_direct_odd.cu), is `44a981241dc66f14dc45b1ad9e806d8db3fd26b1e4a86f169754aedf0ac5ac57`. The CPU source `cpu15_shared.cpp` has hash `383a8e94a4105bcb6899f99dac151b88db6a125313544fa11a94a01f27a1566d`, and the production CPU executable had hash `a4636a3f62650c95493836d83fad283043d093cd0b1652120e6cb32f5c1cde69`. Do not confuse source hashes with binary hashes, and do not require builds in a new environment to be byte-identical.

## 6. Audit and self-tests

From the repository root:

```bash
python3 -m pip install -r a15/requirements.txt

# No GPU required: final joint audit, printing FINAL_INDEPENDENT_AUDIT_OK
make -C a15 audit

# Formula, boundary, coset-mapping, and pruning cross-checks
make -C a15 validate

# Build every production program without starting a search
make -C a15 NVCC=/usr/local/cuda-12.9/bin/nvcc GPU_ARCH=native

# Lightweight numerical self-tests; selftest-cpu / selftest-gpu also work separately
make -C a15 selftest GPU_ID=0
```

[src/audit15_final.py](src/audit15_final.py) accepts only the known 18–20 digit answer scenario; it is not an automatic generalization to arbitrary digit lengths. `--candidate K` verifies a candidate without claiming minimality.

## 7. Correct order for a complete run from scratch

**Warning: the following are long-running jobs, not ordinary tests.** Copy the project and use empty `results15/` and `cpu_results/` directories in a new working directory, retaining the relative-link structure in `src/`. Build first, then start these tasks separately:

```bash
cd a15/src

# Terminal 1: low-digit CPU search, 272 threads; campaign waits for it but does not start it
./cpu15_avx all

# Terminal 2: initial 20-digit GPU search on eight GPUs
python3 run15_d20_fast.py

# Terminal 3: watch candidates and continue with d19, d18, or fallback d21 as appropriate
python3 campaign15.py
```

Run the coordinator only in a dedicated search environment. Its `stop_phase()` uses `pkill -f` to stop matching command lines and `pgrep -f` to wait for them to exit, so an unrelated process with a matching name may also be stopped. The supplied configuration assumes eight GPUs and 272 CPU workers.

`campaign15.py` starts neither the initial d20 search nor the low-digit CPU search; without those prerequisites, it waits. If d20 produces no candidate, the coordinator enters d21. Once a high-digit candidate exists, it runs d19 and d18 and waits for the low-digit CPU search. If there is no earlier candidate, it completes the required high-digit suffixes. Finally, run this explicitly:

```bash
python3 audit15_final.py
```

The completed evidence included here can be used to practice checkpoint continuation, but that is not equivalent to a search from scratch. The 19-digit four-coset conversion and rebalancing are migrations during a run: `prepare15_coset4.py --prepare` and `balance15_d19.py --prepare` require corresponding old checkpoints, and a migration manifest must not be fabricated without those states. A fresh task without a manifest uses the serial period scanner. To enable an optimized path, generate and audit a new manifest after satisfying the prerequisites above.

A complete 18-digit job can directly use `run15_d18.py`, which supports both CPU-prefix classes. `transition15_a5.py` exists only to reproduce the controlled handoff from the old coordinator to the new coordinator used at the time; a new run does not need this historical transition utility. It deliberately asserts the original script and binary hashes, which must not be changed to bypass its safety checks.

For detailed per-phase commands, see `command()` in `run15_d18.py`, `run15_d19.py`, `run15_d20_fast.py`, and `run15_d21.py`. Their production compilation macros match the Makefile.
