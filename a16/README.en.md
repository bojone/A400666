# a(16): Two Small Prime Factors and Windowed Exponentiation

**[中文](README.md) | English**

## Result

$$\boxed{a(16)=10152437877876943330588}$$

The answer lies in the 23-digit interval. Every interval with at most 22 digits completed without a full solution. The 23-digit interval produced ten solutions, of which this is the smallest. See [all ten candidates](results/d23.summary).

For the winning candidate, the primary orbit parameters are

$$x=130029847818,\qquad t=16494262049,\qquad L=153878328221,$$

and direct reconstruction gives

$$k=4(x+Lt)=10152437877876943330588.$$

Verification modulo the complete modulus gives

$$c(10152437877876943330588)\bmod 12345678910111213141516=0.$$

Principal evidence:

- [Answer](results/answer.txt) and [final independent audit](results/final_independent_audit.txt), containing `FINAL_A16_AUDIT_OK`.
- [23-digit summary](results/d23.summary): all ten candidates and the primary-hit count.
- [Block-sum and matrix verification](results/matrix_verification.txt).
- [Campaign log](campaign16.log), [per-shard states, logs, and candidates](results/).
- [Measured campaign summary](results/campaign_summary.txt).

## 1. Factorization and the search space

$$c(16)=12345678910111213141516=4p_1p_2,$$

where

$$p_1=1231026625769,\qquad p_2=2507191691.$$

Their predecessor factorizations are

$$p_1-1=2^3\cdot29\cdot113\cdot1993\cdot23561,$$

$$p_2-1=2\cdot5\cdot1381\cdot181549.$$

Because $`4\mid c(k)`$ is required for the relevant values of $`k`$, write $`k=4j`$. The large prime factors used for a14 and a15 are absent here: both prime moduli are small enough for inexpensive 64-bit arithmetic, and $`p_1`$ has a particularly useful multiplicative order.

For every digit interval $`2\le d\le28`$ used by this build,

$$L=\mathrm{ord}_{p_1}(10^{4d})=153878328221.$$

Digit 29 has a different order and is deliberately rejected by both the manager and the CUDA initialization routine. It is not needed for the completed result because a solution was found and fully audited at 23 digits.

## 2. Primary scan: one linear congruence per orbit point

For a fixed digit length $`d`$, put $`q=10^d\bmod p_1`$ and use the fixed-digit closed form

$$c(k)\equiv Uq^k+Ak+B\pmod{p_1},$$

$$A=(1-q)^{-1}\pmod{p_1},\qquad B=-q(1-q)^{-2}\pmod{p_1}.$$

Here $`U`$ is determined by the remainder at the beginning of the interval. Let

$$g=q^4,\qquad D=4AL,\qquad j=x+Lt,\qquad 0\le x < L.$$

Then $`g^x`$ repeats with period $`L`$, and the zero-remainder condition can be solved for $`t`$:

$$t_*=\alpha g^x+\beta x+\gamma\pmod{p_1},$$

$$\alpha=-UD^{-1},\qquad \beta=-L^{-1},\qquad \gamma=-BD^{-1}\pmod{p_1}.$$

Every actual solution has $`t=t_*+mp_1`$ for some integer $`m\ge0`$. The CUDA loop tests all such $`t`$ whose $`j`$ lies in the actual interval, rather than assuming that only $`t_*`$ itself is valid. A pair of 64-bit words represents $`j`$ and $`k`$, so candidates above $`2^{64}`$ are handled without overflow.

The kernel maintains

$$z_x=\alpha g^x,\qquad w_x=\beta x+\gamma\pmod{p_1},$$

using one Montgomery multiplication and a few additions per point. The fixed multiplier is held in Montgomery-scaled form while the orbit value remains in ordinary representation, so no per-point conversion is needed.

Each interval is split across eight NVIDIA GPUs. A thread processes up to 65,536 consecutive orbit positions; CUDA blocks contain 256 threads, and a launch contains 512 blocks. Every launch atomically replaces its `.state` file with the exact configuration, `next_x`, primary-hit count, and full-hit count. Completed shards are skipped on restart.

## 3. Checking the second prime with four window tables

A primary hit only proves divisibility by $`p_1`$. The second condition is checked independently in the kernel.

For the same digit interval, define $`q_2=10^d\bmod p_2`$, $`g_2=q_2^4`$, and $`h=g_2^L\bmod p_2`$. With the corresponding coefficients $`U_2,A_2,B_2`$, the second remainder is

$$r_2=U_2g_2^x h^t+4A_2x+B_2+(4A_2L)t\pmod{p_2}.$$

The terms involving $`x`$ are updated incrementally. The potentially large exponent $`h^t`$ is evaluated with four tables:

$$T_s[i]=h^{2^{16s}i}\bmod p_2,\qquad s\in\\{0,1,2,3\\},\qquad 0\le i < 65536.$$

Any 64-bit $`t`$ has four base-$`2^{16}`$ digits, so the table evaluation needs four lookups and at most three modular multiplications. The tables occupy 1 MiB per GPU and are rebuilt for each digit interval. This avoids a full exponentiation by squaring for every primary hit.

Only a simultaneous zero modulo both primes, together with $`4\mid k`$, proves divisibility by the full $`c(16)`$. Host-side C++ checks repeat the two prime remainders for every saved hit; Python then independently checks the complete composite modulus.

## 4. Coverage and minimality

For digit $`d`$, the search uses

$$j_{\min}=\left\lceil\frac{\max(17,10^{d-1})}{4}\right\rceil,\qquad j_{\max}=\left\lfloor\frac{10^d-1}{4}\right\rfloor.$$

Each $`x`$ in $`[0,L)`$ is assigned to exactly one of eight shards, and all allowable $`t`$ values are tested. If $`j_{\max}+1 < L`$, the shard limit is clipped to $`j_{\max}+1`$ instead of scanning irrelevant orbits. A new digit begins only after every shard of the previous digit has reached its exact endpoint and passed the audit.

Selected production counts are:

| Digit interval | Primary hits | Full solutions |
|---|---:|---:|
| 22 | `1827789172` | 0 |
| 23 | `18277398093` | 10 |

The main campaign ran for **277.448 seconds**, including compilation/self-tests, device initialization, checkpoint resumes, and final auditing. It reused the already completed 2–6 digit smoke-test checkpoints; that initial end-to-end smoke test took another 69.346 seconds. The hardware was eight NVIDIA GPUs with CUDA 12.9. These are measured records, not a promise for other hardware or a from-scratch benchmark on an empty directory.

## 5. Numerical and independent verification

The CUDA self-test includes:

- 100,000 Montgomery products checked against host-side 128-bit arithmetic.
- 100,000 table-based exponentiations checked against ordinary modular exponentiation.
- Concatenation checks through 1000 and 100 larger inputs using two host formulas.

A separate forward comparison covered the first 1,000,000 orbit points of the 20-digit interval. GPU and Python both reported **114 primary hits**, and their full-candidate sets agreed exactly. See [formula20_test.audit.txt](results/formula20_test.audit.txt).

The independent verifier in [src/verify16.py](src/verify16.py) uses binary block sums. For a fixed-digit block with first term $`s`$, length $`n`$, and $`q=10^d`$, it evaluates

$$r_{\mathrm{end}}=r_{\mathrm{start}}q^n+(s-1)S_0(n)+S_1(n),$$

where

$$S_0(n)=\sum_{i=0}^{n-1}q^i,\qquad S_1(n)=\sum_{i=1}^{n}iq^{n-i}.$$

This method requires only additions and multiplications and works directly modulo the composite $`c(16)`$; it does not use the primary period-reduction formula. The minimum candidate was additionally checked with the matrix-power implementation in [a15/src/verify15_matrix.py](../a15/src/verify15_matrix.py), also giving remainder zero.

The final audit checks every shard configuration, endpoint, completion log, candidate digit range, $`k\bmod4`$, both prime remainders, and the full-modulus remainder. It also verifies that no earlier digit contains a saved solution. As with the other projects, reading the audit is not a fresh exhaustive search: minimality relies on the archived complete coverage as well as the independent candidate checks.

## 6. Audit, build, and reproduce

From the repository root:

```bash
# No GPU required: check the final answer and all completed coverage evidence
make -C a16 audit

# No GPU required: literal-concatenation and verifier self-checks
make -C a16 validate

# Build the CUDA program without starting a search
make -C a16 NVCC=/usr/local/cuda-12.9/bin/nvcc GPU_ARCH=native

# Numerical self-test on one GPU
make -C a16 selftest GPU_ID=0
```

The full campaign entry point is:

```bash
cd a16/src
python3 run16.py --through 28
```

It uses eight GPUs and resumes valid checkpoints. For an independent from-scratch run, copy the source into a new working directory with an empty `results/` directory rather than mixing new partial results with the completed evidence. The supported range is deliberately limited to digits 2–28; digit 29 requires a different period and is not needed for this answer.
