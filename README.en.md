# A400666: GPU Searches for a Concatenated-Integer Divisibility Problem

**[中文](README.md) | English**

**This entire project was completed autonomously by Kimi K3.1 Pro [thinkg: max] + Kimi-Code.**

This project documents the search, optimization, and independent verification of `a(14)`, `a(15)`, `a(16)`, and `a(17)` in [OEIS A400666](https://oeis.org/A400666). Define

$$c(n)=\mathrm{concat}(1,2,\ldots,n),\qquad a(n)=\min\\{k > n:c(n)\mid c(k)\\}.$$

For example, $`c(14)=1234567891011121314`$. The programs never materialize the enormous integer $`c(k)`$; they work modulo $`c(n)`$ or its large prime factor.

## Results

| Term | Value | Search conclusion |
|---|---:|---|
| $`a(14)`$ | `675924569493048966` | No solution with at most 17 digits; exactly one solution in the complete 18-digit interval |
| $`a(15)`$ | `70685893246361615705` | No solution with at most 19 digits; three solutions with 20 digits, of which this is the smallest |
| $`a(16)`$ | `10152437877876943330588` | No solution with at most 22 digits; ten solutions with 23 digits, of which this is the smallest |
| $`a(17)`$ | `553943104744829340555260` | No solution with at most 23 digits; exactly one solution in the complete 24-digit interval |

The other two 20-digit solutions for $`a(15)`$ are `73394079098572402415` and `74341537919751115530`. The ten 23-digit solutions for $`a(16)`$ are listed in [a16/results/d23.summary](a16/results/d23.summary). The orbit coordinates and complete coverage evidence for $`a(17)`$ are in [a17/results/](a17/results/). All four answers have remainder zero modulo the full composite modulus $`c(n)`$, not merely modulo their large prime factors.

The first 13 published values have also been checked for divisibility and minimality. See [a14/oeis_results/verification.txt](a14/oeis_results/verification.txt).

## Hardware and elapsed time

- GPU: eight NVIDIA GPUs, auto-detected architecture.
- CUDA: the production environment used CUDA 12.9, with the compiler at `/usr/local/cuda-12.9/bin/nvcc`.
- CPU: 288 hardware threads in total; the AVX-512 searches for $`a(15)`$ and $`a(17)`$ each used 272 worker threads.
- $`a(14)`$: approximately 80 minutes for the full GPU search.
- $`a(15)`$: approximately 69 hours, including algorithm development, experiments, searches, CPU/GPU handoffs, and the final audit; this was not one unchanged batch job.
- $`a(16)`$: 277.448 seconds for the main campaign, including resumes, initialization, and final audit. The earlier 2–6 digit smoke test took another 69.346 seconds.
- $`a(17)`$: approximately 79.53 hours of total exploration, including algorithm development, optimization, checkpoint resumes, low-digit CPU recovery, and the final audit; this was not one uninterrupted batch job.

This public release omits specific GPU and CPU models. Historical build metadata is de-identified, and architecture-specific disassembly files are excluded.

These are records from this machine, not performance promises for other hardware. Other NVIDIA architectures can be tried by overriding `GPU_ARCH`, but the throughput figures do not transfer automatically. The production CPU programs require AVX-512 support in the compilation target.

## Layout

```text
A400666/
├── README.md               # Default Chinese overview
├── README.en.md            # English overview
├── LICENSE
├── SHA256SUMS
├── a14/
│   ├── README.md             # Default Chinese version
│   ├── README.en.md          # English version
│   ├── Makefile
│   ├── src/                  # CUDA search, scheduling, and independent verification
│   ├── results/              # Logs, checkpoints, and candidates for 36 shards
│   └── oeis_results/         # Exhaustive verification evidence for the first 13 terms
├── a15/
│   ├── README.md             # Default Chinese version
│   ├── README.en.md          # English version
│   ├── EXPERIMENTS.md        # Measured speedups and rejected ideas (Chinese)
│   ├── Makefile
│   ├── requirements.txt      # Python dependencies for audit and validation
│   ├── src/                  # Production CUDA/C++/Python source
│   ├── experiments/          # Control and rejected experiment source
│   ├── results15/            # Active logs, checkpoints, migration manifests, and final audit
│   └── cpu_results/          # Evidence for 272-thread search and CPU/GPU handoff
├── a16/
│   ├── README.md             # Default Chinese version
│   ├── README.en.md          # English version
│   ├── Makefile
│   ├── src/                  # CUDA search, campaign manager, and independent verifier
│   ├── results/              # Logs, checkpoints, candidates, and final audit
│   └── campaign16.log        # Main campaign record
└── a17/
    ├── README.md             # Default Chinese: small-factor filters, periodic reduction, and eight GPUs
    ├── README.en.md          # English version
    ├── Makefile
    ├── requirements.txt      # Python dependency for regenerating filters
    ├── src/                  # CUDA/CPU search, scheduling, snapshots, and independent verification
    ├── filters/              # Small-factor residue classes and generation audits
    ├── experiments/          # Control and rejected experiment source
    ├── results/              # Complete checkpoints, logs, candidates, optimization records, and final audit
    └── campaign17_main.log   # Main 24-digit campaign record
```

To keep the production scripts unchanged, `src/results`, `src/oeis_results`, `src/results15`, and `src/cpu_results` are relative symbolic links to their corresponding evidence directories; `a17` also uses `src/filters` for its filter data. In `a15/experiments/`, `search15_prune.cu` and `search15_period_fast.cu` are also relative links to the production sources, avoiding two inconsistent copies of shared implementations. When running the source directly, first enter the appropriate `src/` directory. From the repository root, prefer the `make -C` commands below.

## Audit the archived results (no GPU required)

Python 3 is required. Some formula checks and the final audit for $`a(15)`$ import SymPy; the environment used here had version 1.14.0. The a16 and a17 audits use only the Python standard library; a17 needs SymPy only when regenerating its small-factor filters.

```bash
python3 -m pip install -r a15/requirements.txt
make -C a14 audit
make -C a14 verify-first13
make -C a15 audit
make -C a16 audit
make -C a17 audit
```

The a15 audit should print:

```text
a(15)=70685893246361615705
block_sums_remainder=0
matrix_remainder=0
...
FINAL_INDEPENDENT_AUDIT_OK
```

The a16 audit should additionally print `a(16)=10152437877876943330588` and `FINAL_A16_AUDIT_OK`. The a17 audit should print `a(17)=553943104744829340555260`, `FINAL_A17_AUDIT_OK`, and `INDEPENDENT_MATRIX17_VERIFIED`.

Each audit recomputes the concatenation remainder of every saved candidate, checks shard configurations, endpoints, completion logs, and any CPU/GPU boundaries, and verifies that no smaller saved solution exists. It does **not** rescan the entire search space from scratch in a few minutes; minimality relies on the archived coverage evidence and the search algorithm itself. The independent remainder checkers use block sums and matrices rather than the period-reduction formulas in the CUDA search.

To check for corruption during file transfer:

```bash
sha256sum -c SHA256SUMS
```

The manifest covers regular files; symbolic links are retained as links by the version-control system.

## Build and run lightweight self-tests

```bash
# If nvcc is not in PATH:
make -C a14 NVCC=/usr/local/cuda-12.9/bin/nvcc
make -C a15 NVCC=/usr/local/cuda-12.9/bin/nvcc
make -C a16 NVCC=/usr/local/cuda-12.9/bin/nvcc
make -C a17 NVCC=/usr/local/cuda-12.9/bin/nvcc

# GPU 0 is the default; these commands do not start a full search
make -C a14 selftest GPU_ID=0
make -C a15 selftest GPU_ID=0
make -C a16 selftest GPU_ID=0
make -C a17 selftest GPU_ID=0
```

You can also run only `make -C a15 audit`, `make -C a15 selftest-cpu`, or `make -C a15 selftest-gpu`. `make all` compiles the programs without starting a search. The a15 CPU self-test invokes the shared wrapper, which runs the same AVX-512 numerical checks and removes its intentional `k=15` test hit afterward. The a17 self-test covers both the CUDA numerical path and the CPU filter.

## Reproduce from scratch

Read [a14/README.en.md](a14/README.en.md), [a15/README.en.md](a15/README.en.md), [a16/README.en.md](a16/README.en.md), and [a17/README.en.md](a17/README.en.md) first. A full run involves up to several days of computation, eight GPUs, 272 CPU workers, and several concurrent phases. A successful build or self-test is not a full reproduction.

This directory contains completed checkpoints, so the scripts resume from them. For an independent run, copy the project and use fresh empty result directories rather than overwriting the archived evidence. For $`a(15)`$, `campaign15.py` is a **continuation coordinator**: before starting it, you must start the low-digit CPU search and the initial 20-digit GPU search as documented. It is not a one-command launcher for every task from scratch. For $`a(17)`$, `make -C a17 run` covers the 2–20-digit CPU filter, the 21–24-digit GPU search, and the final audit in order.

## Source scope and license

The core CUDA/C++ programs contain constants for the fixed moduli and prime factors used for each $`n`$; they are not general solvers for arbitrary $`n`$. The Python block-sum and matrix remainder checkers can be used with a supplied $`k`$ and modulus. Historical experiment sources are kept separately in `a15/experiments/` and `a17/experiments/` and should not be mistaken for the final production path.

Prebuilt local CUDA/CPU executables are not included. Production binary SHA-256 values in the documentation identify the binaries used in the historical logs. Different compilers, drivers, or architectures can produce different hashes without changing the mathematical result.

The project is distributed under the [MIT License](LICENSE). This curated release omits historical snapshots but retains the states, manifests, candidates, and complete logs needed by the current audit.
