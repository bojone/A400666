#include <omp.h>
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using u64 = unsigned long long;
using u128 = unsigned __int128;

static constexpr u64 P = 8230452606740808761ULL;
static constexpr u64 PINV = 12692191010489013751ULL;
static constexpr int THREADS = 272;
static constexpr u64 CHUNK = 1000000;

struct Job {
    int d, a;
    u64 z0, g, w0, ws, r2, begin, end;
};

static inline u64 addmod(u64 a, u64 b) {
    u64 r = a + b;
    if (r >= P) r -= P;
    return r;
}

static inline u64 submod(u64 a, u64 b) {
    return a >= b ? a - b : addmod(a, P - b);
}

static inline u64 mulmod(u64 a, u64 b) {
    return (u64)((u128)a * b % P);
}

static inline u64 mulhi(u64 a, u64 b) {
    return (u64)(((u128)a * b) >> 64);
}

static inline u64 mont(u64 a, u64 b) {
    u64 lo = a * b;
    u64 hi = mulhi(a, b);
    u64 m = lo * PINV;
    u64 mhi = mulhi(m, P);
    u64 r = hi + mhi + (lo != 0);
    if (r >= P) r -= P;
    return r;
}

static u64 powmod(u64 a, u64 e) {
    u64 r = 1;
    while (e) {
        if (e & 1) r = mulmod(r, a);
        a = mulmod(a, a);
        e >>= 1;
    }
    return r;
}

static u64 invmod(u64 a) {
    return powmod(a, P - 2);
}

static u64 mont_pow(u64 a, u64 e, u64 r2) {
    u64 result = mont(1, r2);
    u64 base = mont(a, r2);
    while (e) {
        if (e & 1) result = mont(result, base);
        e >>= 1;
        if (e) base = mont(base, base);
    }
    return mont(result, 1);
}

static u64 concat_mod(u64 k) {
    u64 r = 0, start = 1;
    for (u64 d = 1; d <= 19 && start <= k; ++d) {
        u64 end = std::min(k, start * 10 - 1);
        u64 q = powmod(10, d);
        u64 A = invmod(submod(1, q));
        u64 B = submod(0, mulmod(q, mulmod(A, A)));
        u64 prev = start - 1;
        u64 U = mulmod(submod(r, addmod(mulmod(A, prev % P), B)), powmod(invmod(q), prev));
        r = addmod(mulmod(U, powmod(q, end)), addmod(mulmod(A, end % P), B));
        if (end == k) break;
        start *= 10;
    }
    return r;
}

static u64 concat_mod_sum(u64 k) {
    u64 r = 0, start = 1;
    for (u64 d = 1; d <= 19 && start <= k; ++d) {
        u64 end = std::min(k, start * 10 - 1);
        u64 len = end - start + 1;
        u64 q = powmod(10, d);
        u64 qt = powmod(q, len);
        u64 den = submod(q, 1);
        u64 S0 = mulmod(submod(qt, 1), invmod(den));
        u64 lm = len % P;
        u64 numerator = addmod(submod(powmod(q, len + 1), mulmod(lm, q)), lm);
        numerator = submod(numerator, q);
        u64 S1 = mulmod(numerator, invmod(mulmod(den, den)));
        r = addmod(mulmod(r, qt), addmod(mulmod((start - 1) % P, S0), S1));
        if (end == k) break;
        start *= 10;
    }
    return r;
}

static u64 pow10u(int d) {
    u64 x = 1;
    for (int i = 0; i < d; ++i) x *= 10;
    return x;
}

static Job make_job(int d, int a) {
    u64 start = std::max<u64>(16, pow10u(d - 1));
    u64 q = powmod(10, d);
    u64 A = invmod(submod(1, q));
    u64 B = submod(0, mulmod(q, mulmod(A, A)));
    u64 prev = start - 1;
    u64 rprev = concat_mod(prev);
    u64 U = mulmod(submod(rprev, addmod(mulmod(A, prev % P), B)), powmod(invmod(q), prev));
    Job job{};
    job.d = d; job.a = a;
    job.z0 = mulmod(U, powmod(q, a));
    job.g = powmod(q, 15);
    job.w0 = addmod(mulmod(A, a), B);
    job.ws = mulmod(A, 15);
    u64 R = (u64)(((u128)1 << 64) % P);
    job.r2 = mulmod(R, R);
    job.begin = start <= (u64)a ? 0 : (start - a + 14) / 15;
    u64 hi = start == 16 ? 99 : pow10u(d) - 1;
    job.end = (hi - a) / 15 + 1;
    return job;
}

static std::string job_name(int d, int a) {
    return std::string("d") + (d < 10 ? "0" : "") + std::to_string(d) + "_a" + std::to_string(a);
}

static std::string config_key(const Job& job) {
    return std::to_string(job.d) + ":" + std::to_string(job.a) + ":" +
        std::to_string(job.begin) + ":" + std::to_string(job.end) + ":" + std::to_string(THREADS);
}

static void save_state(const Job& job, const std::vector<std::atomic<u64>>& next) {
    std::string name = job_name(job.d, job.a);
    std::string tmp = "cpu_results/" + name + ".tmp";
    std::ofstream f(tmp);
    f << "config=" << config_key(job) << "\n";
    for (int i = 0; i < THREADS; ++i) f << i << "=" << next[i].load(std::memory_order_relaxed) << "\n";
    f.close();
    std::filesystem::rename(tmp, "cpu_results/" + name + ".state");
}

static void load_state(const Job& job, std::vector<std::atomic<u64>>& next, std::vector<u64>& ends) {
    std::ifstream f("cpu_results/" + job_name(job.d, job.a) + ".state");
    if (!f) {
        for (int i = 0; i < THREADS; ++i) {
            next[i] = job.begin + (job.end - job.begin) * i / THREADS;
            ends[i] = job.begin + (job.end - job.begin) * (i + 1) / THREADS;
        }
        return;
    }
    std::string line;
    if (!std::getline(f, line) || line != "config=" + config_key(job)) throw std::runtime_error("state config mismatch");
    int i = 0;
    while (std::getline(f, line) && i < THREADS) {
        size_t pos = line.find('=');
        next[i] = std::stoull(line.substr(pos + 1));
        ends[i] = job.begin + (job.end - job.begin) * (i + 1) / THREADS;
        ++i;
    }
    if (i != THREADS) throw std::runtime_error("incomplete state");
}

static inline __m512i mulfull_vec(__m512i a, __m512i b, __m512i& lo) {
    const __m512i mask = _mm512_set1_epi64(0xffffffff);
    __m512i a0 = _mm512_and_si512(a, mask), a1 = _mm512_srli_epi64(a, 32);
    __m512i b0 = _mm512_and_si512(b, mask), b1 = _mm512_srli_epi64(b, 32);
    __m512i p00 = _mm512_mul_epu32(a0, b0);
    __m512i p01 = _mm512_mul_epu32(a0, b1);
    __m512i p10 = _mm512_mul_epu32(a1, b0);
    __m512i p11 = _mm512_mul_epu32(a1, b1);
    __m512i s1 = _mm512_slli_epi64(p01, 32), s2 = _mm512_slli_epi64(p10, 32);
    __m512i s = _mm512_add_epi64(s1, s2);
    __mmask8 c1 = _mm512_cmp_epu64_mask(s, s1, _MM_CMPINT_LT);
    lo = _mm512_add_epi64(p00, s);
    __mmask8 c2 = _mm512_cmp_epu64_mask(lo, p00, _MM_CMPINT_LT);
    __m512i hi = _mm512_add_epi64(p11, _mm512_add_epi64(_mm512_srli_epi64(p01, 32), _mm512_srli_epi64(p10, 32)));
    hi = _mm512_mask_add_epi64(hi, c1, hi, _mm512_set1_epi64(1));
    return _mm512_mask_add_epi64(hi, c2, hi, _mm512_set1_epi64(1));
}

static inline __m512i mont_vec(__m512i a, __m512i b) {
    __m512i lo, hi = mulfull_vec(a, b, lo);
    __m512i m = _mm512_mullo_epi64(lo, _mm512_set1_epi64(PINV));
    __m512i unused, mhi = mulfull_vec(m, _mm512_set1_epi64(P), unused);
    __m512i r = _mm512_add_epi64(hi, mhi);
    __mmask8 nz = _mm512_cmp_epu64_mask(lo, _mm512_setzero_si512(), _MM_CMPINT_NE);
    r = _mm512_mask_add_epi64(r, nz, r, _mm512_set1_epi64(1));
    __mmask8 ge = _mm512_cmp_epu64_mask(r, _mm512_set1_epi64(P), _MM_CMPINT_GE);
    return _mm512_mask_sub_epi64(r, ge, r, _mm512_set1_epi64(P));
}

static inline __m512i addmod_vec(__m512i a, __m512i b) {
    __m512i r = _mm512_add_epi64(a, b);
    __mmask8 ge = _mm512_cmp_epu64_mask(r, _mm512_set1_epi64(P), _MM_CMPINT_GE);
    return _mm512_mask_sub_epi64(r, ge, r, _mm512_set1_epi64(P));
}

static void record_hit(const Job& job, u64 k, u64& local_hits) {
    if (concat_mod(k) != 0 || concat_mod_sum(k) != 0) return;
    ++local_hits;
    #pragma omp critical
    {
        std::ofstream f("cpu_results/" + job_name(job.d, job.a) + ".candidates", std::ios::app);
        f << "k=" << k << " remainder_p=0 independent_remainder_p=0\n";
        f.flush();
        std::cout << "HIT d=" << job.d << " a=" << job.a << " k=" << k << "\n";
    }
}

static u64 scan_chunk_scalar(const Job& job, u64 begin, u64 end, u64& local_hits) {
    u64 gx = mont_pow(job.g, begin, job.r2);
    u64 z = mont(job.z0, mont(gx, job.r2));
    u64 w = addmod(mont(job.ws, mont(begin % P, job.r2)), job.w0);
    u64 gm = mont(job.g, job.r2);
    for (u64 j = begin; j < end; ++j) {
        u64 r = addmod(z, w);
        if (r == 0) record_hit(job, job.a + 15 * j, local_hits);
        z = mont(z, gm);
        w = addmod(w, job.ws);
    }
    return end;
}

static u64 scan_chunk(const Job& job, u64 begin, u64 end, u64& local_hits) {
    u64 j = begin;
    if (end - begin < 8) return scan_chunk_scalar(job, begin, end, local_hits);
    alignas(64) u64 za[8], wa[8];
    for (int lane = 0; lane < 8; ++lane) {
        u64 x = begin + lane;
        u64 gx = mont_pow(job.g, x, job.r2);
        za[lane] = mont(job.z0, mont(gx, job.r2));
        wa[lane] = addmod(mont(job.ws, mont(x % P, job.r2)), job.w0);
    }
    __m512i z = _mm512_load_si512(za), w = _mm512_load_si512(wa);
    __m512i gm = _mm512_set1_epi64(mont(job.g, job.r2));
    __m512i ws = _mm512_set1_epi64(job.ws);
    for (; j + 8 <= end; j += 8) {
        __m512i r = addmod_vec(z, w);
        __mmask8 hit = _mm512_cmpeq_epi64_mask(r, _mm512_setzero_si512());
        while (hit) {
            int lane = __builtin_ctz(hit);
            record_hit(job, job.a + 15 * (j + lane), local_hits);
            hit &= hit - 1;
        }
        z = mont_vec(z, gm);
        w = addmod_vec(w, ws);
    }
    if (j < end) scan_chunk_scalar(job, j, end, local_hits);
    return end;
}

static int run_job(int d, int a) {
    Job job = make_job(d, a);
    std::filesystem::create_directories("cpu_results");
    std::vector<std::atomic<u64>> next(THREADS);
    std::vector<u64> ends(THREADS);
    load_state(job, next, ends);
    std::atomic<u64> hits{0};
    std::atomic<bool> stop{false};
    auto started = std::chrono::steady_clock::now();
    #pragma omp parallel num_threads(THREADS)
    {
        int tid = omp_get_thread_num();
        auto last_save = started;
        while (!stop.load(std::memory_order_relaxed)) {
            u64 begin = next[tid].load(std::memory_order_relaxed);
            if (begin >= ends[tid]) break;
            u64 end = std::min(begin + CHUNK, ends[tid]);
            u64 local_hits = 0;
            u64 done = scan_chunk(job, begin, end, local_hits);
            next[tid].store(done, std::memory_order_relaxed);
            hits += local_hits;
            if (tid == 0) {
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - last_save).count() >= 30) {
                    save_state(job, next);
                    std::cout << "PROGRESS d=" << d << " a=" << a << " next0=" << done << "/" << ends[0]
                              << " hits=" << hits.load() << "\n";
                    last_save = now;
                }
            }
        }
    }
    save_state(job, next);
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    u64 remaining = 0;
    for (int i = 0; i < THREADS; ++i) remaining += ends[i] - next[i].load();
    std::cout << "JOB_DONE d=" << d << " a=" << a << " remaining=" << remaining
              << " hits=" << hits.load() << " elapsed=" << elapsed << "\n";
    return 0;
}

static int bench(int d, int a, u64 count) {
    Job job = make_job(d, a);
    job.end = std::min(job.end, job.begin + count);
    auto started = std::chrono::steady_clock::now();
    std::atomic<u64> hits{0};
    #pragma omp parallel num_threads(THREADS)
    {
        int tid = omp_get_thread_num();
        u64 begin = job.begin + (job.end - job.begin) * tid / THREADS;
        u64 end = job.begin + (job.end - job.begin) * (tid + 1) / THREADS;
        u64 local_hits = 0;
        while (begin < end) {
            u64 chunk_end = std::min(begin + CHUNK, end);
            begin = scan_chunk(job, begin, chunk_end, local_hits);
        }
        hits += local_hits;
    }
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << "BENCH d=" << d << " a=" << a << " points=" << job.end - job.begin
              << " elapsed=" << elapsed << " rate=" << (job.end - job.begin) / elapsed
              << " hits=" << hits.load() << "\n";
    return 0;
}

static bool selftest() {
    std::mt19937_64 rng(40066615);
    u64 R = (u64)(((u128)1 << 64) % P);
    u64 rinv = invmod(R);
    for (int i = 0; i < 1000000; ++i) {
        u64 a = rng() % P, b = rng() % P;
        if (mont(a, b) != mulmod(mulmod(a, b), rinv)) return false;
    }
    for (int batch = 0; batch < 10000; ++batch) {
        alignas(64) u64 a[8], b[8], out[8];
        for (int lane = 0; lane < 8; ++lane) { a[lane] = rng() % P; b[lane] = rng() % P; }
        __m512i r = mont_vec(_mm512_load_si512(a), _mm512_load_si512(b));
        _mm512_store_si512(out, r);
        for (int lane = 0; lane < 8; ++lane) if (out[lane] != mont(a[lane], b[lane])) return false;
    }
    for (u64 k = 1; k <= 1000; ++k) {
        if (concat_mod(k) != concat_mod_sum(k)) return false;
    }
    for (int i = 0; i < 100; ++i) {
        u64 k = rng() % 1000000000000000000ULL;
        if (concat_mod(k) != concat_mod_sum(k)) return false;
    }
    Job job = make_job(2, 0);
    u64 hits = 0;
    scan_chunk(job, 1, 2, hits);
    if (hits != 1) {
        std::cerr << "known c(15) scan check failed\n";
        return false;
    }
    std::cout << "selftest ok: 1000000 Montgomery products, concatenation formulas, known k=15 scan\n";
    return true;
}

int main(int argc, char** argv) {
    try {
        std::filesystem::create_directories("cpu_results");
        if (argc == 2 && std::string(argv[1]) == "selftest") return selftest() ? 0 : 1;
        if (argc == 4 && std::string(argv[1]) == "run") return run_job(std::stoi(argv[2]), std::stoi(argv[3]));
        if (argc == 5 && std::string(argv[1]) == "bench") return bench(std::stoi(argv[2]), std::stoi(argv[3]), std::stoull(argv[4]));
        if (argc == 2 && std::string(argv[1]) == "all") {
            for (int d = 2; d <= 17; ++d) {
                run_job(d, 0);
                run_job(d, 5);
            }
            return 0;
        }
        std::cerr << "usage: cpu15 selftest | bench d a count | run d a | all\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
