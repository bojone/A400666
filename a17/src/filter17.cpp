#include <omp.h>
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <vector>

using u64 = unsigned long long;
using u128 = unsigned __int128;

static constexpr u64 P = 584538396786764503ULL;
static constexpr u64 PINV = 17458093328265287449ULL;
static constexpr int THREADS = 272;

struct Residue {
    u128 k0;
    u64 gap, count;
};

struct Job {
    int d;
    u64 period, q, A, B, U, gm, ws;
};

static inline u64 addmod(u64 a, u64 b, u64 p) { return (u64)(((u128)a + b) % p); }
static inline u64 submod(u64 a, u64 b, u64 p) { return a >= b ? a - b : p - (b - a); }
static inline u64 mulmod(u64 a, u64 b, u64 p) { return (u64)((u128)a * b % p); }
static inline u64 mulhi(u64 a, u64 b) { return (u64)(((u128)a * b) >> 64); }

static u64 powmod(u64 a, u128 exponent, u64 p) {
    u64 result = 1;
    while (exponent) {
        if (exponent & 1) result = mulmod(result, a, p);
        a = mulmod(a, a, p);
        exponent >>= 1;
    }
    return result;
}

static u64 invmod(u64 a, u64 p) { return powmod(a, p - 2, p); }

static u64 concat_mod(u128 k, u64 p) {
    u64 remainder = 0;
    u128 start = 1;
    while (start <= k) {
        u128 end = std::min(k, start * 10 - 1);
        int digits = 1;
        for (u128 x = start; x >= 10; x /= 10) ++digits;
        u64 q = powmod(10, digits, p);
        u64 A = invmod(submod(1, q, p), p);
        u64 B = submod(0, mulmod(q, mulmod(A, A, p), p), p);
        u128 previous = start - 1;
        u64 U = mulmod(submod(remainder, addmod(mulmod(A, previous % p, p), B, p), p),
                       powmod(invmod(q, p), previous, p), p);
        remainder = addmod(mulmod(U, powmod(q, end, p), p), addmod(mulmod(A, end % p, p), B, p), p);
        if (end == k) break;
        start *= 10;
    }
    return remainder;
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
    __m512i reduced = _mm512_sub_epi64(r, _mm512_set1_epi64(P));
    __mmask8 negative = _mm512_cmp_epi64_mask(reduced, _mm512_setzero_si512(), _MM_CMPINT_LT);
    return _mm512_mask_sub_epi64(r, negative, r, reduced);
}

static u64 mont(u64 a, u64 b) {
    u64 lo = a * b, hi = mulhi(a, b);
    u64 m = lo * PINV, mhi = mulhi(m, P);
    u64 r = hi + mhi + (lo != 0);
    if (r >= P) r -= P;
    return r;
}

static std::string u128_string(u128 value) {
    if (!value) return "0";
    std::string result;
    while (value) {
        result.push_back(char('0' + value % 10));
        value /= 10;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

static u128 parse_u128(const std::string& text) {
    u128 result = 0;
    for (char c : text) {
        if (c >= '0' && c <= '9') result = result * 10 + (c - '0');
    }
    return result;
}

static std::vector<Residue> load_residues(int d, u64& period) {
    std::ifstream input("filters/d" + std::string(d < 10 ? "0" : "") + std::to_string(d) + ".res");
    if (!input) throw std::runtime_error("missing residue file");
    std::string line;
    std::getline(input, line);
    period = std::stoull(line.substr(line.find('=') + 1));
    std::getline(input, line);
    u64 expected = std::stoull(line.substr(line.find('=') + 1));
    std::vector<Residue> rows;
    u128 hi = 1;
    for (int i = 0; i < d; ++i) hi *= 10;
    --hi;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        size_t space = line.find(' ');
        Residue row{parse_u128(line.substr(0, space)), std::stoull(line.substr(space + 1)), 0};
        if (row.k0 <= hi) row.count = (u64)(1 + (hi - row.k0) / period);
        rows.push_back(row);
    }
    if (rows.size() != expected) throw std::runtime_error("residue count mismatch");
    return rows;
}

static Job make_job(int d, u64 period) {
    u128 start = 1;
    for (int i = 1; i < d; ++i) start *= 10;
    u64 q = powmod(10, d, P);
    u64 A = invmod(submod(1, q, P), P);
    u64 B = submod(0, mulmod(q, mulmod(A, A, P), P), P);
    u64 U = mulmod(submod(concat_mod(start - 1, P),
                          addmod(mulmod(A, (start - 1) % P, P), B, P), P),
                   powmod(invmod(q, P), start - 1, P), P);
    u64 R = (u64)(((u128)1 << 64) % P);
    return Job{d, period, q, A, B, U, mulmod(powmod(q, period, P), R, P), mulmod(A, period % P, P)};
}

static std::vector<u128> scan_group(const Job& job, const std::vector<Residue>& rows, int group, u64 step_limit) {
    alignas(64) u64 za[8], wa[8], counts[8];
    u128 k0[8];
    u64 maximum = 0;
    for (int lane = 0; lane < 8; ++lane) {
        int index = 8 * group + lane;
        counts[lane] = index < (int)rows.size() ? std::min(rows[index].count, step_limit) : 0;
        k0[lane] = index < (int)rows.size() ? rows[index].k0 : 0;
        maximum = std::max(maximum, counts[lane]);
        za[lane] = wa[lane] = 0;
        if (counts[lane]) {
            za[lane] = mulmod(job.U, powmod(job.q, k0[lane], P), P);
            wa[lane] = addmod(mulmod(job.A, k0[lane] % P, P), job.B, P);
        }
    }
    __m512i z = _mm512_load_si512(za), w = _mm512_load_si512(wa);
    __m512i gm = _mm512_set1_epi64(job.gm), ws = _mm512_set1_epi64(job.ws);
    std::vector<u128> hits;
    for (u64 step = 0; step < maximum; ++step) {
        __mmask8 active = 0;
        for (int lane = 0; lane < 8; ++lane) if (step < counts[lane]) active |= 1 << lane;
        __m512i remainder = addmod_vec(z, w);
        __mmask8 hit = _mm512_cmpeq_epi64_mask(remainder, _mm512_setzero_si512()) & active;
        while (hit) {
            int lane = __builtin_ctz(hit);
            hits.push_back(k0[lane] + (u128)step * job.period);
            hit &= hit - 1;
        }
        z = mont_vec(z, gm);
        w = addmod_vec(w, ws);
    }
    return hits;
}

struct State {
    std::vector<unsigned char> done;
    u64 searched = 0, hits = 0;
};

static std::string state_name(int d) {
    return "results/filter_d" + std::string(d < 10 ? "0" : "") + std::to_string(d) + ".state";
}

static std::string candidate_name(int d) {
    return "results/filter_d" + std::string(d < 10 ? "0" : "") + std::to_string(d) + ".candidates";
}

static std::string hex_done(const State& state) {
    static const char digits[] = "0123456789abcdef";
    std::string result;
    for (unsigned char byte : state.done) {
        result.push_back(digits[byte >> 4]);
        result.push_back(digits[byte & 15]);
    }
    return result;
}

static u64 done_groups(const State& state) {
    u64 result = 0;
    for (unsigned char byte : state.done) result += __builtin_popcount((unsigned int)byte);
    return result;
}

static void save_state(int d, u64 period, int groups, const State& state) {
    std::string temporary = state_name(d) + ".tmp";
    std::ofstream output(temporary);
    output << "config=" << d << ":" << period << ":" << groups << "\n"
           << "done=" << done_groups(state)
           << "\nsearched=" << state.searched << "\nhits=" << state.hits
           << "\ndone_hex=" << hex_done(state) << "\n";
    output.close();
    if (!output) throw std::runtime_error("failed to save filter state");
    std::filesystem::rename(temporary, state_name(d));
}

static State load_state(int d, u64 period, int groups, const std::vector<Residue>& rows) {
    State state;
    state.done.assign((groups + 7) / 8, 0);
    std::ifstream input(state_name(d));
    if (!input) return state;
    std::string line, config, hex;
    u64 done_count = 0;
    while (std::getline(input, line)) {
        if (line.rfind("config=", 0) == 0) config = line.substr(7);
        else if (line.rfind("done=", 0) == 0) done_count = std::stoull(line.substr(5));
        else if (line.rfind("searched=", 0) == 0) state.searched = std::stoull(line.substr(9));
        else if (line.rfind("hits=", 0) == 0) state.hits = std::stoull(line.substr(5));
        else if (line.rfind("done_hex=", 0) == 0) hex = line.substr(9);
    }
    if (config != std::to_string(d) + ":" + std::to_string(period) + ":" + std::to_string(groups) ||
        hex.size() != state.done.size() * 2) {
        throw std::runtime_error("filter state config mismatch");
    }
    for (size_t i = 0; i < state.done.size(); ++i) {
        state.done[i] = std::stoi(hex.substr(2 * i, 2), nullptr, 16);
    }
    u64 expected_searched = 0, actual_done = 0;
    for (int group = 0; group < groups; ++group) {
        bool complete = state.done[group / 8] & (1 << (group % 8));
        if (complete) {
            ++actual_done;
            for (int lane = 0; lane < 8 && 8 * group + lane < (int)rows.size(); ++lane) {
                expected_searched += rows[8 * group + lane].count;
            }
        }
    }
    if (actual_done != done_count || expected_searched != state.searched) {
        throw std::runtime_error("filter checkpoint coverage mismatch");
    }
    return state;
}

static bool selftest() {
    std::mt19937_64 rng(40066617);
    u64 R = (u64)(((u128)1 << 64) % P), rinv = invmod(R, P);
    for (int i = 0; i < 100000; ++i) {
        u64 a = rng() % P, b = rng() % P;
        if (mont(a, b) != mulmod(mulmod(a, b, P), rinv, P)) return false;
    }
    for (int batch = 0; batch < 10000; ++batch) {
        alignas(64) u64 a[8], b[8], out[8];
        for (int lane = 0; lane < 8; ++lane) { a[lane] = rng() % P; b[lane] = rng() % P; }
        __m512i r = mont_vec(_mm512_load_si512(a), _mm512_load_si512(b));
        _mm512_store_si512(out, r);
        for (int lane = 0; lane < 8; ++lane) if (out[lane] != mont(a[lane], b[lane])) return false;
    }
    u64 period;
    std::vector<Residue> rows = load_residues(18, period);
    Job job = make_job(18, period);
    std::vector<u128> actual = scan_group(job, rows, 0, 64);
    std::vector<u128> expected;
    for (int lane = 0; lane < 8; ++lane) {
        for (u64 step = 0; step < std::min<u64>(rows[lane].count, 64); ++step) {
            u128 k = rows[lane].k0 + (u128)step * period;
            if (concat_mod(k, P) == 0) expected.push_back(k);
        }
    }
    if (actual != expected) {
        std::cerr << "filter trace mismatch\n";
        return false;
    }
    std::cout << "FILTER_SELFTEST_OK scalar=100000 vector_batches=10000 trace_points=512\n";
    return true;
}

static int run(int d) {
    u64 period;
    std::vector<Residue> rows = load_residues(d, period);
    Job job = make_job(d, period);
    int groups = (rows.size() + 7) / 8;
    State state = load_state(d, period, groups, rows);
    std::atomic<int> next_group{0};
    std::mutex mutex;
    std::ofstream candidates(candidate_name(d), std::ios::app);
    auto started = std::chrono::steady_clock::now();
    auto last_save = started;
    #pragma omp parallel num_threads(THREADS)
    {
        while (true) {
            int group = next_group.fetch_add(1);
            if (group >= groups) break;
            if (state.done[group / 8] & (1 << (group % 8))) continue;
            std::vector<u128> hits = scan_group(job, rows, group, ~0ULL);
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (state.done[group / 8] & (1 << (group % 8))) continue;
                for (u128 k : hits) {
                    candidates << "k=" << u128_string(k) << " d=" << d << "\n";
                    ++state.hits;
                }
                candidates.flush();
                for (int lane = 0; lane < 8 && 8 * group + lane < (int)rows.size(); ++lane) {
                    state.searched += rows[8 * group + lane].count;
                }
                state.done[group / 8] |= 1 << (group % 8);
                if (std::chrono::steady_clock::now() - last_save >= std::chrono::seconds(30)) {
                    save_state(d, period, groups, state);
                    last_save = std::chrono::steady_clock::now();
                }
            }
        }
    }
    save_state(d, period, groups, state);
    candidates.flush();
    std::cout << "DONE_FILTER d=" << d << " searched=" << state.searched << " hits=" << state.hits << "\n";
    return 0;
}

int main(int argc, char** argv) {
    try {
        std::filesystem::create_directories("results");
        if (argc == 2 && std::string(argv[1]) == "selftest") return selftest() ? 0 : 1;
        if (argc == 3 && std::string(argv[1]) == "run") return run(std::stoi(argv[2]));
        std::cerr << "usage: filter17 selftest | run d\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
}
