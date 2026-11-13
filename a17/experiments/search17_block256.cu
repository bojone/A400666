#include <cuda_runtime.h>
#include <algorithm>
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

static constexpr u64 P = 584538396786764503ULL;
static constexpr u64 R17 = 97423066131127417ULL;
static constexpr u64 PINV = 17458093328265287449ULL;
static constexpr int BLOCK_SIZE = 256;
static constexpr int MAX_HITS = 65536;

struct Wide { u64 hi, lo; };
struct Hit { u64 x, t; };

struct Job {
    int d, a;
    u64 z0, g, ws, w0, gm, r2, L, x_end, chunk, t_hi;
    Wide jlo, jhi;
};

__host__ __device__ __forceinline__ bool wide_less(Wide a, Wide b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

__device__ __forceinline__ Wide make_wide(u64 x, u64 period, u64 t) {
    Wide result{__umul64hi(period, t), period * t};
    result.lo += x;
    if (result.lo < x) ++result.hi;
    return result;
}

static Wide from_u128(u128 value) {
    return Wide{(u64)(value >> 64), (u64)value};
}

static u128 to_u128(Wide value) {
    return ((u128)value.hi << 64) | value.lo;
}

__device__ __forceinline__ u64 mont(u64 a, u64 b) {
    u64 lo = a * b;
    u64 hi = __umul64hi(a, b);
    u64 m = lo * PINV;
    u64 mhi = __umul64hi(m, P);
    u64 r = hi + mhi + (lo != 0);
    u64 reduced = r - P;
    return (long long)reduced < 0 ? r : reduced;
}

__host__ __device__ __forceinline__ u64 odd_multiplier(u64 b) {
    return b + ((b & 1) ? 0 : P);
}

__device__ __forceinline__ u64 mont_fixed(u64 a, u64 b) {
    u64 lo = a * b;
    u64 hi = __umul64hi(a, b);
    u64 m = lo * PINV;
    u64 mhi = __umul64hi(m, P);
    u64 r = hi + mhi + 1;
    u64 reduced = r - P;
    return (long long)reduced < 0 ? r : reduced;
}

__host__ __device__ __forceinline__ u64 addmod(u64 a, u64 b) {
    u64 r = a + b;
    u64 reduced = r - P;
    return (long long)reduced < 0 ? r : reduced;
}

__host__ __device__ __forceinline__ u64 advance_mod(u64 a, u64 negative_step) {
    u64 difference = a - negative_step;
    return (long long)difference < 0 ? difference + P : difference;
}

__device__ u64 mont_pow_standard(u64 a, u64 e, u64 r2) {
    u64 result = mont(1, r2);
    u64 base = mont(a, r2);
    while (e) {
        if (e & 1) result = mont(result, base);
        e >>= 1;
        if (e) base = mont(base, base);
    }
    return mont(result, 1);
}

__device__ u64 mul_standard(u64 a, u64 b, u64 r2) {
    return mont(a, mont(b, r2));
}

template <bool SINGLE_T>
__global__ void scan_kernel(Job job, u64 begin, u64* pre_count, int* hit_count, Hit* hits) {
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 x0 = begin + tid * job.chunk;
    if (x0 >= job.x_end) return;
    u64 x1 = min(x0 + job.chunk, job.x_end);
    u64 z = mul_standard(job.z0, mont_pow_standard(job.g, x0, job.r2), job.r2);
    u64 w = addmod(mul_standard(job.ws, x0 % P, job.r2), job.w0);
    u64 local_pre = 0;
#pragma unroll 16
    for (u64 x = x0; x < x1; ++x) {
        u64 first_t = addmod(z, w);
        if (SINGLE_T) {
            if (first_t <= job.t_hi) {
                Wide j = make_wide(x, job.L, first_t);
                if (!wide_less(j, job.jlo) && !wide_less(job.jhi, j)) {
                    ++local_pre;
                    int position = atomicAdd(hit_count, 1);
                    if (position < MAX_HITS) hits[position] = {x, first_t};
                }
            }
        } else {
            for (u64 t = first_t;; t += P) {
                Wide j = make_wide(x, job.L, t);
                if (wide_less(job.jhi, j)) break;
                if (!wide_less(j, job.jlo)) {
                    ++local_pre;
                    int position = atomicAdd(hit_count, 1);
                    if (position < MAX_HITS) hits[position] = {x, t};
                }
                if (t > ~0ULL - P) break;
            }
        }
        z = mont_fixed(z, job.gm);
        w = advance_mod(w, P - job.ws);
    }
    atomicAdd(pre_count, local_pre);
}

__global__ void mont_test_kernel(const u64* a, const u64* b, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = mont(a[i], b[i]);
}

__global__ void fixed_test_kernel(const u64* a, const u64* b, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = mont_fixed(a[i], odd_multiplier(b[i]));
}

__global__ void trace_kernel(Job job, u64 x0, u64* out, int n) {
    if (blockIdx.x || threadIdx.x) return;
    u64 z = mul_standard(job.z0, mont_pow_standard(job.g, x0, job.r2), job.r2);
    u64 w = addmod(mul_standard(job.ws, x0 % P, job.r2), job.w0);
    for (int i = 0; i < n; ++i) {
        out[i] = addmod(z, w);
        z = mont_fixed(z, job.gm);
        w = advance_mod(w, P - job.ws);
    }
}

static void check_cuda(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(error) << "\n";
        std::exit(2);
    }
}

static u64 addmod_host(u64 a, u64 b, u64 p) { return (u64)(((u128)a + b) % p); }
static u64 submod(u64 a, u64 b, u64 p) { return a >= b ? a - b : p - (b - a); }
static u64 mulmod(u64 a, u64 b, u64 p) { return (u64)((u128)a * b % p); }

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

static u64 period_for_digit(int d) {
    if (d < 18 || d > 28) throw std::runtime_error("main search supports digits 18..28");
    if (d % 2 == 0) return R17;
    return 2 * R17;
}

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
        u64 U = mulmod(submod(remainder, addmod_host(mulmod(A, previous % p, p), B, p), p),
                       powmod(invmod(q, p), previous, p), p);
        remainder = addmod_host(mulmod(U, powmod(q, end, p), p),
                                addmod_host(mulmod(A, end % p, p), B, p), p);
        if (end == k) break;
        start *= 10;
    }
    return remainder;
}

static u64 concat_mod_sum(u128 k, u64 p) {
    u64 remainder = 0;
    u128 start = 1;
    while (start <= k) {
        u128 end = std::min(k, start * 10 - 1);
        u128 length = end - start + 1;
        int digits = 1;
        for (u128 x = start; x >= 10; x /= 10) ++digits;
        u64 q = powmod(10, digits, p);
        u64 qt = powmod(q, length, p);
        u64 denominator = submod(q, 1, p);
        u64 s0 = mulmod(submod(qt, 1, p), invmod(denominator, p), p);
        u64 lm = length % p;
        u64 numerator = addmod_host(submod(powmod(q, length + 1, p), mulmod(lm, q, p), p), lm, p);
        numerator = submod(numerator, q, p);
        u64 s1 = mulmod(numerator, invmod(mulmod(denominator, denominator, p), p), p);
        remainder = addmod_host(mulmod(remainder, qt, p),
                                addmod_host(mulmod((start - 1) % p, s0, p), s1, p), p);
        if (end == k) break;
        start *= 10;
    }
    return remainder;
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

struct SmallJob {
    u64 m, q, A, B, U, order;
};

static SmallJob make_small_job(int d, u64 m) {
    u64 q = powmod(10, d, m);
    u64 order = 1, value = q;
    while (value != 1) {
        value = mulmod(value, q, m);
        ++order;
    }
    SmallJob job{m, q, 0, 0, 0, order};
    if (order > 1) {
        job.A = invmod(submod(1, q, m), m);
        job.B = submod(0, mulmod(q, mulmod(job.A, job.A, m), m), m);
        u128 start = 1;
        for (int i = 1; i < d; ++i) start *= 10;
        job.U = mulmod(submod(concat_mod(start - 1, m),
                              addmod_host(mulmod(job.A, (start - 1) % m, m), job.B, m), m),
                       powmod(invmod(q, m), start - 1, m), m);
    }
    return job;
}

static u64 small_remainder(u128 k, const SmallJob& job) {
    if (job.order == 1) return concat_mod(k, job.m);
    u64 power = powmod(job.q, k % job.order, job.m);
    return addmod_host(mulmod(job.U, power, job.m),
                       addmod_host(mulmod(job.A, k % job.m, job.m), job.B, job.m), job.m);
}

static Job make_job(int d, int a, u64 x_end, u64 chunk) {
    u64 L = period_for_digit(d);
    u128 start = 1;
    for (int i = 1; i < d; ++i) start *= 10;
    u128 lo = start, hi = start * 10 - 1;
    u64 q = powmod(10, d, P);
    u64 A = invmod(submod(1, q, P), P);
    u64 B = submod(0, mulmod(q, mulmod(A, A, P), P), P);
    u64 U = mulmod(submod(concat_mod(start - 1, P),
                          addmod_host(mulmod(A, (start - 1) % P, P), B, P), P),
                   powmod(invmod(q, P), start - 1, P), P);
    u64 D = mulmod(mulmod(A, 9, P), L, P);
    u64 denominator_inverse = invmod(D, P);
    u64 R = (u64)(((u128)1 << 64) % P);
    Job job{};
    job.d = d;
    job.a = a;
    job.z0 = submod(0, mulmod(mulmod(U, powmod(q, a, P), P), denominator_inverse, P), P);
    if (!job.z0) throw std::runtime_error("odd fixed multiplier requires a nonzero orbit");
    job.g = powmod(q, 9, P);
    job.ws = submod(0, invmod(L, P), P);
    job.w0 = submod(0, mulmod(addmod_host(mulmod(A, a, P), B, P), denominator_inverse, P), P);
    job.gm = odd_multiplier(mulmod(job.g, R, P));
    job.r2 = mulmod(R, R, P);
    job.L = L;
    job.x_end = x_end;
    job.chunk = chunk;
    job.jlo = from_u128((lo - a + 8) / 9);
    job.jhi = from_u128((hi - a) / 9);
    job.t_hi = (u64)((hi - a) / (9 * L));
    return job;
}

static std::string state_path(const std::string& name) { return "results/" + name + ".state"; }
static std::string hits_path(const std::string& name) { return "results/" + name + ".candidates"; }

struct State { u64 next, pre, hits; };

static void save_state(const std::string& name, const Job& job, u64 begin, u64 end, State state, int blocks) {
    std::string temporary = state_path(name) + ".tmp";
    std::ofstream output(temporary);
    output << "config=" << job.d << ":" << job.a << ":" << begin << ":" << end << "\n"
           << "next_x=" << state.next << "\npre_hits=" << state.pre << "\nhits=" << state.hits
           << "\nperiod=" << job.L << "\nchunk=" << job.chunk << "\nblocks=" << blocks << "\n";
    output.close();
    if (!output) throw std::runtime_error("failed to save state");
    std::filesystem::rename(temporary, state_path(name));
}

static State load_state(const std::string& name, const Job& job, u64 begin, u64 end) {
    std::ifstream input(state_path(name));
    if (!input) return {begin, 0, 0};
    std::string line, config;
    State result{begin, 0, 0};
    while (std::getline(input, line)) {
        if (line.rfind("config=", 0) == 0) config = line.substr(7);
        else if (line.rfind("next_x=", 0) == 0) result.next = std::stoull(line.substr(7));
        else if (line.rfind("pre_hits=", 0) == 0) result.pre = std::stoull(line.substr(9));
        else if (line.rfind("hits=", 0) == 0) result.hits = std::stoull(line.substr(5));
    }
    if (config != std::to_string(job.d) + ":" + std::to_string(job.a) + ":" +
            std::to_string(begin) + ":" + std::to_string(end) || result.next < begin || result.next > end) {
        throw std::runtime_error("state mismatch");
    }
    return result;
}

static bool full_candidate(u128 k, int d, const SmallJob& small47, const SmallJob& small4993) {
    if (k <= 17 || (k % 9 != 0 && k % 9 != 8)) return false;
    if (concat_mod(k, P) != 0) return false;
    if (small_remainder(k, small47) != 0) return false;
    return small_remainder(k, small4993) == 0;
}

static bool check_trace(const Job& job, u64 x0, int n) {
    u64* device_out;
    std::vector<u64> out(n);
    check_cuda(cudaMalloc(&device_out, n * sizeof(u64)), "malloc trace");
    trace_kernel<<<1, 1>>>(job, x0, device_out, n);
    check_cuda(cudaGetLastError(), "trace launch");
    check_cuda(cudaMemcpy(out.data(), device_out, n * sizeof(u64), cudaMemcpyDeviceToHost), "copy trace");
    cudaFree(device_out);
    u64 q = powmod(10, job.d, P);
    u64 A = invmod(submod(1, q, P), P);
    u64 D = mulmod(mulmod(A, 9, P), job.L, P);
    u128 lo = to_u128(job.jlo), hi = to_u128(job.jhi);
    for (int i = 0; i < n; ++i) {
        u64 x = x0 + i;
        u128 t = x < lo ? (lo - x + job.L - 1) / job.L : 0;
        for (u128 delta = 0; delta < 2 && x + job.L * t <= hi; ++delta, ++t) {
            u128 k = job.a + 9 * (x + job.L * t);
            u64 expected = mulmod(D, submod(t % P, out[i], P), P);
            if (concat_mod(k, P) != expected) {
                std::cerr << "trace mismatch x=" << x << " t=" << u128_string(t) << "\n";
                return false;
            }
        }
    }
    return true;
}

static bool selftest(int device) {
    check_cuda(cudaSetDevice(device), "set device");
    const int n = 100000;
    std::mt19937_64 rng(40066617);
    std::vector<u64> a(n), b(n), out(n);
    for (int i = 0; i < n; ++i) {
        a[i] = rng() % P;
        b[i] = rng() % P;
    }
    u64 *da, *db, *dout;
    check_cuda(cudaMalloc(&da, n * sizeof(u64)), "malloc a");
    check_cuda(cudaMalloc(&db, n * sizeof(u64)), "malloc b");
    check_cuda(cudaMalloc(&dout, n * sizeof(u64)), "malloc out");
    check_cuda(cudaMemcpy(da, a.data(), a.size() * sizeof(u64), cudaMemcpyHostToDevice), "copy a");
    check_cuda(cudaMemcpy(db, b.data(), b.size() * sizeof(u64), cudaMemcpyHostToDevice), "copy b");
    mont_test_kernel<<<(n + 255) / 256, 256>>>(da, db, dout, n);
    check_cuda(cudaGetLastError(), "Montgomery launch");
    check_cuda(cudaMemcpy(out.data(), dout, out.size() * sizeof(u64), cudaMemcpyDeviceToHost), "Montgomery result");
    u64 R = (u64)(((u128)1 << 64) % P), rinv = invmod(R, P);
    for (int i = 0; i < n; ++i) {
        if (out[i] != mulmod(mulmod(a[i], b[i], P), rinv, P)) {
            std::cerr << "Montgomery mismatch at " << i << "\n";
            return false;
        }
    }
    for (int i = 0; i < n; ++i) if (!a[i]) a[i] = 1;
    fixed_test_kernel<<<(n + 255) / 256, 256>>>(da, db, dout, n);
    check_cuda(cudaGetLastError(), "fixed Montgomery launch");
    check_cuda(cudaMemcpy(out.data(), dout, out.size() * sizeof(u64), cudaMemcpyDeviceToHost), "fixed Montgomery result");
    for (int i = 0; i < n; ++i) {
        if (out[i] != mulmod(mulmod(a[i], b[i], P), rinv, P)) {
            std::cerr << "fixed Montgomery mismatch at " << i << "\n";
            return false;
        }
    }
    for (u64 k = 1; k <= 1000; ++k) {
        if (concat_mod(k, P) != concat_mod_sum(k, P)) return false;
    }
    for (int i = 0; i < 100; ++i) {
        u128 k = (u128)(rng() % 1000000000000ULL) * 10000000000ULL + rng() % 10000000000ULL;
        if (concat_mod(k, P) != concat_mod_sum(k, P)) return false;
    }
    for (int d : {18, 23, 24}) {
        SmallJob small47 = make_small_job(d, 47), small4993 = make_small_job(d, 4993);
        u128 lo = 1;
        for (int i = 1; i < d; ++i) lo *= 10;
        u128 span = 9 * lo;
        for (int i = 0; i < 100; ++i) {
            u128 random = ((u128)rng() << 64) | rng();
            u128 k = lo + random % span;
            if (small_remainder(k, small47) != concat_mod(k, 47) ||
                small_remainder(k, small4993) != concat_mod(k, 4993)) {
                std::cerr << "small-factor mismatch d=" << d << " k=" << u128_string(k) << "\n";
                return false;
            }
        }
    }
    for (int a : {0, 8}) {
        Job job = make_job(24, a, 1000, 65536);
        for (u64 x0 : {0ULL, 1ULL, 123456789ULL, R17 - 64}) {
            if (!check_trace(job, x0, 64)) return false;
        }
    }
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    std::cout << "SELFTEST_OK montgomery=100000 concat=1100 small_factor=300 traces=512\n";
    return true;
}

static int run(const std::string& name, int d, int a, int device, u64 begin, u64 end, u64 chunk, int blocks) {
    check_cuda(cudaSetDevice(device), "set run device");
    Job job = make_job(d, a, end, chunk);
    Hit* device_hits;
    int* device_count;
    u64* device_pre;
    check_cuda(cudaMalloc(&device_hits, MAX_HITS * sizeof(Hit)), "malloc hits");
    check_cuda(cudaMalloc(&device_count, sizeof(int)), "malloc count");
    check_cuda(cudaMalloc(&device_pre, sizeof(u64)), "malloc pre");
    SmallJob small47 = make_small_job(d, 47), small4993 = make_small_job(d, 4993);
    State state = load_state(name, job, begin, end);
    save_state(name, job, begin, end, state, blocks);
    auto started = std::chrono::steady_clock::now();
    auto last = started;
    while (state.next < end) {
        u64 launch_end = std::min(end, state.next + (u64)blocks * BLOCK_SIZE * chunk);
        check_cuda(cudaMemset(device_count, 0, sizeof(int)), "clear count");
        check_cuda(cudaMemset(device_pre, 0, sizeof(u64)), "clear pre");
        if (job.t_hi < P) scan_kernel<true><<<blocks, BLOCK_SIZE>>>(job, state.next, device_pre, device_count, device_hits);
        else scan_kernel<false><<<blocks, BLOCK_SIZE>>>(job, state.next, device_pre, device_count, device_hits);
        check_cuda(cudaGetLastError(), "scan launch");
        int count = 0;
        u64 pre = 0;
        check_cuda(cudaMemcpy(&count, device_count, sizeof(int), cudaMemcpyDeviceToHost), "copy count");
        check_cuda(cudaMemcpy(&pre, device_pre, sizeof(u64), cudaMemcpyDeviceToHost), "copy pre");
        if (count > MAX_HITS) throw std::runtime_error("candidate buffer overflow");
        u64 launched = launch_end - state.next;
        state.next = launch_end;
        state.pre += pre;
        if (count) {
            std::vector<Hit> hits(count);
            check_cuda(cudaMemcpy(hits.data(), device_hits, count * sizeof(Hit), cudaMemcpyDeviceToHost), "copy hits");
            std::ofstream output(hits_path(name), std::ios::app);
            for (const Hit& hit : hits) {
                u128 k = job.a + 9 * ((u128)hit.x + (u128)job.L * hit.t);
                if (!full_candidate(k, d, small47, small4993)) continue;
                output << "k=" << u128_string(k) << " d=" << d << " a=" << job.a
                       << " x=" << hit.x << " t=" << hit.t << " remainder_p=0 small_factors=0\n";
                ++state.hits;
            }
            output.flush();
        }
        save_state(name, job, begin, end, state, blocks);
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - started).count();
        double dt = std::chrono::duration<double>(now - last).count();
        last = now;
        std::cout << "progress name=" << name << " next_x=" << state.next << "/" << end
                  << " rate=" << launched / dt << " elapsed=" << elapsed
                  << " pre=" << state.pre << " hits=" << state.hits << "\n";
        if (std::filesystem::exists("results/stop") && state.next < end) {
            std::cout << "CLEAN_STOP name=" << name << " next_x=" << state.next << "\n";
            break;
        }
    }
    if (state.next == end) std::cout << "DONE name=" << name << " next_x=" << end
                                     << " pre=" << state.pre << " hits=" << state.hits << "\n";
    cudaFree(device_hits);
    cudaFree(device_count);
    cudaFree(device_pre);
    return 0;
}

int main(int argc, char** argv) {
    try {
        std::filesystem::create_directories("results");
        if (argc == 3 && std::string(argv[1]) == "selftest") return selftest(std::stoi(argv[2])) ? 0 : 1;
        if (argc == 10 && std::string(argv[1]) == "run") {
            return run(argv[2], std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]),
                       std::stoull(argv[6]), std::stoull(argv[7]), std::stoull(argv[8]), std::stoi(argv[9]));
        }
        std::cerr << "usage: search17 selftest device | run name d a device x_begin x_end chunk blocks\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
}
