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

#ifndef DIGITS
#define DIGITS 19
#endif
#ifndef STEP
#define STEP 5
#endif
#ifndef PERIOD
#define PERIOD 823045260674080876ULL
#endif
#ifndef T_LIMIT
#define T_LIMIT 2
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
static constexpr u64 P = 8230452606740808761ULL;
static constexpr u64 PINV = 12692191010489013751ULL;
static constexpr u64 L = PERIOD;

struct Job {
    u64 z0, g, w0, ws, r2, x_end, chunk;
};

struct Hit {
    u64 x, t;
};

__device__ __forceinline__ u64 mont_mul(u64 a, u64 b) {
    u64 lo = a * b;
    u64 hi = __umul64hi(a, b);
    u64 m = lo * 12692191010489013751ULL;
    u64 mhi = __umul64hi(m, 8230452606740808761ULL);
    u64 r = hi + mhi + (lo != 0);
    u64 reduced = r - 8230452606740808761ULL;
    return (long long)reduced < 0 ? r : reduced;
}

__host__ __device__ __forceinline__ u64 addmod(u64 a, u64 b) {
    u64 r = a + b;
    u64 reduced = r - 8230452606740808761ULL;
    return (long long)reduced < 0 ? r : reduced;
}

__device__ u64 mont_pow_standard(u64 a, u64 e, u64 r2) {
    u64 result = mont_mul(1, r2);
    u64 base = mont_mul(a, r2);
    while (e) {
        if (e & 1) result = mont_mul(result, base);
        e >>= 1;
        if (e) base = mont_mul(base, base);
    }
    return mont_mul(result, 1);
}

__device__ u64 mul_standard(u64 a, u64 b, u64 r2) {
    return mont_mul(a, mont_mul(b, r2));
}

__global__ void scan_kernel(Job job, u64 begin, Hit* hits, int* count, int max_hits) {
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 x0 = begin + tid * job.chunk;
    if (x0 >= job.x_end) return;
    u64 x1 = min(x0 + job.chunk, job.x_end);
    u64 z = mul_standard(job.z0, mont_pow_standard(job.g, x0, job.r2), job.r2);
    u64 w = addmod(mul_standard(job.ws, x0, job.r2), job.w0);
    u64 gm = mont_mul(job.g, job.r2);
    for (u64 x = x0; x < x1; ++x) {
        u64 t = addmod(z, w);
        if (t <= T_LIMIT) {
            int pos = atomicAdd(count, 1);
            if (pos < max_hits) hits[pos] = {x, t};
        }
        z = mont_mul(z, gm);
        w = addmod(w, job.ws);
    }
}

__global__ void mont_test_kernel(const u64* a, const u64* b, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = mont_mul(a[i], b[i]);
}

__global__ void trace_kernel(Job job, u64 x0, u64* out, int n) {
    if (blockIdx.x || threadIdx.x) return;
    u64 z = mul_standard(job.z0, mont_pow_standard(job.g, x0, job.r2), job.r2);
    u64 w = addmod(mul_standard(job.ws, x0, job.r2), job.w0);
    u64 gm = mont_mul(job.g, job.r2);
    for (int i = 0; i < n; ++i) {
        out[i] = addmod(z, w);
        z = mont_mul(z, gm);
        w = addmod(w, job.ws);
    }
}

static void check_cuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(e) << "\n";
        std::exit(2);
    }
}

static u64 submod(u64 a, u64 b) {
    return a >= b ? a - b : addmod(a, P - b);
}

static u64 mulmod(u64 a, u64 b) {
    return (u64)((u128)a * b % P);
}

static u64 powmod(u64 a, u128 e) {
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

static u64 concat_mod(u128 k) {
    u64 r = 0;
    u128 start = 1;
    for (u64 d = 1; d <= 21 && start <= k; ++d) {
        u128 end = std::min(k, start * 10 - 1);
        u64 q = powmod(10, d);
        u64 A = invmod(submod(1, q));
        u64 B = submod(0, mulmod(q, mulmod(A, A)));
        u128 prev = start - 1;
        u64 U = mulmod(submod(r, addmod(mulmod(A, prev % P), B)), powmod(invmod(q), prev));
        r = addmod(mulmod(U, powmod(q, end)), addmod(mulmod(A, end % P), B));
        if (end == k) break;
        start *= 10;
    }
    return r;
}

static u64 concat_mod_sum(u128 k) {
    u64 r = 0;
    u128 start = 1;
    for (u64 d = 1; d <= 21 && start <= k; ++d) {
        u128 end = std::min(k, start * 10 - 1);
        u128 len = end - start + 1;
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

static std::string u128str(u128 x) {
    if (!x) return "0";
    std::string s;
    while (x) {
        s.push_back('0' + x % 10);
        x /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

static Job make_job(u64 x_end, u64 chunk) {
    u128 start = 1;
    for (int i = 1; i < DIGITS; ++i) start *= 10;
    u64 q = powmod(10, DIGITS);
    u64 A = invmod(submod(1, q));
    u64 B = submod(0, mulmod(q, mulmod(A, A)));
    u128 prev = start - 1;
    u64 rprev = concat_mod(prev);
    u64 U = mulmod(submod(rprev, addmod(mulmod(A, prev % P), B)), powmod(invmod(q), prev));
    u64 deninv = invmod(mulmod(mulmod(A, STEP), L));
    Job job{};
    job.z0 = submod(0, mulmod(U, deninv));
    job.g = powmod(q, STEP);
    job.w0 = submod(0, mulmod(B, deninv));
    job.ws = submod(0, invmod(L));
    u64 R = (u64)(((u128)1 << 64) % P);
    job.r2 = mulmod(R, R);
    job.x_end = x_end;
    job.chunk = chunk;
    return job;
}

static std::string state_path(const std::string& name) { return "results15/" + name + ".state"; }
static std::string hits_path(const std::string& name) { return "results15/" + name + ".candidates"; }

static std::string config_key(u64 begin, u64 end) {
    return std::to_string(begin) + ":" + std::to_string(end);
}

static void save_state(const std::string& name, u64 begin, u64 end, u64 next, u64 hits, u64 chunk, u64 blocks) {
    std::string tmp = state_path(name) + ".tmp";
    std::ofstream f(tmp);
    f << "config=" << config_key(begin, end) << "\nnext_x=" << next << "\nhits=" << hits
      << "\nchunk=" << chunk << "\nblocks=" << blocks << "\n";
    f.close();
    std::filesystem::rename(tmp, state_path(name));
}

static std::pair<u64, u64> load_state(const std::string& name, u64 begin, u64 end) {
    std::ifstream f(state_path(name));
    if (!f) return {begin, 0};
    std::string line, key;
    u64 next = begin, hits = 0;
    while (std::getline(f, line)) {
        if (line.rfind("config=", 0) == 0) key = line.substr(7);
        else if (line.rfind("next_x=", 0) == 0) next = std::stoull(line.substr(7));
        else if (line.rfind("hits=", 0) == 0) hits = std::stoull(line.substr(5));
    }
    if (key != config_key(begin, end) || next < begin || next > end) throw std::runtime_error("state mismatch");
    return {next, hits};
}

static bool selftest(int device) {
    check_cuda(cudaSetDevice(device), "set device");
    const int n = 100000;
    std::mt19937_64 rng(40066615);
    std::vector<u64> a(n), b(n), out(n);
    for (int i = 0; i < n; ++i) {
        a[i] = rng() % P;
        b[i] = rng() % P;
    }
    u64 *da, *db, *dout;
    check_cuda(cudaMalloc(&da, n * sizeof(u64)), "malloc");
    check_cuda(cudaMalloc(&db, n * sizeof(u64)), "malloc");
    check_cuda(cudaMalloc(&dout, n * sizeof(u64)), "malloc");
    check_cuda(cudaMemcpy(da, a.data(), a.size() * sizeof(u64), cudaMemcpyHostToDevice), "copy");
    check_cuda(cudaMemcpy(db, b.data(), b.size() * sizeof(u64), cudaMemcpyHostToDevice), "copy");
    mont_test_kernel<<<(n + 255) / 256, 256>>>(da, db, dout, n);
    check_cuda(cudaGetLastError(), "mont launch");
    check_cuda(cudaMemcpy(out.data(), dout, out.size() * sizeof(u64), cudaMemcpyDeviceToHost), "mont result");
    u64 R = (u64)(((u128)1 << 64) % P);
    u64 rinv = invmod(R);
    for (int i = 0; i < n; ++i) {
        if (out[i] != mulmod(mulmod(a[i], b[i]), rinv)) {
            std::cerr << "mont mismatch at " << i << "\n";
            return false;
        }
    }
    for (u64 k = 1; k <= 1000; ++k) {
        if (concat_mod(k) != concat_mod_sum(k)) {
            std::cerr << "small concat mismatch at " << k << "\n";
            return false;
        }
    }
    for (int i = 0; i < 100; ++i) {
        u128 k = (u128)(rng() % 10000000000000000000ULL) * 10 + rng() % 10;
        if (concat_mod(k) != concat_mod_sum(k)) {
            std::cerr << "large concat mismatch at " << u128str(k) << "\n";
            return false;
        }
    }
    if (concat_mod(15) != 0) {
        std::cerr << "c(15) check failed\n";
        return false;
    }
    Job job = make_job(1, 1);
    const int traces = 64;
    for (u64 x0 : {0ULL, 1ULL, 17ULL, 123456789ULL, L - 1}) {
        trace_kernel<<<1, 1>>>(job, x0, dout, traces);
        check_cuda(cudaGetLastError(), "trace launch");
        check_cuda(cudaMemcpy(out.data(), dout, traces * sizeof(u64), cudaMemcpyDeviceToHost), "trace result");
        u64 z = mulmod(job.z0, powmod(job.g, x0));
        u64 w = addmod(mulmod(job.ws, x0), job.w0);
        for (int i = 0; i < traces; ++i) {
            if (out[i] != addmod(z, w)) {
                std::cerr << "trace mismatch x0=" << x0 << " i=" << i << "\n";
                return false;
            }
            z = mulmod(z, job.g);
            w = addmod(w, job.ws);
        }
    }
    cudaFree(da); cudaFree(db); cudaFree(dout);
    std::cout << "selftest ok: 100000 Montgomery products, concatenation cross-checks, five 64-point traces\n";
    return true;
}

static int run(const std::string& name, int device, u64 x_begin, u64 x_end, u64 chunk, u64 blocks) {
    check_cuda(cudaSetDevice(device), "set device");
    Job job = make_job(x_end, chunk);
    auto [next, total_hits] = load_state(name, x_begin, x_end);
    std::filesystem::create_directories("results15");
    Hit* d_hits;
    int* d_count;
    check_cuda(cudaMalloc(&d_hits, 65536 * sizeof(Hit)), "malloc hits");
    check_cuda(cudaMalloc(&d_count, sizeof(int)), "malloc count");
    save_state(name, x_begin, x_end, next, total_hits, chunk, blocks);
    auto started = std::chrono::steady_clock::now();
    auto last = started;
    while (next < x_end) {
        int launch_blocks = (int)std::min<u64>(blocks, (x_end - next + chunk - 1) / chunk);
        check_cuda(cudaMemset(d_count, 0, sizeof(int)), "memset");
        scan_kernel<<<launch_blocks, BLOCK_SIZE>>>(job, next, d_hits, d_count, 65536);
        check_cuda(cudaGetLastError(), "scan launch");
        int count = 0;
        check_cuda(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost), "count");
        if (count > 65536) throw std::runtime_error("hit overflow");
        u64 launched = std::min(x_end - next, (u64)launch_blocks * BLOCK_SIZE * chunk);
        next += launched;
        if (count) {
            std::vector<Hit> hits(count);
            check_cuda(cudaMemcpy(hits.data(), d_hits, count * sizeof(Hit), cudaMemcpyDeviceToHost), "hits");
            std::ofstream f(hits_path(name), std::ios::app);
            u128 lo = 1;
            for (int i = 1; i < DIGITS; ++i) lo *= 10;
            u128 hi = lo * 10 - 1;
            for (const Hit& hit : hits) {
                u128 k = (u128)STEP * ((u128)hit.x + (u128)L * hit.t);
                if (k < lo || k > hi || k % 15 != 0 && k % 15 != 5) continue;
                u64 rem = concat_mod(k);
                if (rem != 0 || concat_mod_sum(k) != 0) {
                    std::cerr << "false hit k=" << u128str(k) << "\n";
                    return 3;
                }
                f << "k=" << u128str(k) << " x=" << hit.x << " t=" << hit.t
                  << " remainder_p=0 independent_remainder_p=0 k_mod_15=" << (u64)(k % 15) << "\n";
                f.flush();
                ++total_hits;
                std::cout << "HIT k=" << u128str(k) << " x=" << hit.x << " t=" << hit.t << "\n";
            }
        }
        save_state(name, x_begin, x_end, next, total_hits, chunk, blocks);
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - started).count();
        double dt = std::chrono::duration<double>(now - last).count();
        last = now;
        std::cout << "progress name=" << name << " next_x=" << next << "/" << x_end
                  << " rate=" << launched / dt << " x/s elapsed=" << elapsed << " hits=" << total_hits << "\n";
    }
    cudaFree(d_hits); cudaFree(d_count);
    std::cout << "DONE name=" << name << " next_x=" << next << " hits=" << total_hits << "\n";
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "selftest") return selftest(std::stoi(argv[2])) ? 0 : 1;
        if (argc == 8 && std::string(argv[1]) == "run") {
            return run(argv[2], std::stoi(argv[3]), std::stoull(argv[4]), std::stoull(argv[5]),
                std::stoull(argv[6]), std::stoull(argv[7]));
        }
        std::cerr << "usage: search15 selftest device | run name device x_begin x_end chunk blocks\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
