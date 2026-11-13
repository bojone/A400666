#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using u64 = unsigned long long;
using u128 = unsigned __int128;
using i128 = __int128;

static constexpr u64 KNOWN[14] = {0, 2, 6, 23, 1838, 15000, 6992, 2888377,
    795420, 37366496, 8803255110ULL, 1462927131789ULL, 36810773632272ULL, 2879846318370304ULL};
static constexpr u64 CHUNK = 262144;
static constexpr int BLOCKS = 16384;

struct Job {
    u64 mod, pinv, r2;
    u64 q[20], qm[20], powers[20];
    u64 k_end, chunk;
};

struct Block {
    u64 q, s0, s1, len, lm;
};

__device__ __forceinline__ u64 addmod(u64 a, u64 b, u64 m) {
    u64 r = a + b;
    if (r >= m) r -= m;
    return r;
}

__device__ __forceinline__ u64 mont_mul(u64 a, u64 b, u64 m, u64 pinv) {
    u64 lo = a * b;
    u64 hi = __umul64hi(a, b);
    u64 mm = lo * pinv;
    u64 mhi = __umul64hi(mm, m);
    u64 r = hi + mhi + (lo != 0);
    if (r >= m) r -= m;
    return r;
}

__device__ __forceinline__ u64 mul_standard(u64 a, u64 b, const Job& job) {
    return mont_mul(a, mont_mul(b, job.r2, job.mod, job.pinv), job.mod, job.pinv);
}

__device__ Block combine(Block a, Block b, const Job& job) {
    Block c;
    c.q = mul_standard(a.q, b.q, job);
    c.s0 = addmod(mul_standard(a.s0, b.q, job), b.s0, job.mod);
    c.s1 = addmod(addmod(mul_standard(a.s1, b.q, job), mul_standard(a.lm, b.s0, job), job.mod), b.s1, job.mod);
    c.len = a.len + b.len;
    c.lm = addmod(a.lm, b.lm, job.mod);
    return c;
}

__device__ Block block_sums(u64 q, u64 length, const Job& job) {
    Block result{1 % job.mod, 0, 0, 0, 0};
    Block block{q, 1 % job.mod, 1 % job.mod, 1, 1 % job.mod};
    while (length) {
        if (length & 1) result = combine(result, block, job);
        length >>= 1;
        if (length) block = combine(block, block, job);
    }
    return result;
}

__device__ u64 concat_mod(u64 k, const Job& job) {
    u64 r = 0;
    u64 start = 1;
    for (int d = 1; d <= 19 && start <= k; ++d) {
        u64 end = min(k, start * 10 - 1);
        Block b = block_sums(job.q[d], end - start + 1, job);
        r = addmod(mul_standard(r, b.q, job),
            addmod(mul_standard((start - 1) % job.mod, b.s0, job), b.s1, job.mod), job.mod);
        if (end == k) break;
        start *= 10;
    }
    return r;
}

__device__ int digits10(u64 k) {
    int d = 1;
    u64 p = 10;
    while (k >= p && d < 19) {
        ++d;
        p *= 10;
    }
    return d;
}

__global__ void scan_kernel(Job job, u64 begin, u64* hits, int* count, int max_hits) {
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 k0 = begin + tid * job.chunk;
    if (k0 >= job.k_end) return;
    u64 k1 = min(k0 + job.chunk, job.k_end);
    u64 r = concat_mod(k0 - 1, job);
    int d = digits10(k0);
    u64 qm = job.qm[d];
    u64 next_power = job.powers[d];
    u64 km = k0 % job.mod;
    for (u64 k = k0; k < k1; ++k) {
        r = addmod(mont_mul(r, qm, job.mod, job.pinv), km, job.mod);
        if (r == 0) {
            int pos = atomicAdd(count, 1);
            if (pos < max_hits) hits[pos] = k;
        }
        km = km + 1 == job.mod ? 0 : km + 1;
        if (k + 1 == next_power && d < 19) {
            ++d;
            qm = job.qm[d];
            next_power *= 10;
        }
    }
}

__global__ void mont_test_kernel(Job job, const u64* a, const u64* b, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = mont_mul(a[i], b[i], job.mod, job.pinv);
}

__global__ void concat_test_kernel(Job job, const u64* values, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = concat_mod(values[i], job);
}

static void check_cuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(e) << "\n";
        std::exit(2);
    }
}

static u64 mulmod(u64 a, u64 b, u64 m) {
    return (u64)((u128)a * b % m);
}

static u64 addmod_host(u64 a, u64 b, u64 m) {
    return (u64)(((u128)a + b) % m);
}

static u64 literal(int n) {
    std::string s;
    for (int i = 1; i <= n; ++i) s += std::to_string(i);
    return std::stoull(s);
}

static u64 odd_part(u64 m) {
    while (!(m & 1)) m >>= 1;
    return m;
}

static u64 mont_inv(u64 m) {
    u64 inv = 1;
    for (int i = 0; i < 6; ++i) inv *= 2 - m * inv;
    return 0 - inv;
}

static u64 invmod(u64 a, u64 m) {
    i128 old_r = a, r = m, old_s = 1, s = 0;
    while (r) {
        i128 q = old_r / r;
        i128 nr = old_r - q * r;
        old_r = r; r = nr;
        i128 ns = old_s - q * s;
        old_s = s; s = ns;
    }
    old_s %= m;
    if (old_s < 0) old_s += m;
    return (u64)old_s;
}

static Block combine_host(Block a, Block b, u64 m) {
    return {mulmod(a.q, b.q, m),
        addmod_host(mulmod(a.s0, b.q, m), b.s0, m),
        addmod_host(addmod_host(mulmod(a.s1, b.q, m), mulmod(a.lm, b.s0, m), m), b.s1, m),
        a.len + b.len, addmod_host(a.lm, b.lm, m)};
}

static Block block_sums_host(u64 q, u64 length, u64 m) {
    Block result{1 % m, 0, 0, 0, 0};
    Block block{q, 1 % m, 1 % m, 1, 1 % m};
    while (length) {
        if (length & 1) result = combine_host(result, block, m);
        length >>= 1;
        if (length) block = combine_host(block, block, m);
    }
    return result;
}

static u64 concat_mod_host(u64 k, u64 m) {
    u64 r = 0;
    for (int d = 1; d <= 19; ++d) {
        u64 start = 1;
        for (int i = 1; i < d; ++i) start *= 10;
        if (start > k) break;
        u64 end = std::min(k, start * 10 - 1);
        u64 q = 1;
        for (int i = 0; i < d; ++i) q = mulmod(q, 10, m);
        Block b = block_sums_host(q, end - start + 1, m);
        r = addmod_host(mulmod(r, b.q, m),
            addmod_host(mulmod((start - 1) % m, b.s0, m), b.s1, m), m);
    }
    return r;
}

static Job make_job(u64 modulus) {
    Job job{};
    job.mod = odd_part(modulus);
    job.pinv = mont_inv(job.mod);
    u64 R = (u64)(((u128)1 << 64) % job.mod);
    job.r2 = mulmod(R, R, job.mod);
    job.q[0] = 0;
    job.powers[0] = 1;
    for (int d = 1; d <= 19; ++d) {
        job.q[d] = d == 1 ? 10 % job.mod : mulmod(job.q[d - 1], 10, job.mod);
        job.qm[d] = mulmod(job.q[d], R, job.mod);
        job.powers[d] = d == 1 ? 10 : job.powers[d - 1] * 10;
    }
    job.chunk = CHUNK;
    return job;
}

static std::string state_path(int n, int device) {
    char name[64];
    std::snprintf(name, sizeof(name), "oeis_results/n%02d_dev%d.state", n, device);
    return name;
}

static std::string hit_path(int n, int device) {
    char name[64];
    std::snprintf(name, sizeof(name), "oeis_results/n%02d_dev%d.early", n, device);
    return name;
}

static std::string config_key(int n, u64 begin, u64 end, u64 m) {
    return std::to_string(n) + ":" + std::to_string(begin) + ":" + std::to_string(end) + ":" + std::to_string(m);
}

static void save_state(int n, int device, u64 begin, u64 end, u64 m, u64 next, u64 raw_hits, u64 early_hits) {
    std::string tmp = state_path(n, device) + ".tmp";
    std::ofstream f(tmp);
    f << "config=" << config_key(n, begin, end, m) << "\nnext_k=" << next
      << "\nraw_hits=" << raw_hits << "\nearly_hits=" << early_hits << "\n";
    f.close();
    std::filesystem::rename(tmp, state_path(n, device));
}

static u64 load_state(int n, int device, u64 begin, u64 end, u64 m, u64& raw_hits, u64& early_hits) {
    std::ifstream f(state_path(n, device));
    if (!f) return begin;
    std::string line, key;
    u64 next = begin;
    while (std::getline(f, line)) {
        if (line.rfind("config=", 0) == 0) key = line.substr(7);
        else if (line.rfind("next_k=", 0) == 0) next = std::stoull(line.substr(7));
        else if (line.rfind("raw_hits=", 0) == 0) raw_hits = std::stoull(line.substr(9));
        else if (line.rfind("early_hits=", 0) == 0) early_hits = std::stoull(line.substr(11));
    }
    if (key != config_key(n, begin, end, m) || next < begin || next > end) throw std::runtime_error("state mismatch");
    return next;
}

static bool selftest(int device) {
    check_cuda(cudaSetDevice(device), "set device");
    const int count = 10000;
    std::mt19937_64 rng(400666);
    for (int n = 2; n <= 13; ++n) {
        u64 modulus = literal(n);
        Job job = make_job(modulus);
        std::vector<u64> a(count), b(count), out(count);
        for (int i = 0; i < count; ++i) {
            a[i] = rng() % job.mod;
            b[i] = rng() % job.mod;
        }
        u64 *da, *db, *dout;
        check_cuda(cudaMalloc(&da, count * sizeof(u64)), "malloc");
        check_cuda(cudaMalloc(&db, count * sizeof(u64)), "malloc");
        check_cuda(cudaMalloc(&dout, count * sizeof(u64)), "malloc");
        check_cuda(cudaMemcpy(da, a.data(), a.size() * sizeof(u64), cudaMemcpyHostToDevice), "copy");
        check_cuda(cudaMemcpy(db, b.data(), b.size() * sizeof(u64), cudaMemcpyHostToDevice), "copy");
        mont_test_kernel<<<(count + 255) / 256, 256>>>(job, da, db, dout, count);
        check_cuda(cudaGetLastError(), "mont launch");
        check_cuda(cudaMemcpy(out.data(), dout, out.size() * sizeof(u64), cudaMemcpyDeviceToHost), "mont result");
        u64 R = (u64)(((u128)1 << 64) % job.mod);
        u64 rinv = invmod(R, job.mod);
        for (int i = 0; i < count; ++i) {
            u64 expected = mulmod(mulmod(a[i], b[i], job.mod), rinv, job.mod);
            if (out[i] != expected) {
                std::cerr << "mont mismatch n=" << n << " i=" << i << "\n";
                return false;
            }
        }
        std::vector<u64> values = {1, 2, 9, 10, 14, 99, 100, 999, 1000, KNOWN[n]};
        check_cuda(cudaMemcpy(da, values.data(), values.size() * sizeof(u64), cudaMemcpyHostToDevice), "values");
        concat_test_kernel<<<1, 32>>>(job, da, dout, values.size());
        check_cuda(cudaGetLastError(), "concat launch");
        check_cuda(cudaMemcpy(out.data(), dout, values.size() * sizeof(u64), cudaMemcpyDeviceToHost), "concat result");
        for (int i = 0; i < (int)values.size(); ++i) {
            if (out[i] != concat_mod_host(values[i], job.mod)) {
                std::cerr << "concat mismatch n=" << n << " k=" << values[i] << " gpu=" << out[i]
                          << " host=" << concat_mod_host(values[i], job.mod) << "\n";
                return false;
            }
        }
        cudaFree(da); cudaFree(db); cudaFree(dout);
        if (concat_mod_host(KNOWN[n], modulus) != 0) {
            std::cerr << "published value fails divisibility n=" << n << "\n";
            return false;
        }
        std::cout << "SELFTEST_OK n=" << n << " mod=" << job.mod << "\n";
    }
    return true;
}

static int run_device(int device) {
    check_cuda(cudaSetDevice(device), "set device");
    std::filesystem::create_directories("oeis_results");
    u64* d_hits;
    int* d_count;
    check_cuda(cudaMalloc(&d_hits, 65536 * sizeof(u64)), "malloc hits");
    check_cuda(cudaMalloc(&d_count, sizeof(int)), "malloc count");
    for (int n = 2; n <= 13; ++n) {
        u64 modulus = literal(n);
        u64 full_remainder = concat_mod_host(KNOWN[n], modulus);
        if (full_remainder != 0) {
            std::cerr << "DIVISIBILITY_FAIL n=" << n << " remainder=" << full_remainder << "\n";
            return 3;
        }
        std::cout << "DIVISIBILITY_OK n=" << n << " known=" << KNOWN[n] << "\n";
        u64 lo = n + 1, hi = KNOWN[n];
        u64 total = hi - lo;
        u64 begin = lo + total * device / 8;
        u64 end = lo + total * (device + 1) / 8;
        Job job = make_job(modulus);
        job.k_end = end;
        u64 raw_hits = 0, early_hits = 0;
        u64 next = load_state(n, device, begin, end, job.mod, raw_hits, early_hits);
        save_state(n, device, begin, end, job.mod, next, raw_hits, early_hits);
        auto started = std::chrono::steady_clock::now();
        while (next < end) {
            int blocks = (int)std::min<u64>(BLOCKS, (end - next + CHUNK - 1) / CHUNK);
            check_cuda(cudaMemset(d_count, 0, sizeof(int)), "memset");
            scan_kernel<<<blocks, 256>>>(job, next, d_hits, d_count, 65536);
            check_cuda(cudaGetLastError(), "scan launch");
            int count = 0;
            check_cuda(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost), "count");
            if (count > 65536) throw std::runtime_error("hit overflow");
            u64 launched = std::min(end - next, (u64)blocks * 256 * CHUNK);
            next += launched;
            if (count) {
                std::vector<u64> hits(count);
                check_cuda(cudaMemcpy(hits.data(), d_hits, count * sizeof(u64), cudaMemcpyDeviceToHost), "hits");
                std::sort(hits.begin(), hits.end());
                std::ofstream early(hit_path(n, device), std::ios::app);
                for (u64 k : hits) {
                    ++raw_hits;
                    u64 rem = concat_mod_host(k, modulus);
                    if (rem == 0) {
                        ++early_hits;
                        early << "n=" << n << " k=" << k << " remainder=0\n";
                        early.flush();
                        std::cout << "EARLY_SOLUTION n=" << n << " k=" << k << "\n";
                    }
                }
            }
            save_state(n, device, begin, end, job.mod, next, raw_hits, early_hits);
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::cout << "PROGRESS n=" << n << " device=" << device << " next=" << next << "/" << end
                      << " raw_hits=" << raw_hits << " early_hits=" << early_hits << " elapsed=" << elapsed << "\n";
        }
        std::cout << "SHARD_DONE n=" << n << " device=" << device << " begin=" << begin
                  << " end=" << end << " raw_hits=" << raw_hits << " early_hits=" << early_hits << "\n";
    }
    cudaFree(d_hits); cudaFree(d_count);
    std::cout << "DEVICE_DONE device=" << device << "\n";
    return 0;
}

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "selftest") return selftest(std::stoi(argv[2])) ? 0 : 1;
        if (argc == 3 && std::string(argv[1]) == "run") return run_device(std::stoi(argv[2]));
        std::cerr << "usage: verify_oeis selftest device | run device\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
