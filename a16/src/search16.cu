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
using u32 = unsigned int;

static constexpr u64 P1 = 1231026625769ULL;
static constexpr u64 P2 = 2507191691ULL;
static constexpr u64 L = 153878328221ULL;
static constexpr u64 PINV1 = 1489788291605398695ULL;
static constexpr int BLOCK_SIZE = 256;
static constexpr int TABLE_BITS = 65536;
static constexpr int MAX_HITS = 65536;

struct Wide {
    u64 hi, lo;
};

struct Hit {
    u64 x, t;
};

struct Job {
    int d;
    u64 z0, g, ws, w0, gm, r2;
    u64 z2, g2, ws2, w2, W2, h2;
    Wide jlo, jhi;
    u64 x_end, chunk;
    const u32* tables;
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
    u64 m = lo * PINV1;
    u64 mhi = __umul64hi(m, P1);
    u64 r = hi + mhi + (lo != 0);
    u64 reduced = r - P1;
    return (long long)reduced < 0 ? r : reduced;
}

__host__ __device__ __forceinline__ u64 addmod1(u64 a, u64 b) {
    u64 r = a + b;
    u64 reduced = r - P1;
    return (long long)reduced < 0 ? r : reduced;
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

__device__ __forceinline__ u64 pow2(u64 a, u64 e) {
    u64 result = 1;
    while (e) {
        if (e & 1) result = result * a % P2;
        a = a * a % P2;
        e >>= 1;
    }
    return result;
}

__device__ __forceinline__ u64 table_pow(const u32* tables, u64 exponent) {
    u64 result = tables[exponent & 65535];
    result = result * tables[TABLE_BITS + ((exponent >> 16) & 65535)] % P2;
    result = result * tables[2 * TABLE_BITS + ((exponent >> 32) & 65535)] % P2;
    result = result * tables[3 * TABLE_BITS + (exponent >> 48)] % P2;
    return result;
}

__global__ void scan_kernel(Job job, u64 begin, u64* pre_count, int* hit_count, Hit* hits) {
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 x0 = begin + tid * job.chunk;
    if (x0 >= job.x_end) return;
    u64 x1 = min(x0 + job.chunk, job.x_end);
    u64 z = mul_standard(job.z0, mont_pow_standard(job.g, x0, job.r2), job.r2);
    u64 w = addmod1(mul_standard(job.ws, x0 % P1, job.r2), job.w0);
    u64 z2 = job.z2 * pow2(job.g2, x0) % P2;
    u64 w2 = (job.w2 + job.ws2 * (x0 % P2)) % P2;
    u64 local_pre = 0;
    for (u64 x = x0; x < x1; ++x) {
        u64 first_t = addmod1(z, w);
        for (u64 t = first_t;; t += P1) {
            Wide j = make_wide(x, L, t);
            if (wide_less(job.jhi, j)) break;
            if (!wide_less(j, job.jlo)) {
                ++local_pre;
                u64 remainder = (z2 * table_pow(job.tables, t) % P2 + w2 + job.W2 * (t % P2)) % P2;
                if (remainder == 0) {
                    int position = atomicAdd(hit_count, 1);
                    if (position < MAX_HITS) hits[position] = {x, t};
                }
            }
            if (t > ~0ULL - P1) break;
        }
        z = mont(z, job.gm);
        w = addmod1(w, job.ws);
        z2 = z2 * job.g2 % P2;
        w2 = (w2 + job.ws2) % P2;
    }
    atomicAdd(pre_count, local_pre);
}

__global__ void mont_test_kernel(const u64* a, const u64* b, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = mont(a[i], b[i]);
}

__global__ void table_test_kernel(const u32* tables, const u64* exponents, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = table_pow(tables, exponents[i]);
}

static void check_cuda(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(error) << "\n";
        std::exit(2);
    }
}

static u64 addmod(u64 a, u64 b, u64 p) {
    return (u64)(((u128)a + b) % p);
}

static u64 submod(u64 a, u64 b, u64 p) {
    return a >= b ? a - b : p - (b - a);
}

static u64 mulmod(u64 a, u64 b, u64 p) {
    return (u64)((u128)a * b % p);
}

static u64 powmod(u64 a, u128 exponent, u64 p) {
    u64 result = 1;
    while (exponent) {
        if (exponent & 1) result = mulmod(result, a, p);
        a = mulmod(a, a, p);
        exponent >>= 1;
    }
    return result;
}

static u64 invmod(u64 a, u64 p) {
    return powmod(a, p - 2, p);
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
        u64 U = mulmod(submod(remainder, addmod(mulmod(A, previous % p, p), B, p), p),
                       powmod(invmod(q, p), previous, p), p);
        remainder = addmod(mulmod(U, powmod(q, end, p), p), addmod(mulmod(A, end % p, p), B, p), p);
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
        u64 numerator = addmod(submod(powmod(q, length + 1, p), mulmod(lm, q, p), p), lm, p);
        numerator = submod(numerator, q, p);
        u64 s1 = mulmod(numerator, invmod(mulmod(denominator, denominator, p), p), p);
        remainder = addmod(mulmod(remainder, qt, p),
                           addmod(mulmod((start - 1) % p, s0, p), s1, p), p);
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

static std::vector<u32> make_tables(u64 h) {
    u64 bases[4];
    bases[0] = h;
    for (int level = 1; level < 4; ++level) bases[level] = powmod(bases[level - 1], 65536, P2);
    std::vector<u32> tables(4 * TABLE_BITS);
    for (int level = 0; level < 4; ++level) {
        u64 value = 1;
        for (int i = 0; i < TABLE_BITS; ++i) {
            tables[level * TABLE_BITS + i] = (u32)value;
            value = value * bases[level] % P2;
        }
    }
    return tables;
}

static Job make_job(int d, u64 x_end, u64 chunk, const u32* tables) {
    if (d < 2 || d > 28) throw std::runtime_error("this build supports digits 2..28; digit 29 uses a different period");
    u128 start = 1;
    for (int i = 1; i < d; ++i) start *= 10;
    u128 lo = std::max<u128>(start, 17), hi = start * 10 - 1;
    u64 q1 = powmod(10, d, P1);
    u64 A1 = invmod(submod(1, q1, P1), P1);
    u64 B1 = submod(0, mulmod(q1, mulmod(A1, A1, P1), P1), P1);
    u64 previous_remainder = concat_mod(start - 1, P1);
    u64 U1 = mulmod(submod(previous_remainder,
                           addmod(mulmod(A1, (start - 1) % P1, P1), B1, P1), P1),
                    powmod(invmod(q1, P1), start - 1, P1), P1);
    u64 denominator_inverse = invmod(mulmod(mulmod(A1, 4, P1), L, P1), P1);
    u64 q2 = powmod(10, d, P2);
    u64 A2 = invmod(submod(1, q2, P2), P2);
    u64 B2 = submod(0, mulmod(q2, mulmod(A2, A2, P2), P2), P2);
    u64 U2 = mulmod(submod(concat_mod(start - 1, P2),
                           addmod(mulmod(A2, (start - 1) % P2, P2), B2, P2), P2),
                    powmod(invmod(q2, P2), start - 1, P2), P2);
    u64 R = (u64)(((u128)1 << 64) % P1);
    Job job{};
    job.d = d;
    job.z0 = submod(0, mulmod(U1, denominator_inverse, P1), P1);
    job.g = powmod(q1, 4, P1);
    job.ws = submod(0, invmod(L, P1), P1);
    job.w0 = submod(0, mulmod(B1, denominator_inverse, P1), P1);
    job.gm = mulmod(job.g, R, P1);
    job.r2 = mulmod(R, R, P1);
    job.z2 = U2;
    job.g2 = powmod(q2, 4, P2);
    job.ws2 = mulmod(A2, 4, P2);
    job.w2 = B2;
    job.W2 = mulmod(job.ws2, L % P2, P2);
    job.h2 = powmod(job.g2, L, P2);
    job.jlo = from_u128((lo + 3) / 4);
    job.jhi = from_u128(hi / 4);
    job.x_end = x_end;
    job.chunk = chunk;
    job.tables = tables;
    return job;
}

static std::string state_path(const std::string& name) {
    return "results/" + name + ".state";
}

static std::string hits_path(const std::string& name) {
    return "results/" + name + ".candidates";
}

struct State {
    u64 next, pre, hits;
};

static void save_state(const std::string& name, int d, u64 begin, u64 end, State state, u64 chunk, int blocks) {
    std::string temporary = state_path(name) + ".tmp";
    std::ofstream output(temporary);
    output << "config=" << d << ":" << begin << ":" << end << "\n"
           << "next_x=" << state.next << "\npre_hits=" << state.pre << "\nhits=" << state.hits
           << "\nchunk=" << chunk << "\nblocks=" << blocks << "\n";
    output.close();
    if (!output) throw std::runtime_error("failed to save state");
    std::filesystem::rename(temporary, state_path(name));
}

static State load_state(const std::string& name, int d, u64 begin, u64 end) {
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
    if (config != std::to_string(d) + ":" + std::to_string(begin) + ":" + std::to_string(end) ||
        result.next < begin || result.next > end) {
        throw std::runtime_error("state mismatch");
    }
    return result;
}

static bool selftest(int device) {
    check_cuda(cudaSetDevice(device), "set device");
    const int n = 100000;
    std::mt19937_64 rng(40066616);
    std::vector<u64> a(n), b(n), out(n), exponents(n);
    for (int i = 0; i < n; ++i) {
        a[i] = rng() % P1;
        b[i] = rng() % P1;
        exponents[i] = rng();
    }
    u64 *da, *db, *dout, *dt;
    check_cuda(cudaMalloc(&da, n * sizeof(u64)), "malloc a");
    check_cuda(cudaMalloc(&db, n * sizeof(u64)), "malloc b");
    check_cuda(cudaMalloc(&dout, n * sizeof(u64)), "malloc out");
    check_cuda(cudaMalloc(&dt, n * sizeof(u64)), "malloc exponents");
    check_cuda(cudaMemcpy(da, a.data(), a.size() * sizeof(u64), cudaMemcpyHostToDevice), "copy a");
    check_cuda(cudaMemcpy(db, b.data(), b.size() * sizeof(u64), cudaMemcpyHostToDevice), "copy b");
    check_cuda(cudaMemcpy(dt, exponents.data(), exponents.size() * sizeof(u64), cudaMemcpyHostToDevice), "copy exponents");
    mont_test_kernel<<<(n + 255) / 256, 256>>>(da, db, dout, n);
    check_cuda(cudaGetLastError(), "Montgomery test launch");
    check_cuda(cudaMemcpy(out.data(), dout, out.size() * sizeof(u64), cudaMemcpyDeviceToHost), "Montgomery result");
    u64 R = (u64)(((u128)1 << 64) % P1), rinv = invmod(R, P1);
    for (int i = 0; i < n; ++i) {
        if (out[i] != mulmod(mulmod(a[i], b[i], P1), rinv, P1)) {
            std::cerr << "Montgomery mismatch at " << i << "\n";
            return false;
        }
    }
    Job job = make_job(18, 1000, 65536, nullptr);
    std::vector<u32> tables = make_tables(job.h2);
    u32* device_tables;
    check_cuda(cudaMalloc(&device_tables, tables.size() * sizeof(u32)), "malloc tables");
    check_cuda(cudaMemcpy(device_tables, tables.data(), tables.size() * sizeof(u32), cudaMemcpyHostToDevice), "copy tables");
    table_test_kernel<<<(n + 255) / 256, 256>>>(device_tables, dt, dout, n);
    check_cuda(cudaGetLastError(), "table test launch");
    check_cuda(cudaMemcpy(out.data(), dout, out.size() * sizeof(u64), cudaMemcpyDeviceToHost), "table result");
    for (int i = 0; i < n; ++i) {
        if (out[i] != powmod(job.h2, exponents[i], P2)) {
            std::cerr << "table mismatch at " << i << "\n";
            return false;
        }
    }
    for (u64 k = 1; k <= 1000; ++k) {
        for (u64 p : {P1, P2}) {
            if (concat_mod(k, p) != concat_mod_sum(k, p)) {
                std::cerr << "small concatenation mismatch at " << k << "\n";
                return false;
            }
        }
    }
    for (int i = 0; i < 100; ++i) {
        u128 k = (u128)(rng() % 1000000000000ULL) * 10000000000ULL + rng() % 10000000000ULL;
        for (u64 p : {P1, P2}) {
            if (concat_mod(k, p) != concat_mod_sum(k, p)) {
                std::cerr << "large concatenation mismatch at " << u128_string(k) << "\n";
                return false;
            }
        }
    }
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    cudaFree(dt);
    cudaFree(device_tables);
    std::cout << "SELFTEST_OK montgomery=100000 table_pow=100000 concat_small=1000 concat_large=100\n";
    return true;
}

static int run(const std::string& name, int d, int device, u64 begin, u64 end, u64 chunk, int blocks) {
    check_cuda(cudaSetDevice(device), "set run device");
    u64 R = (u64)(((u128)1 << 64) % P1);
    (void)R;
    std::vector<u32> tables;
    {
        Job initial = make_job(d, end, chunk, nullptr);
        tables = make_tables(initial.h2);
    }
    u32* device_tables;
    check_cuda(cudaMalloc(&device_tables, tables.size() * sizeof(u32)), "malloc run tables");
    check_cuda(cudaMemcpy(device_tables, tables.data(), tables.size() * sizeof(u32), cudaMemcpyHostToDevice), "copy run tables");
    Job job = make_job(d, end, chunk, device_tables);
    Hit* device_hits;
    int* device_count;
    u64* device_pre;
    check_cuda(cudaMalloc(&device_hits, MAX_HITS * sizeof(Hit)), "malloc hits");
    check_cuda(cudaMalloc(&device_count, sizeof(int)), "malloc count");
    check_cuda(cudaMalloc(&device_pre, sizeof(u64)), "malloc pre count");
    State state = load_state(name, d, begin, end);
    save_state(name, d, begin, end, state, chunk, blocks);
    auto started = std::chrono::steady_clock::now();
    auto last = started;
    while (state.next < end) {
        u64 launch_end = std::min(end, state.next + (u64)blocks * BLOCK_SIZE * chunk);
        check_cuda(cudaMemset(device_count, 0, sizeof(int)), "clear count");
        check_cuda(cudaMemset(device_pre, 0, sizeof(u64)), "clear pre count");
        scan_kernel<<<blocks, BLOCK_SIZE>>>(job, state.next, device_pre, device_count, device_hits);
        check_cuda(cudaGetLastError(), "scan launch");
        int count = 0;
        u64 pre = 0;
        check_cuda(cudaMemcpy(&count, device_count, sizeof(int), cudaMemcpyDeviceToHost), "copy count");
        check_cuda(cudaMemcpy(&pre, device_pre, sizeof(u64), cudaMemcpyDeviceToHost), "copy pre count");
        if (count > MAX_HITS) throw std::runtime_error("candidate buffer overflow");
        u64 launched = launch_end - state.next;
        state.next = launch_end;
        state.pre += pre;
        if (count) {
            std::vector<Hit> hits(count);
            check_cuda(cudaMemcpy(hits.data(), device_hits, count * sizeof(Hit), cudaMemcpyDeviceToHost), "copy hits");
            std::ofstream output(hits_path(name), std::ios::app);
            for (const Hit& hit : hits) {
                u128 k = 4 * ((u128)hit.x + (u128)L * hit.t);
                if (k % 4 || k <= 16 || concat_mod(k, P1) != 0 || concat_mod(k, P2) != 0) {
                    std::cerr << "false candidate k=" << u128_string(k) << "\n";
                    return 3;
                }
                output << "k=" << u128_string(k) << " d=" << d << " x=" << hit.x << " t=" << hit.t
                       << " r_p1=0 r_p2=0\n";
            }
            output.flush();
            state.hits += count;
        }
        save_state(name, d, begin, end, state, chunk, blocks);
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - started).count();
        double dt = std::chrono::duration<double>(now - last).count();
        last = now;
        std::cout << "progress name=" << name << " next_x=" << state.next << "/" << end
                  << " rate=" << launched / dt << " elapsed=" << elapsed
                  << " pre=" << state.pre << " hits=" << state.hits << "\n";
    }
    std::cout << "DONE name=" << name << " next_x=" << end << " pre=" << state.pre << " hits=" << state.hits << "\n";
    cudaFree(device_hits);
    cudaFree(device_count);
    cudaFree(device_pre);
    cudaFree(device_tables);
    return 0;
}

int main(int argc, char** argv) {
    try {
        std::filesystem::create_directories("results");
        if (argc == 3 && std::string(argv[1]) == "selftest") {
            return selftest(std::stoi(argv[2])) ? 0 : 1;
        }
        if (argc == 9 && std::string(argv[1]) == "run") {
            return run(argv[2], std::stoi(argv[3]), std::stoi(argv[4]), std::stoull(argv[5]),
                       std::stoull(argv[6]), std::stoull(argv[7]), std::stoi(argv[8]));
        }
        std::cerr << "usage: search16 selftest device | run name d device x_begin x_end chunk blocks\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
}
