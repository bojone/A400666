#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using u64 = unsigned long long;
using u128 = unsigned __int128;

static constexpr u64 P = 205761315168520219ULL;
static constexpr u64 PINV = 8574230492775869933ULL;

__device__ __forceinline__ u64 mont_mul(u64 a, u64 b) {
    u64 lo = a * b;
    u64 hi = __umul64hi(a, b);
    u64 m = lo * 8574230492775869933ULL;
    u64 mhi = __umul64hi(m, 205761315168520219ULL);
    u64 r = hi + mhi + (lo != 0);
    if (r >= 205761315168520219ULL) r -= 205761315168520219ULL;
    return r;
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

struct Job {
    u64 z0, g, w0, ws, r2;
    u64 x_end, chunk;
    u64 period, step, offset;
    u64 j_min, j_max, t_max;
    int period_mode;
};

struct Hit {
    u64 x, t;
};

__global__ void scan_kernel(Job job, u64 launch_begin, Hit* hits, int* hit_count, int max_hits) {
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 x0 = launch_begin + tid * job.chunk;
    if (x0 >= job.x_end) return;
    u64 x1 = min(x0 + job.chunk, job.x_end);
    u64 gx = mont_pow_standard(job.g, x0, job.r2);
    u64 z = mul_standard(job.z0, gx, job.r2);
    u64 w = mul_standard(job.ws, x0 % 205761315168520219ULL, job.r2) + job.w0;
    if (w >= 205761315168520219ULL) w -= 205761315168520219ULL;
    u64 gm = mont_mul(job.g, job.r2);
    for (u64 x = x0; x < x1; ++x) {
        u64 t = z + w;
        if (t >= 205761315168520219ULL) t -= 205761315168520219ULL;
        bool match = job.period_mode ? (t <= job.t_max) : (t == 0);
        if (match) {
            u128 j = job.period_mode ? (u128)x + (u128)job.period * t : (u128)x;
            if (j >= job.j_min && j <= job.j_max) {
                u128 k128 = (u128)job.offset + (u128)job.step * j;
                if (k128 <= ~0ULL) {
                    u64 k = (u64)k128;
                    if (k % 6 == 0 || k % 6 == 2) {
                        int pos = atomicAdd(hit_count, 1);
                        if (pos < max_hits) hits[pos] = {x, t};
                    }
                }
            }
        }
        z = mont_mul(z, gm);
        w += job.ws;
        if (w >= 205761315168520219ULL) w -= 205761315168520219ULL;
    }
}

__global__ void mont_test_kernel(const u64* a, const u64* b, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = mont_mul(a[i], b[i]);
}

static void check_cuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(e) << "\n";
        std::exit(2);
    }
}

static u64 addmod(u64 a, u64 b, u64 m = P) {
    u128 x = (u128)a + b;
    return (u64)(x >= m ? x - m : x);
}

static u64 mulmod(u64 a, u64 b, u64 m = P) {
    return (u64)((u128)a * b % m);
}

static u64 powmod(u64 a, u64 e, u64 m = P) {
    u64 r = 1 % m;
    while (e) {
        if (e & 1) r = mulmod(r, a, m);
        a = mulmod(a, a, m);
        e >>= 1;
    }
    return r;
}

static u64 invmod(u64 a, u64 m = P) {
    return powmod(a, m - 2, m);
}

static u64 submod(u64 a, u64 b, u64 m = P) {
    return a >= b ? a - b : addmod(a, m - b, m);
}

static u64 digits10(u64 k) {
    u64 d = 1, p = 10;
    while (k >= p && d < 19) {
        ++d;
        if (p > ~0ULL / 10) break;
        p *= 10;
    }
    return d;
}

static u64 concat_mod(u64 k) {
    u64 r = 0;
    for (u64 d = 1, s = 1; d <= 19 && s <= k; ++d) {
        u128 next = (u128)s * 10;
        u64 end = next - 1 > k ? k : (u64)(next - 1);
        u64 q = powmod(10, d);
        u64 A = invmod(submod(1, q));
        u64 B = submod(0, mulmod(q, mulmod(A, A)));
        u64 prev = s - 1;
        u64 U = mulmod(submod(r, addmod(mulmod(A, prev % P), B)), powmod(invmod(q), prev));
        r = addmod(mulmod(U, powmod(q, end)), addmod(mulmod(A, end % P), B));
        if (end == k) break;
        s = (u64)next;
    }
    return r;
}

static u64 concat_mod_sum(u64 k) {
    u64 r = 0;
    for (u64 d = 1, s = 1; d <= 19 && s <= k; ++d) {
        u128 next = (u128)s * 10;
        u64 end = next - 1 > k ? k : (u64)(next - 1);
        u64 t = end - s + 1;
        u64 q = powmod(10, d);
        u64 qt = powmod(q, t);
        u64 den = submod(q, 1);
        u64 S0 = mulmod(submod(qt, 1), invmod(den));
        u64 numerator = addmod(submod(powmod(q, t + 1), mulmod(t % P, q)), t % P);
        numerator = submod(numerator, q);
        u64 S1 = mulmod(numerator, invmod(mulmod(den, den)));
        r = addmod(mulmod(r, qt), addmod(mulmod((s - 1) % P, S0), S1));
        if (end == k) break;
        s = (u64)next;
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

static u64 parse_u64(const std::string& s) {
    return std::stoull(s);
}

static bool selftest(int device) {
    check_cuda(cudaSetDevice(device), "set device");
    const int n = 100000;
    std::mt19937_64 rng(0x400666);
    std::vector<u64> a(n), b(n), out(n);
    u64 *da, *db, *dout;
    check_cuda(cudaMalloc(&da, n * sizeof(u64)), "malloc a");
    check_cuda(cudaMalloc(&db, n * sizeof(u64)), "malloc b");
    check_cuda(cudaMalloc(&dout, n * sizeof(u64)), "malloc out");
    for (int i = 0; i < n; ++i) {
        a[i] = rng() % P;
        b[i] = rng() % P;
        if (i < 16) a[i] = b[i] = i;
        if (i == 16) a[i] = b[i] = P - 1;
    }
    check_cuda(cudaMemcpy(da, a.data(), a.size() * sizeof(u64), cudaMemcpyHostToDevice), "copy a");
    check_cuda(cudaMemcpy(db, b.data(), b.size() * sizeof(u64), cudaMemcpyHostToDevice), "copy b");
    mont_test_kernel<<<(n + 255) / 256, 256>>>(da, db, dout, n);
    check_cuda(cudaGetLastError(), "mont test launch");
    check_cuda(cudaMemcpy(out.data(), dout, out.size() * sizeof(u64), cudaMemcpyDeviceToHost), "copy out");
    cudaFree(da); cudaFree(db); cudaFree(dout);
    u64 R = (u64)(((u128)1 << 64) % P);
    u64 rinv = invmod(R);
    for (int i = 0; i < n; ++i) {
        u64 expected = mulmod(mulmod(a[i], b[i]), rinv);
        if (out[i] != expected) {
            std::cerr << "Montgomery mismatch at " << i << ": " << out[i] << " != " << expected << "\n";
            return false;
        }
    }
    u64 slow = 0;
    for (u64 k = 1; k <= 1000; ++k) {
        slow = addmod(mulmod(slow, powmod(10, digits10(k))), k % P);
        if (slow != concat_mod(k) || slow != concat_mod_sum(k)) {
            std::cerr << "concat mismatch at " << k << "\n";
            return false;
        }
    }
    std::vector<u64> special = {14, 999999999999999ULL, 1000000000000000ULL,
        9999999999999999ULL, 10000000000000000ULL, 99999999999999999ULL,
        100000000000000000ULL, 999999999999999999ULL, 1000000000000000000ULL,
        9999999999999999999ULL};
    for (int i = 0; i < 100; ++i) special.push_back(rng() % 10000000000000000000ULL);
    for (u64 k : special) {
        if (concat_mod(k) != concat_mod_sum(k)) {
            std::cerr << "independent large-k concat mismatch at " << k << "\n";
            return false;
        }
    }
    if (concat_mod(14) != 0) {
        std::cerr << "c(14) is not zero modulo p\n";
        return false;
    }
    std::cout << "selftest ok: 100000 Montgomery products, concat through 1000, 110 large-k cross-checks\n";
    return true;
}

struct RunArgs {
    std::string name;
    int device;
    int period_mode;
    u64 d, offset, step, period;
    u64 x_begin, x_end, chunk, blocks;
    u64 k_start, k_end;
};

static std::string config_key(const RunArgs& a) {
    return std::to_string(a.period_mode) + ":" + std::to_string(a.d) + ":" +
        std::to_string(a.offset) + ":" + std::to_string(a.step) + ":" + std::to_string(a.period) + ":" +
        std::to_string(a.x_begin) + ":" + std::to_string(a.x_end) + ":" +
        std::to_string(a.k_start) + ":" + std::to_string(a.k_end);
}

static Job make_job(const RunArgs& args) {
    if (args.d < 1 || args.d > 19) throw std::runtime_error("d must be 1..19");
    if (!args.step || args.offset >= args.step) throw std::runtime_error("require 0 <= offset < step");
    if (args.period_mode && !args.period) throw std::runtime_error("period mode requires nonzero period");
    u64 s = 1;
    for (u64 i = 1; i < args.d; ++i) s *= 10;
    u64 hi = args.d == 19 ? 10000000000000000000ULL - 1 : s * 10 - 1;
    s = std::max(s, args.k_start);
    hi = std::min(hi, args.k_end);
    if (s > hi) throw std::runtime_error("empty k interval");
    u64 q = powmod(10, args.d);
    u64 A = invmod(submod(1, q));
    u64 B = submod(0, mulmod(q, mulmod(A, A)));
    u64 prev = s - 1;
    u64 rprev = concat_mod(prev);
    u64 U = mulmod(submod(rprev, addmod(mulmod(A, prev % P), B)), powmod(invmod(q), prev));
    u64 g = powmod(q, args.step);
    u64 z0 = mulmod(U, powmod(q, args.offset));
    u64 w0 = addmod(mulmod(A, args.offset), B);
    u64 ws = mulmod(A, args.step);
    u64 j_min = s <= args.offset ? 0 : (s - args.offset + args.step - 1) / args.step;
    if (hi < args.offset) throw std::runtime_error("empty residue class");
    u64 j_max = (hi - args.offset) / args.step;
    u64 t_max = 0;
    if (args.period_mode) {
        u64 deninv = invmod(mulmod(ws, args.period % P));
        z0 = submod(0, mulmod(z0, deninv));
        w0 = submod(0, mulmod(w0, deninv));
        ws = submod(0, invmod(args.period % P));
        t_max = j_max / args.period;
    }
    Job job{};
    job.z0 = z0; job.g = g; job.w0 = w0; job.ws = ws;
    u64 R = (u64)(((u128)1 << 64) % P);
    job.r2 = mulmod(R, R);
    job.x_end = args.x_end; job.chunk = args.chunk;
    job.period = args.period; job.step = args.step; job.offset = args.offset;
    job.j_min = j_min; job.j_max = j_max; job.t_max = t_max;
    job.period_mode = args.period_mode;
    return job;
}

static std::string state_path(const std::string& name) { return "results/" + name + ".state"; }
static std::string hits_path(const std::string& name) { return "results/" + name + ".candidates"; }

static void save_state(const RunArgs& args, u64 next_x, u64 hits) {
    std::filesystem::create_directories("results");
    std::string tmp = state_path(args.name) + ".tmp";
    std::ofstream f(tmp);
    f << "config=" << config_key(args) << "\nnext_x=" << next_x << "\nhits=" << hits
      << "\nchunk=" << args.chunk << "\nblocks=" << args.blocks << "\n";
    f.close();
    std::filesystem::rename(tmp, state_path(args.name));
}

static std::pair<u64, u64> load_state(const RunArgs& args) {
    std::ifstream f(state_path(args.name));
    if (!f) return {args.x_begin, 0};
    std::string line, key;
    u64 next = args.x_begin, hits = 0;
    while (std::getline(f, line)) {
        if (line.rfind("config=", 0) == 0) key = line.substr(7);
        if (line.rfind("next_x=", 0) == 0) next = parse_u64(line.substr(7));
        if (line.rfind("hits=", 0) == 0) hits = parse_u64(line.substr(5));
    }
    if (key != config_key(args)) throw std::runtime_error("state config mismatch for " + args.name);
    if (next < args.x_begin || next > args.x_end) throw std::runtime_error("invalid saved next_x");
    return {next, hits};
}

static void append_hits(const RunArgs& args, const std::vector<Hit>& hits, u64& total_hits) {
    if (hits.empty()) return;
    std::ofstream f(hits_path(args.name), std::ios::app);
    for (const Hit& hit : hits) {
        u128 j = args.period_mode ? (u128)hit.x + (u128)args.period * hit.t : (u128)hit.x;
        u128 k = (u128)args.offset + (u128)args.step * j;
        if (k > ~0ULL) continue;
        u64 kk = (u64)k;
        u64 rem = concat_mod(kk);
        if (rem != 0 || concat_mod_sum(kk) != 0) {
            std::cerr << "Rejected false hit k=" << kk << "\n";
            std::exit(3);
        }
        f << "k=" << u128str(k) << " x=" << hit.x << " t=" << hit.t
          << " concat_mod_p=0 independent_mod_p=0 k_mod_6=" << kk % 6 << "\n";
        f.flush();
        ++total_hits;
        std::cout << "HIT k=" << kk << " x=" << hit.x << " t=" << hit.t << "\n";
    }
}

static int run(const RunArgs& args) {
    check_cuda(cudaSetDevice(args.device), "set device");
    Job job = make_job(args);
    auto [next_x, total_hits] = load_state(args);
    std::filesystem::create_directories("results");
    int max_hits = 65536;
    Hit* d_hits;
    int* d_count;
    check_cuda(cudaMalloc(&d_hits, max_hits * sizeof(Hit)), "malloc hits");
    check_cuda(cudaMalloc(&d_count, sizeof(int)), "malloc count");
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "stream");
    save_state(args, next_x, total_hits);
    auto start = std::chrono::steady_clock::now();
    auto last = start;
    while (next_x < args.x_end) {
        int blocks = (int)std::min<u64>(args.blocks, (args.x_end - next_x + job.chunk - 1) / job.chunk);
        if (blocks <= 0) break;
        check_cuda(cudaMemsetAsync(d_count, 0, sizeof(int), stream), "memset count");
        scan_kernel<<<blocks, 256, 0, stream>>>(job, next_x, d_hits, d_count, max_hits);
        check_cuda(cudaGetLastError(), "scan launch");
        int count = 0;
        check_cuda(cudaMemcpyAsync(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream), "copy count");
        check_cuda(cudaStreamSynchronize(stream), "scan sync");
        u64 launched = std::min(args.x_end - next_x, (u64)blocks * 256 * job.chunk);
        next_x += launched;
        if (count > 0) {
            if (count > max_hits) {
                std::cerr << "Hit buffer overflow\n";
                return 4;
            }
            std::vector<Hit> hits(count);
            check_cuda(cudaMemcpy(hits.data(), d_hits, count * sizeof(Hit), cudaMemcpyDeviceToHost), "copy hits");
            append_hits(args, hits, total_hits);
        }
        save_state(args, next_x, total_hits);
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last).count();
        double elapsed = std::chrono::duration<double>(now - start).count();
        u64 total_done = next_x - args.x_begin;
        double rate = dt > 0 ? launched / dt : 0;
        std::cout << "progress name=" << args.name << " next_x=" << next_x << "/" << args.x_end
                  << " fraction=" << (double)total_done / (args.x_end - args.x_begin)
                  << " rate=" << rate << " x/s elapsed=" << elapsed << " hits=" << total_hits << "\n";
        last = now;
    }
    cudaFree(d_hits); cudaFree(d_count); cudaStreamDestroy(stream);
    std::cout << "DONE name=" << args.name << " next_x=" << next_x << " hits=" << total_hits << "\n";
    return 0;
}

static void info() {
    int n = 0;
    check_cuda(cudaGetDeviceCount(&n), "device count");
    for (int i = 0; i < n; ++i) std::cout << "device " << i << "\n";
}

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && std::string(argv[1]) == "info") {
            info();
            return 0;
        }
        if (argc >= 2 && std::string(argv[1]) == "selftest") {
            return selftest(argc > 2 ? std::stoi(argv[2]) : 0) ? 0 : 1;
        }
        if (argc >= 2 && std::string(argv[1]) == "run") {
            if (argc != 15) {
                std::cerr << "usage: search run name device period_mode digits offset step period x_begin x_end chunk blocks k_start k_end\n";
                return 2;
            }
            RunArgs args;
            int i = 2;
            args.name = argv[i++];
            args.device = std::stoi(argv[i++]);
            args.period_mode = std::stoi(argv[i++]);
            args.d = parse_u64(argv[i++]);
            args.offset = parse_u64(argv[i++]);
            args.step = parse_u64(argv[i++]);
            args.period = parse_u64(argv[i++]);
            args.x_begin = parse_u64(argv[i++]);
            args.x_end = parse_u64(argv[i++]);
            args.chunk = parse_u64(argv[i++]);
            args.blocks = parse_u64(argv[i++]);
            args.k_start = parse_u64(argv[i++]);
            args.k_end = parse_u64(argv[i++]);
            return run(args);
        }
        std::cerr << "usage: search info | selftest [device] | run ...\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
