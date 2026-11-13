#include <set>

#define SEARCH15_PRUNE_LIBRARY
#include "search15_prune.cu"

using i128 = __int128;
using u32 = unsigned int;

struct PairParams {
    u64 lambda, lambda2, delta5, C, eta, Wc, Wm, ws18, B18, A18, U18, g18;
    u64 begin, end;
    u32 key1, key2;
};

struct PairHit {
    u64 source, index, value;
    u32 kind, pad;
};

static u64 inverse_period(u64 a, u64 modulus) {
    i128 old_r = modulus, r = a;
    i128 old_s = 0, s = 1;
    while (r) {
        u64 q = old_r / r;
        i128 next_r = old_r - q * r;
        old_r = r;
        r = next_r;
        i128 next_s = old_s - q * s;
        old_s = s;
        s = next_s;
    }
    if (old_r != 1) throw std::runtime_error("period inverse does not exist");
    old_s %= modulus;
    if (old_s < 0) old_s += modulus;
    return (u64)old_s;
}

static PairParams make_pair(const Job& job) {
    const u64 M = 2 * COSET_Q;
    PairParams cfg{};
    u64 inv27 = inverse_period(27, M);
    cfg.C = (u64)((u128)19 * inv27 % M);
    cfg.eta = (u64)((u128)9 * inv27 % M);
    u64 q = powmod(10, 18);
    cfg.A18 = invmod(submod(1, q));
    cfg.B18 = submod(0, mulmod(q, mulmod(cfg.A18, cfg.A18)));
    u128 start = 100000000000000000ULL;
    u64 U = mulmod(submod(concat_mod(start - 1), addmod(mulmod(cfg.A18, (start - 1) % P), cfg.B18)),
                   powmod(invmod(q), start - 1));
    cfg.U18 = U;
    cfg.g18 = powmod(q, 15);
    cfg.lambda = mulmod(U, invmod(job.z0));
    cfg.lambda2 = addmod(cfg.lambda, cfg.lambda);
    cfg.ws18 = mulmod(cfg.A18, 15);
    cfg.delta5 = submod(mulmod(cfg.A18, 5), mulmod(cfg.ws18, cfg.eta % P));
    cfg.Wc = mulmod(cfg.ws18, cfg.C % P);
    cfg.Wm = mulmod(cfg.ws18, M);
    cfg.begin = (start + 14) / 15;
    cfg.end = (start * 10 - 1) / 15 + 1;
    cfg.key1 = (u32)cfg.lambda & 0xffff;
    cfg.key2 = (u32)cfg.lambda2 & 0xffff;
    u64 D = (u64)((u128)54 * inverse_period(19, L) % L);
    if (powmod(job.g, D) != cfg.g18 || powmod(cfg.g18, cfg.eta) != powmod(q, 5))
        throw std::runtime_error("pair exponent mapping mismatch");
    return cfg;
}

__device__ u64 multiply_period(u64 a, u64 b, u64 modulus) {
    u64 hi = __umul64hi(a, b);
    u64 lo = a * b;
    u64 r = hi;
    for (int i = 0; i < 64; ++i) {
        r = r >= modulus - r ? r - (modulus - r) : r + r;
        if (lo >> 63) r = r == modulus - 1 ? 0 : r + 1;
        lo <<= 1;
    }
    return r;
}

__device__ __forceinline__ void pair_store(PairHit* hits, int* count, int max_hits,
                                           u32 kind, u64 source, u64 index, u64 value) {
    int pos = atomicAdd(count, 1);
    if (pos < max_hits) hits[pos] = {source, index, value, kind, 0};
}

__global__ void scan_pair18_kernel(Job job, CosetRange range, PairParams cfg,
                                   PairHit* hits, int* count, int max_hits) {
    const u64 M = 2 * COSET_Q;
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 r0 = range.begin + tid * job.chunk;
    if (r0 >= range.end) return;
    u64 r1 = min(r0 + job.chunk, range.end);
    u64 lambda = cfg.lambda;
    u64 z0 = mul_standard(job.z0, lambda, job.r2);
    u64 z = mul_standard(z0, mont_pow_standard(job.g, r0, job.r2), job.r2);
    u64 ws_scaled = mul_standard(job.ws, lambda, job.r2);
    u64 w = addmod(mul_standard(ws_scaled, r0, job.r2),
                   mul_standard(job.w0, lambda, job.r2));
    u64 gm = mont_mul(job.g, job.r2);
    u64 hm = mont_mul(range.h, job.r2);
    u64 wq = mul_standard(range.w_q, lambda, job.r2);
    u64 js[4], ws[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        u64 rr = r0 + ((r0 + u) & 1);
        u64 y = (rr + COSET_Q * u) / 2;
        js[u] = multiply_period(cfg.C, y, M);
        ws[u] = addmod(mul_standard(cfg.ws18, js[u], job.r2), cfg.B18);
    }
    for (u64 r = r0; r < r1; ++r) {
        u64 zh = mont_mul(z, hm);
        u64 w1 = addmod(w, wq), w2 = addmod(w1, wq), w3 = addmod(w2, wq);
        u64 values[4] = {addmod(z, w), addmod(zh, w1), subtract_mod(w2, z), subtract_mod(w3, zh)};
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            u64 v = values[u];
            #ifdef TEST_ALL_PAIR_HITS
            pair_store(hits, count, max_hits, 0, r + COSET_Q * u, ~0ULL, v);
            #else
            u32 key = (u32)v & 0xffff;
            if ((key == 0 || key == cfg.key1 || key == cfg.key2) &&
                (v == 0 || v == lambda || v == cfg.lambda2)) {
                u64 t = v == 0 ? 0 : v == lambda ? 1 : 2;
                pair_store(hits, count, max_hits, 0, r + COSET_Q * u, t, v);
            }
            #endif
            if (((r + u) & 1) == 0) {
                u64 j = js[u];
                u64 j5 = j >= cfg.eta ? j - cfg.eta : j + M - cfg.eta;
                u64 zu = u == 0 ? z : u == 1 ? zh : u == 2 ? negmod(z) : negmod(zh);
                #ifdef TEST_ALL_PAIR_HITS
                bool valid0 = true, valid5 = true;
                #else
                bool valid0 = j >= cfg.begin && j < cfg.end;
                bool valid5 = j5 >= cfg.begin && j5 < cfg.end;
                #endif
                if (valid0) {
                    u64 value = addmod(zu, ws[u]);
                    #ifdef TEST_ALL_PAIR_HITS
                    pair_store(hits, count, max_hits, 1, r + COSET_Q * u, j, value);
                    #else
                    if (value == 0) pair_store(hits, count, max_hits, 1, r + COSET_Q * u, j, value);
                    #endif
                }
                if (valid5) {
                    u64 w5 = addmod(ws[u], cfg.delta5);
                    if (j < cfg.eta) w5 = addmod(w5, cfg.Wm);
                    u64 value = addmod(zu, w5);
                    #ifdef TEST_ALL_PAIR_HITS
                    pair_store(hits, count, max_hits, 2, r + COSET_Q * u, j5, value);
                    #else
                    if (value == 0) pair_store(hits, count, max_hits, 2, r + COSET_Q * u, j5, value);
                    #endif
                }
                u64 next_j = j + cfg.C;
                bool wrapped = next_j >= M;
                if (wrapped) next_j -= M;
                u64 next_w = addmod(ws[u], cfg.Wc);
                if (wrapped) next_w = subtract_mod(next_w, cfg.Wm);
                js[u] = next_j;
                ws[u] = next_w;
            }
        }
        z = mont_mul(z, gm);
        w = addmod(w, ws_scaled);
    }
}

static bool pair_test(int device) {
    check_cuda(cudaSetDevice(device), "set pair test device");
    Job job = make_job(1, 64);
    PairParams cfg = make_pair(job);
    PairHit* hits;
    int* count;
    check_cuda(cudaMalloc(&hits, 4096 * sizeof(PairHit)), "malloc pair hits");
    check_cuda(cudaMalloc(&count, sizeof(int)), "malloc pair count");
    #ifdef TEST_ALL_PAIR_HITS
    int tested = 0;
    for (u64 begin : {0ULL, 1ULL, 17ULL, 123456789ULL, COSET_Q - 64}) {
        CosetRange range{};
        range.begin = begin;
        range.end = begin + 64;
        range.q = COSET_Q;
        range.h = powmod(job.g, COSET_Q);
        range.w_q = mulmod(job.ws, COSET_Q);
        check_cuda(cudaMemset(count, 0, sizeof(int)), "pair coverage memset");
        scan_pair18_kernel<<<1, BLOCK_SIZE>>>(job, range, cfg, hits, count, 4096);
        int observed = 0;
        check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "pair coverage count");
        if (observed != 512) {
            std::cerr << "pair coverage count mismatch " << observed << "\n";
            return std::cerr << "pair failure line " << __LINE__ << "\n", false;
        }
        std::vector<PairHit> actual(observed);
        check_cuda(cudaMemcpy(actual.data(), hits, observed * sizeof(PairHit), cudaMemcpyDeviceToHost), "pair coverage hits");
        std::set<u64> seen19;
        std::set<std::pair<u64, u64>> seen18;
        for (const PairHit& hit : actual) {
            u64 u = hit.source / COSET_Q;
            u64 r = hit.source % COSET_Q;
            if (u >= 4 || r < begin || r >= begin + 64) return std::cerr << "pair failure line " << __LINE__ << "\n", false;
            if (hit.kind == 0) {
                if (!seen19.insert(hit.source).second) return std::cerr << "pair failure line " << __LINE__ << "\n", false;
                u64 value = addmod(mulmod(job.z0, powmod(job.g, hit.source)),
                                   addmod(mulmod(job.ws, hit.source), job.w0));
                if (hit.value != mulmod(value, cfg.lambda)) return std::cerr << "pair failure line " << __LINE__ << "\n", false;
                continue;
            }
            if ((hit.source & 1) || !seen18.insert({hit.source, hit.kind}).second) return std::cerr << "pair failure line " << __LINE__ << "\n", false;
            u64 j = (u64)((u128)cfg.C * (hit.source / 2) % (2 * COSET_Q));
            u64 z = mulmod(cfg.U18, powmod(cfg.g18, j));
            if (hit.kind == 1) {
                if (hit.index != j) return std::cerr << "pair failure line " << __LINE__ << "\n", false;
                u64 expected = addmod(z, addmod(mulmod(cfg.ws18, j), cfg.B18));
                if (hit.value != expected) return std::cerr << "pair failure line " << __LINE__ << "\n", false;
                if (j >= cfg.begin && j < cfg.end && hit.value != concat_mod(15 * (u128)j)) return std::cerr << "pair failure line " << __LINE__ << "\n", false;
            } else if (hit.kind == 2) {
                u64 j5 = j >= cfg.eta ? j - cfg.eta : j + 2 * COSET_Q - cfg.eta;
                if (hit.index != j5) return std::cerr << "pair failure line " << __LINE__ << "\n", false;
                u64 z5 = mulmod(mulmod(cfg.U18, powmod(10, 90)), powmod(cfg.g18, j5));
                u64 expected = addmod(z5, addmod(mulmod(cfg.A18, 5), addmod(mulmod(cfg.ws18, j5), cfg.B18)));
                if (hit.value != expected) {
                    std::cerr << "kind2 mismatch source=" << hit.source << " j=" << j << " j5=" << j5
                              << " actual=" << hit.value << " expected=" << expected
                              << " z=" << z << " z5=" << z5 << " delta5=" << cfg.delta5 << "\n";
                    return false;
                }
                if (j5 >= cfg.begin && j5 < cfg.end && hit.value != concat_mod((u128)5 + 15 * (u128)j5)) return std::cerr << "pair failure line " << __LINE__ << "\n", false;
            } else return std::cerr << "pair failure line " << __LINE__ << "\n", false;
            ++tested;
        }
    }
    std::cout << "PAIR18_FORCED_COVERAGE_OK points=" << tested << "\n";
    #else
    for (u64 original : {0ULL, 1ULL, 2ULL, 3ULL, P - 1}) {
        Job artificial = make_job(1, 1);
        artificial.z0 = 0;
        artificial.g = 1;
        artificial.ws = 0;
        artificial.w0 = original;
        CosetRange range{};
        range.end = 1;
        range.q = COSET_Q;
        range.h = 1;
        range.w_q = 0;
        check_cuda(cudaMemset(count, 0, sizeof(int)), "pair boundary memset");
        scan_pair18_kernel<<<1, BLOCK_SIZE>>>(artificial, range, cfg, hits, count, 4096);
        int observed = 0;
        check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "pair boundary count");
        std::vector<PairHit> actual(observed);
        check_cuda(cudaMemcpy(actual.data(), hits, observed * sizeof(PairHit), cudaMemcpyDeviceToHost), "pair boundary hits");
        int found = 0;
        for (const PairHit& hit : actual) {
            if (hit.kind == 0) {
                u64 expected_t = original <= 2 ? original : ~0ULL;
                if (hit.index != expected_t) return std::cerr << "pair failure line " << __LINE__ << "\n", false;
                ++found;
            }
        }
        int expected = original <= 2 ? 4 : 0;
        if (found != expected) {
            std::cerr << "pair target mismatch original=" << original << " found=" << found << "\n";
            return std::cerr << "pair failure line " << __LINE__ << "\n", false;
        }
    }
    std::cout << "PAIR18_TARGET_BOUNDARIES_OK scaled_t=0,1,2; rejects=3,P-1\n";
    #endif
    cudaFree(hits);
    cudaFree(count);
    return true;
}

static void pair_benchmark(int device, u64 blocks) {
    check_cuda(cudaSetDevice(device), "set pair benchmark device");
    const u64 chunk = 65536;
    const u64 n = chunk * blocks * BLOCK_SIZE;
    Job job = make_job(4 * n, chunk);
    PairParams cfg = make_pair(job);
    PairHit* hits;
    int* count;
    check_cuda(cudaMalloc(&hits, 65536 * sizeof(PairHit)), "malloc pair benchmark hits");
    check_cuda(cudaMalloc(&count, sizeof(int)), "malloc pair benchmark count");
    CosetRange range{};
    range.end = n;
    for (int u = 0; u < 4; ++u) range.limit[u] = n;
    range.q = COSET_Q;
    range.h = powmod(job.g, COSET_Q);
    range.w_q = mulmod(job.ws, COSET_Q);
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int mode = 0; mode < 2; ++mode) {
            auto start = std::chrono::steady_clock::now();
            check_cuda(cudaMemset(count, 0, sizeof(int)), "pair benchmark memset");
            if (mode == 0) {
                scan_coset4_kernel<true><<<blocks, BLOCK_SIZE>>>(job, range, (Hit*)hits, count, 65536);
            } else {
                scan_pair18_kernel<<<blocks, BLOCK_SIZE>>>(job, range, cfg, hits, count, 65536);
            }
            check_cuda(cudaGetLastError(), "pair benchmark launch");
            int observed = 0;
            check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "pair benchmark sync");
            if (observed > 65536) throw std::runtime_error("pair benchmark overflow");
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << "PAIR18_BENCHMARK mode=" << mode << " repeat=" << repeat
                      << " tested=" << 4 * n << " rate=" << 4 * n / seconds << " x/s hits=" << observed << "\n";
        }
    }
    cudaFree(hits);
    cudaFree(count);
}

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "test") return pair_test(std::stoi(argv[2])) ? 0 : 1;
        if (argc == 4 && std::string(argv[1]) == "benchmark") {
            pair_benchmark(std::stoi(argv[2]), std::stoull(argv[3]));
            return 0;
        }
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
