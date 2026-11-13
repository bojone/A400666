#include <cassert>
#include <cstdio>

#define SEARCH15_PRUNE_LIBRARY
#include "search15_prune.cu"

using i64 = long long;

__host__ __device__ __forceinline__ u64 signed_addmod(u64 a, u64 b) {
    u64 sum = a + b;
    u64 reduced = sum - P;
    return (i64)reduced < 0 ? sum : reduced;
}

__host__ __device__ __forceinline__ u64 signed_submod(u64 a, u64 b) {
    u64 difference = a - b;
    return (i64)difference < 0 ? difference + P : difference;
}

__device__ __forceinline__ u64 signed_mont(u64 a, u64 b) {
    u64 lo = a * b;
    u64 hi = __umul64hi(a, b);
    u64 m = lo * PINV;
    u64 mhi = __umul64hi(m, P);
    u64 r = hi + mhi + (lo != 0);
    u64 reduced = r - P;
    return (i64)reduced < 0 ? r : reduced;
}

__global__ void signed_mont_test_kernel(const u64* a, const u64* b, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = signed_mont(a[i], b[i]);
}

template<bool ALIGNED, bool SIGNED>
__global__ void experimental_scan_kernel(Job job, CosetRange range, Hit* hits, int* count, int max_hits) {
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 r0 = range.begin + tid * job.chunk;
    if (r0 >= range.end) return;
    u64 r1 = min(r0 + job.chunk, range.end);
    u64 z = mul_standard(job.z0, mont_pow_standard(job.g, r0, job.r2), job.r2);
    u64 w = addmod(mul_standard(job.ws, r0, job.r2), job.w0);
    u64 gm = mont_mul(job.g, job.r2);
    u64 hm = mont_mul(range.h, job.r2);
    for (u64 r = r0; r < r1; ++r) {
        u64 zh, w1, w2, w3;
        if constexpr (SIGNED) {
            zh = signed_mont(z, hm);
            w1 = signed_addmod(w, range.w_q);
            w2 = signed_addmod(w1, range.w_q);
            w3 = signed_addmod(w2, range.w_q);
        } else {
            zh = mont_mul(z, hm);
            w1 = addmod(w, range.w_q);
            w2 = addmod(w1, range.w_q);
            w3 = addmod(w2, range.w_q);
        }
        u64 values[4];
        if constexpr (SIGNED) {
            values[0] = signed_addmod(z, w);
            values[1] = signed_addmod(zh, w1);
            values[2] = signed_submod(w2, z);
            values[3] = signed_submod(w3, zh);
        } else {
            values[0] = addmod(z, w);
            values[1] = addmod(zh, w1);
            values[2] = subtract_mod(w2, z);
            values[3] = subtract_mod(w3, zh);
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            if ((ALIGNED || r >= range.next[u]) && values[u] <= T_LIMIT) {
                int pos = atomicAdd(count, 1);
                if (pos < max_hits) hits[pos] = {r + range.q * u, values[u]};
            }
        }
        if constexpr (SIGNED) {
            z = signed_mont(z, gm);
            w = signed_addmod(w, job.ws);
        } else {
            z = mont_mul(z, gm);
            w = addmod(w, job.ws);
        }
    }
}

__global__ void experimental_trace_kernel(Job job, CosetRange range, u64 r0, u64* out, int n) {
    if (blockIdx.x || threadIdx.x) return;
    u64 z = mul_standard(job.z0, mont_pow_standard(job.g, r0, job.r2), job.r2);
    u64 w = addmod(mul_standard(job.ws, r0, job.r2), job.w0);
    u64 gm = mont_mul(job.g, job.r2);
    u64 hm = mont_mul(range.h, job.r2);
    for (int i = 0; i < n; ++i, ++r0) {
        u64 zh = signed_mont(z, hm);
        u64 w1 = signed_addmod(w, range.w_q);
        u64 w2 = signed_addmod(w1, range.w_q);
        u64 w3 = signed_addmod(w2, range.w_q);
        out[4 * i] = signed_addmod(z, w);
        out[4 * i + 1] = signed_addmod(zh, w1);
        out[4 * i + 2] = signed_submod(w2, z);
        out[4 * i + 3] = signed_submod(w3, zh);
        z = signed_mont(z, gm);
        w = signed_addmod(w, job.ws);
    }
}

static bool test(int device) {
    check_cuda(cudaSetDevice(device), "set test device");
    std::mt19937_64 rng(4006661503);
    std::vector<u64> a = {0, 1, 2, 3, P - 1, P - 2, P - 3, P / 2, P / 2 + 1};
    std::vector<u64> b = a;
    for (u64 x : a) assert(x < P);
    for (int i = 0; i < 200000; ++i) {
        a.push_back(rng() % P);
        b.push_back(rng() % P);
    }
    int n = a.size();
    u64 *da, *db, *dout;
    check_cuda(cudaMalloc(&da, n * sizeof(u64)), "malloc test a");
    check_cuda(cudaMalloc(&db, n * sizeof(u64)), "malloc test b");
    check_cuda(cudaMalloc(&dout, 4 * n * sizeof(u64)), "malloc test output");
    check_cuda(cudaMemcpy(da, a.data(), n * sizeof(u64), cudaMemcpyHostToDevice), "copy test a");
    check_cuda(cudaMemcpy(db, b.data(), n * sizeof(u64), cudaMemcpyHostToDevice), "copy test b");
    signed_mont_test_kernel<<<(n + 255) / 256, 256>>>(da, db, dout, n);
    std::vector<u64> out(n);
    check_cuda(cudaMemcpy(out.data(), dout, n * sizeof(u64), cudaMemcpyDeviceToHost), "copy test output");
    u64 R = (u64)(((u128)1 << 64) % P), rinv = invmod(R);
    for (int i = 0; i < n; ++i) {
        assert(out[i] == mulmod(mulmod(a[i], b[i]), rinv));
        if (signed_addmod(a[i], b[i]) != addmod(a[i], b[i]) ||
            signed_submod(a[i], b[i]) != subtract_mod(a[i], b[i])) {
            std::cerr << "signed arithmetic mismatch i=" << i << " a=" << a[i] << " b=" << b[i]
                      << " add=" << signed_addmod(a[i], b[i]) << " expected_add=" << addmod(a[i], b[i])
                      << " sub=" << signed_submod(a[i], b[i]) << " expected_sub=" << subtract_mod(a[i], b[i]) << "\n";
            return false;
        }
    }
    Job job = make_job(1, 1);
    CosetRange range{};
    range.q = COSET_Q;
    range.h = powmod(job.g, COSET_Q);
    range.w_q = mulmod(job.ws, COSET_Q);
    u64* reference;
    check_cuda(cudaMalloc(&reference, 256 * sizeof(u64)), "malloc reference trace");
    for (u64 r0 : {0ULL, 1ULL, 17ULL, 123456789ULL, COSET_Q - 64}) {
        experimental_trace_kernel<<<1, 1>>>(job, range, r0, dout, 64);
        coset_trace_kernel<<<1, 1>>>(job, range.h, range.w_q, r0, reference, 64);
        check_cuda(cudaMemcpy(out.data(), dout, 256 * sizeof(u64), cudaMemcpyDeviceToHost), "copy signed trace");
        std::vector<u64> expected(256);
        check_cuda(cudaMemcpy(expected.data(), reference, 256 * sizeof(u64), cudaMemcpyDeviceToHost), "copy reference trace");
        for (int i = 0; i < 256; ++i) {
            if (out[i] != expected[i]) {
                u64 x = r0 + i / 4 + COSET_Q * (i % 4);
                u64 value = addmod(mulmod(job.z0, powmod(job.g, x)), addmod(mulmod(job.ws, x), job.w0));
                std::cerr << "signed trace mismatch r0=" << r0 << " i=" << i << " x=" << x
                          << " actual=" << out[i] << " reference=" << expected[i] << " independent=" << value << "\n";
                return false;
            }
        }
        for (int i = 0; i < 64; ++i) {
            for (int u = 0; u < 4; ++u) {
                u64 x = r0 + i + COSET_Q * u;
                u64 value = addmod(mulmod(job.z0, powmod(job.g, x)), addmod(mulmod(job.ws, x), job.w0));
                assert(out[4 * i + u] == value);
            }
        }
    }
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    cudaFree(reference);
    std::cout << "SIGNED_TEST_OK montgomery=" << n << " traces=1280 independent_points=1280\n";
    return true;
}

static void benchmark(int device, u64 blocks) {
    check_cuda(cudaSetDevice(device), "set benchmark device");
    const u64 chunk = 65536;
    const u64 n = chunk * blocks * BLOCK_SIZE;
    Job job = make_job(4 * n, chunk);
    Job serial_job = make_job(4 * n, 4 * chunk);
    Hit* hits;
    int* count;
    check_cuda(cudaMalloc(&hits, 65536 * sizeof(Hit)), "malloc benchmark hits");
    check_cuda(cudaMalloc(&count, sizeof(int)), "malloc benchmark count");
    CosetRange range{};
    range.end = n;
    for (int u = 0; u < 4; ++u) range.limit[u] = n;
    range.q = COSET_Q;
    range.h = powmod(job.g, COSET_Q);
    range.w_q = mulmod(job.ws, COSET_Q);
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int mode = 0; mode < 4; ++mode) {
            auto start = std::chrono::steady_clock::now();
            check_cuda(cudaMemset(count, 0, sizeof(int)), "benchmark memset");
            if (mode == 0) scan_kernel<<<blocks, BLOCK_SIZE>>>(serial_job, 0, hits, count, 65536);
            if (mode == 1) experimental_scan_kernel<false, false><<<blocks, BLOCK_SIZE>>>(job, range, hits, count, 65536);
            if (mode == 2) experimental_scan_kernel<false, true><<<blocks, BLOCK_SIZE>>>(job, range, hits, count, 65536);
            if (mode == 3) experimental_scan_kernel<true, true><<<blocks, BLOCK_SIZE>>>(job, range, hits, count, 65536);
            check_cuda(cudaGetLastError(), "benchmark launch");
            int observed = 0;
            check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "benchmark sync");
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << "SIGNED_BENCHMARK mode=" << mode << " repeat=" << repeat
                      << " tested=" << 4 * n << " rate=" << 4 * n / seconds << " x/s\n";
        }
    }
    cudaFree(hits);
    cudaFree(count);
}

#ifndef SIGNED_REFERENCE_LIBRARY
int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "test") return test(std::stoi(argv[2])) ? 0 : 1;
        if (argc == 4 && std::string(argv[1]) == "benchmark") {
            benchmark(std::stoi(argv[2]), std::stoull(argv[3]));
            return 0;
        }
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
#endif
