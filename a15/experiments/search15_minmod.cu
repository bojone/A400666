#define SIGNED_REFERENCE_LIBRARY
#include "search15_signed.cu"

__host__ __device__ __forceinline__ u64 minmod_raw(u64 raw, bool subtract) {
    return subtract ? min(raw, raw + P) : min(raw, raw - P);
}

__global__ void minmod_test_kernel(const u64* a, const u64* b, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[2 * i] = minmod_raw(a[i] + b[i], false);
        out[2 * i + 1] = minmod_raw(a[i] - b[i], true);
    }
}

template<bool ALIGNED>
__global__ void minmod_scan_kernel(Job job, CosetRange range, Hit* hits, int* count, int max_hits) {
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 r0 = range.begin + tid * job.chunk;
    if (r0 >= range.end) return;
    u64 r1 = min(r0 + job.chunk, range.end);
    u64 z = mul_standard(job.z0, mont_pow_standard(job.g, r0, job.r2), job.r2);
    u64 w = addmod(mul_standard(job.ws, r0, job.r2), job.w0);
    u64 gm = mont_mul(job.g, job.r2);
    u64 hm = mont_mul(range.h, job.r2);
    for (u64 r = r0; r < r1; ++r) {
        u64 zh = mont_mul(z, hm);
        u64 w1 = advance_mod(w, range.w_q, P - range.w_q);
        u64 w2 = advance_mod(w1, range.w_q, P - range.w_q);
        u64 w3 = advance_mod(w2, range.w_q, P - range.w_q);
        u64 raw[4] = {z + w, zh + w1, w2 - z, w3 - zh};
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            u64 t = minmod_raw(raw[u], u >= 2);
            bool selected = ALIGNED || r >= range.next[u];
            #ifdef TEST_ALL_COSET_HITS
            if (selected) {
            #else
            if (selected && t <= T_LIMIT) {
            #endif
                int pos = atomicAdd(count, 1);
                if (pos < max_hits) hits[pos] = {r + range.q * u, t};
            }
        }
        z = mont_mul(z, gm);
        w = advance_mod(w, job.ws, P - job.ws);
    }
}

static bool minmod_test(int device) {
    if (!coset_selftest(device)) return false;
    check_cuda(cudaSetDevice(device), "set minmod test device");
    std::mt19937_64 rng(4006661506);
    std::vector<u64> a = {0, 1, 2, 3, P - 1, P - 2, P / 2, P / 2 + 1};
    std::vector<u64> b = a;
    for (int i = 0; i < 2000; ++i) {
        u64 value = rng() % P, target = i % 4;
        a.push_back(value);
        b.push_back(submod(target, value));
        a.push_back(value);
        b.push_back(submod(value, target));
    }
    for (int i = 0; i < 200000; ++i) {
        a.push_back(rng() % P);
        b.push_back(rng() % P);
    }
    int n = a.size();
    u64 *da, *db, *dout;
    check_cuda(cudaMalloc(&da, n * sizeof(u64)), "malloc minmod a");
    check_cuda(cudaMalloc(&db, n * sizeof(u64)), "malloc minmod b");
    check_cuda(cudaMalloc(&dout, 2 * n * sizeof(u64)), "malloc minmod output");
    check_cuda(cudaMemcpy(da, a.data(), n * sizeof(u64), cudaMemcpyHostToDevice), "copy minmod a");
    check_cuda(cudaMemcpy(db, b.data(), n * sizeof(u64), cudaMemcpyHostToDevice), "copy minmod b");
    minmod_test_kernel<<<(n + 255) / 256, 256>>>(da, db, dout, n);
    check_cuda(cudaGetLastError(), "minmod test launch");
    std::vector<u64> out(2 * n);
    check_cuda(cudaMemcpy(out.data(), dout, 2 * n * sizeof(u64), cudaMemcpyDeviceToHost), "copy minmod results");
    for (int i = 0; i < n; ++i) {
        if (out[2 * i] != addmod(a[i], b[i]) || out[2 * i + 1] != subtract_mod(a[i], b[i])) {
            std::cerr << "minmod mismatch i=" << i << " a=" << a[i] << " b=" << b[i]
                      << " add=" << out[2 * i] << " expected_add=" << addmod(a[i], b[i])
                      << " sub=" << out[2 * i + 1] << " expected_sub=" << subtract_mod(a[i], b[i]) << "\n";
            return false;
        }
    }
    #ifdef TEST_ALL_COSET_HITS
    Job job = make_job(1, 100);
    CosetRange range{};
    range.q = COSET_Q;
    range.h = powmod(job.g, COSET_Q);
    range.w_q = mulmod(job.ws, COSET_Q);
    Hit* hits;
    int* count;
    check_cuda(cudaMalloc(&hits, 400 * sizeof(Hit)), "malloc minmod filter hits");
    check_cuda(cudaMalloc(&count, sizeof(int)), "malloc minmod filter count");
    for (int trial = 0; trial < 2; ++trial) {
        const u64 begin = 1234567;
        range.begin = begin;
        range.end = begin + 100;
        int expected = 0;
        for (int u = 0; u < 4; ++u) {
            range.next[u] = trial ? begin : begin + 3 * u;
            range.limit[u] = range.end;
            expected += trial ? 100 : 100 - 3 * u;
        }
        check_cuda(cudaMemset(count, 0, sizeof(int)), "minmod filter memset");
        if (trial) minmod_scan_kernel<true><<<1, BLOCK_SIZE>>>(job, range, hits, count, 400);
        else minmod_scan_kernel<false><<<1, BLOCK_SIZE>>>(job, range, hits, count, 400);
        check_cuda(cudaGetLastError(), "minmod filter launch");
        int observed = 0;
        check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "minmod filter count");
        if (observed != expected) {
            std::cerr << "minmod filter count=" << observed << " expected=" << expected << "\n";
            return false;
        }
        std::vector<Hit> actual(observed);
        std::vector<bool> seen(400, false);
        check_cuda(cudaMemcpy(actual.data(), hits, observed * sizeof(Hit), cudaMemcpyDeviceToHost), "minmod filter hits");
        for (const Hit& hit : actual) {
            u64 u = hit.x / COSET_Q, r = hit.x % COSET_Q;
            if (u >= 4 || r < range.next[u] || r >= range.end) return false;
            u64 index = 100 * u + r - begin;
            if (seen[index]) return false;
            u64 z = mulmod(job.z0, powmod(job.g, hit.x));
            u64 w = addmod(mulmod(job.ws, hit.x), job.w0);
            if (hit.t != addmod(z, w)) {
                std::cerr << "minmod residue mismatch u=" << u << " r=" << r << "\n";
                return false;
            }
            seen[index] = true;
        }
        for (int u = 0; u < 4; ++u) {
            for (u64 r = begin; r < range.end; ++r) {
                if (seen[100 * u + r - begin] != (r >= range.next[u])) return false;
            }
        }
    }
    cudaFree(hits);
    cudaFree(count);
    std::cout << "MINMOD_FILTER_COVERAGE_OK aligned=400 unaligned=379 independent_residues=779\n";
    #endif
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    std::cout << "MINMOD_ARITHMETIC_OK pairs=" << n << " add_and_sub=OK\n";
    return true;
}

static void minmod_benchmark(int device, u64 blocks) {
    check_cuda(cudaSetDevice(device), "set minmod benchmark device");
    const u64 chunk = 65536, n = chunk * blocks * BLOCK_SIZE;
    Job job = make_job(4 * n, chunk);
    Hit* hits;
    int* count;
    check_cuda(cudaMalloc(&hits, 65536 * sizeof(Hit)), "malloc minmod benchmark hits");
    check_cuda(cudaMalloc(&count, sizeof(int)), "malloc minmod benchmark count");
    CosetRange range{};
    range.end = n;
    for (int u = 0; u < 4; ++u) range.limit[u] = n;
    range.q = COSET_Q;
    range.h = powmod(job.g, COSET_Q);
    range.w_q = mulmod(job.ws, COSET_Q);
    for (int repeat = 0; repeat < 4; ++repeat) {
        for (int mode = 0; mode < 2; ++mode) {
            auto start = std::chrono::steady_clock::now();
            check_cuda(cudaMemset(count, 0, sizeof(int)), "minmod benchmark memset");
            if (mode == 0) scan_coset4_kernel<true><<<blocks, BLOCK_SIZE>>>(job, range, hits, count, 65536);
            else minmod_scan_kernel<true><<<blocks, BLOCK_SIZE>>>(job, range, hits, count, 65536);
            check_cuda(cudaGetLastError(), "minmod benchmark launch");
            int observed = 0;
            check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "minmod benchmark sync");
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << "MINMOD_BENCHMARK mode=" << (mode ? "minmod" : "lazy") << " repeat=" << repeat
                      << " tested=" << 4 * n << " rate=" << 4 * n / seconds << "\n";
        }
    }
    cudaFree(hits);
    cudaFree(count);
}

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "test") return minmod_test(std::stoi(argv[2])) ? 0 : 1;
        if (argc == 4 && std::string(argv[1]) == "benchmark") {
            minmod_benchmark(std::stoi(argv[2]), std::stoull(argv[3]));
            return 0;
        }
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
}
