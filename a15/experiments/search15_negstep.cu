#define SIGNED_REFERENCE_LIBRARY
#include "search15_signed.cu"

__global__ void negstep_add_test_kernel(const u64* a, const u64* b, u64* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = signed_submod(a[i], P - b[i]);
}

template<bool ALIGNED>
__global__ void negstep_scan_kernel(Job job, CosetRange range, Hit* hits, int* count, int max_hits) {
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 r0 = range.begin + tid * job.chunk;
    if (r0 >= range.end) return;
    u64 r1 = min(r0 + job.chunk, range.end);
    u64 z = mul_standard(job.z0, mont_pow_standard(job.g, r0, job.r2), job.r2);
    u64 w = addmod(mul_standard(job.ws, r0, job.r2), job.w0);
    u64 gm = mont_mul(job.g, job.r2);
    u64 hm = mont_mul(range.h, job.r2);
    u64 nws = P - job.ws;
    u64 nwq = P - range.w_q;
    for (u64 r = r0; r < r1; ++r) {
        u64 zh = signed_mont(z, hm);
        u64 w1 = signed_submod(w, nwq);
        u64 w2 = signed_submod(w1, nwq);
        u64 w3 = signed_submod(w2, nwq);
        u64 values[4] = {signed_addmod(z, w), signed_addmod(zh, w1),
                         signed_submod(w2, z), signed_submod(w3, zh)};
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            if ((ALIGNED || r >= range.next[u]) && values[u] <= T_LIMIT) {
                int pos = atomicAdd(count, 1);
                if (pos < max_hits) hits[pos] = {r + range.q * u, values[u]};
            }
        }
        z = signed_mont(z, gm);
        w = signed_submod(w, nws);
    }
}

__global__ void negstep_trace_kernel(Job job, CosetRange range, u64 r0, u64* out, int n) {
    if (blockIdx.x || threadIdx.x) return;
    u64 z = mul_standard(job.z0, mont_pow_standard(job.g, r0, job.r2), job.r2);
    u64 w = addmod(mul_standard(job.ws, r0, job.r2), job.w0);
    u64 gm = mont_mul(job.g, job.r2);
    u64 hm = mont_mul(range.h, job.r2);
    u64 nws = P - job.ws;
    u64 nwq = P - range.w_q;
    for (int i = 0; i < n; ++i, ++r0) {
        u64 zh = signed_mont(z, hm);
        u64 w1 = signed_submod(w, nwq);
        u64 w2 = signed_submod(w1, nwq);
        u64 w3 = signed_submod(w2, nwq);
        out[4 * i] = signed_addmod(z, w);
        out[4 * i + 1] = signed_addmod(zh, w1);
        out[4 * i + 2] = signed_submod(w2, z);
        out[4 * i + 3] = signed_submod(w3, zh);
        z = signed_mont(z, gm);
        w = signed_submod(w, nws);
    }
}

static bool negstep_test(int device) {
    if (!test(device)) return false;
    check_cuda(cudaSetDevice(device), "set negstep test device");
    std::mt19937_64 rng(4006661504);
    std::vector<u64> a = {0, 1, 2, 3, P - 1, P - 2, P / 2, P / 2 + 1};
    std::vector<u64> b = a;
    for (int i = 0; i < 200000; ++i) {
        a.push_back(rng() % P);
        b.push_back(rng() % P);
    }
    int n = a.size();
    u64 *da, *db, *dout, *reference;
    check_cuda(cudaMalloc(&da, n * sizeof(u64)), "malloc negstep a");
    check_cuda(cudaMalloc(&db, n * sizeof(u64)), "malloc negstep b");
    check_cuda(cudaMalloc(&dout, n * sizeof(u64)), "malloc negstep output");
    check_cuda(cudaMalloc(&reference, 256 * sizeof(u64)), "malloc negstep reference");
    check_cuda(cudaMemcpy(da, a.data(), n * sizeof(u64), cudaMemcpyHostToDevice), "copy negstep a");
    check_cuda(cudaMemcpy(db, b.data(), n * sizeof(u64), cudaMemcpyHostToDevice), "copy negstep b");
    negstep_add_test_kernel<<<(n + 255) / 256, 256>>>(da, db, dout, n);
    check_cuda(cudaGetLastError(), "negstep addition launch");
    std::vector<u64> out(n), expected(256);
    check_cuda(cudaMemcpy(out.data(), dout, n * sizeof(u64), cudaMemcpyDeviceToHost), "copy negstep additions");
    for (int i = 0; i < n; ++i) {
        if (out[i] != addmod(a[i], b[i])) {
            std::cerr << "negstep addition mismatch i=" << i << " a=" << a[i] << " b=" << b[i] << "\n";
            return false;
        }
    }
    Job job = make_job(1, 1);
    CosetRange range{};
    range.q = COSET_Q;
    range.h = powmod(job.g, COSET_Q);
    range.w_q = mulmod(job.ws, COSET_Q);
    for (u64 r0 : {0ULL, 1ULL, 17ULL, 123456789ULL, COSET_Q - 64}) {
        negstep_trace_kernel<<<1, 1>>>(job, range, r0, dout, 64);
        experimental_trace_kernel<<<1, 1>>>(job, range, r0, reference, 64);
        check_cuda(cudaGetLastError(), "negstep trace launch");
        check_cuda(cudaMemcpy(out.data(), dout, 256 * sizeof(u64), cudaMemcpyDeviceToHost), "copy negstep trace");
        check_cuda(cudaMemcpy(expected.data(), reference, 256 * sizeof(u64), cudaMemcpyDeviceToHost), "copy reference trace");
        for (int i = 0; i < 256; ++i) {
            u64 x = r0 + i / 4 + COSET_Q * (i % 4);
            u64 independent = addmod(mulmod(job.z0, powmod(job.g, x)), addmod(mulmod(job.ws, x), job.w0));
            if (out[i] != expected[i] || out[i] != independent) {
                std::cerr << "negstep trace mismatch r0=" << r0 << " i=" << i << "\n";
                return false;
            }
        }
    }
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    cudaFree(reference);
    std::cout << "NEGSTEP_TEST_OK additions=" << n << " trace_points=1280 independent_points=1280\n";
    return true;
}

static void negstep_benchmark(int device, u64 blocks) {
    check_cuda(cudaSetDevice(device), "set negstep benchmark device");
    const u64 chunk = 65536;
    const u64 n = chunk * blocks * BLOCK_SIZE;
    Job job = make_job(4 * n, chunk);
    Hit* hits;
    int* count;
    check_cuda(cudaMalloc(&hits, 65536 * sizeof(Hit)), "malloc negstep benchmark hits");
    check_cuda(cudaMalloc(&count, sizeof(int)), "malloc negstep benchmark count");
    CosetRange range{};
    range.end = n;
    for (int u = 0; u < 4; ++u) range.limit[u] = n;
    range.q = COSET_Q;
    range.h = powmod(job.g, COSET_Q);
    range.w_q = mulmod(job.ws, COSET_Q);
    const char* names[] = {"signed", "negstep", "production_negstep"};
    for (int repeat = 0; repeat < 4; ++repeat) {
        for (int mode = 0; mode < 3; ++mode) {
            auto start = std::chrono::steady_clock::now();
            check_cuda(cudaMemset(count, 0, sizeof(int)), "negstep benchmark memset");
            if (mode == 0) experimental_scan_kernel<true, true><<<blocks, BLOCK_SIZE>>>(job, range, hits, count, 65536);
            if (mode == 1) negstep_scan_kernel<true><<<blocks, BLOCK_SIZE>>>(job, range, hits, count, 65536);
            if (mode == 2) scan_coset4_kernel<true><<<blocks, BLOCK_SIZE>>>(job, range, hits, count, 65536);
            check_cuda(cudaGetLastError(), "negstep benchmark launch");
            int observed = 0;
            check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "negstep benchmark sync");
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << "NEGSTEP_BENCHMARK mode=" << names[mode] << " repeat=" << repeat
                      << " tested=" << 4 * n << " rate=" << 4 * n / seconds << "\n";
        }
    }
    cudaFree(hits);
    cudaFree(count);
}

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "test") return negstep_test(std::stoi(argv[2])) ? 0 : 1;
        if (argc == 4 && std::string(argv[1]) == "benchmark") {
            negstep_benchmark(std::stoi(argv[2]), std::stoull(argv[3]));
            return 0;
        }
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
}
