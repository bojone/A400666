#define SEARCH15_PRUNE_LIBRARY
#include "search15_prune.cu"

__global__ void scan_coarse_kernel(Job job, CosetRange range, Hit* hits, int* count, int max_hits) {
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
        u64 w1 = addmod(w, range.w_q), w2 = addmod(w1, range.w_q), w3 = addmod(w2, range.w_q);
        u64 values[4] = {addmod(z, w), addmod(zh, w1), subtract_mod(w2, z), subtract_mod(w3, zh)};
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            #ifdef TEST_ALL_COARSE_HITS
            if (true) {
            #else
            if (values[u] <= 0xffffffffULL) {
            #endif
                int pos = atomicAdd(count, 1);
                if (pos < max_hits) hits[pos] = {r + range.q * u, values[u]};
            }
        }
        z = mont_mul(z, gm);
        w = addmod(w, job.ws);
    }
}

static bool test(int device) {
    check_cuda(cudaSetDevice(device), "set coarse test device");
    Hit* hits;
    int* count;
    check_cuda(cudaMalloc(&hits, 1280 * sizeof(Hit)), "malloc coarse hits");
    check_cuda(cudaMalloc(&count, sizeof(int)), "malloc coarse count");
    #ifdef TEST_ALL_COARSE_HITS
    int tested = 0;
    for (u64 begin : {0ULL, 1ULL, 17ULL, 123456789ULL, COSET_Q - 64}) {
        Job job = make_job(1, 64);
        CosetRange range{};
        range.begin = begin;
        range.end = begin + 64;
        range.q = COSET_Q;
        range.h = powmod(job.g, COSET_Q);
        range.w_q = mulmod(job.ws, COSET_Q);
        check_cuda(cudaMemset(count, 0, sizeof(int)), "coverage memset");
        scan_coarse_kernel<<<1, BLOCK_SIZE>>>(job, range, hits, count, 1280);
        int observed = 0;
        check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "coverage count");
        if (observed != 256) return false;
        std::vector<Hit> actual(observed);
        check_cuda(cudaMemcpy(actual.data(), hits, observed * sizeof(Hit), cudaMemcpyDeviceToHost), "coverage hits");
        for (const Hit& hit : actual) {
            u64 u = hit.x / COSET_Q;
            u64 r = hit.x % COSET_Q;
            if (u >= 4 || r < begin || r >= begin + 64) return false;
            u64 z = mulmod(job.z0, powmod(job.g, hit.x));
            u64 w = addmod(mulmod(job.ws, hit.x), job.w0);
            if (hit.t != addmod(z, w)) return false;
            ++tested;
        }
    }
    std::cout << "COARSE_FORCED_COVERAGE_OK points=" << tested << "\n";
    #else
    for (u64 t : {0ULL, 1ULL, 2ULL, 3ULL, 0xffffffffULL, 0x100000000ULL, P - 1}) {
        Job job = make_job(1, 1);
        job.z0 = 0;
        job.g = 1;
        job.ws = 0;
        job.w0 = t;
        CosetRange range{};
        range.end = 1;
        range.q = COSET_Q;
        range.h = 1;
        range.w_q = 0;
        check_cuda(cudaMemset(count, 0, sizeof(int)), "boundary memset");
        scan_coarse_kernel<<<1, BLOCK_SIZE>>>(job, range, hits, count, 4);
        int observed = 0;
        check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "boundary count");
        int expected = t <= 0xffffffffULL ? 4 : 0;
        if (observed != expected) {
            std::cerr << "coarse boundary mismatch t=" << t << " observed=" << observed << " expected=" << expected << "\n";
            return false;
        }
        std::vector<Hit> actual(observed);
        check_cuda(cudaMemcpy(actual.data(), hits, observed * sizeof(Hit), cudaMemcpyDeviceToHost), "boundary hits");
        for (const Hit& hit : actual) {
            if (hit.t != t || hit.x % COSET_Q != 0 || hit.x / COSET_Q >= 4) return false;
            u128 k = (u128)STEP * ((u128)hit.x + (u128)L * hit.t);
            if (t > T_LIMIT && k >= 10000000000000000000ULL) {
                u64 total = 0;
                std::ofstream f;
                if (!verify_hit(hit, total, f) || total != 0) return false;
            }
        }
    }
    std::cout << "COARSE_BOUNDARIES_OK t=0,1,2,3,2^32-1,2^32,P-1; strict_host_rejection=OK\n";
    #endif
    cudaFree(hits);
    cudaFree(count);
    return true;
}

static void benchmark(int device, u64 blocks) {
    check_cuda(cudaSetDevice(device), "set coarse benchmark device");
    const u64 chunk = 65536;
    const u64 n = chunk * blocks * BLOCK_SIZE;
    Job job = make_job(4 * n, chunk);
    Hit* hits;
    int* count;
    check_cuda(cudaMalloc(&hits, 65536 * sizeof(Hit)), "malloc coarse benchmark hits");
    check_cuda(cudaMalloc(&count, sizeof(int)), "malloc coarse benchmark count");
    CosetRange range{};
    range.end = n;
    for (int u = 0; u < 4; ++u) range.limit[u] = n;
    range.q = COSET_Q;
    range.h = powmod(job.g, COSET_Q);
    range.w_q = mulmod(job.ws, COSET_Q);
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int mode = 0; mode < 2; ++mode) {
            auto start = std::chrono::steady_clock::now();
            check_cuda(cudaMemset(count, 0, sizeof(int)), "coarse benchmark memset");
            if (mode == 0) scan_coset4_kernel<true><<<blocks, BLOCK_SIZE>>>(job, range, hits, count, 65536);
            else scan_coarse_kernel<<<blocks, BLOCK_SIZE>>>(job, range, hits, count, 65536);
            check_cuda(cudaGetLastError(), "coarse benchmark launch");
            int observed = 0;
            check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "coarse benchmark sync");
            if (observed > 65536) throw std::runtime_error("coarse benchmark hit overflow");
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << "COARSE_BENCHMARK mode=" << mode << " repeat=" << repeat
                      << " tested=" << 4 * n << " rate=" << 4 * n / seconds << " x/s hits=" << observed << "\n";
        }
    }
    cudaFree(hits);
    cudaFree(count);
}

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
