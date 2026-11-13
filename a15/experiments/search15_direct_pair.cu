#include <set>

#define main direct_serial_main
#include "search15_direct.cu"
#undef main

static constexpr u64 PAIR_Q = L;
static constexpr u64 PAIR_PERIOD = 2 * L;

struct PairRange {
    u64 begin, end, q, w_q;
};

template<bool ALL>
__global__ void scan_direct_pair_kernel(Job job, PairRange range, Hit* hits, int* count, int max_hits) {
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 r0 = range.begin + tid * job.chunk;
    if (r0 >= range.end) return;
    u64 r1 = min(r0 + job.chunk, range.end);
    u64 z = mul_standard(job.z0, mont_pow_standard(job.g, r0, job.r2), job.r2);
    u64 w = addmod(mul_standard(job.ws, r0, job.r2), job.w0);
    u64 gm = mont_mul(job.g, job.r2);
    for (u64 r = r0; r < r1; ++r) {
        u64 w1 = advance_mod(w, range.w_q, P - range.w_q);
        u64 negative_z = z ? P - z : 0;
        u64 values[2] = {z + w, negative_z + w1};
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            if (ALL || small_candidate_raw(values[u])) {
                u64 t = canonical_candidate_raw(values[u]);
                int pos = atomicAdd(count, 1);
                if (pos < max_hits) hits[pos] = {r + range.q * u, t};
            }
        }
        z = mont_mul(z, gm);
        w = advance_mod(w, job.ws, P - job.ws);
    }
}

static bool pair_test(int device) {
    check_cuda(cudaSetDevice(device), "set pair test device");
    for (int a : {0, 5}) {
        Job job = make_job(18, a, PAIR_PERIOD, 64);
        if (powmod(job.g, PAIR_Q) != P - 1 || powmod(job.g, PAIR_PERIOD) != 1) return false;
        PairRange range{};
        range.q = PAIR_Q;
        range.w_q = mulmod(job.ws, PAIR_Q);
        Hit* hits;
        int* count;
        u64* reference;
        check_cuda(cudaMalloc(&hits, 4096 * sizeof(Hit)), "malloc pair hits");
        check_cuda(cudaMalloc(&count, sizeof(int)), "malloc pair count");
        check_cuda(cudaMalloc(&reference, 128 * sizeof(u64)), "malloc pair reference");
        for (u64 begin : {0ULL, 1ULL, 17ULL, 123456789ULL, 6666666666666667ULL, PAIR_Q - 64}) {
            for (u64 length : {1ULL, 2ULL, 63ULL, 64ULL}) {
                range.begin = begin;
                range.end = begin + length;
                check_cuda(cudaMemset(count, 0, sizeof(int)), "pair test memset");
                scan_direct_pair_kernel<true><<<1, BLOCK_SIZE>>>(job, range, hits, count, 4096);
                check_cuda(cudaGetLastError(), "pair test launch");
                int observed = 0;
                check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "pair test count");
                if (observed != 2 * length) return false;
                std::vector<Hit> actual(observed);
                check_cuda(cudaMemcpy(actual.data(), hits, observed * sizeof(Hit), cudaMemcpyDeviceToHost), "pair test hits");
                std::set<std::pair<u64, u64>> seen;
                for (const Hit& hit : actual) {
                    u64 u = hit.x / PAIR_Q, r = hit.x % PAIR_Q;
                    if (u >= 2 || r < begin || r >= range.end || !seen.insert({r, u}).second) return false;
                    u64 expected = addmod(mulmod(job.z0, powmod(job.g, hit.x)),
                                          addmod(mulmod(job.ws, hit.x), job.w0));
                    if (hit.t != expected) {
                        std::cerr << "pair mismatch a=" << a << " x=" << hit.x << "\n";
                        return false;
                    }
                    u128 k = a + 15 * (u128)hit.x;
                    if (k >= 100000000000000000ULL && k < 1000000000000000000ULL && expected != concat_mod(k)) {
                        std::cerr << "independent pair mismatch a=" << a << " x=" << hit.x << "\n";
                        return false;
                    }
                    trace_kernel<<<1, 1>>>(job, hit.x, reference, 1);
                    check_cuda(cudaMemcpy(&expected, reference, sizeof(u64), cudaMemcpyDeviceToHost), "pair direct trace");
                    if (hit.t != expected) return false;
                }
            }
        }
        for (u64 target : {0ULL, 1ULL, P - 1}) {
            Job artificial = job;
            artificial.z0 = 0;
            artificial.g = 1;
            artificial.ws = 0;
            artificial.w0 = target;
            range.begin = 123;
            range.end = 124;
            range.w_q = 0;
            check_cuda(cudaMemset(count, 0, sizeof(int)), "pair boundary memset");
            scan_direct_pair_kernel<false><<<1, BLOCK_SIZE>>>(artificial, range, hits, count, 4096);
            int observed = 0;
            check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "pair boundary count");
            std::vector<Hit> actual(observed);
            check_cuda(cudaMemcpy(actual.data(), hits, observed * sizeof(Hit), cudaMemcpyDeviceToHost), "pair boundary hits");
            int expected = target == 0 ? 2 : 0;
            if (observed != expected) return false;
            for (const Hit& hit : actual) {
                if (hit.t != 0 || (hit.x != 123 && hit.x != 123 + PAIR_Q)) return false;
            }
        }
        cudaFree(hits);
        cudaFree(count);
        cudaFree(reference);
    }
    std::cout << "DIRECT_PAIR_TEST_OK classes=2 starts=6 lengths=1,2,63,64 direct_gpu=3120 "
              << "independent_block_sums=260 boundaries=6\n";
    return true;
}

static void pair_benchmark(int device, u64 blocks) {
    check_cuda(cudaSetDevice(device), "set pair benchmark device");
    const u64 chunk = 65536, n = chunk * blocks * BLOCK_SIZE;
    Job job = make_job(18, 0, 2 * n, chunk);
    PairRange range{0, n, PAIR_Q, mulmod(job.ws, PAIR_Q)};
    Hit* hits;
    int* count;
    check_cuda(cudaMalloc(&hits, 65536 * sizeof(Hit)), "malloc pair benchmark hits");
    check_cuda(cudaMalloc(&count, sizeof(int)), "malloc pair benchmark count");
    for (int repeat = 0; repeat < 4; ++repeat) {
        for (int mode = 0; mode < 2; ++mode) {
            auto start = std::chrono::steady_clock::now();
            check_cuda(cudaMemset(count, 0, sizeof(int)), "pair benchmark memset");
            if (mode == 0) scan_kernel<<<blocks, BLOCK_SIZE>>>(job, 0, hits, count, 65536);
            else scan_direct_pair_kernel<false><<<blocks, BLOCK_SIZE>>>(job, range, hits, count, 65536);
            check_cuda(cudaGetLastError(), "pair benchmark launch");
            int observed = 0;
            check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "pair benchmark sync");
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << "DIRECT_PAIR_BENCHMARK mode=" << (mode ? "pair" : "serial") << " repeat=" << repeat
                      << " tested=" << 2 * n << " rate=" << 2 * n / seconds << " hits=" << observed << "\n";
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
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
}
