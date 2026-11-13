#include <array>
#include <map>

#define main period_serial_main
#include "search15_period_fast.cu"
#undef main

struct CosetRange {
    u64 next[4], limit[4], begin, end, q, h, w_q;
};

static constexpr u64 COSET_Q = L / 4;

__host__ __device__ __forceinline__ u64 negmod(u64 a) {
    return a ? P - a : 0;
}

__host__ __device__ __forceinline__ u64 subtract_mod(u64 a, u64 b) {
    u64 difference = a - b;
    return (long long)difference < 0 ? difference + P : difference;
}

__host__ __device__ __forceinline__ u64 advance_mod(u64 a, u64 step, u64 negative_step) {
    #ifdef NEGATIVE_LINEAR_STEPS
    return subtract_mod(a, negative_step);
    #else
    return addmod(a, step);
    #endif
}

__host__ __device__ __forceinline__ bool small_candidate_raw(u64 raw, bool subtract) {
    return subtract ? (raw <= T_LIMIT || raw + P <= T_LIMIT)
                    : (raw <= T_LIMIT || raw - P <= T_LIMIT);
}

__host__ __device__ __forceinline__ u64 canonical_candidate_raw(u64 raw, bool subtract) {
    if (subtract) return (long long)raw < 0 ? raw + P : raw;
    u64 reduced = raw - P;
    return (long long)reduced < 0 ? raw : reduced;
}

__global__ void scan_pruned_kernel(Job job, CosetRange range, Hit* hits, int* count, int max_hits) {
    u64 begin = range.begin;
    u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    u64 r0 = begin + tid * job.chunk;
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
        #ifdef LAZY_SMALL_FILTER
        u64 values[4] = {z + w, zh + w1, w2 - z, w3 - zh};
        #else
        u64 values[4] = {addmod(z, w), addmod(zh, w1), subtract_mod(w2, z), subtract_mod(w3, zh)};
        #endif
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            if (r >= range.next[u] && r < range.limit[u]) {
                #ifdef TEST_ALL_COSET_HITS
                if (true) {
                #elif defined(LAZY_SMALL_FILTER)
                if (small_candidate_raw(values[u], u >= 2)) {
                #else
                if (values[u] <= T_LIMIT) {
                #endif
                    #ifdef LAZY_SMALL_FILTER
                    u64 t = canonical_candidate_raw(values[u], u >= 2);
                    #else
                    u64 t = values[u];
                    #endif
                    int pos = atomicAdd(count, 1);
                    if (pos < max_hits) hits[pos] = {r + range.q * u, t};
                }
            }
        }
        z = mont_mul(z, gm);
        w = advance_mod(w, job.ws, P - job.ws);
    }
}

template<bool ALIGNED = false>
__global__ void scan_coset4_kernel(Job job, CosetRange range, Hit* hits, int* count, int max_hits) {
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
        #ifdef LAZY_SMALL_FILTER
        u64 values[4] = {z + w, zh + w1, w2 - z, w3 - zh};
        #else
        u64 values[4] = {addmod(z, w), addmod(zh, w1), subtract_mod(w2, z), subtract_mod(w3, zh)};
        #endif
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            #ifdef TEST_ALL_COSET_HITS
            if (ALIGNED || r >= range.next[u]) {
            #elif defined(LAZY_SMALL_FILTER)
            if ((ALIGNED || r >= range.next[u]) && small_candidate_raw(values[u], u >= 2)) {
            #else
            if ((ALIGNED || r >= range.next[u]) && values[u] <= T_LIMIT) {
            #endif
                #ifdef LAZY_SMALL_FILTER
                u64 t = canonical_candidate_raw(values[u], u >= 2);
                #else
                u64 t = values[u];
                #endif
                int pos = atomicAdd(count, 1);
                if (pos < max_hits) hits[pos] = {r + range.q * u, t};
            }
        }
        z = mont_mul(z, gm);
        w = advance_mod(w, job.ws, P - job.ws);
    }
}

__global__ void coset_trace_kernel(Job job, u64 h, u64 w_q, u64 r0, u64* out, int n) {
    if (blockIdx.x || threadIdx.x) return;
    u64 z = mul_standard(job.z0, mont_pow_standard(job.g, r0, job.r2), job.r2);
    u64 w = addmod(mul_standard(job.ws, r0, job.r2), job.w0);
    u64 gm = mont_mul(job.g, job.r2);
    u64 hm = mont_mul(h, job.r2);
    for (int i = 0; i < n; ++i, ++r0) {
        u64 zh = mont_mul(z, hm);
        u64 zs[4] = {z, zh, negmod(z), negmod(zh)};
        u64 wu = w;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            out[4 * i + u] = addmod(zs[u], wu);
            wu = addmod(wu, w_q);
        }
        z = mont_mul(z, gm);
        w = advance_mod(w, job.ws, P - job.ws);
    }
}

static std::map<std::string, std::string> read_state_file(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("missing state: " + path.string());
    std::map<std::string, std::string> values;
    std::string line;
    while (std::getline(f, line)) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) values[line.substr(0, pos)] = line.substr(pos + 1);
    }
    return values;
}

static u64 value_u64(const std::map<std::string, std::string>& values, const std::string& key) {
    auto it = values.find(key);
    if (it == values.end()) throw std::runtime_error("missing state key: " + key);
    return std::stoull(it->second);
}

static std::array<u64, 4> coverage_limits(u64 rlo, u64 rhi, u64 x_limit) {
    std::array<u64, 4> limits;
    for (int u = 0; u < 4; ++u) {
        u64 offset = COSET_Q * u;
        limits[u] = x_limit <= offset + rlo ? rlo : std::min(rhi, x_limit - offset);
    }
    return limits;
}

static u64 common_next(u64 begin, u64 end, u64 rlo, u64 rhi, const std::array<u64, 4>& next,
                       const std::array<u64, 4>& limits) {
    u64 done = 0;
    for (int u = 0; u < 4; ++u) done += next[u] - rlo + rhi - limits[u];
    u64 total = 4 * (rhi - rlo);
    return begin + (u64)((u128)(end - begin) * done / total);
}

static u64 candidate_limit() {
    #ifdef TEST_PRUNE_LIMIT_FILE
    std::ifstream f("results15/test_limit");
    u64 limit;
    if (!(f >> limit)) throw std::runtime_error("missing test limit");
    return limit;
    #elif defined(TEST_PRUNE_LIMIT)
    return TEST_PRUNE_LIMIT;
    #else
    u64 best = 0;
    for (int device = 0; device < 8; ++device) {
        std::ifstream f(hits_path("d19_x" + std::to_string(device)));
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("k=", 0) != 0 || line.find(" independent_remainder_p=0 k_mod_15=") == std::string::npos)
                continue;
            u64 k = std::stoull(line.substr(2));
            if (k < 1000000000000000000ULL || k >= 10000000000000000000ULL ||
                (k % 15 != 0 && k % 15 != 5)) continue;
            if (concat_mod(k) != 0 || concat_mod_sum(k) != 0)
                throw std::runtime_error("invalid candidate cannot be used as a search bound");
            if (!best || k < best) best = k;
        }
    }
    return best ? std::min(L, best / STEP) : L;
    #endif
}

static std::array<u64, 4> initial_next(int device) {
    u64 rlo = COSET_Q * device / 8, rhi = COSET_Q * (device + 1) / 8;
    std::array<u64, 4> next;
    for (int u = 0; u < 4; ++u) {
        int old_device = 2 * u + device / 4;
        std::string old_name = "d19_x" + std::to_string(old_device);
        auto old = read_state_file(state_path(old_name) + ".serial.state");
        u64 begin = L * old_device / 8, end = L * (old_device + 1) / 8;
        if (old.at("config") != config_key(begin, end)) throw std::runtime_error("serial state mismatch");
        u64 old_next = value_u64(old, "next_x");
        if (old_next < begin || old_next > end) throw std::runtime_error("serial checkpoint out of range");
        u64 offset = COSET_Q * u + rlo;
        u64 done = old_next > offset ? old_next - offset : 0;
        next[u] = rlo + std::min(done, rhi - rlo);
    }
    return next;
}

static void save_coset_state(const std::string& name, u64 begin, u64 end, u64 rlo, u64 rhi,
                             const std::array<u64, 4>& next, const std::array<u64, 4>& limits,
                             u64 x_limit, u64 hits, u64 chunk, u64 blocks) {
    u64 next_x = common_next(begin, end, rlo, rhi, next, limits);
    std::string tmp = state_path(name) + ".tmp";
    std::ofstream f(tmp);
    f << "config=" << config_key(begin, end) << "\nnext_x=" << next_x << "\nhits=" << hits
      << "\nchunk=" << chunk << "\nblocks=" << blocks << "\nmode=coset4\nq=" << COSET_Q
      << "\nr_begin=" << rlo << "\nr_end=" << rhi << "\nx_limit=" << x_limit << "\n";
    for (int u = 0; u < 4; ++u) f << "next_r" << u << "=" << next[u] << "\n";
    f.close();
    if (!f) throw std::runtime_error("failed to save checkpoint");
    std::filesystem::rename(tmp, state_path(name));
}

static std::pair<std::array<u64, 4>, u64> load_coset_state(
        const std::string& name, int device, u64 begin, u64 end, u64 rlo, u64 rhi, u64 x_limit) {
    if (!std::filesystem::exists(state_path(name))) return {initial_next(device), 0};
    auto values = read_state_file(state_path(name));
    if (values.at("config") != config_key(begin, end)) throw std::runtime_error("state mismatch");
    if (values.count("mode") == 0) {
        if (values.at("next_x") != read_state_file(state_path(name) + ".serial.state").at("next_x"))
            throw std::runtime_error("state changed during migration");
        return {initial_next(device), value_u64(values, "hits")};
    }
    if (values.at("mode") != "coset4" || value_u64(values, "q") != COSET_Q ||
        value_u64(values, "r_begin") != rlo || value_u64(values, "r_end") != rhi)
        throw std::runtime_error("coset state mismatch");
    u64 stored_limit = values.count("x_limit") ? value_u64(values, "x_limit") : L;
    if (stored_limit < x_limit) throw std::runtime_error("candidate history missing from checkpoint");
    auto limits = coverage_limits(rlo, rhi, stored_limit);
    std::array<u64, 4> next;
    for (int u = 0; u < 4; ++u) {
        next[u] = value_u64(values, "next_r" + std::to_string(u));
        if (next[u] < rlo || next[u] > limits[u]) throw std::runtime_error("coset checkpoint out of range");
    }
    if (value_u64(values, "next_x") != common_next(begin, end, rlo, rhi, next, limits))
        throw std::runtime_error("coset progress mismatch");
    return {next, value_u64(values, "hits")};
}

static bool verify_hit(const Hit& hit, u64& total_hits, std::ofstream& f) {
    u128 k = (u128)STEP * ((u128)hit.x + (u128)L * hit.t);
    u128 lo = 1;
    for (int i = 1; i < DIGITS; ++i) lo *= 10;
    u128 hi = lo * 10 - 1;
    if (k < lo || k > hi || (k % 15 != 0 && k % 15 != 5)) return true;
    if (concat_mod(k) != 0 || concat_mod_sum(k) != 0) {
        std::cerr << "false hit k=" << u128str(k) << "\n";
        return false;
    }
    f << "k=" << u128str(k) << " x=" << hit.x << " t=" << hit.t
      << " remainder_p=0 independent_remainder_p=0 k_mod_15=" << (u64)(k % 15) << "\n";
    f.flush();
    ++total_hits;
    std::cout << "HIT k=" << u128str(k) << " x=" << hit.x << " t=" << hit.t << "\n";
    return true;
}

static bool coset_selftest(int device) {
    if (!selftest(device)) return false;
    Job job = make_job(1, 1);
    u64 h = powmod(job.g, COSET_Q);
    if (powmod(h, 4) != 1 || h == 1 || powmod(h, 2) == 1 || L != 4 * COSET_Q) {
        std::cerr << "coset order check failed\n";
        return false;
    }
    const int n = 64;
    u64 *reference, *actual;
    check_cuda(cudaMalloc(&reference, n * sizeof(u64)), "malloc reference");
    check_cuda(cudaMalloc(&actual, 4 * n * sizeof(u64)), "malloc actual");
    std::vector<u64> a(n), b(4 * n);
    for (u64 r0 : {0ULL, 1ULL, 17ULL, 123456789ULL, COSET_Q - 65}) {
        coset_trace_kernel<<<1, 1>>>(job, h, mulmod(job.ws, COSET_Q), r0, actual, n);
        check_cuda(cudaGetLastError(), "coset trace launch");
        check_cuda(cudaMemcpy(b.data(), actual, sizeof(u64) * b.size(), cudaMemcpyDeviceToHost), "coset trace");
        for (int u = 0; u < 4; ++u) {
            trace_kernel<<<1, 1>>>(job, r0 + COSET_Q * u, reference, n);
            check_cuda(cudaGetLastError(), "reference trace launch");
            check_cuda(cudaMemcpy(a.data(), reference, sizeof(u64) * n, cudaMemcpyDeviceToHost), "reference trace");
            for (int i = 0; i < n; ++i) {
                if (a[i] != b[4 * i + u]) {
                    std::cerr << "coset trace mismatch r0=" << r0 << " u=" << u << " i=" << i << "\n";
                    return false;
                }
                u64 x = r0 + COSET_Q * u + i;
                u64 z = mulmod(job.z0, powmod(job.g, x));
                u64 w = addmod(mulmod(job.ws, x), job.w0);
                u64 t = addmod(z, w);
                u128 j = (u128)x + (u128)L * t;
                u128 k = STEP * j;
                u128 lo = 1000000000000000000ULL, hi = 10000000000000000000ULL;
                if (lo <= k && k < hi && ((concat_mod(k) == 0) != (t <= T_LIMIT) ||
                    (t <= T_LIMIT && concat_mod_sum(k) != 0))) {
                    std::cerr << "independent coset residue mismatch x=" << x << "\n";
                    return false;
                }
            }
        }
    }
    cudaFree(reference);
    cudaFree(actual);
    std::cout << "coset4 selftest ok: order four, 20 independent 64-point traces\n";
    return true;
}

#ifdef TEST_ALL_COSET_HITS
static bool coset_fast_filter_test() {
    Job job = make_job(1, 100);
    CosetRange range{};
    range.next[0] = COSET_Q - 100;
    range.next[1] = COSET_Q - 97;
    range.next[2] = COSET_Q - 93;
    range.next[3] = COSET_Q - 89;
    range.begin = COSET_Q - 100;
    range.end = COSET_Q;
    range.q = COSET_Q;
    range.h = powmod(job.g, COSET_Q);
    range.w_q = mulmod(job.ws, COSET_Q);
    Hit* hits;
    int* count;
    check_cuda(cudaMalloc(&hits, 400 * sizeof(Hit)), "malloc fast filter hits");
    check_cuda(cudaMalloc(&count, sizeof(int)), "malloc fast filter count");
    check_cuda(cudaMemset(count, 0, sizeof(int)), "fast filter memset");
    scan_coset4_kernel<<<1, BLOCK_SIZE>>>(job, range, hits, count, 400);
    check_cuda(cudaGetLastError(), "fast filter scan");
    int observed = 0;
    check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "fast filter count");
    if (observed != 379) {
        std::cerr << "fast filter count mismatch: " << observed << " instead of 379\n";
        return false;
    }
    std::vector<Hit> actual(observed);
    std::vector<bool> seen(400, false);
    check_cuda(cudaMemcpy(actual.data(), hits, actual.size() * sizeof(Hit), cudaMemcpyDeviceToHost), "fast filter hits");
    for (const Hit& hit : actual) {
        u64 u = hit.x / COSET_Q, r = hit.x % COSET_Q;
        if (u >= 4) return false;
        u64 index = 100 * u + r - (COSET_Q - 100);
        if (r < range.next[u] || r >= COSET_Q || seen[index]) {
            std::cerr << "fast filter boundary or duplicate mismatch x=" << hit.x << "\n";
            return false;
        }
        u64 z = mulmod(job.z0, powmod(job.g, hit.x));
        u64 w = addmod(mulmod(job.ws, hit.x), job.w0);
        if (hit.t != addmod(z, w)) {
            std::cerr << "fast filter residue mismatch x=" << hit.x << "\n";
            return false;
        }
        seen[index] = true;
    }
    for (int u = 0; u < 4; ++u) {
        for (u64 r = COSET_Q - 100; r < COSET_Q; ++r) {
            bool required = r >= range.next[u];
            if (seen[100 * u + r - (COSET_Q - 100)] != required) {
                std::cerr << "fast filter coverage mismatch r=" << r << " u=" << u << "\n";
                return false;
            }
        }
    }
    std::fill(range.next, range.next + 4, range.begin);
    check_cuda(cudaMemset(count, 0, sizeof(int)), "aligned filter memset");
    scan_coset4_kernel<true><<<1, BLOCK_SIZE>>>(job, range, hits, count, 400);
    check_cuda(cudaGetLastError(), "aligned filter scan");
    check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "aligned filter count");
    if (observed != 400) {
        std::cerr << "aligned filter count mismatch: " << observed << " instead of 400\n";
        return false;
    }
    actual.resize(observed);
    std::fill(seen.begin(), seen.end(), false);
    check_cuda(cudaMemcpy(actual.data(), hits, actual.size() * sizeof(Hit), cudaMemcpyDeviceToHost), "aligned filter hits");
    for (const Hit& hit : actual) {
        u64 u = hit.x / COSET_Q, r = hit.x % COSET_Q;
        if (u >= 4 || r < range.begin || r >= range.end || seen[100 * u + r - range.begin]) return false;
        u64 z = mulmod(job.z0, powmod(job.g, hit.x));
        u64 w = addmod(mulmod(job.ws, hit.x), job.w0);
        if (hit.t != addmod(z, w)) return false;
        seen[100 * u + r - range.begin] = true;
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) return false;
    cudaFree(hits);
    cudaFree(count);
    std::cout << "exhaustive fast filter test ok: 379 expected hits; aligned test: 400 hits; no omissions or duplicates\n";
    return true;
}

static bool coset_filter_test() {
    if (!coset_fast_filter_test()) return false;
    Job job = make_job(1, 100);
    CosetRange range{};
    range.next[0] = COSET_Q - 100;
    range.next[1] = COSET_Q - 97;
    range.next[2] = COSET_Q - 93;
    range.next[3] = COSET_Q - 89;
    range.begin = COSET_Q - 100;
    range.q = COSET_Q;
    range.h = powmod(job.g, COSET_Q);
    range.w_q = mulmod(job.ws, COSET_Q);
    Hit* hits;
    int* count;
    check_cuda(cudaMalloc(&hits, 400 * sizeof(Hit)), "malloc filter hits");
    check_cuda(cudaMalloc(&count, sizeof(int)), "malloc filter count");
    for (int trial = 0; trial < 2; ++trial) {
        int expected = 0;
        for (int u = 0; u < 4; ++u) {
            range.limit[u] = trial ? COSET_Q - 50 + 10 * u : COSET_Q;
            expected += range.limit[u] - range.next[u];
        }
        range.end = *std::max_element(range.limit, range.limit + 4);
        check_cuda(cudaMemset(count, 0, sizeof(int)), "filter memset");
        scan_pruned_kernel<<<1, BLOCK_SIZE>>>(job, range, hits, count, 400);
        check_cuda(cudaGetLastError(), "filter scan");
        int observed = 0;
        check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "filter count");
        if (observed != expected) {
            std::cerr << "filter count mismatch: " << observed << " instead of " << expected << "\n";
            return false;
        }
        std::vector<Hit> actual(observed);
        std::vector<bool> seen(400, false);
        check_cuda(cudaMemcpy(actual.data(), hits, actual.size() * sizeof(Hit), cudaMemcpyDeviceToHost), "filter hits");
        for (const Hit& hit : actual) {
            u64 u = hit.x / COSET_Q, r = hit.x % COSET_Q;
            if (u >= 4) return false;
            u64 index = 100 * u + r - (COSET_Q - 100);
            if (r < range.next[u] || r >= range.limit[u] || seen[index]) {
                std::cerr << "filter boundary or duplicate mismatch x=" << hit.x << "\n";
                return false;
            }
            u64 z = mulmod(job.z0, powmod(job.g, hit.x));
            u64 w = addmod(mulmod(job.ws, hit.x), job.w0);
            if (hit.t != addmod(z, w)) {
                std::cerr << "filter residue mismatch x=" << hit.x << "\n";
                return false;
            }
            seen[index] = true;
        }
        for (int u = 0; u < 4; ++u) {
            for (u64 r = COSET_Q - 100; r < COSET_Q; ++r) {
                bool required = r >= range.next[u] && r < range.limit[u];
                if (seen[100 * u + r - (COSET_Q - 100)] != required) {
                    std::cerr << "filter coverage mismatch r=" << r << " u=" << u << "\n";
                    return false;
                }
            }
        }
        std::cout << "exhaustive filter test ok: " << observed << " expected hits, no omissions or duplicates\n";
    }
    cudaFree(hits);
    cudaFree(count);
    return true;
}
#endif

#ifdef TEST_SMALL_FILTER
static bool small_filter_test() {
    Job job = make_job(1, 100);
    CosetRange range{};
    range.q = COSET_Q;
    range.h = powmod(job.g, COSET_Q);
    Hit* hits;
    int* count;
    check_cuda(cudaMalloc(&hits, 400 * sizeof(Hit)), "malloc small filter hits");
    check_cuda(cudaMalloc(&count, sizeof(int)), "malloc small filter count");
    int total_expected = 0;
    for (u64 target = 0; target <= 3; ++target) {
        job.z0 = 0;
        job.ws = 0;
        job.w0 = target;
        range.w_q = 0;
        for (int kind = 0; kind < 3; ++kind) {
            const u64 begin = 1234567;
            range.begin = begin;
            range.end = begin + 100;
            int expected = 0;
            for (int u = 0; u < 4; ++u) {
                range.next[u] = kind == 1 ? begin : begin + 3 * u;
                range.limit[u] = kind == 0 ? range.end - 10 * u : range.end;
                if (target <= T_LIMIT) expected += range.limit[u] - range.next[u];
            }
            check_cuda(cudaMemset(count, 0, sizeof(int)), "small filter memset");
            if (kind == 0) scan_pruned_kernel<<<1, BLOCK_SIZE>>>(job, range, hits, count, 400);
            if (kind == 1) scan_coset4_kernel<true><<<1, BLOCK_SIZE>>>(job, range, hits, count, 400);
            if (kind == 2) scan_coset4_kernel<false><<<1, BLOCK_SIZE>>>(job, range, hits, count, 400);
            check_cuda(cudaGetLastError(), "small filter launch");
            int observed = 0;
            check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "small filter count");
            if (observed != expected) {
                std::cerr << "small filter count mismatch target=" << target << " kind=" << kind
                          << " observed=" << observed << " expected=" << expected << "\n";
                return false;
            }
            std::vector<Hit> actual(observed);
            std::vector<bool> seen(400, false);
            if (observed)
                check_cuda(cudaMemcpy(actual.data(), hits, observed * sizeof(Hit), cudaMemcpyDeviceToHost), "small filter hits");
            for (const Hit& hit : actual) {
                u64 u = hit.x / COSET_Q, r = hit.x % COSET_Q;
                if (u >= 4 || r < range.next[u] || r >= range.limit[u] || hit.t != target) {
                    std::cerr << "small filter hit mismatch target=" << target << " kind=" << kind << " x=" << hit.x << " t=" << hit.t << "\n";
                    return false;
                }
                u64 index = 100 * u + r - begin;
                if (seen[index]) {
                    std::cerr << "small filter duplicate target=" << target << " kind=" << kind << " x=" << hit.x << "\n";
                    return false;
                }
                seen[index] = true;
            }
            for (int u = 0; u < 4; ++u) {
                for (u64 r = begin; r < range.end; ++r) {
                    bool required = target <= T_LIMIT && r >= range.next[u] && r < range.limit[u];
                    if (seen[100 * u + r - begin] != required) {
                        std::cerr << "small filter coverage mismatch target=" << target << " kind=" << kind
                                  << " u=" << u << " r=" << r << "\n";
                        return false;
                    }
                }
            }
            total_expected += expected;
        }
    }
    cudaFree(hits);
    cudaFree(count);
    if (total_expected != 3312) return false;
    std::cout << "SMALL_FILTER_PRODUCTION_PATHS_OK positive_hits=3312 rejected_t3_points=1104\n";
    return true;
}
#endif

static int run_coset(const std::string& name, int device, u64 begin, u64 end, u64 chunk, u64 blocks) {
    if (DIGITS != 19 || STEP != 5 || device < 0 || device >= 8 || L % 4 != 0 || blocks % 4 != 0)
        throw std::runtime_error("coset4 requires the eight-shard 19-digit job");
    check_cuda(cudaSetDevice(device), "set device");
    Job job = make_job(end, chunk);
    u64 rlo = COSET_Q * device / 8, rhi = COSET_Q * (device + 1) / 8;
    u64 x_limit = candidate_limit();
    auto [next, total_hits] = load_coset_state(name, device, begin, end, rlo, rhi, x_limit);
    auto limits = coverage_limits(rlo, rhi, x_limit);
    for (int u = 0; u < 4; ++u) next[u] = std::min(next[u], limits[u]);
    std::filesystem::create_directories("results15");
    Hit* d_hits;
    int* d_count;
    check_cuda(cudaMalloc(&d_hits, 65536 * sizeof(Hit)), "malloc hits");
    check_cuda(cudaMalloc(&d_count, sizeof(int)), "malloc count");
    save_coset_state(name, begin, end, rlo, rhi, next, limits, x_limit, total_hits, chunk, blocks);
    auto started = std::chrono::steady_clock::now();
    auto last = started;
    while (true) {
        u64 new_limit = candidate_limit();
        if (new_limit != x_limit) {
            if (new_limit > x_limit) throw std::runtime_error("candidate bound increased");
            x_limit = new_limit;
            limits = coverage_limits(rlo, rhi, x_limit);
            for (int u = 0; u < 4; ++u) next[u] = std::min(next[u], limits[u]);
            std::cout << "PRUNE name=" << name << " x_limit=" << x_limit << "\n";
        }
        save_coset_state(name, begin, end, rlo, rhi, next, limits, x_limit, total_hits, chunk, blocks);
        if (next == limits) break;
        CosetRange range{};
        range.begin = rhi;
        range.end = rlo;
        range.q = COSET_Q;
        range.h = powmod(job.g, COSET_Q);
        range.w_q = mulmod(job.ws, COSET_Q);
        u64 remaining = 0;
        int active = 0;
        for (int u = 0; u < 4; ++u) {
            range.next[u] = next[u];
            range.limit[u] = limits[u];
            if (next[u] < limits[u]) {
                range.begin = std::min(range.begin, next[u]);
                range.end = std::max(range.end, limits[u]);
                remaining = std::max(remaining, limits[u] - next[u]);
                ++active;
            }
        }
        u64 per_lane = std::max<u64>(std::min<u64>(blocks, (remaining + chunk - 1) / chunk), 1);
        u64 base_next = std::min(range.end, range.begin + per_lane * BLOCK_SIZE * chunk);
        bool aligned = x_limit == L && next == std::array<u64, 4>{range.begin, range.begin, range.begin, range.begin};
        check_cuda(cudaMemset(d_count, 0, sizeof(int)), "memset");
        if (aligned) scan_coset4_kernel<true><<<per_lane, BLOCK_SIZE>>>(job, range, d_hits, d_count, 65536);
        else if (x_limit == L) scan_coset4_kernel<<<per_lane, BLOCK_SIZE>>>(job, range, d_hits, d_count, 65536);
        else scan_pruned_kernel<<<per_lane, BLOCK_SIZE>>>(job, range, d_hits, d_count, 65536);
        check_cuda(cudaGetLastError(), "scan launch");
        int count = 0;
        check_cuda(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost), "count");
        if (count > 65536) throw std::runtime_error("hit overflow");
        u64 launched = 0;
        for (int u = 0; u < 4; ++u) {
            u64 advanced = std::min(limits[u], std::max(next[u], base_next));
            launched += advanced - next[u];
            next[u] = advanced;
        }
        if (count) {
            std::vector<Hit> hits(count);
            check_cuda(cudaMemcpy(hits.data(), d_hits, count * sizeof(Hit), cudaMemcpyDeviceToHost), "hits");
            std::ofstream f(hits_path(name), std::ios::app);
            for (const Hit& hit : hits) if (!verify_hit(hit, total_hits, f)) return 3;
        }
        u64 next_x = common_next(begin, end, rlo, rhi, next, limits);
        save_coset_state(name, begin, end, rlo, rhi, next, limits, x_limit, total_hits, chunk, blocks);
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - started).count();
        double dt = std::chrono::duration<double>(now - last).count();
        last = now;
        std::cout << "progress name=" << name << " next_x=" << next_x << "/" << end
                  << " rate=" << launched / dt << " x/s elapsed=" << elapsed << " hits=" << total_hits
                  << " active_cosets=" << active << " x_limit=" << x_limit << std::endl;
    }
    cudaFree(d_hits);
    cudaFree(d_count);
    std::cout << "DONE name=" << name << " next_x=" << end << " hits=" << total_hits << "\n";
    return 0;
}

static void benchmark(int device) {
    check_cuda(cudaSetDevice(device), "set device");
    const u64 chunk = 65536, blocks = 16384;
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
        for (int mode = 0; mode < 2; ++mode) {
            auto start = std::chrono::steady_clock::now();
            check_cuda(cudaMemset(count, 0, sizeof(int)), "benchmark memset");
            if (mode == 0) scan_kernel<<<blocks, BLOCK_SIZE>>>(serial_job, 0, hits, count, 65536);
            else scan_coset4_kernel<<<blocks, BLOCK_SIZE>>>(job, range, hits, count, 65536);
            check_cuda(cudaGetLastError(), "benchmark launch");
            int observed = 0;
            check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "benchmark sync");
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << "BENCHMARK mode=" << (mode ? "coset4" : "serial") << " repeat=" << repeat
                      << " tested=" << 4 * n << " rate=" << 4 * n / seconds << " x/s\n";
        }
    }
    cudaFree(hits);
    cudaFree(count);
}

#ifndef SEARCH15_PRUNE_LIBRARY
int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "selftest") {
            if (!coset_selftest(std::stoi(argv[2]))) return 1;
            #ifdef TEST_ALL_COSET_HITS
            return coset_filter_test() ? 0 : 1;
            #else
            return 0;
            #endif
        }
        if (argc == 3 && std::string(argv[1]) == "benchmark") {
            benchmark(std::stoi(argv[2]));
            return 0;
        }
        if (argc == 8 && std::string(argv[1]) == "run")
            return run_coset(argv[2], std::stoi(argv[3]), std::stoull(argv[4]), std::stoull(argv[5]),
                             std::stoull(argv[6]), std::stoull(argv[7]));
        std::cerr << "usage: search15_coset4 selftest device | run name device x_begin x_end chunk blocks\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
#endif
