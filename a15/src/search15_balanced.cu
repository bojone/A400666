#define SEARCH15_PRUNE_LIBRARY
#include "search15_prune.cu"

struct BalanceSpec {
    std::string config, name;
    int device, source;
    u64 begin, end;
    std::array<u64, 4> initial;
};

static std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> parts;
    size_t begin = 0;
    while (true) {
        size_t end = text.find(delimiter, begin);
        parts.push_back(text.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) return parts;
        begin = end + 1;
    }
}

static BalanceSpec load_spec(const std::string& name, int device) {
    std::ifstream input("results15/d19_balance_jobs.txt");
    if (!input) throw std::runtime_error("missing balanced job manifest");
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::map<std::string, std::string> fields;
        for (const std::string& part : split(line, ';')) {
            size_t equals = part.find('=');
            if (equals == std::string::npos) throw std::runtime_error("invalid balanced job field");
            fields[part.substr(0, equals)] = part.substr(equals + 1);
        }
        if (fields.at("name") != name) continue;
        BalanceSpec spec{};
        spec.config = line;
        spec.name = name;
        spec.device = std::stoi(fields.at("device"));
        spec.source = std::stoi(fields.at("source"));
        spec.begin = std::stoull(fields.at("r_begin"));
        spec.end = std::stoull(fields.at("r_end"));
        auto initial = split(fields.at("initial"), ',');
        if (initial.size() != 4) throw std::runtime_error("invalid balanced initial checkpoint");
        for (int u = 0; u < 4; ++u) spec.initial[u] = std::stoull(initial[u]);
        if (spec.device != device || spec.device < 0 || spec.device >= 8 ||
            spec.source < 0 || spec.source >= 8 || spec.begin >= spec.end || spec.end > COSET_Q)
            throw std::runtime_error("balanced job configuration mismatch");
        for (u64 next : spec.initial)
            if (next < spec.begin || next > spec.end) throw std::runtime_error("balanced initial out of range");
        return spec;
    }
    throw std::runtime_error("balanced job not found: " + name);
}

static u64 balanced_next(const BalanceSpec& spec, const std::array<u64, 4>& next, u64 x_limit) {
    auto limits = coverage_limits(spec.begin, spec.end, x_limit);
    u128 done = 0;
    u64 total = 4 * (spec.end - spec.begin);
    for (int u = 0; u < 4; ++u) done += next[u] - spec.begin + spec.end - limits[u];
    return spec.begin + (u64)((spec.end - spec.begin) * done / total);
}

static void save_balanced_state(const BalanceSpec& spec, const std::array<u64, 4>& next,
                                u64 x_limit, u64 hits, u64 chunk, u64 blocks) {
    std::string temporary = state_path(spec.name) + ".tmp";
    std::ofstream output(temporary);
    output << "config=" << spec.config << "\nnext_x=" << balanced_next(spec, next, x_limit) << "\nhits=" << hits
           << "\nchunk=" << chunk << "\nblocks=" << blocks << "\nmode=balanced\nq=" << COSET_Q
           << "\nsource=" << spec.source << "\nr_begin=" << spec.begin << "\nr_end=" << spec.end
           << "\nx_limit=" << x_limit << "\n";
    for (int u = 0; u < 4; ++u) output << "next_r" << u << "=" << next[u] << "\n";
    output.close();
    if (!output) throw std::runtime_error("failed to save balanced checkpoint");
    std::filesystem::rename(temporary, state_path(spec.name));
}

static std::pair<std::array<u64, 4>, u64> load_balanced_state(
        const BalanceSpec& spec, u64 x_limit) {
    if (!std::filesystem::exists(state_path(spec.name))) return {spec.initial, 0};
    auto values = read_state_file(state_path(spec.name));
    if (values.at("config") != spec.config || values.at("mode") != "balanced" ||
        value_u64(values, "q") != COSET_Q || value_u64(values, "source") != spec.source ||
        value_u64(values, "r_begin") != spec.begin || value_u64(values, "r_end") != spec.end)
        throw std::runtime_error("balanced checkpoint configuration mismatch");
    u64 stored_limit = values.count("x_limit") ? value_u64(values, "x_limit") : L;
    if (stored_limit < x_limit) throw std::runtime_error("candidate history missing from balanced checkpoint");
    auto limits = coverage_limits(spec.begin, spec.end, stored_limit);
    std::array<u64, 4> next;
    for (int u = 0; u < 4; ++u) {
        next[u] = value_u64(values, "next_r" + std::to_string(u));
        if (next[u] < std::min(spec.initial[u], limits[u]) || next[u] > limits[u])
            throw std::runtime_error("balanced checkpoint out of range");
    }
    if (value_u64(values, "next_x") != balanced_next(spec, next, stored_limit))
        throw std::runtime_error("balanced checkpoint progress mismatch");
    return {next, value_u64(values, "hits")};
}

static int run_balanced(const std::string& name, int device, u64 chunk, u64 blocks) {
    if (DIGITS != 19 || STEP != 5 || L != 4 * COSET_Q)
        throw std::runtime_error("balanced scanner requires the 19-digit job");
    check_cuda(cudaSetDevice(device), "set balanced device");
    BalanceSpec spec = load_spec(name, device);
    Job job = make_job(L, chunk);
    u64 x_limit = candidate_limit();
    auto [next, total_hits] = load_balanced_state(spec, x_limit);
    auto limits = coverage_limits(spec.begin, spec.end, x_limit);
    for (int u = 0; u < 4; ++u) next[u] = std::min(next[u], limits[u]);
    std::filesystem::create_directories("results15");
    Hit* d_hits;
    int* d_count;
    check_cuda(cudaMalloc(&d_hits, 65536 * sizeof(Hit)), "malloc balanced hits");
    check_cuda(cudaMalloc(&d_count, sizeof(int)), "malloc balanced count");
    auto started = std::chrono::steady_clock::now();
    auto last = started;
    while (true) {
        u64 new_limit = candidate_limit();
        if (new_limit != x_limit) {
            if (new_limit > x_limit) throw std::runtime_error("candidate bound increased");
            x_limit = new_limit;
            limits = coverage_limits(spec.begin, spec.end, x_limit);
            for (int u = 0; u < 4; ++u) next[u] = std::min(next[u], limits[u]);
            std::cout << "PRUNE name=" << name << " x_limit=" << x_limit << "\n";
        }
        save_balanced_state(spec, next, x_limit, total_hits, chunk, blocks);
        if (next == limits) break;
        CosetRange range{};
        range.begin = spec.end;
        range.end = spec.begin;
        range.q = COSET_Q;
        range.h = powmod(job.g, COSET_Q);
        range.w_q = mulmod(job.ws, COSET_Q);
        u64 remaining = 0;
        for (int u = 0; u < 4; ++u) {
            range.next[u] = next[u];
            range.limit[u] = limits[u];
            if (next[u] < limits[u]) {
                range.begin = std::min(range.begin, next[u]);
                range.end = std::max(range.end, limits[u]);
                remaining = std::max(remaining, limits[u] - next[u]);
            }
        }
        u64 launch_blocks = std::max<u64>(std::min<u64>(blocks, (remaining + chunk - 1) / chunk), 1);
        u64 base_next = std::min(range.end, range.begin + launch_blocks * BLOCK_SIZE * chunk);
        bool aligned = x_limit == L && next == std::array<u64, 4>{range.begin, range.begin, range.begin, range.begin};
        check_cuda(cudaMemset(d_count, 0, sizeof(int)), "balanced memset");
        if (aligned) scan_coset4_kernel<true><<<launch_blocks, BLOCK_SIZE>>>(job, range, d_hits, d_count, 65536);
        else if (x_limit == L) scan_coset4_kernel<false><<<launch_blocks, BLOCK_SIZE>>>(job, range, d_hits, d_count, 65536);
        else scan_pruned_kernel<<<launch_blocks, BLOCK_SIZE>>>(job, range, d_hits, d_count, 65536);
        check_cuda(cudaGetLastError(), "balanced scan launch");
        int count = 0;
        check_cuda(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost), "balanced count");
        if (count > 65536) throw std::runtime_error("balanced hit overflow");
        u64 launched = 0;
        for (int u = 0; u < 4; ++u) {
            u64 advanced = std::min(limits[u], std::max(next[u], base_next));
            launched += advanced - next[u];
            next[u] = advanced;
        }
        if (count) {
            std::vector<Hit> hits(count);
            check_cuda(cudaMemcpy(hits.data(), d_hits, count * sizeof(Hit), cudaMemcpyDeviceToHost), "balanced hits");
            std::ofstream output(hits_path(name), std::ios::app);
            for (const Hit& hit : hits) if (!verify_hit(hit, total_hits, output)) return 3;
        }
        save_balanced_state(spec, next, x_limit, total_hits, chunk, blocks);
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - started).count();
        double dt = std::chrono::duration<double>(now - last).count();
        last = now;
        std::cout << "progress name=" << name << " next_x=" << balanced_next(spec, next, x_limit) << "/" << spec.end
                  << " rate=" << launched / dt << " x/s elapsed=" << elapsed << " hits=" << total_hits
                  << " x_limit=" << x_limit << " r_begin=" << spec.begin << " r_end=" << spec.end << std::endl;
    }
    cudaFree(d_hits);
    cudaFree(d_count);
    std::cout << "DONE name=" << name << " next_x=" << spec.end << " hits=" << total_hits << "\n";
    return 0;
}

#ifdef TEST_ALL_COSET_HITS
static bool balanced_filter_test() {
    Job job = make_job(1, 100);
    const u64 begin = 1234567, end = begin + 100;
    CosetRange range{};
    range.q = COSET_Q;
    range.h = powmod(job.g, COSET_Q);
    range.w_q = mulmod(job.ws, COSET_Q);
    Hit* hits;
    int* count;
    check_cuda(cudaMalloc(&hits, 400 * sizeof(Hit)), "malloc balanced filter hits");
    check_cuda(cudaMalloc(&count, sizeof(int)), "malloc balanced filter count");
    for (int trial = 0; trial < 2; ++trial) {
        range.begin = begin;
        range.end = end;
        int expected = 0;
        for (int u = 0; u < 4; ++u) {
            range.next[u] = trial ? begin + 20 : begin + 3 * u;
            range.limit[u] = trial ? end : end - 10 * u;
            expected += range.limit[u] - range.next[u];
        }
        if (trial) range.begin = begin + 20;
        check_cuda(cudaMemset(count, 0, sizeof(int)), "balanced filter memset");
        if (trial) scan_coset4_kernel<true><<<1, BLOCK_SIZE>>>(job, range, hits, count, 400);
        else scan_pruned_kernel<<<1, BLOCK_SIZE>>>(job, range, hits, count, 400);
        check_cuda(cudaGetLastError(), "balanced filter launch");
        int observed = 0;
        check_cuda(cudaMemcpy(&observed, count, sizeof(int), cudaMemcpyDeviceToHost), "balanced filter count");
        if (observed != expected) {
            std::cerr << "balanced filter count " << observed << " instead of " << expected << "\n";
            return false;
        }
        std::vector<Hit> actual(observed);
        std::vector<bool> seen(400, false);
        check_cuda(cudaMemcpy(actual.data(), hits, observed * sizeof(Hit), cudaMemcpyDeviceToHost), "balanced filter hits");
        for (const Hit& hit : actual) {
            u64 u = hit.x / COSET_Q, r = hit.x % COSET_Q;
            if (u >= 4 || r < range.next[u] || r >= range.limit[u]) return false;
            u64 index = 100 * u + r - begin;
            if (seen[index]) return false;
            u64 z = mulmod(job.z0, powmod(job.g, hit.x));
            u64 w = addmod(mulmod(job.ws, hit.x), job.w0);
            if (hit.t != addmod(z, w)) return false;
            seen[index] = true;
        }
        for (int u = 0; u < 4; ++u) {
            for (u64 r = begin; r < end; ++r) {
                bool required = r >= range.next[u] && r < range.limit[u];
                if (seen[100 * u + r - begin] != required) return false;
            }
        }
    }
    cudaFree(hits);
    cudaFree(count);
    std::cout << "BALANCED_FILTER_OK arbitrary_range=1234567:1234667 pruned_points=259 aligned_points=320\n";
    return true;
}
#endif

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "selftest") {
            if (!coset_selftest(std::stoi(argv[2]))) return 1;
            #ifdef TEST_ALL_COSET_HITS
            if (!coset_filter_test() || !balanced_filter_test()) return 1;
            #endif
            #ifdef TEST_SMALL_FILTER
            if (!small_filter_test()) return 1;
            #endif
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "run")
            return run_balanced(argv[2], std::stoi(argv[3]), std::stoull(argv[4]), std::stoull(argv[5]));
        std::cerr << "usage: search15_balanced selftest device | run name device chunk blocks\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
}
