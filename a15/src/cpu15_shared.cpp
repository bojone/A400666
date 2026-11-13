#include <limits>
#include <mutex>

#define main cpu15_avx_original_main
#include "cpu15_avx.cpp"
#undef main

static std::string SHARED_NAME;
static std::string STOP_PATH;

static void save_shared(const Job& job, u64 next, int stopped) {
    std::string temporary = "cpu_results/" + SHARED_NAME + ".tmp";
    std::ofstream output(temporary);
    output << "config=" << job.d << ":" << job.a << ":" << job.begin << ":" << job.end << ":shared" << THREADS << "\n"
           << "next_x=" << next << "\nstopped=" << stopped << "\nclean_checkpoint=1\n";
    output.close();
    if (!output) throw std::runtime_error("failed to save shared CPU checkpoint");
    std::filesystem::rename(temporary, "cpu_results/" + SHARED_NAME + ".state");
}

static u64 load_shared(const Job& job) {
    std::ifstream input("cpu_results/" + SHARED_NAME + ".state");
    if (!input) return job.begin;
    std::string line;
    if (!std::getline(input, line) || line != "config=" + std::to_string(job.d) + ":" + std::to_string(job.a) +
        ":" + std::to_string(job.begin) + ":" + std::to_string(job.end) + ":shared" + std::to_string(THREADS))
        throw std::runtime_error("shared CPU state config mismatch");
    u64 next = job.begin;
    bool clean = false;
    while (std::getline(input, line)) {
        if (line.rfind("next_x=", 0) == 0) next = std::stoull(line.substr(7));
        if (line == "clean_checkpoint=1") clean = true;
    }
    if (!clean || next < job.begin || next > job.end) throw std::runtime_error("invalid shared CPU checkpoint");
    return next;
}

static int run_shared(int d, int a) {
    if (d != 18 || (a != 0 && a != 5)) throw std::runtime_error("shared CPU runner requires d=18 and a in {0,5}");
    SHARED_NAME = "d18_a" + std::to_string(a) + "_shared";
    STOP_PATH = "cpu_results/" + SHARED_NAME + ".stop";
    Job job = make_job(d, a);
    u64 next_work = load_shared(job);
    std::vector<u64> active(THREADS, std::numeric_limits<u64>::max());
    std::mutex mutex;
    std::atomic<u64> hits{0};
    std::atomic<bool> stop{false};
    auto started = std::chrono::steady_clock::now();
    save_shared(job, next_work, 0);
    #pragma omp parallel num_threads(THREADS)
    {
        int tid = omp_get_thread_num();
        auto last_save = started;
        while (true) {
            if (tid == 0 && !stop.load(std::memory_order_relaxed) && std::filesystem::exists(STOP_PATH))
                stop.store(true, std::memory_order_relaxed);
            if (stop.load(std::memory_order_relaxed)) break;
            u64 begin, end;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (next_work >= job.end) break;
                begin = next_work;
                end = std::min(begin + CHUNK, job.end);
                next_work = end;
                active[tid] = begin;
            }
            u64 local_hits = 0;
            scan_chunk(job, begin, end, local_hits);
            hits += local_hits;
            {
                std::lock_guard<std::mutex> lock(mutex);
                active[tid] = std::numeric_limits<u64>::max();
                if (tid == 0) {
                    auto now = std::chrono::steady_clock::now();
                    if (std::chrono::duration<double>(now - last_save).count() >= 30) {
                        u64 safe = next_work;
                        for (u64 value : active) safe = std::min(safe, value);
                        save_shared(job, safe, 0);
                        std::cout << "SHARED_PROGRESS next=" << safe << "/" << job.end
                                  << " allocated=" << next_work << " hits=" << hits.load() << std::endl;
                        last_save = now;
                    }
                }
            }
        }
        std::lock_guard<std::mutex> lock(mutex);
        active[tid] = std::numeric_limits<u64>::max();
    }
    save_shared(job, next_work, 1);
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << "SHARED_CPU_STOPPED next=" << next_work << "/" << job.end << " hits=" << hits.load()
              << " elapsed=" << elapsed << " clean_prefix=YES" << std::endl;
    return 0;
}

static bool shared_selftest() {
    const std::string path = "cpu_results/d02_a0.candidates";
    const std::string artifact = "k=15 remainder_p=0 independent_remainder_p=0\n";
    auto contents = [&]() {
        if (!std::filesystem::exists(path)) return std::string();
        std::string data(std::filesystem::file_size(path), '\0');
        std::ifstream input(path);
        input.read(data.data(), data.size());
        return data;
    };
    bool existed = std::filesystem::exists(path);
    std::string before = contents();
    bool ok = selftest();
    if (!existed && contents() == artifact) {
        std::filesystem::remove(path);
    } else if (existed && contents() == before + artifact) {
        std::ofstream output(path, std::ios::trunc);
        output << before;
    }
    return ok;
}

int main(int argc, char** argv) {
    try {
        std::filesystem::create_directories("cpu_results");
        if (argc == 2 && std::string(argv[1]) == "selftest") return shared_selftest() ? 0 : 1;
        if (argc == 4 && std::string(argv[1]) == "run") return run_shared(std::stoi(argv[2]), std::stoi(argv[3]));
        std::cerr << "usage: cpu15_shared selftest | run 18 a\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
}
