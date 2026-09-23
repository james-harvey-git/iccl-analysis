// Exact matching of eight unordered module pairs in one shared neuron coordinate system.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
constexpr int H = 16, T = 8, K = 17, S = H * H;
constexpr int MODULES = T * 2 * K * H, OUTPUTS = MODULES + S;
constexpr int STRIDE = (1 + 2 * T) * S;
constexpr double INF = std::numeric_limits<double>::infinity();
using Clock = std::chrono::steady_clock;
thread_local char last_error[1024] = {};

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
void check_costs(const double* costs, int size) {
    for (int i = 0; i < size; ++i)
        require(std::isfinite(costs[i]) && std::abs(costs[i]) < 1e280,
                "assignment costs must be finite and smaller than 1e280 in magnitude");
}
// Successive shortest augmenting paths with feasible row/column potentials.
double hungarian(const double* c, int n, int* permutation) {
    double u[H + 1] = {}, v[H + 1] = {};
    int matched[H + 1] = {}, predecessor[H + 1] = {};
    for (int row = 1; row <= n; ++row) {
        matched[0] = row;
        double distance[H + 1];
        std::fill(distance, distance + n + 1, INF);
        bool seen[H + 1] = {};
        int column = 0;
        do {
            seen[column] = true;
            const int current_row = matched[column];
            double step = INF;
            int next = 0;
            for (int j = 1; j <= n; ++j) {
                if (seen[j]) continue;
                const double reduced = c[(current_row - 1) * n + j - 1]
                                       - u[current_row] - v[j];
                if (reduced < distance[j]) {
                    distance[j] = reduced;
                    predecessor[j] = column;
                }
                if (distance[j] < step) {
                    step = distance[j];
                    next = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (seen[j]) {
                    u[matched[j]] += step;
                    v[j] -= step;
                } else {
                    distance[j] -= step;
                }
            }
            column = next;
        } while (matched[column] != 0);
        do {
            const int prev = predecessor[column];
            matched[column] = matched[prev];
            column = prev;
        } while (column);
    }
    for (int j = 1; j <= n; ++j) permutation[matched[j] - 1] = j - 1;
    double value = 0;
    for (int i = 0; i < n; ++i) value += c[i * n + permutation[i]];
    return value;
}

double select_sum(const double* c, const int* p) {
    double value = 0;
    for (int i = 0; i < H; ++i) value += c[i * H + p[i]];
    return value;
}

// A feasible dual and complementary partial matching can survive a cost change.
struct WarmAssignment {
    double u[H + 1] = {}, v[H + 1] = {};
    int matched[H + 1] = {};

    double prepare(const double* c) {
        int old_column[H + 1] = {};
        for (int j = 1; j <= H; ++j) old_column[matched[j]] = j;
        for (int i = 1; i <= H; ++i) {
            double minimum = INF;
            for (int j = 1; j <= H; ++j)
                minimum = std::min(minimum, c[(i - 1) * H + j - 1] - v[j]);
            u[i] = minimum;
            const int old = old_column[i];
            if (old && c[(i - 1) * H + old - 1] - v[old] != minimum) {
                matched[old] = 0;
                old_column[i] = 0;
            }
        }
        for (int i = 1; i <= H; ++i) {
            if (old_column[i]) continue;
            for (int j = 1; j <= H; ++j) {
                if (!matched[j] && c[(i - 1) * H + j - 1] - v[j] == u[i]) {
                    matched[j] = i;
                    break;
                }
            }
        }
        double lower = 0;
        for (int i = 1; i <= H; ++i) lower += u[i] + v[i];
        return lower;
    }

    void finish(const double* c, int* permutation) {
        bool matched_row[H + 1] = {};
        for (int j = 1; j <= H; ++j) matched_row[matched[j]] = true;
        for (int row = 1; row <= H; ++row) {
            if (matched_row[row]) continue;
            matched[0] = row;
            double distance[H + 1];
            std::fill(distance, distance + H + 1, INF);
            int predecessor[H + 1] = {};
            bool seen[H + 1] = {};
            int column = 0;
            do {
                seen[column] = true;
                const int current_row = matched[column];
                double step = INF;
                int next = 0;
                for (int j = 1; j <= H; ++j) {
                    if (seen[j]) continue;
                    const double reduced = c[(current_row - 1) * H + j - 1]
                                           - u[current_row] - v[j];
                    if (reduced < distance[j]) {
                        distance[j] = reduced;
                        predecessor[j] = column;
                    }
                    if (distance[j] < step) {
                        step = distance[j];
                        next = j;
                    }
                }
                for (int j = 0; j <= H; ++j) {
                    if (seen[j]) {
                        u[matched[j]] += step;
                        v[j] -= step;
                    } else {
                        distance[j] -= step;
                    }
                }
                column = next;
            } while (matched[column]);
            do {
                const int previous = predecessor[column];
                matched[column] = matched[previous];
                column = previous;
            } while (column);
        }
        for (int j = 1; j <= H; ++j) permutation[matched[j] - 1] = j - 1;
    }
};

struct Search {
    const double* costs;
    double best = INF;
    int best_mask = 0;
    int best_p[H] = {};
    std::int64_t solves = 0, nodes = 0;
    double delta[T][2][S];
    double tolerance;

    explicit Search(const double* c) : costs(c) {
        double scale = 0;
        for (int i = 0; i < STRIDE; ++i) scale = std::max(scale, std::abs(c[i]));
        tolerance = 64 * std::numeric_limits<double>::epsilon() * scale * H * (T + 1);
    }

    double actual_value(int mask, const int* p) const {
        double value = select_sum(costs, p);
        for (int t = 0; t < T; ++t)
            value += select_sum(costs + (1 + 2 * t + ((mask >> t) & 1)) * S, p);
        return value;
    }

    void save(int mask, const int* p) {
        const double value = actual_value(mask, p);
        if (value < best) {
            best = value;
            best_mask = mask;
            std::copy(p, p + H, best_p);
        }
    }

    void exhaustive() {
        double matrix[S];
        std::copy(costs, costs + S, matrix);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < S; ++k) matrix[k] += costs[(1 + 2 * t) * S + k];
        int old = 0;
        for (int step = 0; step < (1 << T); ++step) {
            const int mask = step ^ (step >> 1);
            if (step) {
                const int t = __builtin_ctz(static_cast<unsigned>(old ^ mask));
                const int bit = (mask >> t) & 1;
                for (int k = 0; k < S; ++k)
                    matrix[k] += costs[(1 + 2 * t + bit) * S + k]
                               - costs[(1 + 2 * t + 1 - bit) * S + k];
            }
            old = mask;
            ++nodes;
            int permutation[H];
            hungarian(matrix, H, permutation);
            ++solves;
            save(mask, permutation);
        }
    }

    void visit(const double* matrix, unsigned remaining, int fixed_mask,
               const WarmAssignment* previous_assignment = nullptr) {
        ++nodes;
        int permutation[H];
        WarmAssignment current;
        if (previous_assignment) current = *previous_assignment;
        double lower = current.prepare(matrix);
        if (lower >= best - tolerance) return;
        current.finish(matrix, permutation);
        lower = select_sum(matrix, permutation);
        ++solves;
        if (lower >= best - tolerance) return;

        int candidate_mask = fixed_mask;
        int branch_task = -1;
        double largest_gap = -1;
        unsigned tasks = remaining;
        while (tasks) {
            const int t = __builtin_ctz(tasks);
            const double gap0 = select_sum(delta[t][0], permutation);
            const double gap1 = select_sum(delta[t][1], permutation);
            const int bit = gap1 < gap0;
            candidate_mask |= bit << t;
            const double gap = std::min(gap0, gap1);
            if (gap > largest_gap) {
                largest_gap = gap;
                branch_task = t;
            }
            tasks &= tasks - 1;
        }
        save(candidate_mask, permutation);
        if (!remaining || lower >= best - tolerance) return;

        // Fix the swap whose elementwise relaxation was most inconsistent.
        const int preferred = (candidate_mask >> branch_task) & 1;
        const unsigned next = remaining ^ (1u << branch_task);
        for (int order = 0; order < 2; ++order) {
            const int bit = preferred ^ order;
            double child[S];
            for (int k = 0; k < S; ++k)
                child[k] = matrix[k] + delta[branch_task][bit][k];
            visit(child, next, fixed_mask | (bit << branch_task), &current);
        }
    }

    void branch_and_bound() {
        double matrix[S];
        std::copy(costs, costs + S, matrix);
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < S; ++k) {
                const double a = costs[(1 + 2 * t) * S + k];
                const double b = costs[(2 + 2 * t) * S + k];
                const double minimum = std::min(a, b);
                matrix[k] += minimum;
                delta[t][0][k] = a - minimum;
                delta[t][1][k] = b - minimum;
            }
        }
        visit(matrix, (1u << T) - 1, 0);
    }

};

void construct_one(const float* prediction, const float* target,
                   const float* pred_r, const float* target_r,
                   double readout_weight, double* costs) {
    std::fill(costs, costs + STRIDE, 0.0);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < H; ++j) {
            double total = 0;
            for (int k = 0; k < H; ++k) {
                const double d = static_cast<double>(pred_r[i * H + k]) - target_r[j * H + k];
                total += d * d;
            }
            costs[i * H + j] = readout_weight * total;
        }
    }
    for (int t = 0; t < T; ++t) {
        for (int bit = 0; bit < 2; ++bit) {
            double* matrix = costs + (1 + 2 * t + bit) * S;
            for (int a = 0; a < 2; ++a) {
                for (int k = 0; k < K; ++k) {
                    const float* predicted = prediction + ((t * 2 + a) * K + k) * H;
                    const float* truth = target + ((t * 2 + (a ^ bit)) * K + k) * H;
                    for (int i = 0; i < H; ++i) {
                        for (int j = 0; j < H; ++j) {
                            const double d = static_cast<double>(predicted[i]) - truth[j];
                            matrix[i * H + j] += d * d;
                        }
                    }
                }
            }
        }
    }
}

struct Job {
    const double* costs = nullptr;
    const float *prediction = nullptr, *target = nullptr;
    double *values = nullptr, *timings = nullptr;
    int *masks = nullptr, *permutations = nullptr;
    std::int64_t* stats = nullptr;
    std::size_t batch = 0;
    double readout_weight = 1.0;
    bool exhaustive = false, profile = false;
};

class Pool {
    std::vector<std::thread> workers;
    std::mutex mutex, run_mutex, failure_mutex;
    std::condition_variable start, finished;
    std::uint64_t generation = 0;
    bool stop = false;
    int active = 0;
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr failure;
    Job job;

    void process() noexcept {
        try {
            std::size_t e;
            while (!failed && (e = next.fetch_add(1)) < job.batch) {
                double constructed[STRIDE];
                const double* costs = job.costs ? job.costs + e * STRIDE : constructed;
                const auto before = job.profile ? Clock::now() : Clock::time_point{};
                if (!job.costs) {
                    const float* pred = job.prediction + e * OUTPUTS;
                    const float* target = job.target + e * OUTPUTS;
                    construct_one(pred, target, pred + MODULES, target + MODULES,
                                  job.readout_weight, constructed);
                }
                check_costs(costs, STRIDE);
                const auto middle = job.profile ? Clock::now() : Clock::time_point{};
                Search search(costs);
                if (job.exhaustive) search.exhaustive();
                else search.branch_and_bound();
                job.values[e] = search.best;
                job.masks[e] = search.best_mask;
                std::copy(search.best_p, search.best_p + H, job.permutations + e * H);
                job.stats[2 * e] = search.solves;
                job.stats[2 * e + 1] = search.nodes;
                job.timings[2 * e] = job.profile
                    ? std::chrono::duration<double>(middle - before).count() : 0;
                job.timings[2 * e + 1] = job.profile
                    ? std::chrono::duration<double>(Clock::now() - middle).count() : 0;
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) failure = std::current_exception();
            failed = true;
        }
    }

    void shutdown() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        start.notify_all();
        for (auto& worker : workers) if (worker.joinable()) worker.join();
    }

public:
    explicit Pool(int threads) {
        require(threads >= 1 && threads <= 256, "native thread count must be in [1, 256]");
        if (threads == 1) return;
        try {
            for (int w = 0; w < threads; ++w) workers.emplace_back([this] {
                std::uint64_t observed = 0;
                std::unique_lock<std::mutex> lock(mutex);
                while (true) {
                    start.wait(lock, [&] { return stop || generation != observed; });
                    if (stop) return;
                    observed = generation;
                    lock.unlock();
                    process();
                    lock.lock();
                    if (--active == 0) finished.notify_one();
                }
            });
        } catch (...) {
            shutdown();
            throw;
        }
    }

    void run(Job input) {
        std::lock_guard<std::mutex> serial(run_mutex);
        job = input;
        next = 0;
        failed = false;
        failure = nullptr;
        if (workers.empty()) process();
        else {
            std::unique_lock<std::mutex> lock(mutex);
            active = static_cast<int>(workers.size());
            ++generation;
            start.notify_all();
            finished.wait(lock, [&] { return active == 0; });
        }
        if (failure) std::rethrow_exception(failure);
    }

    ~Pool() { shutdown(); }
};

template <typename Action> int guarded(Action action) noexcept {
    last_error[0] = 0;
    try { action(); return 0; }
    catch (const std::exception& error) {
        std::snprintf(last_error, sizeof(last_error), "%s", error.what());
    } catch (...) {
        std::snprintf(last_error, sizeof(last_error), "unknown native assignment error");
    }
    return -1;
}
}  // namespace

extern "C" {
const char* probe_last_error() noexcept { return last_error; }
void* probe_create_pool(int threads) noexcept {
    Pool* pool = nullptr;
    guarded([&] { pool = new Pool(threads); });
    return pool;
}
void probe_destroy_pool(void* pool) noexcept { delete static_cast<Pool*>(pool); }

int probe_match(void* pool, const float* prediction, const float* target,
                const double* costs, std::size_t batch, double readout_weight,
                int exhaustive, int profile, double* values, int* masks,
                int* permutations, std::int64_t* stats, double* timings) noexcept {
    return guarded([&] {
        require(pool != nullptr, "null assignment pool");
        require(std::isfinite(readout_weight) && readout_weight > 0,
                "readout_weight must be positive and finite");
        require(exhaustive == 0 || exhaustive == 1, "invalid matching mode");
        if (batch == 0) return;
        require((costs || (prediction && target)) && values && masks && permutations
                && stats && timings, "null assignment buffer");
        Job job;
        job.costs = costs; job.prediction = prediction; job.target = target;
        job.batch = batch; job.readout_weight = readout_weight;
        job.exhaustive = exhaustive; job.profile = profile;
        job.values = values; job.masks = masks; job.permutations = permutations;
        job.stats = stats; job.timings = timings;
        static_cast<Pool*>(pool)->run(job);
    });
}

int probe_single(const double* costs, int n, double* value, int* permutation) noexcept {
    return guarded([&] {
        require(costs && value && permutation && n >= 1 && n <= H,
                "single assignment requires a square matrix of order 1 through 16");
        check_costs(costs, n * n);
        *value = hungarian(costs, n, permutation);
    });
}

int probe_construct(const float* prediction, const float* target, std::size_t batch,
                    double weight, double* costs) noexcept {
    return guarded([&] {
        require(std::isfinite(weight) && weight > 0, "invalid readout weight");
        require(batch == 0 || (prediction && target && costs), "null cost buffer");
        for (std::size_t e = 0; e < batch; ++e) {
            const float* p = prediction + e * OUTPUTS;
            const float* t = target + e * OUTPUTS;
            construct_one(p, t, p + MODULES, t + MODULES, weight, costs + e * STRIDE);
            check_costs(costs + e * STRIDE, STRIDE);
        }
    });
}
}
