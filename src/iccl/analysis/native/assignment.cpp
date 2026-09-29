// Exact matching of four modules and a fixed readout in one shared neuron basis.
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
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
constexpr int H = 16, M = 4, K = 17, S = H * H;
constexpr int OUTPUTS = S + M * K * H, STRIDE = (1 + M * M) * S;
constexpr int COUNTERS = 3;
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
    int best_q[M] = {}, best_p[H] = {};
    std::int64_t candidates = 0, solves = 0, skips = 0;
    double tolerance;

    explicit Search(const double* c) : costs(c) {
        double scale = 0;
        for (int i = 0; i < STRIDE; ++i) scale = std::max(scale, std::abs(c[i]));
        tolerance = 256 * std::numeric_limits<double>::epsilon() * scale * H * (M + 1);
    }

    const double* edge(int a, int b) const { return costs + (1 + a * M + b) * S; }

    double value(const int* q, const int* p) const {
        double total = select_sum(costs, p);
        for (int a = 0; a < M; ++a) total += select_sum(edge(a, q[a]), p);
        return total;
    }

    void enumerate(bool reuse) {
        int q[M], p[H];
        std::iota(q, q + M, 0);
        WarmAssignment previous;
        do {
            ++candidates;
            double matrix[S];
            std::copy(costs, costs + S, matrix);
            for (int a = 0; a < M; ++a) {
                const double* addition = edge(a, q[a]);
                for (int x = 0; x < S; ++x) matrix[x] += addition[x];
            }
            if (reuse) {
                const double lower = previous.prepare(matrix);
                // Roundoff must make pruning more conservative, never skip a near improvement.
                if (lower >= best + tolerance) { ++skips; continue; }
                previous.finish(matrix, p);
            } else {
                hungarian(matrix, H, p);
            }
            ++solves;
            const double score = value(q, p);
            if (score < best) {
                best = score;
                std::copy(q, q + M, best_q);
                std::copy(p, p + H, best_p);
            }
        } while (std::next_permutation(q, q + M));
    }
};

void construct_one(const float* prediction, const float* target,
                   double readout_weight, double* costs) {
    // Direct squared differences avoid cancellation for exactly reconstructed sets.
    std::fill(costs, costs + STRIDE, 0.0);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < H; ++j) {
            double total = 0;
            for (int k = 0; k < H; ++k) {
                const double d = static_cast<double>(prediction[i * H + k]) - target[j * H + k];
                total += d * d;
            }
            costs[i * H + j] = readout_weight * total;
        }
    }
    for (int a = 0; a < M; ++a) {
        for (int b = 0; b < M; ++b) {
            double* matrix = costs + (1 + a * M + b) * S;
            for (int k = 0; k < K; ++k) {
                const float* predicted = prediction + S + (a * K + k) * H;
                const float* truth = target + S + (b * K + k) * H;
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

struct Job {
    const double* costs = nullptr;
    const float *prediction = nullptr, *target = nullptr;
    double *values = nullptr, *timings = nullptr;
    int *module_permutations = nullptr, *hidden_permutations = nullptr;
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
                    construct_one(pred, target, job.readout_weight, constructed);
                }
                check_costs(costs, STRIDE);
                const auto middle = job.profile ? Clock::now() : Clock::time_point{};
                Search search(costs);
                search.enumerate(!job.exhaustive);
                job.values[e] = search.best;
                std::copy(search.best_q, search.best_q + M, job.module_permutations + e * M);
                std::copy(search.best_p, search.best_p + H, job.hidden_permutations + e * H);
                job.stats[COUNTERS * e] = search.candidates;
                job.stats[COUNTERS * e + 1] = search.solves;
                job.stats[COUNTERS * e + 2] = search.skips;
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
                int exhaustive, int profile, double* values, int* module_permutations,
                int* hidden_permutations, std::int64_t* stats, double* timings) noexcept {
    return guarded([&] {
        require(pool != nullptr, "null assignment pool");
        require(std::isfinite(readout_weight) && readout_weight > 0,
                "readout_weight must be positive and finite");
        require(exhaustive == 0 || exhaustive == 1, "invalid matching mode");
        if (batch == 0) return;
        require((costs || (prediction && target)) && values && module_permutations && hidden_permutations
                && stats && timings, "null assignment buffer");
        Job job;
        job.costs = costs; job.prediction = prediction; job.target = target;
        job.batch = batch; job.readout_weight = readout_weight;
        job.exhaustive = exhaustive; job.profile = profile;
        job.values = values; job.module_permutations = module_permutations; job.hidden_permutations = hidden_permutations;
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
            construct_one(p, t, weight, costs + e * STRIDE);
            check_costs(costs + e * STRIDE, STRIDE);
        }
    });
}
}
