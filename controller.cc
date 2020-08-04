#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <unordered_map>

#include <absl/strings/str_format.h>
#include <absl/synchronization/mutex.h>

#include "controller.h"
#include "lkvs_impl.h"

namespace elf {

class ConcreteController : public Controller {
public:
    ConcreteController() : update_thread(&ConcreteController::update_loop, this) {}
    ~ConcreteController() override {
        stop();
        update_thread.join();
    }

    int64_t join(const std::string &name, update_callback_t callback) override {
        int64_t id;
        {
            absl::MutexLock l(&worker_mux);
            check();
            active_workers++;
            worker_counter++;
            id = worker_counter;
            workers.emplace(std::make_pair(id, new WorkerHandle(id, name, callback)));
            commit();
        }
        return id;
    }

    void leave(int64_t id) override {
        absl::MutexLock l(&worker_mux);
        check();
        if (workers[id]->leave_at == std::numeric_limits<int64_t>::max()) {
            active_workers--;
            workers[id]->leave_at = conf_id + 1;
            commit();
        }
    }

    std::future<BeginBatchResult> begin_batch(int64_t id, int64_t ready_conf_id) override {
        std::future<BeginBatchResult> future;
        {
            absl::MutexLock l(&worker_mux);
            if (workers.at(id)->leave_at <= conf_states.back().conf_id) {
                // already left
                // TODO: conf_states may be empty on wrong usage
                std::promise<BeginBatchResult> p;
                p.set_value({-1, false});
                future = p.get_future();
                return future;
            }
            waiters.emplace_back(id, std::promise<BeginBatchResult>());
            future = waiters.back().second.get_future();
            auto conf_state = conf_states.begin();
            for (; conf_state != conf_states.end(); conf_state++) {
                if (conf_state->conf_id > ready_conf_id) {
                    continue;
                }
                if (conf_state->set_ready(id)) {
                    // the given state is ready
                    int64_t conf_id = conf_state->conf_id;
                    bool requires_broadcast = has_new_worker(conf_states.back(), *conf_state);
                    BeginBatchResult result(conf_id, requires_broadcast);
                    for (auto waiter_it = waiters.begin(); waiter_it != waiters.end();) {
                        if (conf_state->workers.find(waiter_it->first) !=
                            conf_state->workers.end()) {
                            // worker is active in this state
                            waiter_it->second.set_value(result);
                            waiter_it = waiters.erase(waiter_it);
                            continue;
                        }
                        if (workers.at(waiter_it->first)->leave_at <= conf_id) {
                            // the worker can leave
                            waiter_it->second.set_value({-1, false});
                            waiter_it = waiters.erase(waiter_it);
                            continue;
                        }
                        waiter_it++;
                    }
                    conf_state++;
                    for (auto to_clear = conf_states.begin(); to_clear < conf_state; to_clear++) {
                        to_clear->clear();
                    }
                    conf_states.erase(conf_state, conf_states.end());
                    break;
                }
            }
        }
        return future;
    }

    void end_batch(int64_t id) override {}

    void stop() override {
        absl::MutexLock l(&worker_mux);
        conf_id = -1;
    }

    void kv_set(int64_t conf_id, const std::string &key, const std::string &value) override {
        absl::MutexLock l(&kv_mux);
        kv[conf_id].set(key, value);
    }

    std::shared_future<std::string> kv_get(int64_t conf_id, const std::string &key) override {
        absl::MutexLock l(&kv_mux);
        return kv[conf_id].get(key);
    }

private:
    // tracking the last configuration
    int64_t conf_id = 0;
    int64_t processed_conf_id = 0;
    int64_t worker_counter = 0;
    int64_t active_workers = 0;
    struct WorkerHandle {
        int64_t id;
        std::string name;
        update_callback_t callback;
        int64_t leave_at;
        WorkerHandle(int64_t id, const std::string &name, update_callback_t callback)
            : id(id), name(name), callback(callback),
              leave_at(std::numeric_limits<int64_t>::max()) {}
    };
    absl::Mutex worker_mux;
    std::unordered_map<int64_t, std::unique_ptr<WorkerHandle>> workers;
    std::deque<std::pair<int64_t, std::promise<BeginBatchResult>>> waiters;
    std::thread update_thread;
    absl::Mutex kv_mux;
    std::unordered_map<int64_t, LocalKeyValueStore> kv;

    struct ConfState {
        int64_t conf_id;
        int64_t ready_count = 0;
        std::unordered_map<int64_t, bool> workers;
        ConfState(int64_t conf_id, const decltype(ConcreteController::workers) &wmap)
            : conf_id(conf_id) {
            for (auto &w : wmap) {
                if (w.second->leave_at > conf_id) {
                    // not left yet
                    workers[w.first] = false;
                }
            }
        }
        void clear() {
            for (auto &worker : workers) {
                worker.second = false;
            }
            ready_count = 0;
        }
        bool set_ready(int64_t id) {
            auto it = workers.find(id);
            if (it == workers.end()) {
                return false;
            }
            if (it->second == true) {
                return false;
            }
            it->second = true;
            ready_count++;
            return ready_count == workers.size();
        }
    };
    std::deque<ConfState> conf_states;

    void check() {
        if (conf_id == -1) {
            throw std::runtime_error("The controller is stopping and connot accept new requests");
        }
    }

    void update_loop() {
        while (true) {
            absl::MutexLock l(&worker_mux);
            worker_mux.Await(absl::Condition(
                +[](ConcreteController *c) -> bool { return c->conf_id != c->processed_conf_id; },
                this));
            processed_conf_id = conf_id;
            if (conf_id == -1) {
                break;
            }
            broadcast_updates();
        }
    }

    void broadcast_updates() {
        std::vector<WorkerHandle *> active_handles;
        for (auto &w : workers) {
            if (w.second->leave_at > conf_id) {
                active_handles.push_back(w.second.get());
            }
        }
        int64_t rank = 0;
        std::sort(active_handles.begin(), active_handles.end(),
            [](WorkerHandle *a, WorkerHandle *b) -> bool { return a->id < b->id; });
        for (auto &handle : active_handles) {
            if (auto callback = handle->callback) {
                callback({conf_id, rank, active_workers});
            }
            rank++;
        }
    }

    // commit changes
    void commit() {
        conf_id++;
        conf_states.emplace_front(conf_id, workers);
    }

    static bool has_new_worker(const ConfState &a, const ConfState &b) {
        for (auto &worker : b.workers) {
            if (a.workers.find(worker.first) == a.workers.end()) {
                return true;
            }
        }
        return false;
    }
};

std::unique_ptr<Controller> create_controller() {
    return std::make_unique<ConcreteController>();
}

} // namespace elf
