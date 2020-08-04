#include <future>
#include <iostream>
#include <mutex>
#include <unordered_map>

#include <absl/strings/str_format.h>
#include <absl/synchronization/mutex.h>

#include "controller.h"

class LocalController : public Controller {
public:
    LocalController() : update_thread(&LocalController::update_loop, this) {}
    ~LocalController() override {
        stop();
        update_thread.join();
    }

    int64_t join(const std::string &name, update_callback_t callback) override {
        int64_t id;
        {
            absl::MutexLock l(&worker_mux);
            check();
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
        workers.erase(id);
        commit();
    }

    std::future<int64_t> begin_batch(int64_t id, int64_t ready_conf_id) override {

        std::future<int64_t> future;
        {
            absl::MutexLock l(&waiter_mux);
            waiters[id] = std::promise<int64_t>();
            future = waiters[id].get_future();
        }
        {
            absl::MutexLock l(&worker_mux);
            auto conf_state = conf_states.begin();
            for (; conf_state != conf_states.end(); conf_state++) {
                if (conf_state->conf_id > ready_conf_id) {
                    break;
                }
                if (conf_state->set_ready(id)) {
                    int64_t conf_id = conf_state->conf_id;
                    for (auto &w : conf_state->workers) {
                        waiters.at(w.first).set_value(conf_id);
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

private:
    int64_t conf_id = 0;
    int64_t processed_conf_id = 0;
    int64_t worker_counter = 0;
    struct WorkerHandle {
        int64_t id;
        std::string name;
        update_callback_t callback;
        WorkerHandle(int64_t id, const std::string &name, update_callback_t callback)
            : id(id), name(name), callback(callback) {}
    };
    absl::Mutex worker_mux;
    std::unordered_map<int64_t, std::unique_ptr<WorkerHandle>> workers;
    absl::Mutex waiter_mux;
    std::unordered_map<int64_t, std::promise<int64_t>> waiters;
    std::thread update_thread;

    struct ConfState {
        int64_t conf_id;
        int64_t ready_count = 0;
        std::unordered_map<int64_t, bool> workers;
        ConfState(int64_t conf_id, const typeof(LocalController::workers) &wmap)
            : conf_id(conf_id) {
            for (auto &w : wmap) {
                workers[w.first] = false;
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
                +[](LocalController *c) -> bool { return c->conf_id != c->processed_conf_id; },
                this));
            processed_conf_id = conf_id;
            if (conf_id == -1) {
                break;
            }
            broadcast_updates();
        }
    }

    void broadcast_updates() {
        int rank = 0;
        int64_t size = workers.size();
        for (auto &w : workers) {
            if (auto callback = w.second->callback) {
                callback(conf_id, rank, size);
            }
            rank++;
        }
    }

    void kv_set(int64_t conf_id, const std::string &key, const std::string &value) override {}

    std::string kv_get(int64_t conf_id, const std::string &key) override {
        return "not implemented";
    }

    // commit changes
    void commit() {
        conf_id++;
        conf_states.emplace_front(conf_id, workers);
    }
};

std::unique_ptr<Controller> create_controller() {
    return std::make_unique<LocalController>();
}
