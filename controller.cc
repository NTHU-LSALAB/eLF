#include <future>
#include <iostream>
#include <mutex>

#include <absl/strings/str_format.h>
#include <absl/synchronization/mutex.h>

#include "controller.h"

int64_t Controller::join(const std::string &name, update_callback_t callback) {
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

void Controller::leave(int64_t id) {
    absl::MutexLock l(&worker_mux);
    check();
    workers.erase(id);
    commit();
}

void Controller::broadcast_updates() {
    int rank = 0;
    int64_t size = workers.size();
    for (auto &w : workers) {
        if (auto callback = w.second->callback) {
            callback(conf_id, rank, size);
        }
        rank++;
    }
}

std::future<int64_t> Controller::begin_batch(int64_t id, int64_t ready_conf_id) {
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

void Controller::end_batch(int64_t id) {}

void Controller::kv_set(int64_t conf_id, const std::string &key, const std::string &value) {}

std::string Controller::kv_get(int64_t conf_id, const std::string &key) {
    return "";
}

int main() {
    Controller c;
    std::cerr << c.join("", [](int64_t conf_id, int64_t rank, int64_t size) {
        std::cerr << absl::StrFormat("conf_id=%d rank=%d size=%d\n", conf_id, rank, size);
    }) << "\n";
    std::cerr << c.join("", [](int64_t conf_id, int64_t rank, int64_t size) {
        std::cerr << absl::StrFormat("conf_id=%d rank=%d size=%d\n", conf_id, rank, size);
    }) << "\n";
    auto one = c.begin_batch(1, 2);
    auto two = c.begin_batch(2, 2);
    std::cerr << one.get() << "\n";
    std::cerr << two.get() << "\n";
    c.stop();
}
