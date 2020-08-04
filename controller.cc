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
        conf_id++;
        worker_counter++;
        id = worker_counter;
        workers.emplace(std::make_pair(id, new WorkerHandle(id, name, callback)));
    }
    return id;
}

void Controller::leave(int64_t id) {
    absl::MutexLock l(&worker_mux);
    check();
    conf_id++;
    workers.erase(id);
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

int main() {
    Controller c;
    std::cerr << c.join("", [](int64_t conf_id, int64_t rank, int64_t size) {
        std::cerr << absl::StrFormat("conf_id=%d rank=%d size=%d\n", conf_id, rank, size);
    }) << "\n";
    c.leave(1);
    c.stop();
}
