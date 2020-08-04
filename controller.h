#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include <iostream>

#include <absl/synchronization/mutex.h>

class Controller {
public:
    Controller() : update_thread(&Controller::update_loop, this) {}
    Controller(const Controller &) = delete;
    ~Controller() {
        update_thread.join();
    }

    // the callback for worker pool configuration updates
    // conf_id: serial number of the configuration
    // rank: the rank of the worker in this configuration
    // size: the total number of workers in this configuration
    using update_callback_t = std::function<void(int64_t conf_id, int64_t rank, int64_t size)>;

    // joined the worker named by the given name
    // name: the name of the worker
    // callback: the callback function for updates
    // returns the id of the worker; the id is used in other methods of the controller
    int64_t join(const std::string &name, update_callback_t callback);

    // make worker with the supplied id gracefully leave the worker pool
    void leave(int64_t id);

    // start a training batch
    // id: worker identifier
    // ready_conf_id: the configuration that the worker is ready for
    // returns the configuration id to use in this batch
    // -1 is returned if training should be stopped
    int64_t begin_batch(int64_t id, int64_t ready_conf_id);

    // indicate that the batch is finished
    // only used for profiling
    void end_batch(int64_t id);

    // stop the controller
    void stop() {
        {
            absl::MutexLock l(&worker_mux);
            conf_id = -1;
        }
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

    void check() {
        if (conf_id == -1) {
            throw std::runtime_error("The controller is stopping and connot accept new requests");
        }
    }

    std::thread update_thread;
    void update_loop() {
        while (true) {
            absl::MutexLock l(&worker_mux);
            worker_mux.Await(absl::Condition(
                +[](Controller *c) -> bool { return c->conf_id != c->processed_conf_id; }, this));
            processed_conf_id = conf_id;
            if (conf_id == -1) {
                break;
            }
            broadcast_updates();
        }
    }
    void broadcast_updates();
};
