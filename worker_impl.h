#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <absl/synchronization/mutex.h>
#include <thread_pool.h>

#include "communicator.h"
#include "controller.h"
#include "kvs.h"
#include "lkvs_impl.h"
#include "nccl_communicator.h"

class WorkerConf {
public:
    const int64_t id;
    const int64_t rank;
    const int64_t size;

private:
    ThreadPool pool;
    absl::Mutex comms_mux;
    std::unordered_map<std::string, std::unique_ptr<Communicator>> communicators;
    std::shared_ptr<KeyValueStore> kvs;
    int64_t todo;
    bool ready_ = false;

public:
    WorkerConf(int64_t id,
        int64_t rank,
        int64_t size,
        std::shared_ptr<KeyValueStore> kvs,
        const std::vector<std::string> &identifiers)
        : id(id), rank(rank), size(size), pool(8), kvs(kvs), todo(identifiers.size()) {
        for (auto &identifier : identifiers) {
            pool.Schedule([this, &identifier]() { create_communicator_for_variable(identifier); });
        }
    }

    bool ready() {
        absl::ReaderMutexLock l(&comms_mux);
        return ready_;
    }

    void wait_ready() {
        absl::ReaderMutexLock l(&comms_mux);
        comms_mux.Await(absl::Condition(&ready_));
    }

private:
    void create_communicator_for_variable(std::string variable) {
        auto comm = create_nccl_communicator(kvs.get(), variable, rank, size);
        absl::MutexLock l(&comms_mux);
        communicators[variable] = std::move(comm);
        --todo;
        if (todo == 0) {
            ready_ = true;
        }
    }
};

class ControllerKVSAdapter : public KeyValueStore {
    Controller *ctrl;
    int64_t conf_id;

public:
    ControllerKVSAdapter(Controller *ctrl, int64_t conf_id) : ctrl(ctrl), conf_id(conf_id) {}
    ~ControllerKVSAdapter() override {}

    void set(const std::string &k, const std::string &v) override {
        ctrl->kv_set(conf_id, k, v);
    }
    std::shared_future<std::string> get(const std::string &k) override {
        return ctrl->kv_get(conf_id, k);
    }
};

class Worker {
    Controller *ctrl;
    int64_t id = -2222;
    std::vector<std::string> global_variables;
    std::vector<std::string> weight_variables;

    std::list<WorkerConf> confs;
    absl::Mutex conf_mux;
    bool first_conf_pushed = false;

    std::vector<std::string> identifiers;

    decltype(confs)::iterator current_conf;
    decltype(confs)::iterator ready_conf;

public:
    Worker(Controller *ctrl) : ctrl(ctrl) {}
    Worker(const Worker &) = delete;
    ~Worker() {}

    void join(const std::string &name = "") {
        for (auto& gv: global_variables) {
            identifiers.push_back("g-" + gv);
        }
        for (auto& wv: weight_variables) {
            identifiers.push_back("w-" + wv);
        }

        id = ctrl->join(name, [this](Controller::UpdateData data) {
            absl::MutexLock l(&conf_mux);
            confs.emplace_back(data.conf_id, data.rank, data.size,
                std::make_shared<ControllerKVSAdapter>(ctrl, data.conf_id), identifiers);
            first_conf_pushed = true;
        });
        absl::ReaderMutexLock l(&conf_mux);
        conf_mux.Await(absl::Condition(&first_conf_pushed));
        confs.front().wait_ready();
        ready_conf = confs.begin();
    }
    void leave() {
        ctrl->leave(id);
    }

    void add_global_variable(const std::string &identifier) {
        global_variables.push_back(identifier);
    }
    void add_weight_variable(const std::string &identifier) {
        weight_variables.push_back(identifier);
    }

    std::tuple<bool, int64_t> begin_batch() {
        absl::MutexLock l(&conf_mux);
        for (auto next = ready_conf; next != confs.end(); ++next) {
            if (next->ready()) {
                ready_conf = next;
            } else {
                break;
            }
        }
        int64_t batch_conf_id = ctrl->begin_batch(id, ready_conf->id).get();
        while (current_conf->id != batch_conf_id) {
            current_conf++;
        }
        // todo: gc
        return std::make_tuple(true, 0);
    }
};
