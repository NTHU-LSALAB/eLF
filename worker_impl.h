#pragma once

#include <cstdint>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <absl/synchronization/mutex.h>
#include <absl/synchronization/notification.h>

#include "communicator.h"
#include "controller.h"
#include "cuda_helper.h"
#include "kvs.h"
#include "lkvs_impl.h"
#include "nccl_communicator.h"
#include "operator.h"
#include "thread_pool_impl.h"

namespace elf {

class WorkerConf {
public:
    const int gpu;
    const int64_t id;
    const int64_t rank;
    const int64_t size;

private:
    ThreadPool pool;
    absl::Mutex comms_mux;
    std::unordered_map<std::string, std::unique_ptr<Communicator>> communicators;
    std::shared_ptr<KeyValueStore> kvs;
    int64_t todo;
    absl::Notification ready_;

public:
    WorkerConf(int gpu,
        int64_t id,
        int64_t rank,
        int64_t size,
        std::shared_ptr<KeyValueStore> kvs,
        const std::vector<std::string> &identifiers)
        : gpu(gpu), id(id), rank(rank), size(size), pool(8), kvs(kvs), todo(identifiers.size()) {
        for (auto &identifier : identifiers) {
            pool.Schedule([this, identifier]() { create_communicator_for_variable(identifier); });
        }
        if (!todo) {
            ready_.Notify();
        }
    }

    bool ready() {
        return ready_.HasBeenNotified();
    }

    void wait_ready() {
        ready_.WaitForNotification();
    }

    bool schedule_allreduce(const std::string &identifier,
        void *in,
        void *out,
        size_t count,
        Communicator::DataType type,
        std::function<void()> done_callback) {
        pool.Schedule([=]() {
            CUDA_ASSERT(cudaSetDevice(gpu));
            assert(communicators.at(identifier));
            communicators.at(identifier)->allreduce(in, out, count, type);
            done_callback();
        });
        return true;
    }

    bool schedule_broadcast(const std::string &identifier,
        void *in,
        void *out,
        size_t count,
        Communicator::DataType type,
        std::function<void()> done_callback) {
        pool.Schedule([=]() {
            CUDA_ASSERT(cudaSetDevice(gpu));
            assert(communicators.at(identifier));
            communicators.at(identifier)->broadcast(in, out, 0, count, type);
            done_callback();
        });
        return true;
    }

private:
    void create_communicator_for_variable(std::string variable) {
        CUDA_ASSERT(cudaSetDevice(gpu));
        auto comm = create_nccl_communicator(kvs.get(), variable, rank, size);
        {
            absl::MutexLock l(&comms_mux);
            communicators[variable] = std::move(comm);
            --todo;
            if (todo == 0) {
                ready_.Notify();
            }
        }
    }

    void teardown_communicators() {

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
    int gpu;
    std::vector<std::string> global_variables;
    std::vector<std::string> weight_variables;

    std::list<WorkerConf> confs;
    absl::Mutex conf_mux;
    bool first_conf_pushed = false;

    std::vector<std::string> identifiers;

    decltype(confs)::iterator current_conf;
    decltype(confs)::iterator ready_conf;

public:
    Worker(Controller *ctrl) : ctrl(ctrl) {
        CUDA_ASSERT(cudaGetDevice(&gpu));
    }
    Worker(const Worker &) = delete;
    ~Worker() {}

    void join(const std::string &name = "") {
        for (auto &gv : global_variables) {
            identifiers.push_back(gv);
        }
        for (auto &wv : weight_variables) {
            identifiers.push_back(wv);
        }

        id = ctrl->join(name, [this](Controller::UpdateData data) {
            absl::MutexLock l(&conf_mux);
            confs.emplace_back(gpu, data.conf_id, data.rank, data.size,
                std::make_shared<ControllerKVSAdapter>(ctrl, data.conf_id), identifiers);
            first_conf_pushed = true;
        });
        {
            absl::ReaderMutexLock l(&conf_mux);
            conf_mux.Await(absl::Condition(&first_conf_pushed));
            confs.front().wait_ready();
            ready_conf = confs.begin();
            current_conf = confs.begin();
        }
    }
    void leave() {
        ctrl->leave(id);
    }

    class Allreduce : public Operator {
        Worker &worker;
        const std::string identifier;

    public:
        Allreduce(Worker &worker, const std::string &identifier)
            : worker(worker), identifier(identifier) {}
        ~Allreduce() {}
        bool execute_async(void *in,
            void *out,
            size_t count,
            Communicator::DataType type,
            std::function<void()> done_callback) {
            return worker.current_conf->schedule_allreduce(
                identifier, in, out, count, type, done_callback);
        }
    };

    class Broadcast : public Operator {
        Worker &worker;
        const std::string identifier;

    public:
        Broadcast(Worker &worker, const std::string &identifier)
            : worker(worker), identifier(identifier) {}
        ~Broadcast() {}
        bool execute_async(void *in,
            void *out,
            size_t count,
            Communicator::DataType type,
            std::function<void()> done_callback) {
            return worker.current_conf->schedule_broadcast(
                identifier, in, out, count, type, done_callback);
        }
    };

    std::unique_ptr<Operator> add_global_variable(std::string identifier) {
        identifier = "[g]" + identifier;
        global_variables.push_back(identifier);
        return std::make_unique<Allreduce>(*this, identifier);
    }
    std::unique_ptr<Operator> add_weight_variable(std::string identifier) {
        identifier = "[w]" + identifier;
        weight_variables.push_back(identifier);
        return std::make_unique<Broadcast>(*this, identifier);
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
        return std::make_tuple(false, 0);
    }
};

} // namespace elf
