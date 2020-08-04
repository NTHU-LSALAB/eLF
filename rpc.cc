#include <bits/stdint-intn.h>
#include <chrono>
#include <future>
#include <grpcpp/create_channel.h>
#include <grpcpp/impl/codegen/client_context.h>
#include <grpcpp/impl/codegen/sync_stream.h>
#include <grpcpp/security/credentials.h>
#include <queue>
#include <unordered_map>

#include <absl/synchronization/mutex.h>

#include <grpcpp/grpcpp.h>
#include <grpcpp/impl/codegen/completion_queue.h>
#include <grpcpp/impl/codegen/server_context.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server_builder.h>
#include <stdexcept>

#include "absl/time/time.h"
#include "controller.h"
#include "messages.grpc.pb.h"
#include "messages.pb.h"

using grpc::ClientContext;
using grpc::ServerContext;
using grpc::Status;

class UpdateReceiver {
    bool done = false;
    std::queue<Controller::UpdateData> q;
    absl::Mutex mux;
    bool cond() {
        return done || !q.empty();
    }

public:
    enum Status {
        DONE,
        TIMEOUT,
        ITEM,
    };
    Controller::update_callback_t callback() {
        return [this](Controller::UpdateData data) {
            absl::MutexLock l(&mux);
            q.push(data);
        };
    }
    void close() {
        absl::MutexLock l(&mux);
        done = true;
    }
    Status get(Controller::UpdateData *data) {
        absl::MutexLock l(&mux);
        if (!mux.AwaitWithTimeout(
                absl::Condition(this, &UpdateReceiver::cond), absl::Milliseconds(100))) {
            return TIMEOUT;
        }
        if (done) {
            return DONE;
        }
        *data = q.front();
        q.pop();
        return ITEM;
    }
};

class ControllerServiceImpl final : public ControllerRPC::Service {
public:
    explicit ControllerServiceImpl(Controller *c) : c(c) {}
    Status
    Join(ServerContext *sctx, const JoinRequest *in, grpc::ServerWriter<Update> *outw) override {
        UpdateReceiver receiver;
        int64_t id = c->join(in->name(), receiver.callback());
        {
            absl::MutexLock l(&aw_mux);
            active_workers.insert(id);
        }
        while (alive() && !sctx->IsCancelled()) {
            {
                absl::MutexLock l(&aw_mux);
                if (active_workers.find(id) == active_workers.end()) {
                    break;
                }
            }
            Controller::UpdateData data;
            auto status = receiver.get(&data);
            if (status == UpdateReceiver::DONE) {
                return Status::OK;
            } else if (status == UpdateReceiver::TIMEOUT) {
                continue;
            }
            Update update;
            update.set_id(id);
            update.set_conf_id(data.conf_id);
            update.set_rank(data.rank);
            update.set_size(data.size);
            outw->Write(update);
        }
        return Status::CANCELLED;
    }

    Status Leave(ServerContext *, const LeaveRequest *in, LeaveResponse *out) override {
        c->leave(in->id());
        absl::MutexLock l(&aw_mux);
        active_workers.erase(in->id());
        return Status::OK;
    }

    Status
    BeginBatch(ServerContext *, const BeginBatchRequest *in, BeginBatchResponse *out) override {
        auto future = c->begin_batch(in->id(), in->ready_conf_id());
        out->set_conf_id(future.get());
        return Status::OK;
    }

    Status EndBatch(ServerContext *, const EndBatchRequest *in, EndBatchResponse *out) override {
        c->end_batch(in->id());
        return Status::OK;
    }

    Status KVSet(ServerContext *, const KVSetRequest *in, KVSetResponse *out) override {
        c->kv_set(in->conf_id(), in->key(), in->value());
        return Status::OK;
    }

    Status KVGet(ServerContext *sctx, const KVGetRequest *in, KVGetResponse *out) override {
        auto future = c->kv_get(in->conf_id(), in->key());
        while (alive() && !sctx->IsCancelled()) {
            if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
                out->set_value(future.get());
                return Status::OK;
            }
        }
        return Status::CANCELLED;
    }

    void terminate() {
        absl::MutexLock l(&_terminating_mux);
        _terminating = true;
    }

private:
    Controller *c;
    absl::Mutex aw_mux;
    std::unordered_set<int64_t> active_workers;
    absl::Mutex _terminating_mux;
    bool _terminating = false;
    bool alive() {
        absl::ReaderMutexLock l(&_terminating_mux);
        return !_terminating;
    }
};

class ExportedControllerImpl : public ExportedController {
public:
    ExportedControllerImpl(Controller *c, const std::string &address) : service(c) {
        grpc::ServerBuilder builder;
        builder.AddListeningPort(address, grpc::InsecureServerCredentials(), &selected_port);
        builder.RegisterService(&service);
        server = builder.BuildAndStart();
    }
    ~ExportedControllerImpl() override {}

    int listening_port() override {
        return selected_port;
    }
    void stop() override {
        service.terminate();
        server->Shutdown();
        server->Wait();
    }

private:
    ControllerServiceImpl service;
    std::unique_ptr<grpc::Server> server;
    int selected_port = 0;
};

class RemoteController : public Controller {
public:
    RemoteController(const std::string &address)
        : stub(ControllerRPC::NewStub(
              grpc::CreateChannel(address, grpc::InsecureChannelCredentials()))) {}
    ~RemoteController() override {
        for (auto &thread : threads) {
            thread.join();
        }
    }

    int64_t join(const std::string &name, update_callback_t callback) override {
        auto cctx = std::make_unique<ClientContext>();
        JoinRequest in;
        in.set_name(name);
        auto reader = stub->Join(cctx.get(), in);
        Update update;
        assert(reader->Read(&update));
        callback(Controller::UpdateData{update.conf_id(), update.rank(), update.size()});
        threads.emplace_back(
            update_loop, update.id(), std::move(cctx), std::move(reader), callback);
        return update.id();
    }

    void leave(int64_t id) override {
        ClientContext cctx;
        LeaveRequest in;
        LeaveResponse out;
        in.set_id(id);
        check(stub->Leave(&cctx, in, &out));
    }

    std::future<int64_t> begin_batch(int64_t id, int64_t ready_conf_id) override {
        return std::async(std::launch::async, [id, ready_conf_id, this]() -> int64_t {
            ClientContext cctx;
            BeginBatchRequest in;
            BeginBatchResponse out;
            in.set_id(id);
            in.set_ready_conf_id(ready_conf_id);
            check(stub->BeginBatch(&cctx, in, &out));
            return out.conf_id();
        });
    }

    void end_batch(int64_t id) override {
        ClientContext cctx;
        EndBatchRequest in;
        EndBatchResponse out;
        check(stub->EndBatch(&cctx, in, &out));
    }

    void kv_set(int64_t conf_id, const std::string &key, const std::string &value) override {
        ClientContext cctx;
        KVSetRequest in;
        KVSetResponse out;
        in.set_conf_id(conf_id);
        in.set_key(key);
        in.set_value(value);
        check(stub->KVSet(&cctx, in, &out));
    }

    std::shared_future<std::string> kv_get(int64_t conf_id, const std::string &key) override {
        return std::async(std::launch::async, [conf_id, key, this]() -> std::string {
            ClientContext cctx;
            KVGetRequest in;
            KVGetResponse out;
            in.set_conf_id(conf_id);
            in.set_key(key);
            check(stub->KVGet(&cctx, in, &out));
            return out.value();
        });
    }

    void stop() override {
        throw std::runtime_error("you cannot stop a remote controller");
    }

private:
    static void update_loop(int64_t id,
        std::unique_ptr<ClientContext>,
        std::unique_ptr<grpc::ClientReader<Update>> reader,
        update_callback_t callback) {
        Update update;
        while (reader->Read(&update)) {
            callback(Controller::UpdateData{update.conf_id(), update.rank(), update.size()});
        }
    }
    void check(const grpc::Status &status) {
        if (!status.ok()) {
            throw std::runtime_error(status.error_message());
        }
    }
    std::unique_ptr<ControllerRPC::Stub> stub;
    std::vector<std::thread> threads;
};

std::unique_ptr<Controller> connect_controller(const std::string &address) {
    return std::make_unique<RemoteController>(address);
}

std::unique_ptr<ExportedController> export_controller(Controller *c, const std::string &address) {
    return std::make_unique<ExportedControllerImpl>(c, address);
}
