#include "messages.grpc.pb.h"
#include "messages.pb.h"
#include <grpcpp/grpcpp.h>

#include <grpcpp/impl/codegen/client_context.h>
#include <grpcpp/impl/codegen/status.h>
#include <iostream>
#include <string>

#include <unistd.h>

#include <absl/strings/str_format.h>

namespace {
std::string getHostnameAsString() {
    char buf[HOST_NAME_MAX + 1];
    int len = gethostname(buf, HOST_NAME_MAX + 1);
    if (len != 0) {
        throw errno;
    }
    return std::string(buf);
}
} // namespace

class Worker {
    std::unique_ptr<Controller::Stub> stub;
    std::string id;

    std::string defaultId() const {
        return absl::StrFormat("%s-%d-%p", getHostnameAsString(), getpid(), this);
    }

public:
    Worker(const std::string &address)
        : stub(Controller::NewStub(
              grpc::CreateChannel(address, grpc::InsecureChannelCredentials()))) {}
    Worker(const Worker &) = delete;
    void join() {
        grpc::ClientContext context;
        JoinRequest request;
        JoinReply reply;
        request.set_name(defaultId());
        auto status = stub->Join(&context, request, &reply);
        if (!status.ok()) {
            throw std::runtime_error(status.error_message());
        }
        id = reply.name();
        std::cerr << id << "\n";
    }
    const std::string &getId() const {
        return id;
    }
};

int main() {
    Worker w("localhost:2222");
    w.join();
}
