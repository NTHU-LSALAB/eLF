#include <grpcpp/grpcpp.h>
#include <grpcpp/impl/codegen/completion_queue.h>
#include <grpcpp/impl/codegen/server_context.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server_builder.h>

#include "controller.h"
#include "messages.grpc.pb.h"
#include "messages.pb.h"

using grpc::ServerContext;
using grpc::Status;



int main() {
    grpc::ServerBuilder builder;
    builder.AddListeningPort("localhost:1111", grpc::InsecureServerCredentials());
    Coordinator::AsyncService service;
    builder.RegisterService(&service);
    auto cq = builder.AddCompletionQueue();
    auto server = builder.BuildAndStart();
}
