#include <array>

#include <absl/synchronization/notification.h>
#include <catch2/catch.hpp>

#include "controller.h"
#include "worker_impl.h"

TEST_CASE("single worker") {
    auto controller = elf::create_controller();
    elf::Worker w(controller.get());

    auto allreduce_op = w.add_global_variable("var1");
    auto broadcast_op = w.add_weight_variable("var1");
    w.commit_and_join();

    bool needs_broadcast;
    int64_t shard_number;
    std::tie(needs_broadcast, shard_number) = w.begin_batch();
    REQUIRE_FALSE(needs_broadcast);

    auto H = std::array<float, 4>{140, 114, 91, 178};
    gpu_array<float, 4> Dsrc, Ddst;
    Dsrc = H;

    SECTION("allreduce") {
        absl::Notification done;
        allreduce_op->execute_async(
            Dsrc.data(), Ddst.data(), 4, elf::Communicator::f32, [&done]() { done.Notify(); });
        done.WaitForNotification();
        CHECK(Dsrc.cpu() == H);
        CHECK(Ddst.cpu() == H);
    }

    SECTION("broadcast") {
        absl::Notification done;
        broadcast_op->execute_async(
            Dsrc.data(), Ddst.data(), 4, elf::Communicator::f32, [&done]() { done.Notify(); });
        done.WaitForNotification();
        CHECK(Dsrc.cpu() == H);
        CHECK(Ddst.cpu() == H);
    }

    SECTION("make sure the second worker doesn't break it") {
        cudaSetDevice(1);
        elf::Worker w2(controller.get());
        auto allreduce_op2 = w2.add_global_variable("var1");
        auto broadcast_op2 = w2.add_weight_variable("var1");
        w2.commit_and_join();
    }
}
