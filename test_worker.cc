#include <array>

#include <absl/synchronization/notification.h>
#include <catch2/catch.hpp>

#include "controller.h"
#include "worker_impl.h"

TEST_CASE("worker") {
    auto controller = elf::create_controller();
    elf::Worker w(controller.get());

    auto allreduce_op = w.add_global_variable("var1");
    auto broadcast_op = w.add_weight_variable("var1");
    w.commit_and_join();

    bool should_continue;
    bool needs_broadcast;
    int64_t shard_number;
    std::tie(should_continue, needs_broadcast, shard_number) = w.begin_batch();
    CHECK(should_continue);
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

    H = {10, 8, 36, 254};
    Dsrc = H;
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

TEST_CASE("2 workers") {
    absl::Mutex mu;
    auto controller = elf::create_controller();
    absl::Notification worker1_joined;
    absl::Notification test_done;

    std::thread t1([&]() { // worker 1
        elf::Worker w(controller.get());
        auto broadcast_op = w.add_global_variable("var1");
        auto allreduce_op = w.add_weight_variable("var1");
        w.commit_and_join();
        bool should_continue, requires_broadcast;
        int64_t shard_number;
        std::tie(should_continue, requires_broadcast, shard_number) = w.begin_batch();
        CHECK(should_continue);
        CHECK_FALSE(requires_broadcast);
        worker1_joined.Notify();

        while (true) {
            std::tie(should_continue, requires_broadcast, shard_number) = w.begin_batch();
            if (!should_continue) {
                break;
            }
            if (test_done.HasBeenNotified()) {
                w.leave();
            }

            { // broadcast
                auto H = std::array<float, 4>{26, 83, 62, 30};
                gpu_array<float, 4> Dsrc, Ddst;
                Dsrc = H;
                absl::Notification done;
                broadcast_op->execute_async(Dsrc.data(), Ddst.data(), 4, elf::Communicator::f32,
                    [&done]() { done.Notify(); });
                done.WaitForNotification();
                {
                    absl::MutexLock l(&mu);
                    CHECK(Dsrc.cpu() == H);
                }
            }

            { // allreduce
                auto H = std::array<float, 4>{61, 48, 81, 76};
                gpu_array<float, 4> Dsrc, Ddst;
                Dsrc = H;
                absl::Notification done;
                allreduce_op->execute_async(Dsrc.data(), Ddst.data(), 4, elf::Communicator::f32,
                    [&done]() { done.Notify(); });
                done.WaitForNotification();
                {
                    absl::MutexLock l(&mu);
                    CHECK(Dsrc.cpu() == H);
                }
            }
        }
    });

    { // worker 2
        worker1_joined.WaitForNotification();

        elf::Worker w(controller.get());
        auto broadcast_op = w.add_global_variable("var1");
        auto allreduce_op = w.add_weight_variable("var1");
        w.commit_and_join();

        bool should_continue, requires_broadcast;
        int64_t shard_number;
        std::tie(should_continue, requires_broadcast, shard_number) = w.begin_batch();
        REQUIRE(should_continue);
        REQUIRE(requires_broadcast);

        { // broadcast
            auto H = std::array<float, 4>{9700, 800, 4900, 7202};
            gpu_array<float, 4> Dsrc, Ddst;
            Dsrc = H;
            absl::Notification done;
            broadcast_op->execute_async(
                Dsrc.data(), Ddst.data(), 4, elf::Communicator::f32, [&done]() { done.Notify(); });
            done.WaitForNotification();
            {
                absl::MutexLock l(&mu);
                CHECK(Dsrc.cpu() == H);
                CHECK(Ddst.cpu() == std::array<float, 4>{26, 83, 62, 30});
            }
        }

        { // allreduce
            auto H = std::array<float, 4>{100, 200, 300, 400};
            gpu_array<float, 4> Dsrc, Ddst;
            Dsrc = H;
            absl::Notification done;
            allreduce_op->execute_async(
                Dsrc.data(), Ddst.data(), 4, elf::Communicator::f32, [&done]() { done.Notify(); });
            done.WaitForNotification();
            {
                absl::MutexLock l(&mu);
                CHECK(Dsrc.cpu() == H);
                CHECK(Ddst.cpu() == std::array<float, 4>{161, 248, 381, 476});
            }
        }
    }
    test_done.Notify();
    t1.join();
}
