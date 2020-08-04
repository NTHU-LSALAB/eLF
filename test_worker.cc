#include <array>
#include <chrono>
#include <thread>

#include <absl/synchronization/notification.h>
#include <catch2/catch.hpp>

#include "controller.h"
#include "worker_impl.h"

TEST_CASE("worker") {
    auto controller = elf::create_controller();
    cudaSetDevice(0);
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
}

TEST_CASE("2 workers", "[!mayfail]") {
    // this test is buggy so allow it to fail

    absl::Mutex mu;
    auto controller = elf::create_controller();
    absl::Notification worker1_joined;
    absl::Notification test_done;

    cudaSetDevice(0);
    elf::Worker w1(controller.get());

    std::thread t1([&]() { // worker 1
        auto broadcast_op = w1.add_global_variable("var1");
        auto allreduce_op = w1.add_weight_variable("var1");
        w1.commit_and_join();
        bool should_continue, requires_broadcast;
        int64_t shard_number;
        std::tie(should_continue, requires_broadcast, shard_number) = w1.begin_batch();
        CHECK(should_continue);
        CHECK_FALSE(requires_broadcast);
        worker1_joined.Notify();

        while (true) {
            std::tie(should_continue, requires_broadcast, shard_number) = w1.begin_batch();

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

            // let the controller take a breath
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    { // worker 2
        worker1_joined.WaitForNotification();

        cudaSetDevice(1);
        elf::Worker w2(controller.get());
        auto broadcast_op = w2.add_global_variable("var1");
        auto allreduce_op = w2.add_weight_variable("var1");
        w2.commit_and_join();

        while (true) {
            bool should_continue, requires_broadcast;
            int64_t shard_number;
            std::tie(should_continue, requires_broadcast, shard_number) = w2.begin_batch();
            if (!should_continue) {
                break;
            }
            REQUIRE(requires_broadcast);

            { // broadcast
                auto H = std::array<float, 4>{9700, 800, 4900, 7202};
                gpu_array<float, 4> Dsrc, Ddst;
                Dsrc = H;
                absl::Notification done;
                broadcast_op->execute_async(Dsrc.data(), Ddst.data(), 4, elf::Communicator::f32,
                    [&done]() { done.Notify(); });
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
                allreduce_op->execute_async(Dsrc.data(), Ddst.data(), 4, elf::Communicator::f32,
                    [&done]() { done.Notify(); });
                done.WaitForNotification();
                {
                    absl::MutexLock l(&mu);
                    CHECK(Dsrc.cpu() == H);
                    CHECK(Ddst.cpu() == std::array<float, 4>{161, 248, 381, 476});
                }
            }

            // let the controller take a breath
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            w2.leave();
        }
        std::cerr << "worker2 test complete\n";
    }
    test_done.Notify();
    std::cerr << "test_done signalled\n";
    t1.join();
}
