#include <array>

#include <absl/synchronization/mutex.h>
#include <absl/synchronization/notification.h>
#include <catch2/catch.hpp>

#include "communicator.h"
#include "cuda_helper.h"
#include "lkvs_impl.h"
#include "worker_impl.h"

TEST_CASE("workerconf") {
    elf::WorkerConf wc(0, 0, 0, 1, std::make_shared<elf::LocalKeyValueStore>(), {"var1", "var2"});
    wc.wait_ready();
    REQUIRE(wc.ready());

    std::array<int32_t, 4> H{16, 13, 26, 21};
    gpu_array<int32_t, 4> Dsrc, Ddst;
    Dsrc = H;

    SECTION("broadcast") {
        absl::Notification done;
        wc.schedule_broadcast(
            "var1", Dsrc.data(), Ddst.data(), 4, elf::Communicator::i32, [&]() { done.Notify(); });
        done.WaitForNotification();
        CHECK(Dsrc.cpu() == H);
        CHECK(Ddst.cpu() == H);
    }

    SECTION("allreduce") {
        absl::Notification done;
        wc.schedule_allreduce(
            "var1", Dsrc.data(), Ddst.data(), 4, elf::Communicator::i32, [&]() { done.Notify(); });
        done.WaitForNotification();
        CHECK(Dsrc.cpu() == H);
        CHECK(Ddst.cpu() == H);
    }
}

TEST_CASE("workerconf 2 workers") {
    auto kvs = std::make_shared<elf::LocalKeyValueStore>();
    absl::Mutex mu;

    bool test_broadcast = false;
    bool test_allreduce = false;

    SECTION("broadcast") {
        test_broadcast = true;
    }
    SECTION("allreduce") {
        test_allreduce = true;
    }

    std::thread t1([kvs, test_allreduce, test_broadcast, &mu]() {
        elf::WorkerConf wc(0, 0, 0, 2, kvs, {"var1", "var2"});
        wc.wait_ready();
        REQUIRE(wc.ready());

        std::array<int32_t, 4> H{11, 12, 13, 14};
        gpu_array<int32_t, 4> Dsrc, Ddst;
        Dsrc = H;

        if (test_broadcast) {
            absl::Notification done;
            wc.schedule_broadcast(
                "var1", Dsrc.data(), Ddst.data(), 4, elf::Communicator::i32, [&]() {
                    std::cerr << "thread 1 broadcast done\n";
                    done.Notify();
                });
            done.WaitForNotification();
            {
                absl::MutexLock l(&mu);
                INFO("thread 1 broadcast");
                CHECK(Dsrc.cpu() == H);
                CHECK(Ddst.cpu() == std::array<int32_t, 4>{11, 12, 13, 14});
            }
        }

        if (test_allreduce) {
            absl::Notification done;
            wc.schedule_allreduce(
                "var2", Dsrc.data(), Ddst.data(), 4, elf::Communicator::i32, [&]() {
                    std::cerr << "thread 1 allreduce done\n";
                    done.Notify();
                });
            done.WaitForNotification();
            {
                absl::MutexLock l(&mu);
                INFO("thread 1 allreduce");
                CHECK(Dsrc.cpu() == H);
                CHECK(Ddst.cpu() == std::array<int32_t, 4>{32, 34, 36, 38});
            }
        }
    });

    std::thread t2([kvs, test_allreduce, test_broadcast, &mu]() {
        elf::WorkerConf wc(1, 0, 1, 2, kvs, {"var1", "var2"});
        wc.wait_ready();
        REQUIRE(wc.ready());

        std::array<int32_t, 4> H{21, 22, 23, 24};
        gpu_array<int32_t, 4> Dsrc, Ddst;
        Dsrc = H;

        if (test_broadcast) {
            absl::Notification done;
            wc.schedule_broadcast(
                "var1", Dsrc.data(), Ddst.data(), 4, elf::Communicator::i32, [&]() {
                    std::cerr << "thread 2 broadcast done\n";
                    done.Notify();
                });
            done.WaitForNotification();
            {
                absl::MutexLock l(&mu);
                INFO("thread 2 broadcast");
                CHECK(Dsrc.cpu() == H);
                CHECK(Ddst.cpu() == std::array<int32_t, 4>{11, 12, 13, 14});
            }
        }

        if (test_allreduce) {
            absl::Notification done;
            wc.schedule_allreduce(
                "var2", Dsrc.data(), Ddst.data(), 4, elf::Communicator::i32, [&]() {
                    std::cerr << "thread 2 allreduce done\n";
                    done.Notify();
                });
            done.WaitForNotification();
            {
                absl::MutexLock l(&mu);
                INFO("thread 2 allreduce");
                CHECK(Dsrc.cpu() == H);
                CHECK(Ddst.cpu() == std::array<int32_t, 4>{32, 34, 36, 38});
            }
        }
    });

    t1.join();
    t2.join();
}
