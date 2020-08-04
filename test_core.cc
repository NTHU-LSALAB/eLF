#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <iostream>
#include <queue>
#include <stdexcept>

#include <absl/strings/str_format.h>
#include <absl/synchronization/mutex.h>

#include <catch2/catch.hpp>
#include <thread>

#include "controller.h"

constexpr std::chrono::milliseconds wait_time(300);

TEST_CASE("library usage") {
    auto a = std::make_unique<std::promise<int>>();
    a->set_value(10);
    auto fut = a->get_future();

    SECTION("get_future called multiple times") {
        CHECK_THROWS_AS(a->get_future(), std::future_error);
    }

    SECTION("future retrived multiple times") {
        REQUIRE(fut.get() == 10);
        CHECK_THROWS_AS(fut.get(), std::future_error);
    }

    SECTION("associated promise deleted") {
        a.reset();
        REQUIRE(fut.get() == 10); // yes this works!
    }
}

class CallbackMock {
    using ReturnType = Controller::UpdateData;
    int64_t n_calls = 0;
    int64_t last_get = 0;
    ReturnType value;
    absl::Mutex mux;

    struct ValueIsReady {
        int64_t desired_n;
        CallbackMock &mock;
        bool operator()() const {
            return desired_n <= mock.n_calls;
        }
    };

public:
    ReturnType get(int64_t n = -1) {
        absl::MutexLock l(&mux);
        if (n == -1) {
            n = last_get + 1;
        }
        ValueIsReady vr{n, *this};
        bool status = mux.AwaitWithTimeout(absl::Condition(&vr), absl::FromChrono(wait_time));
        if (!status) {
            return ReturnType{-111, -222, -333};
        }
        last_get = n_calls;
        return value;
    }
    std::function<void(ReturnType)> callback() {
        return [this](ReturnType callback_data) {
            absl::MutexLock l(&mux);
            n_calls++;
            value = callback_data;
        };
    };
};

struct TestConcreteController {
    TestConcreteController() : c(create_controller()) {}
    Controller *controller() {
        return c.get();
    }
    std::unique_ptr<Controller> c;
};
struct TestRemoteController {
    TestRemoteController()
        : c(create_controller()), ec(export_controller(c.get(), "localhost:")),
          rc(connect_controller(absl::StrFormat("localhost:%d", ec->listening_port()))) {}
    ~TestRemoteController() {
        ec->stop();
    }
    Controller *controller() {
        return rc.get();
    }
    std::unique_ptr<Controller> c;
    std::unique_ptr<ExportedController> ec;
    std::unique_ptr<Controller> rc;
};

TEMPLATE_TEST_CASE("controller",
    "[controller][core][rpc]",
    TestConcreteController,
    TestRemoteController) {

    TestType helper;
    Controller *c = helper.controller();

    SECTION("first joined worker has id 1") {
        REQUIRE(1 == c->join("", [&](Controller::UpdateData) {}));
    }

    SECTION("one worker") {
        CallbackMock mock;
        int64_t wid = c->join("", mock.callback());
        REQUIRE(1 == wid);
        auto update = mock.get(1);
        REQUIRE(update.conf_id == 1);
        REQUIRE(update.rank == 0);
        REQUIRE(update.size == 1);

        // begin a batch alone
        for (int i = 0; i < 3; i++) {
            auto fut = c->begin_batch(1, 1);
            REQUIRE(fut.wait_for(wait_time) == std::future_status::ready);
            REQUIRE(fut.get() == 1);
        }
    }

    SECTION("two workers") {
        CallbackMock mockA, mockB;
        int64_t a = c->join("workerA", mockA.callback());

        auto confA = mockA.get(1);
        REQUIRE(confA.conf_id == 1);
        REQUIRE(confA.rank == 0);
        REQUIRE(confA.size == 1);

        int64_t b = c->join("workerB", mockB.callback());
        REQUIRE(a != b); // workers must have different ids
        REQUIRE(a > 0);
        REQUIRE(b > 0);

        confA = mockA.get(2);
        auto confB = mockB.get(1);
        REQUIRE(confA.conf_id == 2);
        REQUIRE(confB.conf_id == 2);
        REQUIRE(confA.rank != confB.rank);
        REQUIRE(confA.size == 2);
        REQUIRE(confB.size == 2);

        for (int i = 0; i < 3; i++) {
            // worker A is ready, but worker B is not
            // lets make sure that worker A can proceed with the old configuration
            auto futA = c->begin_batch(a, 1);
            REQUIRE(futA.wait_for(wait_time) == std::future_status::ready);
            int64_t conf_id = futA.get();
            REQUIRE(conf_id == 1);
        }

        for (int i = 0; i < 3; i++) {
            // now that A is ready for the new configuration, but B is still not
            // the old configuration should still be used
            auto futA = c->begin_batch(a, 2);
            REQUIRE(futA.wait_for(wait_time) == std::future_status::ready);
            auto conf_id = futA.get();
            REQUIRE(conf_id == 1);
        }

        for (int i = 0; i < 3; i++) {
            // both of them are ready now
            // the new configuration should be used
            auto futB = c->begin_batch(b, 2);
            auto futA = c->begin_batch(a, 2);
            REQUIRE(futA.wait_for(wait_time) == std::future_status::ready);
            REQUIRE(2 == futA.get());
            REQUIRE(futB.wait_for(wait_time) == std::future_status::ready);
            REQUIRE(2 == futB.get());
        }

        // first worker requests leaving
        c->leave(a);
        confB = mockB.get();
        REQUIRE(confB.conf_id == 3);
        REQUIRE(confB.rank == 0);
        REQUIRE(confB.size == 1);

        for (int i = 0; i < 3; i++) {
            // new configuration is not ready
            // worker A continues training
            auto futB = c->begin_batch(b, 2);
            auto futA = c->begin_batch(a, 2);
            REQUIRE(futA.wait_for(wait_time) == std::future_status::ready);
            REQUIRE(2 == futA.get());
            REQUIRE(futB.wait_for(wait_time) == std::future_status::ready);
            REQUIRE(2 == futB.get());
        }

        SECTION("leaving worker reaches batch first") {
            auto futA = c->begin_batch(a, 2);
            auto futB = c->begin_batch(b, 3);
            REQUIRE(futB.wait_for(wait_time) == std::future_status::ready);
            REQUIRE(3 == futB.get());
            REQUIRE(futA.wait_for(wait_time) == std::future_status::ready);
            REQUIRE(0 == futA.get());
        }

        SECTION("existing workers reaches batch first") {
            for (int i = 0; i < 3; i++) {
                // new configuration is ready, so use new configuration
                auto futB = c->begin_batch(b, 3);
                REQUIRE(futB.wait_for(wait_time) == std::future_status::ready);
                REQUIRE(3 == futB.get());
            }

            // allow A to leave
            auto futA = c->begin_batch(a, 2);
            REQUIRE(futA.wait_for(wait_time) == std::future_status::ready);
            REQUIRE(0 == futA.get());
        }
    }

    SECTION("2 workers together") {
        CallbackMock mockA, mockB;
        int a = c->join("a", mockA.callback());
        int b = c->join("b", mockB.callback());
        CAPTURE(a, b);
        REQUIRE(a > 0);
        REQUIRE(b > 0);
        REQUIRE(a != b);

        // retrieve the updates
        int64_t last_conf_id = 0;
        while (true) {
            auto data = mockA.get();
            CAPTURE(last_conf_id, data.conf_id, data.rank, data.size);
            REQUIRE(data.conf_id > last_conf_id);
            REQUIRE(data.rank < data.size);
            REQUIRE(data.size > 0);
            REQUIRE(data.size <= 2);
            last_conf_id = data.conf_id;
            if (data.size == 2) {
                break;
            }
        }

        last_conf_id = 0;
        while (true) {
            auto data = mockB.get();
            CAPTURE(last_conf_id, data.conf_id, data.rank, data.size);
            REQUIRE(data.conf_id > last_conf_id);
            REQUIRE(data.rank < data.size);
            REQUIRE(data.size > 0);
            REQUIRE(data.size <= 2);
            last_conf_id = data.conf_id;
            if (data.size == 2) {
                break;
            }
        }
    }

    SECTION("a lot of workers") {
        const int64_t test_size = 64;
        std::vector<int64_t> workers;
        std::vector<CallbackMock> mocks(test_size);
        for (int i = 0; i < test_size; i++) {
            std::string worker_name = absl::StrFormat("worker-%d", i);
            UNSCOPED_INFO("joining " << worker_name);
            int64_t id = c->join(worker_name, mocks[i].callback());
            CAPTURE(id);
            workers.push_back(id);
        }
        int64_t last_conf_id;
        for (int64_t i = 0; i < test_size; i++) {
            last_conf_id = 0;
            while (true) {
                auto data = mocks[i].get();
                CAPTURE(i, last_conf_id, data.conf_id, data.rank, data.size);
                REQUIRE(data.conf_id > last_conf_id);
                REQUIRE(data.rank < data.size);
                REQUIRE(data.size > 0);
                REQUIRE(data.size <= test_size);
                last_conf_id = data.conf_id;
                if (data.size == test_size) {
                    break;
                }
            }
        }

        std::vector<std::future<void>> futures;
        std::vector<std::atomic_int64_t> max(test_size);
        for (auto &m : max) {
            m.store(-444);
        }
        for (int64_t i = 0; i < test_size; i++) {
            futures.emplace_back(std::async(std::launch::async, [&, i]() {
                while (true) {
                    int64_t conf_id = c->begin_batch(workers[i], last_conf_id).get();
                    max[i].store(conf_id);
                    if (last_conf_id == conf_id) {
                        break;
                    };
                    std::this_thread::sleep_for(wait_time);
                }
            }));
        }
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        for (int64_t i = 0; i < test_size; i++) {
            INFO(
                absl::StrFormat("Checking worker-%d(%d) conf_id:%d", i, workers[i], max[i].load()));
            CHECK(futures[i].wait_until(deadline) == std::future_status::ready);
            futures[i].get();
        }
    }

    SECTION("scaling a lot of workers") {
        const int64_t step_size = 16;
        const int64_t n_steps = 4;
        std::vector<int64_t> workers;
        std::vector<CallbackMock> mocks(step_size * n_steps);
        for (int64_t step = 0; step < n_steps; step++) {
            const int64_t prev_step = step * step_size;
            const int64_t this_step = prev_step + step_size;
            INFO("scaling from " << prev_step << " workers to " << this_step << " workers");
            for (int i = prev_step; i < this_step; i++) {
                std::string worker_name = absl::StrFormat("worker-%d", i);
                UNSCOPED_INFO("joining " << worker_name);
                int64_t id = c->join(worker_name, mocks[i].callback());
                CAPTURE(id);
                workers.push_back(id);
            }
            int64_t last_conf_id;
            for (int64_t i = 0; i < this_step; i++) {
                last_conf_id = 0;
                while (true) {
                    auto data = mocks[i].get();
                    CAPTURE(i, last_conf_id, data.conf_id, data.rank, data.size);
                    REQUIRE(data.conf_id > last_conf_id);
                    REQUIRE(data.rank < data.size);
                    REQUIRE(data.size > 0);
                    REQUIRE(data.size <= this_step);
                    last_conf_id = data.conf_id;
                    if (data.size == this_step) {
                        break;
                    }
                }
            }

            std::vector<std::future<void>> futures;
            std::vector<std::atomic_int64_t> max(this_step);
            for (auto &m : max) {
                m.store(-444);
            }
            for (int64_t i = 0; i < this_step; i++) {
                futures.emplace_back(std::async(std::launch::async, [&, i]() {
                    while (true) {
                        int64_t conf_id = c->begin_batch(workers[i], last_conf_id).get();
                        max[i].store(conf_id);
                        if (last_conf_id == conf_id) {
                            break;
                        };
                        std::this_thread::sleep_for(wait_time);
                    }
                }));
            }
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            for (int64_t i = 0; i < this_step; i++) {
                INFO(absl::StrFormat("Checking worker-%d(%d) receives conf_id %d/%d", i, workers[i],
                    max[i].load(), last_conf_id));
                CHECK(futures[i].wait_until(deadline) == std::future_status::ready);
                futures[i].get();
            }
        }
    }
}

TEMPLATE_TEST_CASE("controller-kv",
    "[controller][kv][rpc]",
    TestConcreteController,
    TestRemoteController) {

    TestType helper;
    Controller *c = helper.controller();

    SECTION("get after set") {
        c->kv_set(1, "key", "value");
        auto fut = c->kv_get(1, "key");
        REQUIRE(fut.wait_for(wait_time) == std::future_status::ready);
        REQUIRE(fut.get() == "value");
    }

    SECTION("set after get") {
        auto fut = c->kv_get(1, "key");
        REQUIRE(fut.wait_for(wait_time) == std::future_status::timeout);
        c->kv_set(1, "key", "value");
        REQUIRE(fut.wait_for(wait_time) == std::future_status::ready);
        REQUIRE(fut.get() == "value");
    }
}
