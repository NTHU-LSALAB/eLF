#include <chrono>
#include <cstdint>
#include <future>

#include <absl/synchronization/mutex.h>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "controller.h"

constexpr std::chrono::milliseconds wait_time(300);

class CallbackMock {
    using ReturnType = Controller::UpdateData;
    int64_t n_calls = 0;
    int64_t desired_n = 0;
    ReturnType value;
    absl::Mutex mux;

    bool value_is_ready() {
        return n_calls == desired_n;
    }

public:
    ReturnType get(int64_t n = -1) {
        absl::MutexLock l(&mux);
        if (n == -1) {
            n = n_calls + 1;
        }
        assert(desired_n == 0);
        desired_n = n;
        mux.AwaitWithTimeout(
            absl::Condition(this, &CallbackMock::value_is_ready), absl::FromChrono(wait_time));
        desired_n = 0;
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

TEST_CASE("controller") {
    auto c = create_controller();

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

        // first worker leaves
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