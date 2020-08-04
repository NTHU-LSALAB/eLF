#include <catch2/catch.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "controller.h"
#include "worker_impl.h"

TEST_CASE("single worker") {
    auto controller = elf::create_controller();
    elf::Worker w(controller.get());

    w.join();
    bool needs_broadcast;
    int64_t shard_number;
    std::tie(needs_broadcast, shard_number) = w.begin_batch();
    REQUIRE_FALSE(needs_broadcast);
}
