#include <catch2/catch.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "worker_impl.h"
#include "controller.h"

TEST_CASE("smoke") {
    auto controller = elf::create_controller();
    elf::Worker w(controller.get());
}
