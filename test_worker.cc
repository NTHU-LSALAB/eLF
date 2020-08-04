#include <catch2/catch.hpp>

#include "worker_impl.h"
#include "controller.h"

TEST_CASE("smoke") {
    auto controller = create_controller();
    Worker w(controller.get());
}
