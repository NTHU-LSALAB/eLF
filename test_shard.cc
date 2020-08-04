#include <set>

#include <catch2/catch.hpp>

#include "shard_generator_impl.h"

TEST_CASE("not random") {
    ShardGenerator sg(1);

    CHECK(sg.next() == 0);
    CHECK(sg.next() == 1);
    CHECK(sg.next() == 2);
    CHECK(sg.next() == 3);
    CHECK(sg.next() == 4);
}

TEST_CASE("random") {
    ShardGenerator sg(10);

    std::set<int64_t> seen;
    for (int i = 0; i < 1000; i++) {
        int64_t x = sg.next();
        CAPTURE(i, x, seen);
        CHECK(seen.find(x) == seen.end());
        CHECK(x < i + 10);
        seen.insert(x);
    }
}
