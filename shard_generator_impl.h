#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include <pcg_random.hpp>

class ShardGenerator {
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    std::uniform_int_distribution<size_t> dist;
    pcg32 rng;
    int64_t counter;
    std::vector<int64_t> buffer;

public:
    ShardGenerator(size_t buffer_size)
        : dist(0, buffer_size - 1), rng(seed_source), counter(buffer_size), buffer(buffer_size) {
        std::iota(buffer.begin(), buffer.end(), 0);
    }
    int64_t next() {
        size_t i = dist(rng);
        int64_t result = buffer[i];
        buffer[i] = counter++;
        return result;
    }
};
