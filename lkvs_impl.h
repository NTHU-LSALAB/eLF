#pragma once

#include <unordered_map>

#include <absl/synchronization/mutex.h>

#include "kvs.h"

namespace elf {

class LocalKeyValueStore : public KeyValueStore {

    struct Item {
        std::promise<std::string> promise;
        std::shared_future<std::string> future;
        Item() : future(promise.get_future()) {}
    };
    absl::Mutex mux;
    std::unordered_map<std::string, Item> map;

public:
    LocalKeyValueStore() {}
    ~LocalKeyValueStore() override {}

    void set(const std::string &k, const std::string &v) override {
        absl::MutexLock l(&mux);
        map[k].promise.set_value(v);
    }
    std::shared_future<std::string> get(const std::string &k) override {
        absl::MutexLock l(&mux);
        return map[k].future;
    }
};

} // namespace elf
