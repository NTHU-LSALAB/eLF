#pragma once

#include <future>
#include <string>

namespace elf {

class KeyValueStore {
protected:
    KeyValueStore() {}

public:
    KeyValueStore(const KeyValueStore &) = delete;
    virtual ~KeyValueStore() {}
    virtual void set(const std::string &k, const std::string &v) = 0;
    virtual std::shared_future<std::string> get(const std::string &k) = 0;
};

} // namespace elf
