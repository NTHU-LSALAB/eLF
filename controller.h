#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <string>

class Controller {
protected:
    Controller() {}

public:
    Controller(const Controller &) = delete;
    virtual ~Controller() {}

    // the callback data for worker pool configuration updates
    // conf_id: serial number of the configuration
    // rank: the rank of the worker in this configuration
    // size: the total number of workers in this configuration
    struct UpdateData {
        int64_t conf_id;
        int64_t rank;
        int64_t size;
    };
    using update_callback_t = std::function<void(UpdateData)>;

    // joined the worker named by the given name
    // name: the name of the worker
    // callback: the callback function for updates
    // returns the id of the worker; the id is used in other methods of the controller
    virtual int64_t join(const std::string &name, update_callback_t callback) = 0;

    // make worker with the supplied id gracefully leave the worker pool
    virtual void leave(int64_t id) = 0;

    // start a training batch
    // id: worker identifier
    // ready_conf_id: the configuration that the worker is ready for
    // returns the configuration id to use in this batch
    // -1 is returned if training should be stopped
    virtual std::future<int64_t> begin_batch(int64_t id, int64_t ready_conf_id) = 0;

    // indicate that the batch is finished
    // only used for profiling
    virtual void end_batch(int64_t id) = 0;

    // set the value associated with the key
    virtual void kv_set(int64_t conf_id, const std::string &key, const std::string &value) = 0;

    // retrieve the value associated with the key
    virtual std::shared_future<std::string> kv_get(int64_t conf_id, const std::string &key) = 0;

    // stop the controller
    virtual void stop() = 0;
};

class ExportedController {
protected:
    ExportedController() {}

public:
    ExportedController(const ExportedController &) = delete;
    virtual ~ExportedController() {}

    virtual int listening_port() = 0;
    virtual void stop() = 0;
};

std::unique_ptr<Controller> create_controller();

std::unique_ptr<Controller> connect_controller(const std::string &address);

std::unique_ptr<ExportedController> export_controller(Controller *c, const std::string &address);
