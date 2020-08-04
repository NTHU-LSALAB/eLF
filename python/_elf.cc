#include <pybind11/pybind11.h>

#include "controller.h"
#include "operator.h"
#include "worker_impl.h"

namespace py = pybind11;

PYBIND11_MODULE(_elf, m) {
    m.def("create_controller", &elf::create_controller);
    m.def("connect_controller", &elf::connect_controller);
    m.def("export_controller", &elf::export_controller);

    py::class_<elf::Controller>(m, "Controller")
        .def("leave", &elf::Controller::leave)
        .def("stop", &elf::Controller::stop);

    py::class_<elf::ExportedController>(m, "ExportedController")
        .def("listening_port", &elf::ExportedController::listening_port)
        .def("stop", &elf::ExportedController::stop);

    py::class_<elf::Worker>(m, "Worker")
        .def(py::init<elf::Controller *>())
        .def("commit_and_join", &elf::Worker::commit_and_join, py::arg("name") = "")
        .def("leave", &elf::Worker::leave)
        .def("add_global_variable", &elf::Worker::add_global_variable)
        .def("add_weight_variable", &elf::Worker::add_weight_variable)
        .def("begin_batch", &elf::Worker::begin_batch);

    py::class_<elf::Operator>(m, "Operator");
}
