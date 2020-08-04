#include <pybind11/pybind11.h>

#include "controller.h"

namespace py = pybind11;

PYBIND11_MODULE(_elf, m) {
    m.def("create_controller", &elf::create_controller);
    m.def("export_controller", &elf::export_controller);
    py::class_<elf::Controller>(m, "Controller");
    py::class_<elf::ExportedController>(m, "ExportedController");
}
