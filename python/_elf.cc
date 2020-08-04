#include <pybind11/pybind11.h>

#include "controller.h"

namespace py = pybind11;

PYBIND11_MODULE(_elf, m) {
    m.def("create_controller", &create_controller);
    m.def("export_controller", &export_controller);
    py::class_<Controller>(m, "Controller");
    py::class_<ExportedController>(m, "ExportedController");
}
