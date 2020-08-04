project('pybind11', 'cpp', default_options : ['cpp_std=c++11'], license : 'BSD-style', version : '2.3.0')

pymod = import('python')
py3_dep = pymod.find_installation('python3').dependency()

# https://github.com/mesonbuild/meson/issues/4117
if host_machine.system() == 'windows'
	python_ext_dep = py3_dep
else
	python_ext_dep = py3_dep.partial_dependency(compile_args: true)
endif

pybind11_incdir = include_directories('include')

pybind11_dep = declare_dependency(
	include_directories : pybind11_incdir,
	dependencies : python_ext_dep
)
