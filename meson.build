project('elfcore', 'cpp', default_options: ['cpp_std=c++14'])

clang = meson.get_compiler('cpp')
assert(
    clang.get_id() == 'clang',
    'clang must be used. Specify the environment variable CXX=clang++',
)

threads_dep = dependency('threads')
catch2_proj = subproject('catch2')
test_deps = [catch2_proj.get_variable('catch2_dep')]
nccl_deps = [dependency('cuda'), dependency('nccl')]
openmp_dep = dependency('openmp')

absl_proj = subproject('abseil-cpp')
absl_dep = [
    absl_proj.get_variable('absl_strings_dep'),
    absl_proj.get_variable('absl_synchronization_dep')
]

protoc = find_program('protoc')
grpc_cpp_plugin = find_program('grpc_cpp_plugin')
# meson: Using generator(): https://mesonbuild.com/Generating-sources.html#using-generator
# If you want to generate files for general purposes such as for generating headers to be used by several sources, or data that will be installed, and so on, use a custom_target() instead.
messages_generated = custom_target(
    'messages_generated',
    input: ['messages.proto'],
    output: ['@BASENAME@.pb.h', '@BASENAME@.pb.cc', '@BASENAME@.grpc.pb.h', '@BASENAME@.grpc.pb.cc'],
    command: [
        protoc,
        '--plugin=protoc-gen-grpc=' + grpc_cpp_plugin.path(),
        '--proto_path=@CURRENT_SOURCE_DIR@',
        '--cpp_out=@OUTDIR@',
        '--grpc_out=@OUTDIR@',
        '@INPUT@'
    ],
)
grpc_deps = [
    dependency('protobuf'),
    dependency('grpc++'),
]

pymod = import('python')
python_installation = pymod.find_installation('python3')
python_prog = find_program(python_installation.path())
pybind11_proj = subproject('pybind11')
pybind11_dep = pybind11_proj.get_variable('pybind11_dep')

# Note: libelf is something else
elf_core = shared_library(
    'elf_core',
    'controller.cc', 'rpc.cc', messages_generated,
    dependencies: [absl_dep, grpc_deps, threads_dep]
)

elf_comm_nccl = shared_library(
    'elf_comm_nccl',
    'nccl_communicator.cc',
    dependencies: [absl_dep, threads_dep, nccl_deps],
)

test_core = executable(
    'test_core',
    'test_main.cc', 'test_core.cc',
    dependencies: [test_deps, absl_dep, threads_dep],
    link_with: [elf_core],
)

test('test_core', test_core, timeout: 10)

test_nccl = executable(
    'test_nccl',
    'test_main.cc', 'test_nccl.cc',
    dependencies: [test_deps, absl_dep, threads_dep, nccl_deps, openmp_dep],
    link_with: [elf_comm_nccl],
)

test('test_nccl', test_nccl, timeout: 10, env: ['NCCL_DEBUG=WARN'])

test_worker = executable(
    'test_worker',
    'test_main.cc', 'test_worker.cc',
    dependencies: [test_deps, absl_dep, threads_dep],
    link_with: [elf_core, elf_comm_nccl],
)

test('test_worker', test_worker, timeout: 10, env: ['NCCL_DEBUG=WARN'])

python_installation.extension_module(
    '_elf',
    'python/_elf.cc',
    dependencies: [pybind11_dep],
    link_with: [elf_core],
)

test('test_python',
    python_prog,
    args: [files('test_python.py')],
    env: ['PYTHONPATH=' + meson.current_build_dir()]
)
