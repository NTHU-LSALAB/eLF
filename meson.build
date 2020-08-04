project('elfcore', 'cpp')

clang = meson.get_compiler('cpp')
assert(
    clang.get_id() == 'clang',
    'clang must be used. Specify the environment variable CXX=clang++',
)

threads_dep = dependency('threads')
catch2_proj = subproject('catch2')
test_deps = [catch2_proj.get_variable('catch2_dep')]

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

test_core = executable(
    'test_core',
    'test_main.cc', 'test_core.cc', 'controller.cc', 'rpc.cc', messages_generated,
    dependencies: [test_deps, absl_dep, grpc_deps, threads_dep],
)

test('core', test_core, timeout: 10)
