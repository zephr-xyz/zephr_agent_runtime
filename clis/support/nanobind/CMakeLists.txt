# Build Zephr Agent Python native adapters for the host machine's platform,
# which must be either macOS or Linux.
cmake_minimum_required(VERSION 3.30.5)
project(zephr_agent_runtime_nanobind CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

set(NATIVE_DEPS_BRIDGE "${REPO_ROOT}/1_build/native_deps/native_deps.cmake")
if(NOT EXISTS "${NATIVE_DEPS_BRIDGE}")
    message(FATAL_ERROR
        "Native deps bridge not found at ${NATIVE_DEPS_BRIDGE}\n"
        "Run: uv run prepare_dev")
endif()
include("${NATIVE_DEPS_BRIDGE}")


# ---------------------------------------------------------------------------
# Native dependencies are expected to be built by: prepare_native_deps [--platform host]
# cmake expects to find native dependency artifacts laid out according to the conventions arranged by prepare_native_deps
# ---------------------------------------------------------------------------

if(APPLE)
    set(HOST_NATIVE_PLATFORM "macos")
    set(LITERT_LIB_NAME "libLiteRt.dylib")
else()
    set(HOST_NATIVE_PLATFORM "linux")
    set(LITERT_LIB_NAME "libLiteRt.so")
endif()

set(NATIVE_DEPS_DIR "${ZEPHR_NATIVE_DEPS_ROOT}/${HOST_NATIVE_PLATFORM}/2_installed/${CMAKE_BUILD_TYPE}/${CMAKE_HOST_SYSTEM_PROCESSOR}")
if(NOT EXISTS "${NATIVE_DEPS_DIR}")
    message(FATAL_ERROR
        "Native dependencies not found at ${NATIVE_DEPS_DIR}\n"
        "Run: uv run prepare_native_deps && uv run prepare_dev")
endif()

set(TINYLLM_DIR "${REPO_ROOT}/tinylib/tinyllm")
set(ZEPHR_AGENT_DIR "${REPO_ROOT}/tinylib/zephr_agent_runtime")
set(TINYLOG_DIR "${REPO_ROOT}/tinylib/tinylog")
set(ZEPHR_AGENT_NANOBIND_NATIVE_DIR "${REPO_ROOT}/clis/zephr_agent_runtime/nanobind")
set(ZEPHR_AGENT_TOOLS_NANOBIND_NATIVE_DIR "${REPO_ROOT}/clis/zephr_agent_tools/nanobind")


# ---------------------------------------------------------------------------
# find_package
# ---------------------------------------------------------------------------

set(Python_ROOT_DIR "${Python_EXECUTABLE}/../..")
set(Python_FIND_VIRTUALENV ONLY)
set(Python_FIND_STRATEGY LOCATION)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)

list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Native library resolution
# ---------------------------------------------------------------------------

# Sentencepiece
set(SENTENCEPIECE_DIR "${NATIVE_DEPS_DIR}/sentencepiece")
set(SENTENCEPIECE_LIB "${SENTENCEPIECE_DIR}/lib/libsentencepiece.a")
if(NOT EXISTS "${SENTENCEPIECE_LIB}")
    message(FATAL_ERROR "libsentencepiece.a not found at ${SENTENCEPIECE_LIB}")
endif()

# LiteRT
set(LITERT_PKG_DIR "${NATIVE_DEPS_DIR}/litert/lib")
set(LITERT_LIB "${LITERT_PKG_DIR}/${LITERT_LIB_NAME}")
if(NOT EXISTS "${LITERT_LIB}")
    message(FATAL_ERROR
        "libLiteRt not found at ${LITERT_LIB}\n"
        "Run: rebuild_tinyllm --CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}")
endif()
set(LITERT_SOURCE_DIR "${ZEPHR_NATIVE_DEPS_ROOT}/downloads/litert")
set(LITERT_CONFIG_INCLUDE_DIR "${ZEPHR_NATIVE_DEPS_ROOT}/${HOST_NATIVE_PLATFORM}/1_configure/${CMAKE_BUILD_TYPE}/${CMAKE_HOST_SYSTEM_PROCESSOR}/litert/include")
set(LITERT_ABSL_DIR "${ZEPHR_NATIVE_DEPS_ROOT}/${HOST_NATIVE_PLATFORM}/1_configure/${CMAKE_BUILD_TYPE}/${CMAKE_HOST_SYSTEM_PROCESSOR}/litert/abseil-cpp")
if(NOT EXISTS "${LITERT_CONFIG_INCLUDE_DIR}/litert/build_common/build_config.h")
    message(FATAL_ERROR
        "LiteRT generated config headers not found at ${LITERT_CONFIG_INCLUDE_DIR}\n"
        "Run: prepare_native_deps --platform ${HOST_NATIVE_PLATFORM}")
endif()
if(NOT EXISTS "${LITERT_ABSL_DIR}/absl/container/flat_hash_map.h")
    message(FATAL_ERROR
        "LiteRT abseil headers not found at ${LITERT_ABSL_DIR}\n"
        "Run: prepare_native_deps --platform ${HOST_NATIVE_PLATFORM}")
endif()

set(ZEPHR_AGENT_TOOLS_GENERATED_DIR "${REPO_ROOT}/tinylib/zephr_agent_tools/generated")
set(ZEPHR_AGENT_TOOLS_GENERATED_PROTO_DIR "${ZEPHR_AGENT_TOOLS_GENERATED_DIR}/proto")

set(ZEPHR_AGENT_TOOLS_AVAILABLE OFF)
if(EXISTS "${REPO_ROOT}/tinylib/zephr_agent_tools")
    set(ZEPHR_AGENT_TOOLS_AVAILABLE ON)
endif()
option(BUILD_ZEPHR_AGENT_TOOLS_NANOBIND
    "Build private zephr_agent_tools Python native extension"
    ${ZEPHR_AGENT_TOOLS_AVAILABLE})

if(BUILD_ZEPHR_AGENT_TOOLS_NANOBIND)
    if(NOT EXISTS "${ZEPHR_AGENT_TOOLS_GENERATED_PROTO_DIR}/waypoints.pb.cc")
        message(FATAL_ERROR
            "waypoints.pb.cc not found at ${ZEPHR_AGENT_TOOLS_GENERATED_PROTO_DIR}/waypoints.pb.cc\n"
            "Run: uv run prepare_native_deps")
    endif()

    # Protobuf-lite (cross-compiled v3.14.0, same version sentencepiece bundles internally)
    set(PROTOBUF_LITE_DIR "${NATIVE_DEPS_DIR}/protobuf-lite")
    set(PROTOBUF_LITE_LIB "${PROTOBUF_LITE_DIR}/lib/libprotobuf-lite.a")
    if(NOT EXISTS "${PROTOBUF_LITE_LIB}")
        message(FATAL_ERROR "libprotobuf-lite.a not found at ${PROTOBUF_LITE_LIB}")
    endif()
endif()

# ---------------------------------------------------------------------------
# TINYLLM_NATIVE_DEPS — interface target aggregating all native dependencies
# ---------------------------------------------------------------------------

add_library(TINYLLM_NATIVE_DEPS INTERFACE)
target_link_libraries(TINYLLM_NATIVE_DEPS INTERFACE
    "${LITERT_LIB}"
    "${SENTENCEPIECE_LIB}"
)
target_include_directories(TINYLLM_NATIVE_DEPS INTERFACE
    "${SENTENCEPIECE_DIR}/include"
    "${LITERT_SOURCE_DIR}"
    "${LITERT_CONFIG_INCLUDE_DIR}"
    "${LITERT_ABSL_DIR}"
)

add_library(ZEPHR_AGENT_TOOLS_NATIVE_DEPS INTERFACE)
if(BUILD_ZEPHR_AGENT_TOOLS_NANOBIND)
    target_link_libraries(ZEPHR_AGENT_TOOLS_NATIVE_DEPS INTERFACE
        "${PROTOBUF_LITE_LIB}"
    )
    target_include_directories(ZEPHR_AGENT_TOOLS_NATIVE_DEPS INTERFACE
        "${PROTOBUF_LITE_DIR}/include"
    )
endif()

add_library(tinylog INTERFACE)
add_subdirectory("${TINYLOG_DIR}" tinylog_sources)

add_library(tinyllm OBJECT)
target_compile_options(tinyllm PRIVATE -Wno-deprecated-pragma)
target_link_libraries(tinyllm PRIVATE
    tinylog
    TINYLLM_NATIVE_DEPS
)
add_subdirectory("${TINYLLM_DIR}" tinyllm_sources)

add_library(zephr_agent_c_api OBJECT)
target_link_libraries(zephr_agent_c_api PRIVATE
    tinyllm
    tinylog
    TINYLLM_NATIVE_DEPS
)
add_subdirectory("${ZEPHR_AGENT_DIR}" zephr_agent_sources)

# ---------------------------------------------------------------------------
# nanobind extension modules
# ---------------------------------------------------------------------------

nanobind_add_module(zephr_agent_runtime_nanobind NB_STATIC
    ${CMAKE_CURRENT_LIST_DIR}/zephr_agent_runtime_nanobind.cpp
)

install(TARGETS zephr_agent_runtime_nanobind
    LIBRARY DESTINATION "${ZEPHR_AGENT_NANOBIND_NATIVE_DIR}"
)

target_include_directories(zephr_agent_runtime_nanobind PRIVATE
    ${TINYLLM_DIR}
    ${ZEPHR_AGENT_DIR}/core
)

target_link_libraries(zephr_agent_runtime_nanobind PRIVATE
    tinyllm
    zephr_agent_c_api
    tinylog
    TINYLLM_NATIVE_DEPS
)

set_target_properties(zephr_agent_runtime_nanobind PROPERTIES
    BUILD_RPATH "${LITERT_PKG_DIR}"
    INSTALL_RPATH "${LITERT_PKG_DIR}"
)

target_compile_options(zephr_agent_runtime_nanobind PRIVATE -Wall -Wextra -O2)

if(BUILD_ZEPHR_AGENT_TOOLS_NANOBIND)
    nanobind_add_module(zephr_agent_tools_nanobind NB_STATIC
        ${CMAKE_CURRENT_LIST_DIR}/zephr_agent_tools_nanobind.cpp
        ${ZEPHR_AGENT_TOOLS_GENERATED_PROTO_DIR}/waypoints.pb.cc
    )

    install(TARGETS zephr_agent_tools_nanobind
        LIBRARY DESTINATION "${ZEPHR_AGENT_TOOLS_NANOBIND_NATIVE_DIR}"
    )

    target_include_directories(zephr_agent_tools_nanobind PRIVATE
        ${TINYLLM_DIR}
        ${ZEPHR_AGENT_DIR}/core
        ${REPO_ROOT}/tinylib/zephr_agent_tools
        ${ZEPHR_AGENT_TOOLS_GENERATED_DIR}
    )

    target_link_libraries(zephr_agent_tools_nanobind PRIVATE
        tinyllm
        zephr_agent_c_api
        tinylog
        TINYLLM_NATIVE_DEPS
        ZEPHR_AGENT_TOOLS_NATIVE_DEPS
    )

    set_target_properties(zephr_agent_tools_nanobind PROPERTIES
        BUILD_RPATH "${LITERT_PKG_DIR}"
        INSTALL_RPATH "${LITERT_PKG_DIR}"
    )

    target_compile_options(zephr_agent_tools_nanobind PRIVATE -Wall -Wextra -O2)
endif()

# ---------------------------------------------------------------------------
# Type stub generation (.pyi)
# ---------------------------------------------------------------------------

nanobind_add_stub(zephr_agent_runtime_nanobind_stub
    MODULE zephr_agent_runtime_nanobind
    OUTPUT "${ZEPHR_AGENT_NANOBIND_NATIVE_DIR}/zephr_agent_runtime_nanobind.pyi"
    PYTHON_PATH "${ZEPHR_AGENT_NANOBIND_NATIVE_DIR}"
    DEPENDS zephr_agent_runtime_nanobind
)

if(BUILD_ZEPHR_AGENT_TOOLS_NANOBIND)
    nanobind_add_stub(zephr_agent_tools_nanobind_stub
        MODULE zephr_agent_tools_nanobind
        OUTPUT "${ZEPHR_AGENT_TOOLS_NANOBIND_NATIVE_DIR}/zephr_agent_tools_nanobind.pyi"
        PYTHON_PATH "${ZEPHR_AGENT_TOOLS_NANOBIND_NATIVE_DIR}"
        DEPENDS zephr_agent_tools_nanobind
    )
endif()
