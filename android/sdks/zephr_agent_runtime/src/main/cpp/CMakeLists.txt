cmake_minimum_required(VERSION 3.30.5)

project(zephr_agent_android LANGUAGES C CXX)

if(NOT DEFINED REPO_ROOT)
    get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
endif()
if(NOT DEFINED ZEPHR_NATIVE_DEPS_ROOT)
    message(FATAL_ERROR
        "ZEPHR_NATIVE_DEPS_ROOT is not set.\n"
        "Run: uv run prepare_dev before building Android targets.")
endif()
if(NOT DEFINED ZEPHR_NATIVE_DEPS_RECIPE_HASH)
    message(FATAL_ERROR
        "ZEPHR_NATIVE_DEPS_RECIPE_HASH is not set.\n"
        "Run: uv run prepare_dev before building Android targets.")
endif()

set(NATIVE_DEPS_MANIFEST "${ZEPHR_NATIVE_DEPS_ROOT}/manifest.json")
if(NOT EXISTS "${NATIVE_DEPS_MANIFEST}")
    message(FATAL_ERROR
        "Native deps manifest not found at ${NATIVE_DEPS_MANIFEST}\n"
        "Run: uv run prepare_native_deps && uv run prepare_dev")
endif()
file(READ "${NATIVE_DEPS_MANIFEST}" NATIVE_DEPS_MANIFEST_TEXT)
string(FIND "${NATIVE_DEPS_MANIFEST_TEXT}" "\"recipe_hash\": \"${ZEPHR_NATIVE_DEPS_RECIPE_HASH}\"" NATIVE_DEPS_MANIFEST_HASH_OFFSET)
if(NATIVE_DEPS_MANIFEST_HASH_OFFSET EQUAL -1)
    message(FATAL_ERROR
        "Native deps manifest at ${NATIVE_DEPS_MANIFEST} does not match bridge recipe ${ZEPHR_NATIVE_DEPS_RECIPE_HASH}\n"
        "Run: uv run prepare_native_deps && uv run prepare_dev")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(NATIVE_DEPS_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
if(NATIVE_DEPS_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(NATIVE_DEPS_BUILD_TYPE "Release")
endif()

set(NATIVE_DEPS_ANDROID_BRIDGE "${CMAKE_CURRENT_SOURCE_DIR}/native_deps_android/${NATIVE_DEPS_BUILD_TYPE}")
if(NOT EXISTS "${NATIVE_DEPS_ANDROID_BRIDGE}")
    message(FATAL_ERROR
        "Android native deps bridge not found at ${NATIVE_DEPS_ANDROID_BRIDGE}\n"
        "Run: uv run prepare_dev --platform android")
endif()
set(NATIVE_DEPS_DIR "${NATIVE_DEPS_ANDROID_BRIDGE}/${ANDROID_ABI}")
if(NOT EXISTS "${NATIVE_DEPS_DIR}")
    message(FATAL_ERROR
        "Android native deps for ${ANDROID_ABI} not found at ${NATIVE_DEPS_DIR}\n"
        "Run: uv run prepare_native_deps --platform android && uv run prepare_dev --platform android")
endif()
set(ANDROID_TINYLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tinylib")
set(TINYLOG_DIR "${ANDROID_TINYLIB_DIR}/tinylog")
set(TINYLLM_DIR "${ANDROID_TINYLIB_DIR}/tinyllm")
set(ZEPHR_AGENT_DIR "${ANDROID_TINYLIB_DIR}/zephr_agent_runtime")

find_library(android_log log REQUIRED)
find_library(android_lib android REQUIRED)
find_library(z_lib z REQUIRED)

add_library(litert SHARED IMPORTED)
set_target_properties(litert PROPERTIES IMPORTED_LOCATION "${NATIVE_DEPS_DIR}/litert/lib/libLiteRt.so")

add_library(sentencepiece STATIC IMPORTED)
set_target_properties(sentencepiece PROPERTIES IMPORTED_LOCATION "${NATIVE_DEPS_DIR}/sentencepiece/lib/libsentencepiece.a")

add_library(TINYLLM_NATIVE_DEPS INTERFACE)
target_compile_definitions(TINYLLM_NATIVE_DEPS INTERFACE TINYLLM_LITERT_DIRECT_LINK=1)
target_include_directories(TINYLLM_NATIVE_DEPS INTERFACE
    "${NATIVE_DEPS_DIR}/litert/include"
    "${NATIVE_DEPS_DIR}/sentencepiece/include"
)
target_link_libraries(TINYLLM_NATIVE_DEPS INTERFACE
    litert
    sentencepiece
    ${android_log}
    ${android_lib}
    ${z_lib}
)

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
add_library(zephr_agent_android SHARED zephr_agent_jni.cpp)
target_compile_features(zephr_agent_android PRIVATE cxx_std_20)
target_compile_options(zephr_agent_android PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-missing-field-initializers
)
target_link_options(zephr_agent_android PRIVATE
    -Wl,-z,max-page-size=16384
)
target_include_directories(zephr_agent_android PRIVATE
    "${ZEPHR_AGENT_DIR}/public"
)
target_link_libraries(zephr_agent_android PRIVATE
    zephr_agent_c_api
    tinyllm
    tinylog
    TINYLLM_NATIVE_DEPS
)
