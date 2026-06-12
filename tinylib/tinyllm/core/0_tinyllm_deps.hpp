// TinyLLM dependency aggregator.
//
// Single include for all stdlib, platform, and third-party headers used
// across tinyllm. Every other tinyllm header includes only tinyllm files.

#pragma once

// MARK: - Exported dependencies

// --- C++ stdlib ---
#include <algorithm>    // IWYU pragma: export
#include <cerrno>       // IWYU pragma: export
#include <chrono>       // IWYU pragma: export
#include <cmath>        // IWYU pragma: export
#include <cctype>       // IWYU pragma: export
#include <cstdio>       // IWYU pragma: export
#include <cstdint>      // IWYU pragma: export
#include <cstdlib>      // IWYU pragma: export
#include <cstring>      // IWYU pragma: export
#include <filesystem>   // IWYU pragma: export
#include <limits>       // IWYU pragma: export
#include <map>          // IWYU pragma: export
#include <mutex>        // IWYU pragma: export
#include <random>       // IWYU pragma: export
#include <set>          // IWYU pragma: export
#include <sstream>      // IWYU pragma: export
#include <string>       // IWYU pragma: export
#include <thread>       // IWYU pragma: export
#include <vector>       // IWYU pragma: export

// --- POSIX ---
#include <dlfcn.h>      // IWYU pragma: export
#include <fcntl.h>      // IWYU pragma: export
#include <sys/mman.h>   // IWYU pragma: export
#include <sys/stat.h>   // IWYU pragma: export
#include <unistd.h>     // IWYU pragma: export

#if defined(__APPLE__)
#include <TargetConditionals.h> // IWYU pragma: export
#include <mach/mach.h>       // IWYU pragma: export
#include <mach/task_info.h>  // IWYU pragma: export
#endif

// --- Third-party ---
#include <sentencepiece_processor.h> // IWYU pragma: export

// --- tinylib ---
#include "tinylog.hpp" // IWYU pragma: export

// --- LiteRT C API ---
#include "litert_api.h" // IWYU pragma: export
