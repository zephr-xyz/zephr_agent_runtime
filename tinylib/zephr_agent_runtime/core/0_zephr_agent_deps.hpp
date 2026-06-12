// Zephr Agent Runtime dependency aggregator.
//
// Single include for all stdlib headers used across zephr_agent_runtime.
// Every other zephr_agent header includes only zephr_agent or tinyllm files.

#pragma once

// MARK: - Exported dependencies

// --- C++ stdlib ---
#include <algorithm>    // IWYU pragma: export
#include <cctype>     // IWYU pragma: export
#include <chrono>     // IWYU pragma: export
#include <cmath>    // IWYU pragma: export
#include <cstdint>    // IWYU pragma: export
#include <cstdio>     // IWYU pragma: export
#include <map>    // IWYU pragma: export
#include <optional>     // IWYU pragma: export
#include <regex>    // IWYU pragma: export
#include <sstream>    // IWYU pragma: export
#include <string>     // IWYU pragma: export
#include <utility>    // IWYU pragma: export
#include <vector>     // IWYU pragma: export

// --- tinylog ---
#include "tinylog.hpp" // IWYU pragma: export

// --- tinyllm ---
#include "core/tinyllm_engine.hpp" // IWYU pragma: export
