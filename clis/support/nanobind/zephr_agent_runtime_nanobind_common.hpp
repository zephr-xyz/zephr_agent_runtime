// Shared declarations for Zephr Agent Python native modules.

#pragma once

#include "core/tinyllm_engine.hpp"

struct ZephrAgentRuntimeWrapper {
    // Nanobind keeps the GIL for these methods today, so Python calls are
    // serialized. Add explicit owner locking before any gil_scoped_release use.
    TinyLLMEngine engine;
};
