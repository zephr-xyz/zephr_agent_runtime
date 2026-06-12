#pragma once

// Public framework umbrella header.
//
// Swift framework targets do not use an app-style bridging header for their
// own C API. Instead, Swift sees C symbols through the generated Clang module,
// which Xcode builds from public framework headers. Keep this header public so
// ZephrAgentRuntimeCBridge.swift can call the zephr_* API, and only re-export the
// narrow C bridge here; private C++ headers stay out of the module.
//
// Consumers can technically call these C symbols because they are part of the
// public framework module, but they are not the supported SDK contract and are
// not source-stability guaranteed. Use the public Swift API instead.
#import <ZephrAgentRuntime/zephr_agent_c_api.h>
