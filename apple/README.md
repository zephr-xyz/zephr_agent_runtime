# Apple SDK

This directory contains the XcodeGen project description for the Apple SDK.

The workspace contains one framework target:

- `sdks/zephr_agent_runtime`: Swift and Objective-C++ wrapper around the shared
  Zephr Agent C API, including the Swift conversation API and
  `ZephrAgentRuntime.LiteRTLM` facade.

The shared C++ implementation remains in `../tinylib`. Apple targets should
link it through the C API rather than copying native sources.

`apple/project.yml` is the source of truth for Xcode targets. Regenerate
`ZephrAgentRuntime.xcodeproj` after changing target membership or dependencies.

## Setup

From the repository root:

```bash
uv sync
uv run prepare_platform_tools
uv run prepare_native_deps
uv run prepare_dev
```

Build the macOS framework target:

```bash
cd apple
xcodegen generate
xcodebuild -project ZephrAgentRuntime.xcodeproj -scheme zephr_agent_runtime -sdk macosx -configuration Debug build CODE_SIGNING_ALLOWED=NO
```
