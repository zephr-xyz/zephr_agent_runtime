# Zephr Agent Runtime

Zephr Agent Runtime is a small open source generative model runtime built on top of LiteRT that delivers a Gemma4 conversation implementation with tool calling for use on edge devices (supporting android, ios, macos and linux). Conversation inputs can contain both text and images.

We built this runtime to diagnose and workaround issues we found while trying to build edge ai products using another llm conversation runtime built on top of LiteRT.

The particular challenges we encountered when running gemma4 within the existing open source model runtime:

- the model would stop generating valid tool call parameter values as the number of conversation turns increases (we traced this down to precision issues in the model math implementation within the litert calculation and found a workaround)
- the model would intermittently fail to generate valid tool call structures
- using multiple litert context's for text generation and embedding adds avoidable latency to local rag generation processes
- we needed more diagnostic visibility in order to correctly produce/deploy working hardware-specific model variation's for the various LiteRT accelerator delegates
- we needed a place to manage the conversation history when nearing max context
- we needed to handle the model's conversational statefulness when implementing tool calls that issue against the VLM

This runtime SDK contains the core conversational implementation including grammar constrained tool call generation, the Android/kotlin and
Apple(ios+macos)/swift platform bridges, and a python nanobind interface. The conversation api is exposed as mostly-LiteRT-LM-shaped Kotlin and Swift facades. The
Python nanobind api allows directly using the same underlying runtime implementation from python on linux and macos.

## API Example: Kotlin LiteRT-LM Facade

```kotlin
import android.content.Context
import xyz.zephr.sdks.agent.litertlm.Backend
import xyz.zephr.sdks.agent.litertlm.Contents
import xyz.zephr.sdks.agent.litertlm.ConversationConfig
import xyz.zephr.sdks.agent.litertlm.Engine
import xyz.zephr.sdks.agent.litertlm.EngineConfig
import xyz.zephr.sdks.agent.litertlm.Tool
import xyz.zephr.sdks.agent.litertlm.ToolSet
import xyz.zephr.sdks.agent.litertlm.tool

class Tools : ToolSet {
    @Tool(description = "Get the distance and direction of a place")
    fun whereIs(place: String): String = "$place is 80 meters ahead and to the right."
}

suspend fun chatWithZephrAgentRuntime(context: Context): String {
    val engine = Engine(context.applicationContext)
    engine.initialize(
        EngineConfig(
            textBackend = Backend.GPU,
            embeddingBackend = Backend.OFF,
            vlmBackend = Backend.OFF,
            litertCompilationCacheEnabled = true,
        )
    )

    val conversation = engine.createConversation(
        ConversationConfig(
            systemInstruction = Contents.of("Answer briefly. Use tools when helpful."),
            tools = listOf(tool(Tools())),
        )
    )

    return conversation.sendMessage(Contents.of("Where is the library?")).contents.toString()
}
```

## Compared With Kotlin LiteRT-LM

```kotlin
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Tool
import com.google.ai.edge.litertlm.ToolSet
import com.google.ai.edge.litertlm.tool

class Tools : ToolSet {
    @Tool(description = "Get the distance and direction of a place")
    fun whereIs(place: String): String = "$place is 80 meters ahead and to the right."
}

suspend fun chatWithDirectLiteRtLm(modelPath: String, cacheDir: String): String {
    val engine = Engine(
        EngineConfig(
            modelPath = modelPath,
            cacheDir = cacheDir,
            backend = Backend.GPU(),
        )
    )
    engine.initialize()

    val conversation = engine.createConversation(
        ConversationConfig(
            systemInstruction = Contents.of("Answer briefly. Use tools when helpful."),
            tools = listOf(tool(Tools())),
        )
    )

    return conversation.sendMessage("Where is the library?").contents.toString()
}
```

## API Example: Swift LiteRT-LM Facade

```swift
import ZephrAgentRuntime

struct Tools: ZephrAgentRuntime.LiteRTLM.Tool {
    static let name = "where_is"
    static let description = "Get the distance and direction of a place"

    @ZephrAgentRuntime.LiteRTLM.ToolParam(description: "Place name")
    var place: String

    init() {}

    func run() async throws -> Any {
        "\(place) is 80 meters ahead and to the right."
    }
}

@MainActor
func chatWithZephrAgentRuntime() async throws -> String {
    let engine = ZephrAgentRuntime.LiteRTLM.Engine()
    try await engine.initialize(
        configuration: ZephrAgentRuntime.LiteRTLM.EngineConfig(
            textBackend: .gpu,
            embeddingBackend: .off,
            vlmBackend: .off,
            litertCompilationCacheEnabled: true
        )
    )

    let conversation = engine.createConversation(
        config: ZephrAgentRuntime.LiteRTLM.ConversationConfig(
            systemInstruction: .of("Answer briefly. Use tools when helpful."),
            tools: [ZephrAgentRuntime.LiteRTLM.tool(Tools())]
        )
    )

    let response = try await conversation.sendMessage(.of("Where is the library?"))
    return response.contents.description
}
```

## Compared With Swift LiteRT-LM

```swift
import LiteRTLM

struct Tools: Tool {
    static let name = "where_is"
    static let description = "Get the distance and direction of a place"

    @ToolParam(description: "Place name")
    var place: String

    init() {}

    func run() async throws -> Any {
        "\(place) is 80 meters ahead and to the right."
    }
}

@MainActor
func chatWithDirectLiteRtLm(modelPath: String, cacheDir: String) async throws -> String {
    let engine = Engine(
        EngineConfig(
            modelPath: modelPath,
            cacheDir: cacheDir,
            backend: .gpu
        )
    )
    try await engine.initialize()

    let conversation = engine.createConversation(
        config: ConversationConfig(
            systemInstruction: .of("Answer briefly. Use tools when helpful."),
            tools: [Tools()]
        )
    )

    let response = try await conversation.sendMessage(.of("Where is the library?"))
    return response.contents.description
}
```

## Layout

- `tinylib/tinyllm/`: C++ LiteRT inference and Gemma runtime code.
- `tinylib/zephr_agent_runtime/`: shared C API and native SDK bridge.
- `tinylib/tinylog/`: small logging support library.
- `android/sdks/zephr_agent_runtime/`: Android SDK module.
- `apple/sdks/zephr_agent_runtime/`: Apple SDK framework source.
- `clis/zephr_agent_runtime/`: Python package wrapper for the native runtime.
- `clis/support/`: local developer setup and native dependency tooling.
- `data/model_manifests/channels/public.json`: default public model manifest.
- `docs/third_party_native_deps.md`: native dependency audit notes.

Generated local outputs live under `1_build/` and `2_output/`. Shared native
dependency artifacts are stored under
`/opt/zephr/diagnosis/native_dependencies/zephragent`.

## Use the Prebuilt Android AAR

For Android apps, the runtime is published on Maven Central:

```kotlin
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}
```

```kotlin
dependencies {
    implementation("xyz.zephr.sdks.agent:zephr-agent-runtime:0.0.5")
}
```

The AAR includes the Android runtime bridge, native LiteRT libraries, and the
public model manifest:

```kotlin
import xyz.zephr.sdks.agent.litertlm.Engine
import xyz.zephr.sdks.agent.litertlm.EngineConfig
```

## Development Setup

```bash
uv sync
uv run prepare_platform_tools
uv run prepare_native_deps
uv run prepare_native_deps --status
uv run prepare_dev
```

The first `prepare_native_deps` run can take a while because it builds the
pinned native runtime dependencies. Later checkouts with the same recipe hash
reuse the global artifact.

## Build

Build the Android SDK:

```bash
cd android
./gradlew :sdks:zephr_agent_runtime:assembleDebug
```

Build the Apple SDK project:

```bash
cd apple
xcodegen generate
xcodebuild -project ZephrAgentRuntime.xcodeproj -scheme zephr_agent_runtime -sdk macosx -configuration Debug build CODE_SIGNING_ALLOWED=NO
```
