package xyz.zephr.sdks.agent.litertlm

import xyz.zephr.sdks.agent.ZephrAgentRuntime

/**
 * LiteRT-LM shaped public facade over the Zephr Agent runtime.
 *
 * This file intentionally exposes top-level API names so callers can use:
 *
 * import xyz.zephr.sdks.agent.litertlm.*
 */

public typealias Engine = ZephrAgentRuntime.Conversation.Engine

public typealias Conversation = ZephrAgentRuntime.Conversation.Conversation

public typealias ConversationConfig = ZephrAgentRuntime.Conversation.Config

public typealias Contents = ZephrAgentRuntime.Conversation.Contents

public typealias Message = ZephrAgentRuntime.Conversation.Message

public typealias Strategy = ZephrAgentRuntime.Conversation.Strategy

public typealias Tool = ZephrAgentRuntime.Conversation.Tool

public typealias ToolSet = ZephrAgentRuntime.Conversation.ToolSet

public typealias OpenApiTool = ZephrAgentRuntime.Conversation.OpenApiTool

public typealias ToolProvider = ZephrAgentRuntime.Conversation.ToolProvider

public typealias ModelChannel = ZephrAgentRuntime.Lifecycle.ModelChannel

public data class EngineConfig(
    val modelChannel: ModelChannel = ModelChannel.PUBLIC,
    val textBackend: Backend = Backend.GPU,
    val embeddingBackend: Backend = Backend.OFF,
    val vlmBackend: Backend = Backend.OFF,
    val litertCompilationCacheEnabled: Boolean = false,
    val litertRuntimeLibraryDir: String? = null,
    val gemma4Runtime: ZephrAgentRuntime.Lifecycle.Gemma4Options = ZephrAgentRuntime.Lifecycle.Gemma4Options(),
    val diagnosticGemma4: ZephrAgentRuntime.Diagnostics.Gemma4Options = ZephrAgentRuntime.Diagnostics.Gemma4Options(),
    val numThreads: Int = 0,
    val llmExecutionChoiceId: String = "",
    val ragEmbeddingExecutionChoiceId: String = "",
    val vlmExecutionChoiceId: String = "",
) {
    internal fun toLifecycleConfiguration(): ZephrAgentRuntime.Lifecycle.Configuration =
        ZephrAgentRuntime.Lifecycle.Configuration(
            modelChannel = modelChannel,
            llmExecutionChoiceId = llmExecutionChoiceId.ifBlank {
                textChoiceId(textBackend)
            },
            ragEmbeddingExecutionChoiceId = ragEmbeddingExecutionChoiceId.ifBlank {
                embeddingChoiceId(embeddingBackend)
            },
            vlmExecutionChoiceId = vlmExecutionChoiceId.ifBlank {
                visionChoiceId(vlmBackend)
            },
            litertCompilationCacheEnabled = litertCompilationCacheEnabled,
            litertRuntimeLibraryDir = litertRuntimeLibraryDir,
            gemma4Runtime = gemma4Runtime,
            diagnosticGemma4 = diagnosticGemma4,
            numThreads = numThreads,
        )
}

public enum class Backend {
    CPU,
    GPU,
    NPU,
    OFF;
}

public suspend fun Engine.initialize(configuration: EngineConfig) {
    initialize(configuration.toLifecycleConfiguration())
}

public suspend fun Engine.prepare(configuration: EngineConfig) {
    prepare(configuration.toLifecycleConfiguration())
}

public fun tool(toolSet: ToolSet): ToolProvider =
    ZephrAgentRuntime.Conversation.tool(toolSet)

public fun tool(openApiTool: OpenApiTool): ToolProvider =
    ZephrAgentRuntime.Conversation.tool(openApiTool)

private fun textChoiceId(backend: Backend): String =
    when (backend) {
        Backend.CPU -> "gemma4.cpu"
        Backend.GPU -> "gemma4.gpu"
        Backend.NPU -> "gemma4.npu"
        Backend.OFF -> "off"
    }

private fun embeddingChoiceId(backend: Backend): String =
    when (backend) {
        Backend.CPU -> "embedding.gemma3.cpu"
        Backend.GPU -> "embedding.gemma3.gpu"
        Backend.NPU -> "embedding.gemma3.npu"
        Backend.OFF -> "off"
    }

private fun visionChoiceId(backend: Backend): String =
    when (backend) {
        Backend.CPU -> "gemma4_vision.cpu"
        Backend.GPU -> "gemma4_vision.gpu"
        Backend.NPU -> "gemma4_vision.npu"
        Backend.OFF -> "off"
    }
