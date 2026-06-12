import Foundation

extension ZephrAgentRuntime {
    public enum LiteRTLM {}
}

extension ZephrAgentRuntime.LiteRTLM {

public typealias Engine = ZephrAgentRuntime.Conversation.Engine
public typealias Conversation = ZephrAgentRuntime.Conversation.Conversation
public typealias ConversationConfig = ZephrAgentRuntime.Conversation.Config
public typealias Contents = ZephrAgentRuntime.Conversation.Contents
public typealias Message = ZephrAgentRuntime.Conversation.Message
public typealias Tool = ZephrAgentRuntime.Conversation.Tool
public typealias ToolParam = ZephrAgentRuntime.Conversation.ToolParam
public typealias ToolParameterValue = ZephrAgentRuntime.Conversation.ToolParameterValue
public typealias ToolProvider = ZephrAgentRuntime.Conversation.ToolProvider
public typealias OpenApiTool = ZephrAgentRuntime.Conversation.OpenApiTool
public typealias ModelChannel = ZephrAgentRuntime.Lifecycle.ModelChannel

public enum Backend: Sendable {
    case cpu
    case gpu
    case npu
    case off
}

public struct EngineConfig: Sendable {
    public var modelChannel: ModelChannel
    public var textBackend: Backend
    public var embeddingBackend: Backend
    public var vlmBackend: Backend
    public var litertCompilationCacheEnabled: Bool
    public var gemma4Runtime: ZephrAgentRuntime.Lifecycle.Gemma4Options
    public var diagnosticGemma4: ZephrAgentRuntime.Diagnostics.Gemma4Options
    public var numThreads: Int
    public var llmExecutionChoiceID: String
    public var ragEmbeddingExecutionChoiceID: String
    public var vlmExecutionChoiceID: String
    public var modelStorage: ZephrAgentRuntime.Lifecycle.ModelStorage?

    public init(
        modelChannel: ModelChannel = .public,
        textBackend: Backend = .gpu,
        embeddingBackend: Backend = .off,
        vlmBackend: Backend = .off,
        litertCompilationCacheEnabled: Bool = false,
        gemma4Runtime: ZephrAgentRuntime.Lifecycle.Gemma4Options = .automatic,
        diagnosticGemma4: ZephrAgentRuntime.Diagnostics.Gemma4Options = .disabled,
        numThreads: Int = 0,
        llmExecutionChoiceID: String = "",
        ragEmbeddingExecutionChoiceID: String = "",
        vlmExecutionChoiceID: String = "",
        modelStorage: ZephrAgentRuntime.Lifecycle.ModelStorage? = nil
    ) {
        self.modelChannel = modelChannel
        self.textBackend = textBackend
        self.embeddingBackend = embeddingBackend
        self.vlmBackend = vlmBackend
        self.litertCompilationCacheEnabled = litertCompilationCacheEnabled
        self.gemma4Runtime = gemma4Runtime
        self.diagnosticGemma4 = diagnosticGemma4
        self.numThreads = numThreads
        self.llmExecutionChoiceID = llmExecutionChoiceID
        self.ragEmbeddingExecutionChoiceID = ragEmbeddingExecutionChoiceID
        self.vlmExecutionChoiceID = vlmExecutionChoiceID
        self.modelStorage = modelStorage
    }

    public var lifecycleConfiguration: ZephrAgentRuntime.Lifecycle.Configuration {
        ZephrAgentRuntime.Lifecycle.Configuration(
            modelChannel: modelChannel,
            llmExecutionChoiceID: llmExecutionChoiceID.isEmpty ? Self.textChoiceID(textBackend) : llmExecutionChoiceID,
            ragEmbeddingExecutionChoiceID: ragEmbeddingExecutionChoiceID.isEmpty
                ? Self.embeddingChoiceID(embeddingBackend)
                : ragEmbeddingExecutionChoiceID,
            vlmExecutionChoiceID: vlmExecutionChoiceID.isEmpty ? Self.visionChoiceID(vlmBackend) : vlmExecutionChoiceID,
            litertCompilationCacheEnabled: litertCompilationCacheEnabled,
            gemma4Runtime: gemma4Runtime,
            diagnosticGemma4: diagnosticGemma4,
            numThreads: numThreads,
            modelStorage: modelStorage
        )
    }

    private static func textChoiceID(_ backend: Backend) -> String {
        switch backend {
        case .cpu:
            "gemma4.cpu"
        case .gpu:
            "gemma4.gpu"
        case .npu:
            "gemma4.npu"
        case .off:
            "off"
        }
    }

    private static func embeddingChoiceID(_ backend: Backend) -> String {
        switch backend {
        case .cpu:
            "embedding.gemma3.cpu"
        case .gpu:
            "embedding.gemma3.gpu"
        case .npu:
            "embedding.gemma3.npu"
        case .off:
            "off"
        }
    }

    private static func visionChoiceID(_ backend: Backend) -> String {
        switch backend {
        case .cpu:
            "gemma4_vision.cpu"
        case .gpu:
            "gemma4_vision.gpu"
        case .npu:
            "gemma4_vision.npu"
        case .off:
            "off"
        }
    }
}

public static func tool<T: Tool>(_ tool: T) -> ToolProvider {
    ZephrAgentRuntime.Conversation.tool(tool)
}

public static func tool(_ openApiTool: any OpenApiTool) -> ToolProvider {
    ZephrAgentRuntime.Conversation.tool(openApiTool)
}

}

public extension ZephrAgentRuntime.LiteRTLM.Engine {
    func initialize(
        configuration: ZephrAgentRuntime.LiteRTLM.EngineConfig,
        progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler? = nil
    ) async throws {
        try await initialize(configuration: configuration.lifecycleConfiguration, progress: progress)
    }

    func prepare(
        configuration: ZephrAgentRuntime.LiteRTLM.EngineConfig,
        progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler? = nil
    ) async throws {
        try await prepare(configuration: configuration.lifecycleConfiguration, progress: progress)
    }
}
