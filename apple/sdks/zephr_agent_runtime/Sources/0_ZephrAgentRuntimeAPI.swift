import CoreGraphics
import Darwin
import Foundation
import Observation

// MARK: - Namespace roots

public enum ZephrAgentRuntime {}

extension ZephrAgentRuntime {
    public enum Conversation {}
    public enum Diagnostics {}
    public enum Embeddings {}
    public enum Tools {}
    public enum Lifecycle {}
}

extension ZephrAgentRuntime.Diagnostics {

// MARK: - Generation diagnostics

public struct StageStats: Sendable {
    public let tokenizeMs: Int64
    public let prefillMs: Int64
    public let decodeMs: Int64
    public let firstDecodeMs: Int64
    public let inputTokens: Int
    public let outputTokens: Int
    public let mtpRejectedCycles: Int
    public let mtpRejectedAfterPrefix0: Int
    public let mtpRejectedAfterPrefix1: Int
    public let mtpRejectedAfterPrefix2: Int

    public init(
        tokenizeMs: Int64,
        prefillMs: Int64,
        decodeMs: Int64,
        firstDecodeMs: Int64,
        inputTokens: Int,
        outputTokens: Int,
        mtpRejectedCycles: Int = 0,
        mtpRejectedAfterPrefix0: Int = 0,
        mtpRejectedAfterPrefix1: Int = 0,
        mtpRejectedAfterPrefix2: Int = 0
    ) {
        self.tokenizeMs = tokenizeMs
        self.prefillMs = prefillMs
        self.decodeMs = decodeMs
        self.firstDecodeMs = firstDecodeMs
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.mtpRejectedCycles = mtpRejectedCycles
        self.mtpRejectedAfterPrefix0 = mtpRejectedAfterPrefix0
        self.mtpRejectedAfterPrefix1 = mtpRejectedAfterPrefix1
        self.mtpRejectedAfterPrefix2 = mtpRejectedAfterPrefix2
    }
}

public struct GenerationStats: Sendable {
    public let prefillTokens: Int
    public let decodeTokens: Int
    public let prefillMs: Int64
    public let decodeMs: Int64
    public let firstDecodeMs: Int64
    public let modelCalls: Int

    public var prefillTokensPerSecond: Double {
        Self.tokensPerSecond(tokens: prefillTokens, millis: prefillMs)
    }

    public var decodeTokensPerSecond: Double {
        Self.tokensPerSecond(tokens: decodeTokens, millis: decodeMs)
    }

    public init(
        prefillTokens: Int,
        decodeTokens: Int,
        prefillMs: Int64,
        decodeMs: Int64,
        firstDecodeMs: Int64,
        modelCalls: Int = 1
    ) {
        self.prefillTokens = prefillTokens
        self.decodeTokens = decodeTokens
        self.prefillMs = prefillMs
        self.decodeMs = decodeMs
        self.firstDecodeMs = firstDecodeMs
        self.modelCalls = modelCalls
    }

    public func adding(_ other: GenerationStats) -> GenerationStats {
        GenerationStats(
            prefillTokens: prefillTokens + other.prefillTokens,
            decodeTokens: decodeTokens + other.decodeTokens,
            prefillMs: prefillMs + other.prefillMs,
            decodeMs: decodeMs + other.decodeMs,
            firstDecodeMs: firstDecodeMs > 0 ? firstDecodeMs : other.firstDecodeMs,
            modelCalls: modelCalls + other.modelCalls
        )
    }

    private static func tokensPerSecond(tokens: Int, millis: Int64) -> Double {
        guard tokens > 0, millis > 0 else { return 0.0 }
        return Double(tokens) * 1000.0 / Double(millis)
    }
}

public struct MemorySample: Sendable {
    public var label: String
    public var rssMb: Int64?
    public var virtualMb: Int64?
    public var physicalFootprintMb: Int64?
    public var internalMb: Int64?
    public var compressedMb: Int64?

    public init(
        label: String,
        rssMb: Int64? = nil,
        virtualMb: Int64? = nil,
        physicalFootprintMb: Int64? = nil,
        internalMb: Int64? = nil,
        compressedMb: Int64? = nil
    ) {
        self.label = label
        self.rssMb = rssMb
        self.virtualMb = virtualMb
        self.physicalFootprintMb = physicalFootprintMb
        self.internalMb = internalMb
        self.compressedMb = compressedMb
    }

    public static func capture(label: String) -> MemorySample {
        var basic = mach_task_basic_info_data_t()
        var basicCount = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<natural_t>.size)
        let basicStatus = withUnsafeMutablePointer(to: &basic) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(basicCount)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &basicCount)
            }
        }

        var vm = task_vm_info_data_t()
        var vmCount = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let vmStatus = withUnsafeMutablePointer(to: &vm) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(vmCount)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &vmCount)
            }
        }

        return MemorySample(
            label: label,
            rssMb: basicStatus == KERN_SUCCESS ? bytesToMb(UInt64(basic.resident_size)) : nil,
            virtualMb: basicStatus == KERN_SUCCESS ? bytesToMb(UInt64(basic.virtual_size)) : nil,
            physicalFootprintMb: vmStatus == KERN_SUCCESS ? bytesToMb(UInt64(vm.phys_footprint)) : nil,
            internalMb: vmStatus == KERN_SUCCESS ? bytesToMb(UInt64(vm.internal)) : nil,
            compressedMb: vmStatus == KERN_SUCCESS ? bytesToMb(UInt64(vm.compressed)) : nil
        )
    }

    private static func bytesToMb(_ bytes: UInt64) -> Int64 {
        Int64(bytes / (1024 * 1024))
    }
}

}

extension ZephrAgentRuntime.Embeddings {

public enum TaskType: String, Sendable {
    case query
    case document
    case similarity
}

public struct Embedding: Sendable {
    public let vector: [Float]
    public let dimension: Int
    public let durationMs: Int64
    public let taskType: TaskType

    public init(
        vector: [Float],
        dimension: Int? = nil,
        durationMs: Int64 = 0,
        taskType: TaskType = .query
    ) {
        self.vector = vector
        self.dimension = dimension ?? vector.count
        self.durationMs = durationMs
        self.taskType = taskType
    }
}

}

public extension ZephrAgentRuntime.Diagnostics.StageStats {
    static let zero = ZephrAgentRuntime.Diagnostics.StageStats(
        tokenizeMs: 0,
        prefillMs: 0,
        decodeMs: 0,
        firstDecodeMs: 0,
        inputTokens: 0,
        outputTokens: 0
    )
}

extension ZephrAgentRuntime.Tools {

// MARK: - Vision inputs and outputs

// MARK: - Image conversion errors
public enum ImageConversionError: Error, CustomStringConvertible, Sendable {
    case invalidDimensions
    case bitmapContextCreationFailed
    case pixelBufferUnavailable

    public var description: String {
        switch self {
        case .invalidDimensions:
            return "Invalid image dimensions"
        case .bitmapContextCreationFailed:
            return "Failed to create RGB bitmap context"
        case .pixelBufferUnavailable:
            return "Failed to read rendered image pixels"
        }
    }
}

// MARK: - RGB image input
public struct RgbImage: Sendable {
    public let rgb: Data
    public let width: Int
    public let height: Int
    public let rowStride: Int

    public init(rgb: Data, width: Int, height: Int, rowStride: Int? = nil) throws {
        guard width > 0, height > 0 else {
            throw ImageConversionError.invalidDimensions
        }
        let stride = rowStride ?? width * 3
        guard stride >= width * 3, rgb.count >= stride * height else {
            throw ImageConversionError.invalidDimensions
        }
        self.rgb = rgb
        self.width = width
        self.height = height
        self.rowStride = stride
    }

    public init(cgImage: CGImage, maxDimension: Int? = 1600) throws {
        let originalWidth = cgImage.width
        let originalHeight = cgImage.height
        guard originalWidth > 0, originalHeight > 0 else {
            throw ImageConversionError.invalidDimensions
        }

        let scale: CGFloat
        if let maxDimension, maxDimension > 0 {
            scale = min(1, CGFloat(maxDimension) / CGFloat(max(originalWidth, originalHeight)))
        } else {
            scale = 1
        }
        let width = max(1, Int((CGFloat(originalWidth) * scale).rounded()))
        let height = max(1, Int((CGFloat(originalHeight) * scale).rounded()))
        let rgbaStride = width * 4
        let rgbStride = width * 3

        var rgba = Data(count: rgbaStride * height)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue |
            CGImageAlphaInfo.premultipliedLast.rawValue

        let didRender = rgba.withUnsafeMutableBytes { rgbaBuffer -> Bool in
            guard let baseAddress = rgbaBuffer.baseAddress,
                  let context = CGContext(
                    data: baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: rgbaStride,
                    space: colorSpace,
                    bitmapInfo: bitmapInfo
                  ) else {
                return false
            }
            context.interpolationQuality = .high
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard didRender else {
            throw ImageConversionError.bitmapContextCreationFailed
        }

        var rgb = Data(count: rgbStride * height)
        let didPack = rgba.withUnsafeBytes { rgbaBuffer -> Bool in
            guard let rgbaBase = rgbaBuffer.bindMemory(to: UInt8.self).baseAddress else {
                return false
            }
            return rgb.withUnsafeMutableBytes { rgbBuffer -> Bool in
                guard let rgbBase = rgbBuffer.bindMemory(to: UInt8.self).baseAddress else {
                    return false
                }
                for y in 0..<height {
                    let rgbaRow = rgbaBase + y * rgbaStride
                    let rgbRow = rgbBase + y * rgbStride
                    for x in 0..<width {
                        let rgbaIndex = x * 4
                        let rgbIndex = x * 3
                        rgbRow[rgbIndex + 0] = rgbaRow[rgbaIndex + 0]
                        rgbRow[rgbIndex + 1] = rgbaRow[rgbaIndex + 1]
                        rgbRow[rgbIndex + 2] = rgbaRow[rgbaIndex + 2]
                    }
                }
                return true
            }
        }
        guard didPack else {
            throw ImageConversionError.pixelBufferUnavailable
        }

        try self.init(rgb: rgb, width: width, height: height, rowStride: rgbStride)
    }
}

// MARK: - Vision results
public struct VisionResult: Sendable {
    public let response: String
    public let inputPatches: Int
    public let validVisionTokens: Int
    public let imageTokenSlots: Int
    public let resizedWidth: Int
    public let resizedHeight: Int
    public let promptTokens: Int
    public let decodeSteps: Int
    public let firstDecodeMs: Int64
}

}

extension ZephrAgentRuntime.Diagnostics {

// MARK: - Text generation diagnostics
public struct TextResult: Sendable {
    public let response: String
    public let prompt: String
    public let prefillTokens: Int
    public let stats: StageStats
    public let currentPosition: Int

    public init(
        response: String,
        prompt: String,
        prefillTokens: Int,
        stats: StageStats,
        currentPosition: Int
    ) {
        self.response = response
        self.prompt = prompt
        self.prefillTokens = prefillTokens
        self.stats = stats
        self.currentPosition = currentPosition
    }
}

}

// MARK: - Conversation namespace

extension ZephrAgentRuntime.Conversation {

    @MainActor
    @Observable
    public final class Engine {
        @ObservationIgnored let runtime: ZephrAgentRuntime.Runtime

        /// Current runtime status snapshot; may update frequently during downloads.
        public var lifecycleState: ZephrAgentRuntime.Lifecycle.State {
            runtime.lifecycleState
        }

        /// Current ordered lifecycle milestone list; entries are upserted by stable id.
        public var lifecycleTimeline: [ZephrAgentRuntime.Lifecycle.Event] {
            runtime.lifecycleTimeline
        }

        public var resolvedModels: ZephrAgentRuntime.Lifecycle.ResolvedModels? {
            runtime.resolvedModels
        }

        public var detectedModelFamily: String? {
            runtime.detectedModelFamily
        }

        public init() {
            self.runtime = ZephrAgentRuntime.Runtime()
        }

        init(runtime: ZephrAgentRuntime.Runtime) {
            self.runtime = runtime
        }

        public nonisolated static func interruptedInitialization(
            defaults: UserDefaults = .standard
        ) -> ZephrAgentRuntime.Lifecycle.InterruptedInitialization? {
            ZephrAgentRuntime.Runtime.interruptedInitialization(defaults: defaults)
        }

        public nonisolated static func clearInterruptedInitialization(
            defaults: UserDefaults = .standard
        ) {
            ZephrAgentRuntime.Runtime.clearInterruptedInitialization(defaults: defaults)
        }

        public nonisolated static func enableNativeStderrLogging() {
            ZephrAgentRuntimeCBridge.enableNativeStderrLogging()
        }

        public nonisolated static func deleteDownloadedModelsAndCaches(storage: ZephrAgentRuntime.Lifecycle.ModelStorage) async {
            await ZephrAgentRuntime.ModelManager().deleteDownloadedModelsAndCaches(storage: storage)
        }

        public func initialize(
            progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler? = nil
        ) async throws {
            try await _initialize(progress: progress)
        }

        public func initialize(
            configuration: ZephrAgentRuntime.Lifecycle.Configuration,
            progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler? = nil
        ) async throws {
            try await _initialize(configuration: configuration, progress: progress)
        }

        public func start(
            configuration: ZephrAgentRuntime.Lifecycle.Configuration,
            progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler? = nil
        ) {
            runtime.start(configuration: configuration, progress: progress)
        }

        public func prepare(
            configuration: ZephrAgentRuntime.Lifecycle.Configuration,
            progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler? = nil
        ) async throws {
            try await runtime.prepare(configuration: configuration, progress: progress)
        }

        public func retry() {
            runtime.retry()
        }

        public func cancelActiveWork() {
            runtime.cancelActiveWork()
        }

        public func createConversation(
            config: Config = Config()
        ) -> Conversation {
            Conversation(runtime: runtime, config: config)
        }

        public func collectHeavyJson(
            prompt: String,
            collectActivations: Bool,
            topK: Int
        ) async throws -> String {
            try await runtime._collectHeavyJson(
                prompt: prompt,
                collectActivations: collectActivations,
                topK: topK
            )
        }

        public func describeImage(
            _ image: ZephrAgentRuntime.Tools.RgbImage,
            prompt: String = "Briefly describe this image.",
            maxTokens: Int = 0
        ) async throws -> ZephrAgentRuntime.Tools.VisionResult {
            try await runtime.describeImage(image, prompt: prompt, maxTokens: maxTokens)
        }

        public func embedText(
            _ text: String,
            taskType: ZephrAgentRuntime.Embeddings.TaskType = .query
        ) async throws -> ZephrAgentRuntime.Embeddings.Embedding {
            try await runtime.embedText(text, taskType: taskType)
        }

        public func generateText(
            userMessage: String,
            systemMessage: String = "",
            maxTokens: Int = 256,
            temperature: Float = 0.7,
            topK: Int = 40,
            topP: Float = 0.95
        ) async throws -> ZephrAgentRuntime.Diagnostics.TextResult {
            try await runtime.generateText(
                userMessage: userMessage,
                systemMessage: systemMessage,
                maxTokens: maxTokens,
                temperature: temperature,
                topK: topK,
                topP: topP
            )
        }

        public func deleteDownloadedModelsAndCaches(cancelActiveWork: Bool = true) async {
            await runtime.deleteDownloadedModelsAndCaches(cancelActiveWork: cancelActiveWork)
        }

        public func drainModelTimings() async -> [ZephrAgentRuntime.Diagnostics.ModelTiming] {
            await runtime.drainModelTimings()
        }

        @discardableResult
        public func shutdown() async -> ZephrAgentRuntime.Lifecycle.Event {
            await runtime.shutdown()
        }

        public func close() async {
            await runtime.shutdown()
        }
    }

    public struct Config: Sendable {
        public var systemInstruction: Contents?
        public var tools: [ToolProvider]
        public var conversationStrategy: Strategy
        public var maxTokens: Int
        public var temperature: Float
        public var topK: Int
        public var topP: Float
        public var reserveOutputTokens: Int
        public var maxHistoryTurns: Int
        public var imageResolver: ImageResolver?

        public init(
            systemInstruction: Contents? = nil,
            tools: [ToolProvider] = [],
            conversationStrategy: Strategy = .incrementalKV,
            maxTokens: Int = 512,
            temperature: Float = 0,
            topK: Int = 40,
            topP: Float = 0.95,
            reserveOutputTokens: Int = 0,
            maxHistoryTurns: Int = 6,
            imageResolver: ImageResolver? = nil
        ) {
            self.systemInstruction = systemInstruction
            self.tools = tools
            self.conversationStrategy = conversationStrategy
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.topK = topK
            self.topP = topP
            self.reserveOutputTokens = reserveOutputTokens
            self.maxHistoryTurns = maxHistoryTurns
            self.imageResolver = imageResolver
        }

        var toolResponseContinuationMode: ToolResponseContinuationMode {
            conversationStrategy.toolResponseContinuationMode ?? .incrementalKV
        }
    }

    public enum Strategy: Equatable, Sendable {
        case incrementalKV
        case fullReplay

        public var rawValue: String {
            switch self {
            case .incrementalKV:
                "incremental_kv"
            case .fullReplay:
                "full_replay"
            }
        }

        public init?(rawValue: String) {
            switch rawValue {
            case "incremental_kv":
                self = .incrementalKV
            case "full_replay":
                self = .fullReplay
            default:
                return nil
            }
        }

        var toolResponseContinuationMode: ToolResponseContinuationMode? {
            switch self {
            case .incrementalKV:
                .incrementalKV
            case .fullReplay:
                .replayPrompt
            }
        }
    }

    enum ToolResponseContinuationMode: String, Sendable {
        case incrementalKV = "incremental_kv"
        case replayPrompt = "full_replay"

        var conversationStrategy: Strategy {
            switch self {
            case .incrementalKV:
                .incrementalKV
            case .replayPrompt:
                .fullReplay
            }
        }
    }

    public struct Context: Sendable {
        public var promptPrefix: String?

        public init(
            promptPrefix: String? = nil
        ) {
            self.promptPrefix = promptPrefix
        }
    }

    public struct SendOptions: Sendable {
        public var conversationStrategy: Strategy?

        public init(conversationStrategy: Strategy? = nil) {
            self.conversationStrategy = conversationStrategy
        }
    }

    public struct ToolCallRecord: Sendable {
        public var turnIndex: Int
        public var name: String
        public var arguments: [String: String]
    }

    public struct ToolResponseRecord: Sendable {
        public var turnIndex: Int
        public var name: String
        public var response: String
    }

    public struct GenerationStartRecord: Sendable {
        public var turnIndex: Int
        public var label: String
        public var kind: String
        public var prompt: String
        public var usedLiveKvCache: Bool
        public var conversationStrategy: Strategy

        public init(
            turnIndex: Int,
            label: String,
            kind: String,
            prompt: String,
            usedLiveKvCache: Bool,
            conversationStrategy: Strategy
        ) {
            self.turnIndex = turnIndex
            self.label = label
            self.kind = kind
            self.prompt = prompt
            self.usedLiveKvCache = usedLiveKvCache
            self.conversationStrategy = conversationStrategy
        }
    }

    public struct GenerationResultRecord: Sendable {
        public var turnIndex: Int
        public var label: String
        public var kind: String
        public var result: ZephrAgentRuntime.Diagnostics.TextResult
        public var usedLiveKvCache: Bool
        public var conversationStrategy: Strategy

        public init(
            turnIndex: Int,
            label: String,
            kind: String,
            result: ZephrAgentRuntime.Diagnostics.TextResult,
            usedLiveKvCache: Bool,
            conversationStrategy: Strategy
        ) {
            self.turnIndex = turnIndex
            self.label = label
            self.kind = kind
            self.result = result
            self.usedLiveKvCache = usedLiveKvCache
            self.conversationStrategy = conversationStrategy
        }
    }

    public struct GenerationFailureRecord: Sendable {
        public var turnIndex: Int
        public var label: String
        public var kind: String
        public var reason: String
        public var usedLiveKvCache: Bool
        public var conversationStrategy: Strategy

        public init(
            turnIndex: Int,
            label: String,
            kind: String,
            reason: String,
            usedLiveKvCache: Bool,
            conversationStrategy: Strategy
        ) {
            self.turnIndex = turnIndex
            self.label = label
            self.kind = kind
            self.reason = reason
            self.usedLiveKvCache = usedLiveKvCache
            self.conversationStrategy = conversationStrategy
        }
    }

    public struct IncrementalFallbackRecord: Sendable {
        public var turnIndex: Int
        public var label: String
        public var reason: String

        public init(turnIndex: Int, label: String, reason: String) {
            self.turnIndex = turnIndex
            self.label = label
            self.reason = reason
        }
    }

    public struct MemorySampleRecord: Sendable {
        public var turnIndex: Int?
        public var label: String
        public var sample: ZephrAgentRuntime.Diagnostics.MemorySample

        public init(
            turnIndex: Int? = nil,
            label: String,
            sample: ZephrAgentRuntime.Diagnostics.MemorySample
        ) {
            self.turnIndex = turnIndex
            self.label = label
            self.sample = sample
        }
    }

    public struct ToolEffect: Sendable {
        public var name: String
        public var payload: [String: String]

        public init(name: String, payload: [String: String] = [:]) {
            self.name = name
            self.payload = payload
        }
    }

    public struct GenerationRecord: Sendable {
        public var turnIndex: Int
        public var label: String
        public var kind: String
        public var prompt: String
        public var response: String
        public var stats: ZephrAgentRuntime.Diagnostics.StageStats
        public var currentPosition: Int
        public var usedLiveKvCache: Bool
        public var conversationStrategy: Strategy

        public init(
            turnIndex: Int,
            label: String,
            kind: String,
            prompt: String,
            response: String,
            stats: ZephrAgentRuntime.Diagnostics.StageStats,
            currentPosition: Int,
            usedLiveKvCache: Bool,
            conversationStrategy: Strategy
        ) {
            self.turnIndex = turnIndex
            self.label = label
            self.kind = kind
            self.prompt = prompt
            self.response = response
            self.stats = stats
            self.currentPosition = currentPosition
            self.usedLiveKvCache = usedLiveKvCache
            self.conversationStrategy = conversationStrategy
        }
    }

    public struct TurnRecord: Sendable {
        public var conversationTurnIndex: Int
        public var conversationStrategy: Strategy
        public var userContent: Contents
        public var userText: String
        public var finalText: String
        public var generationStats: ZephrAgentRuntime.Diagnostics.GenerationStats?
        public var generations: [GenerationRecord]
        public var toolCalls: [ToolCallRecord]
        public var toolResponses: [ToolResponseRecord]
        public var effects: [ToolEffect]
        public var error: String?

        public init(
            conversationTurnIndex: Int,
            conversationStrategy: Strategy,
            userContent: Contents? = nil,
            userText: String,
            finalText: String,
            generationStats: ZephrAgentRuntime.Diagnostics.GenerationStats? = nil,
            generations: [GenerationRecord] = [],
            toolCalls: [ToolCallRecord] = [],
            toolResponses: [ToolResponseRecord] = [],
            effects: [ToolEffect] = [],
            error: String? = nil
        ) {
            self.conversationTurnIndex = conversationTurnIndex
            self.conversationStrategy = conversationStrategy
            self.userContent = userContent ?? .of(userText)
            self.userText = userText
            self.finalText = finalText
            self.generationStats = generationStats
            self.generations = generations
            self.toolCalls = toolCalls
            self.toolResponses = toolResponses
            self.effects = effects
            self.error = error
        }
    }

    public enum Event: Sendable {
        case assistantDelta(String)
        case generationStart(GenerationStartRecord)
        case generationResult(GenerationResultRecord)
        case generationFailure(GenerationFailureRecord)
        case incrementalFallback(IncrementalFallbackRecord)
        case memorySample(MemorySampleRecord)
        case toolCall(ToolCallRecord)
        case toolResponse(ToolResponseRecord)
        case effectApplied(ToolEffect)
        case turnCompleted(TurnRecord)
    }

    @MainActor
    public final class Conversation {
        @ObservationIgnored let runtime: ZephrAgentRuntime.Runtime
        let config: Config
        var turns: [ConversationProtocolTurn] = []
        var protocolTurns: [(role: String, body: String)] = []
        var conversationTurnIndex = 0
        var closed = false
        var cancelled = false
        var liveKvCacheReady = false

        init(runtime: ZephrAgentRuntime.Runtime, config: Config) {
            self.runtime = runtime
            self.config = config
        }

        public func sendMessageAsync(
            _ contents: Contents,
            options: SendOptions = SendOptions()
        ) -> AsyncThrowingStream<Message, Error> {
            _sendMessageAsync(contents, options: options)
        }

        public func sendMessage(
            _ contents: Contents,
            context: Context = Context(),
            options: SendOptions = SendOptions()
        ) async throws -> Message {
            try await _sendMessage(contents, context: context, options: options)
        }

        public func sendEvents(
            _ contents: Contents,
            context: Context = Context(),
            options: SendOptions = SendOptions()
        ) -> AsyncThrowingStream<Event, Error> {
            _sendEvents(contents, context: context, options: options)
        }

        public func cancelProcess() {
            cancelled = true
        }

        func setToolResponseContinuationMode(_ mode: ToolResponseContinuationMode) {
            liveKvCacheReady = false
        }

        public func close() {
            closed = true
        }
    }

    public struct Message: Sendable {
        public var contents: Contents
        public var generationStats: ZephrAgentRuntime.Diagnostics.GenerationStats?
        public var isFinal: Bool

        public init(
            contents: Contents,
            generationStats: ZephrAgentRuntime.Diagnostics.GenerationStats? = nil,
            isFinal: Bool = false
        ) {
            self.contents = contents
            self.generationStats = generationStats
            self.isFinal = isFinal
        }
    }

    public struct ContentPart: Equatable, Sendable {
        public enum Kind: String, Sendable {
            case text
            case imageRef
            case imageFile
        }

        public var kind: Kind
        public var text: String?
        public var imageRef: String?
        public var imageFile: String?

        public init(kind: Kind, text: String? = nil, imageRef: String? = nil, imageFile: String? = nil) {
            self.kind = kind
            self.text = text
            self.imageRef = imageRef
            self.imageFile = imageFile
        }

        public static func text(_ text: String) -> ContentPart {
            ContentPart(kind: .text, text: text)
        }

        public static func imageRef(_ ref: String) -> ContentPart {
            ContentPart(kind: .imageRef, imageRef: ref)
        }

        public static func imageFile(_ path: String) -> ContentPart {
            ContentPart(kind: .imageFile, imageFile: path)
        }
    }

    public struct Contents: CustomStringConvertible, Sendable {
        public let parts: [ContentPart]

        public var description: String {
            parts.map { part in
                switch part.kind {
                case .text:
                    return part.text ?? ""
                case .imageRef:
                    return "[imageRef:\(part.imageRef ?? "")]"
                case .imageFile:
                    return "[imageFile:\(part.imageFile ?? "")]"
                }
            }.joined(separator: "\n")
        }

        public var text: String {
            parts.compactMap { $0.kind == .text ? $0.text : nil }.joined(separator: "\n")
        }

        public var imageParts: [ContentPart] {
            parts.filter { $0.kind == .imageRef || $0.kind == .imageFile }
        }

        public var hasImageContent: Bool {
            !imageParts.isEmpty
        }

        public init(parts: [ContentPart]) {
            self.parts = parts
        }

        public static func of(_ text: String) -> Contents {
            Contents(parts: [.text(text)])
        }

        public static func of(_ parts: [ContentPart]) -> Contents {
            Contents(parts: parts)
        }
    }

    public typealias ImageResolver = @Sendable (ContentPart) async throws -> ZephrAgentRuntime.Tools.RgbImage?

    public protocol ToolParameterValue: Codable {
        static func jsonSchema() -> [String: Any]
    }

    public protocol Tool: Decodable, Sendable {
        static var name: String { get }
        static var description: String { get }
        init()
        func run() async throws -> Any
    }

    @propertyWrapper
    public struct ToolParam<Value: ToolParameterValue>: Decodable, ToolParamProtocol {
        private var storage: Value?
        private let hasDefaultValue: Bool
        public let description: String

        public init(wrappedValue: Value? = nil, description: String) {
            self.storage = wrappedValue
            self.hasDefaultValue = wrappedValue != nil
            self.description = description
        }

        public var wrappedValue: Value {
            get {
                if let storage {
                    return storage
                }
                if let nilValue = Optional<Any>.none as? Value {
                    return nilValue
                }
                preconditionFailure("ToolParam of type \(Value.self) was not set and has no default value.")
            }
            set {
                storage = newValue
            }
        }

        var wrappedValueAny: Any? {
            storage
        }

        var isRequired: Bool {
            let isOptional = Optional<Any>.none as? Value != nil
            return !isOptional && !hasDefaultValue
        }

        func jsonSchema() -> [String: Any] {
            var schema = Value.jsonSchema()
            if !description.isEmpty {
                schema["description"] = description
            }
            return schema
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if container.decodeNil() {
                if let nilValue = Optional<Any>.none as? Value {
                    storage = nilValue
                } else {
                    throw DecodingError.valueNotFound(
                        Value.self,
                        DecodingError.Context(
                            codingPath: decoder.codingPath,
                            debugDescription: "Received null for non-optional tool parameter."
                        )
                    )
                }
            } else if let value = try? container.decode(Value.self) {
                storage = value
            } else if let text = try? container.decode(String.self),
                      let value = Self.coerceString(text) {
                storage = value
            } else {
                storage = try container.decode(Value.self)
            }
            hasDefaultValue = true
            description = ""
        }

        private static func coerceString(_ text: String) -> Value? {
            if Value.self == String.self {
                return text as? Value
            }
            if Value.self == Int.self {
                return Int(text) as? Value
            }
            if Value.self == Int32.self {
                return Int32(text) as? Value
            }
            if Value.self == Int64.self {
                return Int64(text) as? Value
            }
            if Value.self == Float.self {
                return Float(text) as? Value
            }
            if Value.self == Double.self {
                return Double(text) as? Value
            }
            if Value.self == Bool.self {
                switch text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
                case "true", "1", "yes":
                    return true as? Value
                case "false", "0", "no":
                    return false as? Value
                default:
                    return nil
                }
            }
            return nil
        }
    }

    public protocol OpenApiTool: Sendable {
        func getToolDescriptionJsonString() -> String
        func execute(params: String) async -> String
        func appliedEffects(params: String, response: String) async -> [ToolEffect]
    }

    public struct ToolProvider: Sendable {
        let tools: @Sendable () -> [ConversationInternalTool]

        init(tools: @escaping @Sendable () -> [ConversationInternalTool]) {
            self.tools = tools
        }
    }

    public static func tool(_ openApiTool: any OpenApiTool) -> ToolProvider {
        ToolProvider {
            ConversationToolFactory.tools(from: openApiTool)
        }
    }

    public static func tool<T: Tool>(_ tool: T) -> ToolProvider {
        ToolProvider {
            ConversationToolFactory.tools(from: ReflectedOpenApiTool(tool))
        }
    }

    public static let zephrPointActionToolName = "zephr_point_action"
}

extension String: ZephrAgentRuntime.Conversation.ToolParameterValue {
    public static func jsonSchema() -> [String: Any] {
        ["type": "string"]
    }
}

extension Int: ZephrAgentRuntime.Conversation.ToolParameterValue {
    public static func jsonSchema() -> [String: Any] {
        ["type": "integer"]
    }
}

extension Int32: ZephrAgentRuntime.Conversation.ToolParameterValue {
    public static func jsonSchema() -> [String: Any] {
        ["type": "integer"]
    }
}

extension Int64: ZephrAgentRuntime.Conversation.ToolParameterValue {
    public static func jsonSchema() -> [String: Any] {
        ["type": "integer"]
    }
}

extension Bool: ZephrAgentRuntime.Conversation.ToolParameterValue {
    public static func jsonSchema() -> [String: Any] {
        ["type": "boolean"]
    }
}

extension Float: ZephrAgentRuntime.Conversation.ToolParameterValue {
    public static func jsonSchema() -> [String: Any] {
        ["type": "number"]
    }
}

extension Double: ZephrAgentRuntime.Conversation.ToolParameterValue {
    public static func jsonSchema() -> [String: Any] {
        ["type": "number"]
    }
}

extension Optional: ZephrAgentRuntime.Conversation.ToolParameterValue where Wrapped: ZephrAgentRuntime.Conversation.ToolParameterValue {
    public static func jsonSchema() -> [String: Any] {
        var schema = Wrapped.jsonSchema()
        schema["nullable"] = true
        return schema
    }
}

extension Array: ZephrAgentRuntime.Conversation.ToolParameterValue where Element: ZephrAgentRuntime.Conversation.ToolParameterValue {
    public static func jsonSchema() -> [String: Any] {
        ["type": "array", "items": Element.jsonSchema()]
    }
}

public extension ZephrAgentRuntime.Conversation.OpenApiTool {
    func appliedEffects(params: String, response: String) async -> [ZephrAgentRuntime.Conversation.ToolEffect] {
        []
    }
}

public extension ZephrAgentRuntime.Conversation {
    struct MCPConfiguration: Sendable {
        public var endpoint: URL
        public var apiKey: String
        public var clientName: String
        public var clientVersion: String

        public init(
            endpoint: URL,
            apiKey: String,
            clientName: String = "zephr-agent-runtime",
            clientVersion: String = "1.0.0"
        ) {
            self.endpoint = endpoint
            self.apiKey = apiKey
            self.clientName = clientName
            self.clientVersion = clientVersion
        }
    }

    struct MCPToolSet: Sendable {
        public var providers: [ToolProvider]
        public var toolNames: [String]

        public init(providers: [ToolProvider], toolNames: [String]) {
            self.providers = providers
            self.toolNames = toolNames
        }
    }
}

public extension ZephrAgentRuntime.Conversation.Engine {
    func mcpTools(
        configuration: ZephrAgentRuntime.Conversation.MCPConfiguration
    ) async throws -> ZephrAgentRuntime.Conversation.MCPToolSet {
        try await _mcpTools(configuration: configuration)
    }

}

// MARK: - Lifecycle namespace

extension ZephrAgentRuntime.Lifecycle {

public enum ModelFamily: String, CaseIterable, Identifiable, Codable, Sendable {
    case gemma4 = "gemma4"
    case gemma3Embedding = "gemma3_embedding"

    public var id: String { rawValue }

    public var displayLabel: String {
        switch self {
        case .gemma4:
            "GEMMA 4"
        case .gemma3Embedding:
            "GEMMA 3 EMBEDDING"
        }
    }
}

// MARK: - Model channels
public enum ModelChannel: String, CaseIterable, Identifiable, Codable, Sendable {
    case `public`
    case ben
    case local

    public var id: String { rawValue }
}

// MARK: - Execution delegates
public struct ExecutionDelegates: Equatable, Sendable {
    public var llm: String
    public var rag: String
    public var vlm: String
    public var gemma4Runtime: Gemma4Options
    public var diagnosticGemma4: ZephrAgentRuntime.Diagnostics.Gemma4Options

    public init(
        llm: String = "cpu",
        rag: String = "cpu",
        vlm: String = "gpu",
        gemma4Runtime: Gemma4Options = .automatic,
        diagnosticGemma4: ZephrAgentRuntime.Diagnostics.Gemma4Options = .disabled
    ) {
        self.llm = llm
        self.rag = rag
        self.vlm = vlm
        self.gemma4Runtime = gemma4Runtime
        self.diagnosticGemma4 = diagnosticGemma4
    }
}

// MARK: - Gemma 4 runtime options
public enum Gemma4GpuPrecision: Int32, Equatable, Sendable {
    case automatic = -1
    case precision0 = 0
    case precision1 = 1
    case precision2 = 2
}

public enum RuntimeToggle: Int32, Equatable, Sendable {
    case automatic = -1
    case disabled = 0
    case enabled = 1
}

public struct Gemma4Options: Equatable, Sendable {
    public static let automatic = Gemma4Options()

    public var gpuPrecision: Gemma4GpuPrecision
    public var kvCacheMaxTokens: Int32
    public var constrainedVerifyBatch: RuntimeToggle
    public var mtpEnabled: Bool
    public var mtpTrustVerifyKV: Bool
    public var mtpAdaptiveEnabled: Bool
    public var mtpAdaptiveMinCycles: Int32
    public var mtpAdaptiveMinSavedPerCycle: Float
    public var mtpTrace: Bool

    public init(
        gpuPrecision: Gemma4GpuPrecision = .automatic,
        kvCacheMaxTokens: Int32 = 0,
        constrainedVerifyBatch: RuntimeToggle = .automatic,
        mtpEnabled: Bool = false,
        mtpTrustVerifyKV: Bool = true,
        mtpAdaptiveEnabled: Bool = true,
        mtpAdaptiveMinCycles: Int32 = 4,
        mtpAdaptiveMinSavedPerCycle: Float = 0.5,
        mtpTrace: Bool = false
    ) {
        self.gpuPrecision = gpuPrecision
        self.kvCacheMaxTokens = kvCacheMaxTokens
        self.constrainedVerifyBatch = constrainedVerifyBatch
        self.mtpEnabled = mtpEnabled
        self.mtpTrustVerifyKV = mtpTrustVerifyKV
        self.mtpAdaptiveEnabled = mtpAdaptiveEnabled
        self.mtpAdaptiveMinCycles = mtpAdaptiveMinCycles
        self.mtpAdaptiveMinSavedPerCycle = mtpAdaptiveMinSavedPerCycle
        self.mtpTrace = mtpTrace
    }
}

}

extension ZephrAgentRuntime.Diagnostics {

public struct Gemma4Options: Equatable, Sendable {
    public static let disabled = Gemma4Options()

    public var prefillByDecode: Bool
    public var prefillMaxChunk: Int32
    public var constrainedVerifyTrace: Bool
    public var constrainedVerifyMaxAccept: Int32

    public init(
        prefillByDecode: Bool = false,
        prefillMaxChunk: Int32 = 0,
        constrainedVerifyTrace: Bool = false,
        constrainedVerifyMaxAccept: Int32 = 0
    ) {
        self.prefillByDecode = prefillByDecode
        self.prefillMaxChunk = prefillMaxChunk
        self.constrainedVerifyTrace = constrainedVerifyTrace
        self.constrainedVerifyMaxAccept = constrainedVerifyMaxAccept
    }
}

}

extension ZephrAgentRuntime.Lifecycle {

// MARK: - Model storage
public enum ModelStorage: Equatable, Sendable {
    case appGroup(String)
    case directory(URL)
}

}

extension ZephrAgentRuntime.Lifecycle {

// MARK: - Runtime settings
public struct RuntimeSettings: Equatable, Sendable {
    public var modelChannel: ModelChannel
    public var llmExecutionChoiceID: String
    public var ragEmbeddingExecutionChoiceID: String
    public var vlmExecutionChoiceID: String
    public var crashDuringInitializationForTesting: Bool
    public var litertCompilationCacheEnabled: Bool
    public var gemma4Runtime: Gemma4Options
    public var diagnosticGemma4: ZephrAgentRuntime.Diagnostics.Gemma4Options
    public var numThreads: Int
    public var modelStorage: ModelStorage?

    public init(
        modelChannel: ModelChannel = .public,
        llmExecutionChoiceID: String = "",
        ragEmbeddingExecutionChoiceID: String = "",
        vlmExecutionChoiceID: String = "",
        crashDuringInitializationForTesting: Bool = false,
        litertCompilationCacheEnabled: Bool = false,
        gemma4Runtime: Gemma4Options = .automatic,
        diagnosticGemma4: ZephrAgentRuntime.Diagnostics.Gemma4Options = .disabled,
        numThreads: Int = 0,
        modelStorage: ModelStorage? = nil
    ) {
        self.modelChannel = modelChannel
        self.llmExecutionChoiceID = llmExecutionChoiceID
        self.ragEmbeddingExecutionChoiceID = ragEmbeddingExecutionChoiceID
        self.vlmExecutionChoiceID = vlmExecutionChoiceID
        self.crashDuringInitializationForTesting = crashDuringInitializationForTesting
        self.litertCompilationCacheEnabled = litertCompilationCacheEnabled
        self.gemma4Runtime = gemma4Runtime
        self.diagnosticGemma4 = diagnosticGemma4
        self.numThreads = numThreads
        self.modelStorage = modelStorage
    }

    public var configuration: Configuration {
        Configuration(settings: self)
    }

    public var effectiveHardwareSummary: String {
        configuration.effectiveHardwareSummary
    }

    public var identityKey: String {
        [
            modelChannel.rawValue,
            llmExecutionChoiceID,
            ragEmbeddingExecutionChoiceID,
            vlmExecutionChoiceID,
            crashDuringInitializationForTesting ? "init_crash_test" : "no_init_crash_test",
            litertCompilationCacheEnabled ? "litert_cache" : "no_litert_cache",
            gemma4Runtime.mtpEnabled ? "gemma4_mtp" : "no_gemma4_mtp",
            "gemma4_precision_\(gemma4Runtime.gpuPrecision.rawValue)",
            "gemma4_kv_\(gemma4Runtime.kvCacheMaxTokens)",
            "gemma4_verify_batch_\(gemma4Runtime.constrainedVerifyBatch.rawValue)",
            diagnosticGemma4.prefillByDecode ? "diag_prefill_by_decode" : "no_diag_prefill_by_decode",
            "diag_prefill_chunk_\(diagnosticGemma4.prefillMaxChunk)",
            diagnosticGemma4.constrainedVerifyTrace ? "diag_verify_trace" : "no_diag_verify_trace",
            "diag_verify_accept_\(diagnosticGemma4.constrainedVerifyMaxAccept)",
            "\(numThreads)",
            modelStorage?.identityKey ?? "no_local_storage"
        ].joined(separator: ":")
    }
}

// MARK: - Engine configuration
public struct Configuration: Sendable {
    public var modelChannel: ModelChannel
    public var llmExecutionChoiceID: String
    public var ragEmbeddingExecutionChoiceID: String
    public var vlmExecutionChoiceID: String
    public var crashDuringInitializationForTesting: Bool
    public var litertCompilationCacheEnabled: Bool
    public var gemma4Runtime: Gemma4Options
    public var diagnosticGemma4: ZephrAgentRuntime.Diagnostics.Gemma4Options
    public var numThreads: Int
    public var modelStorage: ModelStorage?

    public init(
        modelChannel: ModelChannel = .public,
        llmExecutionChoiceID: String = "",
        ragEmbeddingExecutionChoiceID: String = "",
        vlmExecutionChoiceID: String = "",
        crashDuringInitializationForTesting: Bool = false,
        litertCompilationCacheEnabled: Bool = false,
        gemma4Runtime: Gemma4Options = .automatic,
        diagnosticGemma4: ZephrAgentRuntime.Diagnostics.Gemma4Options = .disabled,
        numThreads: Int = 0,
        modelStorage: ModelStorage? = nil
    ) {
        self.modelChannel = modelChannel
        self.llmExecutionChoiceID = llmExecutionChoiceID
        self.ragEmbeddingExecutionChoiceID = ragEmbeddingExecutionChoiceID
        self.vlmExecutionChoiceID = vlmExecutionChoiceID
        self.crashDuringInitializationForTesting = crashDuringInitializationForTesting
        self.litertCompilationCacheEnabled = litertCompilationCacheEnabled
        self.gemma4Runtime = gemma4Runtime
        self.diagnosticGemma4 = diagnosticGemma4
        self.numThreads = numThreads
        self.modelStorage = modelStorage
    }

    public init(settings: RuntimeSettings) {
        self.init(
            modelChannel: settings.modelChannel,
            llmExecutionChoiceID: settings.llmExecutionChoiceID,
            ragEmbeddingExecutionChoiceID: settings.ragEmbeddingExecutionChoiceID,
            vlmExecutionChoiceID: settings.vlmExecutionChoiceID,
            crashDuringInitializationForTesting: settings.crashDuringInitializationForTesting,
            litertCompilationCacheEnabled: settings.litertCompilationCacheEnabled,
            gemma4Runtime: settings.gemma4Runtime,
            diagnosticGemma4: settings.diagnosticGemma4,
            numThreads: settings.numThreads,
            modelStorage: settings.modelStorage
        )
    }

    public var effectiveHardwareSummary: String {
        var parts = [
            "llm:\(llmExecutionChoiceID.isEmpty ? "default" : llmExecutionChoiceID)",
            "rag:\(ragEmbeddingExecutionChoiceID.isEmpty ? "default" : ragEmbeddingExecutionChoiceID)",
            "vlm:\(vlmExecutionChoiceID.isEmpty ? "default" : vlmExecutionChoiceID)",
            "mtp:\(gemma4Runtime.mtpEnabled ? "on" : "off")"
        ]
        if gemma4Runtime.gpuPrecision != .automatic {
            parts.append("gemma4_precision:\(gemma4Runtime.gpuPrecision.rawValue)")
        }
        if diagnosticGemma4 != .disabled {
            parts.append("diagnostic_gemma4:on")
        }
        return parts.joined(separator: " ")
    }
}

// MARK: - Model catalog and execution choices

public struct ModelCatalog: Equatable, Sendable {
    public var gemma4LLMVariants: [ModelArtifact]
    public var ragEmbedding: ModelArtifact?
    public var modelArtifacts: [ModelArtifact]
    public var roleChoices: ExecutionChoices

    public init(
        gemma4LLMVariants: [ModelArtifact],
        ragEmbedding: ModelArtifact? = nil,
        modelArtifacts: [ModelArtifact]? = nil,
        roleChoices: ExecutionChoices = ExecutionChoices()
    ) {
        self.gemma4LLMVariants = gemma4LLMVariants
        self.ragEmbedding = ragEmbedding
        self.modelArtifacts = modelArtifacts ?? (gemma4LLMVariants + [ragEmbedding].compactMap { $0 })
        self.roleChoices = roleChoices
    }
}

public struct ExecutionChoice: Equatable, Identifiable, Sendable {
    public var id: String
    public var label: String
    public var family: ModelFamily?
    public var executionPlan: String
    public var requestedPlan: String
    public var artifactID: String
    public var components: [ExecutionPlanComponent]

    public var isOff: Bool {
        id == "off" || requestedPlan == "off" || executionPlan == "off"
    }

    public init(
        id: String,
        label: String,
        family: ModelFamily?,
        executionPlan: String,
        artifactID: String,
        requestedPlan: String? = nil,
        components: [ExecutionPlanComponent] = []
    ) {
        self.id = id
        self.label = label
        self.family = family
        self.executionPlan = executionPlan
        self.requestedPlan = requestedPlan ?? executionPlan
        self.artifactID = artifactID
        self.components = components
    }
}

public struct ExecutionPlanComponent: Equatable, Identifiable, Sendable {
    public enum Role: String, Sendable {
        case text
        case embedding
        case vision
    }

    public var id: String
    public var role: Role
    public var artifactID: String
    public var family: String?
    public var target: String
    public var requestedTarget: String?
    public var reason: String?
    public var signature: String?

    public init(
        id: String,
        role: Role,
        artifactID: String,
        family: String? = nil,
        target: String,
        requestedTarget: String? = nil,
        reason: String? = nil,
        signature: String? = nil
    ) {
        self.id = id
        self.role = role
        self.artifactID = artifactID
        self.family = family
        self.target = target
        self.requestedTarget = requestedTarget
        self.reason = reason
        self.signature = signature
    }

    public var displayLine: String {
        let requested = requestedTarget.map { " requested=\($0)" } ?? ""
        let signature = signature.map { " sig=\($0)" } ?? ""
        return "\(id) -> \(target)\(requested)\(signature)".uppercased()
    }
}

public struct ResolvedExecutionPlan: Equatable, Sendable {
    public var choices: ExecutionSelection
    public var components: [ExecutionPlanComponent]

    public init(
        choices: ExecutionSelection,
        components: [ExecutionPlanComponent]
    ) {
        self.choices = choices
        self.components = components
    }
}

public struct ExecutionChoices: Equatable, Sendable {
    public var llm: [ExecutionChoice]
    public var ragEmbedding: [ExecutionChoice]
    public var vlm: [ExecutionChoice]

    public init(
        llm: [ExecutionChoice] = [],
        ragEmbedding: [ExecutionChoice] = [],
        vlm: [ExecutionChoice] = []
    ) {
        self.llm = llm
        self.ragEmbedding = ragEmbedding
        self.vlm = vlm
    }
}

}

public extension ZephrAgentRuntime.Lifecycle.ExecutionChoices {
    static func uniqueFamilies(in choices: [ZephrAgentRuntime.Lifecycle.ExecutionChoice]) -> [ZephrAgentRuntime.Lifecycle.ModelFamily] {
        choices.compactMap(\.family).reduce(into: []) { families, family in
            if !families.contains(family) {
                families.append(family)
            }
        }
    }

    func selectedLLM(_ requestedID: String) -> ZephrAgentRuntime.Lifecycle.ExecutionChoice? {
        selectedChoice(llm, requestedID: requestedID)
    }

    func selectedRAGEmbedding(_ requestedID: String) -> ZephrAgentRuntime.Lifecycle.ExecutionChoice? {
        selectedChoice(ragEmbedding, requestedID: requestedID)
    }

    func selectedVLM(_ requestedID: String) -> ZephrAgentRuntime.Lifecycle.ExecutionChoice? {
        selectedChoice(vlm, requestedID: requestedID)
    }

    func normalized(_ settings: ZephrAgentRuntime.Lifecycle.RuntimeSettings) -> ZephrAgentRuntime.Lifecycle.RuntimeSettings {
        guard !llm.isEmpty else { return settings }
        return ZephrAgentRuntime.Lifecycle.RuntimeSettings(
            modelChannel: settings.modelChannel,
            llmExecutionChoiceID: selectedLLM(settings.llmExecutionChoiceID)?.id ?? "",
            ragEmbeddingExecutionChoiceID: selectedRAGEmbedding(settings.ragEmbeddingExecutionChoiceID)?.id ?? "",
            vlmExecutionChoiceID: selectedVLM(settings.vlmExecutionChoiceID)?.id ?? "",
            crashDuringInitializationForTesting: settings.crashDuringInitializationForTesting,
            litertCompilationCacheEnabled: settings.litertCompilationCacheEnabled,
            gemma4Runtime: settings.gemma4Runtime,
            diagnosticGemma4: settings.diagnosticGemma4,
            numThreads: settings.numThreads,
            modelStorage: settings.modelStorage
        )
    }

    private func selectedChoice(
        _ choices: [ZephrAgentRuntime.Lifecycle.ExecutionChoice],
        requestedID: String
    ) -> ZephrAgentRuntime.Lifecycle.ExecutionChoice? {
        if !requestedID.isEmpty,
           let choice = choices.first(where: { $0.id == requestedID }) {
            return choice
        }
        return choices.first
    }
}

extension ZephrAgentRuntime.Lifecycle {

public struct ExecutionSelection: Equatable, Sendable {
    public var llm: ExecutionComponent
    public var rag: ExecutionComponent?
    public var vlm: ExecutionComponent?

    public init(
        llm: ExecutionComponent,
        rag: ExecutionComponent? = nil,
        vlm: ExecutionComponent? = nil
    ) {
        self.llm = llm
        self.rag = rag
        self.vlm = vlm
    }
}

}

public extension ZephrAgentRuntime.Lifecycle.ModelCatalog {
    static func availableExecutionChoices(
        channel: ZephrAgentRuntime.Lifecycle.ModelChannel,
        storage: ZephrAgentRuntime.Lifecycle.ModelStorage?,
        fileManager: FileManager = .default
    ) throws -> ZephrAgentRuntime.Lifecycle.ExecutionChoices {
        try _availableExecutionChoices(
            channel: channel,
            storage: storage,
            fileManager: fileManager
        )
    }

    static func resolvedExecutionPlan(
        configuration: ZephrAgentRuntime.Lifecycle.Configuration,
        fileManager: FileManager = .default
    ) throws -> ZephrAgentRuntime.Lifecycle.ResolvedExecutionPlan {
        try _resolvedExecutionPlan(
            configuration: configuration,
            fileManager: fileManager
        )
    }
}

extension ZephrAgentRuntime.Lifecycle {

// MARK: - Execution selection
public struct ExecutionComponent: Equatable, Sendable {
    public var choiceID: String
    public var artifact: String
    public var executionPlan: String
    public var requestedPlan: String
    public var components: [ExecutionPlanComponent]

    public init(
        choiceID: String,
        artifact: String,
        executionPlan: String,
        requestedPlan: String? = nil,
        components: [ExecutionPlanComponent] = []
    ) {
        self.choiceID = choiceID
        self.artifact = artifact
        self.executionPlan = executionPlan
        self.requestedPlan = requestedPlan ?? executionPlan
        self.components = components
    }

    public init(choice: ExecutionChoice) {
        self.init(
            choiceID: choice.id,
            artifact: choice.artifactID,
            executionPlan: choice.executionPlan,
            requestedPlan: choice.requestedPlan,
            components: choice.components
        )
    }
}

// MARK: - Model artifacts
public struct ModelArtifact: Equatable, Identifiable, Sendable {
    public enum Capability: String, Sendable {
        case text
        case vlm
    }

    public enum Role: String, Sendable {
        case text
        case embedding
        case vlm
    }

    public var id: String
    public var role: Role
    public var title: String
    public var family: ModelFamily?
    public var capabilities: [Capability]
    public var filename: String
    public var version: String
    public var quantization: String?
    public var kvCacheMaxLen: Int?
    public var hardwareTarget: String?
    public var executionPlan: String
    public var downloadURL: URL?
    public var sizeBytes: Int64?
    public var sha256: String?

    public init(
        id: String,
        role: Role,
        title: String,
        family: ModelFamily? = nil,
        capabilities: [Capability]? = nil,
        filename: String,
        version: String,
        quantization: String? = nil,
        kvCacheMaxLen: Int? = nil,
        hardwareTarget: String? = nil,
        executionPlan: String = "",
        downloadURL: URL? = nil,
        sizeBytes: Int64? = nil,
        sha256: String? = nil
    ) {
        self.id = id
        self.role = role
        self.title = title
        self.family = family
        self.capabilities = capabilities ?? Self.defaultCapabilities(for: role)
        self.filename = filename
        self.version = version
        self.quantization = quantization
        self.kvCacheMaxLen = kvCacheMaxLen
        self.hardwareTarget = hardwareTarget
        self.executionPlan = executionPlan
        self.downloadURL = downloadURL
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
    }
}

// MARK: - Lifecycle state

public enum Phase: String, Sendable {
    case idle
    case resolvingModels
    case downloadingModels
    case verifyingModels
    case initializingEngine
    case loadingRuntimeData
    case ready
    case failed
}

public struct State: Sendable {
    public var phase: Phase
    public var message: String
    public var artifacts: [ArtifactState]
    public var canRetry: Bool
    public var canUseChat: Bool
    public var errorMessage: String?

    public init(
        phase: Phase,
        message: String,
        artifacts: [ArtifactState] = [],
        canRetry: Bool = false,
        canUseChat: Bool = false,
        errorMessage: String? = nil
    ) {
        self.phase = phase
        self.message = message
        self.artifacts = artifacts
        self.canRetry = canRetry
        self.canUseChat = canUseChat
        self.errorMessage = errorMessage
    }

    public static let idle = State(phase: .idle, message: "Idle")

    public static func ready(artifacts: [ArtifactState]) -> State {
        State(
            phase: .ready,
            message: "Ready",
            artifacts: artifacts,
            canUseChat: true
        )
    }
}

// MARK: - Lifecycle events
public struct Event: Identifiable, Sendable {
    public enum Kind: String, Sendable {
        case modelPreparation = "model_preparation"
        case engineInitialization = "engine_initialization"
        case runtimeDataLoading = "runtime_data_loading"
        case teardown
    }

    public enum Status: String, Sendable {
        case running
        case passed
        case failed
        case skipped
    }

    public var id: UUID
    public var kind: Kind
    public var status: Status
    public var title: String
    public var detail: String
    public var durationMs: Int64?
    public var runtimeDataStats: RuntimeDataLoadResult?
    public var modelTimings: [ZephrAgentRuntime.Diagnostics.ModelTiming]
    public var artifacts: [ArtifactState]

    public init(
        id: UUID = UUID(),
        kind: Kind,
        status: Status,
        title: String,
        detail: String,
        durationMs: Int64? = nil,
        runtimeDataStats: RuntimeDataLoadResult? = nil,
        modelTimings: [ZephrAgentRuntime.Diagnostics.ModelTiming] = [],
        artifacts: [ArtifactState] = []
    ) {
        self.id = id
        self.kind = kind
        self.status = status
        self.title = title
        self.detail = detail
        self.durationMs = durationMs
        self.runtimeDataStats = runtimeDataStats
        self.modelTimings = modelTimings
        self.artifacts = artifacts
    }
}

}

extension ZephrAgentRuntime.Diagnostics {

// MARK: - Model timing diagnostics

public struct ModelTiming: Sendable {
    public var component: String
    public var action: String
    public var detail: String
    public var durationMs: Int64
    public var ok: Bool

    public init(
        component: String,
        action: String,
        detail: String,
        durationMs: Int64,
        ok: Bool
    ) {
        self.component = component
        self.action = action
        self.detail = detail
        self.durationMs = durationMs
        self.ok = ok
    }
}

}

extension ZephrAgentRuntime.Lifecycle {

// MARK: - Artifact state
public struct ArtifactState: Identifiable, Sendable {
    public enum Phase: String, Sendable {
        case pending
        case waitingForSharedDownload
        case checkingLocalFile
        case downloading
        case verifying
        case ready
        case skipped
        case failed
    }

    public var id: String
    public var title: String
    public var purpose: String
    public var phase: Phase
    public var downloadedBytes: Int64
    public var totalBytes: Int64?
    public var downloadBytesPerSecond: Double?
    public var version: String
    public var detail: String?

    public init(
        id: String,
        title: String,
        purpose: String = "",
        phase: Phase,
        downloadedBytes: Int64 = 0,
        totalBytes: Int64? = nil,
        downloadBytesPerSecond: Double? = nil,
        version: String,
        detail: String? = nil
    ) {
        self.id = id
        self.title = title
        self.purpose = purpose
        self.phase = phase
        self.downloadedBytes = downloadedBytes
        self.totalBytes = totalBytes
        self.downloadBytesPerSecond = downloadBytesPerSecond
        self.version = version
        self.detail = detail
    }

    public var fractionComplete: Double? {
        guard let totalBytes, totalBytes > 0 else {
            return nil
        }
        return min(1, max(0, Double(downloadedBytes) / Double(totalBytes)))
    }
}

// MARK: - Resolved models
public struct ResolvedModels: Sendable {
    public var llmModelURL: URL
    public var ragEmbeddingURL: URL?
    public var vlmModelURL: URL?
    public var executionSelection: ExecutionSelection
    public var executionPlan: ResolvedExecutionPlan?
    public var detectedLLMFamily: String?
}

// MARK: - Lifecycle errors
public enum Error: Swift.Error, CustomStringConvertible, Sendable {
    case noDownloadURL(String)
    case missingRequiredArtifact(String)
    case downloadFailed(String)
    case checksumMismatch(String)
    case invalidManifest(String)
    case engineNotReady

    public var description: String {
        switch self {
        case .noDownloadURL(let title):
            return "No download URL configured for \(title)"
        case .missingRequiredArtifact(let title):
            return "Missing required model artifact: \(title)"
        case .downloadFailed(let message):
            return message
        case .checksumMismatch(let title):
            return "Checksum verification failed for \(title)"
        case .invalidManifest(let message):
            return message
        case .engineNotReady:
            return "Agent engine is not ready"
        }
    }
}

public typealias ModelProgressHandler = @Sendable (
    Phase,
    String,
    [ArtifactState]
) async -> Void

// MARK: - Interrupted initialization
public struct InterruptedInitialization: Equatable, Sendable {
    public let modelChannel: ModelChannel?
    public let llmExecutionChoiceID: String?
    public let ragEmbeddingExecutionChoiceID: String?
    public let vlmExecutionChoiceID: String?
    public let startedAt: Date?

    public init(
        modelChannel: ModelChannel?,
        llmExecutionChoiceID: String?,
        ragEmbeddingExecutionChoiceID: String?,
        vlmExecutionChoiceID: String?,
        startedAt: Date?
    ) {
        self.modelChannel = modelChannel
        self.llmExecutionChoiceID = llmExecutionChoiceID
        self.ragEmbeddingExecutionChoiceID = ragEmbeddingExecutionChoiceID
        self.vlmExecutionChoiceID = vlmExecutionChoiceID
        self.startedAt = startedAt
    }

    public var message: String {
        "Previous agent initialization did not finish. Agent startup is paused so you can launch again or adjust settings before initialization."
    }
}

// MARK: - Runtime data loading results
public struct RuntimeDataLoadResult: Sendable {
    public let sourceCount: Int
    public let itemCount: Int
    public let indexSize: Int
    public let durationMs: Int64

    public init(sourceCount: Int, itemCount: Int, indexSize: Int, durationMs: Int64) {
        self.sourceCount = sourceCount
        self.itemCount = itemCount
        self.indexSize = indexSize
        self.durationMs = durationMs
    }
}

}

extension ZephrAgentRuntime {

// MARK: - Internal native diagnostics

enum NativeDiagnostics {
    static func enableNativeStderrLogging() {
        ZephrAgentRuntimeCBridge.enableNativeStderrLogging()
    }

    actor Session {
        private let engine: ZephrAgentRuntimeEngine

        init(
            llmModelPath: String,
            execution: ZephrAgentRuntime.Lifecycle.ExecutionDelegates,
            numThreads: Int = 0,
            litertCompilationCacheEnabled: Bool = false,
            ragEmbeddingPath: String? = nil
        ) throws {
            engine = try ZephrAgentRuntimeEngine(
                llmModelPath: llmModelPath,
                execution: execution,
                numThreads: numThreads,
                litertCompilationCacheEnabled: litertCompilationCacheEnabled,
                ragEmbeddingPath: ragEmbeddingPath
            )
        }

        func shutdown() async {
            _ = await engine.shutdown()
        }

    }
}

// MARK: - Internal runtime
@MainActor
@Observable
final class Runtime {
    /// Current runtime status snapshot; may update frequently during downloads.
    public internal(set) var lifecycleState: ZephrAgentRuntime.Lifecycle.State = .idle

    /// Current ordered lifecycle milestone list; entries are upserted by stable id.
    public internal(set) var lifecycleTimeline: [ZephrAgentRuntime.Lifecycle.Event] = []
    public internal(set) var resolvedModels: ZephrAgentRuntime.Lifecycle.ResolvedModels?
    public internal(set) var detectedModelFamily: String?

    @ObservationIgnored let modelManager: ModelManager
    @ObservationIgnored var engine: ZephrAgentRuntimeEngine?
    @ObservationIgnored var currentConfiguration: ZephrAgentRuntime.Lifecycle.Configuration?
    @ObservationIgnored var startTask: Task<Void, Never>?

    init(modelManager: ModelManager = ModelManager()) {
        self.modelManager = modelManager
    }

    public nonisolated static func interruptedInitialization(
        defaults: UserDefaults = .standard
    ) -> ZephrAgentRuntime.Lifecycle.InterruptedInitialization? {
        _interruptedInitialization(defaults: defaults)
    }

    public nonisolated static func clearInterruptedInitialization(
        defaults: UserDefaults = .standard
    ) {
        _clearInterruptedInitialization(defaults: defaults)
    }

    public func start(
        configuration: ZephrAgentRuntime.Lifecycle.Configuration,
        progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler? = nil
    ) {
        _start(configuration: configuration, progress: progress)
    }

    public func prepare(
        configuration: ZephrAgentRuntime.Lifecycle.Configuration,
        progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler? = nil
    ) async throws {
        try await _prepare(configuration: configuration, progress: progress)
    }

    public func retry() {
        _retry()
    }

    public func cancelActiveWork() {
        _cancelActiveWork()
    }

    func describeImage(
        _ image: ZephrAgentRuntime.Tools.RgbImage,
        prompt: String = "Briefly describe this image.",
        maxTokens: Int = 0
    ) async throws -> ZephrAgentRuntime.Tools.VisionResult {
        try await _describeImage(image, prompt: prompt, maxTokens: maxTokens)
    }

    func embedText(
        _ text: String,
        taskType: ZephrAgentRuntime.Embeddings.TaskType = .query
    ) async throws -> ZephrAgentRuntime.Embeddings.Embedding {
        try await _embedText(text, taskType: taskType)
    }

    func generateText(
        userMessage: String,
        systemMessage: String = "",
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topK: Int = 40,
        topP: Float = 0.95
    ) async throws -> ZephrAgentRuntime.Diagnostics.TextResult {
        try await _generateText(
            userMessage: userMessage,
            systemMessage: systemMessage,
            maxTokens: maxTokens,
            temperature: temperature,
            topK: topK,
            topP: topP
        )
    }

    public func deleteDownloadedModels() {
        _deleteDownloadedModels()
    }

    public func deleteDownloadedModelsAndCaches(cancelActiveWork: Bool = true) async {
        await _deleteDownloadedModelsAndCaches(cancelActiveWork: cancelActiveWork)
    }

    public func drainModelTimings() async -> [ZephrAgentRuntime.Diagnostics.ModelTiming] {
        await _drainModelTimings()
    }

    @discardableResult
    public func shutdown() async -> ZephrAgentRuntime.Lifecycle.Event {
        await _shutdown()
    }
}

// MARK: - Internal model manager

actor ModelManager {
    let storageDirectoryOverride: URL?
    let fileManager: FileManager

    init(
        storageDirectory: URL? = nil,
        fileManager: FileManager = .default
    ) {
        self.fileManager = fileManager
        self.storageDirectoryOverride = storageDirectory
    }

    func resolveModels(
        configuration: ZephrAgentRuntime.Lifecycle.Configuration,
        progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler? = nil
    ) async throws -> ZephrAgentRuntime.Lifecycle.ResolvedModels {
        try await _resolveModels(configuration: configuration, progress: progress)
    }

    func deleteDownloadedModels() {
        _deleteDownloadedModels()
    }

    func deleteDownloadedModels(storage: ZephrAgentRuntime.Lifecycle.ModelStorage) {
        _deleteDownloadedModels(storage: storage)
    }

    func deleteDownloadedModelsAndCaches() {
        _deleteDownloadedModelsAndCaches()
    }

    func deleteDownloadedModelsAndCaches(storage: ZephrAgentRuntime.Lifecycle.ModelStorage) {
        _deleteDownloadedModelsAndCaches(storage: storage)
    }

    func deleteCaches() {
        _deleteCaches()
    }
}

}
