import CoreGraphics
import Foundation

// MARK: - ZephrAgentRuntimeCBridge

// This is the only Swift SDK file that should call zephr_* C symbols directly.
// Keep C-boundary translation here so the public SDK stays platform-native.
enum ZephrAgentRuntimeCBridge {
    static func enableNativeStderrLogging() {
        zephr_enable_stderr_logging()
    }

    static func detectModelFamily(at url: URL) -> String? {
        guard let raw = zephr_detect_model_family(url.path) else {
            return nil
        }
        defer { zephr_free(raw) }
        let value = String(cString: raw)
        return value.isEmpty ? nil : value
    }

}

// MARK: - Error type

enum ZephrAgentRuntimeError: Error, CustomStringConvertible, Sendable {
    case initFailed(String)
    case processFailed
    case vlmFailed
    case textGenerationFailed

    var description: String {
        switch self {
        case .initFailed(let msg): return "ZephrAgentRuntime init failed: \(msg)"
        case .processFailed: return "ZephrAgentRuntime process failed"
        case .vlmFailed: return "ZephrAgentRuntime VLM failed"
        case .textGenerationFailed: return "ZephrAgentRuntime text generation failed"
        }
    }
}

struct ZephrAgentRuntimeNativeToolParam: Sendable {
    var name: String
    var description: String
    var type: String
    var enumValues: [String]
    var required: Bool
}

struct ZephrAgentRuntimeNativeTool: Sendable {
    var name: String
    var description: String
    var params: [ZephrAgentRuntimeNativeToolParam]
}

private final class ZephrTextStreamBox {
    let onChunk: @Sendable (String) -> Bool

    init(_ onChunk: @escaping @Sendable (String) -> Bool) {
        self.onChunk = onChunk
    }
}

private let zephrTextStreamCallback: zephr_text_stream_callback_t = { rawText, userData in
    guard let rawText, let userData else {
        return true
    }
    let box = Unmanaged<ZephrTextStreamBox>.fromOpaque(userData).takeUnretainedValue()
    return box.onChunk(String(cString: rawText))
}

// MARK: - Agent engine

private func makeVLMResult(from resultHandle: zephr_vlm_result_t) -> ZephrAgentRuntime.Tools.VisionResult {
    ZephrAgentRuntime.Tools.VisionResult(
        response: String(cString: zephr_vlm_response(resultHandle)),
        inputPatches: Int(zephr_vlm_input_patches(resultHandle)),
        validVisionTokens: Int(zephr_vlm_valid_vision_tokens(resultHandle)),
        imageTokenSlots: Int(zephr_vlm_image_token_slots(resultHandle)),
        resizedWidth: Int(zephr_vlm_resized_width(resultHandle)),
        resizedHeight: Int(zephr_vlm_resized_height(resultHandle)),
        promptTokens: Int(zephr_vlm_prompt_tokens(resultHandle)),
        decodeSteps: Int(zephr_vlm_decode_steps(resultHandle)),
        firstDecodeMs: zephr_vlm_first_decode_ms(resultHandle)
    )
}

private func makeTextResult(from resultHandle: zephr_text_result_t, agentHandle: zephr_agent_t) -> ZephrAgentRuntime.Diagnostics.TextResult {
    ZephrAgentRuntime.Diagnostics.TextResult(
        response: String(cString: zephr_text_response(resultHandle)),
        prompt: String(cString: zephr_text_prompt(resultHandle)),
        prefillTokens: Int(zephr_text_prefill_tokens(resultHandle)),
        stats: ZephrAgentRuntime.Diagnostics.StageStats(
            tokenizeMs: 0,
            prefillMs: zephr_text_prefill_ms(resultHandle),
            decodeMs: zephr_text_decode_ms(resultHandle),
            firstDecodeMs: zephr_text_first_decode_ms(resultHandle),
            inputTokens: Int(zephr_text_input_tokens(resultHandle)),
            outputTokens: Int(zephr_text_decode_steps(resultHandle)),
            mtpRejectedCycles: Int(zephr_text_mtp_rejected_cycles(resultHandle)),
            mtpRejectedAfterPrefix0: Int(zephr_text_mtp_rejected_after_prefix_0(resultHandle)),
            mtpRejectedAfterPrefix1: Int(zephr_text_mtp_rejected_after_prefix_1(resultHandle)),
            mtpRejectedAfterPrefix2: Int(zephr_text_mtp_rejected_after_prefix_2(resultHandle))
        ),
        currentPosition: Int(zephr_agent_text_current_pos(agentHandle))
    )
}

// MARK: - ZephrAgentRuntime.Lifecycle.ExecutionDelegates
extension ZephrAgentRuntime.Lifecycle.ExecutionDelegates {
    func withCConfig<T>(_ body: (zephr_agent_execution_config_t) -> T) -> T {
        llm.withCString { llmPlan in
            rag.withCString { ragPlan in
                vlm.withCString { vlmPlan in
                    body(zephr_agent_execution_config_t(
                        config_version: 1,
                        llm_execution_plan: llmPlan,
                        rag_execution_plan: ragPlan,
                        vlm_execution_plan: vlmPlan,
                        gemma4_runtime: zephr_agent_gemma4_runtime_config_t(
                            gpu_precision: gemma4Runtime.gpuPrecision.rawValue,
                            kv_cache_max_len: gemma4Runtime.kvCacheMaxTokens,
                            constrained_verify_batch: gemma4Runtime.constrainedVerifyBatch.rawValue,
                            mtp_enabled: gemma4Runtime.mtpEnabled,
                            mtp_trust_verify_kv: gemma4Runtime.mtpTrustVerifyKV,
                            mtp_adaptive_enabled: gemma4Runtime.mtpAdaptiveEnabled,
                            mtp_adaptive_min_cycles: gemma4Runtime.mtpAdaptiveMinCycles,
                            mtp_adaptive_min_saved_per_cycle: gemma4Runtime.mtpAdaptiveMinSavedPerCycle,
                            mtp_trace: gemma4Runtime.mtpTrace
                        ),
                        diagnostic_gemma4: zephr_agent_diagnostic_gemma4_config_t(
                            prefill_by_decode: diagnosticGemma4.prefillByDecode,
                            prefill_max_chunk: diagnosticGemma4.prefillMaxChunk,
                            constrained_verify_trace: diagnosticGemma4.constrainedVerifyTrace,
                            constrained_verify_max_accept: diagnosticGemma4.constrainedVerifyMaxAccept
                        )
                    ))
                }
            }
        }
    }
}

// MARK: - ZephrAgentRuntimeEngine
actor ZephrAgentRuntimeEngine {
    private nonisolated(unsafe) var handle: zephr_agent_t?
    private let nativeQueue = DispatchQueue(label: "xyz.zephr.sdks.agent.native")
    private let nativeQueueID = UUID()
    private static let nativeQueueKey = DispatchSpecificKey<UUID>()

    init(llmModelPath: String,
         execution: ZephrAgentRuntime.Lifecycle.ExecutionDelegates,
         numThreads: Int = 0,
         litertCompilationCacheEnabled: Bool = false,
         ragEmbeddingPath: String? = nil,
         vlmModelPath: String? = nil) throws {
        nativeQueue.setSpecific(key: Self.nativeQueueKey, value: nativeQueueID)
        guard let h = zephr_agent_create() else {
            throw ZephrAgentRuntimeError.initFailed("agent create")
        }

        let cacheDirectory = try Self.litertCompilationCacheRoot(create: litertCompilationCacheEnabled)
        if !litertCompilationCacheEnabled {
            Self.clearLiteRTCompilationCache()
        }

        let initialized: Bool
        initialized = nativeQueue.sync {
            execution.withCConfig { cConfig in
                if litertCompilationCacheEnabled {
                    return cacheDirectory.path.withCString { cachePath in
                        zephr_agent_init(
                            h, cConfig, Int32(numThreads),
                            llmModelPath,
                            ragEmbeddingPath,
                            vlmModelPath,
                            cachePath
                        )
                    }
                }
                return zephr_agent_init(
                    h, cConfig, Int32(numThreads),
                    llmModelPath,
                    ragEmbeddingPath,
                    vlmModelPath,
                    nil
                )
            }
        }

        guard initialized else {
            nativeQueue.sync {
                zephr_agent_destroy(h)
            }
            throw ZephrAgentRuntimeError.initFailed("agent init: \(llmModelPath)")
        }
        handle = h
    }

    deinit {
        if let handle {
            if DispatchQueue.getSpecific(key: Self.nativeQueueKey) == nativeQueueID {
                zephr_agent_destroy(handle)
            } else {
                nativeQueue.sync {
                    zephr_agent_destroy(handle)
                }
            }
        }
    }

    static func clearLiteRTCompilationCache(fileManager: FileManager = .default) {
        if let directory = try? litertCompilationCacheRoot(create: false, fileManager: fileManager) {
            try? fileManager.removeItem(at: directory)
        }
    }

    private static func litertCompilationCacheRoot(
        create: Bool,
        fileManager: FileManager = .default
    ) throws -> URL {
        let base = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let directory = base
            .appendingPathComponent("ZephrAgentRuntime", isDirectory: true)
            .appendingPathComponent("LiteRTCompilationCache", isDirectory: true)
        if create {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        return directory
    }

    func shutdown() -> [ZephrAgentRuntime.Diagnostics.ModelTiming] {
        guard let handle else { return [] }
        let timings = nativeQueue.sync {
            zephr_agent_release_models(handle)
            let timings = drainModelTimingsLocked(handle)
            zephr_agent_destroy(handle)
            return timings
        }
        self.handle = nil
        return timings
    }

    func drainModelTimings() -> [ZephrAgentRuntime.Diagnostics.ModelTiming] {
        guard let handle else { return [] }
        return nativeQueue.sync {
            drainModelTimingsLocked(handle)
        }
    }

    private func drainModelTimingsLocked(
        _ handle: zephr_agent_t
    ) -> [ZephrAgentRuntime.Diagnostics.ModelTiming] {
        let count = Int(zephr_agent_model_lifecycle_timing_count(handle))
        guard count > 0 else { return [] }
        var timings: [ZephrAgentRuntime.Diagnostics.ModelTiming] = []
        timings.reserveCapacity(count)
        for index in 0..<count {
            let cIndex = CInt(index)
            timings.append(ZephrAgentRuntime.Diagnostics.ModelTiming(
                component: String(cString: zephr_agent_model_lifecycle_component(handle, cIndex)),
                action: String(cString: zephr_agent_model_lifecycle_action(handle, cIndex)),
                detail: String(cString: zephr_agent_model_lifecycle_detail(handle, cIndex)),
                durationMs: zephr_agent_model_lifecycle_duration_ms(handle, cIndex),
                ok: zephr_agent_model_lifecycle_ok(handle, cIndex)
            ))
        }
        zephr_agent_model_lifecycle_clear(handle)
        return timings
    }

    // MARK: - Embeddings

    func embedText(
        _ text: String,
        taskType: ZephrAgentRuntime.Embeddings.TaskType = .query
    ) throws -> ZephrAgentRuntime.Embeddings.Embedding {
        guard let handle else {
            throw ZephrAgentRuntimeError.processFailed
        }
        return try nativeQueue.sync {
            guard let r = zephr_agent_embed_text(handle, text, taskType.rawValue) else {
                throw ZephrAgentRuntimeError.processFailed
            }
            defer { zephr_embedding_result_destroy(r) }
            let dimension = Int(zephr_embedding_dimension(r))
            let values: [Float]
            if let data = zephr_embedding_data(r), dimension > 0 {
                values = Array(UnsafeBufferPointer(start: data, count: dimension))
            } else {
                values = []
            }
            return ZephrAgentRuntime.Embeddings.Embedding(
                vector: values,
                dimension: dimension,
                durationMs: zephr_embedding_duration_ms(r),
                taskType: taskType
            )
        }
    }

    func describeImage(
        _ image: ZephrAgentRuntime.Tools.RgbImage,
        prompt: String,
        maxTokens: Int
    ) throws -> ZephrAgentRuntime.Tools.VisionResult {
        guard let handle else {
            throw ZephrAgentRuntimeError.vlmFailed
        }

        return try nativeQueue.sync {
            let resultHandle: zephr_vlm_result_t? = image.rgb.withUnsafeBytes { buffer in
                guard let baseAddress = buffer.bindMemory(to: UInt8.self).baseAddress else {
                    return nil
                }
                return zephr_agent_describe_image_rgb888(
                    handle,
                    baseAddress,
                    Int32(image.width),
                    Int32(image.height),
                    Int32(image.rowStride),
                    prompt,
                    Int32(maxTokens)
                )
            }

            guard let resultHandle else {
                throw ZephrAgentRuntimeError.vlmFailed
            }
            defer { zephr_vlm_result_destroy(resultHandle) }
            return makeVLMResult(from: resultHandle)
        }
    }

    func generateText(
        userMessage: String,
        systemMessage: String = "",
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topK: Int = 40,
        topP: Float = 0.95
    ) async throws -> ZephrAgentRuntime.Diagnostics.TextResult {
        guard let handle else {
            throw ZephrAgentRuntimeError.textGenerationFailed
        }
        let handleAddress = UInt(bitPattern: handle)
        return try await withCheckedThrowingContinuation { continuation in
            nativeQueue.async {
                do {
                    guard let handle = zephr_agent_t(bitPattern: handleAddress) else {
                        throw ZephrAgentRuntimeError.textGenerationFailed
                    }
                    guard let resultHandle = zephr_agent_generate_text(
                        handle,
                        userMessage,
                        systemMessage,
                        Int32(maxTokens),
                        temperature,
                        Int32(topK),
                        topP
                    ) else {
                        throw ZephrAgentRuntimeError.textGenerationFailed
                    }
                    defer { zephr_text_result_destroy(resultHandle) }
                    continuation.resume(returning: makeTextResult(from: resultHandle, agentHandle: handle))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func generateToolAwareText(
        userMessage: String,
        systemMessage: String = "",
        maxTokens: Int = 256,
        temperature: Float = 0,
        topK: Int = 40,
        topP: Float = 0.95,
        tools: [ZephrAgentRuntimeNativeTool],
        onChunk: (@Sendable (String) -> Bool)? = nil
    ) async throws -> ZephrAgentRuntime.Diagnostics.TextResult {
        guard let handle else {
            throw ZephrAgentRuntimeError.textGenerationFailed
        }
        let handleAddress = UInt(bitPattern: handle)
        return try await withCheckedThrowingContinuation { continuation in
            nativeQueue.async {
                do {
                    guard let handle = zephr_agent_t(bitPattern: handleAddress) else {
                        throw ZephrAgentRuntimeError.textGenerationFailed
                    }
                    let result = try Self.withNativeToolSpecs(tools) { toolSpecs in
                        let resultHandle: zephr_text_result_t?
                        if let onChunk {
                            let box = ZephrTextStreamBox(onChunk)
                            resultHandle = zephr_agent_generate_tool_aware_text_stream(
                                handle,
                                userMessage,
                                systemMessage,
                                Int32(maxTokens),
                                temperature,
                                Int32(topK),
                                topP,
                                0,
                                toolSpecs.baseAddress,
                                Int32(toolSpecs.count),
                                zephrTextStreamCallback,
                                Unmanaged.passUnretained(box).toOpaque()
                            )
                        } else {
                            resultHandle = zephr_agent_generate_tool_aware_text(
                                handle,
                                userMessage,
                                systemMessage,
                                Int32(maxTokens),
                                temperature,
                                Int32(topK),
                                topP,
                                0,
                                toolSpecs.baseAddress,
                                Int32(toolSpecs.count)
                            )
                        }
                        guard let resultHandle else {
                            throw ZephrAgentRuntimeError.textGenerationFailed
                        }
                        defer { zephr_text_result_destroy(resultHandle) }
                        return makeTextResult(from: resultHandle, agentHandle: handle)
                    }
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func generateTextFromPromptStreaming(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0,
        topK: Int = 40,
        topP: Float = 0.95,
        onChunk: @escaping @Sendable (String) -> Bool
    ) async throws -> ZephrAgentRuntime.Diagnostics.TextResult {
        guard let handle else {
            throw ZephrAgentRuntimeError.textGenerationFailed
        }
        let handleAddress = UInt(bitPattern: handle)
        return try await withCheckedThrowingContinuation { continuation in
            nativeQueue.async {
                do {
                    guard let handle = zephr_agent_t(bitPattern: handleAddress) else {
                        throw ZephrAgentRuntimeError.textGenerationFailed
                    }
                    let box = ZephrTextStreamBox(onChunk)
                    guard let resultHandle = zephr_agent_generate_text_from_prompt_stream(
                        handle,
                        prompt,
                        Int32(maxTokens),
                        temperature,
                        Int32(topK),
                        topP,
                        zephrTextStreamCallback,
                        Unmanaged.passUnretained(box).toOpaque()
                    ) else {
                        throw ZephrAgentRuntimeError.textGenerationFailed
                    }
                    defer { zephr_text_result_destroy(resultHandle) }
                    continuation.resume(returning: makeTextResult(from: resultHandle, agentHandle: handle))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func collectHeavyJson(
        prompt: String,
        collectActivations: Bool,
        topK: Int
    ) async throws -> String {
        guard let handle else {
            throw ZephrAgentRuntimeError.textGenerationFailed
        }
        let handleAddress = UInt(bitPattern: handle)
        return try await withCheckedThrowingContinuation { continuation in
            nativeQueue.async {
                guard let handle = zephr_agent_t(bitPattern: handleAddress),
                      let jsonPointer = zephr_agent_collect_heavy_json(
                        handle,
                        prompt,
                        collectActivations,
                        Int32(max(1, topK))
                      ) else {
                    continuation.resume(throwing: ZephrAgentRuntimeError.textGenerationFailed)
                    return
                }
                let json = String(cString: jsonPointer)
                zephr_free(UnsafeMutableRawPointer(jsonPointer))
                continuation.resume(returning: json)
            }
        }
    }

    func generateToolAwareTextFromPromptStreaming(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0,
        topK: Int = 40,
        topP: Float = 0.95,
        reserveOutputTokens: Int = 0,
        tools: [ZephrAgentRuntimeNativeTool],
        onChunk: @escaping @Sendable (String) -> Bool
    ) async throws -> ZephrAgentRuntime.Diagnostics.TextResult {
        guard let handle else {
            throw ZephrAgentRuntimeError.textGenerationFailed
        }
        let handleAddress = UInt(bitPattern: handle)
        return try await withCheckedThrowingContinuation { continuation in
            nativeQueue.async {
                do {
                    guard let handle = zephr_agent_t(bitPattern: handleAddress) else {
                        throw ZephrAgentRuntimeError.textGenerationFailed
                    }
                    let result = try Self.withNativeToolSpecs(tools) { toolSpecs in
                        let box = ZephrTextStreamBox(onChunk)
                        guard let resultHandle = zephr_agent_generate_tool_aware_text_from_prompt_stream(
                            handle,
                            prompt,
                            Int32(maxTokens),
                            temperature,
                            Int32(topK),
                            topP,
                            Int32(reserveOutputTokens),
                            toolSpecs.baseAddress,
                            Int32(toolSpecs.count),
                            zephrTextStreamCallback,
                            Unmanaged.passUnretained(box).toOpaque()
                        ) else {
                            throw ZephrAgentRuntimeError.textGenerationFailed
                        }
                        defer { zephr_text_result_destroy(resultHandle) }
                        return makeTextResult(from: resultHandle, agentHandle: handle)
                    }
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func continueAfterToolResponseStreaming(
        toolResponse: String,
        maxTokens: Int = 256,
        temperature: Float = 0,
        topK: Int = 40,
        topP: Float = 0.95,
        reserveOutputTokens: Int = 0,
        onChunk: @escaping @Sendable (String) -> Bool
    ) async throws -> ZephrAgentRuntime.Diagnostics.TextResult {
        guard let handle else {
            throw ZephrAgentRuntimeError.textGenerationFailed
        }
        let handleAddress = UInt(bitPattern: handle)
        return try await withCheckedThrowingContinuation { continuation in
            nativeQueue.async {
                do {
                    guard let handle = zephr_agent_t(bitPattern: handleAddress) else {
                        throw ZephrAgentRuntimeError.textGenerationFailed
                    }
                    let box = ZephrTextStreamBox(onChunk)
                    guard let resultHandle = zephr_agent_continue_after_tool_response_stream(
                        handle,
                        toolResponse,
                        Int32(maxTokens),
                        temperature,
                        Int32(topK),
                        topP,
                        Int32(reserveOutputTokens),
                        zephrTextStreamCallback,
                        Unmanaged.passUnretained(box).toOpaque()
                    ) else {
                        throw ZephrAgentRuntimeError.textGenerationFailed
                    }
                    defer { zephr_text_result_destroy(resultHandle) }
                    continuation.resume(returning: makeTextResult(from: resultHandle, agentHandle: handle))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func continueToolAwareTextStreaming(
        promptSuffix: String,
        maxTokens: Int = 256,
        temperature: Float = 0,
        topK: Int = 40,
        topP: Float = 0.95,
        reserveOutputTokens: Int = 0,
        tools: [ZephrAgentRuntimeNativeTool],
        onChunk: @escaping @Sendable (String) -> Bool
    ) async throws -> ZephrAgentRuntime.Diagnostics.TextResult {
        guard let handle else {
            throw ZephrAgentRuntimeError.textGenerationFailed
        }
        let handleAddress = UInt(bitPattern: handle)
        return try await withCheckedThrowingContinuation { continuation in
            nativeQueue.async {
                do {
                    guard let handle = zephr_agent_t(bitPattern: handleAddress) else {
                        throw ZephrAgentRuntimeError.textGenerationFailed
                    }
                    let result = try Self.withNativeToolSpecs(tools) { toolSpecs in
                        let box = ZephrTextStreamBox(onChunk)
                        guard let resultHandle = zephr_agent_continue_tool_aware_text_stream(
                            handle,
                            promptSuffix,
                            Int32(maxTokens),
                            temperature,
                            Int32(topK),
                            topP,
                            Int32(reserveOutputTokens),
                            toolSpecs.baseAddress,
                            Int32(toolSpecs.count),
                            zephrTextStreamCallback,
                            Unmanaged.passUnretained(box).toOpaque()
                        ) else {
                            throw ZephrAgentRuntimeError.textGenerationFailed
                        }
                        defer { zephr_text_result_destroy(resultHandle) }
                        return makeTextResult(from: resultHandle, agentHandle: handle)
                    }
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func withNativeToolSpecs<T>(
        _ tools: [ZephrAgentRuntimeNativeTool],
        _ body: (UnsafeBufferPointer<zephr_tool_spec_t>) throws -> T
    ) throws -> T {
        var cStrings: [UnsafeMutablePointer<CChar>] = []
        var enumValueBuffers: [UnsafeMutablePointer<UnsafePointer<CChar>?>] = []
        var paramBuffers: [UnsafeMutablePointer<zephr_tool_param_spec_t>] = []

        func makeCString(_ value: String) -> UnsafePointer<CChar>? {
            guard let pointer = strdup(value) else { return nil }
            cStrings.append(pointer)
            return UnsafePointer(pointer)
        }

        defer {
            cStrings.forEach { free($0) }
            enumValueBuffers.forEach { $0.deallocate() }
            paramBuffers.forEach { $0.deallocate() }
        }

        var specs: [zephr_tool_spec_t] = []
        specs.reserveCapacity(tools.count)
        for tool in tools {
            let params = UnsafeMutablePointer<zephr_tool_param_spec_t>.allocate(
                capacity: max(tool.params.count, 1)
            )
            paramBuffers.append(params)
            for (index, param) in tool.params.enumerated() {
                let enumValues = UnsafeMutablePointer<UnsafePointer<CChar>?>.allocate(
                    capacity: max(param.enumValues.count, 1)
                )
                enumValueBuffers.append(enumValues)
                for (enumIndex, value) in param.enumValues.enumerated() {
                    enumValues[enumIndex] = makeCString(value)
                }
                params[index] = zephr_tool_param_spec_t(
                    name: makeCString(param.name),
                    description: makeCString(param.description),
                    type: makeCString(param.type),
                    enum_values: UnsafePointer(enumValues),
                    enum_value_count: Int32(param.enumValues.count),
                    required: param.required
                )
            }
            specs.append(zephr_tool_spec_t(
                name: makeCString(tool.name),
                description: makeCString(tool.description),
                params: UnsafePointer(params),
                param_count: Int32(tool.params.count)
            ))
        }

        return try specs.withUnsafeBufferPointer(body)
    }
}
