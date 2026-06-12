import Foundation

struct ConversationInternalToolParam: Sendable {
    var name: String
    var description: String
    var type: String
    var enumValues: [String]
    var required: Bool
}

struct ConversationInternalTool: Sendable {
    var name: String
    var description: String
    var params: [ConversationInternalToolParam]
    var runtimePolicy: ConversationToolRuntimePolicy = .init()
    var execute: @Sendable ([String: String]) async -> String
    var appliedEffects: @Sendable ([String: String], String) async -> [ZephrAgentRuntime.Conversation.ToolEffect] = { _, _ in [] }

    var nativeTool: ZephrAgentRuntimeNativeTool {
        ZephrAgentRuntimeNativeTool(
            name: name,
            description: description,
            params: params.map {
                ZephrAgentRuntimeNativeToolParam(
                    name: $0.name,
                    description: $0.description,
                    type: $0.type,
                    enumValues: $0.enumValues,
                    required: $0.required
                )
            }
        )
    }
}

struct ConversationToolRuntimePolicy: Sendable {
    var invalidatesLiveTextKV: Bool = false
    var preferredContinuationMode: ZephrAgentRuntime.Conversation.ToolResponseContinuationMode?
    var maxResponseTokensForContinuation: Int?
    var recordInConversationHistory: Bool = true
}

enum ConversationProtocolTurn: Sendable {
    case user(String)
    case assistant(String)
    case tool(name: String, text: String)
}

final class ConversationChunkState: @unchecked Sendable {
    private let lock = NSLock()
    private var emitted = false

    var hasEmittedChunk: Bool {
        lock.withLock { emitted }
    }

    func markEmitted() {
        lock.withLock {
            emitted = true
        }
    }
}

struct ConversationParsedToolCall: Sendable {
    var name: String
    var arguments: [String: String]
}

enum ConversationToolFactory {
    static func imageInspectionTool(
        runtime: ZephrAgentRuntime.Runtime,
        contents: ZephrAgentRuntime.Conversation.Contents,
        resolver: @escaping ZephrAgentRuntime.Conversation.ImageResolver
    ) -> ConversationInternalTool {
        ConversationInternalTool(
            name: "inspect_image",
            description: "Inspect an image attached to the current user turn and answer a focused visual question about it.",
            params: [
                ConversationInternalToolParam(
                    name: "imageRef",
                    description: "The exact imageRef or imageFile value from the current user turn. If there is only one image, this may be omitted.",
                    type: "STRING",
                    enumValues: [],
                    required: false
                ),
                ConversationInternalToolParam(
                    name: "prompt",
                    description: "A focused question or instruction for inspecting the image.",
                    type: "STRING",
                    enumValues: [],
                    required: true
                )
            ],
            runtimePolicy: ConversationToolRuntimePolicy(
                invalidatesLiveTextKV: true,
                preferredContinuationMode: .replayPrompt
            ),
            execute: { args in
                let requestedRef = args["imageRef"]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                let imagePart = Self.imagePart(in: contents, matching: requestedRef)
                guard let imagePart else {
                    return Self.jsonString([
                        "error": "Unknown imageRef",
                        "imageRef": requestedRef
                    ])
                }
                do {
                    guard let image = try await resolver(imagePart) else {
                        return Self.jsonString([
                            "error": "Image resolver returned no image",
                            "imageRef": Self.imageIdentifier(imagePart)
                        ])
                    }
                    let focusedPrompt = args["prompt"]?.trimmingCharacters(in: .whitespacesAndNewlines)
                        ?? "Briefly describe this image."
                    let userText = Self.imageInspectionUserText(contents)
                    let prompt = Self.imageInspectionPrompt(
                        userText: userText,
                        focusedPrompt: focusedPrompt
                    )
                    let result = try await runtime._describeImage(
                        image,
                        prompt: prompt,
                        maxTokens: 0
                    )
                    return Self.jsonString([
                        "imageRef": Self.imageIdentifier(imagePart),
                        "prompt": prompt,
                        "userText": userText,
                        "focusedPrompt": focusedPrompt,
                        "response": result.response,
                        "inputPatches": result.inputPatches,
                        "validVisionTokens": result.validVisionTokens,
                        "imageTokenSlots": result.imageTokenSlots,
                        "resizedWidth": result.resizedWidth,
                        "resizedHeight": result.resizedHeight,
                        "promptTokens": result.promptTokens,
                        "decodeSteps": result.decodeSteps,
                        "firstDecodeMs": result.firstDecodeMs
                    ])
                } catch {
                    return Self.jsonString([
                        "error": String(describing: error),
                        "imageRef": Self.imageIdentifier(imagePart)
                    ])
                }
            }
        )
    }

    static func tools(from openApiTool: any ZephrAgentRuntime.Conversation.OpenApiTool) -> [ConversationInternalTool] {
        guard let data = openApiTool.getToolDescriptionJsonString().data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = json["name"] as? String else {
            return []
        }

        let parameters = json["parameters"] as? [String: Any]
        let properties = parameters?["properties"] as? [String: Any] ?? [:]
        let requiredValues = parameters?["required"] as? [String]
        let required = Set(requiredValues ?? [])
        let params = properties.keys.sorted().map { key in
            let schema = properties[key] as? [String: Any] ?? [:]
            let enumValues = schema["enum"] as? [String] ?? []
            return ConversationInternalToolParam(
                name: key,
                description: schema["description"] as? String ?? "",
                type: gemmaType(from: schema["type"] as? String),
                enumValues: enumValues,
                required: requiredValues == nil || required.contains(key)
            )
        }

        return [
            ConversationInternalTool(
                name: name,
                description: json["description"] as? String ?? "",
                params: params,
                execute: { arguments in
                    let params = paramsJSON(arguments)
                    return await openApiTool.execute(params: params)
                },
                appliedEffects: { arguments, response in
                    let params = paramsJSON(arguments)
                    return await openApiTool.appliedEffects(params: params, response: response)
                }
            )
        ]
    }

    private static func paramsJSON(_ arguments: [String: String]) -> String {
        let data = (try? JSONSerialization.data(withJSONObject: arguments)) ?? Data()
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    private static func gemmaType(from raw: String?) -> String {
        switch raw?.lowercased() {
        case "integer", "int", "long":
            return "INTEGER"
        case "number", "float", "double":
            return "NUMBER"
        case "boolean", "bool":
            return "BOOLEAN"
        default:
            return "STRING"
        }
    }

    private static func jsonString(_ object: [String: Any]) -> String {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let text = String(data: data, encoding: .utf8) else {
            return #"{"error":"failed to encode tool response"}"#
        }
        return text
    }

    private static func imageInspectionUserText(_ contents: ZephrAgentRuntime.Conversation.Contents) -> String {
        let text = contents.text.trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? contents.description : text
    }

    private static func imageInspectionPrompt(userText: String, focusedPrompt: String) -> String {
        """
        Original user request:
        \(userText)

        Focused visual question:
        \(focusedPrompt)

        Inspect only the image. Return concise visual facts relevant to the focused question.
        """
    }

    private static func imagePart(
        in contents: ZephrAgentRuntime.Conversation.Contents,
        matching requestedRef: String
    ) -> ZephrAgentRuntime.Conversation.ContentPart? {
        let imageParts = contents.imageParts
        if requestedRef.isEmpty, imageParts.count == 1 {
            return imageParts.first
        }
        return imageParts.first { imageIdentifier($0) == requestedRef }
    }

    private static func imageIdentifier(_ part: ZephrAgentRuntime.Conversation.ContentPart) -> String {
        switch part.kind {
        case .text:
            return ""
        case .imageRef:
            return part.imageRef ?? ""
        case .imageFile:
            return part.imageFile ?? ""
        }
    }
}

extension ZephrAgentRuntime.Conversation.Engine {
    func _initialize(progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler? = nil) async throws {
        try await _initialize(
            configuration: ZephrAgentRuntime.Lifecycle.Configuration(
                modelChannel: .public,
                llmExecutionChoiceID: "gemma4.gpu",
                ragEmbeddingExecutionChoiceID: "off",
                vlmExecutionChoiceID: "off",
                litertCompilationCacheEnabled: true
            ),
            progress: progress
        )
    }

    func _initialize(
        configuration: ZephrAgentRuntime.Lifecycle.Configuration,
        progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler? = nil
    ) async throws {
        try await runtime.prepare(
            configuration: configuration,
            progress: progress
        )
    }
}

extension ZephrAgentRuntime.Conversation.Conversation {
    func _sendMessageAsync(
        _ contents: ZephrAgentRuntime.Conversation.Contents,
        options: ZephrAgentRuntime.Conversation.SendOptions
    ) -> AsyncThrowingStream<ZephrAgentRuntime.Conversation.Message, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @MainActor in
                do {
                    let chunkState = ConversationChunkState()
                    let finalMessage = try await self.sendMessageInternal(
                        contents,
                        context: ZephrAgentRuntime.Conversation.Context(),
                        options: options,
                        onAssistantChunk: { chunk in
                            guard !chunk.isEmpty else { return true }
                            chunkState.markEmitted()
                            continuation.yield(ZephrAgentRuntime.Conversation.Message(contents: .of(chunk)))
                            return true
                        },
                        onEvent: nil
                    )
                    if !chunkState.hasEmittedChunk {
                        continuation.yield(finalMessage)
                    } else {
                        continuation.yield(ZephrAgentRuntime.Conversation.Message(
                            contents: .of(""),
                            generationStats: finalMessage.generationStats,
                            isFinal: finalMessage.isFinal
                        ))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
                Task { @MainActor in
                    self.cancelProcess()
                }
            }
        }
    }

    func _sendMessage(
        _ contents: ZephrAgentRuntime.Conversation.Contents,
        options: ZephrAgentRuntime.Conversation.SendOptions
    ) async throws -> ZephrAgentRuntime.Conversation.Message {
        try await sendMessageInternal(
            contents,
            context: ZephrAgentRuntime.Conversation.Context(),
            options: options,
            onAssistantChunk: nil,
            onEvent: nil
        )
    }

    func _sendMessage(
        _ contents: ZephrAgentRuntime.Conversation.Contents,
        context: ZephrAgentRuntime.Conversation.Context,
        options: ZephrAgentRuntime.Conversation.SendOptions
    ) async throws -> ZephrAgentRuntime.Conversation.Message {
        try await sendMessageInternal(contents, context: context, options: options, onAssistantChunk: nil, onEvent: nil)
    }

    func _sendEvents(
        _ contents: ZephrAgentRuntime.Conversation.Contents,
        context: ZephrAgentRuntime.Conversation.Context,
        options: ZephrAgentRuntime.Conversation.SendOptions
    ) -> AsyncThrowingStream<ZephrAgentRuntime.Conversation.Event, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @MainActor in
                do {
                    _ = try await self.sendMessageInternal(
                        contents,
                        context: context,
                        options: options,
                        onAssistantChunk: { chunk in
                            continuation.yield(.assistantDelta(chunk))
                            return true
                        },
                        onEvent: { event in
                            continuation.yield(event)
                        }
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
                Task { @MainActor in
                    self.cancelProcess()
                }
            }
        }
    }

    private func sendMessageInternal(
        _ contents: ZephrAgentRuntime.Conversation.Contents,
        context: ZephrAgentRuntime.Conversation.Context,
        options: ZephrAgentRuntime.Conversation.SendOptions,
        onAssistantChunk: (@Sendable (String) -> Bool)?,
        onEvent: ((ZephrAgentRuntime.Conversation.Event) -> Void)?
    ) async throws -> ZephrAgentRuntime.Conversation.Message {
        guard !closed else {
            throw ConversationError.conversationClosed
        }
        cancelled = false
        let currentTurnIndex = conversationTurnIndex
        conversationTurnIndex += 1
        let conversationStrategy = options.conversationStrategy ?? config.conversationStrategy
        let toolResponseContinuationMode = conversationStrategy.toolResponseContinuationMode ?? .incrementalKV
        let imageTools: [ConversationInternalTool]
        if contents.hasImageContent, let imageResolver = config.imageResolver {
            imageTools = [
                ConversationToolFactory.imageInspectionTool(
                    runtime: runtime,
                    contents: contents,
                    resolver: imageResolver
                )
            ]
        } else {
            imageTools = []
        }
        let registry = ConversationToolRegistry(providers: config.tools, extraTools: imageTools)
        let text = contents.description
        var nextUserText = if let promptPrefix = context.promptPrefix {
            userTextWithPromptPrefix(text, promptPrefix: promptPrefix)
        } else {
            text
        }
        var recordedUser = false
        var toolsEnabled = registry.hasTools
        var generationStats: ZephrAgentRuntime.Diagnostics.GenerationStats?
        var generationRecords: [ZephrAgentRuntime.Conversation.GenerationRecord] = []
        var toolCallRecords: [ZephrAgentRuntime.Conversation.ToolCallRecord] = []
        var toolResponseRecords: [ZephrAgentRuntime.Conversation.ToolResponseRecord] = []
        var effectRecords: [ZephrAgentRuntime.Conversation.ToolEffect] = []

        for turnIndex in 0..<Self.recurringToolCallLimit {
            try Task.checkCancellation()
            guard !cancelled else {
                throw ConversationError.generationCancelled
            }

            let nativeTools = registry.nativeTools
            let useLiveKvCache = turnIndex == 0
                && toolsEnabled
                && toolResponseContinuationMode == .incrementalKV
                && liveKvCacheReady
            let useProtocolReplay = turnIndex == 0
                && !useLiveKvCache
                && conversationStrategy == .fullReplay
            let promptUserMessage = (useLiveKvCache || useProtocolReplay)
                ? nextUserText
                : userMessageWithHistory(nextUserText)
            let promptSystemMessage = systemMessage(registry: registry, toolsEnabled: toolsEnabled)
            let result: ZephrAgentRuntime.Diagnostics.TextResult
            var resultLabel = "turn-\(turnIndex)"
            var resultKind = toolsEnabled ? "tool_aware" : "answer_only"
            var resultUsedLiveKvCache = false
            do {
                if useLiveKvCache {
                    let suffix = incrementalUserTurnPrompt(promptUserMessage)
                    resultLabel = "turn-\(turnIndex)-incremental-kv"
                    resultKind = "incremental_tool_aware"
                    resultUsedLiveKvCache = true
                    onEvent?(.generationStart(ZephrAgentRuntime.Conversation.GenerationStartRecord(
                        turnIndex: turnIndex,
                        label: resultLabel,
                        kind: resultKind,
                        prompt: suffix,
                        usedLiveKvCache: true,
                        conversationStrategy: conversationStrategy
                    )))
                    emitMemorySample(
                        turnIndex: turnIndex,
                        label: "\(resultLabel):before_generation",
                        onEvent: onEvent
                    )
                    do {
                        result = try await runtime._continueToolAwareTextStreaming(
                            promptSuffix: suffix,
                            maxTokens: config.maxTokens,
                            temperature: config.temperature,
                            topK: config.topK,
                            topP: config.topP,
                            reserveOutputTokens: config.reserveOutputTokens,
                            tools: nativeTools,
                            onChunk: onAssistantChunk ?? { _ in true }
                        )
                    } catch {
                        onEvent?(.incrementalFallback(ZephrAgentRuntime.Conversation.IncrementalFallbackRecord(
                            turnIndex: turnIndex,
                            label: resultLabel,
                            reason: String(describing: error)
                        )))
                        liveKvCacheReady = false
                        resultLabel = "turn-\(turnIndex)-fallback-replay"
                        resultKind = "tool_aware_fallback_replay"
                        resultUsedLiveKvCache = false
                        emitMemorySample(
                            turnIndex: turnIndex,
                            label: "\(resultLabel):before_generation",
                            onEvent: onEvent
                        )
                        result = try await runtime._generateToolAwareText(
                            userMessage: userMessageWithHistory(nextUserText),
                            systemMessage: promptSystemMessage,
                            maxTokens: config.maxTokens,
                            temperature: config.temperature,
                            topK: config.topK,
                            topP: config.topP,
                            tools: nativeTools,
                            onChunk: onAssistantChunk
                        )
                    }
                } else if useProtocolReplay && toolsEnabled {
                    let prompt = protocolPrompt(
                        currentText: promptUserMessage,
                        systemMessage: promptSystemMessage,
                        includeTools: true,
                        registry: registry
                    )
                    resultLabel = "turn-\(turnIndex)-full-replay"
                    resultKind = "tool_aware_full_replay"
                    onEvent?(.generationStart(ZephrAgentRuntime.Conversation.GenerationStartRecord(
                        turnIndex: turnIndex,
                        label: resultLabel,
                        kind: resultKind,
                        prompt: prompt,
                        usedLiveKvCache: false,
                        conversationStrategy: conversationStrategy
                    )))
                    emitMemorySample(
                        turnIndex: turnIndex,
                        label: "\(resultLabel):before_generation",
                        onEvent: onEvent
                    )
                    result = try await runtime._generateToolAwareTextFromPromptStreaming(
                        prompt: prompt,
                        maxTokens: config.maxTokens,
                        temperature: config.temperature,
                        topK: config.topK,
                        topP: config.topP,
                        reserveOutputTokens: config.reserveOutputTokens,
                        tools: nativeTools,
                        onChunk: onAssistantChunk ?? { _ in true }
                    )
                } else if toolsEnabled {
                    emitMemorySample(
                        turnIndex: turnIndex,
                        label: "\(resultLabel):before_generation",
                        onEvent: onEvent
                    )
                    result = try await runtime._generateToolAwareText(
                        userMessage: promptUserMessage,
                        systemMessage: promptSystemMessage,
                        maxTokens: config.maxTokens,
                        temperature: config.temperature,
                        topK: config.topK,
                        topP: config.topP,
                        tools: nativeTools,
                        onChunk: onAssistantChunk
                    )
                } else if useProtocolReplay {
                    let prompt = protocolPrompt(
                        currentText: promptUserMessage,
                        systemMessage: promptSystemMessage,
                        includeTools: false,
                        registry: registry
                    )
                    resultLabel = "turn-\(turnIndex)-full-replay"
                    resultKind = "answer_only_full_replay"
                    liveKvCacheReady = false
                    onEvent?(.generationStart(ZephrAgentRuntime.Conversation.GenerationStartRecord(
                        turnIndex: turnIndex,
                        label: resultLabel,
                        kind: resultKind,
                        prompt: prompt,
                        usedLiveKvCache: false,
                        conversationStrategy: conversationStrategy
                    )))
                    emitMemorySample(
                        turnIndex: turnIndex,
                        label: "\(resultLabel):before_generation",
                        onEvent: onEvent
                    )
                    result = try await runtime._generateTextFromPromptStreaming(
                        prompt: prompt,
                        maxTokens: config.maxTokens,
                        temperature: config.temperature,
                        topK: config.topK,
                        topP: config.topP,
                        onChunk: onAssistantChunk ?? { _ in true }
                    )
                } else {
                    liveKvCacheReady = false
                    emitMemorySample(
                        turnIndex: turnIndex,
                        label: "\(resultLabel):before_generation",
                        onEvent: onEvent
                    )
                    result = try await runtime.generateText(
                        userMessage: promptUserMessage,
                        systemMessage: promptSystemMessage,
                        maxTokens: config.maxTokens,
                        temperature: config.temperature,
                        topK: config.topK,
                        topP: config.topP
                    )
                }
            } catch {
                onEvent?(.generationFailure(ZephrAgentRuntime.Conversation.GenerationFailureRecord(
                    turnIndex: turnIndex,
                    label: resultLabel,
                    kind: resultKind,
                    reason: String(describing: error),
                    usedLiveKvCache: resultUsedLiveKvCache,
                    conversationStrategy: conversationStrategy
                )))
                throw error
            }
            onEvent?(.generationResult(ZephrAgentRuntime.Conversation.GenerationResultRecord(
                turnIndex: turnIndex,
                label: resultLabel,
                kind: resultKind,
                result: result,
                usedLiveKvCache: resultUsedLiveKvCache,
                conversationStrategy: conversationStrategy
            )))
            emitMemorySample(
                turnIndex: turnIndex,
                label: "\(resultLabel):after_generation",
                onEvent: onEvent
            )

            generationStats = generationStats.adding(result.generationStats)
            generationRecords.append(result.conversationGenerationRecord(
                turnIndex: turnIndex,
                label: resultLabel,
                kind: resultKind,
                usedLiveKvCache: resultUsedLiveKvCache,
                conversationStrategy: conversationStrategy
            ))
            let response = result.response.trimmingCharacters(in: .whitespacesAndNewlines)
            let toolCalls = toolsEnabled ? Self.parseToolCalls(response) : []
            if toolCalls.isEmpty {
                if !recordedUser {
                    turns.append(.user(text))
                    protocolTurns.append((role: "user", body: text))
                    recordedUser = true
                }
                turns.append(.assistant(response))
                protocolTurns.append((role: "model", body: response))
                liveKvCacheReady = toolResponseContinuationMode == .incrementalKV && toolsEnabled
                let message = ZephrAgentRuntime.Conversation.Message(
                    contents: .of(response),
                    generationStats: generationStats,
                    isFinal: true
                )
                onEvent?(.turnCompleted(ZephrAgentRuntime.Conversation.TurnRecord(
                    conversationTurnIndex: currentTurnIndex,
                    conversationStrategy: conversationStrategy,
                    userContent: contents,
                    userText: text,
                    finalText: response,
                    generationStats: generationStats,
                    generations: generationRecords,
                    toolCalls: toolCallRecords,
                    toolResponses: toolResponseRecords,
                    effects: effectRecords
                )))
                return message
            }

            let toolResponses = await registry.execute(toolCalls)
            for toolCall in toolCalls {
                let record = ZephrAgentRuntime.Conversation.ToolCallRecord(
                    turnIndex: turnIndex,
                    name: toolCall.name,
                    arguments: toolCall.arguments
                )
                toolCallRecords.append(record)
                onEvent?(.toolCall(record))
            }
            if !recordedUser {
                turns.append(.user(text))
                protocolTurns.append((role: "user", body: text))
                recordedUser = true
            }
            for toolResponse in toolResponses {
                turns.append(.tool(name: toolResponse.name, text: toolResponse.response))
                let record = ZephrAgentRuntime.Conversation.ToolResponseRecord(
                    turnIndex: turnIndex,
                    name: toolResponse.name,
                    response: toolResponse.response
                )
                toolResponseRecords.append(record)
                onEvent?(.toolResponse(record))
                for effect in toolResponse.effects {
                    effectRecords.append(effect)
                    onEvent?(.effectApplied(effect))
                }
            }
            let effectiveToolResponseContinuationMode = toolResponses.toolResponseContinuationMode(
                defaultMode: toolResponseContinuationMode
            )
            if let continued = try await continueAfterToolResponse(
                firstPass: result,
                responses: toolResponses,
                onAssistantChunk: onAssistantChunk,
                turnIndex: turnIndex,
                toolResponseContinuationMode: effectiveToolResponseContinuationMode,
                conversationStrategy: conversationStrategy,
                onEvent: onEvent
            ) {
                generationStats = generationStats.adding(continued.result.generationStats)
                generationRecords.append(continued.result.conversationGenerationRecord(
                    turnIndex: turnIndex,
                    label: continued.label,
                    kind: continued.kind,
                    usedLiveKvCache: continued.usedLiveKvCache,
                    conversationStrategy: conversationStrategy
                ))
                let finalResponse = continued.result.response.trimmingCharacters(in: .whitespacesAndNewlines)
                turns.append(.assistant(finalResponse))
                protocolTurns.append((
                    role: "model",
                    body: modelProtocolBodyAfterToolResponse(
                        firstPass: result,
                        responses: toolResponses,
                        finalResponse: finalResponse
                    )
                ))
                liveKvCacheReady = continued.usedLiveKvCache
                let message = ZephrAgentRuntime.Conversation.Message(
                    contents: .of(finalResponse),
                    generationStats: generationStats,
                    isFinal: true
                )
                onEvent?(.turnCompleted(ZephrAgentRuntime.Conversation.TurnRecord(
                    conversationTurnIndex: currentTurnIndex,
                    conversationStrategy: conversationStrategy,
                    userContent: contents,
                    userText: text,
                    finalText: finalResponse,
                    generationStats: generationStats,
                    generations: generationRecords,
                    toolCalls: toolCallRecords,
                    toolResponses: toolResponseRecords,
                    effects: effectRecords
                )))
                return message
            }
            nextUserText = continuationPrompt(originalUserText: text, responses: toolResponses)
            toolsEnabled = false
            liveKvCacheReady = false
        }

        throw ConversationError.toolCallLimitExceeded(Self.recurringToolCallLimit)
    }

    private func systemMessage(registry: ConversationToolRegistry, toolsEnabled: Bool) -> String {
        var parts: [String] = []
        let systemInstruction = config.systemInstruction?.description
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let systemInstruction, !systemInstruction.isEmpty {
            parts.append(systemInstruction)
        }
        if toolsEnabled {
            parts.append(
                """
                You have access to the enabled tools declared below.
                Use a single tool call when local place, visual, or navigation state is needed.
                """
            )
        } else if registry.hasTools {
            parts.append(
                "Use the provided tool results to answer directly. Do not call tools or write Gemma 4 tool-call syntax."
            )
        }
        return parts.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func userMessageWithHistory(_ currentText: String) -> String {
        let history = historySlice().map { turn in
            switch turn {
            case .user(let text):
                return "User: \(text)"
            case .assistant(let text):
                return "Assistant: \(text)"
            case .tool(let name, let text):
                return "Tool \(name): \(text)"
            }
        }
        if history.isEmpty {
            return currentText
        }
        return """

        Conversation history:
        \(history.joined(separator: "\n"))

        Current user request:
        \(currentText)
        """
    }

    private func userTextWithPromptPrefix(
        _ text: String,
        promptPrefix: String
    ) -> String {
        """
        \(promptPrefix.trimmingCharacters(in: .whitespacesAndNewlines))

        User request:
        \(text)
        """
    }

    private func historySlice() -> ArraySlice<ConversationProtocolTurn> {
        let maxHistoryTurns = max(0, config.maxHistoryTurns)
        if maxHistoryTurns == 0 {
            return turns[turns.endIndex...]
        }
        var userTurns = 0
        for index in turns.indices.reversed() {
            if case .user = turns[index] {
                userTurns += 1
                if userTurns >= maxHistoryTurns {
                    return turns[index...]
                }
            }
        }
        return turns[...]
    }

    private func protocolHistorySlice() -> ArraySlice<(role: String, body: String)> {
        let maxHistoryTurns = max(0, config.maxHistoryTurns)
        if maxHistoryTurns == 0 {
            return protocolTurns[protocolTurns.endIndex...]
        }
        var userTurns = 0
        for index in protocolTurns.indices.reversed() {
            if protocolTurns[index].role == "user" {
                userTurns += 1
                if userTurns >= maxHistoryTurns {
                    return protocolTurns[index...]
                }
            }
        }
        return protocolTurns[...]
    }

    private func protocolPrompt(
        currentText: String,
        systemMessage: String,
        includeTools: Bool,
        registry: ConversationToolRegistry
    ) -> String {
        var prompt = "<bos>"
        if !systemMessage.isEmpty || (includeTools && registry.hasTools) {
            prompt += "<|turn>system\n"
            prompt += systemMessage
            if includeTools && registry.hasTools {
                prompt += "<|tool>"
                for declaration in registry.declarations {
                    prompt += declaration
                }
                prompt += "<tool|>"
            }
            prompt += "<turn|>\n"
        }
        for turn in protocolHistorySlice() {
            prompt += "<|turn>\(turn.role)\n"
            prompt += turn.body
            prompt += "<turn|>\n"
        }
        prompt += "<|turn>user\n"
        prompt += currentText
        prompt += "<turn|>\n<|turn>model\n"
        return prompt
    }

    private func continuationPrompt(
        originalUserText: String,
        responses: [(name: String, response: String, effects: [ZephrAgentRuntime.Conversation.ToolEffect], runtimePolicy: ConversationToolRuntimePolicy)]
    ) -> String {
        let toolResults = responses.map { "\($0.name): \($0.response)" }.joined(separator: "\n")
        return """
        Original user request:
        \(originalUserText)

        Tool results:
        \(toolResults)

        Use the tool results to answer the original request concisely.
        """
    }

    private struct ToolResponseContinuationResult {
        var result: ZephrAgentRuntime.Diagnostics.TextResult
        var label: String
        var kind: String
        var usedLiveKvCache: Bool
    }

    private func continueAfterToolResponse(
        firstPass: ZephrAgentRuntime.Diagnostics.TextResult,
        responses: [(name: String, response: String, effects: [ZephrAgentRuntime.Conversation.ToolEffect], runtimePolicy: ConversationToolRuntimePolicy)],
        onAssistantChunk: (@Sendable (String) -> Bool)?,
        turnIndex: Int,
        toolResponseContinuationMode: ZephrAgentRuntime.Conversation.ToolResponseContinuationMode,
        conversationStrategy: ZephrAgentRuntime.Conversation.Strategy,
        onEvent: ((ZephrAgentRuntime.Conversation.Event) -> Void)?
    ) async throws -> ToolResponseContinuationResult? {
        guard !responses.isEmpty else { return nil }
        let toolResponseText = responses.map { formatToolResponse(name: $0.name, response: $0.response) }
            .joined()
        let usedLiveKvCache = toolResponseContinuationMode == .incrementalKV
        let label = usedLiveKvCache ? "turn-tool-response-incremental-kv" : "turn-tool-response-replay"
        let kind = usedLiveKvCache ? "tool_response_incremental_kv" : "tool_response_replay"
        let prompt: String
        switch toolResponseContinuationMode {
        case .incrementalKV:
            prompt = incrementalToolResponsePrompt(toolResponseText)
        case .replayPrompt:
            prompt = replayPromptAfterToolResponse(firstPass: firstPass, toolResponseText: toolResponseText)
        }
        onEvent?(.generationStart(ZephrAgentRuntime.Conversation.GenerationStartRecord(
            turnIndex: turnIndex,
            label: label,
            kind: kind,
            prompt: prompt,
            usedLiveKvCache: usedLiveKvCache,
            conversationStrategy: conversationStrategy
        )))
        emitMemorySample(
            turnIndex: turnIndex,
            label: "\(label):before_generation",
            onEvent: onEvent
        )
        do {
            let result: ZephrAgentRuntime.Diagnostics.TextResult
            switch toolResponseContinuationMode {
            case .incrementalKV:
                result = try await runtime._continueAfterToolResponseStreaming(
                    toolResponse: toolResponseText,
                    maxTokens: config.maxTokens,
                    temperature: config.temperature,
                    topK: config.topK,
                    topP: config.topP,
                    reserveOutputTokens: config.reserveOutputTokens,
                    onChunk: onAssistantChunk ?? { _ in true }
                )
            case .replayPrompt:
                result = try await runtime._generateTextFromPromptStreaming(
                    prompt: prompt,
                    maxTokens: config.maxTokens,
                    temperature: config.temperature,
                    topK: config.topK,
                    topP: config.topP,
                    onChunk: onAssistantChunk ?? { _ in true }
                )
            }
            onEvent?(.generationResult(ZephrAgentRuntime.Conversation.GenerationResultRecord(
                turnIndex: turnIndex,
                label: label,
                kind: kind,
                result: result,
                usedLiveKvCache: usedLiveKvCache,
                conversationStrategy: conversationStrategy
            )))
            emitMemorySample(
                turnIndex: turnIndex,
                label: "\(label):after_generation",
                onEvent: onEvent
            )
            if result.response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return nil
            }
            return ToolResponseContinuationResult(
                result: result,
                label: label,
                kind: kind,
                usedLiveKvCache: usedLiveKvCache
            )
        } catch {
            if toolResponseContinuationMode == .incrementalKV {
                onEvent?(.incrementalFallback(ZephrAgentRuntime.Conversation.IncrementalFallbackRecord(
                    turnIndex: turnIndex,
                    label: label,
                    reason: String(describing: error)
                )))
                let fallbackLabel = "turn-tool-response-fallback-replay"
                let fallbackKind = "tool_response_fallback_replay"
                let fallbackPrompt = replayPromptAfterToolResponse(
                    firstPass: firstPass,
                    toolResponseText: toolResponseText
                )
                onEvent?(.generationStart(ZephrAgentRuntime.Conversation.GenerationStartRecord(
                    turnIndex: turnIndex,
                    label: fallbackLabel,
                    kind: fallbackKind,
                    prompt: fallbackPrompt,
                    usedLiveKvCache: false,
                    conversationStrategy: conversationStrategy
                )))
                emitMemorySample(
                    turnIndex: turnIndex,
                    label: "\(fallbackLabel):before_generation",
                    onEvent: onEvent
                )
                do {
                    let result = try await runtime._generateTextFromPromptStreaming(
                        prompt: fallbackPrompt,
                        maxTokens: config.maxTokens,
                        temperature: config.temperature,
                        topK: config.topK,
                        topP: config.topP,
                        onChunk: onAssistantChunk ?? { _ in true }
                    )
                    onEvent?(.generationResult(ZephrAgentRuntime.Conversation.GenerationResultRecord(
                        turnIndex: turnIndex,
                        label: fallbackLabel,
                        kind: fallbackKind,
                        result: result,
                        usedLiveKvCache: false,
                        conversationStrategy: conversationStrategy
                    )))
                    emitMemorySample(
                        turnIndex: turnIndex,
                        label: "\(fallbackLabel):after_generation",
                        onEvent: onEvent
                    )
                    if result.response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        return nil
                    }
                    return ToolResponseContinuationResult(
                        result: result,
                        label: fallbackLabel,
                        kind: fallbackKind,
                        usedLiveKvCache: false
                    )
                } catch {
                    onEvent?(.generationFailure(ZephrAgentRuntime.Conversation.GenerationFailureRecord(
                        turnIndex: turnIndex,
                        label: fallbackLabel,
                        kind: fallbackKind,
                        reason: String(describing: error),
                        usedLiveKvCache: false,
                        conversationStrategy: conversationStrategy
                    )))
                    throw error
                }
            }
            onEvent?(.generationFailure(ZephrAgentRuntime.Conversation.GenerationFailureRecord(
                turnIndex: turnIndex,
                label: label,
                kind: kind,
                reason: String(describing: error),
                usedLiveKvCache: usedLiveKvCache,
                conversationStrategy: conversationStrategy
            )))
            throw error
        }
    }

    private func incrementalUserTurnPrompt(_ userMessage: String) -> String {
        """
        <turn|>
        <|turn>user
        \(userMessage)<turn|>
        <|turn>model

        """
    }

    private func emitMemorySample(
        turnIndex: Int?,
        label: String,
        onEvent: ((ZephrAgentRuntime.Conversation.Event) -> Void)?
    ) {
        onEvent?(.memorySample(ZephrAgentRuntime.Conversation.MemorySampleRecord(
            turnIndex: turnIndex,
            label: label,
            sample: ZephrAgentRuntime.Diagnostics.MemorySample.capture(label: label)
        )))
    }

    private func incrementalToolResponsePrompt(_ toolResponseText: String) -> String {
        "<|tool_response>\(toolResponseText)<tool_response|>"
    }

    private func replayPromptAfterToolResponse(
        firstPass: ZephrAgentRuntime.Diagnostics.TextResult,
        toolResponseText: String
    ) -> String {
        var prompt = firstPass.prompt
        prompt += firstPass.response
        if !firstPass.response.contains("<|tool_response>") {
            prompt += "<|tool_response>"
        }
        prompt += toolResponseText
        prompt += "<tool_response|>"
        return prompt
    }

    private func modelProtocolBodyAfterToolResponse(
        firstPass: ZephrAgentRuntime.Diagnostics.TextResult,
        responses: [(name: String, response: String, effects: [ZephrAgentRuntime.Conversation.ToolEffect], runtimePolicy: ConversationToolRuntimePolicy)],
        finalResponse: String
    ) -> String {
        var body = firstPass.response
        if !firstPass.response.contains("<|tool_response>") {
            body += "<|tool_response>"
        }
        body += responses.map { formatToolResponse(name: $0.name, response: $0.response) }.joined()
        body += "<tool_response|>"
        body += finalResponse
        return body
    }

    private func formatToolResponse(name: String, response: String) -> String {
        let text = response.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.hasPrefix("{") && text.hasSuffix("}") {
            return "response:\(name)\(text)"
        }
        return "response:\(name){result:<|\"|>\(escapeToolResponseText(text))<|\"|>}"
    }

    private func escapeToolResponseText(_ text: String) -> String {
        var escaped = ""
        for character in text {
            switch character {
            case "\\":
                escaped += "\\\\"
            case "\"":
                escaped += "\\\""
            case "\n":
                escaped += "\\n"
            case "\r":
                continue
            case "\t":
                escaped += "\\t"
            default:
                escaped.append(character)
            }
        }
        return escaped
    }

    private static func parseToolCalls(_ response: String) -> [ConversationParsedToolCall] {
        let pattern = #"<\|tool_call>call:([A-Za-z0-9_.-]+)\{(.*?)\}<tool_call\|>"#
        guard let regex = try? NSRegularExpression(
            pattern: pattern,
            options: [.dotMatchesLineSeparators]
        ) else {
            return []
        }
        let range = NSRange(response.startIndex..<response.endIndex, in: response)
        return regex.matches(in: response, range: range).compactMap { match in
            guard let nameRange = Range(match.range(at: 1), in: response),
                  let argsRange = Range(match.range(at: 2), in: response) else {
                return nil
            }
            return ConversationParsedToolCall(
                name: String(response[nameRange]),
                arguments: parseToolArguments(String(response[argsRange]))
            )
        }
    }

    private static func parseToolArguments(_ raw: String) -> [String: String] {
        let pattern = #"([A-Za-z0-9_.-]+):<\|"\|>(.*?)<\|"\|>"#
        guard let regex = try? NSRegularExpression(
            pattern: pattern,
            options: [.dotMatchesLineSeparators]
        ) else {
            return [:]
        }
        let range = NSRange(raw.startIndex..<raw.endIndex, in: raw)
        var result: [String: String] = [:]
        for match in regex.matches(in: raw, range: range) {
            guard let keyRange = Range(match.range(at: 1), in: raw),
                  let valueRange = Range(match.range(at: 2), in: raw) else {
                continue
            }
            result[String(raw[keyRange])] = String(raw[valueRange])
        }
        return result
    }

    private static let recurringToolCallLimit = 25
    private static let defaultMaxTokens = 512
    private static let defaultTemperature: Float = 0
    private static let defaultTopK = 40
    private static let defaultTopP: Float = 0.95
}

private extension Array where Element == (name: String, response: String, effects: [ZephrAgentRuntime.Conversation.ToolEffect], runtimePolicy: ConversationToolRuntimePolicy) {
    func toolResponseContinuationMode(
        defaultMode: ZephrAgentRuntime.Conversation.ToolResponseContinuationMode
    ) -> ZephrAgentRuntime.Conversation.ToolResponseContinuationMode {
        if let preferredMode = compactMap(\.runtimePolicy.preferredContinuationMode).first {
            return preferredMode
        }
        return contains { $0.runtimePolicy.invalidatesLiveTextKV } ? .replayPrompt : defaultMode
    }
}

private struct ConversationToolRegistry {
    let toolsByName: [String: ConversationInternalTool]

    init(providers: [ZephrAgentRuntime.Conversation.ToolProvider], extraTools: [ConversationInternalTool] = []) {
        let tools = providers.flatMap { $0.tools() } + extraTools
        toolsByName = Dictionary(uniqueKeysWithValues: tools.map { ($0.name, $0) })
    }

    var hasTools: Bool {
        !toolsByName.isEmpty
    }

    var nativeTools: [ZephrAgentRuntimeNativeTool] {
        toolsByName.values.sorted { $0.name < $1.name }.map(\.nativeTool)
    }

    var declarations: [String] {
        toolsByName.values.sorted { $0.name < $1.name }.map(conversationToolDeclaration)
    }

    func execute(_ calls: [ConversationParsedToolCall]) async -> [(name: String, response: String, effects: [ZephrAgentRuntime.Conversation.ToolEffect], runtimePolicy: ConversationToolRuntimePolicy)] {
        var responses: [(name: String, response: String, effects: [ZephrAgentRuntime.Conversation.ToolEffect], runtimePolicy: ConversationToolRuntimePolicy)] = []
        for call in calls {
            guard let tool = toolsByName[call.name] else {
                responses.append((call.name, #"{"error":"Unknown tool: \#(Self.jsonEscape(call.name))"}"#, [], .init()))
                continue
            }
            let response = await tool.execute(call.arguments)
            let effects = await tool.appliedEffects(call.arguments, response)
            responses.append((call.name, response, effects, tool.runtimePolicy))
        }
        return responses
    }

    private static func jsonEscape(_ text: String) -> String {
        text
            .replacingOccurrences(of: #"\"#, with: #"\\"#)
            .replacingOccurrences(of: #"""#, with: #"\""#)
            .replacingOccurrences(of: "\n", with: #"\\n"#)
            .replacingOccurrences(of: "\r", with: "")
            .replacingOccurrences(of: "\t", with: #"\\t"#)
    }
}

private func conversationToolDeclaration(_ tool: ConversationInternalTool) -> String {
    let params = tool.params.sorted { $0.name < $1.name }
    let properties = params.map { param in
        var property = "\(param.name):{description:<|\"|>\(conversationToolEscape(param.description))<|\"|>,type:<|\"|>\(param.type)<|\"|>"
        if !param.enumValues.isEmpty {
            let enumValues = param.enumValues
                .map { "<|\"|>\(conversationToolEscape($0))<|\"|>" }
                .joined(separator: ",")
            property += ",enum:[\(enumValues)]"
        }
        property += "}"
        return property
    }.joined(separator: ",")
    let required = params
        .filter(\.required)
        .map { "<|\"|>\($0.name)<|\"|>" }
        .joined(separator: ",")
    return "declaration:\(tool.name){" +
        "description:<|\"|>\(conversationToolEscape(tool.description))<|\"|>," +
        "parameters:{properties:{\(properties)}," +
        "required:[\(required)]," +
        "type:<|\"|>OBJECT<|\"|>}}"
}

private func conversationToolEscape(_ text: String) -> String {
    text
        .replacingOccurrences(of: #"\"#, with: #"\\"#)
        .replacingOccurrences(of: #"""#, with: #"\""#)
        .replacingOccurrences(of: "\n", with: #"\\n"#)
        .replacingOccurrences(of: "\r", with: "")
        .replacingOccurrences(of: "\t", with: #"\\t"#)
}

extension ZephrAgentRuntime.Runtime {
    func _generateToolAwareText(
        userMessage: String,
        systemMessage: String = "",
        maxTokens: Int = 256,
        temperature: Float = 0,
        topK: Int = 40,
        topP: Float = 0.95,
        tools: [ZephrAgentRuntimeNativeTool],
        onChunk: (@Sendable (String) -> Bool)? = nil
    ) async throws -> ZephrAgentRuntime.Diagnostics.TextResult {
        guard let engine else {
            throw ZephrAgentRuntime.Lifecycle.Error.engineNotReady
        }
        return try await engine.generateToolAwareText(
            userMessage: userMessage,
            systemMessage: systemMessage,
            maxTokens: maxTokens,
            temperature: temperature,
            topK: topK,
            topP: topP,
            tools: tools,
            onChunk: onChunk
        )
    }

    func _generateToolAwareTextFromPromptStreaming(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0,
        topK: Int = 40,
        topP: Float = 0.95,
        reserveOutputTokens: Int = 0,
        tools: [ZephrAgentRuntimeNativeTool],
        onChunk: @escaping @Sendable (String) -> Bool
    ) async throws -> ZephrAgentRuntime.Diagnostics.TextResult {
        guard let engine else {
            throw ZephrAgentRuntime.Lifecycle.Error.engineNotReady
        }
        return try await engine.generateToolAwareTextFromPromptStreaming(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topK: topK,
            topP: topP,
            reserveOutputTokens: reserveOutputTokens,
            tools: tools,
            onChunk: onChunk
        )
    }
}

private extension Optional where Wrapped == ZephrAgentRuntime.Diagnostics.GenerationStats {
    func adding(_ other: ZephrAgentRuntime.Diagnostics.GenerationStats) -> ZephrAgentRuntime.Diagnostics.GenerationStats {
        switch self {
        case .some(let existing):
            return existing.adding(other)
        case .none:
            return other
        }
    }
}

private extension ZephrAgentRuntime.Diagnostics.TextResult {
    var generationStats: ZephrAgentRuntime.Diagnostics.GenerationStats {
        ZephrAgentRuntime.Diagnostics.GenerationStats(
            prefillTokens: prefillTokens,
            decodeTokens: stats.outputTokens,
            prefillMs: stats.prefillMs,
            decodeMs: stats.decodeMs,
            firstDecodeMs: stats.firstDecodeMs
        )
    }

    func conversationGenerationRecord(
        turnIndex: Int,
        label: String,
        kind: String,
        usedLiveKvCache: Bool,
        conversationStrategy: ZephrAgentRuntime.Conversation.Strategy
    ) -> ZephrAgentRuntime.Conversation.GenerationRecord {
        ZephrAgentRuntime.Conversation.GenerationRecord(
            turnIndex: turnIndex,
            label: label,
            kind: kind,
            prompt: prompt,
            response: response,
            stats: stats,
            currentPosition: currentPosition,
            usedLiveKvCache: usedLiveKvCache,
            conversationStrategy: conversationStrategy
        )
    }
}

private enum ConversationError: Error, CustomStringConvertible {
    case conversationClosed
    case generationCancelled
    case invalidConfiguration(String)
    case toolCallLimitExceeded(Int)

    var description: String {
        switch self {
        case .conversationClosed:
            return "Conversation is closed"
        case .generationCancelled:
            return "Conversation generation was cancelled"
        case .invalidConfiguration(let message):
            return message
        case .toolCallLimitExceeded(let limit):
            return "Exceeded recurring tool call limit of \(limit)"
        }
    }
}
