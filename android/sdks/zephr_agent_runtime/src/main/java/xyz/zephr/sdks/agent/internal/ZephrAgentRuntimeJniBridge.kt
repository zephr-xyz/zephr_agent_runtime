package xyz.zephr.sdks.agent.internal

// This is the only Kotlin SDK file that should call JNI symbols directly.
// Keep native-boundary translation here so the public SDK stays platform-native.
internal object ZephrAgentRuntimeJniBridge {
    internal fun interface TextStreamCallback {
        fun onText(text: String): Boolean
    }

    init {
        System.loadLibrary("zephr_agent_android")
    }

    external fun createAgent(): Long
    external fun destroyAgent(agentHandle: Long)
    external fun releaseModels(agentHandle: Long)

    external fun initAgent(
        agentHandle: Long,
        llmExecutionPlan: String,
        ragExecutionPlan: String,
        vlmExecutionPlan: String,
        gemma4RuntimeMtpEnabled: Boolean,
        gemma4GpuPrecision: Int,
        gemma4KvCacheMaxTokens: Int,
        gemma4ConstrainedVerifyBatch: Int,
        gemma4MtpTrustVerifyKv: Boolean,
        gemma4MtpAdaptiveEnabled: Boolean,
        gemma4MtpAdaptiveMinCycles: Int,
        gemma4MtpAdaptiveMinSavedPerCycle: Float,
        gemma4MtpTrace: Boolean,
        diagnosticGemma4PrefillByDecode: Boolean,
        diagnosticGemma4PrefillMaxChunk: Int,
        diagnosticGemma4ConstrainedVerifyTrace: Boolean,
        diagnosticGemma4ConstrainedVerifyMaxAccept: Int,
        numThreads: Int,
        llmModelPath: String,
        ragEmbeddingPath: String?,
        vlmModelPath: String?,
        litertCompilationCacheDir: String?,
        litertRuntimeLibraryDir: String?,
    ): Boolean

    external fun textCurrentPosition(agentHandle: Long): Int
    external fun modelLifecycleTimingCount(agentHandle: Long): Int
    external fun modelLifecycleComponent(agentHandle: Long, index: Int): String
    external fun modelLifecycleAction(agentHandle: Long, index: Int): String
    external fun modelLifecycleDetail(agentHandle: Long, index: Int): String
    external fun modelLifecycleDurationMs(agentHandle: Long, index: Int): Long
    external fun modelLifecycleOk(agentHandle: Long, index: Int): Boolean
    external fun clearModelTimings(agentHandle: Long)

    external fun embedText(agentHandle: Long, text: String, taskType: String): Long
    external fun destroyEmbeddingResult(resultHandle: Long)
    external fun embeddingDimension(resultHandle: Long): Int
    external fun embeddingDurationMs(resultHandle: Long): Long
    external fun embeddingData(resultHandle: Long): FloatArray

    external fun describeImageRgb888(
        agentHandle: Long,
        rgb: ByteArray,
        width: Int,
        height: Int,
        rowStride: Int,
        prompt: String,
        maxTokens: Int,
    ): Long

    external fun destroyVisionResult(resultHandle: Long)
    external fun vlmResponse(resultHandle: Long): String
    external fun vlmInputPatches(resultHandle: Long): Int
    external fun vlmValidVisionTokens(resultHandle: Long): Int
    external fun vlmImageTokenSlots(resultHandle: Long): Int
    external fun vlmResizedWidth(resultHandle: Long): Int
    external fun vlmResizedHeight(resultHandle: Long): Int
    external fun vlmPromptTokens(resultHandle: Long): Int
    external fun vlmDecodeSteps(resultHandle: Long): Int
    external fun vlmFirstDecodeMs(resultHandle: Long): Long

    external fun generateText(
        agentHandle: Long,
        userMessage: String,
        systemMessage: String?,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
    ): Long

    external fun generateToolAwareText(
        agentHandle: Long,
        userMessage: String,
        systemMessage: String?,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        reserveOutputTokens: Int,
        toolNames: Array<String>,
        toolDescriptions: Array<String>,
        paramToolIndexes: IntArray,
        paramNames: Array<String>,
        paramDescriptions: Array<String>,
        paramTypes: Array<String>,
        paramRequired: BooleanArray,
        paramEnumValues: Array<String>,
    ): Long

    external fun generateToolAwareTextStreaming(
        agentHandle: Long,
        userMessage: String,
        systemMessage: String?,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        reserveOutputTokens: Int,
        toolNames: Array<String>,
        toolDescriptions: Array<String>,
        paramToolIndexes: IntArray,
        paramNames: Array<String>,
        paramDescriptions: Array<String>,
        paramTypes: Array<String>,
        paramRequired: BooleanArray,
        paramEnumValues: Array<String>,
        callback: TextStreamCallback,
    ): Long

    external fun generateToolAwareTextFromPromptStreaming(
        agentHandle: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        reserveOutputTokens: Int,
        toolNames: Array<String>,
        toolDescriptions: Array<String>,
        paramToolIndexes: IntArray,
        paramNames: Array<String>,
        paramDescriptions: Array<String>,
        paramTypes: Array<String>,
        paramRequired: BooleanArray,
        paramEnumValues: Array<String>,
        callback: TextStreamCallback,
    ): Long

    external fun generateTextFromPromptStreaming(
        agentHandle: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        callback: TextStreamCallback,
    ): Long

    external fun continueAfterToolResponseStreaming(
        agentHandle: Long,
        toolResponse: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        reserveOutputTokens: Int,
        callback: TextStreamCallback,
    ): Long

    external fun continueToolAwareTextStreaming(
        agentHandle: Long,
        promptSuffix: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        reserveOutputTokens: Int,
        toolNames: Array<String>,
        toolDescriptions: Array<String>,
        paramToolIndexes: IntArray,
        paramNames: Array<String>,
        paramDescriptions: Array<String>,
        paramTypes: Array<String>,
        paramRequired: BooleanArray,
        paramEnumValues: Array<String>,
        callback: TextStreamCallback,
    ): Long

    external fun destroyTextResult(resultHandle: Long)
    external fun textResponse(resultHandle: Long): String
    external fun textPrompt(resultHandle: Long): String
    external fun textInputTokens(resultHandle: Long): Int
    external fun textPrefillTokens(resultHandle: Long): Int
    external fun textDecodeSteps(resultHandle: Long): Int
    external fun textMtpRejectedCycles(resultHandle: Long): Int
    external fun textMtpRejectedAfterPrefix0(resultHandle: Long): Int
    external fun textMtpRejectedAfterPrefix1(resultHandle: Long): Int
    external fun textMtpRejectedAfterPrefix2(resultHandle: Long): Int
    external fun textPrefillMs(resultHandle: Long): Long
    external fun textDecodeMs(resultHandle: Long): Long
    external fun textFirstDecodeMs(resultHandle: Long): Long

    external fun collectHeavyJson(
        agentHandle: Long,
        prompt: String,
        collectActivations: Boolean,
        topK: Int,
    ): String

    external fun detectModelFamily(path: String): String
}
