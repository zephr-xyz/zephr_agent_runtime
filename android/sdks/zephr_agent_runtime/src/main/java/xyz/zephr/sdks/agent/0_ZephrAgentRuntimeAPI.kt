package xyz.zephr.sdks.agent

import android.content.Context
import android.os.Build
import android.os.Debug
import android.util.Log
import io.ktor.client.HttpClient
import io.ktor.client.engine.okhttp.OkHttp
import io.modelcontextprotocol.kotlin.sdk.client.Client
import io.modelcontextprotocol.kotlin.sdk.client.StreamableHttpClientTransport
import io.modelcontextprotocol.kotlin.sdk.types.Implementation
import io.modelcontextprotocol.kotlin.sdk.types.TextContent
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.Executors
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonObject
import kotlin.reflect.KClass
import kotlin.reflect.KFunction
import kotlin.reflect.KParameter
import kotlin.reflect.KType
import kotlin.reflect.full.findAnnotation
import kotlin.reflect.full.memberFunctions
import org.json.JSONArray
import org.json.JSONObject
import xyz.zephr.sdks.agent.internal.ZephrAgentRuntimeJniBridge

/**
 * Single public namespace for the Android Zephr Agent Runtime.
 *
 * Public SDK declarations should live here as nested members of [ZephrAgentRuntime].
 * Implementation details should stay private here or internal in subpackages.
 */
public object ZephrAgentRuntime {
    private const val PROMPT_LOG_TAG = "ZephrPrompt"
    private const val PROMPT_LOG_CHUNK_SIZE = 3500

    // MARK: - Configuration

    internal data class Configuration(
        val llmModelPath: String,
        val llmExecutionPlan: String = "cpu",
        val ragExecutionPlan: String = "cpu",
        val vlmExecutionPlan: String = "gpu",
        val gemma4Runtime: Lifecycle.Gemma4Options = Lifecycle.Gemma4Options(),
        val diagnosticGemma4: Diagnostics.Gemma4Options = Diagnostics.Gemma4Options(),
        val numThreads: Int = 0,
        val ragEmbeddingPath: String? = null,
        val vlmModelPath: String? = null,
        val litertCompilationCacheDir: String? = null,
        val litertRuntimeLibraryDir: String? = null,
    )

    public object Diagnostics {
        public data class StageStats(
            val tokenizeMs: Long,
            val prefillMs: Long,
            val decodeMs: Long,
            val firstDecodeMs: Long,
            val inputTokens: Int,
            val outputTokens: Int,
            val mtpRejectedCycles: Int = 0,
            val mtpRejectedAfterPrefix0: Int = 0,
            val mtpRejectedAfterPrefix1: Int = 0,
            val mtpRejectedAfterPrefix2: Int = 0,
        ) {
            public companion object {
                public val zero: StageStats = StageStats(
                    tokenizeMs = 0,
                    prefillMs = 0,
                    decodeMs = 0,
                    firstDecodeMs = 0,
                    inputTokens = 0,
                    outputTokens = 0,
                )
            }
        }

        public data class GenerationStats(
            val prefillTokens: Int,
            val decodeTokens: Int,
            val prefillMs: Long,
            val decodeMs: Long,
            val firstDecodeMs: Long,
            val modelCalls: Int = 1,
        ) {
            public val prefillTokensPerSecond: Double
                get() = tokensPerSecond(prefillTokens, prefillMs)

            public val decodeTokensPerSecond: Double
                get() = tokensPerSecond(decodeTokens, decodeMs)

            public operator fun plus(other: GenerationStats): GenerationStats =
                GenerationStats(
                    prefillTokens = prefillTokens + other.prefillTokens,
                    decodeTokens = decodeTokens + other.decodeTokens,
                    prefillMs = prefillMs + other.prefillMs,
                    decodeMs = decodeMs + other.decodeMs,
                    firstDecodeMs = firstNonZero(firstDecodeMs, other.firstDecodeMs),
                    modelCalls = modelCalls + other.modelCalls,
                )

            private companion object {
                private fun tokensPerSecond(tokens: Int, millis: Long): Double =
                    if (tokens > 0 && millis > 0) tokens * 1000.0 / millis else 0.0

                private fun firstNonZero(lhs: Long, rhs: Long): Long =
                    if (lhs > 0) lhs else rhs
            }
        }

        public data class MemorySample(
            val label: String,
            val rssMb: Long? = null,
            val peakRssMb: Long? = null,
            val nativeHeapAllocatedMb: Long? = null,
            val javaHeapUsedMb: Long? = null,
            val pssMb: Long? = null,
            val privateDirtyMb: Long? = null,
        ) {
            public companion object {
                public fun capture(label: String): MemorySample {
                    val status = procStatusValues("VmRSS", "VmHWM")
                    val runtime = java.lang.Runtime.getRuntime()
                    val info = Debug.MemoryInfo()
                    Debug.getMemoryInfo(info)
                    return MemorySample(
                        label = label,
                        rssMb = status["VmRSS"],
                        peakRssMb = status["VmHWM"],
                        nativeHeapAllocatedMb = Debug.getNativeHeapAllocatedSize().bytesToMb(),
                        javaHeapUsedMb = (runtime.totalMemory() - runtime.freeMemory()).bytesToMb(),
                        pssMb = info.totalPss.kbToMb(),
                        privateDirtyMb = info.totalPrivateDirty.kbToMb(),
                    )
                }

                private fun procStatusValues(vararg names: String): Map<String, Long> {
                    val wanted = names.toSet()
                    return runCatching {
                        File("/proc/self/status").readLines().mapNotNull { line ->
                            val key = line.substringBefore(':')
                            if (key !in wanted) return@mapNotNull null
                            val kb = line.substringAfter(':')
                                .trim()
                                .substringBefore(' ')
                                .toLongOrNull()
                            kb?.let { key to it.kbToMb() }
                        }.toMap()
                    }.getOrDefault(emptyMap())
                }

                private fun Long.bytesToMb(): Long = this / (1024L * 1024L)
                private fun Int.kbToMb(): Long = toLong() / 1024L
                private fun Long.kbToMb(): Long = this / 1024L
            }
        }

        public data class ModelTiming(
            val component: String,
            val action: String,
            val detail: String,
            val durationMs: Long,
            val ok: Boolean,
        )

        public data class TextResult(
            val response: String,
            val prompt: String,
            val prefillTokens: Int,
            val stats: StageStats,
            val currentPosition: Int = 0,
        )

        public data class Gemma4Options(
            val prefillByDecode: Boolean = false,
            val prefillMaxChunk: Int = 0,
            val constrainedVerifyTrace: Boolean = false,
            val constrainedVerifyMaxAccept: Int = 0,
        )
    }

    public object Embeddings {
        public enum class TaskType(internal val wireValue: String) {
            QUERY("query"),
            DOCUMENT("document"),
            SIMILARITY("similarity"),
        }

        public class Embedding(
            public val vector: FloatArray,
            public val dimension: Int = vector.size,
            public val durationMs: Long = 0,
            public val taskType: TaskType = TaskType.QUERY,
        ) {
            init {
                require(dimension == vector.size) { "dimension must match vector size" }
            }

            override fun equals(other: Any?): Boolean {
                if (this === other) return true
                if (other !is Embedding) return false
                return dimension == other.dimension &&
                    durationMs == other.durationMs &&
                    taskType == other.taskType &&
                    vector.contentEquals(other.vector)
            }

            override fun hashCode(): Int {
                var result = vector.contentHashCode()
                result = 31 * result + dimension
                result = 31 * result + durationMs.hashCode()
                result = 31 * result + taskType.hashCode()
                return result
            }

            override fun toString(): String =
                "Embedding(dimension=$dimension, durationMs=$durationMs, taskType=$taskType)"
        }

    }

    public object Tools {
        public data class RgbImage(
            val rgb: ByteArray,
            val width: Int,
            val height: Int,
            val rowStride: Int = width * 3,
        ) {
            init {
                require(width > 0 && height > 0) { "image dimensions must be positive" }
                require(rowStride >= width * 3) { "rowStride must fit RGB888 rows" }
                require(rgb.size >= rowStride * height) { "rgb buffer is too small for image dimensions" }
            }

            override fun equals(other: Any?): Boolean {
                if (this === other) return true
                if (other !is RgbImage) return false
                return width == other.width &&
                    height == other.height &&
                    rowStride == other.rowStride &&
                    rgb.contentEquals(other.rgb)
            }

            override fun hashCode(): Int {
                var result = rgb.contentHashCode()
                result = 31 * result + width
                result = 31 * result + height
                result = 31 * result + rowStride
                return result
            }
        }

        public data class VisionResult(
            val response: String,
            val inputPatches: Int,
            val validVisionTokens: Int,
            val imageTokenSlots: Int,
            val resizedWidth: Int,
            val resizedHeight: Int,
            val promptTokens: Int,
            val decodeSteps: Int,
            val firstDecodeMs: Long,
        )

    }

    // MARK: - ZephrAgentRuntime conversation facade

    public object Conversation {
        public class Engine(
            context: android.content.Context,
        ) : Closeable {
            private val runtime: Runtime = Runtime(context)

            /** Current runtime status snapshot; may update frequently during downloads. */
            public val lifecycleState: StateFlow<Lifecycle.State>
                get() = runtime.lifecycleState

            /** Current ordered lifecycle milestone list; entries are upserted by stable id. */
            public val lifecycleTimeline: StateFlow<List<Lifecycle.Event>>
                get() = runtime.lifecycleTimeline

            public val resolvedModels: Lifecycle.ResolvedModels?
                get() = runtime.resolvedModels

            public val detectedModelFamily: String?
                get() = runtime.detectedModelFamily

            public suspend fun initialize() {
                initialize(
                    configuration = Lifecycle.Configuration(
                        modelChannel = Lifecycle.ModelChannel.PUBLIC,
                        llmExecutionChoiceId = "gemma4.gpu",
                        ragEmbeddingExecutionChoiceId = "off",
                        vlmExecutionChoiceId = "off",
                        litertCompilationCacheEnabled = true,
                    ),
                )
            }

            public suspend fun initialize(configuration: Lifecycle.Configuration) {
                runtime.prepare(configuration)
            }

            public suspend fun prepare(configuration: Lifecycle.Configuration) {
                runtime.prepare(configuration)
            }

            public fun availableExecutionChoices(
                channel: Lifecycle.ModelChannel,
            ): Lifecycle.ExecutionChoices =
                runtime.availableExecutionChoices(channel)

            public fun resolvedExecutionPlan(
                configuration: Lifecycle.Configuration,
            ): Lifecycle.ResolvedExecutionPlan? =
                runtime.resolvedExecutionPlan(configuration)

            public fun createConversation(
                config: Config = Config(),
            ): Conversation =
                Conversation(runtime, config)

            public suspend fun describeImage(
                image: Tools.RgbImage,
                prompt: String = "Briefly describe this image.",
                maxTokens: Int = 0,
            ): Tools.VisionResult =
                runtime.describeImage(image, prompt, maxTokens)

            public suspend fun embedText(
                text: String,
                taskType: Embeddings.TaskType = Embeddings.TaskType.QUERY,
            ): Embeddings.Embedding =
                runtime.embedText(text, taskType)

            public suspend fun drainModelTimings(): List<Diagnostics.ModelTiming> =
                runtime.drainModelTimings()

            public suspend fun collectHeavyJson(
                prompt: String,
                collectActivations: Boolean,
                topK: Int,
            ): String =
                runtime.collectHeavyJson(prompt, collectActivations, topK)

            public suspend fun mcpTools(
                configuration: McpConfiguration,
            ): McpToolSet {
                val client = ZephrMcpClient(configuration)
                val infos = client.listTools()
                return McpToolSet(
                    providers = infos.map { info -> OpenApiToolProvider(ZephrMcpOpenApiTool(info, client)) },
                    toolNames = infos.map { it.name },
                )
            }

            public suspend fun shutdown(): Lifecycle.Event =
                runtime.shutdown()

            public suspend fun deleteDownloadedModelsAndCaches() {
                runtime.deleteDownloadedModelsAndCaches()
            }

            override fun close() {
                runtime.close()
            }
        }

        public data class Config(
            val systemInstruction: Contents? = null,
            val tools: List<ToolProvider> = emptyList(),
            val conversationStrategy: Strategy = Strategy.INCREMENTAL_KV,
            val maxTokens: Int = DEFAULT_MAX_TOKENS,
            val temperature: Float = DEFAULT_TEMPERATURE,
            val topK: Int = DEFAULT_TOP_K,
            val topP: Float = DEFAULT_TOP_P,
            val reserveOutputTokens: Int = 0,
            val maxHistoryTurns: Int = DEFAULT_MAX_HISTORY_TURNS,
            val debugPromptLoggingEnabled: Boolean = false,
            val traceSink: TraceSink? = null,
            val imageResolver: ImageResolver? = null,
        )

        public sealed interface Strategy {
            public val wireName: String

            public data object IncrementalKV : Strategy {
                override val wireName: String = "incremental_kv"
            }

            public data object FullReplay : Strategy {
                override val wireName: String = "full_replay"
            }

            public companion object {
                public val INCREMENTAL_KV: Strategy = IncrementalKV
                public val FULL_REPLAY: Strategy = FullReplay
            }
        }

        internal enum class ToolResponseContinuationMode {
            INCREMENTAL_KV,
            REPLAY_PROMPT,
        }

        internal val ToolResponseContinuationMode.conversationStrategy: Strategy
            get() = when (this) {
                ToolResponseContinuationMode.INCREMENTAL_KV -> Strategy.INCREMENTAL_KV
                ToolResponseContinuationMode.REPLAY_PROMPT -> Strategy.FULL_REPLAY
            }

        internal val Strategy.toolResponseContinuationMode: ToolResponseContinuationMode?
            get() = when (this) {
                Strategy.IncrementalKV -> ToolResponseContinuationMode.INCREMENTAL_KV
                Strategy.FullReplay -> ToolResponseContinuationMode.REPLAY_PROMPT
            }

        public data class Context(
            val promptPrefix: String? = null,
        )

        public data class SendOptions(
            val conversationStrategy: Strategy? = null,
        )

        public data class ToolCallRecord(
            val turnIndex: Int,
            val name: String,
            val arguments: Map<String, String>,
        )

        public data class ToolResponseRecord(
            val turnIndex: Int,
            val name: String,
            val response: String,
        )

        public data class ToolEffect(
            val name: String,
            val payload: Map<String, String> = emptyMap(),
        )

        public data class GenerationRecord(
            val turnIndex: Int,
            val label: String,
            val kind: String,
            val prompt: String,
            val response: String,
            val stats: Diagnostics.StageStats,
            val usedLiveKvCache: Boolean,
            val conversationStrategy: Strategy,
        )

        public data class TurnRecord(
            val conversationTurnIndex: Int,
            val conversationStrategy: Strategy,
            val userContent: Contents,
            val userText: String,
            val finalText: String,
            val generationStats: Diagnostics.GenerationStats? = null,
            val generations: List<GenerationRecord> = emptyList(),
            val toolCalls: List<ToolCallRecord> = emptyList(),
            val toolResponses: List<ToolResponseRecord> = emptyList(),
            val effects: List<ToolEffect> = emptyList(),
            val error: String? = null,
        )

        public fun interface TraceSink {
            public fun onEvent(event: TraceEvent)
        }

        public sealed interface TraceEvent {
            public val timestampMs: Long

            public data class GenerationStart(
                override val timestampMs: Long,
                val turnIndex: Int,
                val label: String,
                val kind: String,
                val prompt: String,
                val usedLiveKvCache: Boolean,
                val conversationStrategy: Strategy,
            ) : TraceEvent

            public data class GenerationResult(
                override val timestampMs: Long,
                val turnIndex: Int,
                val label: String,
                val kind: String,
                val result: Diagnostics.TextResult,
                val usedLiveKvCache: Boolean,
                val conversationStrategy: Strategy,
            ) : TraceEvent

            public data class GenerationFailure(
                override val timestampMs: Long,
                val turnIndex: Int,
                val label: String,
                val kind: String,
                val reason: String,
                val usedLiveKvCache: Boolean,
                val conversationStrategy: Strategy,
            ) : TraceEvent

            public data class IncrementalFallback(
                override val timestampMs: Long,
                val turnIndex: Int,
                val label: String,
                val reason: String,
            ) : TraceEvent

            public data class MemorySample(
                override val timestampMs: Long,
                val turnIndex: Int?,
                val label: String,
                val sample: Diagnostics.MemorySample,
            ) : TraceEvent

            public data class ToolCall(
                override val timestampMs: Long,
                val turnIndex: Int,
                val name: String,
                val arguments: Map<String, String>,
            ) : TraceEvent

            public data class ToolResponse(
                override val timestampMs: Long,
                val turnIndex: Int,
                val name: String,
                val response: String,
            ) : TraceEvent

            public data class EffectApplied(
                override val timestampMs: Long,
                val turnIndex: Int,
                val effect: ToolEffect,
            ) : TraceEvent

            public data class TurnCompleted(
                override val timestampMs: Long,
                val record: TurnRecord,
            ) : TraceEvent
        }

        public class Conversation internal constructor(
            private val runtime: Runtime,
            private val config: Config,
        ) : Closeable {
            private val turns = mutableListOf<ConversationTurn>()
            private val protocolTurns = mutableListOf<ConversationProtocolRecord>()
            @Volatile
            private var closed = false
            @Volatile
            private var cancelled = false
            private var conversationTurnIndex = 0
            private var liveKvCacheReady = false

            public fun sendMessageAsync(
                contents: Contents,
                options: SendOptions = SendOptions(),
            ): Flow<Message> = channelFlow {
                var emittedChunk = false
                val finalMessage = sendMessageInternal(
                    contents = contents,
                    context = Context(),
                    options = options,
                    onAssistantChunk = { chunk ->
                        if (chunk.isEmpty()) {
                            true
                        } else {
                            val sent = trySend(Message(Contents.of(chunk))).isSuccess
                            emittedChunk = emittedChunk || sent
                            sent && !cancelled
                        }
                    },
                )
                if (emittedChunk) {
                    trySend(finalMessage.copy(contents = Contents.of("")))
                } else {
                    trySend(finalMessage)
                }
            }.flowOn(runtime.sdkDispatcher)

            public suspend fun sendMessage(contents: Contents): Message =
                withContext(runtime.sdkDispatcher) {
                    sendMessageInternal(
                        contents,
                        context = Context(),
                        options = SendOptions(),
                        onAssistantChunk = null,
                    )
                }

            public suspend fun sendMessage(
                contents: Contents,
                context: Context,
                options: SendOptions = SendOptions(),
            ): Message =
                withContext(runtime.sdkDispatcher) {
                    sendMessageInternal(contents, context = context, options = options, onAssistantChunk = null)
                }

            private suspend fun sendMessageInternal(
                contents: Contents,
                context: Context,
                options: SendOptions,
                onAssistantChunk: ((String) -> Boolean)?,
            ): Message {
                check(!closed) { "Conversation is closed" }
                cancelled = false
                val currentConversationTurnIndex = conversationTurnIndex++
                val conversationStrategy = options.conversationStrategy ?: config.conversationStrategy
                val toolResponseContinuationMode =
                    conversationStrategy.toolResponseContinuationMode
                        ?: ToolResponseContinuationMode.INCREMENTAL_KV
                val text = contents.toString()
                var nextUserText = context.promptPrefix?.let { prefix ->
                    userTextWithPromptPrefix(text, prefix)
                } ?: text
                var recordedUser = false
                val registry = ToolRegistry(config.tools, imageInspectionTools(contents))
                var toolsEnabled = registry.hasTools
                var generationStats: Diagnostics.GenerationStats? = null
                val generationRecords = mutableListOf<GenerationRecord>()
                val toolCallRecords = mutableListOf<ToolCallRecord>()
                val toolResponseRecords = mutableListOf<ToolResponseRecord>()
                val effectRecords = mutableListOf<ToolEffect>()
                repeat(RECURRING_TOOL_CALL_LIMIT) {
                    check(!cancelled) { "Conversation generation was cancelled" }
                    val nativeTools = registry.nativeSpec()
                    val useLiveKvCache = it == 0 &&
                        toolsEnabled &&
                        toolResponseContinuationMode == ToolResponseContinuationMode.INCREMENTAL_KV &&
                        liveKvCacheReady
                    val useProtocolReplay = it == 0 &&
                        !useLiveKvCache &&
                        conversationStrategy == Strategy.FULL_REPLAY
                    val promptUserMessage =
                        if (useLiveKvCache || useProtocolReplay) nextUserText else userMessageWithHistory(nextUserText)
                    val promptSystemMessage = systemMessage(registry, toolsEnabled)
                    var resultLabel = "turn-$it"
                    var resultKind = if (toolsEnabled) "tool_aware" else "answer_only"
                    var resultUsedLiveKvCache = false
                    val result = if (useLiveKvCache) {
                        val suffix = incrementalUserTurnPrompt(promptUserMessage)
                        resultLabel = "turn-$it-incremental-kv"
                        resultKind = "incremental_tool_aware"
                        resultUsedLiveKvCache = true
                        logPrompt(resultLabel, suffix)
                        emitTrace(TraceEvent.GenerationStart(
                            timestampMs = System.currentTimeMillis(),
                            turnIndex = it,
                            label = resultLabel,
                            kind = resultKind,
                            prompt = suffix,
                            usedLiveKvCache = true,
                            conversationStrategy = conversationStrategy,
                        ))
                        emitMemorySample(it, "$resultLabel:before_generation")
                        runCatching {
                            runtime.continueToolAwareTextStreaming(
                                promptSuffix = suffix,
                                maxTokens = config.maxTokens,
                                temperature = config.temperature,
                                topK = config.topK,
                                topP = config.topP,
                                reserveOutputTokens = config.reserveOutputTokens,
                                toolNames = nativeTools.toolNames,
                                toolDescriptions = nativeTools.toolDescriptions,
                                paramToolIndexes = nativeTools.paramToolIndexes,
                                paramNames = nativeTools.paramNames,
                                paramDescriptions = nativeTools.paramDescriptions,
                                paramTypes = nativeTools.paramTypes,
                                paramRequired = nativeTools.paramRequired,
                                paramEnumValues = nativeTools.paramEnumValues,
                                onChunk = onAssistantChunk ?: { true },
                            )
                        }.getOrElse { error ->
                            emitTrace(TraceEvent.IncrementalFallback(
                                timestampMs = System.currentTimeMillis(),
                                turnIndex = it,
                                label = resultLabel,
                                reason = error.message ?: error::class.java.simpleName,
                            ))
                            liveKvCacheReady = false
                            resultLabel = "turn-$it-fallback-replay"
                            resultKind = "tool_aware_fallback_replay"
                            resultUsedLiveKvCache = false
                            emitMemorySample(it, "$resultLabel:before_generation")
                            generateToolAwareFirstPass(
                                promptUserMessage = userMessageWithHistory(nextUserText),
                                promptSystemMessage = promptSystemMessage,
                                nativeTools = nativeTools,
                                onAssistantChunk = onAssistantChunk,
                            )
                        }
                    } else if (useProtocolReplay && toolsEnabled) {
                        val prompt = protocolPrompt(
                            currentText = promptUserMessage,
                            systemMessage = promptSystemMessage,
                            includeTools = true,
                            registry = registry,
                        )
                        resultLabel = "turn-$it-full-replay"
                        resultKind = "tool_aware_full_replay"
                        logPrompt(resultLabel, prompt)
                        emitTrace(TraceEvent.GenerationStart(
                            timestampMs = System.currentTimeMillis(),
                            turnIndex = it,
                            label = resultLabel,
                            kind = resultKind,
                            prompt = prompt,
                            usedLiveKvCache = false,
                            conversationStrategy = conversationStrategy,
                        ))
                        emitMemorySample(it, "$resultLabel:before_generation")
                        runtime.generateToolAwareTextFromPromptStreaming(
                            prompt = prompt,
                            maxTokens = config.maxTokens,
                            temperature = config.temperature,
                            topK = config.topK,
                            topP = config.topP,
                            reserveOutputTokens = config.reserveOutputTokens,
                            toolNames = nativeTools.toolNames,
                            toolDescriptions = nativeTools.toolDescriptions,
                            paramToolIndexes = nativeTools.paramToolIndexes,
                            paramNames = nativeTools.paramNames,
                            paramDescriptions = nativeTools.paramDescriptions,
                            paramTypes = nativeTools.paramTypes,
                            paramRequired = nativeTools.paramRequired,
                            paramEnumValues = nativeTools.paramEnumValues,
                            onChunk = onAssistantChunk ?: { true },
                        )
                    } else if (toolsEnabled) {
                        emitMemorySample(it, "$resultLabel:before_generation")
                        generateToolAwareFirstPass(
                            promptUserMessage = promptUserMessage,
                            promptSystemMessage = promptSystemMessage,
                            nativeTools = nativeTools,
                            onAssistantChunk = onAssistantChunk,
                        )
                    } else if (useProtocolReplay) {
                        val prompt = protocolPrompt(
                            currentText = promptUserMessage,
                            systemMessage = promptSystemMessage,
                            includeTools = false,
                            registry = registry,
                        )
                        resultLabel = "turn-$it-full-replay"
                        resultKind = "answer_only_full_replay"
                        liveKvCacheReady = false
                        logPrompt(resultLabel, prompt)
                        emitTrace(TraceEvent.GenerationStart(
                            timestampMs = System.currentTimeMillis(),
                            turnIndex = it,
                            label = resultLabel,
                            kind = resultKind,
                            prompt = prompt,
                            usedLiveKvCache = false,
                            conversationStrategy = conversationStrategy,
                        ))
                        emitMemorySample(it, "$resultLabel:before_generation")
                        runtime.generateTextFromPromptStreaming(
                            prompt = prompt,
                            maxTokens = config.maxTokens,
                            temperature = config.temperature,
                            topK = config.topK,
                            topP = config.topP,
                            onChunk = onAssistantChunk ?: { true },
                        )
                    } else {
                        liveKvCacheReady = false
                        emitMemorySample(it, "$resultLabel:before_generation")
                        runtime.generateText(
                            userMessage = promptUserMessage,
                            systemMessage = promptSystemMessage,
                            maxTokens = config.maxTokens,
                            temperature = config.temperature,
                            topK = config.topK,
                            topP = config.topP,
                        )
                    }
                    logPrompt("turn-$it", result.prompt)
                    emitTrace(TraceEvent.GenerationResult(
                        timestampMs = System.currentTimeMillis(),
                        turnIndex = it,
                        label = resultLabel,
                        kind = resultKind,
                        result = result,
                        usedLiveKvCache = resultUsedLiveKvCache,
                        conversationStrategy = conversationStrategy,
                    ))
                    emitMemorySample(it, "$resultLabel:after_generation")
                    generationStats = generationStats.plus(result.toGenerationStats())
                    generationRecords += result.toGenerationRecord(
                        turnIndex = it,
                        label = resultLabel,
                        kind = resultKind,
                        usedLiveKvCache = resultUsedLiveKvCache,
                        conversationStrategy = conversationStrategy,
                    )
                    val response = result.response.trim()

                    val toolCalls = if (toolsEnabled) parseToolCalls(response) else emptyList()
                    if (toolCalls.isEmpty()) {
                        if (!recordedUser) {
                            turns += ConversationTurn.User(text)
                            protocolTurns += ConversationProtocolRecord("user", text)
                            recordedUser = true
                        }
                        turns += ConversationTurn.Assistant(response)
                        protocolTurns += ConversationProtocolRecord("model", response)
                        liveKvCacheReady = toolResponseContinuationMode == ToolResponseContinuationMode.INCREMENTAL_KV &&
                            toolsEnabled
                        val message = Message(
                            contents = Contents.of(response),
                            generationStats = generationStats,
                            isFinal = true,
                        )
                        emitTurnCompleted(
                            conversationTurnIndex = currentConversationTurnIndex,
                            conversationStrategy = conversationStrategy,
                            userContent = contents,
                            userText = text,
                            finalText = response,
                            generationStats = generationStats,
                            generations = generationRecords,
                            toolCalls = toolCallRecords,
                            toolResponses = toolResponseRecords,
                            effects = effectRecords,
                        )
                        return message
                    }

                    val toolResponses = toolCalls.map { call ->
                        emitTrace(TraceEvent.ToolCall(
                            timestampMs = System.currentTimeMillis(),
                            turnIndex = it,
                            name = call.name,
                            arguments = call.arguments,
                        ))
                        toolCallRecords += ToolCallRecord(it, call.name, call.arguments)
                        val result = registry.execute(call)
                        emitTrace(TraceEvent.ToolResponse(
                            timestampMs = System.currentTimeMillis(),
                            turnIndex = it,
                            name = call.name,
                            response = result.response,
                        ))
                        toolResponseRecords += ToolResponseRecord(it, call.name, result.response)
                        result.effects.forEach { effect ->
                            effectRecords += effect
                            emitTrace(TraceEvent.EffectApplied(
                                timestampMs = System.currentTimeMillis(),
                                turnIndex = it,
                                effect = effect,
                            ))
                        }
                        result
                    }
                    if (!recordedUser) {
                        turns += ConversationTurn.User(text)
                        protocolTurns += ConversationProtocolRecord("user", text)
                        recordedUser = true
                    }
                    toolResponses.forEach { turns += ConversationTurn.Tool(it.name, it.response) }
                    val effectiveToolResponseContinuationMode =
                        toolResponses.toolResponseContinuationMode(toolResponseContinuationMode)
                    val continued = continueAfterToolResponse(
                        firstPass = result,
                        responses = toolResponses,
                        onAssistantChunk = onAssistantChunk,
                        turnIndex = it,
                        toolResponseContinuationMode = effectiveToolResponseContinuationMode,
                        conversationStrategy = conversationStrategy,
                    )
                    if (continued != null) {
                        generationStats = generationStats.plus(continued.result.toGenerationStats())
                        generationRecords += continued.result.toGenerationRecord(
                            turnIndex = it,
                            label = continued.label,
                            kind = continued.kind,
                            usedLiveKvCache = continued.usedLiveKvCache,
                            conversationStrategy = conversationStrategy,
                        )
                        val finalResponse = continued.result.response.trim()
                        turns += ConversationTurn.Assistant(finalResponse)
                        protocolTurns += ConversationProtocolRecord(
                            "model",
                            modelProtocolBodyAfterToolResponse(result, toolResponses, finalResponse),
                        )
                        liveKvCacheReady = continued.usedLiveKvCache
                        val message = Message(
                            contents = Contents.of(finalResponse),
                            generationStats = generationStats,
                            isFinal = true,
                        )
                        emitTurnCompleted(
                            conversationTurnIndex = currentConversationTurnIndex,
                            conversationStrategy = conversationStrategy,
                            userContent = contents,
                            userText = text,
                            finalText = finalResponse,
                            generationStats = generationStats,
                            generations = generationRecords,
                            toolCalls = toolCallRecords,
                            toolResponses = toolResponseRecords,
                            effects = effectRecords,
                        )
                        return message
                    }
                    nextUserText = continuationPrompt(text, toolResponses)
                    toolsEnabled = false
                    liveKvCacheReady = false
                }

                error("Exceeded recurring tool call limit of $RECURRING_TOOL_CALL_LIMIT")
            }

            private fun emitTurnCompleted(
                conversationTurnIndex: Int,
                conversationStrategy: Strategy,
                userContent: Contents,
                userText: String,
                finalText: String,
                generationStats: Diagnostics.GenerationStats?,
                generations: List<GenerationRecord> = emptyList(),
                toolCalls: List<ToolCallRecord> = emptyList(),
                toolResponses: List<ToolResponseRecord> = emptyList(),
                effects: List<ToolEffect> = emptyList(),
                error: String? = null,
            ) {
                emitTrace(TraceEvent.TurnCompleted(
                    timestampMs = System.currentTimeMillis(),
                    record = TurnRecord(
                        conversationTurnIndex = conversationTurnIndex,
                        conversationStrategy = conversationStrategy,
                        userContent = userContent,
                        userText = userText,
                        finalText = finalText,
                        generationStats = generationStats,
                        generations = generations,
                        toolCalls = toolCalls,
                        toolResponses = toolResponses,
                        effects = effects,
                        error = error,
                    ),
                ))
            }

            private fun imageInspectionTools(contents: Contents): List<InternalTool> {
                val resolver = config.imageResolver ?: return emptyList()
                if (!contents.hasImageContent) return emptyList()
                return listOf(
                    InternalTool(
                        name = "inspect_image",
                        description = "Inspect an image attached to the current user turn and answer a focused visual question about it.",
                        params = listOf(
                            InternalToolParam(
                                name = "imageRef",
                                description = "The exact imageRef or imageFile value from the current user turn. If there is only one image, this may be omitted.",
                                type = "STRING",
                                required = false,
                            ),
                            InternalToolParam(
                                name = "prompt",
                                description = "A focused question or instruction for inspecting the image.",
                                type = "STRING",
                                required = true,
                            ),
                        ),
                        runtimePolicy = ConversationToolRuntimePolicy(
                            invalidatesLiveTextKv = true,
                            preferredContinuationMode = ToolResponseContinuationMode.REPLAY_PROMPT,
                        ),
                        execute = { args ->
                            val requestedRef = args["imageRef"]?.toString()?.trim().orEmpty()
                            val imagePart = imagePart(contents, requestedRef)
                                ?: return@InternalTool JSONObject()
                                    .put("error", "Unknown imageRef")
                                    .put("imageRef", requestedRef)
                                    .toString()
                            try {
                                val image = resolver.resolve(imagePart)
                                    ?: return@InternalTool JSONObject()
                                        .put("error", "Image resolver returned no image")
                                        .put("imageRef", imageIdentifier(imagePart))
                                        .toString()
                                val focusedPrompt = args["prompt"]?.toString()?.trim()
                                    ?.takeIf { it.isNotBlank() }
                                    ?: "Briefly describe this image."
                                val userText = imageInspectionUserText(contents)
                                val prompt = imageInspectionPrompt(
                                    userText = userText,
                                    focusedPrompt = focusedPrompt,
                                )
                                val result = runtime.describeImage(
                                    image,
                                    prompt,
                                    0,
                                )
                                JSONObject()
                                    .put("imageRef", imageIdentifier(imagePart))
                                    .put("prompt", prompt)
                                    .put("userText", userText)
                                    .put("focusedPrompt", focusedPrompt)
                                    .put("response", result.response)
                                    .put("inputPatches", result.inputPatches)
                                    .put("validVisionTokens", result.validVisionTokens)
                                    .put("imageTokenSlots", result.imageTokenSlots)
                                    .put("resizedWidth", result.resizedWidth)
                                    .put("resizedHeight", result.resizedHeight)
                                    .put("promptTokens", result.promptTokens)
                                    .put("decodeSteps", result.decodeSteps)
                                    .put("firstDecodeMs", result.firstDecodeMs)
                                    .toString()
                            } catch (error: Throwable) {
                                JSONObject()
                                    .put("error", error.message ?: error.toString())
                                    .put("imageRef", imageIdentifier(imagePart))
                                    .toString()
                            }
                        },
                    ),
                )
            }

            private fun imagePart(contents: Contents, requestedRef: String): ContentPart? {
                val imageParts = contents.imageParts
                if (requestedRef.isBlank() && imageParts.size == 1) return imageParts.first()
                return imageParts.firstOrNull { imageIdentifier(it) == requestedRef }
            }

            private fun imageIdentifier(part: ContentPart): String =
                when (part.kind) {
                    ContentPart.Kind.TEXT -> ""
                    ContentPart.Kind.IMAGE_REF -> part.imageRef.orEmpty()
                    ContentPart.Kind.IMAGE_FILE -> part.imageFile.orEmpty()
                }

            private fun imageInspectionUserText(contents: Contents): String =
                contents.text.trim().ifBlank { contents.toString() }

            private fun imageInspectionPrompt(userText: String, focusedPrompt: String): String = """
                Original user request:
                $userText

                Focused visual question:
                $focusedPrompt

                Inspect only the image. Return concise visual facts relevant to the focused question.
            """.trimIndent()

            public fun cancelProcess() {
                cancelled = true
            }

            override fun close() {
                closed = true
            }

            private fun systemMessage(registry: ToolRegistry, toolsEnabled: Boolean): String = buildString {
                config.systemInstruction?.toString()?.takeIf { it.isNotBlank() }?.let {
                    append(it.trim())
                    append("\n")
                }
                if (toolsEnabled) {
                    append(
                        """
                        You have access to the enabled tools declared below.
                        Use a single tool call when local place, visual, or navigation state is needed.
                        """.trimIndent()
                    )
                    append("\n")
                } else if (registry.hasTools) {
                    append("Use the provided tool results to answer directly. Do not call tools or write Gemma 4 tool-call syntax.")
                    append("\n")
                }
            }.trim()

            private fun userMessageWithHistory(currentText: String): String {
                val history = historySlice()
                if (history.isEmpty()) return currentText
                return buildString {
                    append("\nConversation history:\n")
                    history.forEach { turn ->
                        when (turn) {
                            is ConversationTurn.User -> append("User: ${turn.text}\n")
                            is ConversationTurn.Assistant -> append("Assistant: ${turn.text}\n")
                            is ConversationTurn.Tool -> append("Tool ${turn.name}: ${turn.text}\n")
                        }
                    }
                    append("\nCurrent user request:\n")
                    append(currentText)
                }
            }

            private fun userTextWithPromptPrefix(text: String, promptPrefix: String): String =
                buildString {
                    append(promptPrefix.trim())
                    append("\n\nUser request:\n")
                    append(text)
                }

            private fun historySlice(): List<ConversationTurn> {
                val maxHistoryTurns = config.maxHistoryTurns.coerceAtLeast(0)
                if (maxHistoryTurns == 0) return emptyList()
                var userTurns = 0
                for (index in turns.indices.reversed()) {
                    if (turns[index] is ConversationTurn.User) {
                        userTurns += 1
                        if (userTurns >= maxHistoryTurns) {
                            return turns.drop(index)
                        }
                    }
                }
                return turns.toList()
            }

            private fun protocolHistorySlice(): List<ConversationProtocolRecord> {
                val maxHistoryTurns = config.maxHistoryTurns.coerceAtLeast(0)
                if (maxHistoryTurns == 0) return emptyList()
                var userTurns = 0
                for (index in protocolTurns.indices.reversed()) {
                    if (protocolTurns[index].role == "user") {
                        userTurns += 1
                        if (userTurns >= maxHistoryTurns) {
                            return protocolTurns.drop(index)
                        }
                    }
                }
                return protocolTurns.toList()
            }

            private fun protocolPrompt(
                currentText: String,
                systemMessage: String,
                includeTools: Boolean,
                registry: ToolRegistry,
            ): String = buildString {
                append("<bos>")
                if (systemMessage.isNotBlank() || (includeTools && registry.hasTools)) {
                    append("<|turn>system\n")
                    append(systemMessage)
                    if (includeTools && registry.hasTools) {
                        append("<|tool>")
                        registry.declarations.forEach(::append)
                        append("<tool|>")
                    }
                    append("<turn|>\n")
                }
                protocolHistorySlice().forEach { turn ->
                    append("<|turn>")
                    append(turn.role)
                    append("\n")
                    append(turn.body)
                    append("<turn|>\n")
                }
                append("<|turn>user\n")
                append(currentText)
                append("<turn|>\n<|turn>model\n")
            }

            private fun continuationPrompt(
                originalUserText: String,
                responses: List<ToolResponse>,
            ): String = buildString {
                append("Original user request:\n")
                append(originalUserText)
                append("\n\nTool results:\n")
                responses.forEach { response ->
                    append(response.name)
                    append(": ")
                    append(response.response)
                    append("\n")
                }
                append("\nUse the tool results to answer the original request concisely.")
            }

            private suspend fun generateToolAwareFirstPass(
                promptUserMessage: String,
                promptSystemMessage: String,
                nativeTools: NativeToolSpec,
                onAssistantChunk: ((String) -> Boolean)?,
            ): Diagnostics.TextResult =
                if (onAssistantChunk != null) {
                    runtime.generateToolAwareTextStreaming(
                        userMessage = promptUserMessage,
                        systemMessage = promptSystemMessage,
                        maxTokens = config.maxTokens,
                        temperature = config.temperature,
                        topK = config.topK,
                        topP = config.topP,
                        reserveOutputTokens = config.reserveOutputTokens,
                        toolNames = nativeTools.toolNames,
                        toolDescriptions = nativeTools.toolDescriptions,
                        paramToolIndexes = nativeTools.paramToolIndexes,
                        paramNames = nativeTools.paramNames,
                        paramDescriptions = nativeTools.paramDescriptions,
                        paramTypes = nativeTools.paramTypes,
                        paramRequired = nativeTools.paramRequired,
                        paramEnumValues = nativeTools.paramEnumValues,
                        onChunk = onAssistantChunk,
                    )
                } else {
                    runtime.generateToolAwareText(
                        userMessage = promptUserMessage,
                        systemMessage = promptSystemMessage,
                        maxTokens = config.maxTokens,
                        temperature = config.temperature,
                        topK = config.topK,
                        topP = config.topP,
                        reserveOutputTokens = config.reserveOutputTokens,
                        toolNames = nativeTools.toolNames,
                        toolDescriptions = nativeTools.toolDescriptions,
                        paramToolIndexes = nativeTools.paramToolIndexes,
                        paramNames = nativeTools.paramNames,
                        paramDescriptions = nativeTools.paramDescriptions,
                        paramTypes = nativeTools.paramTypes,
                        paramRequired = nativeTools.paramRequired,
                        paramEnumValues = nativeTools.paramEnumValues,
                    )
                }

            private fun incrementalUserTurnPrompt(userMessage: String): String = buildString {
                append("<turn|>\n<|turn>user\n")
                append(userMessage)
                append("<turn|>\n<|turn>model\n")
            }

            private data class ToolResponseContinuationResult(
                val result: Diagnostics.TextResult,
                val label: String,
                val kind: String,
                val usedLiveKvCache: Boolean,
            )

            private suspend fun continueAfterToolResponse(
                firstPass: Diagnostics.TextResult,
                responses: List<ToolResponse>,
                onAssistantChunk: ((String) -> Boolean)?,
                turnIndex: Int,
                toolResponseContinuationMode: ToolResponseContinuationMode,
                conversationStrategy: Strategy,
            ): ToolResponseContinuationResult? {
                if (responses.isEmpty()) return null
                val toolResponseText = responses.joinToString("") { response ->
                    formatToolResponse(response)
                }
                val usedLiveKvCache = toolResponseContinuationMode == ToolResponseContinuationMode.INCREMENTAL_KV
                val label = if (usedLiveKvCache) {
                    "turn-tool-response-incremental-kv"
                } else {
                    "turn-tool-response-replay"
                }
                val kind = if (usedLiveKvCache) {
                    "tool_response_incremental_kv"
                } else {
                    "tool_response_replay"
                }
                val prompt = when (toolResponseContinuationMode) {
                    ToolResponseContinuationMode.INCREMENTAL_KV -> incrementalToolResponsePrompt(toolResponseText)
                    ToolResponseContinuationMode.REPLAY_PROMPT -> replayPromptAfterToolResponse(firstPass, toolResponseText)
                }
                logPrompt(label, prompt)
                emitTrace(TraceEvent.GenerationStart(
                    timestampMs = System.currentTimeMillis(),
                    turnIndex = turnIndex,
                    label = label,
                    kind = kind,
                    prompt = prompt,
                    usedLiveKvCache = usedLiveKvCache,
                    conversationStrategy = conversationStrategy,
                ))
                emitMemorySample(turnIndex, "$label:before_generation")
                return try {
                    val result = when (toolResponseContinuationMode) {
                        ToolResponseContinuationMode.INCREMENTAL_KV -> runtime.continueAfterToolResponseStreaming(
                            toolResponse = toolResponseText,
                            maxTokens = config.maxTokens,
                            temperature = config.temperature,
                            topK = config.topK,
                            topP = config.topP,
                            reserveOutputTokens = config.reserveOutputTokens,
                            onChunk = onAssistantChunk ?: { true },
                        )

                        ToolResponseContinuationMode.REPLAY_PROMPT -> runtime.generateTextFromPromptStreaming(
                            prompt = prompt,
                            maxTokens = config.maxTokens,
                            temperature = config.temperature,
                            topK = config.topK,
                            topP = config.topP,
                            onChunk = onAssistantChunk ?: { true },
                        )
                    }
                    emitTrace(TraceEvent.GenerationResult(
                        timestampMs = System.currentTimeMillis(),
                        turnIndex = turnIndex,
                        label = label,
                        kind = kind,
                        result = result,
                        usedLiveKvCache = usedLiveKvCache,
                        conversationStrategy = conversationStrategy,
                    ))
                    emitMemorySample(turnIndex, "$label:after_generation")
                    result.takeIf { it.response.isNotBlank() }?.let {
                        ToolResponseContinuationResult(
                            result = it,
                            label = label,
                            kind = kind,
                            usedLiveKvCache = usedLiveKvCache,
                        )
                    }
                } catch (error: Throwable) {
                    if (toolResponseContinuationMode == ToolResponseContinuationMode.INCREMENTAL_KV) {
                        emitTrace(TraceEvent.IncrementalFallback(
                            timestampMs = System.currentTimeMillis(),
                            turnIndex = turnIndex,
                            label = label,
                            reason = error.message ?: error::class.java.simpleName,
                        ))
                        val fallbackLabel = "turn-tool-response-fallback-replay"
                        val fallbackKind = "tool_response_fallback_replay"
                        val fallbackPrompt = replayPromptAfterToolResponse(firstPass, toolResponseText)
                        logPrompt(fallbackLabel, fallbackPrompt)
                        emitTrace(TraceEvent.GenerationStart(
                            timestampMs = System.currentTimeMillis(),
                            turnIndex = turnIndex,
                            label = fallbackLabel,
                            kind = fallbackKind,
                            prompt = fallbackPrompt,
                            usedLiveKvCache = false,
                            conversationStrategy = conversationStrategy,
                        ))
                        emitMemorySample(turnIndex, "$fallbackLabel:before_generation")
                        try {
                            val result = runtime.generateTextFromPromptStreaming(
                                prompt = fallbackPrompt,
                                maxTokens = config.maxTokens,
                                temperature = config.temperature,
                                topK = config.topK,
                                topP = config.topP,
                                onChunk = onAssistantChunk ?: { true },
                            )
                            emitTrace(TraceEvent.GenerationResult(
                                timestampMs = System.currentTimeMillis(),
                                turnIndex = turnIndex,
                                label = fallbackLabel,
                                kind = fallbackKind,
                                result = result,
                                usedLiveKvCache = false,
                                conversationStrategy = conversationStrategy,
                            ))
                            emitMemorySample(turnIndex, "$fallbackLabel:after_generation")
                            result.takeIf { it.response.isNotBlank() }?.let {
                                ToolResponseContinuationResult(
                                    result = it,
                                    label = fallbackLabel,
                                    kind = fallbackKind,
                                    usedLiveKvCache = false,
                                )
                            }
                        } catch (fallbackError: Throwable) {
                            emitTrace(TraceEvent.GenerationFailure(
                                timestampMs = System.currentTimeMillis(),
                                turnIndex = turnIndex,
                                label = fallbackLabel,
                                kind = fallbackKind,
                                reason = fallbackError.message ?: fallbackError::class.java.simpleName,
                                usedLiveKvCache = false,
                                conversationStrategy = conversationStrategy,
                            ))
                            throw fallbackError
                        }
                    } else {
                        emitTrace(TraceEvent.GenerationFailure(
                            timestampMs = System.currentTimeMillis(),
                            turnIndex = turnIndex,
                            label = label,
                            kind = kind,
                            reason = error.message ?: error::class.java.simpleName,
                            usedLiveKvCache = usedLiveKvCache,
                            conversationStrategy = conversationStrategy,
                        ))
                        throw error
                    }
                }
            }

            private fun incrementalToolResponsePrompt(toolResponseText: String): String =
                "<|tool_response>${toolResponseText}<tool_response|>"

            private fun replayPromptAfterToolResponse(
                firstPass: Diagnostics.TextResult,
                toolResponseText: String,
            ): String = buildString {
                append(firstPass.prompt)
                append(firstPass.response)
                if (!firstPass.response.contains("<|tool_response>")) {
                    append("<|tool_response>")
                }
                append(toolResponseText)
                append("<tool_response|>")
            }

            private fun modelProtocolBodyAfterToolResponse(
                firstPass: Diagnostics.TextResult,
                responses: List<ToolResponse>,
                finalResponse: String,
            ): String = buildString {
                append(firstPass.response)
                if (!firstPass.response.contains("<|tool_response>")) {
                    append("<|tool_response>")
                }
                responses.forEach { append(formatToolResponse(it)) }
                append("<tool_response|>")
                append(finalResponse)
            }

            private fun formatToolResponse(response: ToolResponse): String {
                val text = response.response.trim()
                return buildString {
                    append("response:")
                    append(response.name)
                    if (text.startsWith("{") && text.endsWith("}")) {
                        append(text)
                    } else {
                        append("{result:<|\"|>")
                        append(escapeToolResponseText(text))
                        append("<|\"|>}")
                    }
                }
            }

            private fun escapeToolResponseText(text: String): String = buildString {
                text.forEach { char ->
                    when (char) {
                        '\\' -> append("\\\\")
                        '"' -> append("\\\"")
                        '\n' -> append("\\n")
                        '\r' -> Unit
                        '\t' -> append("\\t")
                        else -> append(char)
                    }
                }
            }

            private fun logPrompt(label: String, prompt: String) {
                if (!config.debugPromptLoggingEnabled) return
                Log.d(PROMPT_LOG_TAG, "BEGIN $label chars=${prompt.length}")
                if (prompt.isEmpty()) {
                    Log.d(PROMPT_LOG_TAG, "END $label")
                    return
                }
                prompt.chunked(PROMPT_LOG_CHUNK_SIZE).forEachIndexed { index, chunk ->
                    Log.d(PROMPT_LOG_TAG, "$label part=${index + 1}: $chunk")
                }
                Log.d(PROMPT_LOG_TAG, "END $label")
            }

            private fun emitTrace(event: TraceEvent) {
                val sink = config.traceSink ?: return
                runCatching {
                    sink.onEvent(event)
                }.onFailure { error ->
                    Log.w(PROMPT_LOG_TAG, "Conversation trace sink failed", error)
                }
            }

            private fun emitMemorySample(turnIndex: Int?, label: String) {
                emitTrace(TraceEvent.MemorySample(
                    timestampMs = System.currentTimeMillis(),
                    turnIndex = turnIndex,
                    label = label,
                    sample = Diagnostics.MemorySample.capture(label),
                ))
            }

            private fun Diagnostics.TextResult.toGenerationStats(): Diagnostics.GenerationStats =
                Diagnostics.GenerationStats(
                    prefillTokens = prefillTokens,
                    decodeTokens = stats.outputTokens,
                    prefillMs = stats.prefillMs,
                    decodeMs = stats.decodeMs,
                    firstDecodeMs = stats.firstDecodeMs,
                )

            private fun Diagnostics.TextResult.toGenerationRecord(
                turnIndex: Int,
                label: String,
                kind: String,
                usedLiveKvCache: Boolean,
                conversationStrategy: Strategy,
            ): GenerationRecord =
                GenerationRecord(
                    turnIndex = turnIndex,
                    label = label,
                    kind = kind,
                    prompt = prompt,
                    response = response,
                    stats = stats,
                    usedLiveKvCache = usedLiveKvCache,
                    conversationStrategy = conversationStrategy,
                )

            private fun List<ToolResponse>.toolResponseContinuationMode(
                defaultMode: ToolResponseContinuationMode,
            ): ToolResponseContinuationMode {
                firstNotNullOfOrNull { it.runtimePolicy.preferredContinuationMode }?.let { return it }
                return if (any { it.runtimePolicy.invalidatesLiveTextKv }) {
                    ToolResponseContinuationMode.REPLAY_PROMPT
                } else {
                    defaultMode
                }
            }

            private fun Diagnostics.GenerationStats?.plus(other: Diagnostics.GenerationStats): Diagnostics.GenerationStats =
                this?.plus(other) ?: other
        }

        public data class Message(
            val contents: Contents,
            val generationStats: Diagnostics.GenerationStats? = null,
            val isFinal: Boolean = false,
        )

        public data class ContentPart(
            val kind: Kind,
            val text: String? = null,
            val imageRef: String? = null,
            val imageFile: String? = null,
        ) {
            public enum class Kind {
                TEXT,
                IMAGE_REF,
                IMAGE_FILE,
            }

            public companion object {
                public fun text(text: String): ContentPart =
                    ContentPart(kind = Kind.TEXT, text = text)

                public fun imageRef(ref: String): ContentPart =
                    ContentPart(kind = Kind.IMAGE_REF, imageRef = ref)

                public fun imageFile(path: String): ContentPart =
                    ContentPart(kind = Kind.IMAGE_FILE, imageFile = path)
            }
        }

        public class Contents private constructor(
            public val parts: List<ContentPart>,
        ) {
            public val text: String
                get() = parts
                    .filter { it.kind == ContentPart.Kind.TEXT }
                    .joinToString(separator = "\n") { it.text.orEmpty() }

            public val imageParts: List<ContentPart>
                get() = parts.filter {
                    it.kind == ContentPart.Kind.IMAGE_REF || it.kind == ContentPart.Kind.IMAGE_FILE
                }

            public val hasImageContent: Boolean
                get() = imageParts.isNotEmpty()

            override fun toString(): String =
                parts.joinToString(separator = "\n") { part ->
                    when (part.kind) {
                        ContentPart.Kind.TEXT -> part.text.orEmpty()
                        ContentPart.Kind.IMAGE_REF -> "[imageRef:${part.imageRef.orEmpty()}]"
                        ContentPart.Kind.IMAGE_FILE -> "[imageFile:${part.imageFile.orEmpty()}]"
                    }
                }

            public companion object {
                public fun of(text: String): Contents = Contents(listOf(ContentPart.text(text)))

                public fun of(parts: List<ContentPart>): Contents = Contents(parts)
            }
        }

        public fun interface ImageResolver {
            public suspend fun resolve(part: ContentPart): Tools.RgbImage?
        }

        public interface ToolSet

        @Target(AnnotationTarget.FUNCTION)
        @Retention(AnnotationRetention.RUNTIME)
        public annotation class Tool(
            val description: String = "",
        )

        @Target(AnnotationTarget.VALUE_PARAMETER)
        @Retention(AnnotationRetention.RUNTIME)
        public annotation class ToolParam(
            val description: String = "",
        )

        public interface OpenApiTool {
            public fun getToolDescriptionJsonString(): String
            public fun execute(params: String): String
            public fun appliedEffects(params: String, response: String): List<ToolEffect> =
                emptyList()
        }

        public interface ToolProvider

        public data class McpConfiguration(
            val endpoint: String,
            val apiKey: String,
            val clientName: String = "zephr-agent-runtime",
            val clientVersion: String = "1.0.0",
        )

        public data class McpToolSet(
            val providers: List<ToolProvider>,
            val toolNames: List<String>,
        )

        public fun tool(toolSet: ToolSet): ToolProvider =
            ReflectionToolProvider(toolSet)

        public fun tool(openApiTool: OpenApiTool): ToolProvider =
            OpenApiToolProvider(openApiTool)

        private const val RECURRING_TOOL_CALL_LIMIT = 25
        private const val DEFAULT_MAX_TOKENS = 512
        private const val DEFAULT_TEMPERATURE = 0.0f
        private const val DEFAULT_TOP_K = 40
        private const val DEFAULT_TOP_P = 0.95f
        private const val DEFAULT_MAX_HISTORY_TURNS = 6
        private const val MCP_CONNECT_MAX_ATTEMPTS = 4
        private const val MCP_CONNECT_RETRY_DELAY_MS = 2_000L
        public const val ZEPHR_POINT_ACTION_TOOL_NAME: String = "zephr_point_action"

        private interface InternalToolProvider : ToolProvider {
            fun tools(): List<InternalTool>
        }

        private data class InternalTool(
            val name: String,
            val description: String,
            val params: List<InternalToolParam>,
            val runtimePolicy: ConversationToolRuntimePolicy = ConversationToolRuntimePolicy(),
            val execute: suspend (Map<String, Any?>) -> String,
            val appliedEffects: (Map<String, Any?>, String) -> List<ToolEffect> = { _, _ -> emptyList() },
        ) {
            val declaration: String
                get() = formatToolDeclaration(this)
        }

        private data class InternalToolParam(
            val name: String,
            val description: String,
            val type: String,
            val enumValues: List<String> = emptyList(),
            val required: Boolean = true,
        )

        private data class ConversationToolRuntimePolicy(
            val invalidatesLiveTextKv: Boolean = false,
            val preferredContinuationMode: ToolResponseContinuationMode? = null,
            val maxResponseTokensForContinuation: Int? = null,
            val recordInConversationHistory: Boolean = true,
        )

        private data class ParsedToolCall(
            val name: String,
            val arguments: Map<String, String>,
        )

        private data class ToolResponse(
            val name: String,
            val response: String,
            val effects: List<ToolEffect> = emptyList(),
            val runtimePolicy: ConversationToolRuntimePolicy = ConversationToolRuntimePolicy(),
        )

        private data class ConversationProtocolRecord(
            val role: String,
            val body: String,
        )

        private data class NativeToolSpec(
            val toolNames: Array<String>,
            val toolDescriptions: Array<String>,
            val paramToolIndexes: IntArray,
            val paramNames: Array<String>,
            val paramDescriptions: Array<String>,
            val paramTypes: Array<String>,
            val paramRequired: BooleanArray,
            val paramEnumValues: Array<String>,
        )

        private sealed interface ConversationTurn {
            data class User(val text: String) : ConversationTurn
            data class Assistant(val text: String) : ConversationTurn
            data class Tool(val name: String, val text: String) : ConversationTurn
        }

        private class ToolRegistry(
            providers: List<ToolProvider>,
            extraTools: List<InternalTool> = emptyList(),
        ) {
            private val toolsByName: Map<String, InternalTool> =
                (providers.flatMap { provider ->
                    (provider as? InternalToolProvider)?.tools().orEmpty()
                } + extraTools).associateBy { it.name }

            val hasTools: Boolean
                get() = toolsByName.isNotEmpty()

            val declarations: List<String> =
                toolsByName.values.map { it.declaration }

            fun nativeSpec(): NativeToolSpec {
                val tools = toolsByName.values.toList()
                val paramToolIndexes = mutableListOf<Int>()
                val paramNames = mutableListOf<String>()
                val paramDescriptions = mutableListOf<String>()
                val paramTypes = mutableListOf<String>()
                val paramRequired = mutableListOf<Boolean>()
                val paramEnumValues = mutableListOf<String>()

                tools.forEachIndexed { toolIndex, tool ->
                    tool.params.forEach { param ->
                        paramToolIndexes += toolIndex
                        paramNames += param.name
                        paramDescriptions += param.description
                        paramTypes += param.type
                        paramRequired += param.required
                        paramEnumValues += param.enumValues.joinToString(ENUM_VALUE_SEPARATOR)
                    }
                }

                return NativeToolSpec(
                    toolNames = tools.map { it.name }.toTypedArray(),
                    toolDescriptions = tools.map { it.description }.toTypedArray(),
                    paramToolIndexes = paramToolIndexes.toIntArray(),
                    paramNames = paramNames.toTypedArray(),
                    paramDescriptions = paramDescriptions.toTypedArray(),
                    paramTypes = paramTypes.toTypedArray(),
                    paramRequired = paramRequired.toBooleanArray(),
                    paramEnumValues = paramEnumValues.toTypedArray(),
                )
            }

            suspend fun execute(call: ParsedToolCall): ToolResponse =
                withContext(Dispatchers.IO) {
                    val tool = toolsByName[call.name]
                        ?: return@withContext ToolResponse(
                            call.name,
                            """{"error":"Unknown tool: ${jsonEscape(call.name)}"}""",
                        )
                    try {
                        val response = tool.execute(call.arguments)
                        ToolResponse(
                            name = call.name,
                            response = response,
                            effects = tool.appliedEffects(call.arguments, response),
                            runtimePolicy = tool.runtimePolicy,
                        )
                    } catch (error: Throwable) {
                        ToolResponse(
                            call.name,
                            """{"error":"${jsonEscape(error.message ?: error.toString())}"}""",
                        )
                    }
                }
        }

        private data class ZephrMcpToolInfo(
            val name: String,
            val description: String,
            val properties: JsonObject?,
            val required: List<String>?,
        )

        private class ZephrMcpOpenApiTool(
            private val toolInfo: ZephrMcpToolInfo,
            private val client: ZephrMcpClient,
        ) : OpenApiTool {
            override fun getToolDescriptionJsonString(): String {
                val description = JSONObject().apply {
                    put("name", toolInfo.name)
                    put("description", toolInfo.description)
                    put("parameters", JSONObject().apply {
                        put("type", "object")
                        put("properties", toolInfo.properties?.let { JSONObject(it.toString()) } ?: JSONObject())
                        toolInfo.required?.let { put("required", JSONArray(it)) }
                    })
                }
                return description.toString()
            }

            override fun execute(params: String): String {
                val paramsJson = JSONObject(params)
                val arguments = mutableMapOf<String, Any?>()
                paramsJson.keys().forEach { key ->
                    arguments[key] = if (paramsJson.isNull(key)) null else paramsJson.get(key)
                }
                return runBlocking {
                    client.callTool(toolInfo.name, arguments)
                }
            }
        }

        private class ZephrMcpClient(
            private val configuration: McpConfiguration,
        ) {
            private var client: Client? = null

            suspend fun listTools(): List<ZephrMcpToolInfo> = withContext(Dispatchers.IO) {
                ensureConnected()
                val mcpClient = client ?: error("MCP not connected")
                val result = mcpClient.listTools()
                result.tools.map { tool ->
                    ZephrMcpToolInfo(
                        name = tool.name,
                        description = tool.description ?: "",
                        properties = tool.inputSchema.properties,
                        required = tool.inputSchema.required,
                    )
                }
            }

            suspend fun callTool(toolName: String, arguments: Map<String, Any?>): String =
                withContext(Dispatchers.IO) {
                    try {
                        ensureConnected()
                        val mcpClient = client ?: return@withContext """{"error":"MCP not connected"}"""
                        val result = mcpClient.callTool(
                            name = toolName,
                            arguments = arguments,
                        )
                        if (result.isError == true) {
                            val errorText = result.content
                                .filterIsInstance<TextContent>()
                                .joinToString("\n") { it.text }
                            return@withContext JSONObject(mapOf("error" to errorText)).toString()
                        }
                        result.content
                            .filterIsInstance<TextContent>()
                            .joinToString("\n") { it.text }
                    } catch (error: Throwable) {
                        JSONObject(mapOf("error" to (error.message ?: error.toString()))).toString()
                    }
                }

            private suspend fun ensureConnected() {
                if (client != null) {
                    return
                }
                val trimmedKey = configuration.apiKey.trim()
                require(trimmedKey.isNotEmpty()) { "MCP API key is not configured" }
                var lastError: Throwable? = null
                repeat(MCP_CONNECT_MAX_ATTEMPTS) { attemptIndex ->
                    val httpClient = HttpClient(OkHttp)
                    try {
                        val transport = StreamableHttpClientTransport(
                            client = httpClient,
                            url = configuration.endpoint,
                            requestBuilder = {
                                headers.append("x-api-key", trimmedKey)
                            },
                        )
                        val mcpClient = Client(
                            clientInfo = Implementation(
                                name = configuration.clientName,
                                version = configuration.clientVersion,
                            ),
                        )
                        mcpClient.connect(transport)
                        client = mcpClient
                        return
                    } catch (error: Throwable) {
                        if (error is CancellationException) {
                            throw error
                        }
                        lastError = error
                        httpClient.close()
                        if (attemptIndex < MCP_CONNECT_MAX_ATTEMPTS - 1) {
                            delay(MCP_CONNECT_RETRY_DELAY_MS)
                        }
                    }
                }
                throw lastError ?: IllegalStateException("MCP connection failed")
            }
        }

        private class OpenApiToolProvider(
            private val tool: OpenApiTool,
        ) : InternalToolProvider {
            override fun tools(): List<InternalTool> {
                val description = JSONObject(tool.getToolDescriptionJsonString())
                val parameters = description.optJSONObject("parameters") ?: JSONObject()
                val properties = parameters.optJSONObject("properties") ?: JSONObject()
                val required = parameters.optJSONArray("required")
                    ?.strings()
                    ?.toSet()
                    .orEmpty()
                val params = properties.objectEntries().map { (name, schema) ->
                    InternalToolParam(
                        name = name,
                        description = schema.optString("description"),
                        type = schema.optString("type").toGemmaType(),
                        enumValues = schema.optJSONArray("enum").orEmptyStrings(),
                        required = required.isEmpty() || name in required,
                    )
                }

                return listOf(
                    InternalTool(
                        name = description.getString("name"),
                        description = description.optString("description"),
                        params = params,
                        execute = { arguments -> tool.execute(arguments.toJSONObjectString()) },
                        appliedEffects = { arguments, response ->
                            tool.appliedEffects(arguments.toJSONObjectString(), response)
                        },
                    )
                )
            }
        }

        private class ReflectionToolProvider(
            private val toolSet: ToolSet,
        ) : InternalToolProvider {
            override fun tools(): List<InternalTool> =
                toolSet::class.memberFunctions.mapNotNull { function ->
                    val annotation = function.findAnnotation<Tool>() ?: return@mapNotNull null
                    val params = function.parameters
                        .filter { it.kind == KParameter.Kind.VALUE }
                        .map { parameter ->
                            InternalToolParam(
                                name = parameter.name ?: error("Tool parameter names must be available"),
                                description = parameter.findAnnotation<ToolParam>()?.description.orEmpty(),
                                type = parameter.type.toGemmaType(),
                                required = !parameter.isOptional && !parameter.type.isMarkedNullable,
                            )
                        }
                    InternalTool(
                        name = function.name,
                        description = annotation.description,
                        params = params,
                        execute = { arguments ->
                            val result = function.callTool(toolSet, arguments)
                            result?.toString().orEmpty()
                        },
                    )
                }
        }

        private fun Any?.asNullableDouble(): Double? =
            when (this) {
                null -> null
                is Number -> toDouble()
                is String -> toDoubleOrNull()
                else -> null
            }

        private fun KFunction<*>.callTool(
            instance: Any,
            arguments: Map<String, Any?>,
        ): Any? {
            val callArgs = mutableMapOf<KParameter, Any?>()
            parameters.forEach { parameter ->
                when (parameter.kind) {
                    KParameter.Kind.INSTANCE -> callArgs[parameter] = instance
                    KParameter.Kind.VALUE -> {
                        val name = parameter.name ?: return@forEach
                        if (arguments.containsKey(name)) {
                            callArgs[parameter] = arguments[name].convertTo(parameter.type)
                        } else if (!parameter.isOptional && !parameter.type.isMarkedNullable) {
                            error("Missing required tool parameter: $name")
                        }
                    }
                    KParameter.Kind.EXTENSION_RECEIVER -> Unit
                    else -> Unit
                }
            }
            return callBy(callArgs)
        }

        private fun Any?.convertTo(type: KType): Any? {
            if (this == null) return null
            val classifier = type.classifier as? KClass<*> ?: return this
            return when (classifier) {
                String::class -> toString()
                Double::class -> asNumber().toDouble()
                Float::class -> asNumber().toFloat()
                Int::class -> asNumber().toInt()
                Long::class -> asNumber().toLong()
                Boolean::class -> when (this) {
                    is Boolean -> this
                    is String -> equals("true", ignoreCase = true)
                    else -> asNumber().toInt() != 0
                }
                else -> this
            }
        }

        private fun Any.asNumber(): Number =
            when (this) {
                is Number -> this
                is String -> toDouble()
                else -> error("Value is not numeric: $this")
            }

        private fun KType.toGemmaType(): String {
            val classifier = classifier as? KClass<*> ?: return "STRING"
            return when (classifier) {
                String::class -> "STRING"
                Int::class,
                Long::class -> "INTEGER"
                Float::class,
                Double::class -> "NUMBER"
                Boolean::class -> "BOOLEAN"
                else -> "STRING"
            }
        }

        private fun String.toGemmaType(): String =
            when (lowercase()) {
                "integer", "int", "long" -> "INTEGER"
                "number", "float", "double" -> "NUMBER"
                "boolean", "bool" -> "BOOLEAN"
                else -> "STRING"
            }

        private fun formatToolDeclaration(tool: InternalTool): String {
            val params = tool.params.sortedBy { it.name }
            val properties = params.joinToString(",") { param ->
                buildString {
                    append(param.name)
                    append(":{description:<|\"|>")
                    append(toolEscape(param.description))
                    append("<|\"|>,type:<|\"|>")
                    append(param.type)
                    append("<|\"|>")
                    if (param.enumValues.isNotEmpty()) {
                        append(",enum:[")
                        append(param.enumValues.joinToString(",") { "<|\"|>${toolEscape(it)}<|\"|>" })
                        append("]")
                    }
                    append("}")
                }
            }
            val required = params
                .filter { it.required }
                .joinToString(",") { "<|\"|>${toolEscape(it.name)}<|\"|>" }
            return "declaration:${tool.name}{" +
                "description:<|\"|>${toolEscape(tool.description)}<|\"|>," +
                "parameters:{properties:{$properties}," +
                "required:[$required]," +
                "type:<|\"|>OBJECT<|\"|>}}"
        }

        private fun parseToolCalls(response: String): List<ParsedToolCall> {
            val regex = Regex(
                """<\|tool_call>call:([A-Za-z0-9_.-]+)\{(.*?)\}<tool_call\|>""",
                setOf(RegexOption.DOT_MATCHES_ALL),
            )
            return regex.findAll(response).map { match ->
                ParsedToolCall(
                    name = match.groupValues[1],
                    arguments = parseToolArguments(match.groupValues[2]),
                )
            }.toList()
        }

        private fun parseToolArguments(raw: String): Map<String, String> {
            val regex = Regex(
                """([A-Za-z0-9_.-]+):<\|"\|>(.*?)<\|"\|>""",
                setOf(RegexOption.DOT_MATCHES_ALL),
            )
            return regex.findAll(raw).associate { match ->
                match.groupValues[1] to match.groupValues[2]
            }
        }

        private fun Map<String, Any?>.toJSONObjectString(): String {
            val json = JSONObject()
            forEach { (key, value) -> json.put(key, value) }
            return json.toString()
        }

        private fun toolEscape(text: String): String =
            text.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "")
                .replace("\t", "\\t")

        private fun jsonEscape(text: String): String =
            text.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "")
                .replace("\t", "\\t")

        private const val ENUM_VALUE_SEPARATOR = "\u001f"
    }

    // MARK: - Engine

    internal class Engine private constructor(
        @Volatile
        private var handle: Long,
    ) : Closeable {
        public companion object {
            public fun create(configuration: Configuration): Engine {
                val handle = ZephrAgentRuntimeJniBridge.createAgent()
                require(handle != 0L) { "zephr_agent_create returned null" }

                val initialized = ZephrAgentRuntimeJniBridge.initAgent(
                    agentHandle = handle,
                    llmExecutionPlan = configuration.llmExecutionPlan,
                    ragExecutionPlan = configuration.ragExecutionPlan,
                    vlmExecutionPlan = configuration.vlmExecutionPlan,
                    gemma4RuntimeMtpEnabled = configuration.gemma4Runtime.mtpEnabled,
                    gemma4GpuPrecision = configuration.gemma4Runtime.gpuPrecision.wireValue,
                    gemma4KvCacheMaxTokens = configuration.gemma4Runtime.kvCacheMaxTokens,
                    gemma4ConstrainedVerifyBatch = configuration.gemma4Runtime.constrainedVerifyBatch.wireValue,
                    gemma4MtpTrustVerifyKv = configuration.gemma4Runtime.mtpTrustVerifyKv,
                    gemma4MtpAdaptiveEnabled = configuration.gemma4Runtime.mtpAdaptiveEnabled,
                    gemma4MtpAdaptiveMinCycles = configuration.gemma4Runtime.mtpAdaptiveMinCycles,
                    gemma4MtpAdaptiveMinSavedPerCycle = configuration.gemma4Runtime.mtpAdaptiveMinSavedPerCycle,
                    gemma4MtpTrace = configuration.gemma4Runtime.mtpTrace,
                    diagnosticGemma4PrefillByDecode = configuration.diagnosticGemma4.prefillByDecode,
                    diagnosticGemma4PrefillMaxChunk = configuration.diagnosticGemma4.prefillMaxChunk,
                    diagnosticGemma4ConstrainedVerifyTrace = configuration.diagnosticGemma4.constrainedVerifyTrace,
                    diagnosticGemma4ConstrainedVerifyMaxAccept = configuration.diagnosticGemma4.constrainedVerifyMaxAccept,
                    numThreads = configuration.numThreads,
                    llmModelPath = configuration.llmModelPath,
                    ragEmbeddingPath = configuration.ragEmbeddingPath,
                    vlmModelPath = configuration.vlmModelPath,
                    litertCompilationCacheDir = configuration.litertCompilationCacheDir,
                    litertRuntimeLibraryDir = configuration.litertRuntimeLibraryDir,
                )
                if (!initialized) {
                    ZephrAgentRuntimeJniBridge.destroyAgent(handle)
                    error("zephr_agent_init failed: ${configuration.llmModelPath}")
                }
                return Engine(handle)
            }

            public fun detectModelFamily(path: String): String? =
                ZephrAgentRuntimeJniBridge.detectModelFamily(path).ifBlank { null }

        }

        @Synchronized
        internal fun embedText(
            text: String,
            taskType: Embeddings.TaskType = Embeddings.TaskType.QUERY,
        ): Embeddings.Embedding {
            val resultHandle = ZephrAgentRuntimeJniBridge.embedText(
                agentHandle = requireHandle(),
                text = text,
                taskType = taskType.wireValue,
            )
            check(resultHandle != 0L) { "zephr_agent_embed_text returned null" }

            return try {
                Embeddings.Embedding(
                    vector = ZephrAgentRuntimeJniBridge.embeddingData(resultHandle),
                    dimension = ZephrAgentRuntimeJniBridge.embeddingDimension(resultHandle),
                    durationMs = ZephrAgentRuntimeJniBridge.embeddingDurationMs(resultHandle),
                    taskType = taskType,
                )
            } finally {
                ZephrAgentRuntimeJniBridge.destroyEmbeddingResult(resultHandle)
            }
        }

        @Synchronized
        internal fun describeImage(
            image: Tools.RgbImage,
            prompt: String = "Briefly describe this image.",
            maxTokens: Int = 0,
        ): Tools.VisionResult {
            val resultHandle = ZephrAgentRuntimeJniBridge.describeImageRgb888(
                agentHandle = requireHandle(),
                rgb = image.rgb,
                width = image.width,
                height = image.height,
                rowStride = image.rowStride,
                prompt = prompt,
                maxTokens = maxTokens,
            )
            check(resultHandle != 0L) { "zephr_agent_describe_image_rgb888 returned null" }

            return try {
                makeVisionResult(resultHandle)
            } finally {
                ZephrAgentRuntimeJniBridge.destroyVisionResult(resultHandle)
            }
        }

        @Synchronized
        internal fun generateText(
            userMessage: String,
            systemMessage: String? = null,
            maxTokens: Int = 256,
            temperature: Float = 0.7f,
            topK: Int = 40,
            topP: Float = 0.95f,
        ): Diagnostics.TextResult {
            val resultHandle = ZephrAgentRuntimeJniBridge.generateText(
                agentHandle = requireHandle(),
                userMessage = userMessage,
                systemMessage = systemMessage,
                maxTokens = maxTokens,
                temperature = temperature,
                topK = topK,
                topP = topP,
            )
            check(resultHandle != 0L) { "zephr_agent_generate_text returned null" }

            return try {
                makeTextResult(resultHandle)
            } finally {
                ZephrAgentRuntimeJniBridge.destroyTextResult(resultHandle)
            }
        }

        @Synchronized
        internal fun generateToolAwareText(
            userMessage: String,
            systemMessage: String? = null,
            maxTokens: Int = 256,
            temperature: Float = 0.0f,
            topK: Int = 40,
            topP: Float = 0.95f,
            reserveOutputTokens: Int = 0,
            toolNames: Array<String>,
            toolDescriptions: Array<String>,
            paramToolIndexes: IntArray,
            paramNames: Array<String>,
            paramDescriptions: Array<String>,
            paramTypes: Array<String>,
            paramRequired: BooleanArray,
            paramEnumValues: Array<String>,
        ): Diagnostics.TextResult {
            val resultHandle = ZephrAgentRuntimeJniBridge.generateToolAwareText(
                agentHandle = requireHandle(),
                userMessage = userMessage,
                systemMessage = systemMessage,
                maxTokens = maxTokens,
                temperature = temperature,
                topK = topK,
                topP = topP,
                reserveOutputTokens = reserveOutputTokens,
                toolNames = toolNames,
                toolDescriptions = toolDescriptions,
                paramToolIndexes = paramToolIndexes,
                paramNames = paramNames,
                paramDescriptions = paramDescriptions,
                paramTypes = paramTypes,
                paramRequired = paramRequired,
                paramEnumValues = paramEnumValues,
            )
            check(resultHandle != 0L) { "zephr_agent_generate_tool_aware_text returned null" }

            return try {
                makeTextResult(resultHandle)
            } finally {
                ZephrAgentRuntimeJniBridge.destroyTextResult(resultHandle)
            }
        }

        @Synchronized
        internal fun generateToolAwareTextStreaming(
            userMessage: String,
            systemMessage: String? = null,
            maxTokens: Int = 256,
            temperature: Float = 0.0f,
            topK: Int = 40,
            topP: Float = 0.95f,
            reserveOutputTokens: Int = 0,
            toolNames: Array<String>,
            toolDescriptions: Array<String>,
            paramToolIndexes: IntArray,
            paramNames: Array<String>,
            paramDescriptions: Array<String>,
            paramTypes: Array<String>,
            paramRequired: BooleanArray,
            paramEnumValues: Array<String>,
            onChunk: (String) -> Boolean,
        ): Diagnostics.TextResult {
            val callback = ZephrAgentRuntimeJniBridge.TextStreamCallback { text ->
                onChunk(text)
            }
            val resultHandle = ZephrAgentRuntimeJniBridge.generateToolAwareTextStreaming(
                agentHandle = requireHandle(),
                userMessage = userMessage,
                systemMessage = systemMessage,
                maxTokens = maxTokens,
                temperature = temperature,
                topK = topK,
                topP = topP,
                reserveOutputTokens = reserveOutputTokens,
                toolNames = toolNames,
                toolDescriptions = toolDescriptions,
                paramToolIndexes = paramToolIndexes,
                paramNames = paramNames,
                paramDescriptions = paramDescriptions,
                paramTypes = paramTypes,
                paramRequired = paramRequired,
                paramEnumValues = paramEnumValues,
                callback = callback,
            )
            check(resultHandle != 0L) { "zephr_agent_generate_tool_aware_text_stream returned null" }

            return try {
                makeTextResult(resultHandle)
            } finally {
                ZephrAgentRuntimeJniBridge.destroyTextResult(resultHandle)
            }
        }

        @Synchronized
        internal fun generateToolAwareTextFromPromptStreaming(
            prompt: String,
            maxTokens: Int = 256,
            temperature: Float = 0.0f,
            topK: Int = 40,
            topP: Float = 0.95f,
            reserveOutputTokens: Int = 0,
            toolNames: Array<String>,
            toolDescriptions: Array<String>,
            paramToolIndexes: IntArray,
            paramNames: Array<String>,
            paramDescriptions: Array<String>,
            paramTypes: Array<String>,
            paramRequired: BooleanArray,
            paramEnumValues: Array<String>,
            onChunk: (String) -> Boolean,
        ): Diagnostics.TextResult {
            val callback = ZephrAgentRuntimeJniBridge.TextStreamCallback { text ->
                onChunk(text)
            }
            val resultHandle = ZephrAgentRuntimeJniBridge.generateToolAwareTextFromPromptStreaming(
                agentHandle = requireHandle(),
                prompt = prompt,
                maxTokens = maxTokens,
                temperature = temperature,
                topK = topK,
                topP = topP,
                reserveOutputTokens = reserveOutputTokens,
                toolNames = toolNames,
                toolDescriptions = toolDescriptions,
                paramToolIndexes = paramToolIndexes,
                paramNames = paramNames,
                paramDescriptions = paramDescriptions,
                paramTypes = paramTypes,
                paramRequired = paramRequired,
                paramEnumValues = paramEnumValues,
                callback = callback,
            )
            check(resultHandle != 0L) { "zephr_agent_generate_tool_aware_text_from_prompt_stream returned null" }

            return try {
                makeTextResult(resultHandle)
            } finally {
                ZephrAgentRuntimeJniBridge.destroyTextResult(resultHandle)
            }
        }

        @Synchronized
        internal fun generateTextFromPromptStreaming(
            prompt: String,
            maxTokens: Int = 256,
            temperature: Float = 0.0f,
            topK: Int = 40,
            topP: Float = 0.95f,
            onChunk: (String) -> Boolean,
        ): Diagnostics.TextResult {
            val callback = ZephrAgentRuntimeJniBridge.TextStreamCallback { text ->
                onChunk(text)
            }
            val resultHandle = ZephrAgentRuntimeJniBridge.generateTextFromPromptStreaming(
                agentHandle = requireHandle(),
                prompt = prompt,
                maxTokens = maxTokens,
                temperature = temperature,
                topK = topK,
                topP = topP,
                callback = callback,
            )
            check(resultHandle != 0L) { "zephr_agent_generate_text_from_prompt_stream returned null" }

            return try {
                makeTextResult(resultHandle)
            } finally {
                ZephrAgentRuntimeJniBridge.destroyTextResult(resultHandle)
            }
        }

        @Synchronized
        internal fun continueAfterToolResponseStreaming(
            toolResponse: String,
            maxTokens: Int = 256,
            temperature: Float = 0.0f,
            topK: Int = 40,
            topP: Float = 0.95f,
            reserveOutputTokens: Int = 0,
            onChunk: (String) -> Boolean,
        ): Diagnostics.TextResult {
            val callback = ZephrAgentRuntimeJniBridge.TextStreamCallback { text ->
                onChunk(text)
            }
            val resultHandle = ZephrAgentRuntimeJniBridge.continueAfterToolResponseStreaming(
                agentHandle = requireHandle(),
                toolResponse = toolResponse,
                maxTokens = maxTokens,
                temperature = temperature,
                topK = topK,
                topP = topP,
                reserveOutputTokens = reserveOutputTokens,
                callback = callback,
            )
            check(resultHandle != 0L) { "zephr_agent_continue_after_tool_response_stream returned null" }

            return try {
                makeTextResult(resultHandle)
            } finally {
                ZephrAgentRuntimeJniBridge.destroyTextResult(resultHandle)
            }
        }

        @Synchronized
        internal fun continueToolAwareTextStreaming(
            promptSuffix: String,
            maxTokens: Int = 256,
            temperature: Float = 0.0f,
            topK: Int = 40,
            topP: Float = 0.95f,
            reserveOutputTokens: Int = 0,
            toolNames: Array<String>,
            toolDescriptions: Array<String>,
            paramToolIndexes: IntArray,
            paramNames: Array<String>,
            paramDescriptions: Array<String>,
            paramTypes: Array<String>,
            paramRequired: BooleanArray,
            paramEnumValues: Array<String>,
            onChunk: (String) -> Boolean,
        ): Diagnostics.TextResult {
            val callback = ZephrAgentRuntimeJniBridge.TextStreamCallback { text ->
                onChunk(text)
            }
            val resultHandle = ZephrAgentRuntimeJniBridge.continueToolAwareTextStreaming(
                agentHandle = requireHandle(),
                promptSuffix = promptSuffix,
                maxTokens = maxTokens,
                temperature = temperature,
                topK = topK,
                topP = topP,
                reserveOutputTokens = reserveOutputTokens,
                toolNames = toolNames,
                toolDescriptions = toolDescriptions,
                paramToolIndexes = paramToolIndexes,
                paramNames = paramNames,
                paramDescriptions = paramDescriptions,
                paramTypes = paramTypes,
                paramRequired = paramRequired,
                paramEnumValues = paramEnumValues,
                callback = callback,
            )
            check(resultHandle != 0L) { "zephr_agent_continue_tool_aware_text_stream returned null" }

            return try {
                makeTextResult(resultHandle)
            } finally {
                ZephrAgentRuntimeJniBridge.destroyTextResult(resultHandle)
            }
        }

        @Synchronized
        internal fun collectHeavyJson(
            prompt: String,
            collectActivations: Boolean,
            topK: Int,
        ): String =
            ZephrAgentRuntimeJniBridge.collectHeavyJson(
                requireHandle(),
                prompt,
                collectActivations,
                topK.coerceAtLeast(1),
            )

        @Synchronized
        public fun drainModelTimings(): List<Diagnostics.ModelTiming> {
            val current = requireHandle()
            val count = ZephrAgentRuntimeJniBridge.modelLifecycleTimingCount(current)
            if (count <= 0) return emptyList()
            return try {
                buildList {
                    repeat(count) { index ->
                        add(
                            Diagnostics.ModelTiming(
                                component = ZephrAgentRuntimeJniBridge.modelLifecycleComponent(current, index),
                                action = ZephrAgentRuntimeJniBridge.modelLifecycleAction(current, index),
                                detail = ZephrAgentRuntimeJniBridge.modelLifecycleDetail(current, index),
                                durationMs = ZephrAgentRuntimeJniBridge.modelLifecycleDurationMs(current, index),
                                ok = ZephrAgentRuntimeJniBridge.modelLifecycleOk(current, index),
                            )
                        )
                    }
                }
            } finally {
                ZephrAgentRuntimeJniBridge.clearModelTimings(current)
            }
        }

        @Synchronized
        public fun releaseModels(): List<Diagnostics.ModelTiming> {
            ZephrAgentRuntimeJniBridge.releaseModels(requireHandle())
            return drainModelTimings()
        }

        @Synchronized
        override fun close() {
            val current = handle
            if (current != 0L) {
                handle = 0L
                ZephrAgentRuntimeJniBridge.destroyAgent(current)
            }
        }

        private fun requireHandle(): Long =
            checkNotNull(handle.takeIf { it != 0L }) { "Engine is closed" }


        private fun makeVisionResult(resultHandle: Long): Tools.VisionResult =
            Tools.VisionResult(
                response = ZephrAgentRuntimeJniBridge.vlmResponse(resultHandle),
                inputPatches = ZephrAgentRuntimeJniBridge.vlmInputPatches(resultHandle),
                validVisionTokens = ZephrAgentRuntimeJniBridge.vlmValidVisionTokens(resultHandle),
                imageTokenSlots = ZephrAgentRuntimeJniBridge.vlmImageTokenSlots(resultHandle),
                resizedWidth = ZephrAgentRuntimeJniBridge.vlmResizedWidth(resultHandle),
                resizedHeight = ZephrAgentRuntimeJniBridge.vlmResizedHeight(resultHandle),
                promptTokens = ZephrAgentRuntimeJniBridge.vlmPromptTokens(resultHandle),
                decodeSteps = ZephrAgentRuntimeJniBridge.vlmDecodeSteps(resultHandle),
                firstDecodeMs = ZephrAgentRuntimeJniBridge.vlmFirstDecodeMs(resultHandle),
            )

        private fun makeTextResult(resultHandle: Long): Diagnostics.TextResult =
            Diagnostics.TextResult(
                response = ZephrAgentRuntimeJniBridge.textResponse(resultHandle),
                prompt = ZephrAgentRuntimeJniBridge.textPrompt(resultHandle),
                prefillTokens = ZephrAgentRuntimeJniBridge.textPrefillTokens(resultHandle),
                stats = Diagnostics.StageStats(
                    tokenizeMs = 0,
                    prefillMs = ZephrAgentRuntimeJniBridge.textPrefillMs(resultHandle),
                    decodeMs = ZephrAgentRuntimeJniBridge.textDecodeMs(resultHandle),
                    firstDecodeMs = ZephrAgentRuntimeJniBridge.textFirstDecodeMs(resultHandle),
                    inputTokens = ZephrAgentRuntimeJniBridge.textInputTokens(resultHandle),
                    outputTokens = ZephrAgentRuntimeJniBridge.textDecodeSteps(resultHandle),
                    mtpRejectedCycles = ZephrAgentRuntimeJniBridge.textMtpRejectedCycles(resultHandle),
                    mtpRejectedAfterPrefix0 = ZephrAgentRuntimeJniBridge.textMtpRejectedAfterPrefix0(resultHandle),
                    mtpRejectedAfterPrefix1 = ZephrAgentRuntimeJniBridge.textMtpRejectedAfterPrefix1(resultHandle),
                    mtpRejectedAfterPrefix2 = ZephrAgentRuntimeJniBridge.textMtpRejectedAfterPrefix2(resultHandle),
                ),
                currentPosition = ZephrAgentRuntimeJniBridge.textCurrentPosition(requireHandle()),
            )
    }

    private const val ZEPHR_AGENT_HTTP_TIMEOUT_MS = 90_000
    private val TENSOR_G5_REGEX = Regex("""\btensor\s*g5\b""")
    private val TENSOR_G4_REGEX = Regex("""\btensor\s*g4\b""")
    private val TENSOR_G3_REGEX = Regex("""\btensor\s*g3\b""")

    internal object Http {
        fun openGetConnection(url: URL): HttpURLConnection =
            (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                instanceFollowRedirects = true
                connectTimeout = ZEPHR_AGENT_HTTP_TIMEOUT_MS
                readTimeout = ZEPHR_AGENT_HTTP_TIMEOUT_MS
            }
    }

    private fun androidLocalModelRoot(): File =
        File("/data/local/tmp/ZephrAgentRuntimeLocalModels")

    // MARK: - Runtime configuration

    public object Lifecycle {
        public enum class ModelFamily(public val wireName: String) {
            GEMMA4("gemma4"),
            GEMMA3_EMBEDDING("gemma3_embedding");

            public companion object {
                public fun fromWireName(value: String): ModelFamily? =
                    entries.firstOrNull { it.wireName == value }
            }
        }

        public enum class ModelChannel(public val wireName: String) {
            PUBLIC("public"),
            BEN("ben"),
            LOCAL("local");

            public companion object {
                public fun fromWireName(value: String): ModelChannel? =
                    entries.firstOrNull { it.wireName == value }
            }
        }

        public data class Configuration(
            val modelChannel: ModelChannel = ModelChannel.PUBLIC,
            val llmExecutionChoiceId: String = "",
            val ragEmbeddingExecutionChoiceId: String = "",
            val vlmExecutionChoiceId: String = "",
            val litertCompilationCacheEnabled: Boolean = false,
            val litertRuntimeLibraryDir: String? = null,
            val gemma4Runtime: Gemma4Options = Gemma4Options(),
            val diagnosticGemma4: Diagnostics.Gemma4Options = Diagnostics.Gemma4Options(),
            val numThreads: Int = 0,
        ) {
            public val effectiveHardwareSummary: String
                get() = buildList {
                    add("llm:${llmExecutionChoiceId.ifBlank { "default" }}")
                    add("rag:${ragEmbeddingExecutionChoiceId.ifBlank { "default" }}")
                    add("vlm:${vlmExecutionChoiceId.ifBlank { "default" }}")
                    add("mtp:${if (gemma4Runtime.mtpEnabled) "on" else "off"}")
                    if (gemma4Runtime.gpuPrecision != Gemma4GpuPrecision.AUTOMATIC) {
                        add("gemma4_precision:${gemma4Runtime.gpuPrecision.wireValue}")
                    }
                    if (diagnosticGemma4 != Diagnostics.Gemma4Options()) {
                        add("diagnostic_gemma4:on")
                    }
                }.joinToString(" ")
        }

        public enum class Gemma4GpuPrecision(public val wireValue: Int) {
            AUTOMATIC(-1),
            PRECISION_0(0),
            PRECISION_1(1),
            PRECISION_2(2),
        }

        public enum class RuntimeToggle(public val wireValue: Int) {
            AUTOMATIC(-1),
            DISABLED(0),
            ENABLED(1),
        }

        public data class Gemma4Options(
            val gpuPrecision: Gemma4GpuPrecision = Gemma4GpuPrecision.AUTOMATIC,
            val kvCacheMaxTokens: Int = 0,
            val constrainedVerifyBatch: RuntimeToggle = RuntimeToggle.AUTOMATIC,
            val mtpEnabled: Boolean = false,
            val mtpTrustVerifyKv: Boolean = true,
            val mtpAdaptiveEnabled: Boolean = true,
            val mtpAdaptiveMinCycles: Int = 4,
            val mtpAdaptiveMinSavedPerCycle: Float = 0.5f,
            val mtpTrace: Boolean = false,
        )

        public data class ExecutionChoice(
            val id: String,
            val label: String,
            val family: ModelFamily?,
            val executionPlan: String,
            val artifactId: String,
            val requestedPlan: String = executionPlan,
            val components: List<ExecutionPlanComponent> = emptyList(),
        ) {
            public val isOff: Boolean
                get() = id == "off" || executionPlan == "off" || requestedPlan == "off"
        }

        public data class ExecutionPlanComponent(
            val id: String,
            val role: Role,
            val artifactId: String,
            val family: String?,
            val target: String,
            val requestedTarget: String? = null,
            val reason: String? = null,
            val signature: String? = null,
        ) {
            public enum class Role(public val wireName: String) {
                TEXT("text"),
                EMBEDDING("embedding"),
                VISION("vision");

                public companion object {
                    public fun fromWireName(value: String): Role? =
                        entries.firstOrNull { it.wireName == value }
                }
            }
        }

        public data class ExecutionComponent(
            val choiceId: String,
            val artifact: String,
            val executionPlan: String,
            val requestedPlan: String,
            val components: List<ExecutionPlanComponent>,
        ) {
            public companion object {
                internal fun fromChoice(choice: ExecutionChoice): ExecutionComponent =
                    ExecutionComponent(
                        choiceId = choice.id,
                        artifact = choice.artifactId,
                        executionPlan = choice.executionPlan,
                        requestedPlan = choice.requestedPlan,
                        components = choice.components,
                    )
            }
        }

        public data class ResolvedExecutionPlan(
            val selection: ExecutionSelection,
            val components: List<ExecutionPlanComponent>,
        )

        public data class ExecutionChoices(
            val llm: List<ExecutionChoice>,
            val ragEmbedding: List<ExecutionChoice>,
            val vlm: List<ExecutionChoice>,
        ) {
            public fun selectedLlm(id: String): ExecutionChoice? =
                selectedChoice(llm, id)

            public fun selectedRagEmbedding(id: String): ExecutionChoice? =
                selectedChoice(ragEmbedding, id)

            public fun selectedVlm(id: String): ExecutionChoice? =
                selectedChoice(vlm, id)

            public fun normalized(configuration: Configuration): Configuration {
                if (llm.isEmpty()) return configuration
                val selectedLlm = selectedLlm(configuration.llmExecutionChoiceId)
                val selectedRag = selectedRagEmbedding(configuration.ragEmbeddingExecutionChoiceId)
                val selectedVlm = selectedVlm(configuration.vlmExecutionChoiceId)
                return configuration.copy(
                    llmExecutionChoiceId = selectedLlm?.id.orEmpty(),
                    ragEmbeddingExecutionChoiceId = selectedRag?.id.orEmpty(),
                    vlmExecutionChoiceId = selectedVlm?.id.orEmpty(),
                )
            }

            private fun selectedChoice(
                choices: List<ExecutionChoice>,
                id: String,
            ): ExecutionChoice? =
                selectedExecutionChoice(choices, id)
        }

        public fun selectedExecutionChoice(
            choices: List<ExecutionChoice>,
            id: String,
        ): ExecutionChoice? =
            if (id.isBlank()) choices.firstOrNull() else choices.firstOrNull { it.id == id } ?: choices.firstOrNull()

        public data class ExecutionSelection(
            val llm: ExecutionComponent,
            val rag: ExecutionComponent?,
            val vlm: ExecutionComponent?,
        )

        public enum class Phase {
            IDLE,
            RESOLVING_MODELS,
            DOWNLOADING_MODELS,
            VERIFYING_MODELS,
            INITIALIZING_ENGINE,
            LOADING_TILES,
            READY,
            FAILED,
        }

        public data class ArtifactState(
            val id: String,
            val title: String,
            val purpose: String = "",
            val phase: Phase,
            val downloadedBytes: Long = 0,
            val totalBytes: Long? = null,
            val downloadBytesPerSecond: Double? = null,
            val version: String,
            val detail: String? = null,
        ) {
            public enum class Phase {
                PENDING,
                WAITING_FOR_SHARED_DOWNLOAD,
                CHECKING_LOCAL_FILE,
                DOWNLOADING,
                VERIFYING,
                READY,
                SKIPPED,
                FAILED,
            }

            public val fractionComplete: Double?
                get() = totalBytes?.takeIf { it > 0 }?.let { total ->
                    (downloadedBytes.toDouble() / total.toDouble()).coerceIn(0.0, 1.0)
                }
        }

        public data class State(
            val phase: Phase,
            val message: String,
            val artifacts: List<ArtifactState> = emptyList(),
            val canRetry: Boolean = false,
            val canUseChat: Boolean = false,
            val errorMessage: String? = null,
        ) {
            public companion object {
                public val Idle: State =
                    State(Phase.IDLE, "Idle")

                public fun ready(artifacts: List<ArtifactState>): State =
                    State(
                        phase = Phase.READY,
                        message = "Ready",
                        artifacts = artifacts,
                        canUseChat = true,
                    )
            }
        }

        public data class RuntimeDataLoadResult(
            val sourceCount: Int,
            val itemCount: Int,
            val indexSize: Int,
            val durationMs: Long,
        )

        public data class Event(
            val id: String = UUID.randomUUID().toString(),
            val kind: Kind,
            val status: Status,
            val title: String,
            val detail: String,
            val durationMs: Long? = null,
            val runtimeDataStats: RuntimeDataLoadResult? = null,
            val modelTimings: List<Diagnostics.ModelTiming> = emptyList(),
            val artifacts: List<ArtifactState> = emptyList(),
        ) {
            public enum class Kind {
                MODEL_PREPARATION,
                ENGINE_INITIALIZATION,
                RUNTIME_DATA_LOADING,
                TEARDOWN,
            }

            public enum class Status {
                RUNNING,
                PASSED,
                FAILED,
                SKIPPED,
            }
        }

        public data class ResolvedModels(
            val llmModelFile: File,
            val ragEmbeddingFile: File?,
            val vlmModelFile: File?,
            val executionSelection: ExecutionSelection,
            val executionPlan: ResolvedExecutionPlan?,
            val detectedLLMFamily: String?,
        )

        public sealed class Error(message: String) : RuntimeException(message) {
            public class NoDownloadUrl(title: String) :
                Error("No download URL configured for $title")

            public class MissingRequiredArtifact(title: String) :
                Error("Missing required model artifact: $title")

            public class DownloadFailed(message: String) : Error(message)
            public class ChecksumMismatch(title: String) :
                Error("Checksum verification failed for $title")

            public class InvalidManifest(message: String) : Error(message)
            public class EngineNotReady : Error("Agent engine is not ready")
        }
    }

    // MARK: - Runtime

    internal class Runtime(
        context: Context,
    ) : AutoCloseable {
        private val appContext = context.applicationContext
        private val modelManager = ZephrAgentRuntimeModelManager(appContext)
        internal val sdkDispatcher = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "zephr_agent_runtime")
        }.asCoroutineDispatcher()
        private val mutex = Mutex()

        @Volatile
        private var engine: Engine? = null
        @Volatile
        private var closed = false
        private var currentConfiguration: Lifecycle.Configuration? = null

        private val _lifecycleState = MutableStateFlow(Lifecycle.State.Idle)
        private val _lifecycleTimeline = MutableStateFlow<List<Lifecycle.Event>>(emptyList())

        /** Current runtime status snapshot; may update frequently during downloads. */
        public val lifecycleState: StateFlow<Lifecycle.State> = _lifecycleState.asStateFlow()

        /** Current ordered lifecycle milestone list; entries are upserted by stable id. */
        public val lifecycleTimeline: StateFlow<List<Lifecycle.Event>> = _lifecycleTimeline.asStateFlow()

        public var resolvedModels: Lifecycle.ResolvedModels? = null
            private set
        public var detectedModelFamily: String? = null
            private set
        public fun availableExecutionChoices(
            channel: Lifecycle.ModelChannel,
        ): Lifecycle.ExecutionChoices =
            modelManager.availableExecutionChoices(channel)

        public fun resolvedExecutionPlan(
            configuration: Lifecycle.Configuration,
        ): Lifecycle.ResolvedExecutionPlan? =
            modelManager.resolvedExecutionPlan(configuration)

        public suspend fun prepare(
            configuration: Lifecycle.Configuration,
        ): Unit = withContext(sdkDispatcher) {
            mutex.withLock {
                currentConfiguration = configuration
                engine?.close()
                engine = null
                resolvedModels = null
                detectedModelFamily = null
                _lifecycleTimeline.value = emptyList()
                _lifecycleState.value = Lifecycle.State(
                    phase = Lifecycle.Phase.RESOLVING_MODELS,
                    message = "Resolving models",
                )
                val modelPreparationEventId = "model_preparation"
                val modelPreparationStart = System.nanoTime()
                var modelPreparationFinished = false

                fun modelPreparationDurationMs(): Long =
                    (System.nanoTime() - modelPreparationStart) / 1_000_000

                fun upsertModelPreparationEvent(
                    status: Lifecycle.Event.Status,
                    title: String,
                    detail: String,
                    artifacts: List<Lifecycle.ArtifactState> = _lifecycleState.value.artifacts,
                    durationMs: Long? = null,
                ) {
                    upsertLifecycleEvent(
                        Lifecycle.Event(
                            id = modelPreparationEventId,
                            kind = Lifecycle.Event.Kind.MODEL_PREPARATION,
                            status = status,
                            title = title,
                            detail = detail,
                            durationMs = durationMs,
                            artifacts = artifacts,
                        )
                    )
                }
                upsertModelPreparationEvent(
                    status = Lifecycle.Event.Status.RUNNING,
                    title = "Preparing models",
                    detail = "Resolving models",
                )

                try {
                val resolved = modelManager.resolveModels(configuration) { phase, message, artifacts ->
                    _lifecycleState.value = Lifecycle.State(phase, message, artifacts)
                    upsertModelPreparationEvent(
                        status = if (phase == Lifecycle.Phase.FAILED) {
                            Lifecycle.Event.Status.FAILED
                        } else {
                            Lifecycle.Event.Status.RUNNING
                        },
                        title = when (phase) {
                            Lifecycle.Phase.RESOLVING_MODELS -> "Resolving models"
                            Lifecycle.Phase.DOWNLOADING_MODELS -> "Downloading models"
                            Lifecycle.Phase.VERIFYING_MODELS -> "Verifying models"
                            Lifecycle.Phase.INITIALIZING_ENGINE,
                            Lifecycle.Phase.READY -> "Models ready"
                            Lifecycle.Phase.FAILED -> "Model preparation failed"
                            Lifecycle.Phase.IDLE,
                            Lifecycle.Phase.LOADING_TILES -> "Preparing models"
                        },
                        detail = message,
                        artifacts = artifacts,
                    )
                }
                val detected = Engine.detectModelFamily(resolved.llmModelFile.absolutePath)
                detectedModelFamily = detected
                resolvedModels = resolved.copy(detectedLLMFamily = detected)
                val preparedArtifacts = _lifecycleState.value.artifacts
                upsertModelPreparationEvent(
                    status = Lifecycle.Event.Status.PASSED,
                    title = "Models ready",
                    detail = "${resolved.llmModelFile.name} ready",
                    artifacts = preparedArtifacts,
                    durationMs = modelPreparationDurationMs(),
                )
                modelPreparationFinished = true

                _lifecycleState.value = Lifecycle.State(
                    phase = Lifecycle.Phase.INITIALIZING_ENGINE,
                    message = "Compiling and loading models",
                    artifacts = preparedArtifacts,
                )

                engine = Engine.create(
                    Configuration(
                        llmModelPath = resolved.llmModelFile.absolutePath,
                        llmExecutionPlan = resolved.executionSelection.llm.executionPlan,
                        ragExecutionPlan = resolved.executionSelection.rag?.executionPlan ?: "cpu",
                        vlmExecutionPlan = resolved.executionSelection.vlm?.executionPlan ?: "gpu",
                        gemma4Runtime = configuration.gemma4Runtime,
                        diagnosticGemma4 = configuration.diagnosticGemma4,
                        numThreads = configuration.numThreads,
                        ragEmbeddingPath = resolved.ragEmbeddingFile?.absolutePath,
                        vlmModelPath = resolved.vlmModelFile?.absolutePath,
                        litertCompilationCacheDir = if (configuration.litertCompilationCacheEnabled) {
                            File(appContext.cacheDir, "ZephrAgentRuntime/LiteRTCompilationCache")
                                .also { it.mkdirs() }
                                .absolutePath
                        } else {
                            null
                        },
                        litertRuntimeLibraryDir = configuration.litertRuntimeLibraryDir,
                    )
                )
                upsertLifecycleEvent(
                    Lifecycle.Event(
                        kind = Lifecycle.Event.Kind.ENGINE_INITIALIZATION,
                        status = Lifecycle.Event.Status.PASSED,
                        title = "Agent initialized",
                        detail = configuration.effectiveHardwareSummary,
                    )
                )

                _lifecycleState.value = Lifecycle.State.ready(_lifecycleState.value.artifacts)
            } catch (error: CancellationException) {
                engine?.close()
                engine = null
                resolvedModels = null
                detectedModelFamily = null
                _lifecycleState.value = Lifecycle.State.Idle
                throw error
            } catch (error: Throwable) {
                engine?.close()
                engine = null
                val failedArtifacts = markFailure(_lifecycleState.value.artifacts, error.toString())
                if (!modelPreparationFinished) {
                    upsertModelPreparationEvent(
                        status = Lifecycle.Event.Status.FAILED,
                        title = "Model preparation failed",
                        detail = error.toString(),
                        artifacts = failedArtifacts,
                        durationMs = modelPreparationDurationMs(),
                    )
                }
                _lifecycleState.value = Lifecycle.State(
                    phase = Lifecycle.Phase.FAILED,
                    message = "Agent initialization failed",
                    artifacts = failedArtifacts,
                    canRetry = true,
                    errorMessage = error.toString(),
                )
                throw error
            }
        }
        }

        internal suspend fun embedText(
            text: String,
            taskType: Embeddings.TaskType = Embeddings.TaskType.QUERY,
        ): Embeddings.Embedding = withContext(sdkDispatcher) {
            mutex.withLock {
                val currentEngine = engine ?: throw Lifecycle.Error.EngineNotReady()
                currentEngine.embedText(text, taskType)
            }
        }

        internal suspend fun describeImage(
            image: Tools.RgbImage,
            prompt: String = "Briefly describe this image.",
            maxTokens: Int = 0,
        ): Tools.VisionResult = withContext(sdkDispatcher) {
            mutex.withLock {
                val currentEngine = engine ?: throw Lifecycle.Error.EngineNotReady()
                currentEngine.describeImage(
                    image = image,
                    prompt = prompt,
                    maxTokens = maxTokens,
                )
            }
        }

        public suspend fun drainModelTimings(): List<Diagnostics.ModelTiming> =
            withContext(sdkDispatcher) {
                mutex.withLock {
                    val currentEngine = engine ?: return@withLock emptyList()
                    currentEngine.drainModelTimings()
                }
            }

        internal suspend fun generateText(
            userMessage: String,
            systemMessage: String? = null,
            maxTokens: Int = 256,
            temperature: Float = 0.7f,
            topK: Int = 40,
            topP: Float = 0.95f,
        ): Diagnostics.TextResult = withContext(sdkDispatcher) {
            mutex.withLock {
                val currentEngine = engine ?: throw Lifecycle.Error.EngineNotReady()
                currentEngine.generateText(
                    userMessage = userMessage,
                    systemMessage = systemMessage,
                    maxTokens = maxTokens,
                    temperature = temperature,
                    topK = topK,
                    topP = topP,
                )
            }
        }

        internal suspend fun generateToolAwareText(
            userMessage: String,
            systemMessage: String? = null,
            maxTokens: Int = 256,
            temperature: Float = 0.0f,
            topK: Int = 40,
            topP: Float = 0.95f,
            reserveOutputTokens: Int = 0,
            toolNames: Array<String>,
            toolDescriptions: Array<String>,
            paramToolIndexes: IntArray,
            paramNames: Array<String>,
            paramDescriptions: Array<String>,
            paramTypes: Array<String>,
            paramRequired: BooleanArray,
            paramEnumValues: Array<String>,
        ): Diagnostics.TextResult = withContext(sdkDispatcher) {
            mutex.withLock {
                val currentEngine = engine ?: throw Lifecycle.Error.EngineNotReady()
                currentEngine.generateToolAwareText(
                    userMessage = userMessage,
                    systemMessage = systemMessage,
                    maxTokens = maxTokens,
                    temperature = temperature,
                    topK = topK,
                    topP = topP,
                    reserveOutputTokens = reserveOutputTokens,
                    toolNames = toolNames,
                    toolDescriptions = toolDescriptions,
                    paramToolIndexes = paramToolIndexes,
                    paramNames = paramNames,
                    paramDescriptions = paramDescriptions,
                    paramTypes = paramTypes,
                    paramRequired = paramRequired,
                    paramEnumValues = paramEnumValues,
                )
            }
        }

        internal suspend fun generateToolAwareTextStreaming(
            userMessage: String,
            systemMessage: String? = null,
            maxTokens: Int = 256,
            temperature: Float = 0.0f,
            topK: Int = 40,
            topP: Float = 0.95f,
            reserveOutputTokens: Int = 0,
            toolNames: Array<String>,
            toolDescriptions: Array<String>,
            paramToolIndexes: IntArray,
            paramNames: Array<String>,
            paramDescriptions: Array<String>,
            paramTypes: Array<String>,
            paramRequired: BooleanArray,
            paramEnumValues: Array<String>,
            onChunk: (String) -> Boolean,
        ): Diagnostics.TextResult = withContext(sdkDispatcher) {
            mutex.withLock {
                val currentEngine = engine ?: throw Lifecycle.Error.EngineNotReady()
                currentEngine.generateToolAwareTextStreaming(
                    userMessage = userMessage,
                    systemMessage = systemMessage,
                    maxTokens = maxTokens,
                    temperature = temperature,
                    topK = topK,
                    topP = topP,
                    reserveOutputTokens = reserveOutputTokens,
                    toolNames = toolNames,
                    toolDescriptions = toolDescriptions,
                    paramToolIndexes = paramToolIndexes,
                    paramNames = paramNames,
                    paramDescriptions = paramDescriptions,
                    paramTypes = paramTypes,
                    paramRequired = paramRequired,
                    paramEnumValues = paramEnumValues,
                    onChunk = onChunk,
                )
            }
        }

        internal suspend fun generateToolAwareTextFromPromptStreaming(
            prompt: String,
            maxTokens: Int = 256,
            temperature: Float = 0.0f,
            topK: Int = 40,
            topP: Float = 0.95f,
            reserveOutputTokens: Int = 0,
            toolNames: Array<String>,
            toolDescriptions: Array<String>,
            paramToolIndexes: IntArray,
            paramNames: Array<String>,
            paramDescriptions: Array<String>,
            paramTypes: Array<String>,
            paramRequired: BooleanArray,
            paramEnumValues: Array<String>,
            onChunk: (String) -> Boolean,
        ): Diagnostics.TextResult = withContext(sdkDispatcher) {
            mutex.withLock {
                val currentEngine = engine ?: throw Lifecycle.Error.EngineNotReady()
                currentEngine.generateToolAwareTextFromPromptStreaming(
                    prompt = prompt,
                    maxTokens = maxTokens,
                    temperature = temperature,
                    topK = topK,
                    topP = topP,
                    reserveOutputTokens = reserveOutputTokens,
                    toolNames = toolNames,
                    toolDescriptions = toolDescriptions,
                    paramToolIndexes = paramToolIndexes,
                    paramNames = paramNames,
                    paramDescriptions = paramDescriptions,
                    paramTypes = paramTypes,
                    paramRequired = paramRequired,
                    paramEnumValues = paramEnumValues,
                    onChunk = onChunk,
                )
            }
        }

        internal suspend fun generateTextFromPromptStreaming(
            prompt: String,
            maxTokens: Int = 256,
            temperature: Float = 0.0f,
            topK: Int = 40,
            topP: Float = 0.95f,
            onChunk: (String) -> Boolean,
        ): Diagnostics.TextResult = withContext(sdkDispatcher) {
            mutex.withLock {
                val currentEngine = engine ?: throw Lifecycle.Error.EngineNotReady()
                currentEngine.generateTextFromPromptStreaming(
                    prompt = prompt,
                    maxTokens = maxTokens,
                    temperature = temperature,
                    topK = topK,
                    topP = topP,
                    onChunk = onChunk,
                )
            }
        }

        internal suspend fun collectHeavyJson(
            prompt: String,
            collectActivations: Boolean,
            topK: Int,
        ): String = withContext(sdkDispatcher) {
            mutex.withLock {
                val currentEngine = engine ?: throw Lifecycle.Error.EngineNotReady()
                currentEngine.collectHeavyJson(
                    prompt = prompt,
                    collectActivations = collectActivations,
                    topK = topK,
                )
            }
        }

        internal suspend fun continueAfterToolResponseStreaming(
            toolResponse: String,
            maxTokens: Int = 256,
            temperature: Float = 0.0f,
            topK: Int = 40,
            topP: Float = 0.95f,
            reserveOutputTokens: Int = 0,
            onChunk: (String) -> Boolean,
        ): Diagnostics.TextResult = withContext(sdkDispatcher) {
            mutex.withLock {
                val currentEngine = engine ?: throw Lifecycle.Error.EngineNotReady()
                currentEngine.continueAfterToolResponseStreaming(
                    toolResponse = toolResponse,
                    maxTokens = maxTokens,
                    temperature = temperature,
                    topK = topK,
                    topP = topP,
                    reserveOutputTokens = reserveOutputTokens,
                    onChunk = onChunk,
                )
            }
        }

        internal suspend fun continueToolAwareTextStreaming(
            promptSuffix: String,
            maxTokens: Int = 256,
            temperature: Float = 0.0f,
            topK: Int = 40,
            topP: Float = 0.95f,
            reserveOutputTokens: Int = 0,
            toolNames: Array<String>,
            toolDescriptions: Array<String>,
            paramToolIndexes: IntArray,
            paramNames: Array<String>,
            paramDescriptions: Array<String>,
            paramTypes: Array<String>,
            paramRequired: BooleanArray,
            paramEnumValues: Array<String>,
            onChunk: (String) -> Boolean,
        ): Diagnostics.TextResult = withContext(sdkDispatcher) {
            mutex.withLock {
                val currentEngine = engine ?: throw Lifecycle.Error.EngineNotReady()
                currentEngine.continueToolAwareTextStreaming(
                    promptSuffix = promptSuffix,
                    maxTokens = maxTokens,
                    temperature = temperature,
                    topK = topK,
                    topP = topP,
                    reserveOutputTokens = reserveOutputTokens,
                    toolNames = toolNames,
                    toolDescriptions = toolDescriptions,
                    paramToolIndexes = paramToolIndexes,
                    paramNames = paramNames,
                    paramDescriptions = paramDescriptions,
                    paramTypes = paramTypes,
                    paramRequired = paramRequired,
                    paramEnumValues = paramEnumValues,
                    onChunk = onChunk,
                )
            }
        }

        public suspend fun shutdown(): Lifecycle.Event = withContext(sdkDispatcher) {
            mutex.withLock {
                val start = System.nanoTime()
                val modelTimings = engine?.releaseModels().orEmpty()
                engine?.close()
                engine = null
                val elapsed = (System.nanoTime() - start) / 1_000_000
                val event = Lifecycle.Event(
                    kind = Lifecycle.Event.Kind.TEARDOWN,
                    status = Lifecycle.Event.Status.PASSED,
                    title = "Agent released",
                    detail = "models released",
                    durationMs = elapsed,
                    modelTimings = modelTimings,
                )
                upsertLifecycleEvent(event)
                _lifecycleState.value = Lifecycle.State.Idle
                event
            }
        }

        public suspend fun deleteDownloadedModelsAndCaches(): Unit = withContext(sdkDispatcher) {
            mutex.withLock {
                engine?.close()
                engine = null
                resolvedModels = null
                detectedModelFamily = null
                File(appContext.cacheDir, "ZephrAgentRuntimeLocalModels").deleteRecursively()
                androidLocalModelRoot().deleteRecursively()
                File(appContext.cacheDir, "ZephrAgentRuntime").deleteRecursively()
                _lifecycleTimeline.value = emptyList()
                _lifecycleState.value = Lifecycle.State.Idle
            }
        }

        override fun close() {
            val shouldClose = synchronized(this) {
                if (closed) {
                    false
                } else {
                    closed = true
                    true
                }
            }
            if (!shouldClose) return
            runBlocking(sdkDispatcher) {
                mutex.withLock {
                    engine?.close()
                    engine = null
                }
            }
            sdkDispatcher.close()
        }

        private fun upsertLifecycleEvent(event: Lifecycle.Event) {
            val events = _lifecycleTimeline.value.toMutableList()
            val index = events.indexOfFirst { it.id == event.id }
            if (index >= 0) {
                events[index] = event
            } else {
                events += event
            }
            _lifecycleTimeline.value = events
        }

        private fun markFailure(
            artifacts: List<Lifecycle.ArtifactState>,
            message: String,
        ): List<Lifecycle.ArtifactState> =
            artifacts.map { artifact ->
                if (artifact.phase == Lifecycle.ArtifactState.Phase.READY ||
                    artifact.phase == Lifecycle.ArtifactState.Phase.SKIPPED
                ) {
                    artifact
                } else {
                    artifact.copy(
                        phase = Lifecycle.ArtifactState.Phase.FAILED,
                        detail = artifact.detail ?: message,
                    )
                }
            }
    }

    // MARK: - Models

    private data class ModelCatalog(
        val gemma4LLMVariants: List<ModelArtifact>,
        val ragEmbeddingVariants: List<ModelArtifact>,
        val roleChoices: Lifecycle.ExecutionChoices,
    ) {
        fun artifacts(configuration: Lifecycle.Configuration): List<ModelArtifact> {
            val selection = executionSelection(configuration) ?: return emptyList()

            val artifacts = mutableListOf<ModelArtifact>()
            artifact(selection.llm.artifact, setOf(ModelArtifact.Role.TEXT), selection.llm.executionPlan)
                ?.let { artifacts += it }
            selection.rag?.let { rag ->
                artifact(rag.artifact, setOf(ModelArtifact.Role.EMBEDDING), rag.executionPlan)
                    ?.let { artifacts += it }
            }
            selection.vlm?.let { vlm ->
                artifact(vlm.artifact, setOf(ModelArtifact.Role.TEXT), vlm.executionPlan)
                    ?.takeIf { it.capabilities.contains(ModelArtifact.Capability.VLM) }
                    ?.let { artifacts += it.asVlmArtifact(vlm.executionPlan) }
            }
            return artifacts
        }

        fun executionSelection(
            configuration: Lifecycle.Configuration,
        ): Lifecycle.ExecutionSelection? {
            val llm = selectChoice(
                role = "LLM",
                choices = roleChoices.llm,
                requestedId = configuration.llmExecutionChoiceId,
            ) ?: return null
            val rag = selectChoice(
                role = "RAG embedding",
                choices = roleChoices.ragEmbedding,
                requestedId = configuration.ragEmbeddingExecutionChoiceId,
            )
            val vlm = selectChoice(
                role = "VLM",
                choices = roleChoices.vlm,
                requestedId = configuration.vlmExecutionChoiceId,
            )
            return Lifecycle.ExecutionSelection(
                llm = Lifecycle.ExecutionComponent.fromChoice(llm),
                rag = rag?.takeUnless { it.isOff }?.let(Lifecycle.ExecutionComponent::fromChoice),
                vlm = vlm?.takeUnless { it.isOff }?.let(Lifecycle.ExecutionComponent::fromChoice),
            )
        }

        fun resolvedExecutionPlan(
            configuration: Lifecycle.Configuration,
        ): Lifecycle.ResolvedExecutionPlan? {
            val selection = executionSelection(configuration) ?: return null
            val seen = mutableSetOf<String>()
            val components = (selection.llm.components + (selection.vlm?.components ?: emptyList()) + (selection.rag?.components ?: emptyList()))
                .filter { component ->
                    val key = "${component.id}|${component.artifactId}|${component.target}"
                    if (seen.contains(key)) {
                        false
                    } else {
                        seen += key
                        true
                    }
                }
            return Lifecycle.ResolvedExecutionPlan(selection, components)
        }

        private fun artifact(
            id: String,
            roles: Set<ModelArtifact.Role>,
            executionPlan: String,
        ): ModelArtifact? {
            val allArtifacts = gemma4LLMVariants + ragEmbeddingVariants
            return allArtifacts.firstOrNull { it.id == id && roles.contains(it.role) }
                ?.copy(executionPlan = executionPlan)
        }

        private fun selectChoice(
            role: String,
            choices: List<Lifecycle.ExecutionChoice>,
            requestedId: String,
        ): Lifecycle.ExecutionChoice? {
            if (requestedId.isBlank()) {
                return choices.firstOrNull()
            }
            return choices.firstOrNull { it.id == requestedId }
                ?: choices.firstOrNull()
        }

        companion object {
            fun fromManifest(json: JSONObject): ModelCatalog {
                val artifactJson = json.getJSONObject("artifacts")
                val artifacts = artifactJson.objectEntries().mapNotNull { (id, item) ->
                    ModelArtifact.fromManifest(id, item)
                }

                return ModelCatalog(
                    gemma4LLMVariants = artifacts.filter {
                        it.isTextArtifact &&
                            it.family == Lifecycle.ModelFamily.GEMMA4
                    },
                    ragEmbeddingVariants = artifacts.filter { it.isEmbeddingArtifact },
                    roleChoices = parseRoleChoices(json, artifacts),
                )
            }

            private fun parseRoleChoices(
                json: JSONObject,
                artifacts: List<ModelArtifact>,
            ): Lifecycle.ExecutionChoices {
                val artifactsById = artifacts.associateBy { it.id }
                val roles = json.optJSONObject("roles") ?: return Lifecycle.ExecutionChoices(emptyList(), emptyList(), emptyList())
                val compatibleHardwareTargets = androidCompatibleHardwareTargets()
                return Lifecycle.ExecutionChoices(
                    llm = parseRoleChoiceList(
                        roles.optJSONObject("text"),
                        artifactsById,
                        platform = "android",
                        compatibleHardwareTargets = compatibleHardwareTargets,
                    ),
                    ragEmbedding = parseRoleChoiceList(
                        roles.optJSONObject("embedding"),
                        artifactsById,
                        platform = "android",
                        compatibleHardwareTargets = compatibleHardwareTargets,
                    ),
                    vlm = parseRoleChoiceList(
                        roles.optJSONObject("vlm"),
                        artifactsById,
                        platform = "android",
                        compatibleHardwareTargets = compatibleHardwareTargets,
                    ),
                )
            }

        }
    }

    private data class ModelArtifact(
        val id: String,
        val role: Role,
        val title: String,
        val family: Lifecycle.ModelFamily?,
        val capabilities: List<Capability>,
        val filename: String,
        val version: String,
        val executionPlan: String,
        val downloadUrl: String?,
        val sizeBytes: Long?,
        val sha256: String?,
        val hardwareTarget: String?,
    ) {
        enum class Role {
            TEXT,
            EMBEDDING,
            VLM,
        }

        enum class Capability {
            TEXT,
            VLM,
        }

        fun asVlmArtifact(executionPlan: String): ModelArtifact =
            copy(id = "vlm.$id", role = Role.VLM, capabilities = listOf(Capability.VLM), executionPlan = executionPlan)

        companion object {
            fun fromManifest(
                id: String,
                json: JSONObject,
            ): ModelArtifact? {
                val role = role(json.getString("role"))
                val family = json.optString("family").takeIf { it.isNotBlank() }?.let {
                    Lifecycle.ModelFamily.fromWireName(it) ?: return null
                }

                return ModelArtifact(
                    id = id,
                    role = role,
                    title = json.getString("title"),
                    family = family,
                    capabilities = json.optJSONArray("capabilities").orEmptyStrings().map(::capability)
                        .ifEmpty { defaultCapabilities(role) },
                    filename = json.getString("filename"),
                    version = json.getString("version"),
                    executionPlan = "",
                    downloadUrl = json.optString("url").takeIf { it.isNotBlank() },
                    sizeBytes = if (json.has("size_bytes")) json.optLong("size_bytes") else null,
                    sha256 = json.optString("sha256").takeIf { it.isNotBlank() },
                    hardwareTarget = json.optJSONObject("metadata")
                        ?.optString("hardware_target")
                        ?.takeIf { it.isNotBlank() },
                )
            }

            private fun defaultCapabilities(role: Role): List<Capability> =
                when (role) {
                    Role.TEXT -> listOf(Capability.TEXT)
                    Role.VLM -> listOf(Capability.VLM)
                    Role.EMBEDDING -> emptyList()
                }
        }
    }

    private val ModelArtifact.isTextArtifact: Boolean
        get() = role == ModelArtifact.Role.TEXT

    private val ModelArtifact.isEmbeddingArtifact: Boolean
        get() = role == ModelArtifact.Role.EMBEDDING

    private class ZephrAgentRuntimeModelManager(
        private val context: Context,
    ) {
        private companion object {
            const val MINIMUM_DOWNLOAD_PROGRESS_EMIT_NANOS: Long = 200_000_000L
            const val MAX_CONCURRENT_MODEL_DOWNLOADS: Int = 3
            const val MODEL_DOWNLOAD_BUFFER_BYTES: Int = 1024 * 1024
        }

        private enum class ArtifactResolutionMode {
            LOCAL_ONLY,
            DOWNLOAD_IF_MISSING,
        }

        fun availableExecutionChoices(
            channel: Lifecycle.ModelChannel,
        ): Lifecycle.ExecutionChoices =
            loadCatalog(channel).roleChoices

        fun resolvedExecutionPlan(
            configuration: Lifecycle.Configuration,
        ): Lifecycle.ResolvedExecutionPlan? =
            loadCatalog(configuration.modelChannel).resolvedExecutionPlan(configuration)

        suspend fun resolveModels(
            configuration: Lifecycle.Configuration,
            progress: (Lifecycle.Phase, String, List<Lifecycle.ArtifactState>) -> Unit,
        ): Lifecycle.ResolvedModels = withContext(Dispatchers.IO) {
            val catalog = loadCatalog(configuration.modelChannel)
            val executionSelection = catalog.executionSelection(configuration)
                ?: throw Lifecycle.Error.InvalidManifest("Execution selection could not be resolved")
            val executionPlan = catalog.resolvedExecutionPlan(configuration)
            val artifacts = catalog.artifacts(configuration)
            if (artifacts.none { it.isTextArtifact }) {
                throw Lifecycle.Error.MissingRequiredArtifact("LLM")
            }
            val artifactResolutionMode = when (configuration.modelChannel) {
                Lifecycle.ModelChannel.LOCAL -> ArtifactResolutionMode.LOCAL_ONLY
                else -> ArtifactResolutionMode.DOWNLOAD_IF_MISSING
            }

            val statesLock = Any()
            data class DownloadRateSample(
                val downloadedBytes: Long,
                val timestampNanos: Long,
                val bytesPerSecond: Double?,
            )
            val downloadRateSamples = mutableMapOf<String, DownloadRateSample>()
            val states = artifacts.associate { artifact ->
                artifact.id to Lifecycle.ArtifactState(
                    id = artifact.id,
                    title = artifact.title,
                    purpose = artifact.role.progressPurpose,
                    phase = Lifecycle.ArtifactState.Phase.PENDING,
                    totalBytes = artifact.sizeBytes,
                    version = artifact.version,
                )
            }.toMutableMap()

            fun update(
                artifact: ModelArtifact,
                phase: Lifecycle.ArtifactState.Phase,
                lifecyclePhase: Lifecycle.Phase,
                message: String,
                downloadedBytes: Long? = null,
                detail: String? = null,
            ) {
                val snapshot = synchronized(statesLock) {
                    val previous = states.getValue(artifact.id)
                    val downloadRate = if (
                        phase == Lifecycle.ArtifactState.Phase.DOWNLOADING &&
                        downloadedBytes != null &&
                        downloadedBytes > 0
                    ) {
                        val now = System.nanoTime()
                        val sample = downloadRateSamples[artifact.id]
                        var bytesPerSecond = sample?.bytesPerSecond
                        if (sample != null &&
                            downloadedBytes > sample.downloadedBytes &&
                            now > sample.timestampNanos
                        ) {
                            val elapsedSeconds = (now - sample.timestampNanos).toDouble() / 1_000_000_000.0
                            val instantaneous = (downloadedBytes - sample.downloadedBytes).toDouble() / elapsedSeconds
                            bytesPerSecond = sample.bytesPerSecond?.let { existing ->
                                (existing * 0.75) + (instantaneous * 0.25)
                            } ?: instantaneous
                        }
                        downloadRateSamples[artifact.id] = DownloadRateSample(
                            downloadedBytes = downloadedBytes,
                            timestampNanos = now,
                            bytesPerSecond = bytesPerSecond,
                        )
                        bytesPerSecond
                    } else {
                        if (phase != Lifecycle.ArtifactState.Phase.DOWNLOADING) {
                            downloadRateSamples.remove(artifact.id)
                        }
                        null
                    }
                    states[artifact.id] = previous.copy(
                        phase = phase,
                        downloadedBytes = downloadedBytes ?: previous.downloadedBytes,
                        downloadBytesPerSecond = downloadRate,
                        detail = detail ?: previous.detail,
                    )
                    artifacts.mapNotNull { states[it.id] }
                }
                progress(lifecyclePhase, message, snapshot)
            }

            val grouped = artifacts.groupBy { downloadIdentity(it) }
            val semaphore = Semaphore(MAX_CONCURRENT_MODEL_DOWNLOADS)
            val resolved = coroutineScope {
                grouped.values.map { group ->
                    async {
                        semaphore.withPermit {
                            val primary = group.first()
                            group.drop(1).forEach { alias ->
                                update(
                                    alias,
                                    Lifecycle.ArtifactState.Phase.WAITING_FOR_SHARED_DOWNLOAD,
                                    Lifecycle.Phase.RESOLVING_MODELS,
                                    "Preparing shared model download",
                                    detail = "Using ${primary.title}'s downloaded file",
                                )
                            }
                            update(
                                primary,
                                Lifecycle.ArtifactState.Phase.CHECKING_LOCAL_FILE,
                                Lifecycle.Phase.RESOLVING_MODELS,
                                "Checking ${primary.title}",
                            )
                            val file = resolveArtifact(
                                primary,
                                mode = artifactResolutionMode,
                                progress = { downloaded, _ ->
                                    update(
                                        primary,
                                        Lifecycle.ArtifactState.Phase.DOWNLOADING,
                                        Lifecycle.Phase.DOWNLOADING_MODELS,
                                        "Downloading models",
                                        downloadedBytes = downloaded,
                                    )
                                },
                                verifying = {
                                    update(
                                        primary,
                                        Lifecycle.ArtifactState.Phase.VERIFYING,
                                        Lifecycle.Phase.VERIFYING_MODELS,
                                        "Verifying models",
                                    )
                                },
                            )
                            group.map { artifact ->
                                update(
                                    artifact,
                                    Lifecycle.ArtifactState.Phase.READY,
                                    Lifecycle.Phase.RESOLVING_MODELS,
                                    "${artifact.title} ready",
                                    downloadedBytes = artifact.sizeBytes ?: file.length(),
                                    detail = file.absolutePath,
                                )
                                artifact.role to file
                            }
                        }
                    }
                }
                    .awaitAll()
                    .flatten()
                    .toMap()
            }

            val llm = resolved[ModelArtifact.Role.TEXT]
                ?: throw Lifecycle.Error.MissingRequiredArtifact("LLM")
            Lifecycle.ResolvedModels(
                llmModelFile = llm,
                ragEmbeddingFile = resolved[ModelArtifact.Role.EMBEDDING],
                vlmModelFile = resolved[ModelArtifact.Role.VLM],
                executionSelection = executionSelection,
                executionPlan = executionPlan,
                detectedLLMFamily = null,
            )
        }

        private fun loadCatalog(channel: Lifecycle.ModelChannel): ModelCatalog {
            val path = when (channel) {
                Lifecycle.ModelChannel.LOCAL -> "channels/local.json"
                else -> "channels/${channel.wireName}.json"
            }
            val text = if (channel == Lifecycle.ModelChannel.LOCAL) {
                val localManifest = File(localModelRoot(), "local.json")
                if (!localManifest.isFile) {
                    throw Lifecycle.Error.InvalidManifest(
                        "Local model manifest not found: ${localManifest.absolutePath}"
                    )
                }
                localManifest.readText()
            } else {
                context.assets.open(path).bufferedReader().use { it.readText() }
            }
            return ModelCatalog.fromManifest(JSONObject(text))
        }

        private fun resolveArtifact(
            artifact: ModelArtifact,
            mode: ArtifactResolutionMode,
            progress: (Long, Long?) -> Unit,
            verifying: () -> Unit,
        ): File {
            val finalFile = artifactBlobFile(
                when (mode) {
                    ArtifactResolutionMode.LOCAL_ONLY -> localModelRoot()
                    ArtifactResolutionMode.DOWNLOAD_IF_MISSING -> modelRoot()
                },
                artifact,
            )
            val partialFile = File(finalFile.parentFile, "${finalFile.name}.partial")
            val downloadFile = File(finalFile.parentFile, "${finalFile.name}.download")
            if (finalFile.isFile) {
                verifying()
                try {
                    verifyArtifact(artifact, finalFile)
                    return finalFile
                } catch (error: Lifecycle.Error.ChecksumMismatch) {
                    finalFile.delete()
                    partialFile.delete()
                    downloadFile.delete()
                }
            }

            if (mode == ArtifactResolutionMode.LOCAL_ONLY) {
                throw Lifecycle.Error.MissingRequiredArtifact(artifact.title)
            }

            val rawUrl = artifact.downloadUrl ?: throw Lifecycle.Error.NoDownloadUrl(artifact.title)
            val uri = URI(rawUrl)
            if (uri.scheme != "http" && uri.scheme != "https") {
                throw Lifecycle.Error.InvalidManifest("Unsupported artifact URL scheme: $rawUrl")
            }

            finalFile.parentFile?.mkdirs()
            var existingBytes = if (partialFile.isFile) partialFile.length() else 0L
            if (artifact.sizeBytes != null && existingBytes >= artifact.sizeBytes) {
                verifying()
                try {
                    verifyArtifact(artifact, partialFile)
                    if (finalFile.exists()) finalFile.delete()
                    if (!partialFile.renameTo(finalFile)) {
                        throw IOException("Failed to move ${partialFile.absolutePath} to ${finalFile.absolutePath}")
                    }
                    return finalFile
                } catch (error: Lifecycle.Error.ChecksumMismatch) {
                    partialFile.delete()
                    downloadFile.delete()
                    existingBytes = 0L
                }
            }

            val connection = Http.openGetConnection(URL(rawUrl))
            if (existingBytes > 0) {
                connection.setRequestProperty("Range", "bytes=$existingBytes-")
            }

            try {
                downloadFile.delete()
                connection.connect()
                if (connection.responseCode !in 200..299) {
                    throw Lifecycle.Error.DownloadFailed("HTTP ${connection.responseCode} downloading ${artifact.title}")
                }

                val appending = connection.responseCode == HttpURLConnection.HTTP_PARTIAL && existingBytes > 0
                if (!appending) {
                    partialFile.delete()
                    existingBytes = 0L
                } else {
                    partialFile.copyTo(downloadFile, overwrite = true)
                }

                val expectedTotal = artifact.sizeBytes
                    ?: connection.contentLengthLong.takeIf { it > 0 }?.let { existingBytes + it }
                FileOutputStream(downloadFile, appending).buffered(MODEL_DOWNLOAD_BUFFER_BYTES).use { output ->
                    connection.inputStream.use { input ->
                        val buffer = ByteArray(MODEL_DOWNLOAD_BUFFER_BYTES)
                        var downloaded = existingBytes
                        var lastProgressEmitNanos = 0L
                        var lastProgressBytes = -1L
                        while (true) {
                            val count = input.read(buffer)
                            if (count < 0) break
                            output.write(buffer, 0, count)
                            downloaded += count
                            val now = System.nanoTime()
                            if (lastProgressEmitNanos == 0L ||
                                now - lastProgressEmitNanos >= MINIMUM_DOWNLOAD_PROGRESS_EMIT_NANOS
                            ) {
                                progress(downloaded, expectedTotal)
                                lastProgressEmitNanos = now
                                lastProgressBytes = downloaded
                            }
                        }
                        if (downloaded != lastProgressBytes) {
                            progress(downloaded, expectedTotal)
                        }
                    }
                }
                partialFile.delete()
                if (!downloadFile.renameTo(partialFile)) {
                    throw IOException("Failed to move ${downloadFile.absolutePath} to ${partialFile.absolutePath}")
                }
            } catch (error: Throwable) {
                downloadFile.delete()
                throw error
            } finally {
                connection.disconnect()
            }

            verifying()
            try {
                verifyArtifact(artifact, partialFile)
            } catch (error: Lifecycle.Error.ChecksumMismatch) {
                partialFile.delete()
                throw error
            }
            if (finalFile.exists()) finalFile.delete()
            if (!partialFile.renameTo(finalFile)) {
                throw IOException("Failed to move ${partialFile.absolutePath} to ${finalFile.absolutePath}")
            }
            return finalFile
        }

        private fun verifyArtifact(artifact: ModelArtifact, file: File) {
            val expected = artifact.sha256?.takeIf { it.isNotBlank() } ?: return
            val actual = sha256(file)
            if (!actual.equals(expected, ignoreCase = true)) {
                throw Lifecycle.Error.ChecksumMismatch(artifact.title)
            }
        }

        private fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buffer = ByteArray(1024 * 1024)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
        }

        private fun modelRoot(): File =
            File(context.cacheDir, "ZephrAgentRuntimeLocalModels")

        private fun localModelRoot(): File =
            androidLocalModelRoot()

        private fun artifactBlobFile(root: File, artifact: ModelArtifact): File {
            val sha = artifact.sha256?.takeIf { it.isNotBlank() }
                ?: throw Lifecycle.Error.InvalidManifest("Artifact ${artifact.id} has no sha256")
            return File(File(File(root, "blobs"), "sha256"), sha.lowercase())
        }

        private fun downloadIdentity(artifact: ModelArtifact): String =
            artifact.sha256?.takeIf { it.isNotBlank() }?.lowercase()
                ?: listOf(artifact.downloadUrl.orEmpty(), artifact.filename).joinToString("|")
    }

    // MARK: - JSON helpers

    private fun JSONArray?.orEmptyStrings(): List<String> =
        this?.strings().orEmpty()

    private fun JSONArray?.objects(): List<JSONObject> =
        if (this == null) emptyList() else List(length()) { index -> getJSONObject(index) }

    private fun JSONArray.strings(): List<String> =
        List(length()) { index -> getString(index) }

    private fun JSONObject.objectEntries(): List<Pair<String, JSONObject>> =
        keys().asSequence().map { key -> key to getJSONObject(key) }.toList()

    private fun JSONObject.stringMap(): Map<String, String> =
        keys().asSequence().associateWith { key -> getString(key) }

    private fun parseRoleChoiceList(
        roleJson: JSONObject?,
        artifactsById: Map<String, ModelArtifact>,
        platform: String,
        compatibleHardwareTargets: Set<String>,
    ): List<Lifecycle.ExecutionChoice> {
        if (roleJson == null) return emptyList()
        val choicesJson = roleJson.optJSONObject("platform_choices")
            ?.optJSONArray(platform)
        return choicesJson.objects().mapNotNull { choice ->
            val id = choice.getString("id")
            val requestedPlan = choice.optString("requested_plan").takeIf { it.isNotBlank() }
                ?: "cpu"
            val executionPlan = requestedPlan
            val components = parsePlanComponents(
                choice.optJSONArray("components"),
                artifactsById,
                compatibleHardwareTargets,
            ) ?: return@mapNotNull null
            if (id == "off" || executionPlan == "off" || requestedPlan == "off") {
                return@mapNotNull Lifecycle.ExecutionChoice(
                    id = id,
                    label = choice.optString("label").takeIf { it.isNotBlank() } ?: id,
                    family = choice.optString("family").takeIf { it.isNotBlank() }?.let(Lifecycle.ModelFamily::fromWireName),
                    executionPlan = executionPlan,
                    artifactId = "",
                    requestedPlan = requestedPlan,
                    components = components,
                )
            }
            components.firstOrNull()?.let { component ->
                return@mapNotNull Lifecycle.ExecutionChoice(
                    id = id,
                    label = choice.optString("label").takeIf { it.isNotBlank() } ?: id,
                    family = choice.optString("family").takeIf { it.isNotBlank() }?.let(Lifecycle.ModelFamily::fromWireName),
                    executionPlan = executionPlan,
                    artifactId = component.artifactId,
                    requestedPlan = requestedPlan,
                    components = components,
                )
            }
            null
        }
    }

    private fun parsePlanComponents(
        components: JSONArray?,
        artifactsById: Map<String, ModelArtifact>,
        compatibleHardwareTargets: Set<String>,
    ): List<Lifecycle.ExecutionPlanComponent>? {
        val parsed = mutableListOf<Lifecycle.ExecutionPlanComponent>()
        for (component in components.objects()) {
            val artifactId = component.optString("artifact").takeIf { it.isNotBlank() } ?: return null
            val artifact = artifactsById[artifactId] ?: return null
            if (!artifact.compatibleWith(compatibleHardwareTargets)) return null
            val role = Lifecycle.ExecutionPlanComponent.Role.fromWireName(component.optString("role"))
                ?: return null
            parsed.add(
                Lifecycle.ExecutionPlanComponent(
                    id = component.getString("id"),
                    role = role,
                    artifactId = artifactId,
                    family = component.optString("family").takeIf { it.isNotBlank() },
                    target = component.getString("target"),
                    requestedTarget = component.optString("requested_target").takeIf { it.isNotBlank() },
                    reason = component.optString("reason").takeIf { it.isNotBlank() },
                    signature = component.optString("signature").takeIf { it.isNotBlank() },
                )
            )
        }
        return parsed
    }

    private fun ModelArtifact.compatibleWith(compatibleHardwareTargets: Set<String>): Boolean =
        hardwareTarget == null || hardwareTarget in compatibleHardwareTargets

    private fun androidCompatibleHardwareTargets(): Set<String> {
        val fields = buildList {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                add(Build.SOC_MODEL)
                add(Build.SOC_MANUFACTURER)
            }
            add(Build.HARDWARE)
            add(Build.BOARD)
            add(Build.DEVICE)
            add(Build.MODEL)
        }.joinToString(" ").lowercase()

        val specific = when {
            TENSOR_G5_REGEX.containsMatchIn(fields) -> "google_tensor_g5"
            TENSOR_G4_REGEX.containsMatchIn(fields) || "zumapro" in fields -> "google_tensor_g4"
            TENSOR_G3_REGEX.containsMatchIn(fields) || "zuma" in fields -> "google_tensor_g3"
            else -> null
        }
        return buildSet {
            if (specific != null) {
                add("google_tensor")
                add(specific)
            }
        }
    }

    private val ModelArtifact.Role.progressPurpose: String
        get() = when (this) {
            ModelArtifact.Role.TEXT -> "TEXT"
            ModelArtifact.Role.EMBEDDING -> "EMBEDDING"
            ModelArtifact.Role.VLM -> "VLM"
        }

    private fun role(value: String): ModelArtifact.Role =
        when (value) {
            "text" -> ModelArtifact.Role.TEXT
            "embedding" -> ModelArtifact.Role.EMBEDDING
            "vlm" -> ModelArtifact.Role.VLM
            else -> throw Lifecycle.Error.InvalidManifest("Unknown artifact role: $value")
        }

    private fun capability(value: String): ModelArtifact.Capability =
        when (value) {
            "text" -> ModelArtifact.Capability.TEXT
            "vlm" -> ModelArtifact.Capability.VLM
            else -> throw Lifecycle.Error.InvalidManifest("Unknown artifact capability: $value")
        }
}
