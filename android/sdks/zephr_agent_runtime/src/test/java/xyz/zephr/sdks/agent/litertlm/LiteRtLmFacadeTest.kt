package xyz.zephr.sdks.agent.litertlm.tests

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import xyz.zephr.sdks.agent.litertlm.*

class LiteRtLmFacadeTest {
    private class DemoTools : ToolSet {
        @Tool(description = "Return a friendly test string")
        fun hello(): String = "hello"
    }

    @Test
    fun facadeSupportsWildcardImportShape() {
        val engineConfig = EngineConfig(
            textBackend = Backend.GPU,
            embeddingBackend = Backend.OFF,
            vlmBackend = Backend.OFF,
        )
        val conversationConfig = ConversationConfig(
            systemInstruction = Contents.of("You are concise."),
            tools = listOf(tool(DemoTools())),
        )
        val nativeConfig = engineConfig.toLifecycleConfiguration()

        assertEquals("gemma4.gpu", nativeConfig.llmExecutionChoiceId)
        assertEquals("off", nativeConfig.ragEmbeddingExecutionChoiceId)
        assertEquals("off", nativeConfig.vlmExecutionChoiceId)
        assertEquals("You are concise.", conversationConfig.systemInstruction?.text)
        assertTrue(conversationConfig.tools.isNotEmpty())
    }
}
