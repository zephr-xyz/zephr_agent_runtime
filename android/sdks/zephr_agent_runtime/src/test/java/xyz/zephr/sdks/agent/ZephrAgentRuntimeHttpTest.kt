package xyz.zephr.sdks.agent

import com.sun.net.httpserver.HttpServer
import java.net.InetSocketAddress
import java.net.URL
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ZephrAgentRuntimeHttpTest {
    @Test
    fun conversationConfigDefaultsToIncrementalKv() {
        assertEquals(
            ZephrAgentRuntime.Conversation.Strategy.INCREMENTAL_KV,
            ZephrAgentRuntime.Conversation.Config().conversationStrategy,
        )
    }

    @Test
    fun openGetConnectionFollowsHttp302Redirects() {
        val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        server.createContext("/tile") { exchange ->
            exchange.responseHeaders.add(
                "Location",
                "http://127.0.0.1:${server.address.port}/s3-tile",
            )
            exchange.sendResponseHeaders(302, -1)
            exchange.close()
        }
        server.createContext("/s3-tile") { exchange ->
            val body = "redirected tile bytes".toByteArray()
            exchange.sendResponseHeaders(200, body.size.toLong())
            exchange.responseBody.use { it.write(body) }
        }

        server.start()
        try {
            val url = URL("http://127.0.0.1:${server.address.port}/tile")
            val connection = ZephrAgentRuntime.Http.openGetConnection(url)

            val body = connection.inputStream.bufferedReader().use { it.readText() }

            assertTrue(connection.instanceFollowRedirects)
            assertEquals(200, connection.responseCode)
            assertEquals("/s3-tile", connection.url.path)
            assertEquals("redirected tile bytes", body)
        } finally {
            server.stop(0)
        }
    }

    @Test
    fun conversationTurnRecordExposesRawEffects() {
        val turn = ZephrAgentRuntime.Conversation.TurnRecord(
            conversationTurnIndex = 0,
            conversationStrategy = ZephrAgentRuntime.Conversation.Strategy.INCREMENTAL_KV,
            userContent = ZephrAgentRuntime.Conversation.Contents.of("add this place"),
            userText = "add this place",
            finalText = "Saved.",
            effects = listOf(
                ZephrAgentRuntime.Conversation.ToolEffect(
                    name = "waypoint_added",
                    payload = mapOf("name" to "Quiet Corner"),
                )
            ),
        )

        assertEquals("waypoint_added", turn.effects.single().name)
        assertEquals("Quiet Corner", turn.effects.single().payload["name"])
    }
}
