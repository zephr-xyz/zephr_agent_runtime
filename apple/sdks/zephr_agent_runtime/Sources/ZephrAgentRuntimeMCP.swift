import Foundation
import os

extension ZephrAgentRuntime.Conversation.Engine {
    func _mcpTools(
        configuration: ZephrAgentRuntime.Conversation.MCPConfiguration
    ) async throws -> ZephrAgentRuntime.Conversation.MCPToolSet {
        let client = ZephrMCPClient(configuration: configuration)
        let infos = try await client.listTools()
        return ZephrAgentRuntime.Conversation.MCPToolSet(
            providers: infos.map { info in
                ZephrAgentRuntime.Conversation.tool(ZephrMCPOpenApiTool(info: info, client: client))
            },
            toolNames: infos.map(\.name)
        )
    }
}

private struct ZephrMCPToolInfo: Sendable {
    var name: String
    var description: String
    var propertiesJSON: Data
    var required: [String]
}

private struct ZephrMCPOpenApiTool: ZephrAgentRuntime.Conversation.OpenApiTool {
    nonisolated private static let logger = Logger(subsystem: "xyz.zephr.agent.sdk", category: "MCP")

    let info: ZephrMCPToolInfo
    let client: ZephrMCPClient

    nonisolated func getToolDescriptionJsonString() -> String {
        let properties = (try? JSONSerialization.jsonObject(with: info.propertiesJSON)) ?? [:]
        let json: [String: Any] = [
            "name": info.name,
            "description": info.description,
            "parameters": [
                "type": "object",
                "properties": properties,
                "required": info.required
            ]
        ]
        let data = (try? JSONSerialization.data(withJSONObject: json)) ?? Data("{}".utf8)
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    nonisolated func execute(params: String) async -> String {
        let data = Data(params.utf8)
        let json = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
        Self.logger.debug("MCP tool execute name=\(self.info.name, privacy: .public) params=\(params, privacy: .private)")
        let result = await client.callTool(name: info.name, arguments: json)
        Self.logger.debug("MCP tool result name=\(self.info.name, privacy: .public) result=\(result, privacy: .private)")
        return result
    }
}

private actor ZephrMCPClient {
    private static let logger = Logger(subsystem: "xyz.zephr.agent.sdk", category: "MCP")
    private static let connectMaxAttempts = 4
    private static let connectRetryDelayNanoseconds: UInt64 = 2_000_000_000
    private static let requestTimeoutSeconds: TimeInterval = 30

    private let configuration: ZephrAgentRuntime.Conversation.MCPConfiguration
    private var requestID = 0
    private var sessionID: String?
    private var initialized = false

    init(configuration: ZephrAgentRuntime.Conversation.MCPConfiguration) {
        self.configuration = configuration
    }

    func listTools() async throws -> [ZephrMCPToolInfo] {
        guard !configuration.apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ZephrMCPError.missingAPIKey
        }
        try await ensureInitialized()
        let result = try await rpc(method: "tools/list", params: [:])
        guard let tools = result["tools"] as? [[String: Any]] else {
            return []
        }
        return tools.compactMap { tool in
            guard let name = tool["name"] as? String else { return nil }
            let schema = tool["inputSchema"] as? [String: Any]
            let properties = schema?["properties"] as? [String: Any] ?? [:]
            let data = (try? JSONSerialization.data(withJSONObject: properties)) ?? Data("{}".utf8)
            return ZephrMCPToolInfo(
                name: name,
                description: tool["description"] as? String ?? "",
                propertiesJSON: data,
                required: schema?["required"] as? [String] ?? []
            )
        }
    }

    func callTool(name: String, arguments: [String: Any]) async -> String {
        do {
            try await ensureInitialized()
            let result = try await rpc(method: "tools/call", params: [
                "name": name,
                "arguments": arguments
            ])
            if let isError = result["isError"] as? Bool, isError {
                return Self.errorJSON(Self.textContent(from: result))
            }
            return Self.textContent(from: result)
        } catch {
            Self.logger.error("MCP callTool failed name=\(name, privacy: .public) error=\(String(describing: error), privacy: .public)")
            return Self.errorJSON(error.localizedDescription)
        }
    }

    private func ensureInitialized() async throws {
        guard !initialized else { return }
        var lastError: Error?
        for attempt in 1...Self.connectMaxAttempts {
            do {
                sessionID = nil
                _ = try await rpc(method: "initialize", params: [
                    "protocolVersion": "2025-03-26",
                    "clientInfo": [
                        "name": configuration.clientName,
                        "version": configuration.clientVersion
                    ],
                    "capabilities": [:]
                ])
                _ = try await notification(method: "notifications/initialized", params: [:])
                initialized = true
                return
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                lastError = error
                if attempt < Self.connectMaxAttempts {
                    Self.logger.warning(
                        "MCP initialize attempt \(attempt, privacy: .public)/\(Self.connectMaxAttempts, privacy: .public) failed: \(String(describing: error), privacy: .public)"
                    )
                    try await Task.sleep(nanoseconds: Self.connectRetryDelayNanoseconds)
                }
            }
        }
        throw lastError ?? ZephrMCPError.rpc("MCP connection failed")
    }

    private func rpc(method: String, params: [String: Any]) async throws -> [String: Any] {
        requestID += 1
        let id = requestID
        let body: [String: Any] = [
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        ]
        return try await send(body: body, method: method, id: id, expectsResponse: true)
    }

    private func notification(method: String, params: [String: Any]) async throws -> [String: Any] {
        let body: [String: Any] = [
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        ]
        return try await send(body: body, method: method, id: nil, expectsResponse: false)
    }

    private func send(
        body: [String: Any],
        method: String,
        id: Int?,
        expectsResponse: Bool
    ) async throws -> [String: Any] {
        var request = URLRequest(url: configuration.endpoint)
        request.httpMethod = "POST"
        request.timeoutInterval = Self.requestTimeoutSeconds
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
        request.setValue("2025-03-26", forHTTPHeaderField: "MCP-Protocol-Version")
        request.setValue(configuration.apiKey, forHTTPHeaderField: "x-api-key")
        if let sessionID {
            request.setValue(sessionID, forHTTPHeaderField: "Mcp-Session-Id")
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        if let newSessionID = http.value(forHTTPHeaderField: "Mcp-Session-Id"), !newSessionID.isEmpty {
            sessionID = newSessionID
        }
        guard (200...299).contains(http.statusCode) else {
            throw ZephrMCPError.httpStatus(http.statusCode)
        }
        if !expectsResponse || data.isEmpty {
            return [:]
        }
        let json = try Self.jsonResponse(from: data, contentType: http.value(forHTTPHeaderField: "Content-Type") ?? "")
        if let error = json["error"] {
            throw ZephrMCPError.rpc(String(describing: error))
        }
        return json["result"] as? [String: Any] ?? [:]
    }

    private static func textContent(from result: [String: Any]) -> String {
        guard let content = result["content"] as? [[String: Any]] else {
            return ""
        }
        return content.compactMap { item in
            item["text"] as? String
        }.joined(separator: "\n")
    }

    private static func errorJSON(_ message: String) -> String {
        let data = (try? JSONSerialization.data(withJSONObject: ["error": message])) ?? Data(#"{"error":"MCP error"}"#.utf8)
        return String(data: data, encoding: .utf8) ?? #"{"error":"MCP error"}"#
    }

    private static func jsonResponse(from data: Data, contentType: String) throws -> [String: Any] {
        if contentType.localizedCaseInsensitiveContains("text/event-stream") {
            for payload in sseDataPayloads(from: data).reversed() {
                guard let payloadData = payload.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: payloadData) as? [String: Any] else {
                    continue
                }
                return json
            }
            throw ZephrMCPError.rpc("No JSON-RPC payload in event stream")
        }
        return try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    }

    private static func sseDataPayloads(from data: Data) -> [String] {
        guard let text = String(data: data, encoding: .utf8) else {
            return []
        }
        var payloads: [String] = []
        var dataLines: [String] = []
        for rawLine in text.replacingOccurrences(of: "\r\n", with: "\n").split(
            separator: "\n",
            omittingEmptySubsequences: false
        ) {
            let line = String(rawLine)
            if line.isEmpty {
                if !dataLines.isEmpty {
                    payloads.append(dataLines.joined(separator: "\n"))
                    dataLines.removeAll()
                }
                continue
            }
            guard line.hasPrefix("data:") else { continue }
            var value = String(line.dropFirst("data:".count))
            if value.hasPrefix(" ") {
                value.removeFirst()
            }
            dataLines.append(value)
        }
        if !dataLines.isEmpty {
            payloads.append(dataLines.joined(separator: "\n"))
        }
        return payloads
    }
}

private enum ZephrMCPError: LocalizedError {
    case missingAPIKey
    case httpStatus(Int)
    case rpc(String)

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "MCP API key is not configured"
        case .httpStatus(let status):
            return "MCP HTTP status \(status)"
        case .rpc(let message):
            return "MCP RPC error: \(message)"
        }
    }
}
