import Foundation

protocol ToolParamProtocol {
    var description: String { get }
    var isRequired: Bool { get }
    var wrappedValueAny: Any? { get }
    func jsonSchema() -> [String: Any]
}

struct ReflectedOpenApiTool<T: ZephrAgentRuntime.Conversation.Tool>: ZephrAgentRuntime.Conversation.OpenApiTool, @unchecked Sendable {
    private let toolType: T.Type
    private let schema: [String: Any]

    init(_ tool: T) {
        toolType = T.self
        schema = Self.schema(for: tool)
    }

    func getToolDescriptionJsonString() -> String {
        Self.jsonString(schema)
    }

    func execute(params: String) async -> String {
        do {
            let tool = try Self.decodeTool(type: toolType, params: params)
            let result = try await tool.run()
            let normalized = Self.normalize(result)
            if let text = normalized as? String {
                return text
            }
            return Self.jsonString(normalized)
        } catch {
            return Self.jsonString(["error": String(describing: error)])
        }
    }

    private static func schema(for tool: T) -> [String: Any] {
        var properties: [String: Any] = [:]
        var required: [String] = []
        for child in Mirror(reflecting: tool).children {
            guard let label = child.label else { continue }
            guard let param = child.value as? ToolParamProtocol else { continue }
            let name = cleanPropertyLabel(label)
            properties[name] = param.jsonSchema()
            if param.isRequired {
                required.append(name)
            }
        }

        let requiredSorted = required.sorted()
        let parameters: [String: Any] = [
            "type": "object",
            "properties": properties,
            "required": requiredSorted
        ]

        var result: [String: Any] = [
            "name": T.name,
            "description": T.description
        ]
        if !properties.isEmpty {
            result["parameters"] = parameters
        }
        return result
    }

    private static func decodeTool(
        type: T.Type,
        params: String
    ) throws -> T {
        let defaults = defaultValues(for: type.init())
        let provided = try jsonObject(params)
        let merged = defaults.merging(provided) { _, new in new }
        let data = try JSONSerialization.data(withJSONObject: merged)
        return try JSONDecoder().decode(type, from: data)
    }

    private static func defaultValues(for tool: T) -> [String: Any] {
        var defaults: [String: Any] = [:]
        for child in Mirror(reflecting: tool).children {
            guard let label = child.label else { continue }
            guard let param = child.value as? ToolParamProtocol else { continue }
            let name = cleanPropertyLabel(label)
            if let value = param.wrappedValueAny {
                defaults[name] = normalize(value)
            } else if !param.isRequired {
                defaults[name] = NSNull()
            }
        }
        return defaults
    }

    private static func jsonObject(_ text: String) throws -> [String: Any] {
        guard let data = text.data(using: .utf8) else { return [:] }
        guard !data.isEmpty else { return [:] }
        let object = try JSONSerialization.jsonObject(with: data)
        return object as? [String: Any] ?? [:]
    }

    private static func normalize(_ value: Any) -> Any {
        if let value = value as? String { return value }
        if let value = value as? Int { return value }
        if let value = value as? Int32 { return Int(value) }
        if let value = value as? Int64 { return value }
        if let value = value as? Float { return Double(value) }
        if let value = value as? Double { return value }
        if let value = value as? Bool { return value }
        if value is Void { return "" }
        if let value = value as? [Any] {
            return value.map(normalize)
        }
        if let value = value as? [String: Any] {
            return value.mapValues(normalize)
        }
        if let optional = Mirror(reflecting: value).children.first {
            return normalize(optional.value)
        }
        return String(describing: value)
    }

    private static func jsonString(_ object: Any) -> String {
        let value = JSONSerialization.isValidJSONObject(object) ? object : ["value": String(describing: object)]
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
              let text = String(data: data, encoding: .utf8) else {
            return #"{"error":"failed to encode tool response"}"#
        }
        return text
    }

    private static func cleanPropertyLabel(_ label: String) -> String {
        label.hasPrefix("_") ? String(label.dropFirst()) : label
    }
}
