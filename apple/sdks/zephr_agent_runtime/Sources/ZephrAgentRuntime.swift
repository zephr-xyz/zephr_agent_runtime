import CryptoKit
import Foundation

// MARK: - ZephrAgentRuntimeBundleToken
private final class ZephrAgentRuntimeBundleToken {}

private enum ZephrAgentRuntimePlatform {
    #if os(macOS)
    static let manifestPlatform = "macos"
    #else
    static let manifestPlatform = "ios"
    #endif
}

// MARK: - ZephrAgentRuntimeResources
private enum ZephrAgentRuntimeResources {
    static var bundle: Bundle {
        Bundle(for: ZephrAgentRuntimeBundleToken.self)
    }

    static func url(forResource name: String, withExtension ext: String) -> URL? {
        bundle.url(forResource: name, withExtension: ext)
    }
}

// MARK: - ZephrAgentRuntime.Lifecycle.ModelStorage
extension ZephrAgentRuntime.Lifecycle.ModelStorage {
    var identityKey: String {
        switch self {
        case .appGroup(let identifier):
            return "app_group:\(identifier)"
        case .directory(let url):
            return "directory:\(url.standardizedFileURL.path)"
        }
    }
}

// MARK: - Model catalog

extension ZephrAgentRuntime.Lifecycle.ModelCatalog {
    public static func load(
        channel: ZephrAgentRuntime.Lifecycle.ModelChannel,
        storage: ZephrAgentRuntime.Lifecycle.ModelStorage?,
        fileManager: FileManager = .default
    ) throws -> ZephrAgentRuntime.Lifecycle.ModelCatalog? {
        guard let manifestURL = try manifestURL(for: channel, storage: storage, fileManager: fileManager) else {
            return nil
        }

        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(ZephrAgentRuntimeChannelManifest.self, from: data)
        return try ZephrAgentRuntime.Lifecycle.ModelCatalog(manifest: manifest)
    }

    static func _availableExecutionChoices(
        channel: ZephrAgentRuntime.Lifecycle.ModelChannel,
        storage: ZephrAgentRuntime.Lifecycle.ModelStorage?,
        fileManager: FileManager = .default
    ) throws -> ZephrAgentRuntime.Lifecycle.ExecutionChoices {
        let catalog = try load(channel: channel, storage: storage, fileManager: fileManager)
        return catalog?.roleChoices ?? ZephrAgentRuntime.Lifecycle.ExecutionChoices()
    }

    static func _resolvedExecutionPlan(
        configuration: ZephrAgentRuntime.Lifecycle.Configuration,
        fileManager: FileManager = .default
    ) throws -> ZephrAgentRuntime.Lifecycle.ResolvedExecutionPlan {
        let catalog = try resolved(for: configuration, fileManager: fileManager)
        guard let plan = catalog.resolvedExecutionPlan(
            for: configuration,
            platform: ZephrAgentRuntimePlatform.manifestPlatform
        ) else {
            throw ZephrAgentRuntime.Lifecycle.Error.invalidManifest("Execution plan could not be resolved")
        }
        return plan
    }

    static func resolved(
        for configuration: ZephrAgentRuntime.Lifecycle.Configuration,
        fileManager: FileManager = .default
    ) throws -> ZephrAgentRuntime.Lifecycle.ModelCatalog {
        guard let manifestCatalog = try load(
            channel: configuration.modelChannel,
            storage: configuration.modelStorage,
            fileManager: fileManager
        ) else {
            throw ZephrAgentRuntime.Lifecycle.Error.missingRequiredArtifact("\(configuration.modelChannel.rawValue) channel manifest")
        }
        return manifestCatalog
    }

    private static func manifestURL(
        for channel: ZephrAgentRuntime.Lifecycle.ModelChannel,
        storage: ZephrAgentRuntime.Lifecycle.ModelStorage?,
        fileManager: FileManager
    ) throws -> URL? {
        if channel == .local {
            guard let storage else {
                throw ZephrAgentRuntime.Lifecycle.Error.missingRequiredArtifact("local model storage")
            }
            let url = try localModelRoot(storage: storage, fileManager: fileManager)
                .appendingPathComponent("local.json")
            guard fileManager.fileExists(atPath: url.path) else {
                throw ZephrAgentRuntime.Lifecycle.Error.missingRequiredArtifact("local channel manifest at \(url.path)")
            }
            return url
        }

        return ZephrAgentRuntimeResources.url(forResource: channel.rawValue, withExtension: "json")
    }

    static func localModelRoot(
        storage: ZephrAgentRuntime.Lifecycle.ModelStorage,
        fileManager: FileManager = .default
    ) throws -> URL {
        let base: URL
        switch storage {
        case .appGroup(let identifier):
            guard let groupURL = fileManager.containerURL(forSecurityApplicationGroupIdentifier: identifier) else {
                throw ZephrAgentRuntime.Lifecycle.Error.invalidManifest("App group container is unavailable: \(identifier)")
            }
            base = groupURL
        case .directory(let url):
            return url.standardizedFileURL
        }
        return base
            .appendingPathComponent("Library/Caches", isDirectory: true)
            .appendingPathComponent("ZephrAgentRuntimeLocalModels", isDirectory: true)
    }

    private init(manifest: ZephrAgentRuntimeChannelManifest) throws {
        let artifacts = try manifest.artifacts.compactMap { id, item -> ZephrAgentRuntime.Lifecycle.ModelArtifact? in
            try ZephrAgentRuntime.Lifecycle.ModelArtifact(id: id, manifestArtifact: item)
        }

        self.init(
            gemma4LLMVariants: artifacts.filter { $0.isTextArtifact && $0.family == .gemma4 },
            ragEmbedding: artifacts.first { $0.isEmbeddingArtifact },
            modelArtifacts: artifacts,
            roleChoices: manifest.roleChoices(
                artifacts: artifacts,
                platform: ZephrAgentRuntimePlatform.manifestPlatform
            )
        )
    }

    func artifacts(for configuration: ZephrAgentRuntime.Lifecycle.Configuration) -> [ZephrAgentRuntime.Lifecycle.ModelArtifact] {
        guard let selection = executionSelection(
            for: configuration,
            platform: ZephrAgentRuntimePlatform.manifestPlatform
        ) else {
            return []
        }

        var artifacts: [ZephrAgentRuntime.Lifecycle.ModelArtifact] = []
        if let llm = artifact(id: selection.llm.artifact, roles: [.text], executionPlan: selection.llm.executionPlan) {
            artifacts.append(llm)
        }

        if let rag = selection.rag,
           let ragEmbedding = artifact(id: rag.artifact, roles: [.embedding], executionPlan: rag.executionPlan) {
            artifacts.append(ragEmbedding)
        }

        if let vlm = selection.vlm,
           let gemma4ForVLM = artifact(id: vlm.artifact, roles: [.text], executionPlan: vlm.executionPlan),
           gemma4ForVLM.capabilities.contains(.vlm) {
            artifacts.append(gemma4ForVLM.asVLMArtifact(executionPlan: vlm.executionPlan))
        }

        return artifacts
    }

    func executionSelection(
        for configuration: ZephrAgentRuntime.Lifecycle.Configuration,
        platform: String
    ) -> ZephrAgentRuntime.Lifecycle.ExecutionSelection? {
        guard let llm = roleChoices.selectedLLM(configuration.llmExecutionChoiceID) else {
            return nil
        }
        let rag = roleChoices.selectedRAGEmbedding(configuration.ragEmbeddingExecutionChoiceID)
        let vlm = roleChoices.selectedVLM(configuration.vlmExecutionChoiceID)
        return ZephrAgentRuntime.Lifecycle.ExecutionSelection(
            llm: ZephrAgentRuntime.Lifecycle.ExecutionComponent(choice: llm),
            rag: rag?.isOff == true ? nil : rag.map { ZephrAgentRuntime.Lifecycle.ExecutionComponent(choice: $0) },
            vlm: vlm?.isOff == true ? nil : vlm.map { ZephrAgentRuntime.Lifecycle.ExecutionComponent(choice: $0) }
        )
    }

    func resolvedExecutionPlan(
        for configuration: ZephrAgentRuntime.Lifecycle.Configuration,
        platform: String
    ) -> ZephrAgentRuntime.Lifecycle.ResolvedExecutionPlan? {
        guard let selection = executionSelection(for: configuration, platform: platform) else {
            return nil
        }
        var components: [ZephrAgentRuntime.Lifecycle.ExecutionPlanComponent] = []
        var seen = Set<String>()
        for component in selection.llm.components + (selection.vlm?.components ?? []) + (selection.rag?.components ?? []) {
            let key = "\(component.id)|\(component.artifactID)|\(component.target)"
            guard !seen.contains(key) else { continue }
            seen.insert(key)
            components.append(component)
        }
        return ZephrAgentRuntime.Lifecycle.ResolvedExecutionPlan(choices: selection, components: components)
    }

    private func artifact(
        id: String,
        roles: Set<ZephrAgentRuntime.Lifecycle.ModelArtifact.Role>,
        executionPlan: String
    ) -> ZephrAgentRuntime.Lifecycle.ModelArtifact? {
        guard var artifact = modelArtifacts.first(where: { $0.id == id && roles.contains($0.role) }) else {
            return nil
        }
        artifact.executionPlan = executionPlan
        return artifact
    }

}

// MARK: - ZephrAgentRuntimeChannelManifest
fileprivate struct ZephrAgentRuntimeChannelManifest: Decodable {
    var schemaVersion: Int
    var id: String
    var channel: String
    var artifacts: [String: Artifact]
    var roles: Roles?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case id
        case channel
        case artifacts
        case roles
    }

    struct Roles: Decodable {
        var text: RoleChoices?
        var embedding: RoleChoices?
        var vlm: RoleChoices?

        enum CodingKeys: String, CodingKey {
            case text
            case embedding
            case vlm
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            text = try container.decodeIfPresent(RoleChoices.self, forKey: .text)
            embedding = try container.decodeIfPresent(RoleChoices.self, forKey: .embedding)
            vlm = try container.decodeIfPresent(RoleChoices.self, forKey: .vlm)
        }
    }

    struct RoleChoices: Decodable {
        var platformChoices: [String: [Choice]]?

        enum CodingKeys: String, CodingKey {
            case platformChoices = "platform_choices"
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            platformChoices = try container.decodeIfPresent([String: [Choice]].self, forKey: .platformChoices)
        }

        func choices(for platform: String) -> [Choice] {
            platformChoices?[platform] ?? []
        }
    }

    struct Choice: Decodable {
        var id: String
        var label: String?
        var family: String?
        var requestedPlan: String
        var executionPlan: String
        var components: [Component]?

        enum CodingKeys: String, CodingKey {
            case id
            case label
            case family
            case requestedPlan = "requested_plan"
            case components
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            id = try container.decode(String.self, forKey: .id)
            label = try container.decodeIfPresent(String.self, forKey: .label)
            family = try container.decodeIfPresent(String.self, forKey: .family)
            requestedPlan = try container.decodeIfPresent(String.self, forKey: .requestedPlan)
                ?? "cpu"
            executionPlan = requestedPlan
            components = try container.decodeIfPresent([Component].self, forKey: .components)
        }
    }

    struct Component: Decodable {
        var id: String
        var role: String
        var artifact: String
        var family: String?
        var target: String
        var requestedTarget: String?
        var reason: String?
        var signature: String?

        enum CodingKeys: String, CodingKey {
            case id
            case role
            case artifact
            case family
            case target
            case requestedTarget = "requested_target"
            case reason
            case signature
        }
    }

    struct Artifact: Decodable {
        var role: String
        var title: String
        var family: String?
        var capabilities: [String]?
        var filename: String
        var version: String
        var url: URL?
        var sizeBytes: Int64?
        var sha256: String?
        var metadata: Metadata?

        enum CodingKeys: String, CodingKey {
            case role
            case title
            case family
            case capabilities
            case filename
            case version
            case url
            case sizeBytes = "size_bytes"
            case sha256
            case metadata
        }
    }

    struct Metadata: Decodable {
        var quantization: String?
        var kvCacheMaxLen: Int?
        var hardwareTarget: String?

        enum CodingKeys: String, CodingKey {
            case quantization
            case kvCacheMaxLen = "kv_cache_max_len"
            case hardwareTarget = "hardware_target"
        }
    }

}

// MARK: - ZephrAgentRuntimeChannelManifest Validation
private extension ZephrAgentRuntimeChannelManifest {
    func roleChoices(
        artifacts: [ZephrAgentRuntime.Lifecycle.ModelArtifact],
        platform: String
    ) -> ZephrAgentRuntime.Lifecycle.ExecutionChoices {
        let artifactsByID = Dictionary(uniqueKeysWithValues: artifacts.map { ($0.id, $0) })
        return ZephrAgentRuntime.Lifecycle.ExecutionChoices(
            llm: resolvedChoices(roles?.text, artifacts: artifactsByID, platform: platform),
            ragEmbedding: resolvedChoices(roles?.embedding, artifacts: artifactsByID, platform: platform),
            vlm: resolvedChoices(roles?.vlm, artifacts: artifactsByID, platform: platform)
        )
    }

    private func resolvedChoices(
        _ role: RoleChoices?,
        artifacts: [String: ZephrAgentRuntime.Lifecycle.ModelArtifact],
        platform: String
    ) -> [ZephrAgentRuntime.Lifecycle.ExecutionChoice] {
        guard let role else { return [] }
        return role.choices(for: platform).compactMap { choice in
            guard let components = planComponents(
                choice.components ?? [],
                artifacts: artifacts,
                compatibleHardwareTargets: []
            ) else {
                return nil
            }
            if choice.id == "off" || choice.requestedPlan == "off" || choice.executionPlan == "off" {
                return ZephrAgentRuntime.Lifecycle.ExecutionChoice(
                    id: choice.id,
                    label: choice.label ?? choice.id,
                    family: choice.family.flatMap(ZephrAgentRuntime.Lifecycle.ModelFamily.init(rawValue:)),
                    executionPlan: choice.executionPlan,
                    artifactID: "",
                    requestedPlan: choice.requestedPlan,
                    components: components
                )
            }
            if let artifactID = components.first?.artifactID {
                return ZephrAgentRuntime.Lifecycle.ExecutionChoice(
                    id: choice.id,
                    label: choice.label ?? choice.id,
                    family: choice.family.flatMap(ZephrAgentRuntime.Lifecycle.ModelFamily.init(rawValue:)),
                    executionPlan: choice.executionPlan,
                    artifactID: artifactID,
                    requestedPlan: choice.requestedPlan,
                    components: components
                )
            }
            return nil
        }
    }

    private func planComponents(
        _ components: [ZephrAgentRuntimeChannelManifest.Component],
        artifacts: [String: ZephrAgentRuntime.Lifecycle.ModelArtifact],
        compatibleHardwareTargets: Set<String>
    ) -> [ZephrAgentRuntime.Lifecycle.ExecutionPlanComponent]? {
        var parsed: [ZephrAgentRuntime.Lifecycle.ExecutionPlanComponent] = []
        for component in components {
            guard let artifact = artifacts[component.artifact],
                  artifact.isCompatible(with: compatibleHardwareTargets),
                  let role = ZephrAgentRuntime.Lifecycle.ExecutionPlanComponent.Role(rawValue: component.role) else {
                return nil
            }
            parsed.append(ZephrAgentRuntime.Lifecycle.ExecutionPlanComponent(
                id: component.id,
                role: role,
                artifactID: component.artifact,
                family: component.family,
                target: component.target,
                requestedTarget: component.requestedTarget,
                reason: component.reason,
                signature: component.signature
            ))
        }
        return parsed
    }

}

// MARK: - ZephrAgentRuntime.Lifecycle.ExecutionDelegates
extension ZephrAgentRuntime.Lifecycle.ExecutionDelegates {
    init(
        selection: ZephrAgentRuntime.Lifecycle.ExecutionSelection,
        gemma4Runtime: ZephrAgentRuntime.Lifecycle.Gemma4Options = .automatic,
        diagnosticGemma4: ZephrAgentRuntime.Diagnostics.Gemma4Options = .disabled
    ) {
        self.init(
            llm: selection.llm.executionPlan,
            rag: selection.rag?.executionPlan ?? "cpu",
            vlm: selection.vlm?.executionPlan ?? "gpu",
            gemma4Runtime: gemma4Runtime,
            diagnosticGemma4: diagnosticGemma4
        )
    }
}

// MARK: - ZephrAgentRuntime.Lifecycle.ModelArtifact.Role
extension ZephrAgentRuntime.Lifecycle.ModelArtifact.Role {
    init(manifestValue: String) throws {
        switch manifestValue {
        case "text":
            self = .text
        case "embedding":
            self = .embedding
        default:
            throw ZephrAgentRuntime.Lifecycle.Error.invalidManifest("Unknown artifact role: \(manifestValue)")
        }
    }
}

// MARK: - ZephrAgentRuntime.Lifecycle.ModelArtifact
extension ZephrAgentRuntime.Lifecycle.ModelArtifact {
    fileprivate init?(
        id: String,
        manifestArtifact: ZephrAgentRuntimeChannelManifest.Artifact,
    ) throws {
        let role = try Role(manifestValue: manifestArtifact.role)
        let family: ZephrAgentRuntime.Lifecycle.ModelFamily?
        if let rawFamily = manifestArtifact.family {
            guard let value = ZephrAgentRuntime.Lifecycle.ModelFamily(rawValue: rawFamily) else {
                return nil
            }
            family = value
        } else {
            family = nil
        }
        let capabilities = try manifestArtifact.capabilities?.map { raw -> Capability in
            guard let value = Capability(rawValue: raw) else {
                throw ZephrAgentRuntime.Lifecycle.Error.invalidManifest("Unknown artifact capability: \(raw)")
            }
            return value
        }

        self.init(
            id: id,
            role: role,
            title: manifestArtifact.title,
            family: family,
            capabilities: capabilities,
            filename: manifestArtifact.filename,
            version: manifestArtifact.version,
            quantization: manifestArtifact.metadata?.quantization,
            kvCacheMaxLen: manifestArtifact.metadata?.kvCacheMaxLen,
            hardwareTarget: manifestArtifact.metadata?.hardwareTarget,
            executionPlan: "",
            downloadURL: manifestArtifact.url,
            sizeBytes: manifestArtifact.sizeBytes,
            sha256: manifestArtifact.sha256
        )
    }

    func asVLMArtifact(executionPlan: String) -> ZephrAgentRuntime.Lifecycle.ModelArtifact {
        ZephrAgentRuntime.Lifecycle.ModelArtifact(
            id: "vlm.\(id)",
            role: .vlm,
            title: "Gemma 4 VLM",
            family: .gemma4,
            capabilities: [.vlm],
            filename: filename,
            version: version,
            quantization: quantization,
            kvCacheMaxLen: kvCacheMaxLen,
            hardwareTarget: hardwareTarget,
            executionPlan: executionPlan,
            downloadURL: downloadURL,
            sizeBytes: sizeBytes,
            sha256: sha256
        )
    }

    static func defaultCapabilities(for role: Role) -> [Capability] {
        switch role {
        case .text:
            return [.text]
        case .vlm:
            return [.vlm]
        case .embedding:
            return []
        }
    }
}

private extension ZephrAgentRuntime.Lifecycle.ModelArtifact {
    var isTextArtifact: Bool {
        role == .text
    }

    var isEmbeddingArtifact: Bool {
        role == .embedding
    }

    func isCompatible(with compatibleHardwareTargets: Set<String>) -> Bool {
        guard let hardwareTarget else { return true }
        guard !compatibleHardwareTargets.isEmpty else { return true }
        return compatibleHardwareTargets.contains(hardwareTarget)
    }
}

// MARK: - Runtime

// MARK: - ZephrAgentRuntime.Runtime
@MainActor
extension ZephrAgentRuntime.Runtime {
    nonisolated static func _interruptedInitialization(
        defaults: UserDefaults = .standard
    ) -> ZephrAgentRuntime.Lifecycle.InterruptedInitialization? {
        guard defaults.bool(forKey: InitializationRecoveryKeys.inProgress) else {
            return nil
        }

        return ZephrAgentRuntime.Lifecycle.InterruptedInitialization(
            modelChannel: defaults.string(forKey: InitializationRecoveryKeys.modelChannel)
                .flatMap(ZephrAgentRuntime.Lifecycle.ModelChannel.init(rawValue:)),
            llmExecutionChoiceID: defaults.string(forKey: InitializationRecoveryKeys.llmExecutionChoiceID),
            ragEmbeddingExecutionChoiceID: defaults.string(forKey: InitializationRecoveryKeys.ragEmbeddingExecutionChoiceID),
            vlmExecutionChoiceID: defaults.string(forKey: InitializationRecoveryKeys.vlmExecutionChoiceID),
            startedAt: defaults.object(forKey: InitializationRecoveryKeys.startedAt) as? Date
        )
    }

    nonisolated static func _clearInterruptedInitialization(
        defaults: UserDefaults = .standard
    ) {
        clearInitializationRecoveryMarker(defaults: defaults)
    }

    func _start(
        configuration: ZephrAgentRuntime.Lifecycle.Configuration,
        progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler? = nil
    ) {
        startTask?.cancel()
        let task = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.runPreparation(configuration: configuration, progress: progress)
            } catch {
                // UI callers observe lifecycleState.failure; start intentionally does not throw.
            }
        }
        startTask = task
    }

    func _prepare(
        configuration: ZephrAgentRuntime.Lifecycle.Configuration,
        progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler? = nil
    ) async throws {
        startTask?.cancel()
        try await runPreparation(configuration: configuration, progress: progress)
    }

    func _retry() {
        guard let currentConfiguration else {
            return
        }
        _start(configuration: currentConfiguration)
    }

    func _cancelActiveWork() {
        startTask?.cancel()
        startTask = nil
        engine = nil
        lifecycleState = .idle
    }

    func _describeImage(
        _ image: ZephrAgentRuntime.Tools.RgbImage,
        prompt: String = "Briefly describe this image.",
        maxTokens: Int = 0
    ) async throws -> ZephrAgentRuntime.Tools.VisionResult {
        guard let engine else {
            throw ZephrAgentRuntime.Lifecycle.Error.engineNotReady
        }
        return try await engine.describeImage(
            image,
            prompt: prompt,
            maxTokens: maxTokens
        )
    }

    func _embedText(
        _ text: String,
        taskType: ZephrAgentRuntime.Embeddings.TaskType = .query
    ) async throws -> ZephrAgentRuntime.Embeddings.Embedding {
        guard let engine else {
            throw ZephrAgentRuntime.Lifecycle.Error.engineNotReady
        }
        return try await engine.embedText(text, taskType: taskType)
    }

    func _generateText(
        userMessage: String,
        systemMessage: String = "",
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topK: Int = 40,
        topP: Float = 0.95
    ) async throws -> ZephrAgentRuntime.Diagnostics.TextResult {
        guard let engine else {
            throw ZephrAgentRuntime.Lifecycle.Error.engineNotReady
        }
        return try await engine.generateText(
            userMessage: userMessage,
            systemMessage: systemMessage,
            maxTokens: maxTokens,
            temperature: temperature,
            topK: topK,
            topP: topP
        )
    }

    func _generateTextFromPromptStreaming(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0,
        topK: Int = 40,
        topP: Float = 0.95,
        onChunk: @escaping @Sendable (String) -> Bool
    ) async throws -> ZephrAgentRuntime.Diagnostics.TextResult {
        guard let engine else {
            throw ZephrAgentRuntime.Lifecycle.Error.engineNotReady
        }
        return try await engine.generateTextFromPromptStreaming(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topK: topK,
            topP: topP,
            onChunk: onChunk
        )
    }

    func _collectHeavyJson(
        prompt: String,
        collectActivations: Bool,
        topK: Int
    ) async throws -> String {
        guard let engine else {
            throw ZephrAgentRuntime.Lifecycle.Error.engineNotReady
        }
        return try await engine.collectHeavyJson(
            prompt: prompt,
            collectActivations: collectActivations,
            topK: topK
        )
    }

    func _continueAfterToolResponseStreaming(
        toolResponse: String,
        maxTokens: Int = 256,
        temperature: Float = 0,
        topK: Int = 40,
        topP: Float = 0.95,
        reserveOutputTokens: Int = 0,
        onChunk: @escaping @Sendable (String) -> Bool
    ) async throws -> ZephrAgentRuntime.Diagnostics.TextResult {
        guard let engine else {
            throw ZephrAgentRuntime.Lifecycle.Error.engineNotReady
        }
        return try await engine.continueAfterToolResponseStreaming(
            toolResponse: toolResponse,
            maxTokens: maxTokens,
            temperature: temperature,
            topK: topK,
            topP: topP,
            reserveOutputTokens: reserveOutputTokens,
            onChunk: onChunk
        )
    }

    func _continueToolAwareTextStreaming(
        promptSuffix: String,
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
        return try await engine.continueToolAwareTextStreaming(
            promptSuffix: promptSuffix,
            maxTokens: maxTokens,
            temperature: temperature,
            topK: topK,
            topP: topP,
            reserveOutputTokens: reserveOutputTokens,
            tools: tools,
            onChunk: onChunk
        )
    }

    func _deleteDownloadedModels() {
        startTask?.cancel()
        let storage = currentConfiguration?.modelStorage
        Task { [modelManager] in
            if let storage {
                await modelManager.deleteDownloadedModels(storage: storage)
            } else {
                await modelManager.deleteDownloadedModels()
            }
        }
        lifecycleState = .idle
        lifecycleTimeline = []
        resolvedModels = nil
        detectedModelFamily = nil
        engine = nil
    }

    func _deleteDownloadedModelsAndCaches(cancelActiveWork: Bool = true) async {
        if cancelActiveWork {
            startTask?.cancel()
        }
        let storage = currentConfiguration?.modelStorage
        if let storage {
            await modelManager.deleteDownloadedModelsAndCaches(storage: storage)
        } else {
            await modelManager.deleteDownloadedModelsAndCaches()
        }
        lifecycleState = .idle
        lifecycleTimeline = []
        resolvedModels = nil
        detectedModelFamily = nil
        engine = nil
    }

    func _drainModelTimings() async -> [ZephrAgentRuntime.Diagnostics.ModelTiming] {
        guard let engine else { return [] }
        return await engine.drainModelTimings()
    }

    @discardableResult
    func _shutdown() async -> ZephrAgentRuntime.Lifecycle.Event {
        startTask?.cancel()

        let start = DispatchTime.now()
        let modelTimings = await engine?.shutdown() ?? []
        engine = nil

        let elapsed = Int64(Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000)
        let event = ZephrAgentRuntime.Lifecycle.Event(
            kind: .teardown,
            status: .passed,
            title: "Agent released",
            detail: "models released",
            durationMs: elapsed,
            modelTimings: modelTimings
        )
        upsertLifecycleEvent(event)
        lifecycleState = .idle
        return event
    }

    private func runPreparation(
        configuration: ZephrAgentRuntime.Lifecycle.Configuration,
        progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler? = nil
    ) async throws {
        currentConfiguration = configuration
        engine = nil
        resolvedModels = nil
        detectedModelFamily = nil
        lifecycleTimeline = []
        lifecycleState = ZephrAgentRuntime.Lifecycle.State(
            phase: .resolvingModels,
            message: "Resolving models"
        )
        let modelPreparationEventID = UUID()
        let modelPreparationStart = DispatchTime.now()
        var modelPreparationFinished = false

        func modelPreparationDurationMs() -> Int64 {
            Int64(Double(DispatchTime.now().uptimeNanoseconds - modelPreparationStart.uptimeNanoseconds) / 1_000_000)
        }

        func upsertModelPreparationEvent(
            status: ZephrAgentRuntime.Lifecycle.Event.Status,
            title: String,
            detail: String,
            artifacts: [ZephrAgentRuntime.Lifecycle.ArtifactState] = lifecycleState.artifacts,
            durationMs: Int64? = nil
        ) {
            upsertLifecycleEvent(ZephrAgentRuntime.Lifecycle.Event(
                id: modelPreparationEventID,
                kind: .modelPreparation,
                status: status,
                title: title,
                detail: detail,
                durationMs: durationMs,
                artifacts: artifacts
            ))
        }

        upsertModelPreparationEvent(
            status: .running,
            title: "Preparing models",
            detail: "Resolving models"
        )

        do {
            let resolved = try await modelManager.resolveModels(configuration: configuration) { [weak self] phase, message, artifacts in
                await MainActor.run {
                    self?.lifecycleState = ZephrAgentRuntime.Lifecycle.State(
                        phase: phase,
                        message: message,
                        artifacts: artifacts
                    )
                    self?.upsertLifecycleEvent(ZephrAgentRuntime.Lifecycle.Event(
                        id: modelPreparationEventID,
                        kind: .modelPreparation,
                        status: phase == .failed ? .failed : .running,
                        title: Self.modelPreparationTitle(phase: phase),
                        detail: message,
                        artifacts: artifacts
                    ))
                }
                await progress?(phase, message, artifacts)
            }
            let preparedArtifacts = lifecycleState.artifacts
            upsertModelPreparationEvent(
                status: .passed,
                title: "Models ready",
                detail: "\(resolved.llmModelURL.lastPathComponent) ready",
                artifacts: preparedArtifacts,
                durationMs: modelPreparationDurationMs()
            )
            modelPreparationFinished = true

            await progress?(
                .initializingEngine,
                "Compiling and loading models",
                preparedArtifacts
            )
            lifecycleState = ZephrAgentRuntime.Lifecycle.State(
                phase: .initializingEngine,
                message: "Compiling and loading models",
                artifacts: preparedArtifacts
            )

            Self.markInitializationRecoveryMarker(configuration: configuration)
            if configuration.crashDuringInitializationForTesting {
                fatalError("Forced crash during ZephrAgentRuntime.Runtime initialization for startup recovery testing.")
            }
            do {
                let initialized = try await Task.detached(priority: .userInitiated) {
                    let detectedFamily = Self.detectModelFamily(at: resolved.llmModelURL)
                    var resolvedWithDetection = resolved
                    resolvedWithDetection.detectedLLMFamily = detectedFamily
                    let execution = ZephrAgentRuntime.Lifecycle.ExecutionDelegates(
                        selection: resolved.executionSelection,
                        gemma4Runtime: configuration.gemma4Runtime,
                        diagnosticGemma4: configuration.diagnosticGemma4
                    )
                    let engine = try ZephrAgentRuntimeEngine(
                        llmModelPath: resolved.llmModelURL.path,
                        execution: execution,
                        numThreads: configuration.numThreads,
                        litertCompilationCacheEnabled: configuration.litertCompilationCacheEnabled,
                        ragEmbeddingPath: resolved.ragEmbeddingURL?.path,
                        vlmModelPath: resolved.vlmModelURL?.path
                    )
                    return (detectedFamily, resolvedWithDetection, engine)
                }.value

                detectedModelFamily = initialized.0
                resolvedModels = initialized.1
                engine = initialized.2
                Self.clearInitializationRecoveryMarker()
            } catch {
                Self.clearInitializationRecoveryMarker()
                throw error
            }

            lifecycleState = .ready(artifacts: lifecycleState.artifacts)
        } catch is CancellationError {
            lifecycleState = .idle
            throw CancellationError()
        } catch {
            let message = String(describing: error)
            let failedArtifacts = markFailure(in: lifecycleState.artifacts, message: message)
            if !modelPreparationFinished {
                upsertModelPreparationEvent(
                    status: .failed,
                    title: "Model preparation failed",
                    detail: message,
                    artifacts: failedArtifacts,
                    durationMs: modelPreparationDurationMs()
                )
            }
            lifecycleState = ZephrAgentRuntime.Lifecycle.State(
                phase: .failed,
                message: "Agent initialization failed",
                artifacts: failedArtifacts,
                canRetry: true,
                errorMessage: message
            )
            throw error
        }
    }

    private nonisolated static func detectModelFamily(at url: URL) -> String? {
        ZephrAgentRuntimeCBridge.detectModelFamily(at: url)
    }

    private nonisolated static func modelPreparationTitle(phase: ZephrAgentRuntime.Lifecycle.Phase) -> String {
        switch phase {
        case .resolvingModels:
            return "Resolving models"
        case .downloadingModels:
            return "Downloading models"
        case .verifyingModels:
            return "Verifying models"
        case .initializingEngine, .ready:
            return "Models ready"
        case .failed:
            return "Model preparation failed"
        case .idle, .loadingRuntimeData:
            return "Preparing models"
        }
    }

    private nonisolated static func markInitializationRecoveryMarker(
        configuration: ZephrAgentRuntime.Lifecycle.Configuration,
        defaults: UserDefaults = .standard
    ) {
        defaults.set(true, forKey: InitializationRecoveryKeys.inProgress)
        defaults.set(Date(), forKey: InitializationRecoveryKeys.startedAt)
        defaults.set(configuration.modelChannel.rawValue, forKey: InitializationRecoveryKeys.modelChannel)
        defaults.set(configuration.llmExecutionChoiceID, forKey: InitializationRecoveryKeys.llmExecutionChoiceID)
        defaults.set(configuration.ragEmbeddingExecutionChoiceID, forKey: InitializationRecoveryKeys.ragEmbeddingExecutionChoiceID)
        defaults.set(configuration.vlmExecutionChoiceID, forKey: InitializationRecoveryKeys.vlmExecutionChoiceID)
    }

    private nonisolated static func clearInitializationRecoveryMarker(
        defaults: UserDefaults = .standard
    ) {
        defaults.removeObject(forKey: InitializationRecoveryKeys.inProgress)
        defaults.removeObject(forKey: InitializationRecoveryKeys.startedAt)
        defaults.removeObject(forKey: InitializationRecoveryKeys.modelChannel)
        defaults.removeObject(forKey: InitializationRecoveryKeys.llmExecutionChoiceID)
        defaults.removeObject(forKey: InitializationRecoveryKeys.ragEmbeddingExecutionChoiceID)
        defaults.removeObject(forKey: InitializationRecoveryKeys.vlmExecutionChoiceID)
    }

    private enum InitializationRecoveryKeys {
        static let inProgress = "xyz.zephr.agent.runtime.initialization.inProgress"
        static let startedAt = "xyz.zephr.agent.runtime.initialization.startedAt"
        static let modelChannel = "xyz.zephr.agent.runtime.initialization.modelChannel"
        static let llmExecutionChoiceID = "xyz.zephr.agent.runtime.initialization.llmExecutionChoiceID"
        static let ragEmbeddingExecutionChoiceID = "xyz.zephr.agent.runtime.initialization.ragEmbeddingExecutionChoiceID"
        static let vlmExecutionChoiceID = "xyz.zephr.agent.runtime.initialization.vlmExecutionChoiceID"
    }

    private func upsertLifecycleEvent(_ event: ZephrAgentRuntime.Lifecycle.Event) {
        if let index = lifecycleTimeline.firstIndex(where: { $0.id == event.id }) {
            lifecycleTimeline[index] = event
        } else {
            lifecycleTimeline.append(event)
        }
    }

    private func markFailure(
        in artifacts: [ZephrAgentRuntime.Lifecycle.ArtifactState],
        message: String
    ) -> [ZephrAgentRuntime.Lifecycle.ArtifactState] {
        artifacts.map { artifact in
            guard artifact.phase != .ready && artifact.phase != .skipped else {
                return artifact
            }
            var failed = artifact
            failed.phase = .failed
            failed.detail = failed.detail ?? message
            return failed
        }
    }
}

// MARK: - Model manager

extension ZephrAgentRuntime.ModelManager {
    private enum ArtifactResolutionMode {
        case localOnly
        case downloadIfMissing
    }

    func _resolveModels(
        configuration: ZephrAgentRuntime.Lifecycle.Configuration,
        progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler? = nil
    ) async throws -> ZephrAgentRuntime.Lifecycle.ResolvedModels {
        let catalog = try ZephrAgentRuntime.Lifecycle.ModelCatalog.resolved(for: configuration, fileManager: fileManager)
        guard let executionSelection = catalog.executionSelection(
            for: configuration,
            platform: ZephrAgentRuntimePlatform.manifestPlatform
        ) else {
            throw ZephrAgentRuntime.Lifecycle.Error.invalidManifest("Execution selection could not be resolved")
        }
        let executionPlan = catalog.resolvedExecutionPlan(
            for: configuration,
            platform: ZephrAgentRuntimePlatform.manifestPlatform
        )
        let artifacts = catalog.artifacts(for: configuration)
        let storageDirectory = try artifactStorageRoot(for: configuration)
        let artifactResolutionMode: ArtifactResolutionMode
        if configuration.modelChannel == .local {
            artifactResolutionMode = .localOnly
        } else {
            artifactResolutionMode = .downloadIfMissing
        }
        let initialStates = artifacts.map { artifact in
            ZephrAgentRuntime.Lifecycle.ArtifactState(
                id: artifact.id,
                title: artifact.title,
                purpose: artifact.role.progressPurpose,
                phase: .pending,
                totalBytes: artifact.sizeBytes,
                version: artifact.version
            )
        }
        let progressState = ZephrAgentRuntimeProgressAccumulator(
            states: initialStates,
            progress: progress
        )

        try ensureDirectoryExists(storageDirectory)

        var resolved = [ZephrAgentRuntime.Lifecycle.ModelArtifact.Role: URL]()
        let artifactGroups = Self.groupedArtifactsByDownloadIdentity(artifacts)

        try await withThrowingTaskGroup(
            of: [(ZephrAgentRuntime.Lifecycle.ModelArtifact.Role, URL)].self
        ) { group in
            for artifactGroup in artifactGroups {
                group.addTask {
                    try Task.checkCancellation()
                    let artifact = artifactGroup.primary
                    for alias in artifactGroup.aliases {
                        await progressState.update(
                            alias,
                            phase: .waitingForSharedDownload,
                            detail: "Using \(artifact.title)'s downloaded file",
                            lifecyclePhase: .resolvingModels,
                            message: "Preparing shared model download"
                        )
                    }

                    await progressState.update(
                        artifact,
                        phase: .checkingLocalFile,
                        lifecyclePhase: .resolvingModels,
                        message: "Checking \(artifact.title)"
                    )

                    do {
                        let url = try await self.resolveArtifact(
                            artifact,
                            storageDirectory: storageDirectory,
                            mode: artifactResolutionMode
                        ) { downloaded, total in
                            await progressState.update(
                                artifact,
                                phase: .downloading,
                                downloadedBytes: downloaded,
                                totalBytes: total,
                                lifecyclePhase: .downloadingModels,
                                message: "Downloading \(artifact.title)"
                            )
                        } verifying: {
                            await progressState.update(
                                artifact,
                                phase: .verifying,
                                lifecyclePhase: .verifyingModels,
                                message: "Verifying \(artifact.title)"
                            )
                        }

                        await progressState.update(
                            artifact,
                            phase: .ready,
                            downloadedBytes: artifact.sizeBytes ?? 0,
                            detail: url.path,
                            lifecyclePhase: .resolvingModels,
                            message: "\(artifact.title) ready"
                        )
                        for alias in artifactGroup.aliases {
                            await progressState.update(
                                alias,
                                phase: .ready,
                                downloadedBytes: alias.sizeBytes ?? artifact.sizeBytes ?? 0,
                                detail: url.path,
                                lifecyclePhase: .resolvingModels,
                                message: "\(alias.title) ready"
                            )
                        }

                        return ([artifact] + artifactGroup.aliases).map { ($0.role, url) }
                    } catch {
                        await progressState.update(
                            artifact,
                            phase: .failed,
                            detail: String(describing: error),
                            lifecyclePhase: .failed,
                            message: "\(artifact.title) failed"
                        )
                        for alias in artifactGroup.aliases {
                            await progressState.update(
                                alias,
                                phase: .failed,
                                detail: String(describing: error),
                                lifecyclePhase: .failed,
                                message: "\(alias.title) failed"
                            )
                        }
                        throw error
                    }
                }
            }

            for try await results in group {
                for (role, url) in results {
                    resolved[role] = url
                }
            }
        }

        guard let llmModelURL = resolved[.text] else {
            throw ZephrAgentRuntime.Lifecycle.Error.missingRequiredArtifact("LLM")
        }

        return ZephrAgentRuntime.Lifecycle.ResolvedModels(
            llmModelURL: llmModelURL,
            ragEmbeddingURL: resolved[.embedding],
            vlmModelURL: resolved[.vlm],
            executionSelection: executionSelection,
            executionPlan: executionPlan,
            detectedLLMFamily: nil
        )
    }

    private struct ArtifactResolutionGroup: Sendable {
        var primary: ZephrAgentRuntime.Lifecycle.ModelArtifact
        var aliases: [ZephrAgentRuntime.Lifecycle.ModelArtifact]
    }

    private static func groupedArtifactsByDownloadIdentity(
        _ artifacts: [ZephrAgentRuntime.Lifecycle.ModelArtifact]
    ) -> [ArtifactResolutionGroup] {
        var orderedKeys: [String] = []
        var groupedArtifacts: [String: [ZephrAgentRuntime.Lifecycle.ModelArtifact]] = [:]

        for artifact in artifacts {
            let key = downloadIdentity(for: artifact)
            if groupedArtifacts[key] == nil {
                orderedKeys.append(key)
                groupedArtifacts[key] = []
            }
            groupedArtifacts[key]?.append(artifact)
        }

        return orderedKeys.compactMap { key in
            guard var artifacts = groupedArtifacts[key],
                  !artifacts.isEmpty else {
                return nil
            }
            let primary = artifacts.removeFirst()
            return ArtifactResolutionGroup(primary: primary, aliases: artifacts)
        }
    }

    private static func downloadIdentity(for artifact: ZephrAgentRuntime.Lifecycle.ModelArtifact) -> String {
        if let sha256 = artifact.sha256, !sha256.isEmpty {
            return sha256.lowercased()
        }

        return [
            artifact.downloadURL?.absoluteString ?? "",
            artifact.filename,
        ].joined(separator: "|")
    }

    func _deleteDownloadedModels() {
        try? fileManager.removeItem(at: Self.defaultStorageDirectory(fileManager: fileManager))
    }

    func _deleteDownloadedModels(storage: ZephrAgentRuntime.Lifecycle.ModelStorage) {
        try? fileManager.removeItem(at: Self.defaultStorageDirectory(fileManager: fileManager))
        if let groupDirectory = try? Self.downloadStorageDirectory(
            storage: storage,
            override: storageDirectoryOverride,
            fileManager: fileManager
        ) {
            try? fileManager.removeItem(at: groupDirectory.appendingPathComponent("blobs", isDirectory: true))
        }
    }

    func _deleteDownloadedModelsAndCaches() {
        _deleteDownloadedModels()
        _deleteCaches()
    }

    func _deleteDownloadedModelsAndCaches(storage: ZephrAgentRuntime.Lifecycle.ModelStorage) {
        _deleteDownloadedModels(storage: storage)
        _deleteCaches()
    }

    func _deleteCaches() {
        ZephrAgentRuntimeEngine.clearLiteRTCompilationCache(fileManager: fileManager)
    }

    private func resolveArtifact(
        _ artifact: ZephrAgentRuntime.Lifecycle.ModelArtifact,
        storageDirectory: URL,
        mode: ArtifactResolutionMode,
        progress: @escaping @Sendable (Int64, Int64?) async -> Void,
        verifying: () async -> Void
    ) async throws -> URL {
        let finalURL = try artifactBlobURL(root: storageDirectory, artifact: artifact)
        let partialURL = finalURL.deletingLastPathComponent().appendingPathComponent("\(finalURL.lastPathComponent).partial")

        if fileManager.fileExists(atPath: finalURL.path) {
            await verifying()
            do {
                try verifyArtifact(artifact, at: finalURL)
                return finalURL
            } catch {
                guard isChecksumMismatch(error) else {
                    throw error
                }
                try? fileManager.removeItem(at: finalURL)
                try? fileManager.removeItem(at: partialURL)
            }
        }

        if mode == .localOnly {
            throw ZephrAgentRuntime.Lifecycle.Error.missingRequiredArtifact(artifact.title)
        }

        guard let artifactURL = artifact.downloadURL else {
            throw ZephrAgentRuntime.Lifecycle.Error.noDownloadURL(artifact.title)
        }
        guard artifactURL.scheme == "http" || artifactURL.scheme == "https" else {
            throw ZephrAgentRuntime.Lifecycle.Error.invalidManifest("Unsupported artifact URL scheme: \(artifactURL.absoluteString)")
        }
        try ensureDirectoryExists(finalURL.deletingLastPathComponent())

        var existingBytes = fileSize(at: partialURL) ?? 0
        if let expectedSize = artifact.sizeBytes, existingBytes >= expectedSize {
            await verifying()
            do {
                try verifyArtifact(artifact, at: partialURL)
                try? fileManager.removeItem(at: finalURL)
                try fileManager.moveItem(at: partialURL, to: finalURL)
                return finalURL
            } catch {
                guard isChecksumMismatch(error) else {
                    throw error
                }
                try? fileManager.removeItem(at: partialURL)
                existingBytes = 0
            }
        }

        var request = URLRequest(url: artifactURL)
        if existingBytes > 0 {
            request.setValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range")
        }

        let downloader = ZephrAgentRuntimeDownloadDelegate(
            partialURL: partialURL,
            existingBytes: existingBytes,
            artifactSizeBytes: artifact.sizeBytes,
            fileManager: fileManager,
            progress: progress
        )
        let session = URLSession(configuration: .default, delegate: downloader, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        let response = try await downloader.download(request, session: session)
        guard let http = response as? HTTPURLResponse else {
            throw ZephrAgentRuntime.Lifecycle.Error.downloadFailed("No HTTP response for \(artifact.title)")
        }
        guard (200...299).contains(http.statusCode) else {
            throw ZephrAgentRuntime.Lifecycle.Error.downloadFailed("HTTP \(http.statusCode) downloading \(artifact.title)")
        }

        let appending = http.statusCode == 206 && existingBytes > 0
        let expectedTotal = artifact.sizeBytes ?? responseExpectedBytes(http: http, existingBytes: appending ? existingBytes : 0)
        await progress(fileSize(at: partialURL) ?? 0, expectedTotal)

        await verifying()
        do {
            try verifyArtifact(artifact, at: partialURL)
        } catch {
            if isChecksumMismatch(error) {
                try? fileManager.removeItem(at: partialURL)
            }
            throw error
        }

        try? fileManager.removeItem(at: finalURL)
        try fileManager.moveItem(at: partialURL, to: finalURL)
        return finalURL
    }

    private func artifactBlobURL(root: URL, artifact: ZephrAgentRuntime.Lifecycle.ModelArtifact) throws -> URL {
        guard let sha256 = artifact.sha256, !sha256.isEmpty else {
            throw ZephrAgentRuntime.Lifecycle.Error.invalidManifest("Artifact \(artifact.id) has no sha256")
        }
        return root
            .appendingPathComponent("blobs", isDirectory: true)
            .appendingPathComponent("sha256", isDirectory: true)
            .appendingPathComponent(sha256.lowercased())
    }

    private func verifyArtifact(_ artifact: ZephrAgentRuntime.Lifecycle.ModelArtifact, at url: URL) throws {
        guard let expected = artifact.sha256, !expected.isEmpty else {
            return
        }

        let actual = try sha256(url: url)
        guard actual.lowercased() == expected.lowercased() else {
            throw ZephrAgentRuntime.Lifecycle.Error.checksumMismatch(artifact.title)
        }
    }

    private func isChecksumMismatch(_ error: Error) -> Bool {
        if case ZephrAgentRuntime.Lifecycle.Error.checksumMismatch = error {
            return true
        }
        return false
    }

    private func sha256(url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var hasher = SHA256()
        while autoreleasepool(invoking: {
            let data = handle.readData(ofLength: 1024 * 1024)
            guard !data.isEmpty else {
                return false
            }
            hasher.update(data: data)
            return true
        }) {}

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func responseExpectedBytes(http: HTTPURLResponse, existingBytes: Int64) -> Int64? {
        let contentLength = http.value(forHTTPHeaderField: "Content-Length").flatMap(Int64.init)
        return contentLength.map { existingBytes + $0 }
    }

    private func appendFile(_ sourceURL: URL, to destinationURL: URL) throws {
        if !fileManager.fileExists(atPath: destinationURL.path) {
            fileManager.createFile(atPath: destinationURL.path, contents: nil)
        }

        let source = try FileHandle(forReadingFrom: sourceURL)
        defer { try? source.close() }

        let destination = try FileHandle(forWritingTo: destinationURL)
        defer { try? destination.close() }
        try destination.seekToEnd()

        while true {
            let data = source.readData(ofLength: 1024 * 1024)
            guard !data.isEmpty else { break }
            try destination.write(contentsOf: data)
        }

        try? fileManager.removeItem(at: sourceURL)
    }

    private func fileSize(at url: URL) -> Int64? {
        guard let attrs = try? fileManager.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? NSNumber else {
            return nil
        }
        return size.int64Value
    }

    private func ensureDirectoryExists(_ url: URL) throws {
        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
    }

    private func artifactStorageRoot(for configuration: ZephrAgentRuntime.Lifecycle.Configuration) throws -> URL {
        if configuration.modelChannel == .local {
            guard let storage = configuration.modelStorage else {
                throw ZephrAgentRuntime.Lifecycle.Error.missingRequiredArtifact("local model storage")
            }
            return try ZephrAgentRuntime.Lifecycle.ModelCatalog.localModelRoot(storage: storage, fileManager: fileManager)
        }
        return try Self.downloadStorageDirectory(
            storage: configuration.modelStorage,
            override: storageDirectoryOverride,
            fileManager: fileManager
        )
    }

    private static func downloadStorageDirectory(
        storage: ZephrAgentRuntime.Lifecycle.ModelStorage?,
        override: URL?,
        fileManager: FileManager
    ) throws -> URL {
        if let override {
            return override
        }
        guard let storage else {
            return defaultStorageDirectory(fileManager: fileManager)
        }
        return try ZephrAgentRuntime.Lifecycle.ModelCatalog.localModelRoot(storage: storage, fileManager: fileManager)
    }

    private static func defaultStorageDirectory(fileManager: FileManager) -> URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
        let base = appSupport ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appendingPathComponent("ZephrAgentRuntime/Models", isDirectory: true)
    }

}

private extension ZephrAgentRuntime.Lifecycle.ModelArtifact.Role {
    var progressPurpose: String {
        switch self {
        case .text:
            return "TEXT"
        case .embedding:
            return "EMBEDDING"
        case .vlm:
            return "VLM"
        }
    }
}

// MARK: - ZephrAgentRuntimeDownloadDelegate
private final class ZephrAgentRuntimeDownloadDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private static let minimumProgressEmitIntervalNanos: UInt64 = 200_000_000

    private let partialURL: URL
    private let initialExistingBytes: Int64
    private let artifactSizeBytes: Int64?
    private let fileManager: FileManager
    private let progress: @Sendable (Int64, Int64?) async -> Void
    private let lock = NSLock()

    private var continuation: CheckedContinuation<URLResponse, Error>?
    private var task: URLSessionDataTask?
    private var fileHandle: FileHandle?
    private var response: URLResponse?
    private var completionError: Error?
    private var downloadedBytes: Int64 = 0
    private var totalBytes: Int64?
    private var lastProgressEmitNanos: UInt64 = 0

    init(
        partialURL: URL,
        existingBytes: Int64,
        artifactSizeBytes: Int64?,
        fileManager: FileManager,
        progress: @escaping @Sendable (Int64, Int64?) async -> Void
    ) {
        self.partialURL = partialURL
        self.initialExistingBytes = existingBytes
        self.artifactSizeBytes = artifactSizeBytes
        self.fileManager = fileManager
        self.progress = progress
    }

    func download(_ request: URLRequest, session: URLSession) async throws -> URLResponse {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<URLResponse, Error>) in
                let task = session.dataTask(with: request)
                lock.lock()
                self.continuation = continuation
                self.task = task
                lock.unlock()
                task.resume()
            }
        } onCancel: {
            lock.lock()
            let task = self.task
            lock.unlock()
            task?.cancel()
        }
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let http = response as? HTTPURLResponse else {
            fail(ZephrAgentRuntime.Lifecycle.Error.downloadFailed("No HTTP response while downloading model artifact"))
            completionHandler(.cancel)
            return
        }
        guard (200...299).contains(http.statusCode) else {
            fail(ZephrAgentRuntime.Lifecycle.Error.downloadFailed("HTTP \(http.statusCode) downloading model artifact"))
            completionHandler(.cancel)
            return
        }

        let appending = http.statusCode == 206 && initialExistingBytes > 0
        do {
            if !appending {
                try? fileManager.removeItem(at: partialURL)
                downloadedBytes = 0
            } else {
                downloadedBytes = initialExistingBytes
            }
            if !fileManager.fileExists(atPath: partialURL.path) {
                fileManager.createFile(atPath: partialURL.path, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: partialURL)
            try handle.seekToEnd()

            lock.lock()
            self.fileHandle = handle
            self.response = response
            self.totalBytes = artifactSizeBytes ?? Self.responseExpectedBytes(
                http: http,
                existingBytes: appending ? initialExistingBytes : 0
            )
            let progressDownloaded = downloadedBytes
            let progressTotal = totalBytes
            lock.unlock()

            Task {
                await progress(progressDownloaded, progressTotal)
            }
            completionHandler(.allow)
        } catch {
            fail(error)
            completionHandler(.cancel)
        }
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive data: Data
    ) {
        do {
            lock.lock()
            let handle = fileHandle
            lock.unlock()

            guard let handle else {
                throw ZephrAgentRuntime.Lifecycle.Error.downloadFailed("Download started before output file was ready")
            }
            try handle.write(contentsOf: data)

            let now = DispatchTime.now().uptimeNanoseconds
            lock.lock()
            downloadedBytes += Int64(data.count)
            let currentDownloaded = downloadedBytes
            let currentTotal = totalBytes
            let shouldEmit = lastProgressEmitNanos == 0 || now - lastProgressEmitNanos >= Self.minimumProgressEmitIntervalNanos
            if shouldEmit {
                lastProgressEmitNanos = now
            }
            lock.unlock()

            guard shouldEmit else { return }
            Task {
                await progress(currentDownloaded, currentTotal)
            }
        } catch {
            fail(error)
            dataTask.cancel()
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        lock.lock()
        let completionError = self.completionError
        let response = self.response ?? task.response
        let continuation = self.continuation
        self.continuation = nil
        self.task = nil
        let handle = self.fileHandle
        self.fileHandle = nil
        lock.unlock()

        try? handle?.close()

        if let completionError {
            continuation?.resume(throwing: completionError)
            return
        }

        if let error {
            continuation?.resume(throwing: error)
            return
        }

        guard let response else {
            continuation?.resume(
                throwing: ZephrAgentRuntime.Lifecycle.Error.downloadFailed("Download completed without a response")
            )
            return
        }

        continuation?.resume(returning: response)
    }

    private func fail(_ error: Error) {
        lock.lock()
        if completionError == nil {
            completionError = error
        }
        lock.unlock()
    }

    private static func responseExpectedBytes(http: HTTPURLResponse, existingBytes: Int64) -> Int64? {
        let contentLength = http.value(forHTTPHeaderField: "Content-Length").flatMap(Int64.init)
        return contentLength.map { existingBytes + $0 }
    }
}

// MARK: - ZephrAgentRuntimeProgressAccumulator
private actor ZephrAgentRuntimeProgressAccumulator {
    private static let minimumDownloadEmitIntervalNanos: UInt64 = 200_000_000

    private struct DownloadRateSample {
        var downloadedBytes: Int64
        var timestampNanos: UInt64
        var bytesPerSecond: Double?
    }

    private var states: [ZephrAgentRuntime.Lifecycle.ArtifactState]
    private let progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler?
    private var lastDownloadEmitNanos: UInt64 = 0
    private var downloadRateSamples: [String: DownloadRateSample] = [:]

    init(
        states: [ZephrAgentRuntime.Lifecycle.ArtifactState],
        progress: ZephrAgentRuntime.Lifecycle.ModelProgressHandler?
    ) {
        self.states = states
        self.progress = progress
    }

    func update(
        _ artifact: ZephrAgentRuntime.Lifecycle.ModelArtifact,
        phase: ZephrAgentRuntime.Lifecycle.ArtifactState.Phase,
        downloadedBytes: Int64 = 0,
        totalBytes: Int64? = nil,
        detail: String? = nil,
        lifecyclePhase: ZephrAgentRuntime.Lifecycle.Phase,
        message: String
    ) async {
        let phaseChanged: Bool
        if let index = states.firstIndex(where: { $0.id == artifact.id }) {
            phaseChanged = states[index].phase != phase
            states[index].phase = phase
            states[index].downloadedBytes = downloadedBytes
            states[index].totalBytes = totalBytes ?? states[index].totalBytes
            states[index].downloadBytesPerSecond = downloadRate(
                artifactID: artifact.id,
                phase: phase,
                downloadedBytes: downloadedBytes
            )
            states[index].detail = detail
        } else {
            phaseChanged = true
        }

        guard shouldEmit(phase: phase, phaseChanged: phaseChanged) else { return }
        await progress?(lifecyclePhase, message, states)
    }

    private func shouldEmit(
        phase: ZephrAgentRuntime.Lifecycle.ArtifactState.Phase,
        phaseChanged: Bool
    ) -> Bool {
        guard phase == .downloading else { return true }
        guard !phaseChanged else {
            lastDownloadEmitNanos = DispatchTime.now().uptimeNanoseconds
            return true
        }

        let now = DispatchTime.now().uptimeNanoseconds
        guard now - lastDownloadEmitNanos >= Self.minimumDownloadEmitIntervalNanos else {
            return false
        }
        lastDownloadEmitNanos = now
        return true
    }

    private func downloadRate(
        artifactID: String,
        phase: ZephrAgentRuntime.Lifecycle.ArtifactState.Phase,
        downloadedBytes: Int64
    ) -> Double? {
        guard phase == .downloading, downloadedBytes > 0 else {
            if phase != .downloading {
                downloadRateSamples[artifactID] = nil
            }
            return nil
        }

        let now = DispatchTime.now().uptimeNanoseconds
        let previous = downloadRateSamples[artifactID]
        var bytesPerSecond = previous?.bytesPerSecond
        if let previous,
           downloadedBytes > previous.downloadedBytes,
           now > previous.timestampNanos {
            let elapsedSeconds = Double(now - previous.timestampNanos) / 1_000_000_000
            let instantaneous = Double(downloadedBytes - previous.downloadedBytes) / elapsedSeconds
            if let existing = previous.bytesPerSecond {
                bytesPerSecond = (existing * 0.75) + (instantaneous * 0.25)
            } else {
                bytesPerSecond = instantaneous
            }
        }
        downloadRateSamples[artifactID] = DownloadRateSample(
            downloadedBytes: downloadedBytes,
            timestampNanos: now,
            bytesPerSecond: bytesPerSecond
        )
        return bytesPerSecond
    }
}
