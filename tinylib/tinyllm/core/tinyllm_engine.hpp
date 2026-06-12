// TinyLLM engine — shared LiteRT environment with pluggable capabilities.
//
// Single init() creates the LiteRT environment and loads whichever
// capabilities are requested (LLM, RAG, VLM in the future).

#pragma once

#include "tinyllm_inference.hpp"
#include "tinyllm_rag.hpp"

#include <chrono>
#include <memory>

// MARK: - Engine

struct TinyLLMExecutionConfig {
    std::string llm_plan = "cpu";
    std::string rag_plan = "cpu";
    std::string vlm_plan = "cpu";
    Gemma4RuntimeConfig gemma4_runtime;
    Gemma4DiagnosticConfig diagnostic_gemma4;
};

struct VlmExecutionTargets {
    HardwareTarget encoder = HardwareTarget::CPU;
    HardwareTarget decoder = HardwareTarget::CPU;
};

struct ModelLifecycleTiming {
    std::string component;
    std::string action;
    std::string detail;
    int64_t duration_ms = 0;
    bool ok = true;
};

static inline VlmExecutionTargets vlm_execution_targets(Platform platform,
                                                        const std::string& plan) {
    const HardwareTarget requested = parse_hardware_target(plan.c_str());
    // iOS Metal currently fails to prepare the Gemma4 vision encoder after
    // other model compilations. Keep the public choice as "VLM GPU", but run
    // the text decoder on GPU and the vision encoder on CPU until stable.
    if (platform == Platform::APPLE_OS && requested == HardwareTarget::GPU) {
        return VlmExecutionTargets{HardwareTarget::CPU, HardwareTarget::GPU};
    }
    return VlmExecutionTargets{requested, requested};
}

static inline ModelFamily detect_llm_family_from_mapped_model(const MappedFile& model) {
    if (model.size >= 8 && memcmp(model.data, "LITERTLM", 8) == 0) {
        auto sections = parse_litertlm_bundle(model.data, model.size);
        if (find_bundle_section(sections, BundleModelType::EMBEDDER))
            return ModelFamily::GEMMA4;
    }
    return ModelFamily::GEMMA4;
}

static inline int llm_environment_accelerator_mask(ModelFamily family,
                                                   const std::string& plan) {
    if (family == ModelFamily::GEMMA4)
        return gemma4::environment_accelerator_mask(parse_hardware_target(plan.c_str()));
    return kLiteRtHwAcceleratorNone;
}

struct VlmEngine {
    gemma4::Runtime* runtime = nullptr;

    void attach(gemma4::Runtime* value) { runtime = value; }
    void clear() { runtime = nullptr; }
    bool initialized() const {
        return runtime && runtime->opened;
    }
};

struct LiteRtSession {
    LiteRtEnvironment env = nullptr;
    bool process_lifetime_environment = false;
#ifdef __APPLE__
    OwnedLiteRtMetalHandles owned_metal_handles;

    static OwnedLiteRtMetalHandles& process_metal_handles() {
        static OwnedLiteRtMetalHandles handles;
        return handles;
    }
#endif
    std::vector<std::unique_ptr<gemma4::Runtime>> gemma4_runtimes;
    std::unique_ptr<RagEngine> embedding_runtime;

    const LiteRtExternalMetalHandles* active_metal_handles() const {
#ifdef __APPLE__
        if (process_lifetime_environment) {
            auto& handles = process_metal_handles();
            return handles.valid() ? &handles.external : nullptr;
        }
        return owned_metal_handles.valid() ? &owned_metal_handles.external : nullptr;
#else
        return nullptr;
#endif
    }

    bool create(Platform platform,
                HardwareTarget environment_hw,
                int environment_accelerators,
                const char* accelerator_runtime_library_dir = nullptr) {
        if (environment_accelerators == kLiteRtHwAcceleratorNone)
            environment_accelerators = hw_accelerator_mask_for_target(environment_hw);
#ifdef __APPLE__
        if (platform == Platform::APPLE_OS) {
            // Diagnostic mode: keep the MTLDevice process-stable but create a
            // fresh MTLCommandQueue and LiteRT environment for each session.
            // This distinguishes stale env/queue state from process-global
            // state inside libLiteRtMetalAccelerator.
            const bool needs_gpu = (environment_accelerators & kLiteRtHwAcceleratorGpu) != 0;
            if (needs_gpu && !owned_metal_handles.valid()) {
                if (!owned_metal_handles.create()) {
                    tinylog::logger().error("LiteRT Apple session Metal handles unavailable",
                        {{"accelerator_mask", (int64_t)environment_accelerators}});
                    return false;
                }
            }
            const LiteRtExternalMetalHandles* metal_handles =
                needs_gpu && owned_metal_handles.valid()
                    ? &owned_metal_handles.external
                    : nullptr;
            tinylog::logger().debug("LiteRT Apple session Metal handles",
                {{"accelerator_mask", (int64_t)environment_accelerators},
                 {"needs_gpu", needs_gpu},
                 {"valid", owned_metal_handles.valid()},
                 {"device", pointer_hex(owned_metal_handles.device)},
                 {"command_queue", pointer_hex(owned_metal_handles.command_queue)}});
            process_lifetime_environment = false;
            const bool ok = create_environment(&env,
                                               environment_hw,
                                               nullptr,
                                               environment_accelerators,
                                               metal_handles,
                                               accelerator_runtime_library_dir);
            if (ok) {
                tinylog::logger().info("LiteRT Apple session environment created",
                    {{"accelerator_mask", std::to_string(environment_accelerators)}});
            }
            return ok;
        }
#else
        (void)platform;
#endif
        return create_environment(&env,
                                  environment_hw,
                                  nullptr,
                                  environment_accelerators,
                                  active_metal_handles(),
                                  accelerator_runtime_library_dir);
    }

    gemma4::Runtime* find_gemma4_runtime(const char* path) const {
        if (!path) return nullptr;
        const std::string expected(path);
        for (const auto& runtime : gemma4_runtimes) {
            if (runtime && runtime->opened && runtime->bundle.path == expected)
                return runtime.get();
        }
        return nullptr;
    }

    gemma4::Runtime* get_or_create_gemma4_runtime(const char* path) {
        if (auto* existing = find_gemma4_runtime(path)) {
            if (!existing->env)
                existing->env = env;
            return existing;
        }
        auto runtime = std::make_unique<gemma4::Runtime>();
        if (!gemma4::open_runtime(runtime.get(), path, env))
            return nullptr;
        auto* raw = runtime.get();
        gemma4_runtimes.push_back(std::move(runtime));
        return raw;
    }

    gemma4::Runtime* preopen_gemma4_runtime(const char* path) {
        if (auto* existing = find_gemma4_runtime(path))
            return existing;
        auto runtime = std::make_unique<gemma4::Runtime>();
        if (!gemma4::open_runtime(runtime.get(), path, nullptr))
            return nullptr;
        auto* raw = runtime.get();
        gemma4_runtimes.push_back(std::move(runtime));
        return raw;
    }

    void release(bool log_release) {
        ScopedLiteRtMetalLogHandles metal_log_scope(active_metal_handles());
#ifdef __APPLE__
        if (process_lifetime_environment && process_metal_handles().valid())
            process_metal_handles().drain_gpu();
        else if (owned_metal_handles.valid())
            owned_metal_handles.drain_gpu();
#endif
        if (embedding_runtime) {
            embedding_runtime->release();
            embedding_runtime.reset();
        }
        for (auto& runtime : gemma4_runtimes) {
            if (runtime)
                gemma4::destroy_runtime(runtime.get());
        }
        gemma4_runtimes.clear();
        const bool retained_process_env = process_lifetime_environment;
        if (env && !process_lifetime_environment) {
            LITERT(LiteRtDestroyEnvironment)(env);
        }
        env = nullptr;
        process_lifetime_environment = false;
#ifdef __APPLE__
        if (owned_metal_handles.valid()) {
            owned_metal_handles.release();
        }
#endif
        if (log_release) {
            if (retained_process_env) {
                tinylog::logger().info("LiteRT session released",
                    {{"environment", std::string("process-lifetime")}});
            } else {
                tinylog::logger().info("LiteRT environment released");
            }
        }
    }
};

// Threading contract:
// TinyLLMEngine is thread-compatible, not internally thread-safe. Owners must
// externally serialize all access to an engine instance, including calls through
// borrowed subsystem pointers such as llm, rag, text, and vlm. The production
// agent C API provides this serialization at the zephr_agent_t boundary.
struct TinyLLMEngine {
    using LifecycleClock = std::chrono::steady_clock;

    LiteRtSession session;
    TextEngine text;
    InferenceEngine* llm = nullptr;
    RagEngine* rag = nullptr;
    VlmEngine vlm;
    bool initialized = false;
    Platform configured_platform = Platform::APPLE_OS;
    TinyLLMExecutionConfig configured_execution;
    int configured_num_threads = 0;
    std::string configured_rag_bundle_path;
    std::string configured_vlm_model_path;
    std::string configured_compilation_cache_dir;
    std::string configured_litert_runtime_library_dir;
    std::vector<ModelLifecycleTiming> model_lifecycle_timings;

    LifecycleClock::time_point mark_model_lifecycle_start() const {
        return LifecycleClock::now();
    }

    void record_model_lifecycle(const std::string& component,
                                const std::string& action,
                                const std::string& detail,
                                LifecycleClock::time_point start,
                                bool ok) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            LifecycleClock::now() - start).count();
        model_lifecycle_timings.push_back(ModelLifecycleTiming{
            component,
            action,
            detail,
            static_cast<int64_t>(elapsed),
            ok,
        });
    }

    const LiteRtExternalMetalHandles* active_metal_handles() const {
        return session.active_metal_handles();
    }

    void set_litert_runtime_library_dir(const char* path) {
        configured_litert_runtime_library_dir = path ? path : "";
    }

    bool init(Platform platform, HardwareTarget hw, int num_threads = 0,
              const char* llm_model_path = nullptr,
              const char* rag_bundle_path = nullptr,
              const char* vlm_model_path = nullptr,
              const char* compilation_cache_dir = nullptr) {
        TinyLLMExecutionConfig config;
        config.llm_plan = hw_target_name(hw);
        config.rag_plan = hw_target_name(hw);
        config.vlm_plan = vlm_model_path && *vlm_model_path ? "gpu" : hw_target_name(hw);
        return init(platform, config, num_threads, llm_model_path,
                    rag_bundle_path, vlm_model_path, compilation_cache_dir);
    }

    bool init(Platform platform, TinyLLMExecutionConfig config, int num_threads = 0,
              const char* llm_model_path = nullptr,
              const char* rag_bundle_path = nullptr,
              const char* vlm_model_path = nullptr,
              const char* compilation_cache_dir = nullptr) {
        release();
        model_lifecycle_timings.clear();

        const bool has_llm = llm_model_path && *llm_model_path;
        const bool has_rag = rag_bundle_path && *rag_bundle_path;
        const bool has_vlm = vlm_model_path && *vlm_model_path;
        if (config.gemma4_runtime.kv_cache_max_len <= 0) {
            config.gemma4_runtime.kv_cache_max_len =
                gemma4::automatic_kv_cache_max_len_for_platform(platform);
        }
        configured_platform = platform;
        configured_execution = config;
        configured_num_threads = num_threads;
        configured_rag_bundle_path = has_rag ? rag_bundle_path : "";
        configured_vlm_model_path = has_vlm ? vlm_model_path : "";
        configured_compilation_cache_dir = compilation_cache_dir ? compilation_cache_dir : "";

        ModelFamily llm_family = ModelFamily::GEMMA4;
        LiteRtMagicNumberConfigs* llm_magic = nullptr;
        if (has_llm) {
            MappedFile probe;
            if (probe.open(llm_model_path)) {
                llm_family = detect_llm_family_from_mapped_model(probe);
                llm_magic = InferenceEngine::prescan_magic_configs(probe);
                gemma4::apply_magic_configs_to_kv_cache(
                    llm_magic, config.gemma4_runtime.kv_cache_max_len);
                probe.close();
            }
        }

        HardwareTarget environment_hw = HardwareTarget::CPU;
        int environment_accelerators = kLiteRtHwAcceleratorNone;
        auto note_environment_target = [&](HardwareTarget target) {
            if (target == HardwareTarget::NPU) {
                environment_hw = HardwareTarget::NPU;
            } else if (target == HardwareTarget::GPU &&
                       environment_hw != HardwareTarget::NPU) {
                environment_hw = HardwareTarget::GPU;
            }
        };

        if (has_llm) {
            environment_accelerators |= llm_environment_accelerator_mask(llm_family,
                                                                        config.llm_plan);
            note_environment_target(parse_hardware_target(config.llm_plan.c_str()));
        }
        if (has_rag) {
            HardwareTarget rag_hw = parse_hardware_target(config.rag_plan.c_str());
            environment_accelerators |= rag_environment_accelerator_mask(rag_hw);
            note_environment_target(rag_hw);
        }
        if (has_vlm) {
            const VlmExecutionTargets vlm_targets =
                vlm_execution_targets(platform, config.vlm_plan);
            environment_accelerators |= gemma4::vision_environment_accelerator_mask(
                vlm_targets.encoder, vlm_targets.decoder);
            note_environment_target(vlm_targets.encoder);
            note_environment_target(vlm_targets.decoder);
        }

        if (!session.create(
                platform,
                environment_hw,
                environment_accelerators,
                configured_litert_runtime_library_dir.empty()
                    ? nullptr
                    : configured_litert_runtime_library_dir.c_str())) {
            if (llm_magic) std::free(llm_magic);
            session.release(false);
            tinylog::logger().error("TinyLLMEngine init failed",
                {{"stage", std::string("session_create")},
                 {"accelerators", (int64_t)environment_accelerators}});
            return false;
        }
        ScopedLiteRtMetalLogHandles metal_log_scope(session.active_metal_handles());

        if (has_llm) {
            auto* runtime = get_or_create_gemma4_runtime(llm_model_path);
            if (!runtime) {
                if (llm_magic) std::free(llm_magic);
                release();
                return false;
            }
            if (llm_magic) {
                runtime->text.magic_configs = llm_magic;
                llm_magic = nullptr;
            }

            const HardwareTarget llm_hw = parse_hardware_target(config.llm_plan.c_str());
            tinylog::logger().info("InferenceEngine: initializing",
                {{"model", std::string("Gemma4")},
                 {"path", runtime->bundle.path}});
            const auto start = mark_model_lifecycle_start();
            const bool ok = gemma4::ensure_text(runtime,
                                                platform,
                                                llm_hw,
                                                num_threads,
                                                compilation_cache_dir,
                                                /*magic_configs=*/nullptr,
                                                config.gemma4_runtime,
                                                config.diagnostic_gemma4,
                                                /*log_initializing=*/true);
            record_model_lifecycle("gemma4.text",
                                   "init",
                                   std::string(hw_target_name(llm_hw)),
                                   start,
                                   ok);
            if (!ok) {
                release();
                return false;
            }
            text.attach(runtime);
            llm = &text;
            std::string components = gemma4::text_components(llm_hw);
            if (runtime->text.has_mtp_drafter) {
                components += std::string("+mtp(") +
                              (runtime->text.mtp_enabled ? hw_target_name(llm_hw)
                                                         : "disabled") +
                              ")";
            }
            tinylog::logger().info("InferenceEngine: initialized",
                {{"model", std::string("Gemma4")},
                 {"path", runtime->bundle.path},
                 {"hw", std::string(hw_target_name(llm_hw))},
                 {"num_threads", (int64_t)resolve_num_threads(num_threads)},
                 {"components", components},
                 {"prefill_lengths", gemma4::prefill_lengths(runtime->text)},
                 {"kv_cache_max_len", (int64_t)runtime->text.kv_cache_max_len}});
        } else if (llm_magic) {
            std::free(llm_magic);
            llm_magic = nullptr;
        }

        if (has_rag) {
            auto embedding = std::make_unique<RagEngine>();
            const auto start = mark_model_lifecycle_start();
            const bool ok = embedding->init(rag_bundle_path, session.env, platform,
                                            parse_hardware_target(config.rag_plan.c_str()),
                                            num_threads, compilation_cache_dir);
            record_model_lifecycle("embedding.gemma3",
                                   "init",
                                   config.rag_plan,
                                   start,
                                   ok);
            if (!ok) {
                tinylog::logger().error("TinyLLMEngine init failed",
                    {{"stage", std::string("embedding")},
                     {"path", std::string(rag_bundle_path)}});
                release();
                return false;
            }
            rag = embedding.get();
            session.embedding_runtime = std::move(embedding);
        }

        if (has_vlm) {
            const VlmExecutionTargets targets =
                vlm_execution_targets(platform, config.vlm_plan);
            const auto start = mark_model_lifecycle_start();
            const bool ok = init_vlm(platform,
                                     targets.encoder,
                                     targets.decoder,
                                     num_threads,
                                     vlm_model_path,
                                     compilation_cache_dir,
                                     nullptr);
            record_model_lifecycle("gemma4.vlm",
                                   "init",
                                   config.vlm_plan,
                                   start,
                                   ok);
            if (!ok) {
                tinylog::logger().error("TinyLLMEngine init failed",
                    {{"stage", std::string("vlm")},
                     {"path", std::string(vlm_model_path)}});
                release();
                return false;
            }
        }

        initialized = true;
        return true;
    }

    gemma4::Runtime* find_gemma4_runtime(const char* path) const {
        return session.find_gemma4_runtime(path);
    }

    gemma4::Runtime* get_or_create_gemma4_runtime(const char* path) {
        return session.get_or_create_gemma4_runtime(path);
    }

    bool init_vlm(Platform platform,
                  HardwareTarget encoder_hw,
                  HardwareTarget decoder_hw,
                  int num_threads,
                  const char* vlm_model_path,
                  const char* compilation_cache_dir = nullptr,
                  LiteRtMagicNumberConfigs* magic_configs = nullptr) {
        if (!vlm_model_path)
            return true;

        LiteRtMagicNumberConfigs* owned_magic = magic_configs;
        auto free_owned_magic = [&]() {
            if (owned_magic) {
                std::free(owned_magic);
                owned_magic = nullptr;
            }
        };

        auto* runtime = get_or_create_gemma4_runtime(vlm_model_path);
        if (!runtime) {
            free_owned_magic();
            return false;
        }
        if (!owned_magic && !runtime->text.magic_configs) {
            // Keep the large Gemma4 bundle genuinely lazy. The VLM path can be
            // configured during agent init, but the 2GB+ mmap and magic-number
            // prescan should happen only after the embedding model has been
            // evicted for vision mode.
            owned_magic = gemma4::prescan_bundle_magic_configs(runtime->bundle);
            gemma4::apply_magic_configs_to_kv_cache(
                owned_magic,
                gemma4::automatic_kv_cache_max_len_for_platform(platform));
        }

        const bool text_reused_from_inference = text.uses(runtime) && runtime->text_initialized;
        const bool has_end_of_vision = gemma4::bundle_has_end_of_vision(runtime->bundle);
        const std::string components = text_reused_from_inference
            ? gemma4::vision_runtime_components_from_inference_engine(
                encoder_hw, has_end_of_vision)
            : gemma4::vision_runtime_components(
                decoder_hw, encoder_hw, has_end_of_vision);
        if (platform == Platform::APPLE_OS &&
            decoder_hw == HardwareTarget::GPU &&
            encoder_hw == HardwareTarget::CPU) {
            tinylog::logger().info("VLM GPU uses CPU vision encoder on Apple",
                {{"requested_plan", std::string("gpu")},
                 {"components", components},
                 {"reason", std::string("Metal vision_encoder is disabled until stable")}});
        }
        tinylog::logger().info("TinyLLMEngine: VLM initializing",
            {{"path", std::string(vlm_model_path)},
             {"env", std::string("shared")},
             {"components", components},
             {"num_threads", (int64_t)resolve_num_threads(num_threads)}});

        if (!text_reused_from_inference) {
            if (owned_magic && !runtime->text.magic_configs) {
                runtime->text.magic_configs = owned_magic;
                owned_magic = nullptr;
            } else {
                free_owned_magic();
            }
            auto ensure_text = [&]() {
                return gemma4::ensure_text(runtime,
                                           platform,
                                           decoder_hw,
                                           num_threads,
                                           compilation_cache_dir,
                                           /*magic_configs=*/nullptr,
                                           Gemma4RuntimeConfig(),
                                           Gemma4DiagnosticConfig(),
                                           /*log_initializing=*/false);
            };
            auto ensure_vision = [&]() {
                return gemma4::ensure_vision(runtime,
                                             platform,
                                             encoder_hw,
                                             num_threads,
                                             compilation_cache_dir,
                                             /*log_initializing=*/false);
            };
            const bool init_ok = ensure_text() && ensure_vision();
            if (!init_ok) {
                free_owned_magic();
                return false;
            }
        } else {
            free_owned_magic();
            if (!gemma4::ensure_vision(runtime,
                                       platform,
                                       encoder_hw,
                                       num_threads,
                                       compilation_cache_dir,
                                       /*log_initializing=*/false)) {
                return false;
            }
        }
        vlm.attach(runtime);
        tinylog::logger().info("TinyLLMEngine: VLM initialized",
            {{"path", std::string(vlm_model_path)},
             {"env", std::string("shared")},
             {"components", components},
             {"num_threads", (int64_t)resolve_num_threads(num_threads)}});
        return true;
    }

    bool ensure_embedding_model_for_text() {
        if (configured_rag_bundle_path.empty())
            return true;
        if (!session.embedding_runtime) {
            auto embedding = std::make_unique<RagEngine>();
            if (!embedding->init(configured_rag_bundle_path.c_str(),
                                 session.env,
                                 configured_platform,
                                 parse_hardware_target(configured_execution.rag_plan.c_str()),
                                 configured_num_threads,
                                 configured_compilation_cache_dir.empty()
                                     ? nullptr
                                     : configured_compilation_cache_dir.c_str())) {
                return false;
            }
            rag = embedding.get();
            session.embedding_runtime = std::move(embedding);
            return true;
        }
        rag = session.embedding_runtime.get();
        if (!session.embedding_runtime->model_ready) {
            const auto start = mark_model_lifecycle_start();
            const bool ok = session.embedding_runtime->ensure_model();
            record_model_lifecycle("embedding.gemma3",
                                   "init",
                                   "ensure text context",
                                   start,
                                   ok);
            return ok;
        }
        return true;
    }

    void release_vlm_model() {
        ScopedLiteRtMetalLogHandles metal_log_scope(active_metal_handles());
        auto* runtime = vlm.runtime;
        if (!runtime)
            return;
        vlm.clear();
        const auto start = mark_model_lifecycle_start();
#ifdef __APPLE__
        drain_active_litert_metal_queue();
#endif
        if (text.uses(runtime)) {
            gemma4::destroy_vision(&runtime->vision);
            runtime->vision_initialized = false;
            record_model_lifecycle("gemma4.vlm",
                                   "deinit",
                                   "vision components only",
                                   start,
                                   true);
            tinylog::logger().info("TinyLLMEngine: VLM released",
                {{"path", runtime->bundle.path},
                 {"mode", std::string("vision components only")}});
            return;
        }
        for (auto it = session.gemma4_runtimes.begin(); it != session.gemma4_runtimes.end(); ++it) {
            if (it->get() == runtime) {
                tinylog::logger().info("TinyLLMEngine: VLM released",
                    {{"path", runtime->bundle.path},
                     {"mode", std::string("dedicated runtime")}});
                gemma4::destroy_runtime(runtime);
                session.gemma4_runtimes.erase(it);
                record_model_lifecycle("gemma4.vlm",
                                       "deinit",
                                       "dedicated runtime",
                                       start,
                                       true);
                return;
            }
        }
        gemma4::destroy_runtime(runtime);
        record_model_lifecycle("gemma4.vlm",
                               "deinit",
                               "untracked runtime",
                               start,
                               true);
    }

    bool ensure_vlm_model() {
        ScopedLiteRtMetalLogHandles metal_log_scope(active_metal_handles());
        if (configured_vlm_model_path.empty())
            return false;
        if (vlm.initialized())
            return true;
        const VlmExecutionTargets targets =
            vlm_execution_targets(configured_platform, configured_execution.vlm_plan);
        const auto start = mark_model_lifecycle_start();
        const bool ok = init_vlm(configured_platform,
                                 targets.encoder,
                                 targets.decoder,
                                 configured_num_threads,
                                 configured_vlm_model_path.c_str(),
                                 configured_compilation_cache_dir.empty()
                                     ? nullptr
                                     : configured_compilation_cache_dir.c_str(),
                                 nullptr);
        record_model_lifecycle("gemma4.vlm",
                               "init",
                               configured_execution.vlm_plan,
                               start,
                               ok);
        if (!ok) {
            tinylog::logger().error("TinyLLMEngine: failed to initialize VLM",
                {{"path", configured_vlm_model_path}});
            return false;
        }
        return true;
    }

    bool describe_image_rgb888(
            const uint8_t* rgb,
            int width,
            int height,
            int row_stride,
            const std::string& prompt,
            int max_tokens,
            gemma4::VisionTextSmokeResult& result) {
        ScopedLiteRtMetalLogHandles metal_log_scope(active_metal_handles());
        if (!vlm.initialized())
            return false;
        return gemma4::describe_rgb888_with_runtime(
            vlm.runtime,
            rgb,
            width,
            height,
            row_stride,
            prompt,
            max_tokens,
            result);
    }

    void release() {
        release_vlm_model();
        if (session.embedding_runtime && session.embedding_runtime->model_ready) {
            const auto start = mark_model_lifecycle_start();
            session.embedding_runtime->release_model("agent release");
            record_model_lifecycle("embedding.gemma3",
                                   "deinit",
                                   "agent release",
                                   start,
                                   true);
        }
        for (auto& runtime : session.gemma4_runtimes) {
            if (!runtime)
                continue;
            const auto start = mark_model_lifecycle_start();
            gemma4::destroy_runtime(runtime.get());
            record_model_lifecycle("gemma4.runtime",
                                   "deinit",
                                   "agent release",
                                   start,
                                   true);
        }
        session.gemma4_runtimes.clear();
        vlm.clear();
        llm = nullptr;
        text.clear();
        rag = nullptr;
        session.release(initialized);
        initialized = false;
        configured_rag_bundle_path.clear();
        configured_vlm_model_path.clear();
        configured_compilation_cache_dir.clear();
    }
};
