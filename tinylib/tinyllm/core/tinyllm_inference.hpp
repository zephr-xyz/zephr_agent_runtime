// TinyLLM inference — prefill/decode/constrained-decode for LLM models.
//
// Dispatches across supported text model families.
// Owned by TinyLLMEngine; does not manage its own LiteRtEnvironment.

#pragma once

#include "tinyllm_model_gemma4.hpp"

#include <variant>

// MARK: - Bundle utilities (no engine init required)

static inline std::string detect_model_family(const char* path) {
    MappedFile mf;
    if (!mf.open(path)) return "";

    if (mf.size < 8 || memcmp(mf.data, "LITERTLM", 8) != 0) {
        mf.close();
        return "";
    }

    auto sections = parse_litertlm_bundle(mf.data, mf.size);
    mf.close();

    if (find_bundle_section(sections, BundleModelType::EMBEDDER))
        return "GEMMA4";
    return "";
}

static inline std::vector<std::pair<std::string, size_t>>
list_bundle_sections(const char* path) {
    std::vector<std::pair<std::string, size_t>> result;
    MappedFile mf;
    if (!mf.open(path)) return result;

    if (mf.size >= 8 && memcmp(mf.data, "LITERTLM", 8) == 0) {
        auto sections = parse_litertlm_bundle(mf.data, mf.size);
        for (auto& sec : sections)
            result.push_back({bundle_model_type_name(sec.type), sec.size});
    }
    mf.close();
    return result;
}

// MARK: - Model-family prompt/completion formatting

static inline std::string format_tool_call_prompt(
        ModelFamily family,
        const ToolDef& tool,
        const std::string& user_message,
        const std::string& system_message = "",
        const std::string& prompt_profile = "") {
    switch (family) {
        case ModelFamily::GEMMA4:
            return gemma4::format_fc_prompt(tool, user_message, system_message);
        case ModelFamily::GEMMA3_EMBEDDING:
        case ModelFamily::GEMMA4_VISION_ENCODER:
            break;
    }
    return "";
}

static inline std::string format_tool_call_completion(
        ModelFamily family,
        const ToolDef& tool,
        const std::map<std::string, std::string>& arguments,
        bool include_function_response = true) {
    switch (family) {
        case ModelFamily::GEMMA4:
            return gemma4::format_gemma4_completion(
                tool, arguments, include_function_response);
        case ModelFamily::GEMMA3_EMBEDDING:
        case ModelFamily::GEMMA4_VISION_ENCODER:
            break;
    }
    throw std::runtime_error("Unsupported model family for tool-call completion formatting");
}

// MARK: - Text role facade

struct TextEngine {
    using Backend = std::variant<std::monostate, gemma4::Runtime*>;

    Backend backend;
    ModelFamily family = ModelFamily::GEMMA4;
    bool initialized = false;

    // Pre-scan the text model for magic number configs that must be scoped
    // around decoder compilation.
    static LiteRtMagicNumberConfigs* prescan_magic_configs(const MappedFile& mf) {
        const void* model_data = mf.data;
        size_t model_size = mf.size;
        const char* source = "raw";
        if (mf.size >= 8 && memcmp(mf.data, "LITERTLM", 8) == 0) {
            auto sections = parse_litertlm_bundle(mf.data, mf.size);
            auto* decoder_section = find_decoder_section(sections);
            if (!decoder_section) {
                fprintf(stderr, "LiteRT text magic prescan: no decoder section\n");
                fflush(stderr);
                return nullptr;
            }
            model_data = decoder_section->data;
            model_size = decoder_section->size;
            source = "decoder_section";
        }

        LiteRtModel probe = nullptr;
        LiteRtStatus status = LITERT(LiteRtCreateModelFromBuffer)(
                nullptr, model_data, model_size, &probe);
        if (status != kLiteRtStatusOk) {
            fprintf(stderr,
                    "LiteRT text magic prescan: create probe failed source=%s status=%d bytes=%zu\n",
                    source,
                    static_cast<int>(status),
                    model_size);
            fflush(stderr);
            return nullptr;
        }
        auto* configs = build_magic_number_configs(probe);
        fprintf(stderr, "LiteRT text magic prescan: source=%s configs=%s\n",
                source,
                magic_configs_summary(configs).c_str());
        fflush(stderr);
        LITERT(LiteRtDestroyModel)(probe);
        return configs;
    }

    void clear() {
        backend = std::monostate{};
        family = ModelFamily::GEMMA4;
        initialized = false;
    }

    void attach(gemma4::Runtime* runtime) {
        backend = runtime;
        family = ModelFamily::GEMMA4;
        initialized = runtime && runtime->text_initialized;
    }

    gemma4::Runtime* gemma4_runtime() const {
        auto* slot = std::get_if<gemma4::Runtime*>(&backend);
        return slot ? *slot : nullptr;
    }

    bool uses(const gemma4::Runtime* runtime) const {
        return gemma4_runtime() == runtime;
    }

    const std::string& model_path() const {
        if (auto* runtime = gemma4_runtime())
            return runtime->bundle.path;
        static const std::string empty;
        return empty;
    }

    Tokenizer* tokenizer() {
        if (auto* runtime = gemma4_runtime())
            return &runtime->tokenizer;
        return nullptr;
    }

    void reset() {
        if (auto* runtime = gemma4_runtime())
            gemma4::reset_kv_cache(&runtime->text);
    }

    // --- inference ---

    bool prefill(const int32_t* tokens, int count) {
        if (auto* runtime = gemma4_runtime()) {
            if (runtime->text.diagnostic_config.prefill_by_decode) {
                for (int pos = 0; pos < count; pos++) {
                    if (!gemma4::run_decode_step_kv_only(&runtime->text, tokens[pos]))
                        return false;
                }
                return true;
            }
            const int max_prefill_chunk =
                runtime->text.diagnostic_config.prefill_max_chunk > 0
                    ? std::clamp(runtime->text.diagnostic_config.prefill_max_chunk,
                                 1,
                                 gemma4::KV_CACHE_MAX_LEN_MAGIC_CAP)
                    : gemma4::KV_CACHE_MAX_LEN_MAGIC_CAP;
            int pos = 0;
            while (pos < count) {
                const int requested = std::min(count - pos, max_prefill_chunk);
                auto* ps = gemma4::pick_prefill_sig(&runtime->text, requested);
                if (!ps) return false;
                int chunk = std::min(ps->seq_len, requested);
                if (!gemma4::run_prefill_chunk(&runtime->text, tokens + pos, chunk, pos))
                    return false;
                pos += chunk;
            }
            return true;
        }
        return false;
    }

    bool decode_logits(int32_t token_id, float* logits_out) {
        if (auto* runtime = gemma4_runtime())
            return gemma4::run_decode_step(&runtime->text, token_id, logits_out);
        return false;
    }

    int activation_dim() const {
        if (auto* runtime = gemma4_runtime())
            return runtime->text.shape.embedding_dim;
        return 0;
    }

    bool decode_logits_activations(
            int32_t token_id, float* logits_out, float* activations_out) {
        if (auto* runtime = gemma4_runtime())
            return gemma4::run_decode_step_with_activations(
                &runtime->text, token_id, logits_out, activations_out);
        return false;
    }

    int decode_sample(int32_t token_id, float temperature, int top_k, float top_p) {
        std::vector<float> logits(VOCAB_SIZE);
        if (!decode_logits(token_id, logits.data())) return -1;
        return sample_topk_topp(logits.data(), VOCAB_SIZE, temperature, top_k, top_p, rng());
    }

    ConstrainedResult constrained_decode(
        const Tokenizer& tok, int pending_token,
        const std::string& tool_name,
        const std::vector<ConstrainedParam>& params,
        float temperature, int top_k, float top_p, int diag_top_k)
    {
        if (auto* runtime = gemma4_runtime()) {
            Gemma4Grammar grammar(tok, tool_name, params, gemma4::STOP_TOKENS);
            return gemma4::generate_constrained_with_verify_batches<Gemma4Grammar>(
                &runtime->text, grammar, pending_token,
                temperature, top_k, top_p, diag_top_k);
        }
        return {};
    }

    // Format prompt + encode + prefill + constrained decode in one call.
    GenerateResult generate_tool_call(
        const ToolDef& tool,
        const std::string& user_message,
        float temperature = 0.0f, int top_k = 40, float top_p = 1.0f,
        const std::string& system_message = "",
        int diag_top_k = 8,
        const std::string& prompt_profile = "")
    {
        std::string prompt = format_tool_call_prompt(
            family, tool, user_message, system_message, prompt_profile);

        std::vector<ConstrainedParam> cparams;
        cparams.reserve(tool.params.size());
        for (auto& p : tool.params)
            cparams.push_back({p.name, p.enum_values});

        GenerateResult result;
        if (auto* runtime = gemma4_runtime()) {
            result = gemma4::generate(&runtime->text, prompt, tool.name, cparams,
                                      /*max_tokens=*/256, temperature, top_k, top_p,
                                      diag_top_k);
        }
        result.prompt = std::move(prompt);
        return result;
    }

    GenerateResult generate_text(
        const std::string& user_message,
        const std::string& system_message = "",
        int max_tokens = 256,
        float temperature = 0.7f,
        int top_k = 40,
        float top_p = 0.95f)
    {
        GenerateResult result;
        auto* runtime = gemma4_runtime();
        if (!runtime)
            return result;

        std::string prompt = gemma4::format_chat_prompt(user_message, system_message);
        result = gemma4::generate(
            &runtime->text,
            prompt,
            /*tool_name=*/"",
            /*params=*/{},
            max_tokens,
            temperature,
            top_k,
            top_p);
        result.prompt = std::move(prompt);
        return result;
    }

    GenerateResult generate_tool_aware_text(
        const std::vector<ToolDef>& tools,
        const std::string& user_message,
        const std::string& system_message = "",
        int max_tokens = 256,
        float temperature = 0.0f,
        int top_k = 40,
        float top_p = 0.95f,
        int reserve_output_tokens = 0,
        std::function<bool(const std::string&)> stream_callback = {})
    {
        GenerateResult result;
        auto* runtime = gemma4_runtime();
        if (!runtime)
            return result;

        std::string prompt = tools.empty()
            ? gemma4::format_chat_prompt(user_message, system_message)
            : gemma4::format_fc_prompt(tools, user_message, system_message);
        result = gemma4::generate_tool_aware(
            &runtime->text,
            prompt,
            tools,
            max_tokens,
            temperature,
            top_k,
            top_p,
            reserve_output_tokens,
            /*diag_top_k=*/8,
            std::move(stream_callback));
        result.prompt = std::move(prompt);
        return result;
    }

    GenerateResult generate_tool_aware_text_from_prompt(
        const std::vector<ToolDef>& tools,
        const std::string& prompt,
        int max_tokens = 256,
        float temperature = 0.0f,
        int top_k = 40,
        float top_p = 0.95f,
        int reserve_output_tokens = 0,
        std::function<bool(const std::string&)> stream_callback = {})
    {
        GenerateResult result;
        auto* runtime = gemma4_runtime();
        if (!runtime)
            return result;

        result = gemma4::generate_tool_aware(
            &runtime->text,
            prompt,
            tools,
            max_tokens,
            temperature,
            top_k,
            top_p,
            reserve_output_tokens,
            /*diag_top_k=*/8,
            std::move(stream_callback));
        result.prompt = prompt;
        return result;
    }

    GenerateResult generate_text_from_prompt(
        const std::string& prompt,
        int max_tokens = 256,
        float temperature = 0.0f,
        int top_k = 40,
        float top_p = 0.95f,
        std::function<bool(const std::string&)> stream_callback = {})
    {
        GenerateResult result;
        auto* runtime = gemma4_runtime();
        if (!runtime)
            return result;

        result = gemma4::generate_tool_aware(
            &runtime->text,
            prompt,
            /*tools=*/{},
            max_tokens,
            temperature,
            top_k,
            top_p,
            /*reserve_output_tokens=*/0,
            /*diag_top_k=*/8,
            std::move(stream_callback));
        result.prompt = prompt;
        return result;
    }

    GenerateResult continue_after_tool_response(
        const std::string& tool_response,
        int max_tokens = 256,
        float temperature = 0.0f,
        int top_k = 40,
        float top_p = 0.95f,
        int reserve_output_tokens = 0,
        std::function<bool(const std::string&)> stream_callback = {})
    {
        GenerateResult result;
        auto* runtime = gemma4_runtime();
        if (!runtime)
            return result;

        result = gemma4::continue_after_tool_response(
            &runtime->text,
            tool_response,
            max_tokens,
            temperature,
            top_k,
            top_p,
            reserve_output_tokens,
            std::move(stream_callback));
        return result;
    }

    GenerateResult continue_tool_aware_text(
        const std::vector<ToolDef>& tools,
        const std::string& prompt_suffix,
        int max_tokens = 256,
        float temperature = 0.0f,
        int top_k = 40,
        float top_p = 0.95f,
        int reserve_output_tokens = 0,
        std::function<bool(const std::string&)> stream_callback = {})
    {
        GenerateResult result;
        auto* runtime = gemma4_runtime();
        if (!runtime)
            return result;

        result = gemma4::continue_tool_aware(
            &runtime->text,
            prompt_suffix,
            tools,
            max_tokens,
            temperature,
            top_k,
            top_p,
            reserve_output_tokens,
            /*diag_top_k=*/8,
            std::move(stream_callback));
        return result;
    }

    gemma4::MtpDebugResult gemma4_mtp_debug_verify(
        const std::string& user_message,
        const std::string& system_message = "")
    {
        auto* runtime = gemma4_runtime();
        if (!runtime) {
            gemma4::MtpDebugResult result;
            result.error = "InferenceEngine is not initialized with Gemma4";
            return result;
        }
        std::string prompt = gemma4::format_chat_prompt(user_message, system_message);
        return gemma4::debug_mtp_verify(&runtime->text, prompt);
    }

    bool gemma4_mtp_available() const {
        if (auto* runtime = gemma4_runtime())
            return runtime->text.has_mtp_drafter && runtime->text.has_verify_sig;
        return false;
    }

    bool gemma4_mtp_enabled() const {
        if (auto* runtime = gemma4_runtime())
            return runtime->text.mtp_enabled && runtime->text.mtp_runtime_enabled;
        return false;
    }

    int prefill_seq_len() const {
        if (auto* runtime = gemma4_runtime()) {
            for (int i = gemma4::NUM_PREFILL_SIZES - 1; i >= 0; i--)
                if (runtime->text.prefill_sigs[i].seq_len > 0)
                    return runtime->text.prefill_sigs[i].seq_len;
            return 0;
        }
        return 0;
    }

    int current_pos() const {
        if (auto* runtime = gemma4_runtime())
            return runtime->text.current_pos;
        return 0;
    }

    std::mt19937& rng() {
        if (auto* runtime = gemma4_runtime())
            return runtime->text.rng;
        static thread_local std::mt19937 fallback{1234};
        return fallback;
    }
};

using InferenceEngine = TextEngine;
