// Zephr Agent nanobind module — Python binding over the TinyLLM runtime.
//
// Links directly against libLiteRt.dylib from the ai-edge-litert pip package.
// Supports Gemma4 text/vision and EmbeddingGemma developer bindings.

#include "zephr_agent_runtime_nanobind_common.hpp"

#include "core/tinyllm_bundle_inspect.hpp" // IWYU pragma: keep
#include "core/tinyllm_model_gemma4.hpp" // IWYU pragma: keep

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

static tinylog::Logger& tlog() {
    static tinylog::Logger* instance = [] {
        auto* l = new tinylog::Logger;
        tinylog::add_platform_default_sink(
            *l, "xyz.zephr.sdks.agent", "zephr_agent_runtime_nanobind", tinylog::level_from_env());
        return l;
    }();
    return *instance;
}

static struct _TllmLogInit {
    _TllmLogInit() { tinylog::set_default_logger(&tlog()); }
} _tllm_log_init;

namespace nb = nanobind;

static ToolDef tool_from_args(
        const std::string& tool_name,
        const std::string& tool_description,
        const std::vector<std::tuple<std::string, std::string, std::string,
                                     std::vector<std::string>, bool>>& params) {
    ToolDef tool;
    tool.name = tool_name;
    tool.description = tool_description;
    for (auto& [name, desc, type, enum_values, required] : params)
        tool.params.push_back({name, desc, type, enum_values, required});
    return tool;
}

static std::vector<ToolDef> tools_from_python_specs(
        const std::vector<std::tuple<
            std::string,
            std::string,
            std::vector<std::tuple<std::string, std::string, std::string,
                                   std::vector<std::string>, bool>>>>& specs) {
    std::vector<ToolDef> tools;
    tools.reserve(specs.size());
    for (const auto& [name, description, params] : specs)
        tools.push_back(tool_from_args(name, description, params));
    return tools;
}

static ModelFamily model_family_from_name(const std::string& name) {
    std::string s = name;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    if (s == "gemma4" || s == "gemma_4" || s == "g4")
        return ModelFamily::GEMMA4;
    if (s == "gemma3_embedding" || s == "gemma3" || s == "embedding")
        return ModelFamily::GEMMA3_EMBEDDING;
    throw std::runtime_error("Unknown model family: " + name);
}

static nb::dict generate_result_to_stats_dict(const GenerateResult& result) {
    nb::dict d;
    d["response"] = result.response;
    d["prompt"] = result.prompt;
    d["input_ids_count"] = result.input_ids_count;
    d["prefill_tokens"] = result.prefill_tokens;
    d["decode_steps"] = result.decode_steps;
    d["tokenize_ms"] = result.tokenize_ms;
    d["prefill_ms"] = result.prefill_ms;
    d["decode_ms"] = result.decode_ms;
    d["first_decode_ms"] = result.first_decode_ms;
    nb::dict params;
    for (auto& [key, value] : result.params)
        params[nb::str(key.c_str())] = nb::str(value.c_str());
    d["params"] = params;
    d["constrained_target_decode_calls"] = result.constrained_target_decode_calls;
    d["constrained_verify_batches"] = result.constrained_verify_batches;
    d["constrained_verify_fixed_tokens"] = result.constrained_verify_fixed_tokens;
    d["constrained_verify_rows"] = result.constrained_verify_rows;
    d["constrained_target_decode_ms"] = result.constrained_target_decode_ms;
    d["constrained_verify_ms"] = result.constrained_verify_ms;
    d["mtp_cycles"] = result.mtp_cycles;
    d["mtp_trust_verify_kv"] = result.mtp_trust_verify_kv;
    d["mtp_adaptive_enabled"] = result.mtp_adaptive_enabled;
    d["mtp_adaptive_disabled"] = result.mtp_adaptive_disabled;
    d["mtp_adaptive_disable_cycle"] = result.mtp_adaptive_disable_cycle;
    d["mtp_adaptive_disable_output_tokens"] =
        result.mtp_adaptive_disable_output_tokens;
    d["mtp_max_draft_tokens"] = result.mtp_max_draft_tokens;
    d["mtp_target_decode_calls"] = result.mtp_target_decode_calls;
    d["mtp_drafter_calls"] = result.mtp_drafter_calls;
    d["mtp_verify_calls"] = result.mtp_verify_calls;
    d["mtp_draft_tokens"] = result.mtp_draft_tokens;
    d["mtp_accepted_tokens"] = result.mtp_accepted_tokens;
    d["mtp_rejected_cycles"] = result.mtp_rejected_cycles;
    d["mtp_rejected_after_prefix_0"] = result.mtp_rejected_after_prefix_0;
    d["mtp_rejected_after_prefix_1"] = result.mtp_rejected_after_prefix_1;
    d["mtp_rejected_after_prefix_2"] = result.mtp_rejected_after_prefix_2;
    d["mtp_full_accept_cycles"] = result.mtp_full_accept_cycles;
    d["mtp_shadow_verify_cycles"] = result.mtp_shadow_verify_cycles;
    d["mtp_replacement_tokens"] = result.mtp_replacement_tokens;
    d["mtp_bonus_tokens"] = result.mtp_bonus_tokens;
    d["mtp_fallback_cycles"] = result.mtp_fallback_cycles;
    d["mtp_rebuilds"] = result.mtp_rebuilds;
    d["mtp_local_repairs"] = result.mtp_local_repairs;
    d["mtp_local_repair_tokens"] = result.mtp_local_repair_tokens;
    d["mtp_target_decode_ms"] = result.mtp_target_decode_ms;
    d["mtp_drafter_ms"] = result.mtp_drafter_ms;
    d["mtp_verify_ms"] = result.mtp_verify_ms;
    d["mtp_rejection_ms"] = result.mtp_rejection_ms;
    d["mtp_rebuild_ms"] = result.mtp_rebuild_ms;
    d["mtp_local_repair_ms"] = result.mtp_local_repair_ms;
    d["mtp_target_model_us"] = result.mtp_target_model_us;
    d["mtp_target_logits_read_us"] = result.mtp_target_logits_read_us;
    d["mtp_target_activation_read_us"] = result.mtp_target_activation_read_us;
    d["mtp_target_sample_us"] = result.mtp_target_sample_us;
    d["mtp_drafter_model_us"] = result.mtp_drafter_model_us;
    d["mtp_drafter_logits_read_us"] = result.mtp_drafter_logits_read_us;
    d["mtp_drafter_activation_read_us"] = result.mtp_drafter_activation_read_us;
    d["mtp_drafter_sample_us"] = result.mtp_drafter_sample_us;
    d["mtp_verify_model_us"] = result.mtp_verify_model_us;
    d["mtp_verify_logits_read_us"] = result.mtp_verify_logits_read_us;
    d["mtp_verify_activation_read_us"] = result.mtp_verify_activation_read_us;
    d["mtp_verify_sample_us"] = result.mtp_verify_sample_us;
    d["mtp_model_us"] =
        result.mtp_target_model_us + result.mtp_drafter_model_us +
        result.mtp_verify_model_us;
    d["mtp_logits_read_us"] =
        result.mtp_target_logits_read_us + result.mtp_drafter_logits_read_us +
        result.mtp_verify_logits_read_us;
    d["mtp_activation_read_us"] =
        result.mtp_target_activation_read_us +
        result.mtp_drafter_activation_read_us +
        result.mtp_verify_activation_read_us;
    d["mtp_sample_us"] =
        result.mtp_target_sample_us + result.mtp_drafter_sample_us +
        result.mtp_verify_sample_us;
    d["mtp_target_logits_rows_read"] = result.mtp_target_logits_rows_read;
    d["mtp_drafter_logits_rows_read"] = result.mtp_drafter_logits_rows_read;
    d["mtp_verify_logits_rows_read"] = result.mtp_verify_logits_rows_read;
    d["mtp_logits_rows_read"] =
        result.mtp_target_logits_rows_read +
        result.mtp_drafter_logits_rows_read +
        result.mtp_verify_logits_rows_read;
    d["mtp_target_activation_rows_read"] = result.mtp_target_activation_rows_read;
    d["mtp_drafter_activation_rows_read"] = result.mtp_drafter_activation_rows_read;
    d["mtp_verify_activation_rows_read"] = result.mtp_verify_activation_rows_read;
    d["mtp_activation_rows_read"] =
        result.mtp_target_activation_rows_read +
        result.mtp_drafter_activation_rows_read +
        result.mtp_verify_activation_rows_read;
    d["mtp_acceptance_rate"] = result.mtp_draft_tokens > 0
        ? (double)result.mtp_accepted_tokens / (double)result.mtp_draft_tokens
        : 0.0;
    d["mtp_rejected_cycle_rate"] = result.mtp_cycles > 0
        ? (double)result.mtp_rejected_cycles / (double)result.mtp_cycles
        : 0.0;
    d["mtp_avg_drafter_ms"] = result.mtp_drafter_calls > 0
        ? (double)result.mtp_drafter_ms / (double)result.mtp_drafter_calls
        : 0.0;
    d["mtp_avg_verify_ms"] = result.mtp_verify_calls > 0
        ? (double)result.mtp_verify_ms / (double)result.mtp_verify_calls
        : 0.0;
    d["mtp_saved_target_decode_calls"] =
        result.mtp_accepted_tokens + result.mtp_replacement_tokens +
        result.mtp_bonus_tokens;
    d["mtp_net_saved_target_decode_calls"] =
        (result.mtp_accepted_tokens + result.mtp_replacement_tokens +
         result.mtp_bonus_tokens) - result.mtp_local_repair_tokens;
    return d;
}

static nb::list int_vector_to_list(const std::vector<int>& values) {
    nb::list out;
    for (int value : values)
        out.append(value);
    return out;
}

static nb::dict mtp_debug_result_to_dict(const gemma4::MtpDebugResult& result) {
    nb::dict d;
    d["ok"] = result.ok;
    d["error"] = result.error;
    d["input_ids_count"] = result.input_ids_count;
    d["prefill_tokens"] = result.prefill_tokens;
    d["pending_token"] = result.pending_token;
    d["verify_start_pos"] = result.verify_start_pos;
    d["good_token"] = result.good_token;
    d["sequential_tokens"] = int_vector_to_list(result.sequential_tokens);
    d["draft_tokens"] = int_vector_to_list(result.draft_tokens);
    d["verifier_tokens"] = int_vector_to_list(result.verifier_tokens);
    d["oracle_draft_tokens"] = int_vector_to_list(result.oracle_draft_tokens);
    d["oracle_verifier_tokens"] = int_vector_to_list(result.oracle_verifier_tokens);
    d["oracle_prevpos_verifier_tokens"] =
        int_vector_to_list(result.oracle_prevpos_verifier_tokens);
    d["oracle_anchored_verifier_tokens"] =
        int_vector_to_list(result.oracle_anchored_verifier_tokens);
    d["oracle_verify_kv_next_token"] = result.oracle_verify_kv_next_token;
    d["accepted_prefix"] = result.accepted_prefix;
    d["oracle_accepted_prefix"] = result.oracle_accepted_prefix;
    d["oracle_prevpos_accepted_prefix"] = result.oracle_prevpos_accepted_prefix;
    d["oracle_anchored_accepted_prefix"] = result.oracle_anchored_accepted_prefix;
    d["good_matches_sequential"] =
        !result.sequential_tokens.empty() &&
        result.good_token == result.sequential_tokens[0];
    d["verifier_row0_matches_sequential_after_good"] =
        result.sequential_tokens.size() > 1 &&
        !result.verifier_tokens.empty() &&
        result.verifier_tokens[0] == result.sequential_tokens[1];
    d["verifier_shifted_row_matches_sequential_after_good"] =
        result.sequential_tokens.size() > 1 &&
        result.verifier_tokens.size() > gemma4::MTP_VERIFY_LOGIT_OFFSET &&
        result.verifier_tokens[gemma4::MTP_VERIFY_LOGIT_OFFSET] ==
            result.sequential_tokens[1];
    return d;
}

static nb::dict tensor_metadata_to_python(const TinyLLMTensorMetadata& tensor) {
    nb::dict out;
    out["name"] = nb::str(tensor.name.c_str());
    out["element_type"] = nb::str(tensor.element_type.c_str());
    out["shape"] = nb::cast(tensor.shape);
    return out;
}

static nb::list inspect_litertlm_bundle(const std::string& path) {
    nb::list sections;
    for (const auto& section : inspect_litertlm_bundle_metadata(path.c_str())) {
        nb::dict section_dict;
        section_dict["type"] = nb::str(section.type.c_str());
        section_dict["data_type"] = nb::str(section.data_type.c_str());
        section_dict["begin_offset"] = section.begin_offset;
        section_dict["end_offset"] = section.end_offset;
        section_dict["byte_size"] = section.byte_size;
        section_dict["model_loaded"] = section.model_loaded;

        nb::list signatures;
        for (const auto& sig : section.signatures) {
            nb::dict sig_dict;
            sig_dict["index"] = sig.index;
            sig_dict["key"] = nb::str(sig.key.c_str());

            nb::list inputs;
            for (const auto& input : sig.inputs)
                inputs.append(tensor_metadata_to_python(input));
            sig_dict["inputs"] = inputs;

            nb::list outputs;
            for (const auto& output : sig.outputs)
                outputs.append(tensor_metadata_to_python(output));
            sig_dict["outputs"] = outputs;

            signatures.append(sig_dict);
        }
        section_dict["signatures"] = signatures;
        sections.append(section_dict);
    }
    return sections;
}

static nb::dict run_gemma4_vision_smoke(const std::string& path, int num_patches) {
    auto smoke = gemma4::run_vision_smoke(path.c_str(), num_patches);
    nb::dict result;
    result["input_patches"] = smoke.input_patches;
    result["valid_vision_tokens"] = smoke.valid_vision_tokens;
    result["embedding_dim"] = smoke.embedding_dim;
    result["first_embedding_value"] = smoke.first_embedding_value;
    result["last_valid_embedding_value"] = smoke.last_valid_embedding_value;
    return result;
}

static nb::dict vision_text_result_to_python(const gemma4::VisionTextSmokeResult& smoke) {
    nb::dict result;
    result["input_patches"] = smoke.input_patches;
    result["valid_vision_tokens"] = smoke.valid_vision_tokens;
    result["image_token_slots"] = smoke.image_token_slots;
    result["resized_width"] = smoke.resized_width;
    result["resized_height"] = smoke.resized_height;
    result["prompt_tokens"] = smoke.prompt_tokens;
    result["prefill_tokens"] = smoke.prefill_tokens;
    result["decode_steps"] = smoke.decode_steps;
    result["first_decode_ms"] = smoke.first_decode_ms;
    result["response"] = nb::str(smoke.response.c_str());
    return result;
}

static nb::dict run_gemma4_vlm_text_smoke(
        const std::string& path,
        const std::string& prompt,
        int num_patches,
        int max_tokens) {
    auto smoke = gemma4::run_vision_text_smoke(
        path.c_str(), prompt, num_patches, max_tokens);
    return vision_text_result_to_python(smoke);
}

static nb::dict run_gemma4_vlm_rgb888(
        const std::string& path,
        nb::bytes rgb,
        int width,
        int height,
        int row_stride,
        const std::string& prompt,
        int max_tokens,
        const std::string& encoder_delegate = "cpu",
        const std::string& decoder_delegate = "cpu") {
    const int expected_min = height * row_stride;
    if (width <= 0 || height <= 0 || row_stride < width * 3 ||
        (int)rgb.size() < expected_min)
        throw std::runtime_error("invalid RGB888 image buffer");
    HardwareTarget encoder_hw = HardwareTarget::CPU;
    if (encoder_delegate == "gpu") {
        encoder_hw = HardwareTarget::GPU;
    } else if (encoder_delegate != "cpu") {
        throw std::runtime_error("encoder_delegate must be 'cpu' or 'gpu'");
    }
    HardwareTarget decoder_hw = HardwareTarget::CPU;
    if (decoder_delegate == "gpu") {
        decoder_hw = HardwareTarget::GPU;
    } else if (decoder_delegate != "cpu") {
        throw std::runtime_error("decoder_delegate must be 'cpu' or 'gpu'");
    }
    auto result = gemma4::run_vision_text_rgb888(
        path.c_str(),
        reinterpret_cast<const uint8_t*>(rgb.c_str()),
        width,
        height,
        row_stride,
        prompt,
        max_tokens,
        host_platform(),
        encoder_hw,
        decoder_hw);
    return vision_text_result_to_python(result);
}

static nb::dict run_engine_init_smoke(
        const std::string& llm_model,
        const std::string& rag_embedding,
        const std::string& vlm_model,
        const std::string& llm_plan,
        const std::string& rag_plan,
        const std::string& vlm_plan,
        int num_threads) {
    TinyLLMEngine engine;
    TinyLLMExecutionConfig config;
    config.llm_plan = llm_plan;
    config.rag_plan = rag_plan;
    config.vlm_plan = vlm_plan;

    bool ok = engine.init(
        host_platform(),
        config,
        num_threads,
        llm_model.empty() ? nullptr : llm_model.c_str(),
        rag_embedding.empty() ? nullptr : rag_embedding.c_str(),
        vlm_model.empty() ? nullptr : vlm_model.c_str(),
        nullptr);
    const bool vision_ok = vlm_model.empty() || engine.vlm.initialized();
    ok = ok && vision_ok;

    nb::dict result;
    result["ok"] = ok;
    result["vision_mode_ok"] = vision_ok;
    result["llm_plan"] = llm_plan;
    result["rag_plan"] = rag_plan;
    result["vlm_plan"] = vlm_plan;
    result["has_text"] = engine.llm != nullptr;
    result["has_rag"] = engine.rag != nullptr;
    result["has_vlm"] = engine.vlm.initialized();
    engine.release();
    return result;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

NB_MODULE(zephr_agent_runtime_nanobind, m) {
    m.doc() = "Zephr Agent core native inference engine";

    m.def("format_tool_call_prompt",
        [](const std::string& model_family,
           const std::string& tool_name,
           const std::string& tool_description,
           const std::vector<std::tuple<std::string, std::string, std::string,
                                        std::vector<std::string>, bool>>& params,
           const std::string& user_message,
           const std::string& system_message,
           const std::string& prompt_profile) {
            auto tool = tool_from_args(tool_name, tool_description, params);
            return ::format_tool_call_prompt(
                model_family_from_name(model_family), tool, user_message,
                system_message, prompt_profile);
        },
        nb::arg("model_family"), nb::arg("tool_name"), nb::arg("tool_description"),
        nb::arg("params"), nb::arg("user_message"), nb::arg("system_message") = "",
        nb::arg("prompt_profile") = "",
        "Format a tool-call prompt using the selected model-family formatter");

    m.def("format_tool_call_completion",
        [](const std::string& model_family,
           const std::string& tool_name,
           const std::string& tool_description,
           const std::vector<std::tuple<std::string, std::string, std::string,
                                        std::vector<std::string>, bool>>& params,
           const std::map<std::string, std::string>& arguments,
           bool include_function_response) {
            auto tool = tool_from_args(tool_name, tool_description, params);
            return ::format_tool_call_completion(
                model_family_from_name(model_family), tool, arguments, include_function_response);
        },
        nb::arg("model_family"), nb::arg("tool_name"), nb::arg("tool_description"),
        nb::arg("params"), nb::arg("arguments"), nb::arg("include_function_response") = true,
        "Format a tool-call completion using the selected model-family formatter");

    m.def("set_log_level", [](const std::string& level) {
        std::string s = level;
        for (auto& c : s) c = (char)std::toupper((unsigned char)c);
        tinylog::Level l = tinylog::Level::Info;
        if (s == "TRACE") l = tinylog::Level::Trace;
        else if (s == "DEBUG") l = tinylog::Level::Debug;
        else if (s == "INFO") l = tinylog::Level::Info;
        else if (s == "WARN") l = tinylog::Level::Warn;
        else if (s == "ERROR") l = tinylog::Level::Error;
        tlog().set_level(l);
    }, nb::arg("level"), "Set native log level (trace/debug/info/warn/error)");

    m.def("detect_model_family", [](const std::string& path) {
        auto result = detect_model_family(path.c_str());
        if (result.empty()) throw std::runtime_error("Failed to open: " + path);
        return result;
    }, nb::arg("path"), "Detect model family from .litertlm bundle");

    m.def("list_bundle_sections", [](const std::string& path) {
        auto sections = list_bundle_sections(path.c_str());
        if (sections.empty()) throw std::runtime_error("No LiteRT-LM bundle sections found: " + path);
        return sections;
    }, nb::arg("path"), "List .litertlm bundle sections as (type, byte_size) tuples");

    m.def("inspect_litertlm_bundle", &inspect_litertlm_bundle, nb::arg("path"),
          "Inspect .litertlm bundle sections, signatures, tensor names, and tensor shapes");

    m.def("run_gemma4_vision_smoke", &run_gemma4_vision_smoke,
          nb::arg("path"), nb::arg("num_patches") = 2304,
          "Run Gemma4 vision encoder+adapter on a synthetic patch tensor");

    m.def("run_gemma4_vlm_text_smoke", &run_gemma4_vlm_text_smoke,
          nb::arg("path"), nb::arg("prompt") = "Describe the image.",
          nb::arg("num_patches") = 2304, nb::arg("max_tokens") = 16,
          "Run Gemma4 synthetic image embeddings through the text decoder");

    m.def("run_gemma4_vlm_rgb888", &run_gemma4_vlm_rgb888,
          nb::arg("path"), nb::arg("rgb"), nb::arg("width"), nb::arg("height"),
          nb::arg("row_stride"), nb::arg("prompt") = "Describe the image.",
          nb::arg("max_tokens") = 64, nb::arg("encoder_delegate") = "cpu",
          nb::arg("decoder_delegate") = "cpu",
          "Run Gemma4 VLM on an RGB888 image buffer");

    m.def("run_engine_init_smoke", &run_engine_init_smoke,
          nb::arg("llm_model"), nb::arg("rag_embedding") = "",
          nb::arg("vlm_model") = "", nb::arg("llm_plan") = "cpu",
          nb::arg("rag_plan") = "cpu", nb::arg("vlm_plan") = "gpu",
          nb::arg("num_threads") = 0,
          "Initialize TinyLLMEngine with explicit role plans and report prepared capabilities");

    nb::class_<Tokenizer>(m, "Tokenizer")
        .def(nb::init<>())
        .def("load", [](Tokenizer& self, const std::string& path) -> bool {
            MappedFile mf;
            if (!mf.open(path.c_str())) return false;
            if (mf.size >= 8 && memcmp(mf.data, "LITERTLM", 8) == 0) {
                auto [data, size] = find_bundle_tokenizer(mf.data, mf.size);
                if (!data) {
                    tlog().error("no SP_TOKENIZER section in bundle", {{"path", path}});
                    mf.close();
                    return false;
                }
                bool ok = self.load_from_buffer(data, size);
                mf.close();
                return ok;
            }
            mf.close();
            return self.load(path);
        })
        .def("encode", &Tokenizer::encode)
        .def("decode", &Tokenizer::decode)
        .def("piece_to_id", &Tokenizer::piece_to_id)
        .def("id_to_piece", &Tokenizer::id_to_piece)
        .def("vocab_size", &Tokenizer::vocab_size);

    // --- InferenceEngine (accessed via engine.text) ---

    nb::class_<InferenceEngine>(m, "InferenceEngine")
        .def("reset", &InferenceEngine::reset)
        .def("prefill", [](InferenceEngine& e, const std::vector<int32_t>& tokens) {
            return e.prefill(tokens.data(), (int)tokens.size());
        })
        .def("decode", [](InferenceEngine& e, int32_t token_id, float temperature, int top_k, float top_p) {
            return e.decode_sample(token_id, temperature, top_k, top_p);
        }, nb::arg("token_id"), nb::arg("temperature") = 0.0f,
           nb::arg("top_k") = 40, nb::arg("top_p") = 1.0f)
        .def("decode_with_logits", [](InferenceEngine& e, int32_t token_id, float temperature,
                                      int top_k, float top_p, int top_n) {
            std::vector<float> logits(VOCAB_SIZE);
            if (!e.decode_logits(token_id, logits.data()))
                return nb::make_tuple(-1, nb::list());
            int sampled = sample_topk_topp(logits.data(), VOCAB_SIZE,
                                            temperature, top_k, top_p, e.rng());
            std::vector<std::pair<int, float>> top;
            extract_topk(logits.data(), VOCAB_SIZE, top_n, top);
            nb::list top_list;
            for (auto& [id, logit] : top)
                top_list.append(nb::make_tuple(id, logit));
            return nb::make_tuple(sampled, top_list);
        }, nb::arg("token_id"), nb::arg("temperature") = 0.0f,
           nb::arg("top_k") = 40, nb::arg("top_p") = 1.0f, nb::arg("top_n") = 10)
        .def("activation_dim", &InferenceEngine::activation_dim)
        .def("decode_with_logits_and_activations", [](InferenceEngine& e, int32_t token_id,
                                                      float temperature, int top_k,
                                                      float top_p, int top_n) {
            int activation_dim = e.activation_dim();
            if (activation_dim <= 0)
                return nb::make_tuple(-1, nb::list(), nb::list());
            std::vector<float> logits(VOCAB_SIZE);
            std::vector<float> activations((size_t)activation_dim);
            if (!e.decode_logits_activations(token_id, logits.data(), activations.data()))
                return nb::make_tuple(-1, nb::list(), nb::list());
            int sampled = sample_topk_topp(logits.data(), VOCAB_SIZE,
                                            temperature, top_k, top_p, e.rng());
            std::vector<std::pair<int, float>> top;
            extract_topk(logits.data(), VOCAB_SIZE, top_n, top);
            nb::list top_list;
            for (auto& [id, logit] : top)
                top_list.append(nb::make_tuple(id, logit));
            nb::list activation_list;
            for (float value : activations)
                activation_list.append(value);
            return nb::make_tuple(sampled, top_list, activation_list);
        }, nb::arg("token_id"), nb::arg("temperature") = 0.0f,
           nb::arg("top_k") = 40, nb::arg("top_p") = 1.0f, nb::arg("top_n") = 10)
        .def("constrained_decode", [](InferenceEngine& e, const Tokenizer& tok, int pending_token,
                                      const std::string& tool_name,
                                      const std::vector<std::pair<std::string, std::vector<std::string>>>& param_defs,
                                      float temperature, int top_k, float top_p, int diag_top_k) {
            std::vector<ConstrainedParam> params;
            for (auto& [name, enums] : param_defs)
                params.push_back({name, enums});

            auto cr = e.constrained_decode(tok, pending_token, tool_name, params,
                                           temperature, top_k, top_p, diag_top_k);

            nb::dict params_dict;
            for (auto& [k, v] : cr.params)
                params_dict[nb::str(k.c_str())] = nb::str(v.c_str());

            nb::list logs;
            for (auto& entry : cr.logs) {
                nb::list top;
                for (auto& [id, logit] : entry.top_logits)
                    top.append(nb::make_tuple(id, logit));
                logs.append(nb::make_tuple(entry.step, entry.valid_count, entry.sampled_id, top));
            }

            return nb::make_tuple(cr.response, params_dict, logs, cr.decode_steps);
        }, nb::arg("tok"), nb::arg("pending_token"),
           nb::arg("tool_name"), nb::arg("params"),
           nb::arg("temperature") = 0.0f, nb::arg("top_k") = 40,
           nb::arg("top_p") = 1.0f, nb::arg("diag_top_k") = 8)
        .def("generate_tool_call", [](InferenceEngine& e,
                                  const std::string& tool_name,
                                  const std::string& tool_description,
                                  const std::vector<std::tuple<std::string, std::string, std::string, std::vector<std::string>, bool>>& param_defs,
                                  const std::string& user_message,
                                  float temperature, int top_k, float top_p,
                                  const std::string& system_message,
                                  const std::string& prompt_profile) {
            ToolDef tool;
            tool.name = tool_name;
            tool.description = tool_description;
            for (auto& [name, desc, type, enums, req] : param_defs)
                tool.params.push_back({name, desc, type, enums, req});

            auto result = e.generate_tool_call(tool, user_message,
                                           temperature, top_k, top_p, system_message, 8,
                                           prompt_profile);

            nb::dict params_dict;
            for (auto& [k, v] : result.params)
                params_dict[nb::str(k.c_str())] = nb::str(v.c_str());

            return nb::make_tuple(
                params_dict,
                result.prefill_tokens,
                result.decode_steps,
                result.prefill_ms,
                result.decode_ms,
                result.first_decode_ms
            );
        }, nb::arg("tool_name"), nb::arg("tool_description"),
           nb::arg("params"),
           nb::arg("user_message"),
           nb::arg("temperature") = 0.0f, nb::arg("top_k") = 40,
           nb::arg("top_p") = 1.0f, nb::arg("system_message") = "",
           nb::arg("prompt_profile") = "")
        .def("generate_tool_call_stats", [](InferenceEngine& e,
                                  const std::string& tool_name,
                                  const std::string& tool_description,
                                  const std::vector<std::tuple<std::string, std::string, std::string, std::vector<std::string>, bool>>& param_defs,
                                  const std::string& user_message,
                                  float temperature, int top_k, float top_p,
                                  const std::string& system_message,
                                  const std::string& prompt_profile) {
            ToolDef tool;
            tool.name = tool_name;
            tool.description = tool_description;
            for (auto& [name, desc, type, enums, req] : param_defs)
                tool.params.push_back({name, desc, type, enums, req});

            auto result = e.generate_tool_call(tool, user_message,
                                           temperature, top_k, top_p, system_message, 8,
                                           prompt_profile);
            return generate_result_to_stats_dict(result);
        }, nb::arg("tool_name"), nb::arg("tool_description"),
           nb::arg("params"),
           nb::arg("user_message"),
           nb::arg("temperature") = 0.0f, nb::arg("top_k") = 40,
           nb::arg("top_p") = 1.0f, nb::arg("system_message") = "",
           nb::arg("prompt_profile") = "")
        .def("generate_text", [](InferenceEngine& e,
                                 const std::string& user_message,
                                 const std::string& system_message,
                                 int max_tokens,
                                 float temperature,
                                 int top_k,
                                 float top_p) {
            auto result = e.generate_text(
                user_message, system_message, max_tokens, temperature, top_k, top_p);
            return nb::make_tuple(
                result.response,
                result.input_ids_count,
                result.prefill_tokens,
                result.decode_steps,
                result.prefill_ms,
                result.decode_ms,
                result.first_decode_ms
            );
        }, nb::arg("user_message"), nb::arg("system_message") = "",
           nb::arg("max_tokens") = 256, nb::arg("temperature") = 0.7f,
           nb::arg("top_k") = 40, nb::arg("top_p") = 0.95f)
        .def("generate_text_stats", [](InferenceEngine& e,
                                       const std::string& user_message,
                                       const std::string& system_message,
                                       int max_tokens,
                                       float temperature,
                                       int top_k,
                                       float top_p) {
            auto result = e.generate_text(
                user_message, system_message, max_tokens, temperature, top_k, top_p);
            return generate_result_to_stats_dict(result);
        }, nb::arg("user_message"), nb::arg("system_message") = "",
           nb::arg("max_tokens") = 256, nb::arg("temperature") = 0.7f,
           nb::arg("top_k") = 40, nb::arg("top_p") = 0.95f)
        .def("generate_tool_aware_text_stats", [](InferenceEngine& e,
                                                  const std::vector<std::tuple<
                                                      std::string,
                                                      std::string,
                                                      std::vector<std::tuple<
                                                          std::string,
                                                          std::string,
                                                          std::string,
                                                          std::vector<std::string>,
                                                          bool>>>>& tool_specs,
                                                  const std::string& user_message,
                                                  const std::string& system_message,
                                                  int max_tokens,
                                                  float temperature,
                                                  int top_k,
                                                  float top_p,
                                                  int reserve_output_tokens) {
            auto tools = tools_from_python_specs(tool_specs);
            auto result = e.generate_tool_aware_text(
                tools, user_message, system_message, max_tokens, temperature, top_k,
                top_p, reserve_output_tokens);
            return generate_result_to_stats_dict(result);
        }, nb::arg("tools"), nb::arg("user_message"), nb::arg("system_message") = "",
           nb::arg("max_tokens") = 256, nb::arg("temperature") = 0.0f,
           nb::arg("top_k") = 40, nb::arg("top_p") = 0.95f,
           nb::arg("reserve_output_tokens") = 0)
        .def("generate_tool_aware_text_from_prompt_stats", [](InferenceEngine& e,
                                                               const std::vector<std::tuple<
                                                                   std::string,
                                                                   std::string,
                                                                   std::vector<std::tuple<
                                                                       std::string,
                                                                       std::string,
                                                                       std::string,
                                                                       std::vector<std::string>,
                                                                       bool>>>>& tool_specs,
                                                               const std::string& prompt,
                                                               int max_tokens,
                                                               float temperature,
                                                               int top_k,
                                                               float top_p,
                                                               int reserve_output_tokens) {
            auto tools = tools_from_python_specs(tool_specs);
            auto result = e.generate_tool_aware_text_from_prompt(
                tools, prompt, max_tokens, temperature, top_k, top_p,
                reserve_output_tokens);
            return generate_result_to_stats_dict(result);
        }, nb::arg("tools"), nb::arg("prompt"),
           nb::arg("max_tokens") = 256, nb::arg("temperature") = 0.0f,
           nb::arg("top_k") = 40, nb::arg("top_p") = 0.95f,
           nb::arg("reserve_output_tokens") = 0)
        .def("generate_text_from_prompt_stats", [](InferenceEngine& e,
                                                   const std::string& prompt,
                                                   int max_tokens,
                                                   float temperature,
                                                   int top_k,
                                                   float top_p) {
            auto result = e.generate_text_from_prompt(
                prompt, max_tokens, temperature, top_k, top_p);
            return generate_result_to_stats_dict(result);
        }, nb::arg("prompt"), nb::arg("max_tokens") = 256,
           nb::arg("temperature") = 0.0f, nb::arg("top_k") = 40,
           nb::arg("top_p") = 0.95f)
        .def("continue_after_tool_response_stats", [](InferenceEngine& e,
                                                      const std::string& tool_response,
                                                      int max_tokens,
                                                      float temperature,
                                                      int top_k,
                                                      float top_p,
                                                      int reserve_output_tokens) {
            auto result = e.continue_after_tool_response(
                tool_response, max_tokens, temperature, top_k, top_p,
                reserve_output_tokens);
            return generate_result_to_stats_dict(result);
        }, nb::arg("tool_response"), nb::arg("max_tokens") = 256,
           nb::arg("temperature") = 0.0f, nb::arg("top_k") = 40,
           nb::arg("top_p") = 0.95f, nb::arg("reserve_output_tokens") = 0)
        .def("continue_tool_aware_text_stats", [](InferenceEngine& e,
                                                  const std::vector<std::tuple<
                                                      std::string,
                                                      std::string,
                                                      std::vector<std::tuple<
                                                          std::string,
                                                          std::string,
                                                          std::string,
                                                          std::vector<std::string>,
                                                          bool>>>>& tool_specs,
                                                  const std::string& prompt_suffix,
                                                  int max_tokens,
                                                  float temperature,
                                                  int top_k,
                                                  float top_p,
                                                  int reserve_output_tokens) {
            auto tools = tools_from_python_specs(tool_specs);
            auto result = e.continue_tool_aware_text(
                tools, prompt_suffix, max_tokens, temperature, top_k, top_p,
                reserve_output_tokens);
            return generate_result_to_stats_dict(result);
        }, nb::arg("tools"), nb::arg("prompt_suffix"),
           nb::arg("max_tokens") = 256, nb::arg("temperature") = 0.0f,
           nb::arg("top_k") = 40, nb::arg("top_p") = 0.95f,
           nb::arg("reserve_output_tokens") = 0)
        .def("gemma4_mtp_debug_verify", [](InferenceEngine& e,
                                           const std::string& user_message,
                                           const std::string& system_message) {
            return mtp_debug_result_to_dict(
                e.gemma4_mtp_debug_verify(user_message, system_message));
        }, nb::arg("user_message"), nb::arg("system_message") = "")
        .def_prop_ro("prefill_seq_len", &InferenceEngine::prefill_seq_len)
        .def_prop_ro("current_pos", &InferenceEngine::current_pos)
        .def_prop_ro("gemma4_mtp_enabled", &InferenceEngine::gemma4_mtp_enabled)
        .def_prop_ro("gemma4_mtp_available", &InferenceEngine::gemma4_mtp_available);

    // --- RagEngine (accessed via engine.rag) ---

    nb::class_<RagEngine>(m, "RagEngine")
        .def("embed", [](RagEngine& e, const std::string& text, const std::string& task_type) {
            EmbeddingTaskType tt = EmbeddingTaskType::RETRIEVAL_QUERY;
            if (task_type == "document") tt = EmbeddingTaskType::RETRIEVAL_DOCUMENT;
            else if (task_type == "similarity") tt = EmbeddingTaskType::SEMANTIC_SIMILARITY;
            std::vector<float> out;
            if (!e.embed(text, tt, out))
                throw std::runtime_error("embedding inference failed");
            return out;
        }, nb::arg("text"), nb::arg("task_type") = "query")
        .def("add", [](RagEngine& e, const std::string& id,
                        const std::vector<float>& embedding, double lat, double lon) {
            e.add(id, embedding.data(), (int)embedding.size(), lat, lon);
        }, nb::arg("id"), nb::arg("embedding"), nb::arg("lat") = 0.0, nb::arg("lon") = 0.0)
        .def("clear", &RagEngine::clear)
        .def("query", [](RagEngine& e, const std::string& text,
                          double lat, double lon, double radius_m, int top_k,
                          float similarity_threshold) {
            auto results = e.query(text, lat, lon, radius_m, top_k, similarity_threshold);
            nb::list out;
            for (auto& r : results)
                out.append(nb::make_tuple(r.id, r.similarity, r.distance_m, r.combined_score));
            return out;
        }, nb::arg("text"), nb::arg("lat") = 0.0, nb::arg("lon") = 0.0,
           nb::arg("radius_m") = 0.0, nb::arg("top_k") = 5,
           nb::arg("similarity_threshold") = 0.3f)
        .def("query_visual", [](RagEngine& e, const std::string& text,
                                 double lat, double lon, double radius_m, int top_k,
                                 float similarity_threshold) {
            auto results = e.query_visual(text, lat, lon, radius_m, top_k, similarity_threshold);
            nb::list out;
            for (auto& r : results)
                out.append(nb::make_tuple(r.id, r.similarity, r.distance_m, r.combined_score));
            return out;
        }, nb::arg("text"), nb::arg("lat") = 0.0, nb::arg("lon") = 0.0,
           nb::arg("radius_m") = 0.0, nb::arg("top_k") = 5,
           nb::arg("similarity_threshold") = 0.25f)
        .def("query_by_embedding", [](RagEngine& e, const std::vector<float>& embedding,
                                       double lat, double lon, double radius_m, int top_k,
                                       float similarity_threshold) {
            auto results = e.query_by_embedding(embedding.data(), (int)embedding.size(),
                                                 lat, lon, radius_m, top_k, similarity_threshold);
            nb::list out;
            for (auto& r : results)
                out.append(nb::make_tuple(r.id, r.similarity, r.distance_m, r.combined_score));
            return out;
        }, nb::arg("embedding"), nb::arg("lat") = 0.0, nb::arg("lon") = 0.0,
           nb::arg("radius_m") = 0.0, nb::arg("top_k") = 5,
           nb::arg("similarity_threshold") = 0.3f)
        .def_prop_ro("corpus_size", &RagEngine::corpus_size)
        .def_prop_ro("embedding_dim", [](const RagEngine& e) { return e.embedding_dim; })
        .def_prop_ro("max_seq_len", [](const RagEngine& e) { return e.max_seq_len; });

    // --- ZephrAgentRuntime (developer wrapper over the native engine) ---

    nb::class_<ZephrAgentRuntimeWrapper>(m, "ZephrAgentRuntime")
        .def(nb::init<>())
        .def("init", [](ZephrAgentRuntimeWrapper& self, bool gpu, int num_threads,
                        const std::string& llm_model,
                        const std::string& rag_embedding,
                        const std::string& vlm_model,
                        const std::string& llm_plan,
                        const std::string& rag_plan,
                        const std::string& vlm_plan,
                        bool gemma4_mtp_enabled,
                        int gemma4_gpu_precision,
                        int gemma4_kv_cache_max_len,
                        int gemma4_constrained_verify_batch,
                        bool gemma4_mtp_trust_verify_kv,
                        bool gemma4_mtp_adaptive_enabled,
                        int gemma4_mtp_adaptive_min_cycles,
                        float gemma4_mtp_adaptive_min_saved_per_cycle,
                        bool gemma4_mtp_trace,
                        bool diagnostic_gemma4_prefill_by_decode,
                        int diagnostic_gemma4_prefill_max_chunk,
                        bool diagnostic_gemma4_constrained_verify_trace,
                        int diagnostic_gemma4_constrained_verify_max_accept) {
            TinyLLMExecutionConfig config;
            config.llm_plan = llm_plan.empty() ? (gpu ? "gpu" : "cpu") : llm_plan;
            config.rag_plan = rag_plan.empty() ? (gpu ? "gpu" : "cpu") : rag_plan;
            config.vlm_plan = vlm_plan.empty() ? (gpu ? "gpu" : "cpu") : vlm_plan;
            config.gemma4_runtime.gpu_precision = gemma4_gpu_precision;
            config.gemma4_runtime.kv_cache_max_len = gemma4_kv_cache_max_len;
            config.gemma4_runtime.constrained_verify_batch = gemma4_constrained_verify_batch;
            config.gemma4_runtime.mtp_enabled = gemma4_mtp_enabled;
            config.gemma4_runtime.mtp_trust_verify_kv = gemma4_mtp_trust_verify_kv;
            config.gemma4_runtime.mtp_adaptive_enabled = gemma4_mtp_adaptive_enabled;
            config.gemma4_runtime.mtp_adaptive_min_cycles = gemma4_mtp_adaptive_min_cycles;
            config.gemma4_runtime.mtp_adaptive_min_saved_per_cycle =
                gemma4_mtp_adaptive_min_saved_per_cycle;
            config.gemma4_runtime.mtp_trace = gemma4_mtp_trace;
            config.diagnostic_gemma4.prefill_by_decode =
                diagnostic_gemma4_prefill_by_decode;
            config.diagnostic_gemma4.prefill_max_chunk =
                diagnostic_gemma4_prefill_max_chunk;
            config.diagnostic_gemma4.constrained_verify_trace =
                diagnostic_gemma4_constrained_verify_trace;
            config.diagnostic_gemma4.constrained_verify_max_accept =
                diagnostic_gemma4_constrained_verify_max_accept;
            bool ok = self.engine.init(
                host_platform(),
                config,
                num_threads,
                llm_model.empty() ? nullptr : llm_model.c_str(),
                rag_embedding.empty() ? nullptr : rag_embedding.c_str(),
                vlm_model.empty() ? nullptr : vlm_model.c_str(),
                nullptr);
            return ok;
        }, nb::arg("gpu") = false, nb::arg("num_threads") = 0,
           nb::arg("llm_model") = "", nb::arg("rag_embedding") = "",
           nb::arg("vlm_model") = "", nb::arg("llm_plan") = "",
           nb::arg("rag_plan") = "", nb::arg("vlm_plan") = "",
           nb::arg("gemma4_mtp_enabled") = false,
           nb::arg("gemma4_gpu_precision") = -1,
           nb::arg("gemma4_kv_cache_max_len") = 0,
           nb::arg("gemma4_constrained_verify_batch") = -1,
           nb::arg("gemma4_mtp_trust_verify_kv") = true,
           nb::arg("gemma4_mtp_adaptive_enabled") = true,
           nb::arg("gemma4_mtp_adaptive_min_cycles") = 4,
           nb::arg("gemma4_mtp_adaptive_min_saved_per_cycle") = 0.5f,
           nb::arg("gemma4_mtp_trace") = false,
           nb::arg("diagnostic_gemma4_prefill_by_decode") = false,
           nb::arg("diagnostic_gemma4_prefill_max_chunk") = 0,
           nb::arg("diagnostic_gemma4_constrained_verify_trace") = false,
           nb::arg("diagnostic_gemma4_constrained_verify_max_accept") = 0)
        .def("drain_model_lifecycle_timings", [](ZephrAgentRuntimeWrapper& self) {
            nb::list out;
            for (const auto& timing : self.engine.model_lifecycle_timings) {
                nb::dict item;
                item["component"] = timing.component;
                item["action"] = timing.action;
                item["detail"] = timing.detail;
                item["durationMs"] = timing.duration_ms;
                item["ok"] = timing.ok;
                out.append(item);
            }
            self.engine.model_lifecycle_timings.clear();
            return out;
        })
        .def("describe_image_rgb888", [](ZephrAgentRuntimeWrapper& self,
                                         nb::bytes rgb,
                                         int width,
                                         int height,
                                         int row_stride,
                                         const std::string& prompt,
                                         int max_tokens) {
            const int expected_min = height * row_stride;
            if (width <= 0 || height <= 0 || row_stride < width * 3 ||
                (int)rgb.size() < expected_min)
                throw std::runtime_error("invalid RGB888 image buffer");
            gemma4::VisionTextSmokeResult result;
            if (!self.engine.describe_image_rgb888(
                    reinterpret_cast<const uint8_t*>(rgb.c_str()),
                    width,
                    height,
                    row_stride,
                    prompt,
                    max_tokens,
                    result))
                throw std::runtime_error("ZephrAgentRuntime VLM describe failed");
            return vision_text_result_to_python(result);
        }, nb::arg("rgb"), nb::arg("width"), nb::arg("height"),
           nb::arg("row_stride"), nb::arg("prompt") = "Describe the image.",
           nb::arg("max_tokens") = 0)
        .def("generate_text_stats", [](ZephrAgentRuntimeWrapper& self,
                                       const std::string& user_message,
                                       const std::string& system_message,
                                       int max_tokens,
                                       float temperature,
                                       int top_k,
                                       float top_p) {
            if (!self.engine.llm)
                throw std::runtime_error("ZephrAgentRuntime LLM is not initialized");
            auto result = self.engine.llm->generate_text(
                user_message, system_message, max_tokens, temperature, top_k, top_p);
            return generate_result_to_stats_dict(result);
        }, nb::arg("user_message"), nb::arg("system_message") = "",
           nb::arg("max_tokens") = 256, nb::arg("temperature") = 0.7f,
           nb::arg("top_k") = 40, nb::arg("top_p") = 0.95f)
        .def("release", [](ZephrAgentRuntimeWrapper& self) { self.engine.release(); })
        .def("clear_loaded_corpus", [](ZephrAgentRuntimeWrapper& self) {
            if (self.engine.rag) self.engine.rag->clear();
        })
        .def_prop_ro("corpus_size", [](ZephrAgentRuntimeWrapper& self) -> int {
            return self.engine.rag ? (int)self.engine.rag->corpus_size() : 0;
        });

    // --- TinyLLMEngine (top-level) ---
    //
    // Raw Engine/InferenceEngine/RagEngine bindings are intentionally
    // single-threaded developer APIs. They expose borrowed subsystem pointers;
    // callers must serialize access and must not use a subsystem after release().

    nb::class_<TinyLLMEngine>(m, "Engine")
        .def(nb::init<>())
        .def("init", [](TinyLLMEngine& e, bool gpu, int num_threads,
                        const std::string& llm_model,
                        const std::string& rag_embedding,
                        bool gemma4_mtp_enabled,
                        int gemma4_gpu_precision,
                        int gemma4_kv_cache_max_len,
                        int gemma4_constrained_verify_batch,
                        bool gemma4_mtp_trust_verify_kv,
                        bool gemma4_mtp_adaptive_enabled,
                        int gemma4_mtp_adaptive_min_cycles,
                        float gemma4_mtp_adaptive_min_saved_per_cycle,
                        bool gemma4_mtp_trace,
                        bool diagnostic_gemma4_prefill_by_decode,
                        int diagnostic_gemma4_prefill_max_chunk,
                        bool diagnostic_gemma4_constrained_verify_trace,
                        int diagnostic_gemma4_constrained_verify_max_accept) {
            TinyLLMExecutionConfig config;
            config.llm_plan = gpu ? "gpu" : "cpu";
            config.rag_plan = gpu ? "gpu" : "cpu";
            config.vlm_plan = gpu ? "gpu" : "cpu";
            config.gemma4_runtime.gpu_precision = gemma4_gpu_precision;
            config.gemma4_runtime.kv_cache_max_len = gemma4_kv_cache_max_len;
            config.gemma4_runtime.constrained_verify_batch = gemma4_constrained_verify_batch;
            config.gemma4_runtime.mtp_enabled = gemma4_mtp_enabled;
            config.gemma4_runtime.mtp_trust_verify_kv = gemma4_mtp_trust_verify_kv;
            config.gemma4_runtime.mtp_adaptive_enabled = gemma4_mtp_adaptive_enabled;
            config.gemma4_runtime.mtp_adaptive_min_cycles = gemma4_mtp_adaptive_min_cycles;
            config.gemma4_runtime.mtp_adaptive_min_saved_per_cycle =
                gemma4_mtp_adaptive_min_saved_per_cycle;
            config.gemma4_runtime.mtp_trace = gemma4_mtp_trace;
            config.diagnostic_gemma4.prefill_by_decode =
                diagnostic_gemma4_prefill_by_decode;
            config.diagnostic_gemma4.prefill_max_chunk =
                diagnostic_gemma4_prefill_max_chunk;
            config.diagnostic_gemma4.constrained_verify_trace =
                diagnostic_gemma4_constrained_verify_trace;
            config.diagnostic_gemma4.constrained_verify_max_accept =
                diagnostic_gemma4_constrained_verify_max_accept;
            return e.init(Platform::APPLE_OS,
                          config,
                          num_threads,
                          llm_model.empty() ? nullptr : llm_model.c_str(),
                          rag_embedding.empty() ? nullptr : rag_embedding.c_str());
        }, nb::arg("gpu") = false, nb::arg("num_threads") = 0,
           nb::arg("llm_model") = "", nb::arg("rag_embedding") = "",
           nb::arg("gemma4_mtp_enabled") = false,
           nb::arg("gemma4_gpu_precision") = -1,
           nb::arg("gemma4_kv_cache_max_len") = 0,
           nb::arg("gemma4_constrained_verify_batch") = -1,
           nb::arg("gemma4_mtp_trust_verify_kv") = true,
           nb::arg("gemma4_mtp_adaptive_enabled") = true,
           nb::arg("gemma4_mtp_adaptive_min_cycles") = 4,
           nb::arg("gemma4_mtp_adaptive_min_saved_per_cycle") = 0.5f,
           nb::arg("gemma4_mtp_trace") = false,
           nb::arg("diagnostic_gemma4_prefill_by_decode") = false,
           nb::arg("diagnostic_gemma4_prefill_max_chunk") = 0,
           nb::arg("diagnostic_gemma4_constrained_verify_trace") = false,
           nb::arg("diagnostic_gemma4_constrained_verify_max_accept") = 0)
        .def("release", &TinyLLMEngine::release)
        .def_prop_ro("text", [](TinyLLMEngine& e) -> InferenceEngine* { return e.llm; },
                     nb::rv_policy::reference_internal)
        .def_prop_ro("llm", [](TinyLLMEngine& e) -> InferenceEngine* { return e.llm; },
                     nb::rv_policy::reference_internal)
        .def_prop_ro("rag", [](TinyLLMEngine& e) -> RagEngine* { return e.rag; },
                     nb::rv_policy::reference_internal);
}
