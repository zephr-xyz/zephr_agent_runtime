// TinyLLM Gemma 4 LiteRT-LM runtime.
//
// Contains the Gemma 4 text decoder implementation and the companion vision
// encoder/adapter runtime used by VLM bundles.
//
// Multi-model inference: embedder + per-layer-embedder + decoder.
// The decoder takes float embeddings (not token IDs) — embedder and PLE
// are run first to convert tokens → embeddings.
//
// Model structure:
//   - Embedder: tokens[1] → embeddings[1, embedding_dim]
//   - Per-layer embedder: tokens[1] → per_layer_embeddings[1, ...]
//   - Decoder signatures: prefill_128, prefill_1024, decode
//
// Decoder input tensors (S = seq_len for prefill, 1 for decode; C = kv_cache_max_len):
//
//   embeddings          float32 [S, embedding_dim] Output of embedder for each token.
//   per_layer_embeddings float32 [S, ...]     Output of PLE for each token.
//   input_pos           int32   [S]           Absolute position indices (0-based).
//   mask                float32/float16 [S, C] Causal attention mask.
//                         0.0 = attend, large negative = masked.
//                         Row q attends to kv positions [0..pos_offset+q].
//                         Padding rows (q >= num_tokens) are fully masked.
//   param_tensor        int32   [1, 1, 1, 7]  GPU single-buffer KV cache control.
//                         [start_idx, end_idx, end_idx, 0, 0, 0, 0]
//                         Tells the GPU kernel which cache slice to update.
//   kv_cache_k_{N}      float32 [1, C, head_dim]  Per-layer K cache.
//   kv_cache_v_{N}      float32 [1, C, head_dim]  Per-layer V cache.
//                         GPU: pass nullptr (Metal delegate uses internal buffers).
//                         CPU: pass same buffer as output (double-duty).
//   activations         float32 (output only, MTP drafter — unused, must be allocated)
//
// Decoder output tensors:
//   logits              float32 [1, vocab_size] Vocabulary logits (decode sig only).
//   kv_cache_k_{N}      GPU: placeholder (delegate writes to internal buffer).
//                        CPU: same buffer as input.
//   kv_cache_v_{N}      Same pattern as kv_cache_k.

#pragma once

#include "tinyllm_grammar.hpp"

#include <algorithm>
#include <functional>
#include <limits>

namespace gemma4 {

struct TextComponentTargets {
    HardwareTarget decoder = HardwareTarget::CPU;
    HardwareTarget embedder = HardwareTarget::CPU;
    HardwareTarget ple = HardwareTarget::CPU;
};

static inline TextComponentTargets text_component_targets(HardwareTarget decoder_hw) {
    return TextComponentTargets{
        decoder_hw,
        HardwareTarget::CPU,
        HardwareTarget::CPU,
    };
}

static inline std::string text_components(const TextComponentTargets& targets) {
    return std::string("decoder(") + hw_target_name(targets.decoder) + ")+embedder(" +
           hw_target_name(targets.embedder) + ")+ple(" + hw_target_name(targets.ple) + ")";
}

static inline std::string text_components(HardwareTarget decoder_hw) {
    return text_components(text_component_targets(decoder_hw));
}

static inline int environment_accelerator_mask(HardwareTarget decoder_hw) {
    const TextComponentTargets targets = text_component_targets(decoder_hw);
    return litert_options_accelerator_mask(targets.decoder, ModelFamily::GEMMA4) |
           litert_options_accelerator_mask(targets.embedder, ModelFamily::GEMMA4) |
           litert_options_accelerator_mask(targets.ple, ModelFamily::GEMMA4);
}

// MARK: - Text constants

static constexpr int NUM_LAYERS = 15;
static constexpr int EMBEDDING_DIM = 1536;
static constexpr int PLE_LAYERS = 35;
static constexpr int PLE_DIM = 256;
static constexpr int PLE_FLOATS = PLE_LAYERS * PLE_DIM;  // 8960
static constexpr int MTP_DRAFTER_INPUT_DIM = EMBEDDING_DIM * 2;
static constexpr int MTP_VERIFY_LOGIT_OFFSET = 0;
// Fallback context length for fixed-shape or unscanned Gemma4 bundles.
static constexpr int KV_CACHE_MAX_LEN_DEFAULT = 1536;
// Cap applied when a Gemma4 LiteRT-LM bundle exposes a magic-number context.
// Use an 8k cap for longer live conversations when the bundle supports it.
static constexpr int KV_CACHE_MAX_LEN_MAGIC_CAP = 8192;
// iOS Metal runs out of working room at the 8k Gemma4 decoder shape once the
// agent also owns embedding/VLM components. Keep automatic iOS runs at 4k while
// preserving explicit diagnostic overrides up to the model cap.
static constexpr int KV_CACHE_MAX_LEN_IOS_AUTOMATIC_CAP = 4096;
static constexpr int PARAM_TENSOR_SIZE = 7;

static inline int automatic_kv_cache_max_len_for_platform(Platform platform) {
#if defined(__APPLE__) && TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
    if (platform == Platform::APPLE_OS)
        return KV_CACHE_MAX_LEN_IOS_AUTOMATIC_CAP;
#else
    (void)platform;
#endif
    return KV_CACHE_MAX_LEN_MAGIC_CAP;
}

struct Gemma4ModelShape {
    int num_kv_layers = NUM_LAYERS;
    int embedding_dim = EMBEDDING_DIM;
    int ple_floats = PLE_FLOATS;
    int mtp_drafter_input_dim = MTP_DRAFTER_INPUT_DIM;
    int vocab_size = VOCAB_SIZE;
};

// Stop tokens: <pad>=0, <eos>=1, <turn|>=106
static const std::set<int> STOP_TOKENS = {0, 1, 106};

// Available prefill signature sizes (sorted ascending).
static constexpr int PREFILL_SIZES[] = {128, 1024};
static constexpr int NUM_PREFILL_SIZES = 2;

struct MtpStepTimings {
    int64_t model_us = 0;
    int64_t logits_read_us = 0;
    int64_t activation_read_us = 0;
    int64_t sample_us = 0;
    int logits_rows_read = 0;
    int activation_rows_read = 0;
};

static inline int64_t elapsed_us(
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();
}

// MARK: - Text prefill signature state

struct PrefillSig {
    int seq_len = 0;
    SignatureInfo sig;

    // Decoder input buffers for this prefill size
    LiteRtTensorBuffer embeddings_buf = nullptr;  // [seq, embedding_dim]
    LiteRtTensorBuffer ple_buf = nullptr;          // [seq, PLE_LAYERS, PLE_DIM]
    LiteRtTensorBuffer pos_buf = nullptr;          // [seq]
    LiteRtTensorBuffer mask_buf = nullptr;         // [seq, kv_cache_max_len]
    LiteRtTensorBuffer param_buf = nullptr;        // [1,1,1,7]
    LiteRtTensorBuffer activations_buf = nullptr;  // MTP output (unused)

    std::vector<LiteRtTensorBuffer> input_bufs;
    std::vector<LiteRtTensorBuffer> output_bufs;
};

struct VerifySig {
    SignatureInfo sig;

    LiteRtTensorBuffer embeddings_buf = nullptr;
    LiteRtTensorBuffer ple_buf = nullptr;
    LiteRtTensorBuffer pos_buf = nullptr;
    LiteRtTensorBuffer mask_buf = nullptr;
    LiteRtTensorBuffer param_buf = nullptr;
    LiteRtTensorBuffer logits_buf = nullptr;
    LiteRtTensorBuffer activations_buf = nullptr;
    std::vector<LiteRtTensorBuffer> kv_k;
    std::vector<LiteRtTensorBuffer> kv_v;

    std::vector<LiteRtTensorBuffer> input_bufs;
    std::vector<LiteRtTensorBuffer> output_bufs;
};

// MARK: - Text runtime state

struct State {
    LiteRtEnvironment env = nullptr;
    int execution_mode = EXEC_MODE_GPU;
    LiteRtMagicNumberConfigs* magic_configs = nullptr;  // must outlive env
    Gemma4RuntimeConfig runtime_config;
    Gemma4DiagnosticConfig diagnostic_config;
    int kv_cache_max_len = KV_CACHE_MAX_LEN_DEFAULT;
    Gemma4ModelShape shape;

    // Three models
    LiteRtModel decoder_model = nullptr;
    LiteRtModel embedder_model = nullptr;
    LiteRtModel ple_model = nullptr;
    LiteRtModel mtp_model = nullptr;
    MappedRegion decoder_region;
    MappedRegion mtp_region;
    const void* decoder_original_data = nullptr;
    uint64_t decoder_original_offset = 0;
    uint64_t decoder_original_length = 0;

    // Three compiled models
    LiteRtCompiledModel decoder_cm = nullptr;
    LiteRtCompiledModel embedder_cm = nullptr;
    LiteRtCompiledModel ple_cm = nullptr;
    LiteRtCompiledModel mtp_cm = nullptr;

    // Decoder options
    LiteRtOptions decoder_options = nullptr;
    LiteRtOptions embedder_options = nullptr;
    LiteRtOptions ple_options = nullptr;
    LiteRtOptions mtp_options = nullptr;

    // Prefill signatures (one per size)
    PrefillSig prefill_sigs[NUM_PREFILL_SIZES];

    // Decode signature
    SignatureInfo decode_sig;
    LiteRtTensorBuffer decode_embeddings_buf = nullptr;
    LiteRtTensorBuffer decode_ple_buf = nullptr;
    LiteRtTensorBuffer decode_pos_buf = nullptr;
    LiteRtTensorBuffer decode_mask_buf = nullptr;
    LiteRtTensorBuffer decode_param_buf = nullptr;
    LiteRtTensorBuffer decode_logits_buf = nullptr;
    LiteRtTensorBuffer decode_activations_buf = nullptr;  // MTP drafter output (unused, but must be allocated)
    std::vector<LiteRtTensorBuffer> decode_input_bufs;
    std::vector<LiteRtTensorBuffer> decode_output_bufs;
    std::vector<float> decode_embedding_scratch;
    std::vector<float> decode_ple_scratch;

    // Verify signature (target model parallel verification for MTP).
    VerifySig verify_sig;
    bool has_verify_sig = false;

    // MTP drafter signature + buffers. The drafter consumes the current token
    // embedding followed by the target decoder's activations.
    SignatureInfo mtp_sig;
    bool has_mtp_drafter = false;
    bool mtp_enabled = false;
    bool mtp_runtime_enabled = false;
    bool mtp_trust_verify_kv = false;
    bool mtp_trace = false;
    int mtp_verify_tokens = 0;
    int mtp_draft_tokens = 0;
    bool mtp_adaptive_enabled = true;
    int mtp_adaptive_min_cycles = 4;
    float mtp_adaptive_min_saved_per_cycle = 0.5f;
    bool constrained_verify_batch_enabled = true;
    LiteRtTensorBuffer mtp_pos_buf = nullptr;
    LiteRtTensorBuffer mtp_activations_buf = nullptr;
    LiteRtTensorBuffer mtp_param_buf = nullptr;
    LiteRtTensorBuffer mtp_mask_buf = nullptr;
    LiteRtTensorBuffer mtp_logits_buf = nullptr;
    LiteRtTensorBuffer mtp_projected_activations_buf = nullptr;
    std::vector<LiteRtTensorBuffer> mtp_input_bufs;
    std::vector<LiteRtTensorBuffer> mtp_output_bufs;

    // KV cache — shared between prefill and decode (single-buffered for GPU)
    std::vector<LiteRtTensorBuffer> kv_k;
    std::vector<LiteRtTensorBuffer> kv_v;

    // Embedder signature + buffers
    SignatureInfo emb_sig;
    LiteRtTensorBuffer emb_tokens_buf = nullptr;
    LiteRtTensorBuffer emb_output_buf = nullptr;
    std::vector<LiteRtTensorBuffer> emb_input_bufs;
    std::vector<LiteRtTensorBuffer> emb_output_bufs;

    // Per-layer embedder signature + buffers
    SignatureInfo ple_sig;
    LiteRtTensorBuffer ple_tokens_buf = nullptr;
    LiteRtTensorBuffer ple_output_buf = nullptr;
    std::vector<LiteRtTensorBuffer> ple_input_bufs;
    std::vector<LiteRtTensorBuffer> ple_output_bufs;

    int current_pos = 0;
    LiteRtElementType mask_element_type = kLiteRtElementTypeFloat32;
    LiteRtElementType verify_mask_element_type = kLiteRtElementTypeFloat32;
    LiteRtElementType mtp_mask_element_type = kLiteRtElementTypeBool;
    float masked_value = MASKED_FP16;

    std::mt19937 rng{std::random_device{}()};
    Tokenizer* tokenizer = nullptr;
};

static inline std::string prefill_lengths(const State& s) {
    std::string lengths;
    for (int i = 0; i < NUM_PREFILL_SIZES; i++) {
        if (s.prefill_sigs[i].seq_len > 0) {
            if (!lengths.empty()) lengths += ",";
            lengths += std::to_string(s.prefill_sigs[i].seq_len);
        }
    }
    return lengths;
}

static inline void log_bundle_section_digest(
        const char* phase,
        const char* label,
        const char* path,
        const void* data,
        uint64_t offset,
        uint64_t length) {
    const std::string ptr = pointer_hex(data);
    const std::string hash = stable_sampled_bytes_hash(data, static_cast<size_t>(length));
    tinylog::logger().info("Gemma4 bundle section digest",
        {{"phase", std::string(phase ? phase : "")},
         {"section", std::string(label ? label : "")},
         {"path", std::string(path ? path : "")},
         {"offset", (int64_t)offset},
         {"bytes", (int64_t)length},
         {"ptr", ptr},
         {"sample_hash", hash}});
}

// MARK: - Text initialization
// The .litertlm file is memory-mapped and models are loaded directly from
// their byte ranges via LiteRtCreateModelFromBuffer. The MappedFile must
// outlive the State (model data is read from the mapped buffer).

static inline bool load_models_from_bundle(State* s, const MappedFile& bundle) {
    auto sections = parse_litertlm_bundle(bundle.data, bundle.size);

    for (const auto& sec : sections) {
        if (sec.data_type != BundleSectionDataType::TFLITE_WEIGHTS)
            continue;
        tinylog::logger().error("unsupported Gemma4 external weights section",
            {{"model_type", std::string(bundle_model_type_name(sec.type))},
             {"path", bundle.path},
             {"offset", (int64_t)sec.begin_offset},
             {"bytes", (int64_t)sec.size}});
        return false;
    }

    auto load_section = [&](BundleModelType type,
                            const std::vector<BundleSection>& secs,
                            LiteRtModel* out, const char* label) -> bool {
        auto* sec = find_bundle_section(secs, type);
        if (!sec) {
            tinylog::logger().error("bundle section not found", {{"type", (int64_t)type}});
            return false;
        }
        tinylog::logger().debug("CreateModelFromBuffer",
            {{"label", std::string(label)}, {"path", bundle.path}, {"bytes", (int64_t)sec->size}});
        return litert_check(
            LITERT(LiteRtCreateModelFromBuffer)(s->env, sec->data, sec->size, out),
            "LiteRtCreateModelFromBuffer");
    };

    auto* decoder_sec = find_decoder_section(sections);
    if (!decoder_sec) {
        tinylog::logger().error("no decoder section found in bundle");
        return false;
    }
    // LiteRT's Gemma4 magic-number pass patches the decoder FlatBuffer during
    // compiled-model creation. Map just that section private-writable instead
    // of making the whole 2GB+ bundle writable.
    s->decoder_original_data = decoder_sec->data;
    s->decoder_original_offset = decoder_sec->begin_offset;
    s->decoder_original_length = decoder_sec->size;
    log_bundle_section_digest("load/original", "decoder", bundle.path.c_str(),
                              s->decoder_original_data, s->decoder_original_offset,
                              s->decoder_original_length);
    log_metal_memory_snapshot("before gemma4/decoder section mmap");
    log_process_memory_snapshot("before gemma4/decoder section mmap");
    if (!s->decoder_region.open(
            bundle.path.c_str(), decoder_sec->begin_offset, decoder_sec->size,
            MappedFileAccess::PrivateWritable))
        return false;
    log_bundle_section_digest("load/private_mmap", "decoder", bundle.path.c_str(),
                              s->decoder_region.data, s->decoder_original_offset,
                              s->decoder_region.size);
    log_metal_memory_snapshot("after gemma4/decoder section mmap");
    log_process_memory_snapshot("after gemma4/decoder section mmap");
    tinylog::logger().debug("CreateModelFromBuffer",
        {{"label", std::string("decoder")}, {"path", bundle.path}, {"bytes", (int64_t)decoder_sec->size}});
    if (!litert_check(
            LITERT(LiteRtCreateModelFromBuffer)(s->env, s->decoder_region.data,
                                                 s->decoder_region.size,
                                                 &s->decoder_model),
            "LiteRtCreateModelFromBuffer(decoder)"))
        return false;
    if (!load_section(BundleModelType::EMBEDDER, sections, &s->embedder_model, "embedder"))
        return false;
    if (!load_section(BundleModelType::PER_LAYER_EMBEDDER, sections, &s->ple_model, "ple"))
        return false;
    if (auto* mtp_sec = find_bundle_section(sections, BundleModelType::MTP_DRAFTER)) {
        if (!s->mtp_region.open(
                bundle.path.c_str(), mtp_sec->begin_offset, mtp_sec->size,
                MappedFileAccess::PrivateWritable))
            return false;
        tinylog::logger().debug("CreateModelFromBuffer",
            {{"label", std::string("mtp_drafter")}, {"path", bundle.path},
             {"bytes", (int64_t)mtp_sec->size}});
        if (!litert_check(
                LITERT(LiteRtCreateModelFromBuffer)(s->env, s->mtp_region.data,
                                                     s->mtp_region.size,
                                                     &s->mtp_model),
                "LiteRtCreateModelFromBuffer(mtp_drafter)"))
            return false;
        s->has_mtp_drafter = true;
    }

    return true;
}

static inline LiteRtMagicNumberConfigs* prescan_bundle_magic_configs(const MappedFile& bundle) {
    if (bundle.size < 8 || memcmp(bundle.data, "LITERTLM", 8) != 0)
        return nullptr;
    auto sections = parse_litertlm_bundle(bundle.data, bundle.size);
    if (!find_bundle_section(sections, BundleModelType::EMBEDDER))
        return nullptr;
    auto* decoder_section = find_decoder_section(sections);
    if (!decoder_section)
        return nullptr;

    LiteRtModel decoder_probe = nullptr;
    if (LITERT(LiteRtCreateModelFromBuffer)(
            nullptr, decoder_section->data, decoder_section->size,
            &decoder_probe) != kLiteRtStatusOk)
        return nullptr;

    auto* configs = build_magic_number_configs(decoder_probe);
    LITERT(LiteRtDestroyModel)(decoder_probe);
    return configs;
}

static inline int apply_magic_configs_to_kv_cache(
        LiteRtMagicNumberConfigs* configs,
        int configured_kv_cache_max_len = 0) {
    if (!configs)
        return KV_CACHE_MAX_LEN_DEFAULT;

    int64_t context_magic = 0;
    int64_t context_target = 0;
    for (int64_t i = 0; i < configs->num_configs; i++) {
        const int64_t target = configs->configs[i].target_number;
        if (!configs->configs[i].signature_prefix ||
            target > context_target) {
            context_magic = configs->configs[i].magic_number;
            context_target = target;
        }
    }
    if (context_magic > 0 && context_target > 0) {
        int kv_cache_cap = KV_CACHE_MAX_LEN_MAGIC_CAP;
        if (configured_kv_cache_max_len > 0) {
            kv_cache_cap = std::clamp(
                configured_kv_cache_max_len,
                KV_CACHE_MAX_LEN_DEFAULT,
                KV_CACHE_MAX_LEN_MAGIC_CAP);
        }
        int kv_cache_max_len = std::min((int)context_target, kv_cache_cap);
        for (int64_t i = 0; i < configs->num_configs; i++) {
            if (configs->configs[i].magic_number == context_magic)
                configs->configs[i].target_number = kv_cache_max_len;
        }
        return kv_cache_max_len;
    }
    return KV_CACHE_MAX_LEN_DEFAULT;
}

static inline void apply_magic_configs_to_state(State* s) {
    if (!s || !s->magic_configs)
        return;
    s->kv_cache_max_len = apply_magic_configs_to_kv_cache(
        s->magic_configs, s->runtime_config.kv_cache_max_len);
}

// MARK: - Text param_tensor helpers
// param_tensor[1,1,1,7] = {start_index, end_index, end_index, 0, 0, 0, 0}
// See: LiteRT-LM litert_compiled_model_executor_utils.cc:303-323

static inline bool write_param_tensor(LiteRtTensorBuffer buf,
                                      int start_index, int update_length) {
    int32_t params[PARAM_TENSOR_SIZE] = {};
    int end_index = start_index + update_length;
    params[0] = start_index;
    params[1] = end_index;
    params[2] = end_index;
    return write_int_buf(buf, params, PARAM_TENSOR_SIZE);
}

// MARK: - Text embedding lookup

static inline bool lookup_embedding(State* s, int32_t token_id,
                                    float* embedding_out) {
    // tinylog::logger().trace("lookup_embedding", {{"token", (int64_t)token_id}});
    write_int_buf(s->emb_tokens_buf, &token_id, 1);

    if (!litert_check(LITERT(LiteRtRunCompiledModel)(
                          s->embedder_cm, s->emb_sig.sig_index,
                          s->emb_input_bufs.size(), s->emb_input_bufs.data(),
                          s->emb_output_bufs.size(), s->emb_output_bufs.data()),
                      "RunCompiledModel(embedder)"))
        return false;

    return read_float_buf(s->emb_output_buf, embedding_out, s->shape.embedding_dim);
}

static inline bool lookup_ple(State* s, int32_t token_id,
                              float* ple_out) {
    write_int_buf(s->ple_tokens_buf, &token_id, 1);

    if (!litert_check(LITERT(LiteRtRunCompiledModel)(
                          s->ple_cm, s->ple_sig.sig_index,
                          s->ple_input_bufs.size(), s->ple_input_bufs.data(),
                          s->ple_output_bufs.size(), s->ple_output_bufs.data()),
                      "RunCompiledModel(ple)"))
        return false;

    return read_float_buf(s->ple_output_buf, ple_out, s->shape.ple_floats);
}

// MARK: - Text buffer array builders

static inline bool uses_gpu_single_buffer_kv_cache(const State* s) {
    return s->execution_mode == EXEC_MODE_GPU &&
           s->decode_sig.input_index_of("param_tensor") >= 0;
}

static inline int parse_kv_cache_layer_index(const std::string& name,
                                             const char* prefix,
                                             int num_layers) {
    const size_t prefix_len = std::strlen(prefix);
    if (name.rfind(prefix, 0) != 0 || name.size() == prefix_len)
        return -1;
    int layer = 0;
    for (size_t i = prefix_len; i < name.size(); i++) {
        const char c = name[i];
        if (c < '0' || c > '9')
            return -1;
        layer = layer * 10 + (c - '0');
        if (layer >= num_layers)
            return -1;
    }
    return layer;
}

static inline LiteRtTensorBuffer kv_cache_buffer_for_name(
        const std::string& name,
        const char* prefix,
        std::vector<LiteRtTensorBuffer>& buffers,
        bool gpu_input,
        bool* ok) {
    const int layer = parse_kv_cache_layer_index(name, prefix, (int)buffers.size());
    if (layer < 0 || layer >= (int)buffers.size()) {
        tinylog::logger().error("invalid Gemma4 KV cache tensor name",
            {{"tensor", name}, {"prefix", std::string(prefix)}});
        if (ok) *ok = false;
        return nullptr;
    }
    return gpu_input ? nullptr : buffers[layer];
}

static inline bool rebuild_prefill_arrays(State* s, PrefillSig& ps) {
    const auto& sig = ps.sig;
    const bool gpu = uses_gpu_single_buffer_kv_cache(s);
    bool ok = true;
    for (size_t i = 0; i < sig.num_inputs; i++) {
        const auto& name = sig.input_names[i];
        if (name == "embeddings") ps.input_bufs[i] = ps.embeddings_buf;
        else if (name == "per_layer_embeddings") ps.input_bufs[i] = ps.ple_buf;
        else if (name == "input_pos") ps.input_bufs[i] = ps.pos_buf;
        else if (name == "mask") ps.input_bufs[i] = ps.mask_buf;
        else if (name == "param_tensor") ps.input_bufs[i] = ps.param_buf;
        else if (name.rfind("kv_cache_k_", 0) == 0)
            ps.input_bufs[i] = kv_cache_buffer_for_name(name, "kv_cache_k_", s->kv_k, gpu, &ok);
        else if (name.rfind("kv_cache_v_", 0) == 0)
            ps.input_bufs[i] = kv_cache_buffer_for_name(name, "kv_cache_v_", s->kv_v, gpu, &ok);
    }
    for (size_t i = 0; i < sig.num_outputs; i++) {
        const auto& name = sig.output_names[i];
        if (name == "activations") ps.output_bufs[i] = ps.activations_buf;
        else if (name.rfind("kv_cache_k_", 0) == 0)
            ps.output_bufs[i] = kv_cache_buffer_for_name(name, "kv_cache_k_", s->kv_k, false, &ok);
        else if (name.rfind("kv_cache_v_", 0) == 0)
            ps.output_bufs[i] = kv_cache_buffer_for_name(name, "kv_cache_v_", s->kv_v, false, &ok);
    }
    return ok;
}

static inline bool rebuild_decode_arrays(State* s) {
    const auto& sig = s->decode_sig;
    const bool gpu = uses_gpu_single_buffer_kv_cache(s);
    bool ok = true;
    for (size_t i = 0; i < sig.num_inputs; i++) {
        const auto& name = sig.input_names[i];
        if (name == "embeddings") s->decode_input_bufs[i] = s->decode_embeddings_buf;
        else if (name == "per_layer_embeddings") s->decode_input_bufs[i] = s->decode_ple_buf;
        else if (name == "input_pos") s->decode_input_bufs[i] = s->decode_pos_buf;
        else if (name == "mask") s->decode_input_bufs[i] = s->decode_mask_buf;
        else if (name == "param_tensor") s->decode_input_bufs[i] = s->decode_param_buf;
        else if (name.rfind("kv_cache_k_", 0) == 0)
            s->decode_input_bufs[i] = kv_cache_buffer_for_name(name, "kv_cache_k_", s->kv_k, gpu, &ok);
        else if (name.rfind("kv_cache_v_", 0) == 0)
            s->decode_input_bufs[i] = kv_cache_buffer_for_name(name, "kv_cache_v_", s->kv_v, gpu, &ok);
    }
    for (size_t i = 0; i < sig.num_outputs; i++) {
        const auto& name = sig.output_names[i];
        if (name == "logits") s->decode_output_bufs[i] = s->decode_logits_buf;
        else if (name == "activations") s->decode_output_bufs[i] = s->decode_activations_buf;
        else if (name.rfind("kv_cache_k_", 0) == 0)
            s->decode_output_bufs[i] = kv_cache_buffer_for_name(name, "kv_cache_k_", s->kv_k, false, &ok);
        else if (name.rfind("kv_cache_v_", 0) == 0)
            s->decode_output_bufs[i] = kv_cache_buffer_for_name(name, "kv_cache_v_", s->kv_v, false, &ok);
    }
    return ok;
}

static inline bool rebuild_verify_arrays(State* s) {
    auto& vs = s->verify_sig;
    const auto& sig = vs.sig;
    const bool gpu = uses_gpu_single_buffer_kv_cache(s);
    bool ok = true;
    for (size_t i = 0; i < sig.num_inputs; i++) {
        const auto& name = sig.input_names[i];
        if (name == "embeddings") vs.input_bufs[i] = vs.embeddings_buf;
        else if (name == "per_layer_embeddings") vs.input_bufs[i] = vs.ple_buf;
        else if (name == "input_pos") vs.input_bufs[i] = vs.pos_buf;
        else if (name == "mask") vs.input_bufs[i] = vs.mask_buf;
        else if (name == "param_tensor") vs.input_bufs[i] = vs.param_buf;
        else if (name.rfind("kv_cache_k_", 0) == 0)
            vs.input_bufs[i] = kv_cache_buffer_for_name(name, "kv_cache_k_", s->kv_k, gpu, &ok);
        else if (name.rfind("kv_cache_v_", 0) == 0)
            vs.input_bufs[i] = kv_cache_buffer_for_name(name, "kv_cache_v_", s->kv_v, gpu, &ok);
    }
    for (size_t i = 0; i < sig.num_outputs; i++) {
        const auto& name = sig.output_names[i];
        if (name == "logits") vs.output_bufs[i] = vs.logits_buf;
        else if (name == "activations") vs.output_bufs[i] = vs.activations_buf;
        else if (name.rfind("kv_cache_k_", 0) == 0) {
            const int layer = parse_kv_cache_layer_index(
                name, "kv_cache_k_", (int)s->kv_k.size());
            if (layer < 0 || layer >= (int)s->kv_k.size()) ok = false;
            else vs.output_bufs[i] = vs.kv_k[layer] ? vs.kv_k[layer] : s->kv_k[layer];
        } else if (name.rfind("kv_cache_v_", 0) == 0) {
            const int layer = parse_kv_cache_layer_index(
                name, "kv_cache_v_", (int)s->kv_v.size());
            if (layer < 0 || layer >= (int)s->kv_v.size()) ok = false;
            else vs.output_bufs[i] = vs.kv_v[layer] ? vs.kv_v[layer] : s->kv_v[layer];
        }
    }
    return ok;
}

static inline bool duplicate_run_buffers(
        const SignatureInfo& sig,
        const std::vector<LiteRtTensorBuffer>& src,
        bool inputs,
        ScopedTensorBufferDuplicates& owned,
        std::vector<LiteRtTensorBuffer>& dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        if (!src[i]) {
            dst[i] = nullptr;
            continue;
        }
        const std::string& name = inputs ? sig.input_names[i] : sig.output_names[i];
        dst[i] = owned.duplicate_or_original(src[i], name.c_str());
    }
    return true;
}

static inline bool rebuild_mtp_arrays(State* s) {
    const auto& sig = s->mtp_sig;
    bool ok = true;
    for (size_t i = 0; i < sig.num_inputs; i++) {
        const auto& name = sig.input_names[i];
        if (name == "input_pos") s->mtp_input_bufs[i] = s->mtp_pos_buf;
        else if (name == "activations") s->mtp_input_bufs[i] = s->mtp_activations_buf;
        else if (name == "mask") s->mtp_input_bufs[i] = s->mtp_mask_buf;
        else if (name == "param_tensor") s->mtp_input_bufs[i] = s->mtp_param_buf;
        else if (name.rfind("kv_cache_k_", 0) == 0)
            s->mtp_input_bufs[i] = kv_cache_buffer_for_name(name, "kv_cache_k_", s->kv_k, false, &ok);
        else if (name.rfind("kv_cache_v_", 0) == 0)
            s->mtp_input_bufs[i] = kv_cache_buffer_for_name(name, "kv_cache_v_", s->kv_v, false, &ok);
    }
    for (size_t i = 0; i < sig.num_outputs; i++) {
        const auto& name = sig.output_names[i];
        if (name == "logits") s->mtp_output_bufs[i] = s->mtp_logits_buf;
        else if (name == "projected_activations")
            s->mtp_output_bufs[i] = s->mtp_projected_activations_buf;
    }
    return ok;
}

// MARK: - Text signature discovery

static inline int signature_first_dim(
        const SignatureInfo& sig, const char* name, bool input) {
    const int idx = input ? sig.input_index_of(name) : sig.output_index_of(name);
    if (idx < 0) return 0;

    LiteRtRankedTensorType tensor_type = {};
    const bool ok = input
        ? get_input_tensor_type(sig.sig, (LiteRtParamIndex)idx, &tensor_type)
        : get_output_tensor_type(sig.sig, (LiteRtParamIndex)idx, &tensor_type);
    if (!ok || tensor_type.layout.rank < 1)
        return 0;
    return tensor_type.layout.dimensions[0];
}

static inline int signature_tensor_element_count(
        const SignatureInfo& sig, const char* name, bool input) {
    const int idx = input ? sig.input_index_of(name) : sig.output_index_of(name);
    if (idx < 0) return 0;

    LiteRtRankedTensorType tensor_type = {};
    const bool ok = input
        ? get_input_tensor_type(sig.sig, (LiteRtParamIndex)idx, &tensor_type)
        : get_output_tensor_type(sig.sig, (LiteRtParamIndex)idx, &tensor_type);
    if (!ok || tensor_type.layout.rank < 1)
        return 0;
    int64_t elements = 1;
    for (int i = 0; i < tensor_type.layout.rank; i++) {
        const int dim = tensor_type.layout.dimensions[i];
        if (dim <= 0) return 0;
        elements *= dim;
        if (elements > std::numeric_limits<int>::max()) return 0;
    }
    return (int)elements;
}

static inline int signature_last_dim(
        const SignatureInfo& sig, const char* name, bool input) {
    const int idx = input ? sig.input_index_of(name) : sig.output_index_of(name);
    if (idx < 0) return 0;

    LiteRtRankedTensorType tensor_type = {};
    const bool ok = input
        ? get_input_tensor_type(sig.sig, (LiteRtParamIndex)idx, &tensor_type)
        : get_output_tensor_type(sig.sig, (LiteRtParamIndex)idx, &tensor_type);
    if (!ok || tensor_type.layout.rank < 1)
        return 0;
    return tensor_type.layout.dimensions[tensor_type.layout.rank - 1];
}

static inline int max_kv_cache_layer_from_names(
        const std::vector<std::string>& names) {
    int max_layer = -1;
    for (const auto& name : names) {
        int layer = -1;
        if (name.rfind("kv_cache_k_", 0) == 0)
            layer = parse_kv_cache_layer_index(name, "kv_cache_k_", 10000);
        else if (name.rfind("kv_cache_v_", 0) == 0)
            layer = parse_kv_cache_layer_index(name, "kv_cache_v_", 10000);
        if (layer > max_layer) max_layer = layer;
    }
    return max_layer;
}

static inline void discover_model_shape(State* s) {
    Gemma4ModelShape shape;

    int max_layer = -1;
    auto scan_sig_layers = [&](const SignatureInfo& sig) {
        max_layer = std::max(max_layer, max_kv_cache_layer_from_names(sig.input_names));
        max_layer = std::max(max_layer, max_kv_cache_layer_from_names(sig.output_names));
    };
    scan_sig_layers(s->decode_sig);
    for (int i = 0; i < NUM_PREFILL_SIZES; i++) {
        if (s->prefill_sigs[i].seq_len > 0)
            scan_sig_layers(s->prefill_sigs[i].sig);
    }
    if (s->has_verify_sig)
        scan_sig_layers(s->verify_sig.sig);
    if (max_layer >= 0)
        shape.num_kv_layers = max_layer + 1;

    const int emb_out = signature_tensor_element_count(
        s->emb_sig, "embeddings", /*input=*/false);
    const int emb_fallback = signature_tensor_element_count(
        s->emb_sig, "output", /*input=*/false);
    const int decoder_emb = signature_last_dim(
        s->decode_sig, "embeddings", /*input=*/true);
    if (decoder_emb > 0)
        shape.embedding_dim = decoder_emb;
    else if (emb_out > 0)
        shape.embedding_dim = emb_out;
    else if (emb_fallback > 0)
        shape.embedding_dim = emb_fallback;

    int ple = signature_tensor_element_count(
        s->decode_sig, "per_layer_embeddings", /*input=*/true);
    if (ple <= 0)
        ple = signature_tensor_element_count(
            s->ple_sig, "per_layer_embeddings", /*input=*/false);
    if (ple <= 0)
        ple = signature_tensor_element_count(s->ple_sig, "output", /*input=*/false);
    if (ple > 0)
        shape.ple_floats = ple;

    int logits = signature_last_dim(s->decode_sig, "logits", /*input=*/false);
    if (logits > 0)
        shape.vocab_size = logits;

    shape.mtp_drafter_input_dim = shape.embedding_dim * 2;
    if (s->mtp_enabled) {
        int mtp_input = signature_tensor_element_count(
            s->mtp_sig, "activations", /*input=*/true);
        if (mtp_input > 0)
            shape.mtp_drafter_input_dim = mtp_input;
    }

    s->shape = shape;
    s->kv_k.assign(shape.num_kv_layers, nullptr);
    s->kv_v.assign(shape.num_kv_layers, nullptr);
    s->verify_sig.kv_k.assign(shape.num_kv_layers, nullptr);
    s->verify_sig.kv_v.assign(shape.num_kv_layers, nullptr);

    tinylog::logger().info("Gemma4 model shape discovered",
        {{"kv_layers", (int64_t)s->shape.num_kv_layers},
         {"embedding_dim", (int64_t)s->shape.embedding_dim},
         {"ple_floats", (int64_t)s->shape.ple_floats},
         {"mtp_input_dim", (int64_t)s->shape.mtp_drafter_input_dim},
         {"vocab_size", (int64_t)s->shape.vocab_size}});
}

static inline int infer_verify_token_count(const SignatureInfo& sig) {
    // LiteRT-LM derives the MTP draft count from verify.input_pos shape:
    // verify rows = current token + draft tokens.
    const char* input_candidates[] = {
        "input_pos",
        "embeddings",
        "per_layer_embeddings",
    };
    for (const char* name : input_candidates) {
        int dim = signature_first_dim(sig, name, /*input=*/true);
        if (dim > 1) return dim;
    }
    int dim = signature_first_dim(sig, "logits", /*input=*/false);
    return dim > 1 ? dim : 0;
}

static inline bool discover_decoder_signatures(State* s) {
    LiteRtParamIndex num_sigs = 0;
    LITERT(LiteRtGetNumModelSignatures)(s->decoder_model, &num_sigs);

    bool found_decode = false;
    bool found_verify = false;
    int prefill_found = 0;

    for (LiteRtParamIndex i = 0; i < num_sigs; i++) {
        LiteRtSignature sig = nullptr;
        if (LITERT(LiteRtGetModelSignature)(s->decoder_model, i, &sig) != kLiteRtStatusOk)
            continue;

        const char* key = nullptr;
        LITERT(LiteRtGetSignatureKey)(sig, &key);
        if (!key) continue;

        std::string key_str(key);

        if (key_str == "decode") {
            if (!discover_signature(s->decoder_model, i, s->decode_sig)) return false;
            found_decode = true;
        } else if (key_str == "verify") {
            if (!discover_signature(s->decoder_model, i, s->verify_sig.sig)) return false;
            found_verify = true;
        } else {
            // Try to match prefill_NNN signatures
            for (int p = 0; p < NUM_PREFILL_SIZES; p++) {
                char expected[32];
                snprintf(expected, sizeof(expected), "prefill_%d", PREFILL_SIZES[p]);
                if (key_str == expected) {
                    if (!discover_signature(s->decoder_model, i, s->prefill_sigs[p].sig))
                        return false;
                    s->prefill_sigs[p].seq_len = PREFILL_SIZES[p];
                    prefill_found++;
                    break;
                }
            }
        }
    }

    if (!found_decode) {
        tinylog::logger().error("decode signature not found");
        return false;
    }
    if (prefill_found == 0) {
        tinylog::logger().error("no prefill signatures found");
        return false;
    }

    s->mtp_verify_tokens = 0;
    s->mtp_draft_tokens = 0;
    if (found_verify) {
        s->mtp_verify_tokens = infer_verify_token_count(s->verify_sig.sig);
        s->mtp_draft_tokens = s->mtp_verify_tokens - 1;
        if (s->mtp_draft_tokens <= 0) {
            tinylog::logger().warn("verify signature has invalid MTP token shape",
                {{"verify_tokens", (int64_t)s->mtp_verify_tokens}});
            found_verify = false;
            s->mtp_verify_tokens = 0;
            s->mtp_draft_tokens = 0;
        }
    }

    s->has_verify_sig = found_verify;
    tinylog::logger().debug("discovered decoder signatures",
        {{"prefill_count", (int64_t)prefill_found},
         {"verify", found_verify ? 1 : 0},
         {"mtp_verify_tokens", (int64_t)s->mtp_verify_tokens},
         {"mtp_draft_tokens", (int64_t)s->mtp_draft_tokens}});
    return true;
}

// MARK: - Text buffer allocation

static inline bool allocate_embedder_buffers(State* s) {
    const auto& sig = s->emb_sig;

    int tok_idx = sig.input_index_of("tokens");
    if (tok_idx < 0) tok_idx = sig.input_index_of("input_ids");
    if (tok_idx < 0) {
        // Fall back to first input
        if (sig.num_inputs > 0) tok_idx = 0;
        else return false;
    }
    s->emb_tokens_buf = alloc_managed_input(s->env, s->embedder_cm, sig, tok_idx, "emb/tokens");
    if (!s->emb_tokens_buf) return false;

    int out_idx = sig.output_index_of("embeddings");
    if (out_idx < 0) out_idx = sig.output_index_of("output");
    if (out_idx < 0 && sig.num_outputs > 0) out_idx = 0;
    if (out_idx < 0) return false;
    s->emb_output_buf = alloc_managed_output(s->env, s->embedder_cm, sig, out_idx, "emb/output");
    if (!s->emb_output_buf) return false;

    s->emb_input_bufs.resize(sig.num_inputs);
    s->emb_output_bufs.resize(sig.num_outputs);
    s->emb_input_bufs[tok_idx] = s->emb_tokens_buf;
    s->emb_output_bufs[out_idx] = s->emb_output_buf;

    return true;
}

static inline bool allocate_ple_buffers(State* s) {
    const auto& sig = s->ple_sig;

    int tok_idx = sig.input_index_of("tokens");
    if (tok_idx < 0) tok_idx = sig.input_index_of("input_ids");
    if (tok_idx < 0 && sig.num_inputs > 0) tok_idx = 0;
    if (tok_idx < 0) return false;
    s->ple_tokens_buf = alloc_managed_input(s->env, s->ple_cm, sig, tok_idx, "ple/tokens");
    if (!s->ple_tokens_buf) return false;

    int out_idx = sig.output_index_of("per_layer_embeddings");
    if (out_idx < 0) out_idx = sig.output_index_of("output");
    if (out_idx < 0 && sig.num_outputs > 0) out_idx = 0;
    if (out_idx < 0) return false;
    s->ple_output_buf = alloc_managed_output(s->env, s->ple_cm, sig, out_idx, "ple/output");
    if (!s->ple_output_buf) return false;

    s->ple_input_bufs.resize(sig.num_inputs);
    s->ple_output_bufs.resize(sig.num_outputs);
    s->ple_input_bufs[tok_idx] = s->ple_tokens_buf;
    s->ple_output_bufs[out_idx] = s->ple_output_buf;

    return true;
}

static inline bool allocate_prefill_sig_buffers(State* s, PrefillSig& ps) {
    if (ps.seq_len == 0) return true;  // not discovered

    const auto& sig = ps.sig;
    LiteRtCompiledModel cm = s->decoder_cm;

    // Embeddings input
    int idx = sig.input_index_of("embeddings");
    if (idx >= 0) {
        ps.embeddings_buf = alloc_managed_input(s->env, cm, sig, idx, "prefill/embeddings");
        if (!ps.embeddings_buf) return false;
    }

    // Per-layer embeddings input
    idx = sig.input_index_of("per_layer_embeddings");
    if (idx >= 0) {
        ps.ple_buf = alloc_managed_input(s->env, cm, sig, idx, "prefill/ple");
        if (!ps.ple_buf) return false;
    }

    // Position input
    idx = sig.input_index_of("input_pos");
    if (idx >= 0) {
        ps.pos_buf = alloc_managed_input(s->env, cm, sig, idx, "prefill/pos");
        if (!ps.pos_buf) return false;
    }

    // Mask input
    idx = sig.input_index_of("mask");
    if (idx >= 0) {
        ps.mask_buf = alloc_managed_input(s->env, cm, sig, idx, "prefill/mask");
        if (!ps.mask_buf) return false;
    }

    // param_tensor input
    idx = sig.input_index_of("param_tensor");
    if (idx >= 0) {
        ps.param_buf = alloc_managed_input(s->env, cm, sig, idx, "prefill/param");
        if (!ps.param_buf) return false;
    }

    // activations output (MTP drafter — unused, but buffer must exist)
    int out_idx = sig.output_index_of("activations");
    if (out_idx >= 0) {
        ps.activations_buf = alloc_managed_output(s->env, cm, sig, out_idx, "prefill/activations");
        if (!ps.activations_buf) return false;
    }

    ps.input_bufs.resize(sig.num_inputs);
    ps.output_bufs.resize(sig.num_outputs);

    return true;
}

static inline bool allocate_verify_buffers(State* s) {
    if (!s->has_verify_sig) return true;
    auto& vs = s->verify_sig;
    const auto& sig = vs.sig;
    LiteRtCompiledModel cm = s->decoder_cm;

    int idx = sig.input_index_of("embeddings");
    if (idx >= 0) {
        vs.embeddings_buf = alloc_managed_input(s->env, cm, sig, idx, "verify/embeddings");
        if (!vs.embeddings_buf) return false;
    }
    idx = sig.input_index_of("per_layer_embeddings");
    if (idx >= 0) {
        vs.ple_buf = alloc_managed_input(s->env, cm, sig, idx, "verify/ple");
        if (!vs.ple_buf) return false;
    }
    idx = sig.input_index_of("input_pos");
    if (idx >= 0) {
        vs.pos_buf = alloc_managed_input(s->env, cm, sig, idx, "verify/pos");
        if (!vs.pos_buf) return false;
    }
    idx = sig.input_index_of("mask");
    if (idx >= 0) {
        LiteRtRankedTensorType mask_type = {};
        if (get_input_tensor_type(sig.sig, idx, &mask_type))
            s->verify_mask_element_type = mask_type.element_type;
        vs.mask_buf = alloc_managed_input(s->env, cm, sig, idx, "verify/mask");
        if (!vs.mask_buf) return false;
    }
    idx = sig.input_index_of("param_tensor");
    if (idx >= 0) {
        vs.param_buf = alloc_managed_input(s->env, cm, sig, idx, "verify/param");
        if (!vs.param_buf) return false;
    }
    idx = sig.output_index_of("logits");
    if (idx < 0) return false;
    vs.logits_buf = alloc_managed_output(s->env, cm, sig, idx, "verify/logits");
    if (!vs.logits_buf) return false;
    idx = sig.output_index_of("activations");
    if (idx >= 0) {
        vs.activations_buf = alloc_managed_output(s->env, cm, sig, idx, "verify/activations");
        if (!vs.activations_buf) return false;
    }
    vs.input_bufs.resize(sig.num_inputs);
    vs.output_bufs.resize(sig.num_outputs);
    return true;
}

static inline bool allocate_mtp_buffers(State* s) {
    if (!s->has_mtp_drafter || !s->mtp_enabled) return true;
    const auto& sig = s->mtp_sig;
    LiteRtCompiledModel cm = s->mtp_cm;

    int idx = sig.input_index_of("input_pos");
    if (idx >= 0) {
        s->mtp_pos_buf = alloc_managed_input(s->env, cm, sig, idx, "mtp/input_pos");
        if (!s->mtp_pos_buf) return false;
    }
    idx = sig.input_index_of("activations");
    if (idx >= 0) {
        s->mtp_activations_buf = alloc_managed_input(s->env, cm, sig, idx, "mtp/activations");
        if (!s->mtp_activations_buf) return false;
    }
    idx = sig.input_index_of("param_tensor");
    if (idx >= 0) {
        s->mtp_param_buf = alloc_managed_input(s->env, cm, sig, idx, "mtp/param");
        if (!s->mtp_param_buf) return false;
    }
    idx = sig.input_index_of("mask");
    if (idx >= 0) {
        LiteRtRankedTensorType mask_type = {};
        if (get_input_tensor_type(sig.sig, idx, &mask_type))
            s->mtp_mask_element_type = mask_type.element_type;
        s->mtp_mask_buf = alloc_managed_input(s->env, cm, sig, idx, "mtp/mask");
        if (!s->mtp_mask_buf) return false;
    }
    idx = sig.output_index_of("logits");
    if (idx < 0) return false;
    s->mtp_logits_buf = alloc_managed_output(s->env, cm, sig, idx, "mtp/logits");
    if (!s->mtp_logits_buf) return false;
    idx = sig.output_index_of("projected_activations");
    if (idx < 0) return false;
    s->mtp_projected_activations_buf =
        alloc_managed_output(s->env, cm, sig, idx, "mtp/projected_activations");
    if (!s->mtp_projected_activations_buf) return false;

    s->mtp_input_bufs.resize(sig.num_inputs);
    s->mtp_output_bufs.resize(sig.num_outputs);
    return true;
}

static inline bool allocate_buffers(State* s) {
    LiteRtCompiledModel cm = s->decoder_cm;
    const auto& ds = s->decode_sig;

    // KV cache buffers.
    // GPU mode: the Metal dispatch delegate maintains its own internal KV
    // buffers per signature (not accessible externally). We allocate output
    // KV buffers as required placeholders for Run(), but the actual KV state
    // lives in the delegate's internal Metal buffer. Cross-signature KV
    // sharing doesn't work, so GPU prefill uses the decode-only path (see
    // Engine::prefill). We pass nullptr for KV inputs on GPU.
    // CPU mode: allocate from prefill INPUT requirements and bind the same
    // buffer to both input and output slots.
    if (s->execution_mode == EXEC_MODE_GPU) {
        // GPU: allocate from prefill output requirements (type 34,
        // MetalBufferPacked). These are placeholders required by Run().
        const PrefillSig* kv_prefill = nullptr;
        for (int i = 0; i < NUM_PREFILL_SIZES; i++) {
            if (s->prefill_sigs[i].seq_len > 0) {
                kv_prefill = &s->prefill_sigs[i];
                break;
            }
        }
        if (!kv_prefill) return false;

        for (int layer = 0; layer < s->shape.num_kv_layers; layer++) {
            char name_k[32], name_v[32];
            snprintf(name_k, sizeof(name_k), "kv_cache_k_%d", layer);
            snprintf(name_v, sizeof(name_v), "kv_cache_v_%d", layer);
            int out_idx_k = kv_prefill->sig.output_index_of(name_k);
            int out_idx_v = kv_prefill->sig.output_index_of(name_v);
            if (out_idx_k < 0 || out_idx_v < 0) {
                tinylog::logger().error("KV cache output not found in prefill sig",
                    {{"k", std::string(name_k)}, {"v", std::string(name_v)}});
                return false;
            }
            s->kv_k[layer] = alloc_managed_output(s->env, cm, kv_prefill->sig,
                                                   out_idx_k, name_k);
            s->kv_v[layer] = alloc_managed_output(s->env, cm, kv_prefill->sig,
                                                   out_idx_v, name_v);
            if (!s->kv_k[layer] || !s->kv_v[layer]) return false;
        }
    } else {
        const PrefillSig* kv_sig_source = nullptr;
        for (int i = 0; i < NUM_PREFILL_SIZES; i++) {
            if (s->prefill_sigs[i].seq_len > 0) {
                kv_sig_source = &s->prefill_sigs[i];
                break;
            }
        }
        if (!kv_sig_source) return false;

        for (int layer = 0; layer < s->shape.num_kv_layers; layer++) {
            char name_k[32], name_v[32];
            snprintf(name_k, sizeof(name_k), "kv_cache_k_%d", layer);
            snprintf(name_v, sizeof(name_v), "kv_cache_v_%d", layer);
            int in_idx_k = kv_sig_source->sig.input_index_of(name_k);
            int in_idx_v = kv_sig_source->sig.input_index_of(name_v);
            if (in_idx_k < 0 || in_idx_v < 0) {
                tinylog::logger().error("KV cache input not found in prefill sig",
                    {{"k", std::string(name_k)}, {"v", std::string(name_v)}});
                return false;
            }
            s->kv_k[layer] = alloc_managed_input(s->env, cm, kv_sig_source->sig,
                                                  in_idx_k, name_k);
            s->kv_v[layer] = alloc_managed_input(s->env, cm, kv_sig_source->sig,
                                                  in_idx_v, name_v);
            if (!s->kv_k[layer] || !s->kv_v[layer]) return false;
        }
    }

    // Allocate per-prefill-size buffers
    for (int i = 0; i < NUM_PREFILL_SIZES; i++) {
        if (!allocate_prefill_sig_buffers(s, s->prefill_sigs[i])) return false;
    }
    if (!allocate_verify_buffers(s)) return false;

    // Decode buffers
    {
        int idx = ds.input_index_of("embeddings");
        if (idx >= 0) {
            s->decode_embeddings_buf = alloc_managed_input(s->env, cm, ds, idx,
                                                           "decode/embeddings");
            if (!s->decode_embeddings_buf) return false;
        }
        idx = ds.input_index_of("per_layer_embeddings");
        if (idx >= 0) {
            s->decode_ple_buf = alloc_managed_input(s->env, cm, ds, idx, "decode/ple");
            if (!s->decode_ple_buf) return false;
        }
        idx = ds.input_index_of("input_pos");
        if (idx >= 0) {
            s->decode_pos_buf = alloc_managed_input(s->env, cm, ds, idx, "decode/pos");
            if (!s->decode_pos_buf) return false;
        }
        idx = ds.input_index_of("mask");
        if (idx >= 0) {
            s->decode_mask_buf = alloc_managed_input(s->env, cm, ds, idx, "decode/mask");
            if (!s->decode_mask_buf) return false;
        }
        idx = ds.input_index_of("param_tensor");
        if (idx >= 0) {
            s->decode_param_buf = alloc_managed_input(s->env, cm, ds, idx, "decode/param");
            if (!s->decode_param_buf) return false;
        }
        idx = ds.output_index_of("logits");
        if (idx < 0) return false;
        s->decode_logits_buf = alloc_managed_output(s->env, cm, ds, idx, "decode/logits");
        if (!s->decode_logits_buf) return false;
        idx = ds.output_index_of("activations");
        if (idx >= 0) {
            s->decode_activations_buf = alloc_managed_output(s->env, cm, ds, idx, "decode/activations");
            if (!s->decode_activations_buf) return false;
        }
    }

    s->decode_input_bufs.resize(ds.num_inputs);
    s->decode_output_bufs.resize(ds.num_outputs);

    // Embedder + PLE buffers
    if (!allocate_embedder_buffers(s)) return false;
    if (!allocate_ple_buffers(s)) return false;
    if (!allocate_mtp_buffers(s)) return false;

    return true;
}

// MARK: - Text core inference

static inline void reset_kv_cache(State* s) {
    s->current_pos = 0;
}

// Pick the best prefill signature for a given chunk size.
static inline PrefillSig* pick_prefill_sig(State* s, int num_tokens) {
    for (int i = 0; i < NUM_PREFILL_SIZES; i++) {
        if (s->prefill_sigs[i].seq_len > 0 &&
            s->prefill_sigs[i].seq_len >= num_tokens) {
            return &s->prefill_sigs[i];
        }
    }
    // Fall back to the largest available
    for (int i = NUM_PREFILL_SIZES - 1; i >= 0; i--) {
        if (s->prefill_sigs[i].seq_len > 0) return &s->prefill_sigs[i];
    }
    return nullptr;
}

static inline bool run_prefill_embedding_chunk(
        State* s,
        const float* embeddings,
        const float* per_layer_embeddings,
        int num_tokens,
        int pos_offset) {
    PrefillSig* ps = pick_prefill_sig(s, num_tokens);
    if (!ps) return false;
    if (pos_offset < 0 || num_tokens < 0 ||
        pos_offset + num_tokens > s->kv_cache_max_len) {
        tinylog::logger().error("Gemma4 prefill position exceeds KV cache",
            {{"pos", (int64_t)pos_offset},
             {"tokens", (int64_t)num_tokens},
             {"kv_cache_max_len", (int64_t)s->kv_cache_max_len}});
        return false;
    }
    const int SEQ = ps->seq_len;
    tinylog::logger().debug("prefill_embedding_chunk",
        {{"tokens", (int64_t)num_tokens}, {"seq", (int64_t)SEQ}, {"pos", (int64_t)pos_offset}});

    {
        std::vector<float> emb_buf(SEQ * s->shape.embedding_dim, 0.0f);
        std::vector<float> ple_buf(SEQ * s->shape.ple_floats, 0.0f);
        memcpy(emb_buf.data(), embeddings, num_tokens * s->shape.embedding_dim * sizeof(float));
        memcpy(ple_buf.data(), per_layer_embeddings, num_tokens * s->shape.ple_floats * sizeof(float));

        tinylog::logger().debug("writing embeddings",
            {{"seq", (int64_t)SEQ}, {"dim", (int64_t)s->shape.embedding_dim}});
        write_float_buf(ps->embeddings_buf, emb_buf.data(), SEQ * s->shape.embedding_dim);
        write_float_buf(ps->ple_buf, ple_buf.data(), SEQ * s->shape.ple_floats);
    }

    // Positions
    {
        std::vector<int32_t> input_pos(SEQ, 0);
        for (int i = 0; i < num_tokens; i++) input_pos[i] = pos_offset + i;
        write_int_buf(ps->pos_buf, input_pos.data(), SEQ);
    }

    tinylog::logger().debug("writing mask");
    // Mask
    write_prefill_mask(ps->mask_buf, s->mask_element_type, s->masked_value,
                       SEQ, s->kv_cache_max_len, num_tokens, pos_offset);

    // param_tensor
    if (ps->param_buf)
        write_param_tensor(ps->param_buf, pos_offset, num_tokens);

    tinylog::logger().debug("running prefill RunCompiledModel");
    if (!rebuild_prefill_arrays(s, *ps))
        return false;
    ScopedTensorBufferDuplicates duplicates;
    std::vector<LiteRtTensorBuffer> input_bufs;
    std::vector<LiteRtTensorBuffer> output_bufs;
    if (!duplicate_run_buffers(ps->sig, ps->input_bufs, true, duplicates, input_bufs) ||
        !duplicate_run_buffers(ps->sig, ps->output_bufs, false, duplicates, output_bufs))
        return false;
    clear_buffer_events(output_bufs.data(), output_bufs.size());

    if (!litert_check(LITERT(LiteRtRunCompiledModel)(
                          s->decoder_cm, ps->sig.sig_index,
                          input_bufs.size(), input_bufs.data(),
                          output_bufs.size(), output_bufs.data()),
                      "RunCompiledModel(gemma4/prefill)"))
        return false;

    s->current_pos = pos_offset + num_tokens;
    return true;
}

static inline bool run_prefill_chunk(State* s, const int32_t* token_ids,
                                     int num_tokens, int pos_offset) {
    std::vector<float> emb_buf(num_tokens * s->shape.embedding_dim, 0.0f);
    std::vector<float> ple_buf(num_tokens * s->shape.ple_floats, 0.0f);
    std::vector<float> single_emb(s->shape.embedding_dim);
    std::vector<float> single_ple(s->shape.ple_floats);

    for (int i = 0; i < num_tokens; i++) {
        if (!lookup_embedding(s, token_ids[i], single_emb.data())) return false;
        memcpy(&emb_buf[i * s->shape.embedding_dim], single_emb.data(),
               s->shape.embedding_dim * sizeof(float));

        if (!lookup_ple(s, token_ids[i], single_ple.data())) return false;
        memcpy(&ple_buf[i * s->shape.ple_floats], single_ple.data(),
               s->shape.ple_floats * sizeof(float));
    }

    return run_prefill_embedding_chunk(
        s, emb_buf.data(), ple_buf.data(), num_tokens, pos_offset);
}

static inline bool run_decode_embedding_step(
        State* s,
        const float* embedding,
        const float* per_layer_embedding,
        float* logits_out,
        float* activations_out = nullptr,
        MtpStepTimings* timings = nullptr) {
    write_float_buf(s->decode_embeddings_buf, embedding, s->shape.embedding_dim);
    write_float_buf(s->decode_ple_buf, per_layer_embedding, s->shape.ple_floats);

    // Position
    int32_t pos = s->current_pos;
    if (pos < 0 || pos >= s->kv_cache_max_len) {
        tinylog::logger().error("Gemma4 decode position exceeds KV cache",
            {{"pos", (int64_t)pos},
             {"kv_cache_max_len", (int64_t)s->kv_cache_max_len}});
        return false;
    }
    write_int_buf(s->decode_pos_buf, &pos, 1);

    // Mask
    write_decode_mask(s->decode_mask_buf, s->mask_element_type, s->masked_value,
                      s->kv_cache_max_len, pos);

    // param_tensor
    if (s->decode_param_buf)
        write_param_tensor(s->decode_param_buf, pos, 1);

    if (!rebuild_decode_arrays(s))
        return false;
    ScopedTensorBufferDuplicates duplicates;
    std::vector<LiteRtTensorBuffer> input_bufs;
    std::vector<LiteRtTensorBuffer> output_bufs;
    if (!duplicate_run_buffers(
            s->decode_sig, s->decode_input_bufs, true, duplicates, input_bufs) ||
        !duplicate_run_buffers(
            s->decode_sig, s->decode_output_bufs, false, duplicates, output_bufs))
        return false;
    clear_buffer_events(output_bufs.data(), output_bufs.size());

    auto model_start = std::chrono::steady_clock::now();
    bool model_ok = litert_check(LITERT(LiteRtRunCompiledModel)(
                          s->decoder_cm, s->decode_sig.sig_index,
                          input_bufs.size(), input_bufs.data(),
                          output_bufs.size(), output_bufs.data()),
                      "RunCompiledModel(gemma4/decode)");
    if (timings)
        timings->model_us += elapsed_us(model_start, std::chrono::steady_clock::now());
    if (!model_ok)
        return false;

    if (logits_out) {
        LiteRtTensorBuffer logits_buf = s->decode_logits_buf;
        int logits_idx = s->decode_sig.output_index_of("logits");
        if (logits_idx >= 0 && logits_idx < (int)output_bufs.size() &&
            output_bufs[logits_idx])
            logits_buf = output_bufs[logits_idx];
        auto read_start = std::chrono::steady_clock::now();
        bool read_ok = read_float_buf(logits_buf, logits_out, s->shape.vocab_size);
        if (timings) {
            timings->logits_read_us += elapsed_us(
                read_start, std::chrono::steady_clock::now());
            timings->logits_rows_read += 1;
        }
        if (!read_ok)
            return false;
    }
    if (activations_out && s->decode_activations_buf) {
        LiteRtTensorBuffer activations_buf = s->decode_activations_buf;
        int activations_idx = s->decode_sig.output_index_of("activations");
        if (activations_idx >= 0 && activations_idx < (int)output_bufs.size() &&
            output_bufs[activations_idx])
            activations_buf = output_bufs[activations_idx];
        auto read_start = std::chrono::steady_clock::now();
        bool read_ok = read_float_buf(activations_buf, activations_out,
                                      s->shape.embedding_dim);
        if (timings) {
            timings->activation_read_us += elapsed_us(
                read_start, std::chrono::steady_clock::now());
            timings->activation_rows_read += 1;
        }
        if (!read_ok)
            return false;
    }

    s->current_pos++;
    return true;
}

static inline bool run_decode_step_with_activations(
        State* s,
        int32_t token_id,
        float* logits_out,
        float* activations_out,
        MtpStepTimings* timings = nullptr) {
    // Look up embedding + PLE for this single token
    s->decode_embedding_scratch.resize(s->shape.embedding_dim);
    if (!lookup_embedding(s, token_id, s->decode_embedding_scratch.data())) return false;

    s->decode_ple_scratch.resize(s->shape.ple_floats);
    if (!lookup_ple(s, token_id, s->decode_ple_scratch.data())) return false;

    return run_decode_embedding_step(
        s, s->decode_embedding_scratch.data(), s->decode_ple_scratch.data(),
        logits_out, activations_out, timings);
}

static inline bool run_decode_step(State* s, int32_t token_id, float* logits_out) {
    return run_decode_step_with_activations(s, token_id, logits_out, nullptr);
}

// Decode step that only builds KV cache — skips the logits readback.
// Used for GPU decode-only prefill where we don't need intermediate logits.
static inline bool run_decode_step_kv_only(
        State* s,
        int32_t token_id,
        MtpStepTimings* timings = nullptr) {
    s->decode_embedding_scratch.resize(s->shape.embedding_dim);
    if (!lookup_embedding(s, token_id, s->decode_embedding_scratch.data())) return false;

    s->decode_ple_scratch.resize(s->shape.ple_floats);
    if (!lookup_ple(s, token_id, s->decode_ple_scratch.data())) return false;

    return run_decode_embedding_step(
        s, s->decode_embedding_scratch.data(), s->decode_ple_scratch.data(),
        nullptr, nullptr, timings);
}

static inline bool run_verify_step(
        State* s,
        int32_t current_token,
        const int32_t* token_ids,
        int start_pos,
        float* logits_out,
        float* activations_out = nullptr,
        MtpStepTimings* timings = nullptr) {
    if (!s->has_verify_sig || !logits_out) return false;
    const int verify_tokens = s->mtp_verify_tokens;
    const int draft_tokens = s->mtp_draft_tokens;
    if (verify_tokens <= 1 || draft_tokens <= 0 || !token_ids)
        return false;
    if (start_pos < 0 || start_pos + verify_tokens > s->kv_cache_max_len) {
        tinylog::logger().error("Gemma4 verify position exceeds KV cache",
            {{"pos", (int64_t)start_pos},
             {"tokens", (int64_t)verify_tokens},
             {"kv_cache_max_len", (int64_t)s->kv_cache_max_len}});
        return false;
    }

    std::vector<float> emb_buf(verify_tokens * s->shape.embedding_dim, 0.0f);
    std::vector<float> ple_buf(verify_tokens * s->shape.ple_floats, 0.0f);
    std::vector<float> single_emb(s->shape.embedding_dim);
    std::vector<float> single_ple(s->shape.ple_floats);

    std::vector<int32_t> verify_ids(verify_tokens, 0);
    verify_ids[0] = current_token;
    for (int i = 0; i < draft_tokens; i++)
        verify_ids[i + 1] = token_ids[i];

    for (int i = 0; i < verify_tokens; i++) {
        if (!lookup_embedding(s, verify_ids[i], single_emb.data())) return false;
        memcpy(&emb_buf[i * s->shape.embedding_dim], single_emb.data(),
               s->shape.embedding_dim * sizeof(float));
        if (!lookup_ple(s, verify_ids[i], single_ple.data())) return false;
        memcpy(&ple_buf[i * s->shape.ple_floats], single_ple.data(),
               s->shape.ple_floats * sizeof(float));
    }

    auto& vs = s->verify_sig;
    write_float_buf(vs.embeddings_buf, emb_buf.data(),
                    verify_tokens * s->shape.embedding_dim);
    write_float_buf(vs.ple_buf, ple_buf.data(),
                    verify_tokens * s->shape.ple_floats);

    std::vector<int32_t> positions(verify_tokens, 0);
    for (int i = 0; i < verify_tokens; i++)
        positions[i] = start_pos + i;
    write_int_buf(vs.pos_buf, positions.data(), verify_tokens);

    write_prefill_mask(vs.mask_buf, s->verify_mask_element_type, s->masked_value,
                       verify_tokens, s->kv_cache_max_len,
                       verify_tokens, start_pos);
    if (vs.param_buf)
        write_param_tensor(vs.param_buf, start_pos, verify_tokens);

    if (!rebuild_verify_arrays(s))
        return false;
    ScopedTensorBufferDuplicates duplicates;
    std::vector<LiteRtTensorBuffer> input_bufs;
    std::vector<LiteRtTensorBuffer> output_bufs;
    if (!duplicate_run_buffers(vs.sig, vs.input_bufs, true, duplicates, input_bufs) ||
        !duplicate_run_buffers(vs.sig, vs.output_bufs, false, duplicates, output_bufs))
        return false;

    clear_buffer_events(output_bufs.data(), output_bufs.size());

    auto model_start = std::chrono::steady_clock::now();
    bool model_ok = litert_check(LITERT(LiteRtRunCompiledModel)(
                          s->decoder_cm, vs.sig.sig_index,
                          input_bufs.size(), input_bufs.data(),
                          output_bufs.size(), output_bufs.data()),
                      "RunCompiledModel(gemma4/verify)");
    if (timings)
        timings->model_us += elapsed_us(model_start, std::chrono::steady_clock::now());
    if (!model_ok)
        return false;

    LiteRtTensorBuffer logits_buf = vs.logits_buf;
    int logits_idx = vs.sig.output_index_of("logits");
    if (logits_idx >= 0 && logits_idx < (int)output_bufs.size() && output_bufs[logits_idx])
        logits_buf = output_bufs[logits_idx];
    auto logits_read_start = std::chrono::steady_clock::now();
    bool logits_ok = read_float_buf(
        logits_buf, logits_out, verify_tokens * s->shape.vocab_size);
    if (timings) {
        timings->logits_read_us += elapsed_us(
            logits_read_start, std::chrono::steady_clock::now());
        timings->logits_rows_read += verify_tokens;
    }
    if (!logits_ok)
        return false;
    if (activations_out && vs.activations_buf) {
        LiteRtTensorBuffer activations_buf = vs.activations_buf;
        int activations_idx = vs.sig.output_index_of("activations");
        if (activations_idx >= 0 && activations_idx < (int)output_bufs.size() &&
            output_bufs[activations_idx])
            activations_buf = output_bufs[activations_idx];
        auto activation_read_start = std::chrono::steady_clock::now();
        bool activations_ok = read_float_buf(
            activations_buf, activations_out,
            verify_tokens * s->shape.embedding_dim);
        if (timings) {
            timings->activation_read_us += elapsed_us(
                activation_read_start, std::chrono::steady_clock::now());
            timings->activation_rows_read += verify_tokens;
        }
        if (!activations_ok)
            return false;
    }

    s->current_pos = start_pos + verify_tokens;
    return true;
}

static inline bool run_mtp_drafter_step(
        State* s,
        int32_t token_id,
        const float* source_activation,
        int token_pos,
        float* logits_out,
        float* projected_activation_out,
        MtpStepTimings* timings = nullptr) {
    if (!s->mtp_enabled || !source_activation || !logits_out ||
        !projected_activation_out)
        return false;

    std::vector<float> emb(s->shape.embedding_dim);
    if (!lookup_embedding(s, token_id, emb.data())) return false;

    if (s->shape.mtp_drafter_input_dim < s->shape.embedding_dim * 2)
        return false;
    std::vector<float> mtp_input(s->shape.mtp_drafter_input_dim);
    memcpy(mtp_input.data(), emb.data(), s->shape.embedding_dim * sizeof(float));
    memcpy(mtp_input.data() + s->shape.embedding_dim, source_activation,
           s->shape.embedding_dim * sizeof(float));
    write_float_buf(s->mtp_activations_buf, mtp_input.data(), mtp_input.size());

    const int drafter_pos = std::max(0, token_pos - 1);
    int32_t pos = drafter_pos;
    write_int_buf(s->mtp_pos_buf, &pos, 1);
    write_decode_mask(s->mtp_mask_buf, s->mtp_mask_element_type, s->masked_value,
                      s->kv_cache_max_len, drafter_pos);
    if (s->mtp_param_buf)
        write_param_tensor(s->mtp_param_buf, drafter_pos, 1);

    if (!rebuild_mtp_arrays(s))
        return false;
    ScopedTensorBufferDuplicates duplicates;
    std::vector<LiteRtTensorBuffer> input_bufs;
    std::vector<LiteRtTensorBuffer> output_bufs;
    if (!duplicate_run_buffers(s->mtp_sig, s->mtp_input_bufs, true, duplicates, input_bufs) ||
        !duplicate_run_buffers(s->mtp_sig, s->mtp_output_bufs, false, duplicates, output_bufs)) {
        tinylog::logger().error("Gemma4 MTP drafter buffer duplication failed");
        return false;
    }

    clear_buffer_events(output_bufs.data(), output_bufs.size());

    auto model_start = std::chrono::steady_clock::now();
    bool model_ok = litert_check(LITERT(LiteRtRunCompiledModel)(
                          s->mtp_cm, s->mtp_sig.sig_index,
                          input_bufs.size(), input_bufs.data(),
                          output_bufs.size(), output_bufs.data()),
                      "RunCompiledModel(gemma4/mtp_drafter)");
    if (timings)
        timings->model_us += elapsed_us(model_start, std::chrono::steady_clock::now());
    if (!model_ok) {
        tinylog::logger().error("Gemma4 MTP drafter run failed");
        return false;
    }

    LiteRtTensorBuffer logits_buf = s->mtp_logits_buf;
    int logits_idx = s->mtp_sig.output_index_of("logits");
    if (logits_idx >= 0 && logits_idx < (int)output_bufs.size() && output_bufs[logits_idx])
        logits_buf = output_bufs[logits_idx];
    auto logits_read_start = std::chrono::steady_clock::now();
    bool logits_ok = read_float_buf(logits_buf, logits_out, s->shape.vocab_size);
    if (timings) {
        timings->logits_read_us += elapsed_us(
            logits_read_start, std::chrono::steady_clock::now());
        timings->logits_rows_read += 1;
    }
    if (!logits_ok) {
        tinylog::logger().error("Gemma4 MTP drafter logits read failed");
        return false;
    }
    LiteRtTensorBuffer projected_buf = s->mtp_projected_activations_buf;
    int projected_idx = s->mtp_sig.output_index_of("projected_activations");
    if (projected_idx >= 0 && projected_idx < (int)output_bufs.size() &&
        output_bufs[projected_idx])
        projected_buf = output_bufs[projected_idx];
    auto activation_read_start = std::chrono::steady_clock::now();
    bool activation_ok = read_float_buf(projected_buf, projected_activation_out,
                                        s->shape.embedding_dim);
    if (timings) {
        timings->activation_read_us += elapsed_us(
            activation_read_start, std::chrono::steady_clock::now());
        timings->activation_rows_read += 1;
    }
    if (!activation_ok) {
        tinylog::logger().error("Gemma4 MTP drafter activation read failed");
        return false;
    }
    return true;
}

static inline bool draft_mtp_tokens(
        State* s,
        int32_t current_token,
        const float* target_activation,
        int current_token_pos,
        int32_t* draft_tokens,
        MtpStepTimings* timings = nullptr,
        bool allow_stop_tokens = false) {
    const int draft_token_count = s->mtp_draft_tokens;
    if (draft_token_count <= 0 || !draft_tokens)
        return false;
    std::vector<float> logits(s->shape.vocab_size);
    std::vector<float> activation(target_activation, target_activation + s->shape.embedding_dim);
    std::vector<float> projected(s->shape.embedding_dim);
    int32_t token = current_token;

    for (int i = 0; i < draft_token_count; i++) {
        if (!run_mtp_drafter_step(
                s, token, activation.data(), current_token_pos,
                logits.data(), projected.data(), timings))
            return false;
        auto sample_start = std::chrono::steady_clock::now();
        token = sample_topk_topp(logits.data(), s->shape.vocab_size,
                                 /*temperature=*/0.0f, /*top_k=*/1,
                                 /*top_p=*/1.0f, s->rng);
        if (timings)
            timings->sample_us += elapsed_us(
                sample_start, std::chrono::steady_clock::now());
        if (!allow_stop_tokens && STOP_TOKENS.count(token))
            return false;
        draft_tokens[i] = token;
        activation.swap(projected);
    }
    return true;
}

// Multi-chunk prefill. Returns the last token (pending — first decode input).
static inline int prefill_all(State* s, const int32_t* all_tokens, int count) {
    if (!all_tokens || count <= 0) return -1;
    if (count == 1) return all_tokens[0];

    int prefill_count = count - 1;
    int pending = all_tokens[count - 1];
    int offset = 0;

    while (offset < prefill_count) {
        // Pick the best signature for the remaining tokens
        int remaining = prefill_count - offset;
        PrefillSig* ps = pick_prefill_sig(s, remaining);
        if (!ps) return -1;
        int chunk = std::min(ps->seq_len, remaining);
        if (!run_prefill_chunk(s, all_tokens + offset, chunk, offset))
            return -1;
        offset += chunk;
    }
    return pending;
}

// Multi-chunk prefill that appends to the current KV position instead of
// starting at zero. Returns the last token as the pending decode input.
static inline int prefill_all_from_current(State* s, const int32_t* all_tokens, int count) {
    if (!all_tokens || count <= 0) return -1;
    if (count == 1) return all_tokens[0];

    const int base_pos = s->current_pos;
    const int prefill_count = count - 1;
    const int pending = all_tokens[count - 1];
    if (s->diagnostic_config.prefill_by_decode) {
        for (int i = 0; i < prefill_count; i++) {
            if (!run_decode_step_kv_only(s, all_tokens[i]))
                return -1;
        }
        return pending;
    }

    int offset = 0;
    const int max_prefill_chunk = s->diagnostic_config.prefill_max_chunk > 0
        ? std::clamp(s->diagnostic_config.prefill_max_chunk, 1, KV_CACHE_MAX_LEN_MAGIC_CAP)
        : KV_CACHE_MAX_LEN_MAGIC_CAP;

    while (offset < prefill_count) {
        const int remaining = prefill_count - offset;
        const int requested = std::min(remaining, max_prefill_chunk);
        PrefillSig* ps = pick_prefill_sig(s, requested);
        if (!ps) return -1;
        int chunk = std::min(ps->seq_len, requested);
        if (!run_prefill_chunk(s, all_tokens + offset, chunk, base_pos + offset))
            return -1;
        offset += chunk;
    }
    return pending;
}

static inline bool encode_prompt_for_context(
        const Tokenizer& tok,
        const std::string& prompt,
        int max_context_tokens,
        std::vector<int>& ids) {
    ids = tok.encode(prompt);
    if (ids.empty())
        return false;
    if ((int)ids.size() <= max_context_tokens)
        return true;

    const int original_tokens = (int)ids.size();
    const std::string user_turn_marker = "<|turn>user\n";
    const size_t marker_pos = prompt.find(user_turn_marker);
    if (marker_pos == std::string::npos) {
        ids.erase(ids.begin(), ids.begin() + (original_tokens - max_context_tokens));
        tinylog::logger().warn("Gemma4 prompt exceeded context; kept suffix",
            {{"original_tokens", (int64_t)original_tokens},
             {"kept_tokens", (int64_t)ids.size()},
             {"max_context_tokens", (int64_t)max_context_tokens}});
        return true;
    }

    const size_t user_content_pos = marker_pos + user_turn_marker.size();
    std::vector<int> prefix_ids = tok.encode(prompt.substr(0, user_content_pos));
    std::vector<int> suffix_ids = tok.encode(prompt.substr(user_content_pos));
    if (prefix_ids.empty() || (int)prefix_ids.size() >= max_context_tokens) {
        tinylog::logger().error("Gemma4 prompt prefix exceeds context",
            {{"prefix_tokens", (int64_t)prefix_ids.size()},
             {"original_tokens", (int64_t)original_tokens},
             {"max_context_tokens", (int64_t)max_context_tokens}});
        ids.clear();
        return false;
    }

    const int suffix_budget = max_context_tokens - (int)prefix_ids.size();
    if ((int)suffix_ids.size() > suffix_budget) {
        suffix_ids.erase(
            suffix_ids.begin(),
            suffix_ids.begin() + ((int)suffix_ids.size() - suffix_budget));
    }
    ids = std::move(prefix_ids);
    ids.insert(ids.end(), suffix_ids.begin(), suffix_ids.end());
    tinylog::logger().warn("Gemma4 prompt exceeded context; preserved prefix",
        {{"original_tokens", (int64_t)original_tokens},
         {"kept_tokens", (int64_t)ids.size()},
         {"prefix_tokens", (int64_t)(max_context_tokens - suffix_budget)},
         {"suffix_tokens", (int64_t)suffix_ids.size()},
         {"max_context_tokens", (int64_t)max_context_tokens}});
    return true;
}

static inline int rebuild_text_context(
        State* s,
        const std::vector<int>& prompt_ids,
        const std::vector<int>& output_ids) {
    if (prompt_ids.empty()) return -1;

    reset_kv_cache(s);
    int pending = prefill_all(s, prompt_ids.data(), (int)prompt_ids.size());
    if (pending < 0) return -1;

    int current = pending;
    for (int token_id : output_ids) {
        if (!run_decode_step(s, current, nullptr))
            return -1;
        current = token_id;
    }
    return current;
}

static inline int greedy_select_token(const float* logits, int vocab_size) {
    int best = 0;
    for (int i = 1; i < vocab_size; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

static inline std::string mtp_token_array_string(const int32_t* tokens, int count) {
    std::string out = "[";
    for (int i = 0; i < count; i++) {
        if (i > 0) out += ",";
        out += std::to_string(tokens[i]);
    }
    out += "]";
    return out;
}

static inline std::string mtp_logits_token_string(const float* logits, int rows, int vocab_size) {
    if (!logits || rows <= 0) return "[]";
    std::string out = "[";
    for (int i = 0; i < rows; i++) {
        if (i > 0) out += ",";
        out += std::to_string(greedy_select_token(
            logits + i * vocab_size, vocab_size));
    }
    out += "]";
    return out;
}

struct MtpDebugResult {
    bool ok = false;
    std::string error;
    int input_ids_count = 0;
    int prefill_tokens = 0;
    int pending_token = -1;
    int verify_start_pos = 0;
    int good_token = -1;
    std::vector<int> sequential_tokens;
    std::vector<int> draft_tokens;
    std::vector<int> verifier_tokens;
    std::vector<int> oracle_draft_tokens;
    std::vector<int> oracle_verifier_tokens;
    std::vector<int> oracle_prevpos_verifier_tokens;
    std::vector<int> oracle_anchored_verifier_tokens;
    int oracle_verify_kv_next_token = -1;
    int accepted_prefix = 0;
    int oracle_accepted_prefix = 0;
    int oracle_prevpos_accepted_prefix = 0;
    int oracle_anchored_accepted_prefix = 0;
};

static inline MtpDebugResult debug_mtp_verify(
        State* s,
        const std::string& prompt) {
    MtpDebugResult result;
    auto& tok = *s->tokenizer;

    if (!s->has_verify_sig) {
        result.error = "Gemma4 verify signature is not available";
        return result;
    }
    if (!s->mtp_enabled || !s->mtp_runtime_enabled || !s->has_mtp_drafter) {
        result.error = "Gemma4 MTP is not enabled";
        return result;
    }
    const int verify_tokens = s->mtp_verify_tokens;
    const int draft_tokens_count = s->mtp_draft_tokens;
    if (verify_tokens <= 1 || draft_tokens_count <= 0) {
        result.error = "Gemma4 verify signature has invalid MTP shape";
        return result;
    }

    auto ids = tok.encode(prompt);
    if (ids.empty()) {
        result.error = "Prompt encoded to no tokens";
        return result;
    }
    if ((int)ids.size() > s->kv_cache_max_len)
        ids.erase(ids.begin(), ids.begin() + ((int)ids.size() - s->kv_cache_max_len));

    result.input_ids_count = (int)ids.size();
    result.prefill_tokens = (int)ids.size() - 1;

    std::vector<float> logits(s->shape.vocab_size);

    reset_kv_cache(s);
    int pending = prefill_all(s, ids.data(), (int)ids.size());
    if (pending < 0) {
        result.error = "Sequential prefill failed";
        return result;
    }
    result.pending_token = pending;

    int current = pending;
    for (int i = 0; i < verify_tokens + 2; i++) {
        if (!run_decode_step(s, current, logits.data())) {
            result.error = "Sequential decode failed";
            return result;
        }
        int next = greedy_select_token(logits.data(), s->shape.vocab_size);
        result.sequential_tokens.push_back(next);
        if (STOP_TOKENS.count(next)) break;
        current = next;
    }
    if (result.sequential_tokens.empty()) {
        result.error = "Sequential decode produced no tokens";
        return result;
    }

    reset_kv_cache(s);
    pending = prefill_all(s, ids.data(), (int)ids.size());
    if (pending < 0) {
        result.error = "MTP prefill failed";
        return result;
    }

    std::vector<float> activation(s->shape.embedding_dim);
    if (!run_decode_step_with_activations(
            s, pending, logits.data(), activation.data())) {
        result.error = "MTP target decode failed";
        return result;
    }

    result.good_token = greedy_select_token(logits.data(), s->shape.vocab_size);
    result.verify_start_pos = s->current_pos;
    if (STOP_TOKENS.count(result.good_token)) {
        result.error = "MTP target decode produced a stop token";
        return result;
    }

    std::vector<int32_t> drafts(draft_tokens_count, 0);
    if (!draft_mtp_tokens(
            s, result.good_token, activation.data(), result.verify_start_pos,
            drafts.data(), nullptr, /*allow_stop_tokens=*/true)) {
        result.error = "MTP drafter failed";
        return result;
    }
    for (int i = 0; i < draft_tokens_count; i++)
        result.draft_tokens.push_back(drafts[i]);

    std::vector<float> verify_logits(verify_tokens * s->shape.vocab_size);
    if (!run_verify_step(
            s, result.good_token, drafts.data(), result.verify_start_pos,
            verify_logits.data())) {
        result.error = "MTP verifier failed";
        return result;
    }
    for (int i = 0; i < verify_tokens; i++) {
        result.verifier_tokens.push_back(greedy_select_token(
            verify_logits.data() + i * s->shape.vocab_size, s->shape.vocab_size));
    }

    int accepted = 0;
    for (int i = 0; i < draft_tokens_count; i++) {
        int row = i + MTP_VERIFY_LOGIT_OFFSET;
        if (row >= (int)result.verifier_tokens.size()) break;
        if (result.verifier_tokens[row] != drafts[i]) break;
        accepted++;
    }
    result.accepted_prefix = accepted;

    if ((int)result.sequential_tokens.size() > draft_tokens_count) {
        reset_kv_cache(s);
        pending = prefill_all(s, ids.data(), (int)ids.size());
        if (pending < 0) {
            result.error = "Oracle verifier prefill failed";
            return result;
        }
        std::vector<float> oracle_activation(s->shape.embedding_dim);
        if (!run_decode_step_with_activations(
                s, pending, logits.data(), oracle_activation.data())) {
            result.error = "Oracle verifier target decode failed";
            return result;
        }

        std::vector<int32_t> oracle_drafts(draft_tokens_count, 0);
        for (int i = 0; i < draft_tokens_count; i++) {
            oracle_drafts[i] = result.sequential_tokens[i + 1];
            result.oracle_draft_tokens.push_back(oracle_drafts[i]);
        }

        std::vector<float> oracle_verify_logits(verify_tokens * s->shape.vocab_size);
        if (!run_verify_step(
                s, result.good_token, oracle_drafts.data(), result.verify_start_pos,
                oracle_verify_logits.data())) {
            result.error = "Oracle verifier failed";
            return result;
        }
        for (int i = 0; i < verify_tokens; i++) {
            result.oracle_verifier_tokens.push_back(greedy_select_token(
                oracle_verify_logits.data() + i * s->shape.vocab_size, s->shape.vocab_size));
        }
        for (int i = 0; i < draft_tokens_count; i++) {
            int row = i + MTP_VERIFY_LOGIT_OFFSET;
            if (row >= (int)result.oracle_verifier_tokens.size()) break;
            if (result.oracle_verifier_tokens[row] != oracle_drafts[i]) break;
            result.oracle_accepted_prefix++;
        }
        if ((int)result.sequential_tokens.size() > verify_tokens + 1 &&
            (int)result.oracle_verifier_tokens.size() >= verify_tokens &&
            result.oracle_verifier_tokens[verify_tokens - 1] ==
                result.sequential_tokens[verify_tokens]) {
            if (run_decode_step(
                    s, result.sequential_tokens[verify_tokens],
                    logits.data())) {
                result.oracle_verify_kv_next_token =
                    greedy_select_token(logits.data(), s->shape.vocab_size);
            }
        }

        reset_kv_cache(s);
        pending = prefill_all(s, ids.data(), (int)ids.size());
        if (pending < 0) {
            result.error = "Oracle prevpos verifier prefill failed";
            return result;
        }
        if (!run_decode_step_with_activations(
                s, pending, logits.data(), oracle_activation.data())) {
            result.error = "Oracle prevpos verifier target decode failed";
            return result;
        }
        std::vector<float> oracle_prevpos_logits(verify_tokens * s->shape.vocab_size);
        if (!run_verify_step(
                s, result.good_token, oracle_drafts.data(),
                std::max(0, result.verify_start_pos - 1),
                oracle_prevpos_logits.data())) {
            result.error = "Oracle prevpos verifier failed";
            return result;
        }
        for (int i = 0; i < verify_tokens; i++) {
            result.oracle_prevpos_verifier_tokens.push_back(greedy_select_token(
                oracle_prevpos_logits.data() + i * s->shape.vocab_size, s->shape.vocab_size));
        }
        for (int i = 0; i < draft_tokens_count; i++) {
            int row = i + MTP_VERIFY_LOGIT_OFFSET;
            if (row >= (int)result.oracle_prevpos_verifier_tokens.size()) break;
            if (result.oracle_prevpos_verifier_tokens[row] != oracle_drafts[i]) break;
            result.oracle_prevpos_accepted_prefix++;
        }

        reset_kv_cache(s);
        pending = prefill_all(s, ids.data(), (int)ids.size());
        if (pending < 0) {
            result.error = "Oracle anchored verifier prefill failed";
            return result;
        }
        if (!run_decode_step_with_activations(
                s, pending, logits.data(), oracle_activation.data())) {
            result.error = "Oracle anchored verifier target decode failed";
            return result;
        }
        std::vector<int32_t> anchored_tokens(draft_tokens_count, 0);
        anchored_tokens[0] = result.good_token;
        for (int i = 1; i < draft_tokens_count; i++)
            anchored_tokens[i] = oracle_drafts[i - 1];
        std::vector<float> oracle_anchored_logits(verify_tokens * s->shape.vocab_size);
        if (!run_verify_step(
                s, result.pending_token, anchored_tokens.data(),
                std::max(0, result.verify_start_pos - 1),
                oracle_anchored_logits.data())) {
            result.error = "Oracle anchored verifier failed";
            return result;
        }
        for (int i = 0; i < verify_tokens; i++) {
            result.oracle_anchored_verifier_tokens.push_back(greedy_select_token(
                oracle_anchored_logits.data() + i * s->shape.vocab_size, s->shape.vocab_size));
        }
        for (int i = 0; i < draft_tokens_count; i++) {
            int row = i + 1;
            if (row >= (int)result.oracle_anchored_verifier_tokens.size()) break;
            if (result.oracle_anchored_verifier_tokens[row] != oracle_drafts[i]) break;
            result.oracle_anchored_accepted_prefix++;
        }
    }

    result.ok = true;
    return result;
}

static inline int prefill_embeddings_all(
        State* s,
        const std::vector<float>& embeddings,
        const std::vector<float>& per_layer_embeddings,
        int count,
        int pending_token_id) {
    if (count <= 0) return -1;
    if ((int)embeddings.size() < count * s->shape.embedding_dim ||
        (int)per_layer_embeddings.size() < count * s->shape.ple_floats)
        return -1;
    if (count == 1) return pending_token_id;

    int prefill_count = count - 1;
    int offset = 0;

    while (offset < prefill_count) {
        int remaining = prefill_count - offset;
        PrefillSig* ps = pick_prefill_sig(s, remaining);
        if (!ps) return -1;
        int chunk = std::min(ps->seq_len, remaining);
        if (!run_prefill_embedding_chunk(
                s,
                embeddings.data() + offset * s->shape.embedding_dim,
                per_layer_embeddings.data() + offset * s->shape.ple_floats,
                chunk,
                offset))
            return -1;
        offset += chunk;
    }
    return pending_token_id;
}

// ---------------------------------------------------------------------------
// Full init: load models, discover signatures, compile, allocate buffers.
// Caller provides an already-created LiteRtEnvironment (platform-specific).
// The MappedFile must outlive State.
// ---------------------------------------------------------------------------

static inline bool init(State* s, LiteRtEnvironment env,
                        const MappedFile& mapped_model,
                        Platform platform, HardwareTarget hw, int num_threads,
                        const char* compilation_cache_dir = nullptr,
                        Gemma4RuntimeConfig runtime_config = Gemma4RuntimeConfig(),
                        Gemma4DiagnosticConfig diagnostic_config = Gemma4DiagnosticConfig(),
                        bool log_initializing = true) {
    s->env = env;
    if (runtime_config.kv_cache_max_len <= 0)
        runtime_config.kv_cache_max_len =
            automatic_kv_cache_max_len_for_platform(platform);
    s->runtime_config = runtime_config;
    s->diagnostic_config = diagnostic_config;

    if (!load_models_from_bundle(s, mapped_model))
        return false;

    if (!discover_decoder_signatures(s))
        return false;

    // Detect mask element type from decode signature
    {
        int mask_idx = s->decode_sig.input_index_of("mask");
        if (mask_idx >= 0) {
            LiteRtRankedTensorType mask_type = {};
            if (get_input_tensor_type(s->decode_sig.sig, mask_idx, &mask_type))
                s->mask_element_type = mask_type.element_type;
        }
    }

    s->execution_mode = (hw == HardwareTarget::GPU) ? EXEC_MODE_GPU : EXEC_MODE_CPU;
    s->masked_value = (s->mask_element_type == kLiteRtElementTypeFloat16 ||
                       s->mask_element_type == kLiteRtElementTypeBFloat16)
                      ? MASKED_FP16 : MASKED_FP32;
    s->mtp_runtime_enabled = runtime_config.mtp_enabled;
    s->mtp_trust_verify_kv = runtime_config.mtp_trust_verify_kv;
    s->mtp_adaptive_enabled = runtime_config.mtp_adaptive_enabled;
    s->mtp_adaptive_min_cycles =
        std::clamp(runtime_config.mtp_adaptive_min_cycles, 1, 64);
    s->mtp_adaptive_min_saved_per_cycle =
        std::clamp(runtime_config.mtp_adaptive_min_saved_per_cycle, 0.0f, 4.0f);
    s->mtp_trace = runtime_config.mtp_trace;
    s->constrained_verify_batch_enabled =
        runtime_config.constrained_verify_batch < 0
            ? true
            : runtime_config.constrained_verify_batch != 0;
    const bool mtp_available =
        s->has_mtp_drafter && s->has_verify_sig && s->mtp_draft_tokens > 0;
    s->mtp_enabled = mtp_available && s->mtp_runtime_enabled;

    const TextComponentTargets targets = text_component_targets(hw);
    const std::string hw_name = hw_target_name(targets.decoder);
    std::string components = text_components(targets);
    if (s->has_mtp_drafter) {
        if (s->mtp_enabled)
            components += std::string("+mtp(") + hw_target_name(targets.decoder) + ")";
        else
            components += "+mtp(disabled)";
    }
    if (log_initializing) {
	        tinylog::logger().info("Gemma4: initializing",
	            {{"path", mapped_model.path},
	             {"hw", hw_name},
	             {"components", components},
	             {"num_threads", (int64_t)resolve_num_threads(num_threads)},
	             {"gpu_precision", (int64_t)resolve_gemma4_gpu_precision(
	                 runtime_config.gpu_precision)},
	             {"kv_cache_max_len", (int64_t)s->kv_cache_max_len},
	             {"mtp_available", mtp_available ? 1 : 0},
	             {"mtp_runtime_enabled", s->mtp_runtime_enabled ? 1 : 0},
	             {"mtp_verify_tokens", (int64_t)s->mtp_verify_tokens},
	             {"mtp_draft_tokens", (int64_t)s->mtp_draft_tokens},
	             {"mtp_adaptive", s->mtp_adaptive_enabled ? 1 : 0},
	             {"mtp_adaptive_min_cycles", (int64_t)s->mtp_adaptive_min_cycles},
	             {"mtp_adaptive_min_saved_per_cycle",
	              s->mtp_adaptive_min_saved_per_cycle},
	             {"constrained_verify_batch",
	              s->constrained_verify_batch_enabled ? 1 : 0},
	             {"diagnostic_prefill_by_decode",
	              diagnostic_config.prefill_by_decode ? 1 : 0},
	             {"diagnostic_prefill_max_chunk",
	              (int64_t)diagnostic_config.prefill_max_chunk},
	             {"diagnostic_constrained_verify_trace",
	              diagnostic_config.constrained_verify_trace ? 1 : 0},
	             {"diagnostic_constrained_verify_max_accept",
	              (int64_t)diagnostic_config.constrained_verify_max_accept}});
    }

    // Create options: decoder gets the requested hw, embedder/PLE always CPU
    // (matches LiteRT-LM — embedder lookups are too small to benefit from GPU)
    s->decoder_options = create_litert_options(
        platform, targets.decoder, ModelFamily::GEMMA4, num_threads,
        "decoder", mapped_model.path.c_str(), compilation_cache_dir,
        runtime_config.gpu_precision);
    s->embedder_options = create_litert_options(
        platform, targets.embedder, ModelFamily::GEMMA4, num_threads,
        "embedder", mapped_model.path.c_str(), compilation_cache_dir,
        runtime_config.gpu_precision);
    s->ple_options = create_litert_options(
        platform, targets.ple, ModelFamily::GEMMA4, num_threads,
        "ple", mapped_model.path.c_str(), compilation_cache_dir,
        runtime_config.gpu_precision);
    if (s->mtp_enabled) {
        s->mtp_options = create_litert_options(
            platform, targets.decoder, ModelFamily::GEMMA4, num_threads,
            "mtp_drafter", mapped_model.path.c_str(), compilation_cache_dir,
            runtime_config.gpu_precision);
    }
    if (!s->decoder_options || !s->embedder_options || !s->ple_options ||
        (s->mtp_enabled && !s->mtp_options))
        return false;

    tinylog::logger().info("Gemma4: compiling decoder",
        {{"path", mapped_model.path}, {"hw", std::string(hw_target_name(targets.decoder))}});
    log_bundle_section_digest("before_compile/original", "decoder", mapped_model.path.c_str(),
                              s->decoder_original_data, s->decoder_original_offset,
                              s->decoder_original_length);
    log_bundle_section_digest("before_compile/private_mmap", "decoder", mapped_model.path.c_str(),
                              s->decoder_region.data, s->decoder_original_offset,
                              s->decoder_region.size);
    log_metal_memory_snapshot("before gemma4/decoder compile");
    const bool decoder_compile_ok = create_compiled_model_with_magic(
        s->env, s->decoder_model, s->decoder_options, &s->decoder_cm,
        s->magic_configs, "LiteRtCreateCompiledModel(decoder)");
    log_bundle_section_digest(decoder_compile_ok ? "after_compile/original"
                                                 : "failed_compile/original",
                              "decoder", mapped_model.path.c_str(),
                              s->decoder_original_data, s->decoder_original_offset,
                              s->decoder_original_length);
    log_bundle_section_digest(decoder_compile_ok ? "after_compile/private_mmap"
                                                 : "failed_compile/private_mmap",
                              "decoder", mapped_model.path.c_str(),
                              s->decoder_region.data, s->decoder_original_offset,
                              s->decoder_region.size);
    log_metal_memory_snapshot(decoder_compile_ok
                                  ? "after gemma4/decoder compile"
                                  : "failed gemma4/decoder compile");
    if (!decoder_compile_ok)
        return false;
    tinylog::logger().info("Gemma4: compiling embedder",
        {{"path", mapped_model.path}, {"hw", std::string(hw_target_name(targets.embedder))}});
    if (!create_compiled_model_with_magic(s->env, s->embedder_model,
                                          s->embedder_options, &s->embedder_cm,
                                          nullptr,
                                          "LiteRtCreateCompiledModel(embedder)"))
        return false;
    tinylog::logger().info("Gemma4: compiling ple",
        {{"path", mapped_model.path}, {"hw", std::string(hw_target_name(targets.ple))}});
    if (!create_compiled_model_with_magic(s->env, s->ple_model,
                                          s->ple_options, &s->ple_cm,
                                          nullptr,
                                          "LiteRtCreateCompiledModel(ple)"))
        return false;
    if (s->mtp_enabled) {
        tinylog::logger().info("Gemma4: compiling mtp_drafter",
            {{"path", mapped_model.path}, {"hw", std::string(hw_target_name(targets.decoder))}});
        if (!create_compiled_model_with_magic(s->env, s->mtp_model,
                                              s->mtp_options, &s->mtp_cm,
                                              s->magic_configs,
                                              "LiteRtCreateCompiledModel(mtp_drafter)"))
            return false;
    }

    // Discover embedder/PLE signatures (single signature each, index 0)
    if (!discover_signature(s->embedder_model, 0, s->emb_sig) ||
        !discover_signature(s->ple_model, 0, s->ple_sig))
        return false;
    if (s->mtp_enabled && !discover_signature(s->mtp_model, 0, s->mtp_sig))
        return false;
    discover_model_shape(s);

    if (!allocate_buffers(s)) {
        tinylog::logger().error("buffer allocation failed");
        return false;
    }

    reset_kv_cache(s);

    return true;
}

// MARK: - Text cleanup

static inline void destroy_prefill_sig(PrefillSig& ps) {
    destroy_buffer(ps.embeddings_buf);
    destroy_buffer(ps.ple_buf);
    destroy_buffer(ps.pos_buf);
    destroy_buffer(ps.mask_buf);
    destroy_buffer(ps.param_buf);
    destroy_buffer(ps.activations_buf);
}

static inline void destroy_verify_sig(VerifySig& vs) {
    destroy_buffer(vs.embeddings_buf);
    destroy_buffer(vs.ple_buf);
    destroy_buffer(vs.pos_buf);
    destroy_buffer(vs.mask_buf);
    destroy_buffer(vs.param_buf);
    destroy_buffer(vs.logits_buf);
    destroy_buffer(vs.activations_buf);
    for (auto& buffer : vs.kv_k) destroy_buffer(buffer);
    for (auto& buffer : vs.kv_v) destroy_buffer(buffer);
    vs.kv_k.clear();
    vs.kv_v.clear();
    vs.input_bufs.clear();
    vs.output_bufs.clear();
}

static inline void destroy_buffers(State* s) {
    // Compiled models own delegate state prepared against these buffers.
    // Release them before destroying the backing tensor buffers.
    if (s->decoder_cm) LITERT(LiteRtDestroyCompiledModel)(s->decoder_cm);
    if (s->embedder_cm) LITERT(LiteRtDestroyCompiledModel)(s->embedder_cm);
    if (s->ple_cm) LITERT(LiteRtDestroyCompiledModel)(s->ple_cm);
    if (s->mtp_cm) LITERT(LiteRtDestroyCompiledModel)(s->mtp_cm);
    s->decoder_cm = s->embedder_cm = s->ple_cm = s->mtp_cm = nullptr;
    drain_active_litert_metal_queue();

    // Prefill buffers
    for (int i = 0; i < NUM_PREFILL_SIZES; i++)
        destroy_prefill_sig(s->prefill_sigs[i]);
    destroy_verify_sig(s->verify_sig);

    // Decode buffers
    destroy_buffer(s->decode_embeddings_buf);
    destroy_buffer(s->decode_ple_buf);
    destroy_buffer(s->decode_pos_buf);
    destroy_buffer(s->decode_mask_buf);
    destroy_buffer(s->decode_param_buf);
    destroy_buffer(s->decode_logits_buf);
    destroy_buffer(s->decode_activations_buf);

    destroy_buffer(s->mtp_pos_buf);
    destroy_buffer(s->mtp_activations_buf);
    destroy_buffer(s->mtp_param_buf);
    destroy_buffer(s->mtp_mask_buf);
    destroy_buffer(s->mtp_logits_buf);
    destroy_buffer(s->mtp_projected_activations_buf);
    s->mtp_input_bufs.clear();
    s->mtp_output_bufs.clear();

    // KV cache
    for (auto& buffer : s->kv_k) destroy_buffer(buffer);
    for (auto& buffer : s->kv_v) destroy_buffer(buffer);
    s->kv_k.clear();
    s->kv_v.clear();

    // Embedder + PLE
    destroy_buffer(s->emb_tokens_buf);
    destroy_buffer(s->emb_output_buf);
    destroy_buffer(s->ple_tokens_buf);
    destroy_buffer(s->ple_output_buf);

    if (s->decoder_options) LITERT(LiteRtDestroyOptions)(s->decoder_options);
    if (s->embedder_options) LITERT(LiteRtDestroyOptions)(s->embedder_options);
    if (s->ple_options) LITERT(LiteRtDestroyOptions)(s->ple_options);
    if (s->mtp_options) LITERT(LiteRtDestroyOptions)(s->mtp_options);
    s->decoder_options = s->embedder_options = s->ple_options = s->mtp_options = nullptr;

    if (s->decoder_model) LITERT(LiteRtDestroyModel)(s->decoder_model);
    if (s->embedder_model) LITERT(LiteRtDestroyModel)(s->embedder_model);
    if (s->ple_model) LITERT(LiteRtDestroyModel)(s->ple_model);
    if (s->mtp_model) LITERT(LiteRtDestroyModel)(s->mtp_model);
    s->decoder_model = s->embedder_model = s->ple_model = s->mtp_model = nullptr;
    log_bundle_section_digest("destroy/original", "decoder", s->decoder_region.path.c_str(),
                              s->decoder_original_data, s->decoder_original_offset,
                              s->decoder_original_length);
    log_bundle_section_digest("destroy/private_mmap", "decoder", s->decoder_region.path.c_str(),
                              s->decoder_region.data, s->decoder_original_offset,
                              s->decoder_region.size);
    s->decoder_region.close();
    s->mtp_region.close();
    s->decoder_original_data = nullptr;
    s->decoder_original_offset = 0;
    s->decoder_original_length = 0;

    s->tokenizer = nullptr;
    s->has_verify_sig = false;
    s->has_mtp_drafter = false;
    s->runtime_config = Gemma4RuntimeConfig();
    s->diagnostic_config = Gemma4DiagnosticConfig();
    s->mtp_enabled = false;
    s->mtp_runtime_enabled = false;
    s->mtp_trust_verify_kv = false;
    s->mtp_trace = false;
    s->mtp_verify_tokens = 0;
    s->mtp_draft_tokens = 0;
    s->mtp_adaptive_enabled = true;
    s->mtp_adaptive_min_cycles = 4;
    s->mtp_adaptive_min_saved_per_cycle = 0.5f;
    s->constrained_verify_batch_enabled = true;

    if (s->magic_configs) { std::free(s->magic_configs); s->magic_configs = nullptr; }
}

static inline int sample_for_constraint(
        const GrammarConstraint& constraint,
        float* logits,
        std::mt19937& rng,
        float temperature,
        int top_k,
        float top_p,
        int vocab_size,
        int diag_top_k,
        int decode_step,
        bool& intent_selected,
        std::vector<TokenSetLogEntry>& logs) {
    std::vector<std::pair<int, float>> diag_topk_snapshot;
    if (diag_top_k > 0)
        extract_topk(logits, vocab_size, diag_top_k, diag_topk_snapshot);

    int sampled = -1;
    int valid_count = 1;
    switch (constraint.type) {
    case ConstraintType::SINGLE_TOKEN:
        sampled = constraint.single_token_id;
        break;

    case ConstraintType::TOKEN_SET:
        valid_count = (int)constraint.valid_ids.size();
        sampled = constrained_sample(logits, vocab_size, constraint.valid_ids,
                                     temperature, top_k, top_p, rng);
        break;

    case ConstraintType::FREE_TEXT:
        intent_selected = true;
        sampled = sample_topk_topp(logits, vocab_size,
                                   temperature, top_k, top_p, rng);
        break;
    }

    if (!diag_topk_snapshot.empty()) {
        TokenSetLogEntry entry;
        entry.step = decode_step;
        entry.valid_count = valid_count;
        entry.sampled_id = sampled;
        entry.top_logits = std::move(diag_topk_snapshot);
        logs.push_back(std::move(entry));
    }

    return sampled;
}

template<typename Grammar>
static inline ConstrainedResult generate_constrained_with_verify_batches(
        State* s,
        Grammar& grammar,
        int pending_token,
        float temperature,
        int top_k,
        float top_p,
        int diag_top_k = 8,
        std::chrono::steady_clock::time_point decode_start =
            std::chrono::steady_clock::now()) {
    ConstrainedResult result;
    int current_token = pending_token;
    bool intent_selected = false;
    const bool trace_batches = s->diagnostic_config.constrained_verify_trace;
    const int verify_tokens = s->mtp_verify_tokens;
    const int draft_tokens_count = s->mtp_draft_tokens;
    const int max_verify_accept = draft_tokens_count > 0
        ? (s->diagnostic_config.constrained_verify_max_accept > 0
              ? std::clamp(s->diagnostic_config.constrained_verify_max_accept,
                           1,
                           draft_tokens_count)
              : draft_tokens_count)
        : 0;

    using clock = std::chrono::steady_clock;
    auto elapsed_ms = [](clock::time_point start, clock::time_point end) -> int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    };
    auto record_first_decode = [&]() {
        if (result.decode_steps == 0) {
            result.first_decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                clock::now() - decode_start).count();
        }
    };

    std::vector<float> logits(s->shape.vocab_size);
    std::vector<float> verify_logits(
        std::max(0, verify_tokens) * s->shape.vocab_size);

    while (!grammar.is_done()) {
        auto target_start = clock::now();
        if (!run_decode_step(s, current_token, logits.data()))
            break;
        result.target_decode_calls++;
        result.target_decode_ms += elapsed_ms(target_start, clock::now());
        record_first_decode();

        const GrammarConstraint constraint = grammar.get_constraint();
        const int model_greedy = greedy_select_token(logits.data(), s->shape.vocab_size);
        int sampled = sample_for_constraint(
            constraint, logits.data(), s->rng, temperature, top_k, top_p,
            s->shape.vocab_size, diag_top_k, result.decode_steps, intent_selected, result.logs);
        if (sampled < 0)
            break;
        if (trace_batches) {
            tinylog::logger().info("Gemma4 constrained verify trace",
                {{"kind", std::string("target")},
                 {"pos", (int64_t)(s->current_pos - 1)},
                 {"current", (int64_t)current_token},
                 {"constraint", (int64_t)static_cast<int>(constraint.type)},
                 {"model_greedy", (int64_t)model_greedy},
                 {"sampled", (int64_t)sampled}});
        }

        result.decode_steps++;
        grammar.advance(sampled);
        current_token = sampled;

        int fixed_current = -1;
        const bool sampled_was_fixed =
            constraint_fixed_token(constraint, &fixed_current) &&
            fixed_current == sampled;
        const bool can_verify_batch =
            s->constrained_verify_batch_enabled &&
            sampled_was_fixed &&
            s->has_verify_sig &&
            verify_tokens > 1 &&
            draft_tokens_count > 0 &&
            s->current_pos + verify_tokens <= s->kv_cache_max_len;
        const std::vector<int> fixed_tokens = can_verify_batch
            ? peek_fixed_tokens(grammar, draft_tokens_count)
            : std::vector<int>();

        if ((int)fixed_tokens.size() >= draft_tokens_count) {
            const int verify_start_pos = s->current_pos;
            std::vector<int32_t> verify_fixed(draft_tokens_count, 0);
            for (int i = 0; i < draft_tokens_count; i++)
                verify_fixed[i] = fixed_tokens[i];
            auto verify_start = clock::now();
            if (!run_verify_step(
                    s, current_token, verify_fixed.data(), verify_start_pos,
                    verify_logits.data(), nullptr, nullptr))
                break;
            result.verify_ms += elapsed_ms(verify_start, clock::now());
            result.verify_batches++;
            result.verify_rows += verify_tokens;
            record_first_decode();

            int accepted_fixed = 0;
            for (int i = 0; i < draft_tokens_count; i++) {
                const int verifier_token = greedy_select_token(
                    verify_logits.data() + i * s->shape.vocab_size, s->shape.vocab_size);
                if (verifier_token != fixed_tokens[i])
                    break;
                accepted_fixed++;
            }
            accepted_fixed = std::min(accepted_fixed, max_verify_accept);
            if (accepted_fixed > 0)
                result.verify_fixed_tokens += accepted_fixed + 1;
            if (trace_batches) {
                tinylog::logger().info("Gemma4 constrained verify trace",
                    {{"kind", std::string("verify")},
                     {"start_pos", (int64_t)verify_start_pos},
                     {"current", (int64_t)current_token},
                     {"fixed", mtp_token_array_string(
                          verify_fixed.data(), draft_tokens_count)},
                     {"verifier", mtp_logits_token_string(
                          verify_logits.data(), verify_tokens, s->shape.vocab_size)},
                     {"accepted", (int64_t)accepted_fixed}});
            }

            if (accepted_fixed == 0) {
                s->current_pos = verify_start_pos;
            } else {
                for (int i = 0; i < accepted_fixed; i++) {
                    int token = fixed_tokens[i];
                    if (!advance_fixed_token(grammar, token))
                        return result;
                    current_token = token;
                    result.decode_steps++;
                }
                s->current_pos = verify_start_pos + accepted_fixed;
                continue;
            }
        }
    }

    result.params = grammar.get_result();
    result.response = grammar.build_response_string(result.params);

    if (result.verify_batches > 0) {
        tinylog::logger().info("Gemma4 constrained verify batch stats",
            {{"verify_batches", (int64_t)result.verify_batches},
             {"verify_fixed_tokens", (int64_t)result.verify_fixed_tokens},
             {"verify_rows", (int64_t)result.verify_rows},
             {"target_decode_calls", (int64_t)result.target_decode_calls},
             {"decode_steps", (int64_t)result.decode_steps},
             {"verify_ms", result.verify_ms},
             {"target_decode_ms", result.target_decode_ms}});
    }

    return result;
}

// MARK: - Text generation

static inline GenerateResult generate(
    State* s,
    const std::string& prompt,
    const std::string& tool_name,
    const std::vector<ConstrainedParam>& params,
    int max_tokens,
    float temperature, int top_k, float top_p,
    int diag_top_k = 8)
{
    GenerateResult result;
    result.mtp_trust_verify_kv = s->mtp_trust_verify_kv;
    result.mtp_max_draft_tokens = s->mtp_draft_tokens;
    result.mtp_adaptive_enabled = s->mtp_adaptive_enabled;
    auto& tok = *s->tokenizer;
    using clock = std::chrono::steady_clock;
    auto elapsed_ms = [](clock::time_point start, clock::time_point end) -> int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    };

    // Encode
    auto t0 = clock::now();
    tinylog::logger().info("encoding prompt", {{"chars", (int64_t)prompt.size()}});
    std::vector<int> ids;
    if (!encode_prompt_for_context(tok, prompt, s->kv_cache_max_len, ids))
        return result;
    result.tokenize_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count();
    result.input_ids_count = (int)ids.size();
    tinylog::logger().info("starting prefill", {{"tokens", (int64_t)ids.size()}});

    // Reset + prefill
    auto t1 = clock::now();
    reset_kv_cache(s);
    int pending = prefill_all(s, ids.data(), (int)ids.size());
    tinylog::logger().info("prefill done", {{"pending", (int64_t)pending}});
    if (pending < 0) return result;
    result.prefill_tokens = (int)ids.size() - 1;
    result.prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t1).count();

    auto t2 = clock::now();
    if (!tool_name.empty()) {
        // Grammar-constrained (Gemma 4 format)
        Gemma4Grammar grammar(tok, tool_name, params, STOP_TOKENS);

        auto cr = generate_constrained_with_verify_batches<Gemma4Grammar>(
            s, grammar, pending, temperature, top_k, top_p, diag_top_k, t2);

        result.response = std::move(cr.response);
        result.params = std::move(cr.params);
        result.logs = std::move(cr.logs);
        result.decode_steps = cr.decode_steps;
        result.first_decode_ms = cr.first_decode_ms;
        result.constrained_target_decode_calls = cr.target_decode_calls;
        result.constrained_verify_batches = cr.verify_batches;
        result.constrained_verify_fixed_tokens = cr.verify_fixed_tokens;
        result.constrained_verify_rows = cr.verify_rows;
        result.constrained_target_decode_ms = cr.target_decode_ms;
        result.constrained_verify_ms = cr.verify_ms;
    } else {
        // Unconstrained
        const int verify_tokens = s->mtp_verify_tokens;
        const int draft_tokens_count = s->mtp_draft_tokens;
        std::vector<float> logits(s->shape.vocab_size);
        std::vector<float> verify_logits(
            std::max(0, verify_tokens) * s->shape.vocab_size);
        std::vector<float> activation(s->shape.embedding_dim);
        std::vector<float> verifier_activations(
            std::max(0, verify_tokens) * s->shape.embedding_dim);
        bool has_verifier_activation = false;
        int last_verified_activation_idx = 0;
        std::vector<int> output_ids;
        int current = pending;
        auto add_target_timing = [&](const MtpStepTimings& t) {
            result.mtp_target_model_us += t.model_us;
            result.mtp_target_logits_read_us += t.logits_read_us;
            result.mtp_target_activation_read_us += t.activation_read_us;
            result.mtp_target_sample_us += t.sample_us;
            result.mtp_target_logits_rows_read += t.logits_rows_read;
            result.mtp_target_activation_rows_read += t.activation_rows_read;
        };
        auto add_drafter_timing = [&](const MtpStepTimings& t) {
            result.mtp_drafter_model_us += t.model_us;
            result.mtp_drafter_logits_read_us += t.logits_read_us;
            result.mtp_drafter_activation_read_us += t.activation_read_us;
            result.mtp_drafter_sample_us += t.sample_us;
            result.mtp_drafter_logits_rows_read += t.logits_rows_read;
            result.mtp_drafter_activation_rows_read += t.activation_rows_read;
        };
        auto add_verify_timing = [&](const MtpStepTimings& t) {
            result.mtp_verify_model_us += t.model_us;
            result.mtp_verify_logits_read_us += t.logits_read_us;
            result.mtp_verify_activation_read_us += t.activation_read_us;
            result.mtp_verify_sample_us += t.sample_us;
            result.mtp_verify_logits_rows_read += t.logits_rows_read;
            result.mtp_verify_activation_rows_read += t.activation_rows_read;
        };
        auto rebuild_after_speculation = [&]() -> bool {
            auto rebuild_start = clock::now();
            int rebuilt_current = rebuild_text_context(s, ids, output_ids);
            result.mtp_rebuilds++;
            result.mtp_rebuild_ms += elapsed_ms(rebuild_start, clock::now());
            if (rebuilt_current < 0) return false;
            current = rebuilt_current;
            has_verifier_activation = false;
            return true;
        };
        auto repair_after_speculation =
            [&](int32_t cycle_token,
                const std::vector<int32_t>& emitted_after_cycle,
                int repair_start_pos) -> bool {
                if (emitted_after_cycle.empty()) return true;

                auto repair_start = clock::now();
                s->current_pos = repair_start_pos;
                int32_t repair_current = cycle_token;
                for (int32_t emitted_token : emitted_after_cycle) {
                    MtpStepTimings repair_timing;
                    if (!run_decode_step_kv_only(
                            s, repair_current, &repair_timing)) {
                        result.mtp_local_repair_ms += elapsed_ms(
                            repair_start, clock::now());
                        return false;
                    }
                    add_target_timing(repair_timing);
                    result.mtp_local_repair_tokens++;
                    repair_current = emitted_token;
                }

                result.mtp_local_repairs++;
                result.mtp_local_repair_ms += elapsed_ms(
                    repair_start, clock::now());
                current = repair_current;
                has_verifier_activation = false;
                return true;
            };
        auto mtp_net_saved_target_decode_calls = [&]() -> int {
            return result.mtp_accepted_tokens + result.mtp_replacement_tokens +
                   result.mtp_bonus_tokens - result.mtp_local_repair_tokens;
        };
        auto maybe_disable_mtp_adaptively = [&]() {
            if (!s->mtp_adaptive_enabled || result.mtp_adaptive_disabled)
                return;
            if (result.mtp_cycles < s->mtp_adaptive_min_cycles)
                return;
            const float saved_per_cycle =
                (float)mtp_net_saved_target_decode_calls() /
                (float)std::max(1, result.mtp_cycles);
            if (saved_per_cycle >= s->mtp_adaptive_min_saved_per_cycle)
                return;
            result.mtp_adaptive_disabled = true;
            result.mtp_adaptive_disable_cycle = result.mtp_cycles;
            result.mtp_adaptive_disable_output_tokens = (int)output_ids.size();
            has_verifier_activation = false;
            if (s->mtp_trace) {
                tinylog::logger().info("Gemma4 MTP adaptive disable",
                    {{"cycles", (int64_t)result.mtp_cycles},
                     {"output_tokens", (int64_t)output_ids.size()},
                     {"net_saved_target_decode_calls",
                      (int64_t)mtp_net_saved_target_decode_calls()},
                     {"saved_per_cycle", saved_per_cycle},
                     {"min_saved_per_cycle",
                      s->mtp_adaptive_min_saved_per_cycle}});
            }
        };

        auto run_mtp_cycle = [&](int32_t cycle_token,
                                 const float* source_activation,
                                 int verify_start_pos,
                                 bool cycle_token_already_output,
                                 int anchor_token) -> bool {
            if (!source_activation) return false;

            std::vector<int32_t> drafts(draft_tokens_count, 0);
            const int output_size_before_cycle = (int)output_ids.size();
            std::vector<int32_t> emitted_after_cycle;
            if (!cycle_token_already_output) {
                output_ids.push_back(cycle_token);
                current = cycle_token;
                if ((int)output_ids.size() >= max_tokens) return false;
            } else {
                current = cycle_token;
            }

            result.mtp_cycles++;
            const int cycle_index = result.mtp_cycles;
            auto drafter_start = clock::now();
            MtpStepTimings drafter_timing;
            bool drafted = draft_mtp_tokens(
                s, current, source_activation, verify_start_pos, drafts.data(),
                &drafter_timing);
            auto drafter_end = clock::now();
            add_drafter_timing(drafter_timing);
            result.mtp_drafter_ms += elapsed_ms(drafter_start, drafter_end);
            if (drafted) {
                result.mtp_drafter_calls += draft_tokens_count;
                result.mtp_draft_tokens += draft_tokens_count;
            }

            auto verify_start = clock::now();
            int verify_logit_offset = MTP_VERIFY_LOGIT_OFFSET;
            bool verified = false;
            MtpStepTimings verify_timing;
            if (drafted) {
                verified = run_verify_step(
                    s, current, drafts.data(), verify_start_pos, verify_logits.data(),
                    s->mtp_trust_verify_kv ? verifier_activations.data() : nullptr,
                    &verify_timing);
            }
            auto verify_end = clock::now();
            add_verify_timing(verify_timing);
            if (drafted) {
                result.mtp_verify_calls++;
                result.mtp_verify_ms += elapsed_ms(verify_start, verify_end);
            }

            if (verified) {
                const std::string verifier_tokens_trace = s->mtp_trace
                    ? mtp_logits_token_string(verify_logits.data(), verify_tokens,
                                              s->shape.vocab_size)
                    : std::string();
                bool rejected = false;
                bool stop_decode = false;
                int accepted_this_cycle = 0;
                int next_verified_activation_idx = 0;
                auto record_rejected_cycle = [&](int accepted_prefix) {
                    result.mtp_rejected_cycles++;
                    if (accepted_prefix <= 0)
                        result.mtp_rejected_after_prefix_0++;
                    else if (accepted_prefix == 1)
                        result.mtp_rejected_after_prefix_1++;
                    else
                        result.mtp_rejected_after_prefix_2++;
                };
                const int max_verify_draft_tokens = s->mtp_trust_verify_kv
                    ? draft_tokens_count
                    : 1;
                if (max_verify_draft_tokens < draft_tokens_count)
                    result.mtp_shadow_verify_cycles++;
                auto rejection_start = clock::now();

                for (int i = 0; i < max_verify_draft_tokens &&
                                (int)output_ids.size() < max_tokens; i++) {
                    const int verify_row = i + verify_logit_offset;
                    if (verify_row >= verify_tokens) break;
                    const float* target_logits =
                        verify_logits.data() + verify_row * s->shape.vocab_size;
                    auto sample_start = clock::now();
                    int target_token = greedy_select_token(target_logits, s->shape.vocab_size);
                    result.mtp_verify_sample_us += elapsed_us(sample_start, clock::now());
                    if (STOP_TOKENS.count(target_token)) {
                        rejected = true;
                        next_verified_activation_idx = i;
                        if (s->mtp_trust_verify_kv &&
                            (verify_logit_offset == 0 || accepted_this_cycle > 0)) {
                            stop_decode = true;
                            current = target_token;
                        } else {
                            record_rejected_cycle(accepted_this_cycle);
                            s->current_pos = verify_start_pos + accepted_this_cycle;
                        }
                        break;
                    }
                    if (target_token != drafts[i]) {
                        rejected = true;
                        next_verified_activation_idx = i;
                        record_rejected_cycle(accepted_this_cycle);
                        if (accepted_this_cycle > 0)
                            current = drafts[accepted_this_cycle - 1];
                        else
                            current = cycle_token;
                        s->current_pos = verify_start_pos + accepted_this_cycle;
                        break;
                    }
                    output_ids.push_back(drafts[i]);
                    emitted_after_cycle.push_back(drafts[i]);
                    current = drafts[i];
                    accepted_this_cycle++;
                }
                result.mtp_accepted_tokens += accepted_this_cycle;
                result.mtp_rejection_ms += elapsed_ms(rejection_start, clock::now());

                if (stop_decode) return false;
                const bool capped_after_accept =
                    !rejected && accepted_this_cycle == max_verify_draft_tokens &&
                    max_verify_draft_tokens < draft_tokens_count;
                if (!rejected && !capped_after_accept) {
                    result.mtp_full_accept_cycles++;
                    next_verified_activation_idx = draft_tokens_count;
                    if (verify_logit_offset == 0 &&
                        accepted_this_cycle == draft_tokens_count &&
                        (int)output_ids.size() < max_tokens) {
                        const float* bonus_logits =
                            verify_logits.data() + draft_tokens_count * s->shape.vocab_size;
                        auto sample_start = clock::now();
                        int bonus_token = greedy_select_token(bonus_logits, s->shape.vocab_size);
                        result.mtp_verify_sample_us += elapsed_us(sample_start, clock::now());
                        if (STOP_TOKENS.count(bonus_token)) return false;
                        output_ids.push_back(bonus_token);
                        emitted_after_cycle.push_back(bonus_token);
                        current = bonus_token;
                        s->current_pos = verify_start_pos + accepted_this_cycle + 1;
                        result.mtp_bonus_tokens++;
                    } else {
                        s->current_pos = verify_start_pos + accepted_this_cycle;
                    }
                } else if (!rejected && capped_after_accept) {
                    s->current_pos = verify_start_pos + accepted_this_cycle;
                }
                const bool trust_verifier_kv_for_cycle =
                    s->mtp_trust_verify_kv &&
                    !rejected &&
                    !capped_after_accept &&
                    accepted_this_cycle == draft_tokens_count;
                const bool should_rebuild_after_speculation =
                    !s->mtp_trust_verify_kv ||
                    (verify_logit_offset != 0 && rejected &&
                     accepted_this_cycle == 0);
                const bool should_repair_after_speculation =
                    s->mtp_trust_verify_kv &&
                    !trust_verifier_kv_for_cycle &&
                    !emitted_after_cycle.empty();
                if (s->mtp_trace) {
                    tinylog::logger().info("Gemma4 MTP trace",
                        {{"cycle", (int64_t)cycle_index},
                         {"out_tokens_before_cycle", (int64_t)output_size_before_cycle},
                         {"anchor_token", (int64_t)anchor_token},
                         {"target_token", (int64_t)cycle_token},
                         {"drafts", mtp_token_array_string(
                              drafts.data(), draft_tokens_count)},
                         {"verifier_tokens", verifier_tokens_trace},
                         {"active_draft_tokens", (int64_t)draft_tokens_count},
                         {"verify_start_pos", (int64_t)verify_start_pos},
                         {"verify_logit_offset", (int64_t)verify_logit_offset},
                         {"verify_rows_used", (int64_t)max_verify_draft_tokens},
                         {"accepted", (int64_t)accepted_this_cycle},
                         {"rejected_after_prefix", rejected ? (int64_t)accepted_this_cycle : -1},
                         {"rejected", rejected ? 1 : 0},
                         {"stop", stop_decode ? 1 : 0},
                         {"verification_capped", capped_after_accept ? 1 : 0},
                         {"trust_verifier_kv_for_cycle",
                          trust_verifier_kv_for_cycle ? 1 : 0},
                         {"trust_verify_kv", s->mtp_trust_verify_kv ? 1 : 0},
                         {"activation_carry", cycle_token_already_output ? 1 : 0},
                         {"next_activation_idx", (int64_t)next_verified_activation_idx},
                         {"will_local_repair", should_repair_after_speculation ? 1 : 0},
                         {"local_repair_tokens", (int64_t)emitted_after_cycle.size()},
                         {"will_rebuild", should_rebuild_after_speculation ? 1 : 0},
                         {"current_pos", (int64_t)s->current_pos}});
                }
                if (should_rebuild_after_speculation && !rebuild_after_speculation())
                    return false;
                if (should_repair_after_speculation &&
                    !repair_after_speculation(cycle_token, emitted_after_cycle, verify_start_pos))
                    return false;
                has_verifier_activation = trust_verifier_kv_for_cycle;
                last_verified_activation_idx = next_verified_activation_idx;
                maybe_disable_mtp_adaptively();
                return true;
            }

            s->current_pos = verify_start_pos;
            has_verifier_activation = false;
            result.mtp_fallback_cycles++;
            if (s->mtp_trace) {
                tinylog::logger().info("Gemma4 MTP trace",
                    {{"cycle", (int64_t)cycle_index},
                     {"out_tokens_before_cycle", (int64_t)output_size_before_cycle},
                     {"anchor_token", (int64_t)anchor_token},
                     {"target_token", (int64_t)cycle_token},
                     {"drafts", drafted ? mtp_token_array_string(
                          drafts.data(), draft_tokens_count) : std::string("[]")},
                     {"verifier_tokens", std::string("[]")},
                     {"active_draft_tokens", (int64_t)draft_tokens_count},
                     {"verify_start_pos", (int64_t)verify_start_pos},
                     {"verify_logit_offset", (int64_t)verify_logit_offset},
                     {"accepted", (int64_t)0},
                     {"rejected_after_prefix", (int64_t)-1},
                     {"rejected", 0},
                     {"stop", 0},
                     {"trust_verify_kv", s->mtp_trust_verify_kv ? 1 : 0},
                     {"activation_carry", cycle_token_already_output ? 1 : 0},
                     {"fallback", drafted ? 0 : 1},
                     {"will_rebuild", (drafted && !s->mtp_trust_verify_kv) ? 1 : 0},
                     {"current_pos", (int64_t)s->current_pos}});
            }
            if (drafted && !s->mtp_trust_verify_kv && !rebuild_after_speculation())
                return false;
            maybe_disable_mtp_adaptively();
            return true;
        };

        while ((int)output_ids.size() < max_tokens) {
            const bool use_mtp = s->mtp_enabled && s->mtp_runtime_enabled &&
                                 !result.mtp_adaptive_disabled &&
                                 verify_tokens > 1 &&
                                 draft_tokens_count > 0 &&
                                 (max_tokens - (int)output_ids.size()) >= verify_tokens;
            if (use_mtp && s->mtp_trust_verify_kv && has_verifier_activation) {
                const float* carried_activation =
                    verifier_activations.data() +
                    last_verified_activation_idx * s->shape.embedding_dim;
                if (!run_mtp_cycle(
                        current, carried_activation, s->current_pos,
                        /*cycle_token_already_output=*/true,
                        /*anchor_token=*/current))
                    break;
                continue;
            }

            const int anchor_token = current;
            auto target_decode_start = clock::now();
            MtpStepTimings target_timing;
            bool decoded = use_mtp
                ? run_decode_step_with_activations(
                      s, current, logits.data(), activation.data(),
                      &target_timing)
                : run_decode_step(s, current, logits.data());
            auto target_decode_end = clock::now();
            if (use_mtp) {
                add_target_timing(target_timing);
                result.mtp_target_decode_calls++;
                result.mtp_target_decode_ms += elapsed_ms(
                    target_decode_start, target_decode_end);
            }
            if (!decoded) break;
            if (output_ids.empty()) {
                result.first_decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    clock::now() - t2).count();
            }

            auto sample_start = clock::now();
            int next = sample_topk_topp(logits.data(), s->shape.vocab_size,
                                         temperature, top_k, top_p, s->rng);
            if (use_mtp)
                result.mtp_target_sample_us += elapsed_us(sample_start, clock::now());
            if (STOP_TOKENS.count(next)) break;

            if (use_mtp) {
                const int verify_start_pos = s->current_pos;
                if (!run_mtp_cycle(
                        next, activation.data(), verify_start_pos,
                        /*cycle_token_already_output=*/false, anchor_token))
                    break;
                continue;
            }

            output_ids.push_back(next);
            current = next;
        }

        result.response = tok.decode(output_ids);
        result.decode_steps = (int)output_ids.size();
    }
    result.decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t2).count();
    if (result.mtp_cycles > 0) {
        tinylog::logger().info("Gemma4 MTP stats",
            {{"cycles", (int64_t)result.mtp_cycles},
             {"max_draft_tokens", (int64_t)result.mtp_max_draft_tokens},
             {"adaptive_enabled", result.mtp_adaptive_enabled ? 1 : 0},
             {"adaptive_disabled", result.mtp_adaptive_disabled ? 1 : 0},
             {"adaptive_disable_cycle",
              (int64_t)result.mtp_adaptive_disable_cycle},
             {"adaptive_disable_output_tokens",
              (int64_t)result.mtp_adaptive_disable_output_tokens},
             {"target_decode_calls", (int64_t)result.mtp_target_decode_calls},
             {"drafter_calls", (int64_t)result.mtp_drafter_calls},
             {"verify_calls", (int64_t)result.mtp_verify_calls},
             {"draft_tokens", (int64_t)result.mtp_draft_tokens},
             {"accepted_tokens", (int64_t)result.mtp_accepted_tokens},
             {"rejected_cycles", (int64_t)result.mtp_rejected_cycles},
             {"rejected_after_prefix_0",
              (int64_t)result.mtp_rejected_after_prefix_0},
             {"rejected_after_prefix_1",
              (int64_t)result.mtp_rejected_after_prefix_1},
             {"rejected_after_prefix_2",
              (int64_t)result.mtp_rejected_after_prefix_2},
             {"full_accept_cycles", (int64_t)result.mtp_full_accept_cycles},
             {"shadow_verify_cycles", (int64_t)result.mtp_shadow_verify_cycles},
             {"replacement_tokens", (int64_t)result.mtp_replacement_tokens},
             {"bonus_tokens", (int64_t)result.mtp_bonus_tokens},
             {"fallback_cycles", (int64_t)result.mtp_fallback_cycles},
             {"rebuilds", (int64_t)result.mtp_rebuilds},
             {"local_repairs", (int64_t)result.mtp_local_repairs},
             {"local_repair_tokens", (int64_t)result.mtp_local_repair_tokens},
             {"target_decode_ms", result.mtp_target_decode_ms},
             {"drafter_ms", result.mtp_drafter_ms},
             {"verify_ms", result.mtp_verify_ms},
             {"rejection_ms", result.mtp_rejection_ms},
             {"rebuild_ms", result.mtp_rebuild_ms},
             {"local_repair_ms", result.mtp_local_repair_ms},
             {"model_us", result.mtp_target_model_us +
                          result.mtp_drafter_model_us +
                          result.mtp_verify_model_us},
             {"logits_read_us", result.mtp_target_logits_read_us +
                                result.mtp_drafter_logits_read_us +
                                result.mtp_verify_logits_read_us},
             {"activation_read_us", result.mtp_target_activation_read_us +
                                    result.mtp_drafter_activation_read_us +
                                    result.mtp_verify_activation_read_us},
             {"sample_us", result.mtp_target_sample_us +
                           result.mtp_drafter_sample_us +
                           result.mtp_verify_sample_us},
             {"logits_rows_read",
              (int64_t)(result.mtp_target_logits_rows_read +
                        result.mtp_drafter_logits_rows_read +
                        result.mtp_verify_logits_rows_read)},
             {"activation_rows_read",
              (int64_t)(result.mtp_target_activation_rows_read +
                        result.mtp_drafter_activation_rows_read +
                        result.mtp_verify_activation_rows_read)}});
    }

    return result;
}

static inline std::vector<ConstrainedParam> constrained_params_for_tool(
        const ToolDef& tool) {
    std::vector<ConstrainedParam> cparams;
    cparams.reserve(tool.params.size());
    for (auto& p : tool.params)
        cparams.push_back({p.name, p.enum_values});
    return cparams;
}

static inline GenerateResult generate_tool_aware(
    State* s,
    const std::string& prompt,
    const std::vector<ToolDef>& tools,
    int max_tokens,
    float temperature, int top_k, float top_p,
    int reserve_output_tokens = 0,
    int diag_top_k = 8,
    std::function<bool(const std::string&)> stream_callback = {})
{
    GenerateResult result;
    result.mtp_trust_verify_kv = s->mtp_trust_verify_kv;
    result.mtp_max_draft_tokens = s->mtp_draft_tokens;
    result.mtp_adaptive_enabled = s->mtp_adaptive_enabled;
    auto& tok = *s->tokenizer;
    using clock = std::chrono::steady_clock;
    auto elapsed_ms = [](clock::time_point start, clock::time_point end) -> int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    };

    auto t0 = clock::now();
    tinylog::logger().info("encoding prompt", {{"chars", (int64_t)prompt.size()}});
    std::vector<int> ids;
    const int reserve = std::max(0, reserve_output_tokens);
    const int max_prompt_tokens = std::max(1, s->kv_cache_max_len - reserve);
    if (!encode_prompt_for_context(tok, prompt, max_prompt_tokens, ids))
        return result;
    result.tokenize_ms = elapsed_ms(t0, clock::now());
    result.input_ids_count = (int)ids.size();
    tinylog::logger().info("starting prefill", {{"tokens", (int64_t)ids.size()}});

    auto t1 = clock::now();
    reset_kv_cache(s);
    int pending = prefill_all(s, ids.data(), (int)ids.size());
    tinylog::logger().info("prefill done", {{"pending", (int64_t)pending}});
    if (pending < 0) return result;
    result.prefill_tokens = (int)ids.size() - 1;
    result.prefill_ms = elapsed_ms(t1, clock::now());

    auto t2 = clock::now();
    std::vector<float> logits(s->shape.vocab_size);
    int current = pending;

    auto run_target_decode = [&](int token_id) -> bool {
        auto start = clock::now();
        const bool ok = run_decode_step(s, token_id, logits.data());
        result.constrained_target_decode_calls++;
        result.constrained_target_decode_ms += elapsed_ms(start, clock::now());
        if (ok && result.decode_steps == 0) {
            result.first_decode_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    clock::now() - t2).count();
        }
        return ok;
    };

    if (!run_target_decode(current)) {
        result.decode_ms = elapsed_ms(t2, clock::now());
        return result;
    }

    int first = sample_topk_topp(logits.data(), s->shape.vocab_size,
                                 temperature, top_k, top_p, s->rng);
    if (STOP_TOKENS.count(first)) {
        result.decode_ms = elapsed_ms(t2, clock::now());
        return result;
    }

    const int start_tool_call_id = tok.piece_to_id("<|tool_call>");
    result.decode_steps++;

    if (tools.empty() || first != start_tool_call_id) {
        std::vector<int> output_ids;
        std::string emitted_text;
        auto emit_decoded_delta = [&]() -> bool {
            if (!stream_callback)
                return true;
            const std::string decoded = tok.decode(output_ids);
            std::string delta;
            if (decoded.rfind(emitted_text, 0) == 0) {
                delta = decoded.substr(emitted_text.size());
            } else {
                delta = decoded;
            }
            emitted_text = decoded;
            return delta.empty() || stream_callback(delta);
        };
        output_ids.push_back(first);
        current = first;
        if (!emit_decoded_delta()) {
            result.response = tok.decode(output_ids);
            result.decode_steps = (int)output_ids.size();
            result.decode_ms = elapsed_ms(t2, clock::now());
            return result;
        }

        while ((int)output_ids.size() < max_tokens) {
            if (!run_decode_step(s, current, logits.data()))
                break;
            int next = sample_topk_topp(logits.data(), s->shape.vocab_size,
                                        temperature, top_k, top_p, s->rng);
            if (STOP_TOKENS.count(next))
                break;
            output_ids.push_back(next);
            current = next;
            result.decode_steps = (int)output_ids.size();
            if (!emit_decoded_delta())
                break;
        }

        result.response = tok.decode(output_ids);
        result.decode_steps = (int)output_ids.size();
        result.decode_ms = elapsed_ms(t2, clock::now());
        return result;
    }

    struct ToolCandidate {
        int tool_index = -1;
        std::vector<int> tokens;
    };

    std::vector<ToolCandidate> candidates;
    candidates.reserve(tools.size());
    for (size_t i = 0; i < tools.size(); i++) {
        auto name_tokens = tok.encode("call:" + tools[i].name + "{");
        if (!name_tokens.empty())
            candidates.push_back({(int)i, std::move(name_tokens)});
    }
    if (candidates.empty()) {
        result.decode_ms = elapsed_ms(t2, clock::now());
        return result;
    }

    current = first;
    size_t candidate_pos = 0;
    int selected_tool_index = -1;
    std::vector<int> selected_prefix;

    while (selected_tool_index < 0) {
        if (candidates.size() == 1 &&
            candidate_pos >= candidates[0].tokens.size()) {
            selected_tool_index = candidates[0].tool_index;
            selected_prefix = candidates[0].tokens;
            break;
        }

        std::vector<int> valid_ids;
        for (auto& candidate : candidates) {
            if (candidate_pos < candidate.tokens.size())
                valid_ids.push_back(candidate.tokens[candidate_pos]);
        }
        std::sort(valid_ids.begin(), valid_ids.end());
        valid_ids.erase(std::unique(valid_ids.begin(), valid_ids.end()), valid_ids.end());
        if (valid_ids.empty())
            break;

        if (!run_target_decode(current))
            break;
        std::vector<std::pair<int, float>> diag_topk_snapshot;
        if (diag_top_k > 0)
            extract_topk(logits.data(), s->shape.vocab_size, diag_top_k, diag_topk_snapshot);
        int sampled = constrained_sample(logits.data(), s->shape.vocab_size, valid_ids,
                                         temperature, top_k, top_p, s->rng);
        if (sampled < 0)
            break;

        if (!diag_topk_snapshot.empty()) {
            TokenSetLogEntry entry;
            entry.step = result.decode_steps;
            entry.valid_count = (int)valid_ids.size();
            entry.sampled_id = sampled;
            entry.top_logits = std::move(diag_topk_snapshot);
            result.logs.push_back(std::move(entry));
        }

        result.decode_steps++;
        current = sampled;
        candidate_pos++;

        std::vector<ToolCandidate> remaining;
        remaining.reserve(candidates.size());
        for (auto& candidate : candidates) {
            if (candidate_pos <= candidate.tokens.size() &&
                candidate.tokens[candidate_pos - 1] == sampled)
                remaining.push_back(std::move(candidate));
        }
        candidates = std::move(remaining);
        if (candidates.empty())
            break;
    }

    if (selected_tool_index < 0 ||
        selected_tool_index >= (int)tools.size() ||
        selected_prefix.empty()) {
        result.decode_ms = elapsed_ms(t2, clock::now());
        return result;
    }

    const ToolDef& selected_tool = tools[selected_tool_index];
    auto cparams = constrained_params_for_tool(selected_tool);
    Gemma4Grammar grammar(tok, selected_tool.name, cparams, STOP_TOKENS);
    grammar.advance(start_tool_call_id);
    for (int token : selected_prefix)
        grammar.advance(token);

    auto cr = generate_constrained_with_verify_batches<Gemma4Grammar>(
        s, grammar, current, temperature, top_k, top_p, diag_top_k, t2);

    result.response = std::move(cr.response);
    result.params = std::move(cr.params);
    result.logs.insert(result.logs.end(), cr.logs.begin(), cr.logs.end());
    result.decode_steps += cr.decode_steps;
    if (result.first_decode_ms == 0)
        result.first_decode_ms = cr.first_decode_ms;
    result.constrained_target_decode_calls += cr.target_decode_calls;
    result.constrained_verify_batches = cr.verify_batches;
    result.constrained_verify_fixed_tokens = cr.verify_fixed_tokens;
    result.constrained_verify_rows = cr.verify_rows;
    result.constrained_target_decode_ms += cr.target_decode_ms;
    result.constrained_verify_ms = cr.verify_ms;
    result.decode_ms = elapsed_ms(t2, clock::now());
    return result;
}

static inline GenerateResult continue_tool_aware(
    State* s,
    const std::string& prompt_suffix,
    const std::vector<ToolDef>& tools,
    int max_tokens,
    float temperature, int top_k, float top_p,
    int reserve_output_tokens = 0,
    int diag_top_k = 8,
    std::function<bool(const std::string&)> stream_callback = {})
{
    GenerateResult result;
    result.mtp_trust_verify_kv = s->mtp_trust_verify_kv;
    result.mtp_max_draft_tokens = s->mtp_draft_tokens;
    result.mtp_adaptive_enabled = s->mtp_adaptive_enabled;
    auto& tok = *s->tokenizer;
    using clock = std::chrono::steady_clock;
    auto elapsed_ms = [](clock::time_point start, clock::time_point end) -> int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    };

    auto t0 = clock::now();
    tinylog::logger().info("encoding incremental prompt suffix", {{"chars", (int64_t)prompt_suffix.size()}});
    std::vector<int> ids = tok.encode(prompt_suffix);
    result.tokenize_ms = elapsed_ms(t0, clock::now());
    result.input_ids_count = (int)ids.size();
    result.prompt = prompt_suffix;
    if (ids.empty())
        return result;
    const int reserve = std::max(0, reserve_output_tokens);
    if (s->current_pos < 0 || s->current_pos + (int)ids.size() + reserve > s->kv_cache_max_len) {
        tinylog::logger().warn("Gemma4 incremental prompt exceeds KV cache",
            {{"current_pos", (int64_t)s->current_pos},
             {"tokens", (int64_t)ids.size()},
             {"reserve_output_tokens", (int64_t)reserve},
             {"kv_cache_max_len", (int64_t)s->kv_cache_max_len}});
        return result;
    }
    tinylog::logger().info("starting incremental prefill",
        {{"current_pos", (int64_t)s->current_pos},
         {"tokens", (int64_t)ids.size()}});

    auto t1 = clock::now();
    int pending = prefill_all_from_current(s, ids.data(), (int)ids.size());
    tinylog::logger().info("incremental prefill done", {{"pending", (int64_t)pending}});
    if (pending < 0) return result;
    result.prefill_tokens = (int)ids.size() - 1;
    result.prefill_ms = elapsed_ms(t1, clock::now());

    auto t2 = clock::now();
    std::vector<float> logits(s->shape.vocab_size);
    int current = pending;

    auto run_target_decode = [&](int token_id) -> bool {
        auto start = clock::now();
        const bool ok = run_decode_step(s, token_id, logits.data());
        result.constrained_target_decode_calls++;
        result.constrained_target_decode_ms += elapsed_ms(start, clock::now());
        if (ok && result.decode_steps == 0) {
            result.first_decode_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    clock::now() - t2).count();
        }
        return ok;
    };

    if (!run_target_decode(current)) {
        result.decode_ms = elapsed_ms(t2, clock::now());
        return result;
    }

    int first = sample_topk_topp(logits.data(), s->shape.vocab_size,
                                 temperature, top_k, top_p, s->rng);
    if (STOP_TOKENS.count(first)) {
        result.decode_ms = elapsed_ms(t2, clock::now());
        return result;
    }

    const int start_tool_call_id = tok.piece_to_id("<|tool_call>");
    result.decode_steps++;

    if (tools.empty() || first != start_tool_call_id) {
        std::vector<int> output_ids;
        std::string emitted_text;
        auto emit_decoded_delta = [&]() -> bool {
            if (!stream_callback)
                return true;
            const std::string decoded = tok.decode(output_ids);
            std::string delta;
            if (decoded.rfind(emitted_text, 0) == 0) {
                delta = decoded.substr(emitted_text.size());
            } else {
                delta = decoded;
            }
            emitted_text = decoded;
            return delta.empty() || stream_callback(delta);
        };
        output_ids.push_back(first);
        current = first;
        if (!emit_decoded_delta()) {
            result.response = tok.decode(output_ids);
            result.decode_steps = (int)output_ids.size();
            result.decode_ms = elapsed_ms(t2, clock::now());
            return result;
        }

        while ((int)output_ids.size() < max_tokens) {
            if (!run_decode_step(s, current, logits.data()))
                break;
            int next = sample_topk_topp(logits.data(), s->shape.vocab_size,
                                        temperature, top_k, top_p, s->rng);
            if (STOP_TOKENS.count(next))
                break;
            output_ids.push_back(next);
            current = next;
            result.decode_steps = (int)output_ids.size();
            if (!emit_decoded_delta())
                break;
        }

        result.response = tok.decode(output_ids);
        result.decode_steps = (int)output_ids.size();
        result.decode_ms = elapsed_ms(t2, clock::now());
        return result;
    }

    struct ToolCandidate {
        int tool_index = -1;
        std::vector<int> tokens;
    };

    std::vector<ToolCandidate> candidates;
    candidates.reserve(tools.size());
    for (size_t i = 0; i < tools.size(); i++) {
        auto name_tokens = tok.encode("call:" + tools[i].name + "{");
        if (!name_tokens.empty())
            candidates.push_back({(int)i, std::move(name_tokens)});
    }
    if (candidates.empty()) {
        result.decode_ms = elapsed_ms(t2, clock::now());
        return result;
    }

    current = first;
    size_t candidate_pos = 0;
    int selected_tool_index = -1;
    std::vector<int> selected_prefix;

    while (selected_tool_index < 0) {
        if (candidates.size() == 1 &&
            candidate_pos >= candidates[0].tokens.size()) {
            selected_tool_index = candidates[0].tool_index;
            selected_prefix = candidates[0].tokens;
            break;
        }

        std::vector<int> valid_ids;
        for (auto& candidate : candidates) {
            if (candidate_pos < candidate.tokens.size())
                valid_ids.push_back(candidate.tokens[candidate_pos]);
        }
        std::sort(valid_ids.begin(), valid_ids.end());
        valid_ids.erase(std::unique(valid_ids.begin(), valid_ids.end()), valid_ids.end());
        if (valid_ids.empty())
            break;

        if (!run_target_decode(current))
            break;
        std::vector<std::pair<int, float>> diag_topk_snapshot;
        if (diag_top_k > 0)
            extract_topk(logits.data(), s->shape.vocab_size, diag_top_k, diag_topk_snapshot);
        int sampled = constrained_sample(logits.data(), s->shape.vocab_size, valid_ids,
                                         temperature, top_k, top_p, s->rng);
        if (sampled < 0)
            break;

        if (!diag_topk_snapshot.empty()) {
            TokenSetLogEntry entry;
            entry.step = result.decode_steps;
            entry.valid_count = (int)valid_ids.size();
            entry.sampled_id = sampled;
            entry.top_logits = std::move(diag_topk_snapshot);
            result.logs.push_back(std::move(entry));
        }

        result.decode_steps++;
        current = sampled;
        candidate_pos++;

        std::vector<ToolCandidate> remaining;
        remaining.reserve(candidates.size());
        for (auto& candidate : candidates) {
            if (candidate_pos <= candidate.tokens.size() &&
                candidate.tokens[candidate_pos - 1] == sampled)
                remaining.push_back(std::move(candidate));
        }
        candidates = std::move(remaining);
        if (candidates.empty())
            break;
    }

    if (selected_tool_index < 0 ||
        selected_tool_index >= (int)tools.size() ||
        selected_prefix.empty()) {
        result.decode_ms = elapsed_ms(t2, clock::now());
        return result;
    }

    const ToolDef& selected_tool = tools[selected_tool_index];
    auto cparams = constrained_params_for_tool(selected_tool);
    Gemma4Grammar grammar(tok, selected_tool.name, cparams, STOP_TOKENS);
    grammar.advance(start_tool_call_id);
    for (int token : selected_prefix)
        grammar.advance(token);

    auto cr = generate_constrained_with_verify_batches<Gemma4Grammar>(
        s, grammar, current, temperature, top_k, top_p, diag_top_k, t2);

    result.response = std::move(cr.response);
    result.params = std::move(cr.params);
    result.logs.insert(result.logs.end(), cr.logs.begin(), cr.logs.end());
    result.decode_steps += cr.decode_steps;
    if (result.first_decode_ms == 0)
        result.first_decode_ms = cr.first_decode_ms;
    result.constrained_target_decode_calls += cr.target_decode_calls;
    result.constrained_verify_batches = cr.verify_batches;
    result.constrained_verify_fixed_tokens = cr.verify_fixed_tokens;
    result.constrained_verify_rows = cr.verify_rows;
    result.constrained_target_decode_ms += cr.target_decode_ms;
    result.constrained_verify_ms = cr.verify_ms;
    result.decode_ms = elapsed_ms(t2, clock::now());
    return result;
}

static inline GenerateResult continue_after_tool_response(
    State* s,
    const std::string& tool_response,
    int max_tokens,
    float temperature, int top_k, float top_p,
    int reserve_output_tokens = 0,
    std::function<bool(const std::string&)> stream_callback = {})
{
    GenerateResult result;
    result.mtp_trust_verify_kv = s->mtp_trust_verify_kv;
    result.mtp_max_draft_tokens = s->mtp_draft_tokens;
    result.mtp_adaptive_enabled = s->mtp_adaptive_enabled;
    auto& tok = *s->tokenizer;
    using clock = std::chrono::steady_clock;
    auto elapsed_ms = [](clock::time_point start, clock::time_point end) -> int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    };

    auto t0 = clock::now();
    std::vector<int> ids;
    const int tool_response_id = tok.piece_to_id("<|tool_response>");
    if (tool_response_id < 0) {
        tinylog::logger().error("Gemma4 missing tool response token");
        return result;
    }
    ids.push_back(tool_response_id);

    const std::string suffix = tool_response + "<tool_response|>";
    std::vector<int> suffix_ids = tok.encode(suffix);
    ids.insert(ids.end(), suffix_ids.begin(), suffix_ids.end());
    result.tokenize_ms = elapsed_ms(t0, clock::now());
    result.input_ids_count = (int)ids.size();
    result.prompt = suffix;

    const int reserve = std::max(0, reserve_output_tokens);
    if (s->current_pos < 0 || s->current_pos + (int)ids.size() + reserve > s->kv_cache_max_len) {
        tinylog::logger().error("Gemma4 tool response continuation exceeds KV cache",
            {{"current_pos", (int64_t)s->current_pos},
             {"tokens", (int64_t)ids.size()},
             {"reserve_output_tokens", (int64_t)reserve},
             {"kv_cache_max_len", (int64_t)s->kv_cache_max_len}});
        return result;
    }

    tinylog::logger().info("Gemma4 continuing after tool response",
        {{"current_pos", (int64_t)s->current_pos},
         {"tokens", (int64_t)ids.size()}});

    auto t1 = clock::now();
    int pending = prefill_all_from_current(s, ids.data(), (int)ids.size());
    if (pending < 0) {
        tinylog::logger().error("Gemma4 tool response continuation prefill failed");
        return result;
    }
    result.prefill_tokens = (int)ids.size() - 1;
    result.prefill_ms = elapsed_ms(t1, clock::now());

    auto t2 = clock::now();
    std::vector<float> logits(s->shape.vocab_size);
    std::vector<int> output_ids;
    std::string emitted_text;
    int current = pending;

    auto emit_decoded_delta = [&]() -> bool {
        if (!stream_callback)
            return true;
        const std::string decoded = tok.decode(output_ids);
        std::string delta;
        if (decoded.rfind(emitted_text, 0) == 0) {
            delta = decoded.substr(emitted_text.size());
        } else {
            delta = decoded;
        }
        emitted_text = decoded;
        return delta.empty() || stream_callback(delta);
    };

    while ((int)output_ids.size() < max_tokens) {
        if (!run_decode_step(s, current, logits.data()))
            break;
        if (result.decode_steps == 0) {
            result.first_decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                clock::now() - t2).count();
        }
        int next = sample_topk_topp(logits.data(), s->shape.vocab_size,
                                    temperature, top_k, top_p, s->rng);
        if (STOP_TOKENS.count(next))
            break;
        output_ids.push_back(next);
        current = next;
        result.decode_steps = (int)output_ids.size();
        if (!emit_decoded_delta())
            break;
    }

    result.response = tok.decode(output_ids);
    result.decode_steps = (int)output_ids.size();
    result.decode_ms = elapsed_ms(t2, clock::now());
    return result;
}

// MARK: - Text prompt formatting

static inline std::string escape_tool_text(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': break;
            case '\t': out += "\\t"; break;
            default:   out += c;
        }
    }
    return out;
}

static inline std::string format_tool_declaration(const ToolDef& tool) {
    std::string props;

    auto sorted_params = tool.params;
    std::sort(sorted_params.begin(), sorted_params.end(),
              [](const ToolParam& a, const ToolParam& b) { return a.name < b.name; });

    for (size_t i = 0; i < sorted_params.size(); i++) {
        auto& p = sorted_params[i];
        if (i > 0) props += ",";
        std::string type = p.type.empty() ? "STRING" : p.type;
        for (auto& c : type) c = (char)std::toupper((unsigned char)c);
        props += p.name + ":{description:<|\"|>" + escape_tool_text(p.description) +
            "<|\"|>,type:<|\"|>" + type + "<|\"|>";
        if (!p.enum_values.empty()) {
            props += ",enum:[";
            for (size_t j = 0; j < p.enum_values.size(); j++) {
                if (j > 0) props += ",";
                props += "<|\"|>" + escape_tool_text(p.enum_values[j]) + "<|\"|>";
            }
            props += "]";
        }
        props += "}";
    }

    std::string required;
    for (auto& p : sorted_params) {
        if (!p.required) continue;
        if (!required.empty()) required += ",";
        required += "<|\"|>" + p.name + "<|\"|>";
    }

    return "declaration:" + tool.name + "{"
        "description:<|\"|>" + escape_tool_text(tool.description) + "<|\"|>,"
        "parameters:{properties:{" + props + "},"
        "required:[" + required + "],"
        "type:<|\"|>OBJECT<|\"|>}}";
}

static inline std::string format_fc_prompt(const ToolDef& tool,
                                           const std::string& user_message,
                                           const std::string& system_message = "") {
    std::string prompt = "<bos><|turn>system\n";
    if (!system_message.empty())
        prompt += system_message;
    prompt += "<|tool>";
    prompt += format_tool_declaration(tool);
    prompt += "<tool|><turn|>\n<|turn>user\n";
    prompt += user_message;
    prompt += "<turn|>\n<|turn>model\n";
    return prompt;
}

static inline std::string format_fc_prompt(const std::vector<ToolDef>& tools,
                                           const std::string& user_message,
                                           const std::string& system_message = "") {
    std::string prompt = "<bos><|turn>system\n";
    if (!system_message.empty())
        prompt += system_message;
    if (!tools.empty()) {
        prompt += "<|tool>";
        for (auto& tool : tools)
            prompt += format_tool_declaration(tool);
        prompt += "<tool|>";
    }
    prompt += "<turn|>\n<|turn>user\n";
    prompt += user_message;
    prompt += "<turn|>\n<|turn>model\n";
    return prompt;
}

static inline std::string format_chat_prompt(const std::string& user_message,
                                             const std::string& system_message = "") {
    std::string prompt = "<bos>";
    if (!system_message.empty()) {
        prompt += "<|turn>system\n";
        prompt += system_message;
        prompt += "<turn|>\n";
    }
    prompt += "<|turn>user\n";
    prompt += user_message;
    prompt += "<turn|>\n<|turn>model\n";
    return prompt;
}

static inline std::string format_gemma4_completion(
        const ToolDef& tool,
        const std::map<std::string, std::string>& arguments,
        bool include_function_response = true) {
    std::string out = "<|tool_call>call:" + tool.name + "{";
    bool first = true;
    for (auto& p : tool.params) {
        auto it = arguments.find(p.name);
        if (it == arguments.end()) continue;
        if (!first) out += ",";
        first = false;
        out += p.name + ":<|\"|>" + escape_tool_text(it->second) + "<|\"|>";
    }
    out += "}<tool_call|>";
    if (include_function_response)
        out += "<|tool_response>";
    return out;
}

// MARK: - Vision runtime

// Gemma 4 vision encoder/adapter support.
//
// Vision-side companion to the Gemma4 text decoder. The .litertlm bundle may
// contain three vision sections: vision_encoder, vision_adapter, and optional
// end_of_vision. This file preprocesses RGB888 images, runs those vision
// components, and returns text-decoder-sized image embeddings.
//
// Model structure:
//   - Vision encoder: image patches + positions -> mask + 768-d features.
//   - Vision adapter: encoder features -> Gemma4 1536-d image token embeddings.
//   - End-of-vision: optional learned 1536-d terminator embedding.
//
// Tensor flow:
//   RGB888 image
//     -> resize to patch budget and patchify
//     -> images float32 [signature_patches,16*16*3], positions_xy int32 [signature_patches,2]
//     -> encoder mask uint8 [signature_tokens], features float32 [signature_tokens,768]
//     -> adapter embeddings float32 [signature_tokens,1536]
//     -> optional eoi_embedding float32 [1536]
//
// Prompt assembly and decoder splicing live with the main Gemma4 generation
// path; this file stops at reusable image embeddings plus mask metadata.

// MARK: - Vision constants

static constexpr int VISION_ENCODER_DIM = 768;
static constexpr int VISION_DEFAULT_IMAGE_TOKEN_BUDGET = 196;
static constexpr int VISION_PATCH_SIZE = 16;
static constexpr int VISION_PATCH_GRID = 48;
static constexpr int VISION_POOLING_KERNEL_SIZE = 3;
static constexpr int VISION_PATCH_DIM = VISION_PATCH_SIZE * VISION_PATCH_SIZE * 3;
static constexpr int VISION_BOI_TOKEN_ID = 255999;     // <|image>
static constexpr int VISION_IMAGE_TOKEN_ID = 258880;   // <|image|>
static constexpr int VISION_EOI_TOKEN_ID = 258882;     // <image|>

static constexpr const char* VISION_ENCODER_SIGNATURE_KEY = "vision_70";
static constexpr const char* VISION_ADAPTER_SIGNATURE_KEY = "vision_adapter_70";

// MARK: - Vision data

struct VisionPreprocessResult {
    std::vector<float> patches;
    std::vector<int32_t> positions;
    int input_patches = 0;
    int resized_width = 0;
    int resized_height = 0;
};

struct VisionEmbeddingResult {
    int input_patches = 0;
    int valid_tokens = 0;
    int max_tokens = 0;
    std::vector<uint8_t> mask;
    std::vector<float> embeddings;  // [max_tokens, EMBEDDING_DIM]
    std::vector<float> end_embedding;  // [EMBEDDING_DIM]
};

struct VisionState {
    LiteRtEnvironment env = nullptr;
    LiteRtModel encoder_model = nullptr;
    LiteRtModel adapter_model = nullptr;
    LiteRtModel eoi_model = nullptr;
    LiteRtCompiledModel encoder_cm = nullptr;
    LiteRtCompiledModel adapter_cm = nullptr;
    LiteRtCompiledModel eoi_cm = nullptr;
    LiteRtOptions encoder_options = nullptr;
    LiteRtOptions adapter_options = nullptr;
    LiteRtOptions eoi_options = nullptr;

    SignatureInfo encoder_sig;
    SignatureInfo adapter_sig;
    SignatureInfo eoi_sig;

    LiteRtTensorBuffer encoder_images_buf = nullptr;
    LiteRtTensorBuffer encoder_positions_buf = nullptr;
    LiteRtTensorBuffer encoder_mask_buf = nullptr;
    LiteRtTensorBuffer encoder_features_buf = nullptr;
    LiteRtTensorBuffer adapter_input_buf = nullptr;
    LiteRtTensorBuffer adapter_output_buf = nullptr;
    LiteRtTensorBuffer eoi_output_buf = nullptr;

    std::vector<LiteRtTensorBuffer> encoder_inputs;
    std::vector<LiteRtTensorBuffer> encoder_outputs;
    std::vector<LiteRtTensorBuffer> adapter_inputs;
    std::vector<LiteRtTensorBuffer> adapter_outputs;
    std::vector<LiteRtTensorBuffer> eoi_inputs;
    std::vector<LiteRtTensorBuffer> eoi_outputs;

    int max_patches = 0;
    int max_tokens = 0;
    int image_token_budget = 0;
    int encoder_dim = VISION_ENCODER_DIM;

    bool initialized = false;
};

// MARK: - Vision initialization

struct VisionComponentTargets {
    HardwareTarget encoder = HardwareTarget::CPU;
    HardwareTarget adapter = HardwareTarget::CPU;
    HardwareTarget end_of_vision = HardwareTarget::CPU;
};

static inline VisionComponentTargets vision_component_targets(HardwareTarget encoder_hw) {
    return VisionComponentTargets{
        encoder_hw,
        HardwareTarget::CPU,
        HardwareTarget::CPU,
    };
}

static inline std::string vision_components(
        const VisionComponentTargets& targets,
        bool has_end_of_vision) {
    std::string components =
        std::string("vision_encoder(") + hw_target_name(targets.encoder) +
        ")+vision_adapter(" + hw_target_name(targets.adapter) + ")";
    if (has_end_of_vision)
        components += std::string("+end_of_vision(") + hw_target_name(targets.end_of_vision) + ")";
    return components;
}

static inline std::string vision_components(HardwareTarget encoder_hw, bool has_end_of_vision) {
    return vision_components(vision_component_targets(encoder_hw), has_end_of_vision);
}

static inline std::string vision_runtime_components(
        HardwareTarget decoder_hw,
        HardwareTarget encoder_hw,
        bool has_end_of_vision) {
    return text_components(decoder_hw) + "+" +
           vision_components(encoder_hw, has_end_of_vision);
}

static inline std::string vision_runtime_components_from_inference_engine(
        HardwareTarget encoder_hw,
        bool has_end_of_vision) {
    return std::string("decoder(inference_engine)+embedder(inference_engine)+ple(inference_engine)+") +
           vision_components(encoder_hw, has_end_of_vision);
}

static inline bool bundle_has_end_of_vision(const MappedFile& bundle) {
    auto sections = parse_litertlm_bundle(bundle.data, bundle.size);
    return find_bundle_section(sections, BundleModelType::END_OF_VISION) != nullptr;
}

static inline bool tensor_dim(const LiteRtRankedTensorType& tensor,
                              int axis,
                              int& value) {
    if (axis < 0 || axis >= static_cast<int>(tensor.layout.rank))
        return false;
    value = tensor.layout.dimensions[axis];
    return value > 0;
}

static inline bool trailing_tensor_dim(const LiteRtRankedTensorType& tensor,
                                       int from_end,
                                       int& value) {
    if (from_end <= 0)
        return false;
    return tensor_dim(tensor, static_cast<int>(tensor.layout.rank) - from_end, value);
}

static inline bool configure_vision_shapes(VisionState* s,
                                           int images_idx,
                                           int positions_idx,
                                           int mask_idx,
                                           int features_idx,
                                           const char* path) {
    if (!s) return false;

    LiteRtRankedTensorType images_type = {};
    LiteRtRankedTensorType positions_type = {};
    LiteRtRankedTensorType mask_type = {};
    LiteRtRankedTensorType features_type = {};
    LiteRtRankedTensorType adapter_input_type = {};
    LiteRtRankedTensorType adapter_output_type = {};
    if (!get_input_tensor_type(s->encoder_sig.sig, images_idx, &images_type) ||
        !get_input_tensor_type(s->encoder_sig.sig, positions_idx, &positions_type) ||
        !get_output_tensor_type(s->encoder_sig.sig, mask_idx, &mask_type) ||
        !get_output_tensor_type(s->encoder_sig.sig, features_idx, &features_type) ||
        !get_input_tensor_type(s->adapter_sig.sig, 0, &adapter_input_type) ||
        !get_output_tensor_type(s->adapter_sig.sig, 0, &adapter_output_type))
        return false;

    int image_patches = 0;
    int image_patch_dim = 0;
    int position_patches = 0;
    int position_dim = 0;
    int mask_tokens = 0;
    int feature_tokens = 0;
    int feature_dim = 0;
    int adapter_input_tokens = 0;
    int adapter_input_dim = 0;
    int adapter_output_tokens = 0;
    int adapter_output_dim = 0;
    if (!trailing_tensor_dim(images_type, 2, image_patches) ||
        !trailing_tensor_dim(images_type, 1, image_patch_dim) ||
        !trailing_tensor_dim(positions_type, 2, position_patches) ||
        !trailing_tensor_dim(positions_type, 1, position_dim) ||
        !trailing_tensor_dim(mask_type, 1, mask_tokens) ||
        !trailing_tensor_dim(features_type, 2, feature_tokens) ||
        !trailing_tensor_dim(features_type, 1, feature_dim) ||
        !trailing_tensor_dim(adapter_input_type, 2, adapter_input_tokens) ||
        !trailing_tensor_dim(adapter_input_type, 1, adapter_input_dim) ||
        !trailing_tensor_dim(adapter_output_type, 2, adapter_output_tokens) ||
        !trailing_tensor_dim(adapter_output_type, 1, adapter_output_dim)) {
        tinylog::logger().error("Gemma4 VLM signature has dynamic or unexpected rank",
            {{"path", std::string(path ? path : "")},
             {"encoder_signature", s->encoder_sig.key},
             {"adapter_signature", s->adapter_sig.key}});
        return false;
    }

    const int expected_patch_dim = VISION_PATCH_DIM;
    if (image_patches != position_patches ||
        image_patch_dim != expected_patch_dim ||
        position_dim != 2 ||
        mask_tokens != feature_tokens ||
        mask_tokens != adapter_input_tokens ||
        mask_tokens != adapter_output_tokens ||
        feature_dim != VISION_ENCODER_DIM ||
        adapter_input_dim != VISION_ENCODER_DIM ||
        adapter_output_dim != EMBEDDING_DIM ||
        image_patches != mask_tokens * VISION_POOLING_KERNEL_SIZE * VISION_POOLING_KERNEL_SIZE) {
        tinylog::logger().error("Gemma4 VLM signature shape mismatch",
            {{"path", std::string(path ? path : "")},
             {"encoder_signature", s->encoder_sig.key},
             {"adapter_signature", s->adapter_sig.key},
             {"image_patches", (int64_t)image_patches},
             {"image_patch_dim", (int64_t)image_patch_dim},
             {"position_patches", (int64_t)position_patches},
             {"position_dim", (int64_t)position_dim},
             {"mask_tokens", (int64_t)mask_tokens},
             {"feature_tokens", (int64_t)feature_tokens},
             {"feature_dim", (int64_t)feature_dim},
             {"adapter_input_tokens", (int64_t)adapter_input_tokens},
             {"adapter_input_dim", (int64_t)adapter_input_dim},
             {"adapter_output_tokens", (int64_t)adapter_output_tokens},
             {"adapter_output_dim", (int64_t)adapter_output_dim}});
        return false;
    }

    s->max_patches = image_patches;
    s->max_tokens = mask_tokens;
    s->image_token_budget = std::min(VISION_DEFAULT_IMAGE_TOKEN_BUDGET, s->max_tokens);
    s->encoder_dim = feature_dim;
    tinylog::logger().info("Gemma4 VLM: signature shapes",
        {{"path", std::string(path ? path : "")},
         {"encoder_signature", s->encoder_sig.key},
         {"adapter_signature", s->adapter_sig.key},
         {"max_patches", (int64_t)s->max_patches},
         {"max_tokens", (int64_t)s->max_tokens},
         {"image_token_budget", (int64_t)s->image_token_budget},
         {"encoder_dim", (int64_t)s->encoder_dim},
         {"embedding_dim", (int64_t)EMBEDDING_DIM}});
    return true;
}

static inline bool init_vision(
        VisionState* s,
        LiteRtEnvironment env,
        const MappedFile& bundle,
        Platform platform,
        HardwareTarget encoder_hw,
        int num_threads,
        const char* compilation_cache_dir = nullptr,
        bool log_initializing = true) {
    if (!s) return false;
    auto sections = parse_litertlm_bundle(bundle.data, bundle.size);
    auto* encoder_section = find_bundle_section(sections, BundleModelType::VISION_ENCODER);
    auto* adapter_section = find_bundle_section(sections, BundleModelType::VISION_ADAPTER);
    auto* eoi_section = find_bundle_section(sections, BundleModelType::END_OF_VISION);
    if (!encoder_section || !adapter_section) {
        tinylog::logger().error("Gemma4 vision sections not found", {{"path", bundle.path}});
        return false;
    }

    if (log_initializing) {
        tinylog::logger().info("Gemma4 VLM: initializing",
            {{"path", bundle.path},
             {"components", vision_components(encoder_hw, eoi_section != nullptr)},
             {"num_threads", (int64_t)resolve_num_threads(num_threads)}});
    }

    const VisionComponentTargets targets = vision_component_targets(encoder_hw);
    s->env = env;
    if (!litert_check(LITERT(LiteRtCreateModelFromBuffer)(
            env, encoder_section->data, encoder_section->size, &s->encoder_model),
            "LiteRtCreateModelFromBuffer(gemma4/vision_encoder)") ||
        !litert_check(LITERT(LiteRtCreateModelFromBuffer)(
            env, adapter_section->data, adapter_section->size, &s->adapter_model),
            "LiteRtCreateModelFromBuffer(gemma4/vision_adapter)"))
        return false;
    if (eoi_section &&
        !litert_check(LITERT(LiteRtCreateModelFromBuffer)(
            env, eoi_section->data, eoi_section->size, &s->eoi_model),
            "LiteRtCreateModelFromBuffer(gemma4/end_of_vision)"))
        return false;

    s->encoder_options = create_litert_options(
        platform, targets.encoder, ModelFamily::GEMMA4_VISION_ENCODER, num_threads,
        "vision_encoder", bundle.path.c_str(), compilation_cache_dir);
    s->adapter_options = create_litert_options(
        platform, targets.adapter, ModelFamily::GEMMA4, num_threads,
        "vision_adapter", bundle.path.c_str(), compilation_cache_dir);
    if (s->eoi_model) {
        s->eoi_options = create_litert_options(
            platform, targets.end_of_vision, ModelFamily::GEMMA4, num_threads,
            "end_of_vision", bundle.path.c_str(), compilation_cache_dir);
    }
    if (!s->encoder_options || !s->adapter_options)
        return false;
    if (s->eoi_model && !s->eoi_options)
        return false;

    if (!discover_signature_by_key(s->encoder_model, VISION_ENCODER_SIGNATURE_KEY, s->encoder_sig) ||
        !discover_signature_by_key(s->adapter_model, VISION_ADAPTER_SIGNATURE_KEY, s->adapter_sig))
        return false;
    if (s->eoi_model && !discover_signature(s->eoi_model, 0, s->eoi_sig))
        return false;

    int images_idx = s->encoder_sig.input_index_of("images");
    int positions_idx = s->encoder_sig.input_index_of("positions_xy");
    int mask_idx = s->encoder_sig.output_index_of("mask");
    int features_idx = s->encoder_sig.output_index_of("features");
    if (images_idx < 0 || positions_idx < 0 || mask_idx < 0 || features_idx < 0) {
        tinylog::logger().error("Gemma4 vision encoder signature mismatch");
        return false;
    }
    if (!configure_vision_shapes(s, images_idx, positions_idx, mask_idx, features_idx,
                                 bundle.path.c_str()))
        return false;

    tinylog::logger().info("Gemma4 VLM: compiling vision_encoder",
        {{"path", bundle.path}, {"hw", std::string(hw_target_name(targets.encoder))}});
    log_metal_memory_snapshot("before gemma4/vision_encoder compile");
    const bool encoder_compile_ok = create_compiled_model_with_magic(
        env, s->encoder_model, s->encoder_options, &s->encoder_cm,
        nullptr,
        "LiteRtCreateCompiledModel(gemma4/vision_encoder)");
    log_metal_memory_snapshot(encoder_compile_ok
                                  ? "after gemma4/vision_encoder compile"
                                  : "failed gemma4/vision_encoder compile");
    if (!encoder_compile_ok)
        return false;

    tinylog::logger().info("Gemma4 VLM: compiling vision_adapter",
        {{"path", bundle.path}, {"hw", std::string(hw_target_name(targets.adapter))}});
    if (!create_compiled_model_with_magic(
            env, s->adapter_model, s->adapter_options, &s->adapter_cm,
            nullptr,
            "LiteRtCreateCompiledModel(gemma4/vision_adapter)"))
        return false;

    if (s->eoi_model) {
        tinylog::logger().info("Gemma4 VLM: compiling end_of_vision",
            {{"path", bundle.path}, {"hw", std::string(hw_target_name(targets.end_of_vision))}});
        if (!create_compiled_model_with_magic(
            env, s->eoi_model, s->eoi_options, &s->eoi_cm,
            nullptr,
            "LiteRtCreateCompiledModel(gemma4/end_of_vision)"))
            return false;
    }

    s->encoder_images_buf = alloc_managed_input(
        env, s->encoder_cm, s->encoder_sig, images_idx, "vision/images");
    s->encoder_positions_buf = alloc_managed_input(
        env, s->encoder_cm, s->encoder_sig, positions_idx, "vision/positions_xy");
    s->encoder_mask_buf = alloc_managed_output(
        env, s->encoder_cm, s->encoder_sig, mask_idx, "vision/mask");
    s->encoder_features_buf = alloc_managed_output(
        env, s->encoder_cm, s->encoder_sig, features_idx, "vision/features");
    s->adapter_input_buf = alloc_managed_input(
        env, s->adapter_cm, s->adapter_sig, 0, "vision_adapter/soft_tokens");
    s->adapter_output_buf = alloc_managed_output(
        env, s->adapter_cm, s->adapter_sig, 0, "vision_adapter/mm_embedding");
    if (s->eoi_cm) {
        int eoi_output_idx = s->eoi_sig.output_index_of("eoi_embedding");
        if (eoi_output_idx < 0 && s->eoi_sig.num_outputs > 0)
            eoi_output_idx = 0;
        if (eoi_output_idx < 0)
            return false;
        s->eoi_output_buf = alloc_managed_output(
            env, s->eoi_cm, s->eoi_sig, eoi_output_idx, "end_of_vision/eoi_embedding");
    }
    if (!s->encoder_images_buf || !s->encoder_positions_buf || !s->encoder_mask_buf ||
        !s->encoder_features_buf || !s->adapter_input_buf || !s->adapter_output_buf)
        return false;
    if (s->eoi_cm && !s->eoi_output_buf)
        return false;

    s->encoder_inputs.assign(s->encoder_sig.num_inputs, nullptr);
    s->encoder_inputs[images_idx] = s->encoder_images_buf;
    s->encoder_inputs[positions_idx] = s->encoder_positions_buf;
    s->encoder_outputs.assign(s->encoder_sig.num_outputs, nullptr);
    s->encoder_outputs[mask_idx] = s->encoder_mask_buf;
    s->encoder_outputs[features_idx] = s->encoder_features_buf;
    s->adapter_inputs.assign(1, s->adapter_input_buf);
    s->adapter_outputs.assign(1, s->adapter_output_buf);
    if (s->eoi_cm) {
        s->eoi_inputs.assign(s->eoi_sig.num_inputs, nullptr);
        s->eoi_outputs.assign(s->eoi_sig.num_outputs, nullptr);
        int eoi_output_idx = s->eoi_sig.output_index_of("eoi_embedding");
        if (eoi_output_idx < 0 && s->eoi_sig.num_outputs > 0)
            eoi_output_idx = 0;
        s->eoi_outputs[eoi_output_idx] = s->eoi_output_buf;
    }

    s->initialized = true;
    tinylog::logger().info("Gemma4 VLM: initialized",
        {{"path", bundle.path},
         {"components", s->eoi_cm
            ? std::string("vision_encoder(") + hw_target_name(encoder_hw) + ")+vision_adapter(cpu)+end_of_vision(cpu)"
            : std::string("vision_encoder(") + hw_target_name(encoder_hw) + ")+vision_adapter(cpu)" }});
    return true;
}

// MARK: - Vision cleanup

static inline void destroy_vision(VisionState* s) {
    if (!s) return;
    if (s->encoder_cm) LITERT(LiteRtDestroyCompiledModel)(s->encoder_cm);
    if (s->adapter_cm) LITERT(LiteRtDestroyCompiledModel)(s->adapter_cm);
    if (s->eoi_cm) LITERT(LiteRtDestroyCompiledModel)(s->eoi_cm);
    s->encoder_cm = s->adapter_cm = s->eoi_cm = nullptr;
    drain_active_litert_metal_queue();

    destroy_buffer(s->encoder_images_buf);
    destroy_buffer(s->encoder_positions_buf);
    destroy_buffer(s->encoder_mask_buf);
    destroy_buffer(s->encoder_features_buf);
    destroy_buffer(s->adapter_input_buf);
    destroy_buffer(s->adapter_output_buf);
    destroy_buffer(s->eoi_output_buf);
    if (s->encoder_options) LITERT(LiteRtDestroyOptions)(s->encoder_options);
    if (s->adapter_options) LITERT(LiteRtDestroyOptions)(s->adapter_options);
    if (s->eoi_options) LITERT(LiteRtDestroyOptions)(s->eoi_options);
    if (s->encoder_model) LITERT(LiteRtDestroyModel)(s->encoder_model);
    if (s->adapter_model) LITERT(LiteRtDestroyModel)(s->adapter_model);
    if (s->eoi_model) LITERT(LiteRtDestroyModel)(s->eoi_model);
    *s = VisionState{};
}

static inline void gemma4_vision_target_size(
        int width,
        int height,
        int image_token_budget,
        int& target_width,
        int& target_height) {
    const int max_patches = image_token_budget * VISION_POOLING_KERNEL_SIZE * VISION_POOLING_KERNEL_SIZE;
    const int num_patches_h = height / VISION_PATCH_SIZE;
    const int num_patches_w = width / VISION_PATCH_SIZE;
    const int num_patches = num_patches_h * num_patches_w;

    // Match LiteRT-LM's StbImagePreprocessor patchify resize policy.
    target_height = height;
    target_width = width;
    if (num_patches > max_patches) {
        const double factor = std::sqrt(static_cast<double>(max_patches) /
                                        static_cast<double>(num_patches));
        target_height = static_cast<int>(height * factor);
        target_width = static_cast<int>(width * factor);
    }

    target_height = (target_height / VISION_PATCH_SIZE) * VISION_PATCH_SIZE;
    target_width = (target_width / VISION_PATCH_SIZE) * VISION_PATCH_SIZE;
    target_height = std::max(target_height, VISION_PATCH_SIZE);
    target_width = std::max(target_width, VISION_PATCH_SIZE);

    while ((target_height / VISION_PATCH_SIZE) * (target_width / VISION_PATCH_SIZE) > max_patches) {
        if (target_height >= target_width) {
            target_height -= VISION_PATCH_SIZE;
        } else {
            target_width -= VISION_PATCH_SIZE;
        }
        target_height = std::max(target_height, VISION_PATCH_SIZE);
        target_width = std::max(target_width, VISION_PATCH_SIZE);
    }

    if (target_width <= 0 || target_height <= 0)
        throw std::runtime_error("Gemma4 VLM invalid target image size");
}

static inline int patchify_rgb_image(
        const float* rgb,
        int width,
        int height,
        int max_patches,
        std::vector<float>& patches,
        std::vector<int32_t>& positions) {
    if (width <= 0 || height <= 0 ||
        width % VISION_PATCH_SIZE != 0 ||
        height % VISION_PATCH_SIZE != 0)
        throw std::runtime_error("Gemma4 VLM patchify expects dimensions divisible by 16");

    const int patch_width = width / VISION_PATCH_SIZE;
    const int patch_height = height / VISION_PATCH_SIZE;
    const int num_patches = patch_width * patch_height;
    if (num_patches <= 0 || num_patches > max_patches)
        throw std::runtime_error("Gemma4 VLM patch count exceeds model input budget");

    patches.assign(max_patches * VISION_PATCH_DIM, 0.0f);
    positions.assign(max_patches * 2, -1);

    for (int row = 0; row < patch_height; row++) {
        for (int col = 0; col < patch_width; col++) {
            int patch_idx = row * patch_width + col;
            positions[patch_idx * 2] = col;
            positions[patch_idx * 2 + 1] = row;

            for (int ph = 0; ph < VISION_PATCH_SIZE; ph++) {
                for (int pw = 0; pw < VISION_PATCH_SIZE; pw++) {
                    for (int c = 0; c < 3; c++) {
                        int src_y = row * VISION_PATCH_SIZE + ph;
                        int src_x = col * VISION_PATCH_SIZE + pw;
                        int src_idx = (src_y * width + src_x) * 3 + c;
                        int dst_idx = patch_idx * VISION_PATCH_DIM +
                                      (ph * VISION_PATCH_SIZE + pw) * 3 + c;
                        patches[dst_idx] = rgb[src_idx];
                    }
                }
            }
        }
    }
    return num_patches;
}

static inline float sample_rgb888_bilinear(
        const uint8_t* rgb,
        int width,
        int height,
        int row_stride,
        double x,
        double y,
        int channel) {
    x = std::max(0.0, std::min(x, static_cast<double>(width - 1)));
    y = std::max(0.0, std::min(y, static_cast<double>(height - 1)));
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = std::min(x0 + 1, width - 1);
    int y1 = std::min(y0 + 1, height - 1);
    double wx = x - x0;
    double wy = y - y0;

    auto pixel = [&](int px, int py) -> double {
        return static_cast<double>(rgb[py * row_stride + px * 3 + channel]) / 255.0;
    };
    double top = pixel(x0, y0) * (1.0 - wx) + pixel(x1, y0) * wx;
    double bottom = pixel(x0, y1) * (1.0 - wx) + pixel(x1, y1) * wx;
    return static_cast<float>(top * (1.0 - wy) + bottom * wy);
}

static inline VisionPreprocessResult preprocess_rgb888_image(
        const uint8_t* rgb,
        int width,
        int height,
        int row_stride,
        int max_patches,
        int image_token_budget) {
    if (!rgb || width <= 0 || height <= 0 || row_stride < width * 3)
        throw std::runtime_error("Gemma4 VLM invalid RGB888 image input");
    if (max_patches <= 0 || image_token_budget <= 0)
        throw std::runtime_error("Gemma4 VLM invalid signature dimensions");

    VisionPreprocessResult result;
    gemma4_vision_target_size(width, height, image_token_budget,
                              result.resized_width, result.resized_height);

    std::vector<float> resized(result.resized_width * result.resized_height * 3, 0.0f);
    const double scale_x = static_cast<double>(width) / result.resized_width;
    const double scale_y = static_cast<double>(height) / result.resized_height;
    for (int y = 0; y < result.resized_height; y++) {
        double src_y = (static_cast<double>(y) + 0.5) * scale_y - 0.5;
        for (int x = 0; x < result.resized_width; x++) {
            double src_x = (static_cast<double>(x) + 0.5) * scale_x - 0.5;
            for (int c = 0; c < 3; c++) {
                resized[(y * result.resized_width + x) * 3 + c] =
                    sample_rgb888_bilinear(rgb, width, height, row_stride, src_x, src_y, c);
            }
        }
    }

    result.input_patches = patchify_rgb_image(
        resized.data(), result.resized_width, result.resized_height, max_patches,
        result.patches, result.positions);
    return result;
}

static inline bool encode_image_patches(
        VisionState* s,
        const std::vector<float>& patches,
        const std::vector<int32_t>& positions,
        int input_patches,
        VisionEmbeddingResult& result) {
    if (!s || !s->initialized) return false;
    if (s->max_patches <= 0 || s->max_tokens <= 0 || s->encoder_dim <= 0 ||
        (int)patches.size() != s->max_patches * VISION_PATCH_DIM ||
        (int)positions.size() != s->max_patches * 2 ||
        input_patches <= 0 ||
        input_patches > s->max_patches) {
        tinylog::logger().error("Gemma4 VLM: invalid patch input");
        return false;
    }

    if (!write_float_buf(s->encoder_images_buf, patches.data(), patches.size()))
        return false;
    if (!write_int_buf(s->encoder_positions_buf, positions.data(), positions.size()))
        return false;

    if (!litert_check(LITERT(LiteRtRunCompiledModel)(
            s->encoder_cm, s->encoder_sig.sig_index,
            s->encoder_inputs.size(), s->encoder_inputs.data(),
            s->encoder_outputs.size(), s->encoder_outputs.data()),
            "RunCompiledModel(gemma4/vision_encoder)"))
        return false;

    result.mask.assign(s->max_tokens, 0);
    if (!read_u8_buf(s->encoder_mask_buf, result.mask.data(), result.mask.size()))
        return false;
    result.valid_tokens = 0;
    result.max_tokens = s->max_tokens;
    for (auto value : result.mask) {
        if (value) result.valid_tokens++;
    }

    std::vector<float> features(s->max_tokens * s->encoder_dim, 0.0f);
    if (!read_float_buf(s->encoder_features_buf, features.data(), features.size()))
        return false;
    if (!write_float_buf(s->adapter_input_buf, features.data(), features.size()))
        return false;

    if (!litert_check(LITERT(LiteRtRunCompiledModel)(
            s->adapter_cm, s->adapter_sig.sig_index,
            s->adapter_inputs.size(), s->adapter_inputs.data(),
            s->adapter_outputs.size(), s->adapter_outputs.data()),
            "RunCompiledModel(gemma4/vision_adapter)"))
        return false;

    result.input_patches = input_patches;
    result.embeddings.assign(s->max_tokens * EMBEDDING_DIM, 0.0f);
    if (!read_float_buf(s->adapter_output_buf, result.embeddings.data(), result.embeddings.size()))
        return false;
    if (s->eoi_cm) {
        LiteRtTensorBuffer* eoi_inputs = s->eoi_inputs.empty() ? nullptr : s->eoi_inputs.data();
        if (!litert_check(LITERT(LiteRtRunCompiledModel)(
                s->eoi_cm, s->eoi_sig.sig_index,
                s->eoi_inputs.size(), eoi_inputs,
                s->eoi_outputs.size(), s->eoi_outputs.data()),
                "RunCompiledModel(gemma4/end_of_vision)"))
            return false;
        result.end_embedding.assign(EMBEDDING_DIM, 0.0f);
        if (!read_float_buf(s->eoi_output_buf, result.end_embedding.data(), result.end_embedding.size()))
            return false;
    }
    return true;
}

static inline bool encode_rgb888_image(
        VisionState* s,
        const uint8_t* rgb,
        int width,
        int height,
        int row_stride,
        VisionEmbeddingResult& result,
        VisionPreprocessResult* preprocess_out = nullptr) {
    if (!s || !s->initialized) return false;
    VisionPreprocessResult preprocess = preprocess_rgb888_image(
        rgb, width, height, row_stride, s->max_patches, s->image_token_budget);
    bool ok = encode_image_patches(
        s, preprocess.patches, preprocess.positions, preprocess.input_patches, result);
    if (ok && preprocess_out)
        *preprocess_out = std::move(preprocess);
    return ok;
}

static inline bool encode_synthetic_image_patches(
        VisionState* s,
        int input_patches,
        VisionEmbeddingResult& result) {
    if (!s || input_patches <= 0 || input_patches > s->max_patches)
        return false;

    std::vector<float> patches(s->max_patches * VISION_PATCH_DIM, 0.0f);
    std::vector<int32_t> positions(s->max_patches * 2, -1);
    for (int i = 0; i < input_patches; i++) {
        positions[i * 2] = i % VISION_PATCH_GRID;
        positions[i * 2 + 1] = i / VISION_PATCH_GRID;
    }
    return encode_image_patches(s, patches, positions, input_patches, result);
}

struct VisionSmokeResult {
    int input_patches = 0;
    int valid_vision_tokens = 0;
    int embedding_dim = 0;
    float first_embedding_value = 0.0f;
    float last_valid_embedding_value = 0.0f;
};

struct VisionTextSmokeResult {
    int input_patches = 0;
    int valid_vision_tokens = 0;
    int image_token_slots = 0;
    int resized_width = 0;
    int resized_height = 0;
    int prompt_tokens = 0;
    int prefill_tokens = 0;
    int decode_steps = 0;
    int first_decode_ms = 0;
    std::string response;
};

struct Runtime {
    MappedFile bundle;
    LiteRtEnvironment env = nullptr;
    bool owns_env = false;
    Tokenizer tokenizer;
    State text;
    VisionState vision;
    bool opened = false;
    bool text_initialized = false;
    bool vision_initialized = false;
};

static inline int vision_environment_accelerator_mask(HardwareTarget encoder_hw,
                                                      HardwareTarget decoder_hw) {
    const VisionComponentTargets vision_targets = vision_component_targets(encoder_hw);
    return environment_accelerator_mask(decoder_hw) |
           litert_options_accelerator_mask(vision_targets.encoder,
                                           ModelFamily::GEMMA4_VISION_ENCODER) |
           litert_options_accelerator_mask(vision_targets.adapter, ModelFamily::GEMMA4) |
           litert_options_accelerator_mask(vision_targets.end_of_vision, ModelFamily::GEMMA4);
}

static inline std::vector<int> build_vision_prompt_token_ids(
        Tokenizer& tokenizer,
        const std::string& prompt,
        int image_token_slots) {
    std::vector<int> ids = tokenizer.encode("<bos><|turn>user\n");
    ids.push_back(VISION_BOI_TOKEN_ID);
    for (int i = 0; i < image_token_slots; i++)
        ids.push_back(VISION_IMAGE_TOKEN_ID);
    ids.push_back(VISION_EOI_TOKEN_ID);

    auto suffix = tokenizer.encode("\n" + prompt + "<turn|>\n<|turn>model\n");
    ids.insert(ids.end(), suffix.begin(), suffix.end());
    return ids;
}

static inline bool build_vision_prompt_embeddings(
        State* llm,
        const VisionEmbeddingResult& vision,
        const std::string& prompt,
        std::vector<int>& token_ids,
        std::vector<float>& embeddings,
        std::vector<float>& per_layer_embeddings) {
    if (!llm || !llm->tokenizer)
        return false;
    if (vision.max_tokens <= 0 ||
        (int)vision.embeddings.size() < vision.max_tokens * EMBEDDING_DIM)
        return false;

    auto& tokenizer = *llm->tokenizer;
    int image_token_slots = vision.valid_tokens;
    if (image_token_slots <= 0 || image_token_slots > vision.max_tokens)
        return false;

    token_ids = build_vision_prompt_token_ids(tokenizer, prompt, image_token_slots);
    if (token_ids.empty())
        return false;

    embeddings.assign(token_ids.size() * EMBEDDING_DIM, 0.0f);
    per_layer_embeddings.assign(token_ids.size() * PLE_FLOATS, 0.0f);

    std::vector<float> token_embedding(EMBEDDING_DIM);
    std::vector<float> token_ple(PLE_FLOATS);
    std::vector<float> pad_ple(PLE_FLOATS);
    if (!lookup_ple(llm, /*<pad>*/ 0, pad_ple.data()))
        return false;

    int image_index = 0;
    for (size_t i = 0; i < token_ids.size(); i++) {
        float* embedding_dst = embeddings.data() + i * EMBEDDING_DIM;
        float* ple_dst = per_layer_embeddings.data() + i * PLE_FLOATS;

        if (token_ids[i] == VISION_IMAGE_TOKEN_ID && image_index < image_token_slots) {
            memcpy(embedding_dst,
                   vision.embeddings.data() + image_index * EMBEDDING_DIM,
                   EMBEDDING_DIM * sizeof(float));
            memcpy(ple_dst, pad_ple.data(), PLE_FLOATS * sizeof(float));
            image_index++;
            continue;
        }

        if (token_ids[i] == VISION_EOI_TOKEN_ID && !vision.end_embedding.empty()) {
            memcpy(embedding_dst, vision.end_embedding.data(), EMBEDDING_DIM * sizeof(float));
            memcpy(ple_dst, pad_ple.data(), PLE_FLOATS * sizeof(float));
            continue;
        }

        if (!lookup_embedding(llm, token_ids[i], token_embedding.data()))
            return false;
        if (!lookup_ple(llm, token_ids[i], token_ple.data()))
            return false;
        memcpy(embedding_dst, token_embedding.data(), EMBEDDING_DIM * sizeof(float));
        memcpy(ple_dst, token_ple.data(), PLE_FLOATS * sizeof(float));
    }

    return image_index == image_token_slots;
}

static inline bool generate_vision_text(
        State* llm,
        Tokenizer& tokenizer,
        const VisionEmbeddingResult& vision,
        const std::string& prompt,
        int max_tokens,
        VisionTextSmokeResult& result) {
    std::vector<int> token_ids;
    std::vector<float> embeddings;
    std::vector<float> per_layer_embeddings;
    if (!build_vision_prompt_embeddings(
            llm, vision, prompt, token_ids, embeddings, per_layer_embeddings))
        return false;

    result.input_patches = vision.input_patches;
    result.valid_vision_tokens = vision.valid_tokens;
    result.image_token_slots = vision.valid_tokens;
    result.prompt_tokens = (int)token_ids.size();
    const int remaining_context_tokens = std::max(
        0,
        llm->kv_cache_max_len - result.prompt_tokens);
    const int effective_max_tokens = max_tokens > 0
        ? std::min(max_tokens, remaining_context_tokens)
        : remaining_context_tokens;
    if (effective_max_tokens <= 0) {
        tinylog::logger().error("Gemma4 VLM prompt leaves no output context",
            {{"prompt_tokens", (int64_t)result.prompt_tokens},
             {"kv_cache_max_len", (int64_t)llm->kv_cache_max_len}});
        return false;
    }
    if (max_tokens > remaining_context_tokens) {
        tinylog::logger().warn("Gemma4 VLM max tokens clamped to remaining context",
            {{"requested_max_tokens", (int64_t)max_tokens},
             {"effective_max_tokens", (int64_t)effective_max_tokens},
             {"prompt_tokens", (int64_t)result.prompt_tokens},
             {"kv_cache_max_len", (int64_t)llm->kv_cache_max_len}});
    }

    using clock = std::chrono::steady_clock;
    auto decode_start = clock::now();
    reset_kv_cache(llm);
    int pending = prefill_embeddings_all(
        llm, embeddings, per_layer_embeddings, (int)token_ids.size(), token_ids.back());
    if (pending < 0)
        return false;
    result.prefill_tokens = (int)token_ids.size() - 1;

    std::vector<float> logits(VOCAB_SIZE);
    std::vector<int> output_ids;
    int current = pending;
    for (int step = 0; step < effective_max_tokens; step++) {
        if (!run_decode_step(llm, current, logits.data()))
            break;
        if (step == 0) {
            result.first_decode_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                clock::now() - decode_start).count();
        }
        int next = sample_topk_topp(logits.data(), VOCAB_SIZE,
                                    /*temperature=*/0.0f, /*top_k=*/1, /*top_p=*/1.0f,
                                    llm->rng);
        if (STOP_TOKENS.count(next)) break;
        output_ids.push_back(next);
        current = next;
    }
    result.response = tokenizer.decode(output_ids);
    result.decode_steps = (int)output_ids.size();
    return true;
}

static inline void destroy_runtime(Runtime* runtime) {
    if (!runtime) return;
    destroy_vision(&runtime->vision);
    destroy_buffers(&runtime->text);
    if (runtime->env && runtime->owns_env) {
        LITERT(LiteRtDestroyEnvironment)(runtime->env);
    }
    runtime->env = nullptr;
    runtime->owns_env = false;
    runtime->bundle.close();
    runtime->opened = false;
    runtime->text_initialized = false;
    runtime->vision_initialized = false;
}

static inline bool open_runtime(
        Runtime* runtime,
        const char* path,
        LiteRtEnvironment shared_env) {
    if (!runtime || !path)
        return false;

    destroy_runtime(runtime);

    if (!runtime->bundle.open(path))
        return false;

    auto [tok_data, tok_size] = find_bundle_tokenizer(runtime->bundle.data, runtime->bundle.size);
    if (!tok_data || !runtime->tokenizer.load_from_buffer(tok_data, tok_size)) {
        destroy_runtime(runtime);
        return false;
    }

    runtime->env = shared_env;
    runtime->owns_env = false;
    runtime->text.tokenizer = &runtime->tokenizer;
    runtime->opened = true;
    return true;
}

static inline bool ensure_text(
        Runtime* runtime,
        Platform platform,
        HardwareTarget decoder_hw,
        int num_threads,
        const char* compilation_cache_dir = nullptr,
        LiteRtMagicNumberConfigs* magic_configs = nullptr,
        Gemma4RuntimeConfig runtime_config = Gemma4RuntimeConfig(),
        Gemma4DiagnosticConfig diagnostic_config = Gemma4DiagnosticConfig(),
        bool log_initializing = true) {
    if (!runtime || !runtime->opened || !runtime->env)
        return false;
    if (runtime->text_initialized)
        return true;

    if (magic_configs)
        runtime->text.magic_configs = magic_configs;
    if (!runtime->text.magic_configs)
        runtime->text.magic_configs = prescan_bundle_magic_configs(runtime->bundle);
    runtime->text.runtime_config = runtime_config;
    runtime->text.diagnostic_config = diagnostic_config;
    apply_magic_configs_to_state(&runtime->text);
    runtime->text.tokenizer = &runtime->tokenizer;

    if (!init(&runtime->text,
              runtime->env,
              runtime->bundle,
              platform,
              decoder_hw,
              num_threads,
              compilation_cache_dir,
              runtime_config,
              diagnostic_config,
              log_initializing))
        return false;

    runtime->text_initialized = true;
    return true;
}

static inline bool ensure_vision(
        Runtime* runtime,
        Platform platform,
        HardwareTarget encoder_hw,
        int num_threads,
        const char* compilation_cache_dir = nullptr,
        bool log_initializing = true) {
    if (!runtime || !runtime->opened || !runtime->env)
        return false;
    if (runtime->vision_initialized)
        return true;

    if (!init_vision(&runtime->vision,
                     runtime->env,
                     runtime->bundle,
                     platform,
                     encoder_hw,
                     num_threads,
                     compilation_cache_dir,
                     log_initializing))
        return false;

    runtime->vision_initialized = true;
    return true;
}

static inline bool init_runtime(
        Runtime* runtime,
        const char* path,
        Platform platform = host_platform(),
        HardwareTarget encoder_hw = HardwareTarget::CPU,
        HardwareTarget decoder_hw = HardwareTarget::CPU,
        int num_threads = 0,
        const char* compilation_cache_dir = nullptr,
        LiteRtEnvironment shared_env = nullptr,
        const LiteRtExternalMetalHandles* metal_handles = nullptr,
        bool log_initializing = true) {
    if (!runtime || !path)
        return false;

    if (!open_runtime(runtime, path, shared_env))
        return false;

    runtime->text.magic_configs = prescan_bundle_magic_configs(runtime->bundle);
    apply_magic_configs_to_state(&runtime->text);

    if (shared_env) {
        runtime->env = shared_env;
        runtime->owns_env = false;
    } else {
        const HardwareTarget environment_hw =
            (encoder_hw == HardwareTarget::GPU || decoder_hw == HardwareTarget::GPU)
                ? HardwareTarget::GPU
                : HardwareTarget::CPU;
        const int environment_accelerators =
            vision_environment_accelerator_mask(encoder_hw, decoder_hw);
        if (!create_environment(&runtime->env,
                                environment_hw,
                                nullptr,
                                environment_accelerators,
                                metal_handles)) {
            destroy_runtime(runtime);
            return false;
        }
        runtime->owns_env = true;
    }

    const bool has_end_of_vision = bundle_has_end_of_vision(runtime->bundle);

    if (log_initializing) {
        tinylog::logger().info("Gemma4 runtime: initializing",
            {{"path", std::string(path)},
             {"env", std::string(shared_env ? "shared" : "owned")},
             {"components", vision_runtime_components(decoder_hw, encoder_hw, has_end_of_vision)},
             {"num_threads", (int64_t)resolve_num_threads(num_threads)}});
    }

    if (!ensure_text(runtime,
                     platform,
                     decoder_hw,
                     num_threads,
                     compilation_cache_dir,
                     /*magic_configs=*/nullptr,
                     Gemma4RuntimeConfig(),
                     Gemma4DiagnosticConfig(),
                     /*log_initializing=*/false)) {
        destroy_runtime(runtime);
        return false;
    }

    if (!ensure_vision(runtime,
                       platform,
                       encoder_hw,
                       num_threads,
                       compilation_cache_dir,
                       /*log_initializing=*/false)) {
        destroy_runtime(runtime);
        return false;
    }

    if (log_initializing) {
        tinylog::logger().info("Gemma4 runtime: initialized",
            {{"path", std::string(path)},
             {"env", std::string(shared_env ? "shared" : "owned")},
             {"components", vision_runtime_components(decoder_hw, encoder_hw, has_end_of_vision)},
             {"num_threads", (int64_t)resolve_num_threads(num_threads)}});
    }
    return true;
}

static inline bool describe_rgb888_with_runtime(
        Runtime* runtime,
        const uint8_t* rgb,
        int width,
        int height,
        int row_stride,
        const std::string& prompt,
        int max_tokens,
        VisionTextSmokeResult& result) {
    if (!runtime || !runtime->text_initialized || !runtime->vision_initialized ||
        !rgb || width <= 0 || height <= 0 || row_stride < width * 3)
        return false;

    VisionEmbeddingResult vision;
    VisionPreprocessResult preprocess;
    if (!encode_rgb888_image(
            &runtime->vision, rgb, width, height, row_stride, vision, &preprocess))
        return false;

    tinylog::logger().info("Gemma4 VLM: image preprocessed",
        {{"source_width", (int64_t)width},
         {"source_height", (int64_t)height},
         {"resized_width", (int64_t)preprocess.resized_width},
         {"resized_height", (int64_t)preprocess.resized_height},
         {"input_patches", (int64_t)vision.input_patches},
         {"image_token_slots", (int64_t)vision.valid_tokens},
         {"image_token_budget", (int64_t)runtime->vision.image_token_budget},
         {"max_image_tokens", (int64_t)vision.max_tokens}});

    if (!generate_vision_text(&runtime->text, runtime->tokenizer, vision, prompt, max_tokens, result))
        return false;
    result.resized_width = preprocess.resized_width;
    result.resized_height = preprocess.resized_height;
    return true;
}

static inline bool describe_rgb888_with_vision_components(
        VisionState* vision_state,
        State* llm_state,
        Tokenizer& tokenizer,
        const uint8_t* rgb,
        int width,
        int height,
        int row_stride,
        const std::string& prompt,
        int max_tokens,
        VisionTextSmokeResult& result) {
    if (!vision_state || !vision_state->initialized || !llm_state ||
        !rgb || width <= 0 || height <= 0 || row_stride < width * 3)
        return false;

    VisionEmbeddingResult vision;
    VisionPreprocessResult preprocess;
    if (!encode_rgb888_image(
            vision_state, rgb, width, height, row_stride, vision, &preprocess))
        return false;

    tinylog::logger().info("Gemma4 VLM: image preprocessed",
        {{"source_width", (int64_t)width},
         {"source_height", (int64_t)height},
         {"resized_width", (int64_t)preprocess.resized_width},
         {"resized_height", (int64_t)preprocess.resized_height},
         {"input_patches", (int64_t)vision.input_patches},
         {"image_token_slots", (int64_t)vision.valid_tokens},
         {"image_token_budget", (int64_t)vision_state->image_token_budget},
         {"max_image_tokens", (int64_t)vision.max_tokens}});

    if (!generate_vision_text(llm_state, tokenizer, vision, prompt, max_tokens, result))
        return false;
    result.resized_width = preprocess.resized_width;
    result.resized_height = preprocess.resized_height;
    return true;
}

static inline VisionSmokeResult run_vision_smoke(
        const char* path,
        int num_patches,
        Platform platform = host_platform()) {
    MappedFile mf;
    if (!mf.open(path))
        throw std::runtime_error(std::string("Failed to open: ") + path);

    LiteRtEnvironment env = nullptr;
    if (!create_environment(&env, HardwareTarget::CPU)) {
        mf.close();
        throw std::runtime_error("Failed to create LiteRT environment");
    }

    VisionState state;
    if (!init_vision(&state, env, mf, platform, HardwareTarget::CPU, /*num_threads=*/0, nullptr)) {
        destroy_vision(&state);
        if (env) LITERT(LiteRtDestroyEnvironment)(env);
        mf.close();
        throw std::runtime_error("Failed to initialize Gemma4 VLM");
    }

    VisionEmbeddingResult embeddings;
    if (!encode_synthetic_image_patches(&state, num_patches, embeddings)) {
        destroy_vision(&state);
        LITERT(LiteRtDestroyEnvironment)(env);
        mf.close();
        throw std::runtime_error("Failed to run Gemma4 VLM smoke");
    }

    VisionSmokeResult smoke;
    smoke.input_patches = embeddings.input_patches;
    smoke.valid_vision_tokens = embeddings.valid_tokens;
    smoke.embedding_dim = EMBEDDING_DIM;
    smoke.first_embedding_value =
        embeddings.embeddings.empty() ? 0.0f : embeddings.embeddings[0];
    smoke.last_valid_embedding_value =
        embeddings.valid_tokens > 0
            ? embeddings.embeddings[(embeddings.valid_tokens - 1) * EMBEDDING_DIM]
            : 0.0f;

    destroy_vision(&state);
    LITERT(LiteRtDestroyEnvironment)(env);
    mf.close();
    return smoke;
}

static inline VisionTextSmokeResult run_vision_text_smoke(
        const char* path,
        const std::string& prompt,
        int num_patches,
        int max_tokens,
        Platform platform = host_platform()) {
    MappedFile mf;
    if (!mf.open(path))
        throw std::runtime_error(std::string("Failed to open: ") + path);

    State llm;
    llm.magic_configs = prescan_bundle_magic_configs(mf);
    apply_magic_configs_to_state(&llm);

    LiteRtEnvironment env = nullptr;
    if (!create_environment(&env, HardwareTarget::CPU, nullptr)) {
        if (llm.magic_configs) { std::free(llm.magic_configs); llm.magic_configs = nullptr; }
        mf.close();
        throw std::runtime_error("Failed to create LiteRT environment");
    }

    Tokenizer tokenizer;
    auto [tok_data, tok_size] = find_bundle_tokenizer(mf.data, mf.size);
    if (!tok_data || !tokenizer.load_from_buffer(tok_data, tok_size)) {
        if (llm.magic_configs) { std::free(llm.magic_configs); llm.magic_configs = nullptr; }
        LITERT(LiteRtDestroyEnvironment)(env);
        mf.close();
        throw std::runtime_error("Failed to load tokenizer");
    }

    llm.tokenizer = &tokenizer;
    if (!init(&llm, env, mf, platform, HardwareTarget::CPU, /*num_threads=*/0, nullptr)) {
        destroy_buffers(&llm);
        LITERT(LiteRtDestroyEnvironment)(env);
        mf.close();
        throw std::runtime_error("Failed to initialize Gemma4 text model");
    }

    VisionState vision_state;
    if (!init_vision(&vision_state, env, mf, platform, HardwareTarget::CPU, /*num_threads=*/0, nullptr)) {
        destroy_vision(&vision_state);
        destroy_buffers(&llm);
        LITERT(LiteRtDestroyEnvironment)(env);
        mf.close();
        throw std::runtime_error("Failed to initialize Gemma4 vision model");
    }

    VisionEmbeddingResult vision;
    if (!encode_synthetic_image_patches(&vision_state, num_patches, vision)) {
        destroy_vision(&vision_state);
        destroy_buffers(&llm);
        LITERT(LiteRtDestroyEnvironment)(env);
        mf.close();
        throw std::runtime_error("Failed to encode synthetic Gemma4 image");
    }

    VisionTextSmokeResult result;
    if (!generate_vision_text(&llm, tokenizer, vision, prompt, max_tokens, result)) {
        destroy_vision(&vision_state);
        destroy_buffers(&llm);
        LITERT(LiteRtDestroyEnvironment)(env);
        mf.close();
        throw std::runtime_error("Failed to generate Gemma4 VLM response");
    }

    destroy_vision(&vision_state);
    destroy_buffers(&llm);
    LITERT(LiteRtDestroyEnvironment)(env);
    mf.close();
    return result;
}

static inline VisionTextSmokeResult run_vision_text_rgb888(
        const char* path,
        const uint8_t* rgb,
        int width,
        int height,
        int row_stride,
        const std::string& prompt,
        int max_tokens,
        Platform platform = host_platform(),
        HardwareTarget encoder_hw = HardwareTarget::CPU,
        HardwareTarget decoder_hw = HardwareTarget::CPU) {
    Runtime runtime;
    if (!init_runtime(&runtime, path, platform, encoder_hw, decoder_hw,
                      /*num_threads=*/0, nullptr))
        throw std::runtime_error(std::string("Failed to open: ") + path);

    VisionTextSmokeResult result;
    if (!describe_rgb888_with_runtime(
            &runtime, rgb, width, height, row_stride, prompt, max_tokens, result)) {
        destroy_runtime(&runtime);
        throw std::runtime_error("Failed to generate Gemma4 VLM response");
    }
    destroy_runtime(&runtime);
    return result;
}
}  // namespace gemma4
