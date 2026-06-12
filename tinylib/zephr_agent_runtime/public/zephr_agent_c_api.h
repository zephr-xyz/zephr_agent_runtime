// Zephr Agent Runtime C API — native bridge for Swift/JNI.
//
// Provides the native runtime bridge: engine init (LLM + embeddings + VLM),
// text generation, embeddings, VLM, and generic tool-aware generation.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// MARK: - Opaque handles

typedef struct zephr_agent_s*       zephr_agent_t;
typedef struct zephr_embedding_result_s* zephr_embedding_result_t;
typedef struct zephr_vlm_result_s*  zephr_vlm_result_t;
typedef struct zephr_text_result_s* zephr_text_result_t;

typedef bool (*zephr_text_stream_callback_t)(const char* text, void* user_data);

typedef struct zephr_agent_gemma4_runtime_config_s {
    // -1 means automatic/default; otherwise LiteRT GPU precision values 0, 1, or 2.
    int gpu_precision;
    // 0 means automatic from model metadata/cap.
    int kv_cache_max_len;
    // -1 means automatic/default, 0 disabled, 1 enabled.
    int constrained_verify_batch;
    bool mtp_enabled;
    bool mtp_trust_verify_kv;
    bool mtp_adaptive_enabled;
    int mtp_adaptive_min_cycles;
    float mtp_adaptive_min_saved_per_cycle;
    bool mtp_trace;
} zephr_agent_gemma4_runtime_config_t;

typedef struct zephr_agent_diagnostic_gemma4_config_s {
    bool prefill_by_decode;
    // 0 means automatic/full available prefill chunk.
    int prefill_max_chunk;
    bool constrained_verify_trace;
    // 0 means automatic/all drafter tokens.
    int constrained_verify_max_accept;
} zephr_agent_diagnostic_gemma4_config_t;

typedef struct zephr_agent_execution_config_s {
    // Version 1 enables the typed runtime/diagnostic config structs below.
    // Zero-initialized callers keep the historical automatic defaults.
    uint32_t config_version;
    const char* llm_execution_plan;
    const char* rag_execution_plan;
    const char* vlm_execution_plan;
    zephr_agent_gemma4_runtime_config_t gemma4_runtime;
    zephr_agent_diagnostic_gemma4_config_t diagnostic_gemma4;
} zephr_agent_execution_config_t;

typedef struct zephr_tool_param_spec_s {
    const char* name;
    const char* description;
    const char* type;
    const char* const* enum_values;
    int enum_value_count;
    bool required;
} zephr_tool_param_spec_t;

typedef struct zephr_tool_spec_s {
    const char* name;
    const char* description;
    const zephr_tool_param_spec_t* params;
    int param_count;
} zephr_tool_spec_t;

// MARK: - Return conventions
//
// Result-producing functions return NULL on failure. Destroy functions accept
// NULL. String accessors always return a non-NULL, NUL-terminated string; they
// return "" for null handles, invalid indexes, missing optional fields, or
// invalid keys. Numeric and bool accessors return 0/false for null handles or
// invalid indexes. Optional keyed values expose explicit presence checks where
// empty string is a valid value.

// MARK: - Memory

void zephr_free(void* ptr);

// MARK: - Engine lifecycle

zephr_agent_t zephr_agent_create(void);
void          zephr_agent_destroy(zephr_agent_t agent);
void          zephr_agent_release_models(zephr_agent_t agent);
void          zephr_agent_set_litert_runtime_library_dir(zephr_agent_t agent,
                                                         const char* path);

// Initialize LLM (required), RAG (optional), and VLM (optional).
bool zephr_agent_init(zephr_agent_t agent,
                      zephr_agent_execution_config_t execution,
                      int num_threads,
                      const char* llm_model_path,
                      const char* rag_embedding_path,
                      const char* vlm_model_path,
                      const char* litert_compilation_cache_dir);

// Current text-decoder KV position, or 0 when no text model is loaded.
int zephr_agent_text_current_pos(zephr_agent_t agent);

// Native model lifecycle timing buffer. Timings are appended by init/deinit
// operations and can be drained by diagnostics after high-level SDK calls.
int         zephr_agent_model_lifecycle_timing_count(zephr_agent_t agent);
const char* zephr_agent_model_lifecycle_component(zephr_agent_t agent, int index);
const char* zephr_agent_model_lifecycle_action(zephr_agent_t agent, int index);
const char* zephr_agent_model_lifecycle_detail(zephr_agent_t agent, int index);
int64_t     zephr_agent_model_lifecycle_duration_ms(zephr_agent_t agent, int index);
bool        zephr_agent_model_lifecycle_ok(zephr_agent_t agent, int index);
void        zephr_agent_model_lifecycle_clear(zephr_agent_t agent);

// MARK: - Embeddings

// Embed text with the configured embedding model. Returns a normalized vector.
// The task type is one of: "query", "document", or "similarity". Empty or
// unknown values default to "query".
zephr_embedding_result_t zephr_agent_embed_text(
    zephr_agent_t agent,
    const char* text,
    const char* task_type);

void         zephr_embedding_result_destroy(zephr_embedding_result_t r);
int          zephr_embedding_dimension(zephr_embedding_result_t r);
int64_t      zephr_embedding_duration_ms(zephr_embedding_result_t r);
const float* zephr_embedding_data(zephr_embedding_result_t r);

// MARK: - Raw Gemma4 text generation

zephr_text_result_t zephr_agent_generate_text(
    zephr_agent_t agent,
    const char* user_message,
    const char* system_message,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p);

zephr_text_result_t zephr_agent_generate_tool_aware_text(
    zephr_agent_t agent,
    const char* user_message,
    const char* system_message,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p,
    int reserve_output_tokens,
    const zephr_tool_spec_t* tools,
    int tool_count);

zephr_text_result_t zephr_agent_generate_tool_aware_text_stream(
    zephr_agent_t agent,
    const char* user_message,
    const char* system_message,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p,
    int reserve_output_tokens,
    const zephr_tool_spec_t* tools,
    int tool_count,
    zephr_text_stream_callback_t stream_callback,
    void* stream_user_data);

zephr_text_result_t zephr_agent_generate_tool_aware_text_from_prompt_stream(
    zephr_agent_t agent,
    const char* prompt,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p,
    int reserve_output_tokens,
    const zephr_tool_spec_t* tools,
    int tool_count,
    zephr_text_stream_callback_t stream_callback,
    void* stream_user_data);

zephr_text_result_t zephr_agent_generate_text_from_prompt_stream(
    zephr_agent_t agent,
    const char* prompt,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p,
    zephr_text_stream_callback_t stream_callback,
    void* stream_user_data);

zephr_text_result_t zephr_agent_continue_tool_aware_text_stream(
    zephr_agent_t agent,
    const char* prompt_suffix,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p,
    int reserve_output_tokens,
    const zephr_tool_spec_t* tools,
    int tool_count,
    zephr_text_stream_callback_t stream_callback,
    void* stream_user_data);

zephr_text_result_t zephr_agent_continue_after_tool_response_stream(
    zephr_agent_t agent,
    const char* tool_response,
    int max_tokens,
    float temperature,
    int top_k,
    float top_p,
    int reserve_output_tokens,
    zephr_text_stream_callback_t stream_callback,
    void* stream_user_data);

void        zephr_text_result_destroy(zephr_text_result_t r);
const char* zephr_text_response(zephr_text_result_t r);
const char* zephr_text_prompt(zephr_text_result_t r);
int         zephr_text_input_tokens(zephr_text_result_t r);
int         zephr_text_prefill_tokens(zephr_text_result_t r);
int         zephr_text_decode_steps(zephr_text_result_t r);
int         zephr_text_mtp_rejected_cycles(zephr_text_result_t r);
int         zephr_text_mtp_rejected_after_prefix_0(zephr_text_result_t r);
int         zephr_text_mtp_rejected_after_prefix_1(zephr_text_result_t r);
int         zephr_text_mtp_rejected_after_prefix_2(zephr_text_result_t r);
int64_t     zephr_text_prefill_ms(zephr_text_result_t r);
int64_t     zephr_text_decode_ms(zephr_text_result_t r);
int64_t     zephr_text_first_decode_ms(zephr_text_result_t r);

// MARK: - Diagnostic heavy collection
//
// Materialize one native next-token probe for a fully formatted prompt. The
// returned string is a JSON object owned by the caller; release it with
// zephr_free. The probe resets the text KV cache before and after collection.
char* zephr_agent_collect_heavy_json(zephr_agent_t agent,
                                     const char* prompt,
                                     bool collect_activations,
                                     int top_k);

// MARK: - Agent VLM image description

// Run the agent's initialized VLM capability on an RGB888 image buffer. The
// image is expected to contain 3 bytes per pixel, with row_stride bytes between
// rows. Pass max_tokens <= 0 to use the remaining text decoder context after
// image and prompt tokens. Pass a non-NULL vlm_model_path to zephr_agent_init
// to enable this.
zephr_vlm_result_t zephr_agent_describe_image_rgb888(
    zephr_agent_t agent,
    const uint8_t* rgb,
    int width,
    int height,
    int row_stride,
    const char* prompt,
    int max_tokens);

void        zephr_vlm_result_destroy(zephr_vlm_result_t r);
const char* zephr_vlm_response(zephr_vlm_result_t r);
int         zephr_vlm_input_patches(zephr_vlm_result_t r);
int         zephr_vlm_valid_vision_tokens(zephr_vlm_result_t r);
int         zephr_vlm_image_token_slots(zephr_vlm_result_t r);
int         zephr_vlm_resized_width(zephr_vlm_result_t r);
int         zephr_vlm_resized_height(zephr_vlm_result_t r);
int         zephr_vlm_prompt_tokens(zephr_vlm_result_t r);
int         zephr_vlm_decode_steps(zephr_vlm_result_t r);
int64_t     zephr_vlm_first_decode_ms(zephr_vlm_result_t r);

// MARK: - Utilities

void  zephr_set_log_level(const char* level);
void  zephr_enable_stderr_logging(void);
char* zephr_detect_model_family(const char* path);  // caller must free with zephr_free()

#ifdef __cplusplus
}
#endif
