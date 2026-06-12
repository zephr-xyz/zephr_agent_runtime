// Zephr AI Agent C API implementation.
//
#include "core/0_zephr_agent_deps.hpp"
#include "public/zephr_agent_c_api.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// MARK: - Logging bootstrap

static tinylog::Logger& agent_log() {
    static tinylog::Logger* instance = [] {
        auto* l = new tinylog::Logger;
        tinylog::add_platform_default_sink(
            *l, "xyz.zephr.sdks.agent", "zephr_agent_runtime", tinylog::level_from_env());
        return l;
    }();
    return *instance;
}

static struct _AgentLogInit {
    _AgentLogInit() { tinylog::set_default_logger(&agent_log()); }
} _agent_log_init;

// MARK: - Opaque structs

struct zephr_agent_s {
    std::recursive_mutex mutex;
    TinyLLMEngine engine;
};

struct zephr_embedding_result_s {
    std::vector<float> values;
    int64_t duration_ms = 0;
};

struct zephr_vlm_result_s {
    gemma4::VisionTextSmokeResult result;
};

struct zephr_text_result_s {
    GenerateResult result;
};

// MARK: - Memory

static const std::array<const char*, 8> PROXIMITY_KEYWORDS = {
    "nearby",
    "near me",
    "around me",
    "close by",
    "closest",
    "nearest",
    "around here",
    "near here",
};

void zephr_free(void* ptr) {
    std::free(ptr);
}

static char* strdup_c(const std::string& s) {
    char* p = static_cast<char*>(std::malloc(s.size() + 1));
    if (!p) return nullptr;
    std::memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

static std::vector<ToolDef> tool_defs_from_c(
        const zephr_tool_spec_t* tools,
        int tool_count) {
    std::vector<ToolDef> out;
    if (!tools || tool_count <= 0)
        return out;
    out.reserve(tool_count);
    for (int i = 0; i < tool_count; i++) {
        const auto& src_tool = tools[i];
        if (!src_tool.name || src_tool.name[0] == '\0')
            continue;
        ToolDef tool;
        tool.name = src_tool.name;
        tool.description = src_tool.description ? src_tool.description : "";
        if (src_tool.params && src_tool.param_count > 0) {
            tool.params.reserve(src_tool.param_count);
            for (int j = 0; j < src_tool.param_count; j++) {
                const auto& src_param = src_tool.params[j];
                if (!src_param.name || src_param.name[0] == '\0')
                    continue;
                ToolParam param;
                param.name = src_param.name;
                param.description = src_param.description ? src_param.description : "";
                param.type = src_param.type ? src_param.type : "STRING";
                param.required = src_param.required;
                if (src_param.enum_values && src_param.enum_value_count > 0) {
                    param.enum_values.reserve(src_param.enum_value_count);
                    for (int k = 0; k < src_param.enum_value_count; k++) {
                        const char* value = src_param.enum_values[k];
                        if (value)
                            param.enum_values.push_back(value);
                    }
                }
                tool.params.push_back(std::move(param));
            }
        }
        out.push_back(std::move(tool));
    }
    return out;
}

static std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (unsigned char c : text) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char)c);
                }
        }
    }
    return out;
}

static int64_t elapsed_ms(std::chrono::steady_clock::time_point t0) {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
}

static std::string normalized_rag_query(const std::string& raw_query) {
    std::string query = raw_query;
    std::string lower_query = query;
    std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    for (auto kw : PROXIMITY_KEYWORDS) {
        size_t pos = lower_query.find(kw);
        if (pos != std::string::npos) {
            query.erase(pos, std::string(kw).length());
            lower_query.erase(pos, std::string(kw).length());
        }
    }
    while (!query.empty() && query.front() == ' ') query.erase(0, 1);
    while (!query.empty() && query.back() == ' ') query.pop_back();
    return query.empty() ? raw_query : query;
}

static uint32_t rotr32(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32 - bits));
}

static std::string sha256_hex(const std::vector<uint8_t>& data) {
    static constexpr uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };
    std::vector<uint8_t> bytes = data;
    uint64_t bit_len = static_cast<uint64_t>(bytes.size()) * 8;
    bytes.push_back(0x80);
    while ((bytes.size() % 64) != 56) bytes.push_back(0);
    for (int i = 7; i >= 0; --i)
        bytes.push_back(static_cast<uint8_t>((bit_len >> (i * 8)) & 0xff));

    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };
    for (size_t offset = 0; offset < bytes.size(); offset += 64) {
        uint32_t w[64] = {};
        for (int i = 0; i < 16; ++i) {
            size_t j = offset + static_cast<size_t>(i) * 4;
            w[i] = (static_cast<uint32_t>(bytes[j]) << 24) |
                   (static_cast<uint32_t>(bytes[j + 1]) << 16) |
                   (static_cast<uint32_t>(bytes[j + 2]) << 8) |
                   static_cast<uint32_t>(bytes[j + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = hh + s1 + ch + k[i] + w[i];
            uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = s0 + maj;
            hh = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (uint32_t value : h)
        oss << std::setw(8) << value;
    return oss.str();
}

static std::string activation_summary_json(const std::vector<float>& values) {
    std::ostringstream oss;
    oss << std::setprecision(9);
    std::vector<uint8_t> packed;
    packed.reserve(values.size() * sizeof(float));
    double total = 0.0;
    double l2 = 0.0;
    float min_value = values.empty() ? 0.0f : values.front();
    float max_value = min_value;
    for (float value : values) {
        total += static_cast<double>(value);
        l2 += static_cast<double>(value) * static_cast<double>(value);
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit");
        std::memcpy(&bits, &value, sizeof(bits));
        packed.push_back(static_cast<uint8_t>(bits & 0xff));
        packed.push_back(static_cast<uint8_t>((bits >> 8) & 0xff));
        packed.push_back(static_cast<uint8_t>((bits >> 16) & 0xff));
        packed.push_back(static_cast<uint8_t>((bits >> 24) & 0xff));
    }
    oss << "{\"dimension\":" << values.size()
        << ",\"sha256Float32LE\":\"" << sha256_hex(packed) << "\""
        << ",\"mean\":" << (values.empty() ? 0.0 : total / static_cast<double>(values.size()))
        << ",\"min\":" << min_value
        << ",\"max\":" << max_value
        << ",\"l2\":" << std::sqrt(l2)
        << ",\"sample\":[";
    const int sample_size = 16;
    std::vector<int> sample_indices;
    if (values.size() <= static_cast<size_t>(sample_size)) {
        for (int i = 0; i < static_cast<int>(values.size()); ++i)
            sample_indices.push_back(i);
    } else {
        for (int i = 0; i < sample_size; ++i) {
            int index = static_cast<int>(std::llround(
                static_cast<double>(i) * static_cast<double>(values.size() - 1) /
                static_cast<double>(sample_size - 1)));
            if (sample_indices.empty() || sample_indices.back() != index)
                sample_indices.push_back(index);
        }
    }
    for (size_t i = 0; i < sample_indices.size(); ++i) {
        int index = sample_indices[i];
        if (i) oss << ",";
        oss << "{\"index\":" << index << ",\"value\":" << values[static_cast<size_t>(index)] << "}";
    }
    oss << "]}";
    return oss.str();
}

// MARK: - Engine lifecycle

zephr_agent_t zephr_agent_create(void) {
    try {
        return new zephr_agent_s();
    } catch (const std::exception& e) {
        agent_log().error("agent create failed", {{"error", e.what()}});
        return nullptr;
    } catch (...) {
        agent_log().error("agent create failed", {{"error", "unknown"}});
        return nullptr;
    }
}

void zephr_agent_release_models(zephr_agent_t agent) {
    if (!agent) return;
    try {
        std::lock_guard<std::recursive_mutex> lock(agent->mutex);
        agent->engine.release();
    } catch (const std::exception& e) {
        agent_log().error("agent release models failed", {{"error", e.what()}});
    } catch (...) {
        agent_log().error("agent release models failed", {{"error", "unknown"}});
    }
}

void zephr_agent_set_litert_runtime_library_dir(zephr_agent_t agent,
                                                const char* path) {
    if (!agent) return;
    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    agent->engine.set_litert_runtime_library_dir(path);
}

void zephr_agent_destroy(zephr_agent_t agent) {
    if (!agent) return;
    try {
        {
            std::lock_guard<std::recursive_mutex> lock(agent->mutex);
            agent->engine.release();
        }
    } catch (const std::exception& e) {
        agent_log().error("agent destroy release failed", {{"error", e.what()}});
    } catch (...) {
        agent_log().error("agent destroy release failed", {{"error", "unknown"}});
    }
    delete agent;
}

bool zephr_agent_init(zephr_agent_t agent,
                      zephr_agent_execution_config_t execution,
                      int num_threads,
                      const char* llm_model_path,
                      const char* rag_embedding_path,
                      const char* vlm_model_path,
                      const char* litert_compilation_cache_dir) {
    if (!agent || !llm_model_path) return false;
    std::lock_guard<std::recursive_mutex> lock(agent->mutex);

    TinyLLMExecutionConfig config;
    config.llm_plan = execution.llm_execution_plan && *execution.llm_execution_plan
        ? execution.llm_execution_plan
        : "cpu";
    config.rag_plan = execution.rag_execution_plan && *execution.rag_execution_plan
        ? execution.rag_execution_plan
        : "cpu";
    config.vlm_plan = execution.vlm_execution_plan && *execution.vlm_execution_plan
        ? execution.vlm_execution_plan
        : "gpu";
    if (execution.config_version >= 1) {
        config.gemma4_runtime.gpu_precision = execution.gemma4_runtime.gpu_precision;
        config.gemma4_runtime.kv_cache_max_len = execution.gemma4_runtime.kv_cache_max_len;
        config.gemma4_runtime.constrained_verify_batch =
            execution.gemma4_runtime.constrained_verify_batch;
        config.gemma4_runtime.mtp_enabled = execution.gemma4_runtime.mtp_enabled;
        config.gemma4_runtime.mtp_trust_verify_kv =
            execution.gemma4_runtime.mtp_trust_verify_kv;
        config.gemma4_runtime.mtp_adaptive_enabled =
            execution.gemma4_runtime.mtp_adaptive_enabled;
        config.gemma4_runtime.mtp_adaptive_min_cycles =
            execution.gemma4_runtime.mtp_adaptive_min_cycles;
        config.gemma4_runtime.mtp_adaptive_min_saved_per_cycle =
            execution.gemma4_runtime.mtp_adaptive_min_saved_per_cycle;
        config.gemma4_runtime.mtp_trace = execution.gemma4_runtime.mtp_trace;
        config.diagnostic_gemma4.prefill_by_decode =
            execution.diagnostic_gemma4.prefill_by_decode;
        config.diagnostic_gemma4.prefill_max_chunk =
            execution.diagnostic_gemma4.prefill_max_chunk;
        config.diagnostic_gemma4.constrained_verify_trace =
            execution.diagnostic_gemma4.constrained_verify_trace;
        config.diagnostic_gemma4.constrained_verify_max_accept =
            execution.diagnostic_gemma4.constrained_verify_max_accept;
    }

    try {
        bool ok = agent->engine.init(
            host_platform(),
            config,
            num_threads,
            llm_model_path,
            rag_embedding_path,
            vlm_model_path,
            litert_compilation_cache_dir);

        return ok;
    } catch (const std::exception& e) {
        agent_log().error("agent init failed", {{"error", e.what()}});
        return false;
    } catch (...) {
        agent_log().error("agent init failed", {{"error", "unknown"}});
        return false;
    }
}

int zephr_agent_text_current_pos(zephr_agent_t agent) {
    if (!agent) return 0;
    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    return agent->engine.llm ? agent->engine.llm->current_pos() : 0;
}

static const ModelLifecycleTiming* zephr_agent_model_lifecycle_at(
        zephr_agent_t agent,
        int index) {
    if (!agent || index < 0 ||
        index >= static_cast<int>(agent->engine.model_lifecycle_timings.size())) {
        return nullptr;
    }
    return &agent->engine.model_lifecycle_timings[static_cast<size_t>(index)];
}

int zephr_agent_model_lifecycle_timing_count(zephr_agent_t agent) {
    if (!agent) return 0;
    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    return agent ? static_cast<int>(agent->engine.model_lifecycle_timings.size()) : 0;
}

const char* zephr_agent_model_lifecycle_component(zephr_agent_t agent, int index) {
    if (!agent) return "";
    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    auto* timing = zephr_agent_model_lifecycle_at(agent, index);
    return timing ? timing->component.c_str() : "";
}

const char* zephr_agent_model_lifecycle_action(zephr_agent_t agent, int index) {
    if (!agent) return "";
    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    auto* timing = zephr_agent_model_lifecycle_at(agent, index);
    return timing ? timing->action.c_str() : "";
}

const char* zephr_agent_model_lifecycle_detail(zephr_agent_t agent, int index) {
    if (!agent) return "";
    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    auto* timing = zephr_agent_model_lifecycle_at(agent, index);
    return timing ? timing->detail.c_str() : "";
}

int64_t zephr_agent_model_lifecycle_duration_ms(zephr_agent_t agent, int index) {
    if (!agent) return 0;
    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    auto* timing = zephr_agent_model_lifecycle_at(agent, index);
    return timing ? timing->duration_ms : 0;
}

bool zephr_agent_model_lifecycle_ok(zephr_agent_t agent, int index) {
    if (!agent) return false;
    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    auto* timing = zephr_agent_model_lifecycle_at(agent, index);
    return timing ? timing->ok : false;
}

void zephr_agent_model_lifecycle_clear(zephr_agent_t agent) {
    if (!agent) return;
    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    agent->engine.model_lifecycle_timings.clear();
}

// MARK: - Embeddings

static EmbeddingTaskType embedding_task_type_from_c(const char* value) {
    if (!value || value[0] == '\0')
        return EmbeddingTaskType::RETRIEVAL_QUERY;
    std::string task(value);
    std::transform(task.begin(), task.end(), task.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    if (task == "document" || task == "retrieval_document")
        return EmbeddingTaskType::RETRIEVAL_DOCUMENT;
    if (task == "similarity" || task == "semantic_similarity")
        return EmbeddingTaskType::SEMANTIC_SIMILARITY;
    return EmbeddingTaskType::RETRIEVAL_QUERY;
}

zephr_embedding_result_t zephr_agent_embed_text(
        zephr_agent_t agent,
        const char* text,
        const char* task_type) {
    if (!agent || !text) return nullptr;
    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    try {
        if (!agent->engine.ensure_embedding_model_for_text()) return nullptr;
        if (!agent->engine.rag) return nullptr;

        auto* result = new zephr_embedding_result_s();
        auto t0 = std::chrono::steady_clock::now();
        if (!agent->engine.rag->embed(
                text,
                embedding_task_type_from_c(task_type),
                result->values)) {
            delete result;
            return nullptr;
        }
        result->duration_ms = elapsed_ms(t0);
        return result;
    } catch (const std::exception& e) {
        agent_log().error("agent embed text failed", {{"error", e.what()}});
        return nullptr;
    } catch (...) {
        agent_log().error("agent embed text failed", {{"error", "unknown"}});
        return nullptr;
    }
}

void zephr_embedding_result_destroy(zephr_embedding_result_t r) { delete r; }

int zephr_embedding_dimension(zephr_embedding_result_t r) {
    return r ? (int)r->values.size() : 0;
}

int64_t zephr_embedding_duration_ms(zephr_embedding_result_t r) {
    return r ? r->duration_ms : 0;
}

const float* zephr_embedding_data(zephr_embedding_result_t r) {
    return (r && !r->values.empty()) ? r->values.data() : nullptr;
}

// MARK: - Raw Gemma4 text generation

zephr_text_result_t zephr_agent_generate_text(
        zephr_agent_t agent,
        const char* user_message,
        const char* system_message,
        int max_tokens,
        float temperature,
        int top_k,
        float top_p) {
    if (!agent || !user_message) {
        return nullptr;
    }

    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    if (!agent->engine.llm) {
        return nullptr;
    }

    try {
        auto gen = agent->engine.llm->generate_text(
            std::string(user_message),
            system_message ? std::string(system_message) : std::string(),
            max_tokens > 0 ? max_tokens : 256,
            temperature,
            top_k > 0 ? top_k : 40,
            top_p > 0.0f ? top_p : 0.95f);
        if (gen.response.empty() && gen.decode_steps == 0) {
            agent_log().error("Gemma4 text generation failed");
            return nullptr;
        }
        auto* r = new zephr_text_result_s();
        r->result = std::move(gen);
        return r;
    } catch (const std::exception& e) {
        agent_log().error("Gemma4 text generation failed", {{"error", e.what()}});
        return nullptr;
    } catch (...) {
        agent_log().error("Gemma4 text generation failed", {{"error", "unknown"}});
        return nullptr;
    }
}

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
        int tool_count) {
    return zephr_agent_generate_tool_aware_text_stream(
        agent,
        user_message,
        system_message,
        max_tokens,
        temperature,
        top_k,
        top_p,
        reserve_output_tokens,
        tools,
        tool_count,
        nullptr,
        nullptr);
}

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
        void* stream_user_data) {
    if (!agent || !user_message) {
        return nullptr;
    }

    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    if (!agent->engine.llm) {
        return nullptr;
    }

    try {
        auto tool_defs = tool_defs_from_c(tools, tool_count);
        auto gen = agent->engine.llm->generate_tool_aware_text(
            tool_defs,
            std::string(user_message),
            system_message ? std::string(system_message) : std::string(),
            max_tokens > 0 ? max_tokens : 256,
            temperature,
            top_k > 0 ? top_k : 40,
            top_p > 0.0f ? top_p : 0.95f,
            reserve_output_tokens,
            stream_callback
                ? std::function<bool(const std::string&)>(
                    [stream_callback, stream_user_data](const std::string& text) {
                        return stream_callback(text.c_str(), stream_user_data);
                    })
                : std::function<bool(const std::string&)>());
        if (gen.response.empty() && gen.decode_steps == 0) {
            agent_log().error("Gemma4 tool-aware text generation failed");
            return nullptr;
        }
        auto* r = new zephr_text_result_s();
        r->result = std::move(gen);
        return r;
    } catch (const std::exception& e) {
        agent_log().error("Gemma4 tool-aware text generation failed", {{"error", e.what()}});
        return nullptr;
    } catch (...) {
        agent_log().error("Gemma4 tool-aware text generation failed", {{"error", "unknown"}});
        return nullptr;
    }
}

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
        void* stream_user_data) {
    if (!agent || !prompt) {
        return nullptr;
    }

    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    if (!agent->engine.llm) {
        return nullptr;
    }

    try {
        auto tool_defs = tool_defs_from_c(tools, tool_count);
        auto gen = agent->engine.llm->generate_tool_aware_text_from_prompt(
            tool_defs,
            std::string(prompt),
            max_tokens > 0 ? max_tokens : 256,
            temperature,
            top_k > 0 ? top_k : 40,
            top_p > 0.0f ? top_p : 0.95f,
            reserve_output_tokens,
            stream_callback
                ? std::function<bool(const std::string&)>(
                    [stream_callback, stream_user_data](const std::string& text) {
                        return stream_callback(text.c_str(), stream_user_data);
                    })
                : std::function<bool(const std::string&)>());
        if (gen.response.empty() && gen.decode_steps == 0) {
            agent_log().error("Gemma4 tool-aware prompt text generation failed");
            return nullptr;
        }
        auto* r = new zephr_text_result_s();
        r->result = std::move(gen);
        return r;
    } catch (const std::exception& e) {
        agent_log().error("Gemma4 tool-aware prompt text generation failed", {{"error", e.what()}});
        return nullptr;
    } catch (...) {
        agent_log().error("Gemma4 tool-aware prompt text generation failed", {{"error", "unknown"}});
        return nullptr;
    }
}

zephr_text_result_t zephr_agent_generate_text_from_prompt_stream(
        zephr_agent_t agent,
        const char* prompt,
        int max_tokens,
        float temperature,
        int top_k,
        float top_p,
        zephr_text_stream_callback_t stream_callback,
        void* stream_user_data) {
    if (!agent || !prompt) {
        return nullptr;
    }

    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    if (!agent->engine.llm) {
        return nullptr;
    }

    try {
        auto gen = agent->engine.llm->generate_text_from_prompt(
            std::string(prompt),
            max_tokens > 0 ? max_tokens : 256,
            temperature,
            top_k > 0 ? top_k : 40,
            top_p > 0.0f ? top_p : 0.95f,
            stream_callback
                ? std::function<bool(const std::string&)>(
                    [stream_callback, stream_user_data](const std::string& text) {
                        return stream_callback(text.c_str(), stream_user_data);
                    })
                : std::function<bool(const std::string&)>());
        if (gen.response.empty() && gen.decode_steps == 0) {
            agent_log().error("Gemma4 raw prompt text generation failed");
            return nullptr;
        }
        auto* r = new zephr_text_result_s();
        r->result = std::move(gen);
        return r;
    } catch (const std::exception& e) {
        agent_log().error("Gemma4 raw prompt text generation failed", {{"error", e.what()}});
        return nullptr;
    } catch (...) {
        agent_log().error("Gemma4 raw prompt text generation failed", {{"error", "unknown"}});
        return nullptr;
    }
}

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
        void* stream_user_data) {
    if (!agent || !prompt_suffix) {
        return nullptr;
    }

    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    if (!agent->engine.llm) {
        return nullptr;
    }

    try {
        auto tool_defs = tool_defs_from_c(tools, tool_count);
        auto gen = agent->engine.llm->continue_tool_aware_text(
            tool_defs,
            std::string(prompt_suffix),
            max_tokens > 0 ? max_tokens : 256,
            temperature,
            top_k > 0 ? top_k : 40,
            top_p > 0.0f ? top_p : 0.95f,
            reserve_output_tokens,
            stream_callback
                ? std::function<bool(const std::string&)>(
                    [stream_callback, stream_user_data](const std::string& text) {
                        return stream_callback(text.c_str(), stream_user_data);
                    })
                : std::function<bool(const std::string&)>());
        if (gen.response.empty() && gen.decode_steps == 0) {
            agent_log().warn("Gemma4 incremental tool-aware text generation failed");
            return nullptr;
        }
        auto* r = new zephr_text_result_s();
        r->result = std::move(gen);
        return r;
    } catch (const std::exception& e) {
        agent_log().warn("Gemma4 incremental tool-aware text generation failed", {{"error", e.what()}});
        return nullptr;
    } catch (...) {
        agent_log().warn("Gemma4 incremental tool-aware text generation failed", {{"error", "unknown"}});
        return nullptr;
    }
}

zephr_text_result_t zephr_agent_continue_after_tool_response_stream(
        zephr_agent_t agent,
        const char* tool_response,
        int max_tokens,
        float temperature,
        int top_k,
        float top_p,
        int reserve_output_tokens,
        zephr_text_stream_callback_t stream_callback,
        void* stream_user_data) {
    if (!agent || !tool_response) {
        return nullptr;
    }

    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    if (!agent->engine.llm) {
        return nullptr;
    }

    try {
        auto gen = agent->engine.llm->continue_after_tool_response(
            std::string(tool_response),
            max_tokens > 0 ? max_tokens : 256,
            temperature,
            top_k > 0 ? top_k : 40,
            top_p > 0.0f ? top_p : 0.95f,
            reserve_output_tokens,
            stream_callback
                ? std::function<bool(const std::string&)>(
                    [stream_callback, stream_user_data](const std::string& text) {
                        return stream_callback(text.c_str(), stream_user_data);
                    })
                : std::function<bool(const std::string&)>());
        if (gen.response.empty() && gen.decode_steps == 0) {
            agent_log().error("Gemma4 tool response continuation failed");
            return nullptr;
        }
        auto* r = new zephr_text_result_s();
        r->result = std::move(gen);
        return r;
    } catch (const std::exception& e) {
        agent_log().error("Gemma4 tool response continuation failed", {{"error", e.what()}});
        return nullptr;
    } catch (...) {
        agent_log().error("Gemma4 tool response continuation failed", {{"error", "unknown"}});
        return nullptr;
    }
}

void zephr_text_result_destroy(zephr_text_result_t r) { delete r; }

const char* zephr_text_response(zephr_text_result_t r) {
    return r ? r->result.response.c_str() : "";
}

const char* zephr_text_prompt(zephr_text_result_t r) {
    return r ? r->result.prompt.c_str() : "";
}

int zephr_text_input_tokens(zephr_text_result_t r) {
    return r ? r->result.input_ids_count : 0;
}

int zephr_text_prefill_tokens(zephr_text_result_t r) {
    return r ? r->result.prefill_tokens : 0;
}

int zephr_text_decode_steps(zephr_text_result_t r) {
    return r ? r->result.decode_steps : 0;
}

int zephr_text_mtp_rejected_cycles(zephr_text_result_t r) {
    return r ? r->result.mtp_rejected_cycles : 0;
}

int zephr_text_mtp_rejected_after_prefix_0(zephr_text_result_t r) {
    return r ? r->result.mtp_rejected_after_prefix_0 : 0;
}

int zephr_text_mtp_rejected_after_prefix_1(zephr_text_result_t r) {
    return r ? r->result.mtp_rejected_after_prefix_1 : 0;
}

int zephr_text_mtp_rejected_after_prefix_2(zephr_text_result_t r) {
    return r ? r->result.mtp_rejected_after_prefix_2 : 0;
}

int64_t zephr_text_prefill_ms(zephr_text_result_t r) {
    return r ? r->result.prefill_ms : 0;
}

int64_t zephr_text_decode_ms(zephr_text_result_t r) {
    return r ? r->result.decode_ms : 0;
}

int64_t zephr_text_first_decode_ms(zephr_text_result_t r) {
    return r ? r->result.first_decode_ms : 0;
}

char* zephr_agent_collect_heavy_json(zephr_agent_t agent,
                                     const char* prompt,
                                     bool collect_activations,
                                     int top_k) {
    auto error_json = [](const std::string& error) {
        return strdup_c("{\"error\":\"" + json_escape(error) + "\"}");
    };
    if (!agent || !prompt) return error_json("missing agent or prompt");
    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    try {
        auto* llm = agent->engine.llm;
        if (!llm)
            return error_json("text model is not loaded");
        auto* tok = llm->tokenizer();
        if (!tok)
            return error_json("tokenizer is not available");

        std::string prompt_string(prompt);
        std::vector<int> token_ids = tok->encode(prompt_string);
        if (token_ids.empty())
            return error_json("prompt tokenized to zero tokens");

        std::vector<int32_t> prefill_ids;
        prefill_ids.reserve(token_ids.size() > 0 ? token_ids.size() - 1 : 0);
        for (size_t i = 0; i + 1 < token_ids.size(); ++i)
            prefill_ids.push_back(static_cast<int32_t>(token_ids[i]));

        std::vector<float> logits(VOCAB_SIZE);
        std::vector<float> activations;
        llm->reset();
        if (!prefill_ids.empty() && !llm->prefill(prefill_ids.data(), static_cast<int>(prefill_ids.size()))) {
            llm->reset();
            return error_json("prefill failed");
        }
        int32_t target_token = static_cast<int32_t>(token_ids.back());
        bool decode_ok = false;
        if (collect_activations) {
            int dim = llm->activation_dim();
            if (dim <= 0) {
                llm->reset();
                return error_json("activation collection is not supported by this backend");
            }
            activations.resize(static_cast<size_t>(dim));
            decode_ok = llm->decode_logits_activations(target_token, logits.data(), activations.data());
        } else {
            decode_ok = llm->decode_logits(target_token, logits.data());
        }
        if (!decode_ok) {
            llm->reset();
            return error_json("decode logits failed");
        }
        int bounded_top_k = std::min(VOCAB_SIZE, std::max(1, top_k));
        std::vector<std::pair<int, float>> top;
        extract_topk(logits.data(), VOCAB_SIZE, bounded_top_k, top);
        int sampled = sample_topk_topp(logits.data(), VOCAB_SIZE, 0.0f, bounded_top_k, 1.0f, llm->rng());
        llm->reset();

        std::ostringstream oss;
        oss << std::setprecision(9);
        oss << "{\"error\":null"
            << ",\"inputTokenCount\":" << token_ids.size()
            << ",\"inputLastTokenId\":" << target_token
            << ",\"sampledTokenId\":" << sampled
            << ",\"topLogits\":[";
        for (size_t i = 0; i < top.size(); ++i) {
            if (i) oss << ",";
            oss << "{\"tokenId\":" << top[i].first << ",\"logit\":" << top[i].second << "}";
        }
        oss << "]";
        if (collect_activations)
            oss << ",\"activation\":" << activation_summary_json(activations);
        oss << "}";
        return strdup_c(oss.str());
    } catch (const std::exception& e) {
        if (agent && agent->engine.llm) {
            try { agent->engine.llm->reset(); } catch (...) {}
        }
        return error_json(e.what());
    } catch (...) {
        if (agent && agent->engine.llm) {
            try { agent->engine.llm->reset(); } catch (...) {}
        }
        return error_json("unknown heavy collection failure");
    }
}

// MARK: - Agent VLM image description

zephr_vlm_result_t zephr_agent_describe_image_rgb888(
        zephr_agent_t agent,
        const uint8_t* rgb,
        int width,
        int height,
        int row_stride,
        const char* prompt,
        int max_tokens) {
    if (!agent || !rgb || width <= 0 || height <= 0 || row_stride < width * 3) {
        return nullptr;
    }

    std::lock_guard<std::recursive_mutex> lock(agent->mutex);
    try {
        if (!agent->engine.ensure_vlm_model()) {
            agent_log().error("Gemma4 VLM image description failed",
                {{"error", "VLM is not initialized"}});
            return nullptr;
        }
        auto* r = new zephr_vlm_result_s();
        if (!agent->engine.describe_image_rgb888(
            rgb,
            width,
            height,
            row_stride,
            prompt ? std::string(prompt) : std::string("Describe the image."),
            max_tokens,
            r->result)) {
            delete r;
            agent_log().error("Gemma4 VLM image description failed");
            return nullptr;
        }
        return r;
    } catch (const std::exception& e) {
        agent_log().error("Gemma4 VLM image description failed", {{"error", e.what()}});
        return nullptr;
    } catch (...) {
        agent_log().error("Gemma4 VLM image description failed", {{"error", "unknown"}});
        return nullptr;
    }
}

void zephr_vlm_result_destroy(zephr_vlm_result_t r) { delete r; }

const char* zephr_vlm_response(zephr_vlm_result_t r) {
    return r ? r->result.response.c_str() : "";
}

int zephr_vlm_input_patches(zephr_vlm_result_t r) {
    return r ? r->result.input_patches : 0;
}

int zephr_vlm_valid_vision_tokens(zephr_vlm_result_t r) {
    return r ? r->result.valid_vision_tokens : 0;
}

int zephr_vlm_image_token_slots(zephr_vlm_result_t r) {
    return r ? r->result.image_token_slots : 0;
}

int zephr_vlm_resized_width(zephr_vlm_result_t r) {
    return r ? r->result.resized_width : 0;
}

int zephr_vlm_resized_height(zephr_vlm_result_t r) {
    return r ? r->result.resized_height : 0;
}

int zephr_vlm_prompt_tokens(zephr_vlm_result_t r) {
    return r ? r->result.prompt_tokens : 0;
}

int zephr_vlm_decode_steps(zephr_vlm_result_t r) {
    return r ? r->result.decode_steps : 0;
}

int64_t zephr_vlm_first_decode_ms(zephr_vlm_result_t r) {
    return r ? r->result.first_decode_ms : 0;
}

// MARK: - Utilities

void zephr_set_log_level(const char* level) {
    if (!level) return;
    std::string s(level);
    for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    tinylog::Level l = tinylog::Level::Info;
    if (s == "TRACE") l = tinylog::Level::Trace;
    else if (s == "DEBUG") l = tinylog::Level::Debug;
    else if (s == "INFO") l = tinylog::Level::Info;
    else if (s == "WARN") l = tinylog::Level::Warn;
    else if (s == "ERROR") l = tinylog::Level::Error;
    agent_log().set_level(l);
}

void zephr_enable_stderr_logging(void) {
    static bool enabled = false;
    if (enabled)
        return;
    enabled = true;
    // iOS collect launches through `xcrun devicectl --console`, which streams
    // stdout/stderr but not OSLog. Mirror native logs to stderr when requested
    // so the live xcrun view still shows detailed init diagnostics.
    agent_log().add_sink(std::cerr, tinylog::Format::Human, tinylog::Level::Trace);
}

char* zephr_detect_model_family(const char* path) {
    if (!path) return strdup_c("");
    try {
        return strdup_c(detect_model_family(path));
    } catch (const std::exception& e) {
        agent_log().error("detect model family failed", {{"error", e.what()}});
        return strdup_c("");
    } catch (...) {
        agent_log().error("detect model family failed", {{"error", "unknown"}});
        return strdup_c("");
    }
}
