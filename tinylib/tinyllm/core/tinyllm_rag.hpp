// TinyLLM RAG engine — embedding inference + vector similarity search.
//
// Loads an EmbeddingGemma .litertlm bundle containing the embedding model and
// SentencePiece tokenizer.
// Provides: embed text → 768-dim vector, corpus management, similarity search.
//
// Requires tinyllm_core.hpp to be included first (for LiteRT helpers,
// Tokenizer, MappedFile, buffer read/write, logging).

#pragma once

#include "tinyllm_core.hpp"
#include <unordered_map>

// ---------------------------------------------------------------------------
// Task type prefixes for EmbeddingGemma.
//
// Keep these aligned with the model-card / AI Edge RAG EmbedData task strings:
//   RETRIEVAL_QUERY    -> task: search result | query: ...
//   RETRIEVAL_DOCUMENT -> title: none | text: ...
//   SEMANTIC_SIMILARITY -> task: sentence similarity | query: ...
// ---------------------------------------------------------------------------

enum class EmbeddingTaskType {
    RETRIEVAL_QUERY,
    RETRIEVAL_DOCUMENT,
    SEMANTIC_SIMILARITY,
};

static inline int rag_environment_accelerator_mask(HardwareTarget hw) {
    return litert_options_accelerator_mask(hw, ModelFamily::GEMMA3_EMBEDDING);
}

static inline const char* task_prefix(EmbeddingTaskType t) {
    switch (t) {
        case EmbeddingTaskType::RETRIEVAL_QUERY:       return "task: search result | query: ";
        case EmbeddingTaskType::RETRIEVAL_DOCUMENT:    return "title: none | text: ";
        case EmbeddingTaskType::SEMANTIC_SIMILARITY:   return "task: sentence similarity | query: ";
    }
    return "";
}

// MARK: - POI metadata (populated from waypoint tile protos)

struct PoiInfo {
    std::string name;
    std::string description;
    std::string embedding_document;
    std::string types;           // "cafe,food,restaurant"
    float rating = 0.0f;
    std::string place_id;
    int64_t created_at = 0;      // unix timestamp
    int64_t expires_at = 0;      // 0 = permanent
    std::string visual_description;
    std::vector<float> visual_embedding;

    struct RoadSegment {
        std::string name;
        std::string road_class;
        double start_lat = 0, start_lon = 0;
        double end_lat = 0, end_lon = 0;
    };
    std::vector<RoadSegment> enclosing_roads;
};

// MARK: - Corpus entry

struct RagEntry {
    std::string id;
    std::vector<float> embedding;  // L2-normalized
    double lat = 0.0;
    double lon = 0.0;
    PoiInfo poi;
};

// MARK: - Query result

struct RagResult {
    std::string id;
    float similarity = 0.0f;
    double distance_m = 0.0;
    float combined_score = 0.0f;
};

// MARK: - Geo math

static inline double haversine_m(double lat1, double lon1, double lat2, double lon2) {
    constexpr double R = 6371000.0;
    double dlat = (lat2 - lat1) * M_PI / 180.0;
    double dlon = (lon2 - lon1) * M_PI / 180.0;
    double a = std::sin(dlat / 2) * std::sin(dlat / 2) +
               std::cos(lat1 * M_PI / 180.0) * std::cos(lat2 * M_PI / 180.0) *
               std::sin(dlon / 2) * std::sin(dlon / 2);
    return R * 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
}

// MARK: - RAG Engine

struct RagEngine {
    Tokenizer tokenizer;
    MappedFile mapped_bundle;

    LiteRtEnvironment env = nullptr;  // borrowed, not owned
    LiteRtModel model = nullptr;
    LiteRtCompiledModel compiled_model = nullptr;
    LiteRtOptions options = nullptr;

    SignatureInfo sig;
    LiteRtTensorBuffer input_ids_buf = nullptr;
    LiteRtTensorBuffer output_buf = nullptr;
    std::vector<LiteRtTensorBuffer> input_bufs;
    std::vector<LiteRtTensorBuffer> output_bufs;

    int max_seq_len = 256;
    int embedding_dim = 768;
    bool configured = false;
    bool tokenizer_loaded = false;
    bool model_ready = false;

    std::string bundle_path;
    Platform platform = Platform::APPLE_OS;
    HardwareTarget hw = HardwareTarget::CPU;
    int num_threads = 0;
    std::string compilation_cache_dir_storage;

    std::vector<RagEntry> corpus;
    std::unordered_map<std::string, size_t> corpus_index_by_id;

    // Scoring weights (match Android defaults)
    float similarity_weight = 0.6f;
    float distance_weight = 0.4f;

    // --- lifecycle ---

    bool init(const char* bundle_path,
              LiteRtEnvironment shared_env,
              Platform platform, HardwareTarget hw, int num_threads = 0,
              const char* compilation_cache_dir = nullptr) {

        release_all();

        this->bundle_path = bundle_path ? bundle_path : "";
        env = shared_env;
        this->platform = platform;
        this->hw = hw;
        this->num_threads = num_threads;
        compilation_cache_dir_storage = compilation_cache_dir ? compilation_cache_dir : "";
        configured = !this->bundle_path.empty() && env != nullptr;
        if (!configured)
            return false;

        if (!ensure_model()) {
            release_all();
            return false;
        }
        return true;
    }

    bool ensure_model() {
        if (model_ready)
            return true;
        if (!configured || !env)
            return false;

        release_model();

        const char* path = bundle_path.c_str();
        if (!mapped_bundle.open(path)) {
            tinylog::logger().error("RagEngine: failed to mmap bundle",
                {{"path", bundle_path}});
            return false;
        }

        auto sections = parse_litertlm_bundle(mapped_bundle.data, mapped_bundle.size);
        const BundleSection* model_section =
            find_bundle_section(sections, BundleModelType::EMBEDDER);
        if (!model_section) {
            for (const auto& section : sections) {
                if (section.data_type == BundleSectionDataType::TFLITE_MODEL) {
                    model_section = &section;
                    break;
                }
            }
        }
        if (!model_section) {
            tinylog::logger().error("RagEngine: no TFLite model section in bundle",
                {{"path", bundle_path}});
            release_model();
            return false;
        }

        if (!tokenizer_loaded) {
            auto [tokenizer_data, tokenizer_size] =
                find_bundle_tokenizer(mapped_bundle.data, mapped_bundle.size);
            if (!tokenizer_data || tokenizer_size == 0) {
                tinylog::logger().error("RagEngine: no SentencePiece tokenizer section in bundle",
                    {{"path", bundle_path}});
                release_model();
                return false;
            }

            if (!tokenizer.load_from_buffer(tokenizer_data, tokenizer_size)) {
                tinylog::logger().error("RagEngine: failed to load tokenizer from bundle",
                    {{"path", bundle_path}});
                release_model();
                return false;
            }
            tokenizer_loaded = true;
        }

        tinylog::logger().info("RagEngine: initializing", {
            {"bundle", bundle_path},
            {"hw", std::string(hw_target_name(this->hw))},
            {"env", std::string("shared")},
        });

        if (LITERT(LiteRtCreateModelFromBuffer)(
                env, model_section->data, model_section->size, &model) != kLiteRtStatusOk) {
            tinylog::logger().error("RagEngine: LiteRtCreateModelFromBuffer failed");
            release_model();
            return false;
        }

        options = create_litert_options(this->platform,
                                        this->hw,
                                        ModelFamily::GEMMA3_EMBEDDING,
                                        this->num_threads,
                                        "rag_embedding",
                                        path,
                                        compilation_cache_dir_storage.empty()
                                            ? nullptr
                                            : compilation_cache_dir_storage.c_str());
        if (!options) {
            release_model();
            return false;
        }

        drain_active_litert_metal_queue();
        log_metal_memory_snapshot("before rag_embedding compile");
        const bool compile_ok = create_compiled_model_with_magic(
            env, model, options, &compiled_model, nullptr,
            "LiteRtCreateCompiledModel(rag_embedding)");
        log_metal_memory_snapshot(compile_ok
                                      ? "after rag_embedding compile"
                                      : "failed rag_embedding compile");
        if (!compile_ok) {
            tinylog::logger().error("RagEngine: compilation failed");
            release_model();
            return false;
        }

        if (!discover_signature(model, 0, sig)) {
            release_model();
            return false;
        }

        for (size_t i = 0; i < sig.num_inputs; i++) {
            LiteRtRankedTensorType ttype = {};
            std::string dtype_str = "?";
            std::string shape_str;
            if (get_input_tensor_type(sig.sig, i, &ttype)) {
                switch (ttype.element_type) {
                    case kLiteRtElementTypeInt32:   dtype_str = "int32"; break;
                    case kLiteRtElementTypeFloat32: dtype_str = "float32"; break;
                    case kLiteRtElementTypeInt8:    dtype_str = "int8"; break;
                    case kLiteRtElementTypeFloat16: dtype_str = "float16"; break;
                    default: dtype_str = "type_" + std::to_string(ttype.element_type); break;
                }
                for (int d = 0; d < ttype.layout.rank; d++) {
                    if (d > 0) shape_str += "x";
                    shape_str += std::to_string(ttype.layout.dimensions[d]);
                }
            }
            tinylog::logger().debug("  input", {
                {"idx", std::to_string(i)}, {"name", sig.input_names[i]},
                {"dtype", dtype_str}, {"shape", shape_str}});
        }
        for (size_t i = 0; i < sig.num_outputs; i++) {
            LiteRtRankedTensorType ttype = {};
            std::string dtype_str = "?";
            std::string shape_str;
            if (get_output_tensor_type(sig.sig, i, &ttype)) {
                switch (ttype.element_type) {
                    case kLiteRtElementTypeInt32:   dtype_str = "int32"; break;
                    case kLiteRtElementTypeFloat32: dtype_str = "float32"; break;
                    case kLiteRtElementTypeInt8:    dtype_str = "int8"; break;
                    case kLiteRtElementTypeFloat16: dtype_str = "float16"; break;
                    default: dtype_str = "type_" + std::to_string(ttype.element_type); break;
                }
                for (int d = 0; d < ttype.layout.rank; d++) {
                    if (d > 0) shape_str += "x";
                    shape_str += std::to_string(ttype.layout.dimensions[d]);
                }
            }
            tinylog::logger().debug("  output", {
                {"idx", std::to_string(i)}, {"name", sig.output_names[i]},
                {"dtype", dtype_str}, {"shape", shape_str}});
        }

        if (!allocate_buffers()) {
            release_model();
            return false;
        }

        model_ready = true;
        tinylog::logger().info("RagEngine: initialized", {
            {"bundle", bundle_path},
            {"seq_len", std::to_string(max_seq_len)},
            {"embedding_dim", std::to_string(embedding_dim)},
        });
        return true;
    }

    void release_model(const char* reason = nullptr) {
        const bool had_residency =
            model_ready || mapped_bundle.data || model || compiled_model || options ||
            input_ids_buf || output_buf || !input_bufs.empty() || !output_bufs.empty();
        if (compiled_model) { LITERT(LiteRtDestroyCompiledModel)(compiled_model); compiled_model = nullptr; }
        drain_active_litert_metal_queue();
        destroy_all_buffers();
        if (options) { LITERT(LiteRtDestroyOptions)(options); options = nullptr; }
        if (model) { LITERT(LiteRtDestroyModel)(model); model = nullptr; }
        mapped_bundle.close();
        if (had_residency) {
            log_metal_memory_snapshot("after rag_embedding release");
            log_process_memory_snapshot("after rag_embedding release");
        }
        sig = SignatureInfo{};
        max_seq_len = 256;
        input_bufs.clear();
        output_bufs.clear();
        model_ready = false;
        if (reason && had_residency) {
            tinylog::logger().info("RagEngine: embedding model released",
                {{"bundle", bundle_path}, {"reason", std::string(reason)},
                 {"corpus_size", std::to_string(corpus.size())}});
        }
    }

    void release_all() {
        release_model();
        env = nullptr;  // borrowed, not owned
        corpus.clear();
        corpus_index_by_id.clear();
        configured = false;
        tokenizer_loaded = false;
        bundle_path.clear();
        compilation_cache_dir_storage.clear();
    }

    void release() {
        release_all();
    }

    // --- embedding ---

    bool embed(const std::string& text, EmbeddingTaskType task_type,
               std::vector<float>& out) {
        if (!ensure_model()) return false;

        std::string prefixed = std::string(task_prefix(task_type)) + text;
        std::vector<int> token_ids = tokenizer.encode(prefixed);

        // EmbeddingGemma expects BOS + tokens + EOS (matching Python pipeline)
        static constexpr int BOS_ID = 2;
        static constexpr int EOS_ID = 1;
        token_ids.insert(token_ids.begin(), BOS_ID);
        token_ids.push_back(EOS_ID);

        if ((int)token_ids.size() > max_seq_len)
            token_ids.resize(max_seq_len);

        std::vector<int32_t> padded(max_seq_len, 0);
        for (size_t i = 0; i < token_ids.size(); i++)
            padded[i] = token_ids[i];

        if (!write_int_buf(input_ids_buf, padded.data(), max_seq_len))
            return false;

        // Clear output buffer events (GPU)
        clear_buffer_events(output_bufs.data(), output_bufs.size());

        if (LITERT(LiteRtRunCompiledModel)(
                compiled_model, sig.sig_index,
                (LiteRtParamIndex)input_bufs.size(), input_bufs.data(),
                (LiteRtParamIndex)output_bufs.size(), output_bufs.data()) != kLiteRtStatusOk) {
            tinylog::logger().error("RagEngine: inference failed");
            return false;
        }

        out.resize(embedding_dim);
        if (!read_float_buf(output_buf, out.data(), embedding_dim))
            return false;

        l2_normalize(out.data(), embedding_dim);
        return true;
    }

    // --- corpus ---

    void add(const std::string& id, const float* embedding, int dims,
             double lat, double lon) {
        RagEntry entry;
        entry.id = id;
        entry.embedding.assign(embedding, embedding + dims);
        l2_normalize(entry.embedding.data(), dims);
        entry.lat = lat;
        entry.lon = lon;
        upsert_entry(std::move(entry));
    }

    void clear() {
        corpus.clear();
        corpus_index_by_id.clear();
    }

    size_t corpus_size() const { return corpus.size(); }

    // Optional zephr_agent_tools extension for Python compatibility.
    int load_tile(const void* data, size_t size);

    bool upsert_entry(RagEntry entry) {
        if (!entry.id.empty()) {
            auto it = corpus_index_by_id.find(entry.id);
            if (it != corpus_index_by_id.end() && it->second < corpus.size()) {
                corpus[it->second] = std::move(entry);
                return false;
            }
        }

        const std::string id = entry.id;
        corpus.push_back(std::move(entry));
        if (!id.empty()) {
            corpus_index_by_id[id] = corpus.size() - 1;
        }
        return true;
    }

    // --- search ---

    std::vector<RagResult> query(const std::string& text, double lat, double lon,
                                  double radius_m, int top_k,
                                  float similarity_threshold = 0.3f) {
        std::vector<float> query_emb;
        if (!embed(text, EmbeddingTaskType::RETRIEVAL_QUERY, query_emb))
            return {};

        return query_by_embedding(query_emb.data(), (int)query_emb.size(),
                                   lat, lon, radius_m, top_k, similarity_threshold);
    }

    std::vector<RagResult> query_visual(const std::string& text, double lat, double lon,
                                         double radius_m, int top_k,
                                         float similarity_threshold = 0.25f) {
        std::vector<float> query_emb;
        if (!embed(text, EmbeddingTaskType::SEMANTIC_SIMILARITY, query_emb))
            return {};

        return query_visual_by_embedding(query_emb.data(), (int)query_emb.size(),
                                         lat, lon, radius_m, top_k, similarity_threshold);
    }

    std::vector<RagResult> query_by_embedding(const float* query_emb, int dims,
                                               double lat, double lon,
                                               double radius_m, int top_k,
                                               float similarity_threshold = 0.3f) {
        std::vector<RagResult> results;

        for (const auto& entry : corpus) {
            double dist = haversine_m(lat, lon, entry.lat, entry.lon);
            if (radius_m > 0 && dist > radius_m) continue;

            int min_dims = std::min(dims, (int)entry.embedding.size());
            float sim = dot_product(query_emb, entry.embedding.data(), min_dims);

            if (sim < similarity_threshold) continue;

            float dist_score = (radius_m > 0)
                ? (float)(1.0 - dist / radius_m)
                : 1.0f;
            if (dist_score < 0.0f) dist_score = 0.0f;

            float combined = similarity_weight * sim + distance_weight * dist_score;

            results.push_back({entry.id, sim, dist, combined});
        }

        std::sort(results.begin(), results.end(),
                  [](const RagResult& a, const RagResult& b) {
                      return a.combined_score > b.combined_score;
                  });

        if ((int)results.size() > top_k)
            results.resize(top_k);

        return results;
    }

    std::vector<RagResult> query_visual_by_embedding(const float* query_emb, int dims,
                                                     double lat, double lon,
                                                     double radius_m, int top_k,
                                                     float similarity_threshold = 0.25f) {
        std::vector<RagResult> results;

        for (const auto& entry : corpus) {
            if (entry.poi.visual_embedding.empty()) continue;
            if ((int)entry.poi.visual_embedding.size() != dims) continue;

            double dist = haversine_m(lat, lon, entry.lat, entry.lon);
            if (radius_m > 0 && dist > radius_m) continue;

            float sim = dot_product(query_emb, entry.poi.visual_embedding.data(), dims);
            if (sim < similarity_threshold) continue;

            float dist_score = (radius_m > 0)
                ? (float)(1.0 - dist / radius_m)
                : 1.0f;
            if (dist_score < 0.0f) dist_score = 0.0f;

            float combined = similarity_weight * sim + distance_weight * dist_score;

            results.push_back({entry.id, sim, dist, combined});
        }

        std::sort(results.begin(), results.end(),
                  [](const RagResult& a, const RagResult& b) {
                      return a.combined_score > b.combined_score;
                  });

        if ((int)results.size() > top_k)
            results.resize(top_k);

        return results;
    }

private:

    void destroy_all_buffers() {
        auto destroy_unique = [](LiteRtTensorBuffer& candidate,
                                 std::vector<LiteRtTensorBuffer>& destroyed) {
            if (!candidate)
                return;
            for (auto existing : destroyed) {
                if (existing == candidate) {
                    candidate = nullptr;
                    return;
                }
            }
            destroyed.push_back(candidate);
            destroy_buffer(candidate);
        };

        std::vector<LiteRtTensorBuffer> destroyed;
        destroy_unique(input_ids_buf, destroyed);
        destroy_unique(output_buf, destroyed);
        for (auto& buffer : input_bufs)
            destroy_unique(buffer, destroyed);
        for (auto& buffer : output_bufs)
            destroy_unique(buffer, destroyed);
    }

    bool allocate_buffers() {
        LiteRtRankedTensorType input_type = {};
        int input_idx = sig.input_index_of("input_ids");
        if (input_idx < 0) input_idx = sig.input_index_of("text_batch");
        if (input_idx < 0) input_idx = 0;
        if (get_input_tensor_type(sig.sig, input_idx, &input_type)) {
            if (input_type.layout.rank >= 1)
                max_seq_len = input_type.layout.dimensions[input_type.layout.rank - 1];
        }

        // Discover output tensor shape to learn embedding_dim
        LiteRtRankedTensorType output_type = {};
        if (get_output_tensor_type(sig.sig, 0, &output_type)) {
            if (output_type.layout.rank >= 1)
                embedding_dim = output_type.layout.dimensions[output_type.layout.rank - 1];
        }

        input_ids_buf = alloc_managed_input(env, compiled_model, sig, input_idx, "input_ids");
        if (!input_ids_buf) return false;

        output_buf = alloc_managed_output(env, compiled_model, sig, 0, "embeddings");
        if (!output_buf) return false;

        // Build buffer arrays for RunCompiledModel
        input_bufs.resize(sig.num_inputs, nullptr);
        output_bufs.resize(sig.num_outputs, nullptr);

        input_bufs[input_idx] = input_ids_buf;
        output_bufs[0] = output_buf;

        // Allocate any remaining inputs (e.g., attention_mask, position_ids)
        for (size_t i = 0; i < sig.num_inputs; i++) {
            if (input_bufs[i]) continue;
            input_bufs[i] = alloc_managed_input(env, compiled_model, sig, i,
                                                 sig.input_names[i].c_str());
            if (!input_bufs[i]) {
                tinylog::logger().warn("RagEngine: could not allocate input",
                    {{"name", sig.input_names[i]}});
                return false;
            }
        }

        // Allocate any remaining outputs
        for (size_t i = 0; i < sig.num_outputs; i++) {
            if (output_bufs[i]) continue;
            output_bufs[i] = alloc_managed_output(env, compiled_model, sig, i,
                                                    sig.output_names[i].c_str());
            if (!output_bufs[i]) {
                tinylog::logger().warn("RagEngine: could not allocate output",
                    {{"name", sig.output_names[i]}});
                return false;
            }
        }

        return true;
    }

    static float dot_product(const float* a, const float* b, int n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    static void l2_normalize(float* v, int n) {
        float sum_sq = 0.0f;
        for (int i = 0; i < n; i++) sum_sq += v[i] * v[i];
        if (sum_sq > 0.0f) {
            float inv_norm = 1.0f / std::sqrt(sum_sq);
            for (int i = 0; i < n; i++) v[i] *= inv_norm;
        }
    }
};
