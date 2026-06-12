// TinyLLM core infrastructure — model-agnostic shared code.
//
// This header provides the platform-independent building blocks used by all
// model backends (Gemma 4 text/vision, EmbeddingGemma, etc.):
//
//   - SentencePiece tokenizer wrapper
//   - LiteRT buffer helpers (alloc, read, write, lock/unlock)
//   - Signature discovery
//   - Sampling (top-k/top-p, constrained)
//   - Diagnostic types
//
// Before using this header, call tinylog::set_default_logger().

#pragma once

#include "0_tinyllm_deps.hpp"  // IWYU pragma: export

// MARK: - SentencePiece tokenizer wrapper
class Tokenizer {
public:
    bool load(const std::string& model_path) {
        auto status = processor_.Load(model_path);
        if (!status.ok()) {
            fprintf(stderr, "SentencePiece Load failed: %s\n", status.ToString().c_str());
            return false;
        }
        return init_control_tokens();
    }

    bool load_from_buffer(const void* data, size_t size) {
        auto status = processor_.LoadFromSerializedProto(
            {reinterpret_cast<const char*>(data), size});
        if (!status.ok()) {
            fprintf(stderr, "SentencePiece LoadFromSerializedProto failed: %s\n",
                    status.ToString().c_str());
            return false;
        }
        return init_control_tokens();
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        encode_with_controls(text, ids);
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string text;
        processor_.Decode(ids, &text);
        return text;
    }

    int piece_to_id(const std::string& piece) const {
        return processor_.PieceToId(piece);
    }

    std::string id_to_piece(int id) const {
        if (id < 0 || id >= processor_.GetPieceSize()) return "?";
        return processor_.IdToPiece(id);
    }

    int vocab_size() const {
        return processor_.GetPieceSize();
    }

private:
    sentencepiece::SentencePieceProcessor processor_;

    struct ControlToken {
        std::string piece;
        int id;
    };
    std::vector<ControlToken> control_tokens_;

    bool init_control_tokens() {
        int vs = processor_.GetPieceSize();
        for (int id = 0; id < vs; id++) {
            if (processor_.IsControl(id)) {
                const auto& piece = processor_.IdToPiece(id);
                if (piece.size() > 1 && piece[0] == '<' && piece.back() == '>') {
                    control_tokens_.push_back({piece, id});
                }
            }
        }
        std::sort(control_tokens_.begin(), control_tokens_.end(),
                  [](const ControlToken& a, const ControlToken& b) {
                      return a.piece.size() > b.piece.size();
                  });
        return true;
    }

    void encode_with_controls(const std::string& text, std::vector<int>& out) const {
        if (control_tokens_.empty()) {
            processor_.Encode(text, &out);
            return;
        }

        size_t pos = 0;
        size_t seg_start = 0;

        while (pos < text.size()) {
            if (text[pos] == '<') {
                bool matched = false;
                for (const auto& ct : control_tokens_) {
                    if (pos + ct.piece.size() <= text.size() &&
                        text.compare(pos, ct.piece.size(), ct.piece) == 0) {
                        if (pos > seg_start) {
                            std::string segment = text.substr(seg_start, pos - seg_start);
                            std::vector<int> seg_ids;
                            processor_.Encode(segment, &seg_ids);
                            out.insert(out.end(), seg_ids.begin(), seg_ids.end());
                        }
                        out.push_back(ct.id);
                        pos += ct.piece.size();
                        seg_start = pos;
                        matched = true;
                        break;
                    }
                }
                if (!matched) pos++;
            } else {
                pos++;
            }
        }

        if (seg_start < text.size()) {
            std::string segment = text.substr(seg_start);
            std::vector<int> seg_ids;
            processor_.Encode(segment, &seg_ids);
            out.insert(out.end(), seg_ids.begin(), seg_ids.end());
        }
    }
};

// LITERT(name) — dispatch macro for LiteRT C API calls.
//
// Direct linking: symbols resolved at link time (Python nanobind, iOS,
// Android SDK CMake).
// Dynamic loading (Android): routes through fn_##name function pointers
// loaded via dlsym. The JNI translation unit declares and loads these
// pointers using DECL_FN / LOAD_FN before any LITERT() calls.
#ifndef LITERT
#if defined(TINYLLM_LITERT_DIRECT_LINK)
#define LITERT(name) name
#elif defined(__ANDROID__)
#define LITERT(name) fn_##name
#else
#define LITERT(name) name
#endif
#endif


// MARK: - Constants

static constexpr int VOCAB_SIZE = 262144;  // Shared across all Gemma models

static constexpr float ATTEND = 0.0f;
static constexpr float MASKED_FP16 = -45824.0f;
static constexpr float MASKED_FP32 = -0.7f * std::numeric_limits<float>::max();

// Execution modes
static constexpr int EXEC_MODE_CPU = 0;
static constexpr int EXEC_MODE_GPU = 1;
static constexpr int EXEC_MODE_TPU = 3;

static inline int64_t bytes_to_mb(uint64_t bytes) {
    return static_cast<int64_t>(bytes / (1024ull * 1024ull));
}

#ifdef __APPLE__
static inline void log_process_memory_snapshot(const char* label) {
    mach_task_basic_info_data_t basic = {};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t basic_status = task_info(
        mach_task_self(), MACH_TASK_BASIC_INFO,
        reinterpret_cast<task_info_t>(&basic), &basic_count);

    task_vm_info_data_t vm = {};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    kern_return_t vm_status = task_info(
        mach_task_self(), TASK_VM_INFO,
        reinterpret_cast<task_info_t>(&vm), &vm_count);

    if (basic_status != KERN_SUCCESS && vm_status != KERN_SUCCESS) {
        tinylog::logger().warn("Process memory snapshot failed", {
            {"label", std::string(label ? label : "")},
            {"basic_status", (int64_t)basic_status},
            {"vm_status", (int64_t)vm_status},
        });
        return;
    }

    tinylog::logger().info("Process memory snapshot", {
        {"label", std::string(label ? label : "")},
        {"resident_mb", basic_status == KERN_SUCCESS
            ? bytes_to_mb(static_cast<uint64_t>(basic.resident_size))
            : -1},
        {"virtual_mb", basic_status == KERN_SUCCESS
            ? bytes_to_mb(static_cast<uint64_t>(basic.virtual_size))
            : -1},
        {"physical_footprint_mb", vm_status == KERN_SUCCESS
            ? bytes_to_mb(static_cast<uint64_t>(vm.phys_footprint))
            : -1},
        {"internal_mb", vm_status == KERN_SUCCESS
            ? bytes_to_mb(static_cast<uint64_t>(vm.internal))
            : -1},
        {"compressed_mb", vm_status == KERN_SUCCESS
            ? bytes_to_mb(static_cast<uint64_t>(vm.compressed))
            : -1},
    });
}
#else
static inline void log_process_memory_snapshot(const char*) {}
#endif

// MARK: - Memory-mapped file wrapper

enum class MappedFileAccess {
    ReadOnly,
    PrivateWritable,
};

struct MappedFile {
    void* data = nullptr;
    size_t size = 0;
    int fd = -1;
    MappedFileAccess access_mode = MappedFileAccess::ReadOnly;
    std::string path;

    bool open(const char* p, MappedFileAccess access = MappedFileAccess::ReadOnly) {
        path = p;
        access_mode = access;
        fd = ::open(p, O_RDONLY);
        if (fd < 0) {
            tinylog::logger().error("MappedFile open failed",
                {{"path", std::string(p)}, {"errno", errno}, {"error", std::strerror(errno)}});
            std::fprintf(stderr, "MappedFile open failed path=%s errno=%d error=%s\n",
                         p, errno, std::strerror(errno));
            access_mode = MappedFileAccess::ReadOnly;
            return false;
        }
        struct stat st;
        if (fstat(fd, &st) < 0) {
            tinylog::logger().error("MappedFile stat failed",
                {{"path", std::string(p)}, {"errno", errno}, {"error", std::strerror(errno)}});
            std::fprintf(stderr, "MappedFile stat failed path=%s errno=%d error=%s\n",
                         p, errno, std::strerror(errno));
            ::close(fd); fd = -1; access_mode = MappedFileAccess::ReadOnly; return false;
        }
        size = (size_t)st.st_size;
        int prot = PROT_READ;
        if (access == MappedFileAccess::PrivateWritable)
            prot |= PROT_WRITE;
        const int flags = access == MappedFileAccess::PrivateWritable
            ? MAP_PRIVATE
            : MAP_SHARED;
        data = mmap(nullptr, size, prot, flags, fd, 0);
        if (data == MAP_FAILED) {
            int mmap_errno = errno;
            tinylog::logger().error("MappedFile mmap failed",
                {{"path", std::string(p)}, {"bytes", (int64_t)size},
                 {"access", access == MappedFileAccess::PrivateWritable
                    ? std::string("private_writable")
                    : std::string("read_only")},
                 {"errno", mmap_errno}, {"error", std::strerror(mmap_errno)}});
            std::fprintf(stderr,
                         "MappedFile mmap failed path=%s bytes=%lld access=%s errno=%d error=%s\n",
                         p, static_cast<long long>(size),
                         access == MappedFileAccess::PrivateWritable
                            ? "private_writable"
                            : "read_only",
                         mmap_errno, std::strerror(mmap_errno));
            data = nullptr; ::close(fd); fd = -1; access_mode = MappedFileAccess::ReadOnly; return false;
        }
        ::close(fd);
        fd = -1;
        return true;
    }

    void close() {
        if (data && data != MAP_FAILED) {
            if (access_mode == MappedFileAccess::PrivateWritable)
                madvise(data, size, MADV_DONTNEED);
            munmap(data, size);
            data = nullptr;
        }
        if (fd >= 0) { ::close(fd); fd = -1; }
        size = 0;
        access_mode = MappedFileAccess::ReadOnly;
    }

    ~MappedFile() { close(); }

    // Non-copyable
    MappedFile() = default;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& o) noexcept
        : data(o.data), size(o.size), fd(o.fd), access_mode(o.access_mode),
          path(std::move(o.path)) {
        o.data = nullptr; o.size = 0; o.fd = -1; o.access_mode = MappedFileAccess::ReadOnly;
    }
};

struct MappedRegion {
    void* mapping = nullptr;
    void* data = nullptr;
    size_t mapping_size = 0;
    size_t size = 0;
    int fd = -1;
    MappedFileAccess access_mode = MappedFileAccess::ReadOnly;
    std::string path;

    bool open(const char* p, size_t offset, size_t byte_count,
              MappedFileAccess access = MappedFileAccess::ReadOnly) {
        path = p;
        access_mode = access;
        fd = ::open(p, O_RDONLY);
        if (fd < 0) {
            tinylog::logger().error("MappedRegion open failed",
                {{"path", std::string(p)}, {"errno", errno}, {"error", std::strerror(errno)}});
            std::fprintf(stderr, "MappedRegion open failed path=%s errno=%d error=%s\n",
                         p, errno, std::strerror(errno));
            access_mode = MappedFileAccess::ReadOnly;
            return false;
        }

        long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) page_size = 4096;
        size_t page_mask = static_cast<size_t>(page_size - 1);
        size_t aligned_offset = offset & ~page_mask;
        size_t delta = offset - aligned_offset;
        mapping_size = delta + byte_count;
        size = byte_count;

        int prot = PROT_READ;
        if (access == MappedFileAccess::PrivateWritable)
            prot |= PROT_WRITE;
        const int flags = access == MappedFileAccess::PrivateWritable
            ? MAP_PRIVATE
            : MAP_SHARED;
        const bool log_large_private_mapping =
            access == MappedFileAccess::PrivateWritable &&
            byte_count >= 64ull * 1024ull * 1024ull;
        if (log_large_private_mapping) {
            tinylog::logger().info("MappedRegion mmap begin", {
                {"path", std::string(p)},
                {"offset", (int64_t)offset},
                {"bytes", (int64_t)byte_count},
                {"access", std::string("private_writable")},
            });
            log_process_memory_snapshot("before large private mmap");
        }
        mapping = mmap(nullptr, mapping_size, prot, flags, fd, static_cast<off_t>(aligned_offset));
        if (mapping == MAP_FAILED) {
            int mmap_errno = errno;
            tinylog::logger().error("MappedRegion mmap failed",
                {{"path", std::string(p)}, {"offset", (int64_t)offset},
                 {"bytes", (int64_t)byte_count},
                 {"access", access == MappedFileAccess::PrivateWritable
                    ? std::string("private_writable")
                    : std::string("read_only")},
                 {"errno", mmap_errno}, {"error", std::strerror(mmap_errno)}});
            std::fprintf(stderr,
                         "MappedRegion mmap failed path=%s offset=%lld bytes=%lld access=%s errno=%d error=%s\n",
                         p, static_cast<long long>(offset),
                         static_cast<long long>(byte_count),
                         access == MappedFileAccess::PrivateWritable
                            ? "private_writable"
                            : "read_only",
                         mmap_errno, std::strerror(mmap_errno));
            mapping = nullptr; data = nullptr; ::close(fd); fd = -1; mapping_size = size = 0;
            access_mode = MappedFileAccess::ReadOnly;
            return false;
        }
        if (log_large_private_mapping) {
            tinylog::logger().info("MappedRegion mmap succeeded", {
                {"path", std::string(p)},
                {"offset", (int64_t)offset},
                {"bytes", (int64_t)byte_count},
                {"access", std::string("private_writable")},
            });
            log_process_memory_snapshot("after large private mmap");
        }
        ::close(fd);
        fd = -1;
        data = static_cast<uint8_t*>(mapping) + delta;
        return true;
    }

    void close() {
        if (mapping && mapping != MAP_FAILED) {
            const bool log_large_private_mapping =
                access_mode == MappedFileAccess::PrivateWritable &&
                mapping_size >= 64ull * 1024ull * 1024ull;
            if (log_large_private_mapping)
                log_process_memory_snapshot("before large private munmap");
            if (access_mode == MappedFileAccess::PrivateWritable)
                madvise(mapping, mapping_size, MADV_DONTNEED);
            munmap(mapping, mapping_size);
            if (log_large_private_mapping)
                log_process_memory_snapshot("after large private munmap");
            mapping = nullptr;
        }
        if (fd >= 0) { ::close(fd); fd = -1; }
        data = nullptr;
        mapping_size = 0;
        size = 0;
        access_mode = MappedFileAccess::ReadOnly;
    }

    ~MappedRegion() { close(); }

    MappedRegion() = default;
    MappedRegion(const MappedRegion&) = delete;
    MappedRegion& operator=(const MappedRegion&) = delete;
    MappedRegion(MappedRegion&& o) noexcept
        : mapping(o.mapping), data(o.data), mapping_size(o.mapping_size), size(o.size),
          fd(o.fd), access_mode(o.access_mode), path(std::move(o.path)) {
        o.mapping = nullptr; o.data = nullptr; o.mapping_size = 0; o.size = 0; o.fd = -1;
        o.access_mode = MappedFileAccess::ReadOnly;
    }
};

// MARK: - .litertlm bundle parser — extracts individual TFLite models by type
// Format: "LITERTLM" magic | version (3×uint32) | 4B pad | header_end (uint64)
//         | FlatBuffer header (section metadata)
// We do a minimal FlatBuffer parse: enough to read SectionObject offsets
// and model_type KeyValuePair strings.

// Known model types within a .litertlm bundle
enum class BundleModelType {
    PREFILL_DECODE,      // "tf_lite_prefill_decode" — main decoder
    EMBEDDER,            // "tf_lite_embedder"
    PER_LAYER_EMBEDDER,  // "tf_lite_per_layer_embedder"
    VISION_ENCODER,      // "tf_lite_vision_encoder"
    VISION_ADAPTER,      // "tf_lite_vision_adapter"
    MTP_DRAFTER,         // "tf_lite_mtp_drafter"
    AUDIO_ENCODER,       // "tf_lite_audio_encoder_hw"
    AUDIO_ADAPTER,       // "tf_lite_audio_adapter"
    END_OF_AUDIO,        // "tf_lite_end_of_audio"
    END_OF_VISION,       // "tf_lite_end_of_vision"
    UNKNOWN
};

enum class BundleSectionDataType {
    TFLITE_MODEL,
    TFLITE_WEIGHTS,
    OTHER
};

static inline const char* bundle_model_type_name(BundleModelType t) {
    switch (t) {
        case BundleModelType::PREFILL_DECODE:     return "PREFILL_DECODE";
        case BundleModelType::EMBEDDER:           return "EMBEDDER";
        case BundleModelType::PER_LAYER_EMBEDDER: return "PER_LAYER_EMBEDDER";
        case BundleModelType::VISION_ENCODER:     return "VISION_ENCODER";
        case BundleModelType::VISION_ADAPTER:     return "VISION_ADAPTER";
        case BundleModelType::MTP_DRAFTER:        return "MTP_DRAFTER";
        case BundleModelType::AUDIO_ENCODER:      return "AUDIO_ENCODER";
        case BundleModelType::AUDIO_ADAPTER:      return "AUDIO_ADAPTER";
        case BundleModelType::END_OF_AUDIO:       return "END_OF_AUDIO";
        case BundleModelType::END_OF_VISION:      return "END_OF_VISION";
        default:                                  return "UNKNOWN";
    }
}

static inline const char* bundle_section_data_type_name(BundleSectionDataType t) {
    switch (t) {
        case BundleSectionDataType::TFLITE_MODEL:   return "TFLITE_MODEL";
        case BundleSectionDataType::TFLITE_WEIGHTS: return "TFLITE_WEIGHTS";
        default:                                    return "OTHER";
    }
}

struct BundleSection {
    BundleModelType type = BundleModelType::UNKNOWN;
    BundleSectionDataType data_type = BundleSectionDataType::OTHER;
    uint8_t raw_data_type = 0;
    uint64_t begin_offset = 0;
    uint64_t end_offset = 0;
    const void* data = nullptr;  // pointer into mapped file
    size_t size = 0;
    std::string backend_constraint;
};

// Minimal FlatBuffer reader for the LiteRTLM header.
// The header is the root table of type LiteRTLMMetaData, which contains
// section_metadata (SectionMetadata) → objects (vector of SectionObject).
// Each SectionObject has: begin_offset, end_offset, data_type, items (KeyValuePairs).
//
// FlatBuffer wire format:
//   - Root: offset to root table at position 0 (4 bytes)
//   - Table: vtable offset (int32, signed), then field data at vtable-specified offsets
//   - Vector: length (uint32), then length × element offsets
//   - String: length (uint32), then UTF-8 data (null-terminated)
//
// We hardcode field indices from the schema (.fbs) above.

namespace detail {

static inline uint32_t fb_u32(const uint8_t* p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}
static inline int32_t fb_i32(const uint8_t* p) {
    int32_t v; memcpy(&v, p, 4); return v;
}
static inline uint64_t fb_u64(const uint8_t* p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}
static inline uint16_t fb_u16(const uint8_t* p) {
    uint16_t v; memcpy(&v, p, 2); return v;
}

// Dereference a FlatBuffer offset: pointer at `base` + offset stored at `base`.
static inline const uint8_t* fb_deref(const uint8_t* base) {
    return base + fb_u32(base);
}

// Read a field from a FlatBuffer table.
// Returns nullptr if the field is not present (vtable entry is 0).
static inline const uint8_t* fb_field(const uint8_t* table, int field_index) {
    int32_t vtable_offset = fb_i32(table);
    const uint8_t* vtable = table - vtable_offset;
    uint16_t vtable_size = fb_u16(vtable);
    int field_byte_offset = 4 + field_index * 2;  // skip vtable_size(2) + table_size(2)
    if (field_byte_offset + 2 > vtable_size) return nullptr;
    uint16_t data_offset = fb_u16(vtable + field_byte_offset);
    if (data_offset == 0) return nullptr;
    return table + data_offset;
}

// Read a FlatBuffer string: returns {ptr, length} or {"", 0} if null.
static inline std::string fb_string(const uint8_t* str_offset_ptr) {
    if (!str_offset_ptr) return {};
    const uint8_t* str = fb_deref(str_offset_ptr);
    uint32_t len = fb_u32(str);
    return std::string(reinterpret_cast<const char*>(str + 4), len);
}

// Read a FlatBuffer vector length + element offset pointers.
struct FbVector {
    const uint8_t* data;
    uint32_t length;
};
static inline FbVector fb_vector(const uint8_t* vec_offset_ptr) {
    if (!vec_offset_ptr) return {nullptr, 0};
    const uint8_t* vec = fb_deref(vec_offset_ptr);
    return {vec + 4, fb_u32(vec)};
}

static inline BundleModelType parse_model_type(const std::string& s) {
    if (s == "TF_LITE_PREFILL_DECODE" || s == "tf_lite_prefill_decode")
        return BundleModelType::PREFILL_DECODE;
    if (s == "TF_LITE_EMBEDDER" || s == "tf_lite_embedder")
        return BundleModelType::EMBEDDER;
    if (s == "TF_LITE_PER_LAYER_EMBEDDER" || s == "tf_lite_per_layer_embedder")
        return BundleModelType::PER_LAYER_EMBEDDER;
    if (s == "TF_LITE_VISION_ENCODER" || s == "tf_lite_vision_encoder")
        return BundleModelType::VISION_ENCODER;
    if (s == "TF_LITE_VISION_ADAPTER" || s == "tf_lite_vision_adapter")
        return BundleModelType::VISION_ADAPTER;
    if (s == "TF_LITE_MTP_DRAFTER" || s == "tf_lite_mtp_drafter")
        return BundleModelType::MTP_DRAFTER;
    if (s == "TF_LITE_AUDIO_ENCODER_HW" || s == "tf_lite_audio_encoder_hw")
        return BundleModelType::AUDIO_ENCODER;
    if (s == "TF_LITE_AUDIO_ADAPTER" || s == "tf_lite_audio_adapter")
        return BundleModelType::AUDIO_ADAPTER;
    if (s == "TF_LITE_END_OF_AUDIO" || s == "tf_lite_end_of_audio")
        return BundleModelType::END_OF_AUDIO;
    if (s == "TF_LITE_END_OF_VISION" || s == "tf_lite_end_of_vision")
        return BundleModelType::END_OF_VISION;
    return BundleModelType::UNKNOWN;
}

}  // namespace detail

// Parse a memory-mapped .litertlm file and return the TFLite model sections.
// The returned sections have `data` pointing into the mapped file buffer.
static inline std::vector<BundleSection> parse_litertlm_bundle(
    const void* file_data, size_t file_size)
{
    std::vector<BundleSection> sections;
    const uint8_t* d = static_cast<const uint8_t*>(file_data);

    // Check magic
    if (file_size < 32 || memcmp(d, "LITERTLM", 8) != 0) {
        tinylog::logger().error("parse_litertlm_bundle: invalid magic");
        return sections;
    }

    // Skip: magic(8) + version(12) + padding(4) = 24 bytes
    // Then: header_end_offset (uint64)
    uint64_t header_end = detail::fb_u64(d + 24);
    if (header_end > file_size) {
        tinylog::logger().error("parse_litertlm_bundle: header_end_offset > file_size",
            {{"header_end", (int64_t)header_end}, {"file_size", (int64_t)file_size}});
        return sections;
    }

    // FlatBuffer starts at byte 32 (after magic+version+pad+offset)
    const uint8_t* fb_base = d + 32;
    size_t fb_size = (size_t)(header_end - 32);
    if (fb_size < 8) return sections;

    // Root table: LiteRTLMMetaData
    const uint8_t* root = detail::fb_deref(fb_base);

    // field 1: section_metadata (field index 1 in LiteRTLMMetaData)
    const uint8_t* sec_meta_ptr = detail::fb_field(root, 1);
    if (!sec_meta_ptr) return sections;
    const uint8_t* sec_meta = detail::fb_deref(sec_meta_ptr);

    // SectionMetadata field 0: objects (vector of SectionObject)
    const uint8_t* objects_ptr = detail::fb_field(sec_meta, 0);
    if (!objects_ptr) return sections;
    auto objects = detail::fb_vector(objects_ptr);

    // AnySectionDataType enum values (from schema)
    constexpr uint8_t DATA_TYPE_TFLITE_MODEL = 3;
    constexpr uint8_t DATA_TYPE_TFLITE_WEIGHTS = 7;

    for (uint32_t i = 0; i < objects.length; i++) {
        // Each element is an offset to a SectionObject table
        const uint8_t* obj = detail::fb_deref(objects.data + i * 4);

        // SectionObject fields:
        //   0: items (vector of KeyValuePair)
        //   1: begin_offset (uint64)
        //   2: end_offset (uint64)
        //   3: data_type (uint8)

        const uint8_t* dtype_ptr = detail::fb_field(obj, 3);
        if (!dtype_ptr) continue;
        uint8_t data_type = *dtype_ptr;

        if (data_type != DATA_TYPE_TFLITE_MODEL &&
            data_type != DATA_TYPE_TFLITE_WEIGHTS) continue;

        const uint8_t* begin_ptr = detail::fb_field(obj, 1);
        const uint8_t* end_ptr = detail::fb_field(obj, 2);
        if (!begin_ptr || !end_ptr) continue;

        uint64_t begin = detail::fb_u64(begin_ptr);
        uint64_t end = detail::fb_u64(end_ptr);
        if (end > file_size || begin >= end) continue;

        BundleSection sec;
        sec.raw_data_type = data_type;
        sec.data_type = data_type == DATA_TYPE_TFLITE_MODEL
            ? BundleSectionDataType::TFLITE_MODEL
            : BundleSectionDataType::TFLITE_WEIGHTS;
        sec.begin_offset = begin;
        sec.end_offset = end;
        sec.data = d + begin;
        sec.size = (size_t)(end - begin);

        // Look for model_type in KeyValuePair items
        const uint8_t* items_ptr = detail::fb_field(obj, 0);
        std::string model_type_str;
        if (items_ptr) {
            auto items = detail::fb_vector(items_ptr);
            for (uint32_t j = 0; j < items.length; j++) {
                const uint8_t* kv = detail::fb_deref(items.data + j * 4);
                // KeyValuePair: field 0 = key (string), field 1 = value (union)
                const uint8_t* key_ptr = detail::fb_field(kv, 0);
                std::string key = detail::fb_string(key_ptr);
                if (key == "model_type" || key == "backend_constraint") {
                    // Value is a union — field 1 is the type byte, field 2 is the offset
                    // For StringValue union member, we need to read the string from it.
                    // Union: type at field index 1 (uint8), value at field index 2 (offset)
                    const uint8_t* val_ptr = detail::fb_field(kv, 2);
                    if (val_ptr) {
                        const uint8_t* val_table = detail::fb_deref(val_ptr);
                        // StringValue table: field 0 = value (string)
                        const uint8_t* str_ptr = detail::fb_field(val_table, 0);
                        std::string value = detail::fb_string(str_ptr);
                        if (key == "model_type") {
                            model_type_str = value;
                            sec.type = detail::parse_model_type(model_type_str);
                        } else {
                            sec.backend_constraint = value;
                        }
                    }
                }
            }
        }

        if (sec.type == BundleModelType::UNKNOWN && !model_type_str.empty()) {
            tinylog::logger().warn("unrecognized bundle section",
                {{"section", (int64_t)i}, {"model_type", model_type_str}, {"bytes", (int64_t)sec.size}});
        }

        sections.push_back(sec);
    }

    tinylog::logger().trace("parse_litertlm_bundle",
        {{"sections", (int64_t)sections.size()}});
    for (auto& sec : sections) {
        tinylog::logger().trace("  bundle section",
            {{"type", (int64_t)sec.type},
             {"data_type", std::string(bundle_section_data_type_name(sec.data_type))},
             {"begin", (int64_t)sec.begin_offset},
             {"end", (int64_t)sec.end_offset}, {"bytes", (int64_t)sec.size}});
    }

    return sections;
}

// Find a section by model type.
static inline const BundleSection* find_bundle_section(
    const std::vector<BundleSection>& sections, BundleModelType type,
    BundleSectionDataType data_type = BundleSectionDataType::TFLITE_MODEL)
{
    for (auto& sec : sections)
        if (sec.type == type && sec.data_type == data_type) return &sec;
    return nullptr;
}

// Find the decoder section: explicit PREFILL_DECODE if present, otherwise
// the largest UNKNOWN section (bundles may omit model_type for the decoder).
static inline const BundleSection* find_decoder_section(
    const std::vector<BundleSection>& sections)
{
    auto* explicit_pd = find_bundle_section(sections, BundleModelType::PREFILL_DECODE);
    if (explicit_pd) return explicit_pd;

    const BundleSection* best = nullptr;
    for (auto& sec : sections) {
        if (sec.type == BundleModelType::UNKNOWN) {
            if (sec.data_type != BundleSectionDataType::TFLITE_MODEL) continue;
            if (!best || sec.size > best->size)
                best = &sec;
        }
    }
    return best;
}

// Find the SP_TOKENIZER section in a mmapped .litertlm bundle.
// Returns {data, size} or {nullptr, 0} if not found.
static inline std::pair<const void*, size_t> find_bundle_tokenizer(
    const void* bundle_data, size_t bundle_size)
{
    const auto* d = reinterpret_cast<const uint8_t*>(bundle_data);
    if (bundle_size < 32 || memcmp(d, "LITERTLM", 8) != 0)
        return {nullptr, 0};

    uint64_t header_end = detail::fb_u64(d + 24);
    if (header_end > bundle_size) return {nullptr, 0};

    const uint8_t* fb_base = d + 32;
    size_t fb_size = (size_t)(header_end - 32);
    if (fb_size < 8) return {nullptr, 0};

    const uint8_t* root = detail::fb_deref(fb_base);
    const uint8_t* sec_meta_ptr = detail::fb_field(root, 1);
    if (!sec_meta_ptr) return {nullptr, 0};
    const uint8_t* sec_meta = detail::fb_deref(sec_meta_ptr);
    const uint8_t* objects_ptr = detail::fb_field(sec_meta, 0);
    if (!objects_ptr) return {nullptr, 0};
    auto objects = detail::fb_vector(objects_ptr);

    constexpr uint8_t DATA_TYPE_SP_TOKENIZER = 4;

    for (uint32_t i = 0; i < objects.length; i++) {
        const uint8_t* obj = detail::fb_deref(objects.data + i * 4);
        const uint8_t* dtype_ptr = detail::fb_field(obj, 3);
        if (!dtype_ptr || *dtype_ptr != DATA_TYPE_SP_TOKENIZER) continue;

        const uint8_t* begin_ptr = detail::fb_field(obj, 1);
        const uint8_t* end_ptr = detail::fb_field(obj, 2);
        if (!begin_ptr || !end_ptr) continue;

        uint64_t begin = detail::fb_u64(begin_ptr);
        uint64_t end = detail::fb_u64(end_ptr);
        if (end > bundle_size || begin >= end) continue;

        return {d + begin, (size_t)(end - begin)};
    }
    return {nullptr, 0};
}

// MARK: - Helpers

static inline bool litert_check(LiteRtStatus s, const char* label) {
    if (s == kLiteRtStatusOk) return true;
    tinylog::logger().error("LiteRT call failed",
        {{"call", std::string(label)}, {"status", (int64_t)s}});
    return false;
}

// MARK: - LiteRT option construction
// One function per LiteRT opaque option identifier. Each takes the full
// (platform, hardware, model) triple so dispatch is explicit and all
// variation for a given identifier lives in one place.

enum class Platform { APPLE_OS, ANDROID_OS, LINUX_OS };
enum class HardwareTarget { CPU, GPU, NPU };
enum class ModelFamily { GEMMA4, GEMMA4_VISION_ENCODER, GEMMA3_EMBEDDING };

struct Gemma4RuntimeConfig {
    // -1 means automatic/default; otherwise LiteRT GPU precision values 0, 1, or 2.
    int gpu_precision = -1;
    // 0 means automatic from model metadata/cap.
    int kv_cache_max_len = 0;
    // -1 means automatic/default, 0 disabled, 1 enabled.
    int constrained_verify_batch = -1;
    bool mtp_enabled = false;
    bool mtp_trust_verify_kv = true;
    bool mtp_adaptive_enabled = true;
    int mtp_adaptive_min_cycles = 4;
    float mtp_adaptive_min_saved_per_cycle = 0.5f;
    bool mtp_trace = false;
};

struct Gemma4DiagnosticConfig {
    bool prefill_by_decode = false;
    // 0 means automatic/full available prefill chunk.
    int prefill_max_chunk = 0;
    bool constrained_verify_trace = false;
    // 0 means automatic/all drafter tokens.
    int constrained_verify_max_accept = 0;
};

static inline bool is_ios_family_platform_build() {
#if defined(__APPLE__) && TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
    return true;
#else
    return false;
#endif
}

struct LiteRtExternalMetalHandles {
    void* device = nullptr;
    void* command_queue = nullptr;
};

#ifdef __APPLE__
static inline void drain_metal_command_queue(const LiteRtExternalMetalHandles* handles) {
    if (!handles || !handles->command_queue) return;
    auto* raw_msg_send = dlsym(RTLD_DEFAULT, "objc_msgSend");
    auto* sel_fn = reinterpret_cast<void* (*)(const char*)>(
        dlsym(RTLD_DEFAULT, "sel_registerName"));
    if (!raw_msg_send || !sel_fn) return;
    using MsgSendNoArgs = void* (*)(void*, void*);
    auto send = reinterpret_cast<MsgSendNoArgs>(raw_msg_send);
    using MsgSendVoid = void (*)(void*, void*);
    auto send_void = reinterpret_cast<MsgSendVoid>(raw_msg_send);
    void* cb = send(handles->command_queue, sel_fn("commandBuffer"));
    if (!cb) return;
    send_void(cb, sel_fn("commit"));
    send_void(cb, sel_fn("waitUntilCompleted"));
}

struct OwnedLiteRtMetalHandles {
    void* metal_framework = nullptr;
    void* device = nullptr;
    void* command_queue = nullptr;
    LiteRtExternalMetalHandles external;

    OwnedLiteRtMetalHandles() = default;
    OwnedLiteRtMetalHandles(const OwnedLiteRtMetalHandles&) = delete;
    OwnedLiteRtMetalHandles& operator=(const OwnedLiteRtMetalHandles&) = delete;

    ~OwnedLiteRtMetalHandles() { release(); }

    bool valid() const {
        return device != nullptr && command_queue != nullptr;
    }

    static void* shared_metal_framework() {
        static void* handle = dlopen(
            "/System/Library/Frameworks/Metal.framework/Metal",
            RTLD_NOW | RTLD_LOCAL);
        return handle;
    }

    struct SharedHandles {
        void* metal_framework = nullptr;
        void* device = nullptr;
        bool attempted = false;
    };

    static SharedHandles& shared_handles() {
        static SharedHandles handles;
        return handles;
    }

    bool create() {
        if (valid()) return true;

        auto& shared = shared_handles();
        if (!shared.device) {
            if (shared.attempted)
                return false;

            shared.attempted = true;
            shared.metal_framework = shared_metal_framework();
            metal_framework = shared.metal_framework;
            if (!metal_framework) {
                release();
                return false;
            }

            auto create_device = reinterpret_cast<void* (*)()>(
                dlsym(metal_framework, "MTLCreateSystemDefaultDevice"));
            if (!create_device) {
                release();
                return false;
            }

            shared.device = create_device();
            if (!shared.device) {
                release();
                return false;
            }
        }

        metal_framework = shared.metal_framework;
        device = shared.device;
        auto* raw_msg_send = dlsym(RTLD_DEFAULT, "objc_msgSend");
        auto* sel_fn = reinterpret_cast<void* (*)(const char*)>(
            dlsym(RTLD_DEFAULT, "sel_registerName"));
        if (!raw_msg_send || !sel_fn) {
            release();
            return false;
        }

        using MsgSendNoArgs = void* (*)(void*, void*);
        auto send_no_args = reinterpret_cast<MsgSendNoArgs>(raw_msg_send);
        command_queue = send_no_args(device, sel_fn("newCommandQueue"));
        if (!command_queue) {
            release();
            return false;
        }

        external.device = device;
        external.command_queue = command_queue;
        return true;
    }

    void drain_gpu() {
        drain_metal_command_queue(valid() ? &external : nullptr);
    }

    void release() {
        // The Metal device is intentionally process-stable through
        // shared_handles(); this per-session command queue is dropped with the
        // LiteRT environment for diagnostics around stale queue/env state.
        command_queue = nullptr;
        device = nullptr;
        external = {};
        metal_framework = nullptr;
    }
};

static thread_local const LiteRtExternalMetalHandles* g_litert_metal_log_handles = nullptr;

static inline void drain_active_litert_metal_queue() {
    drain_metal_command_queue(g_litert_metal_log_handles);
}

struct ScopedLiteRtMetalLogHandles {
    const LiteRtExternalMetalHandles* previous = nullptr;

    explicit ScopedLiteRtMetalLogHandles(const LiteRtExternalMetalHandles* handles) {
        previous = g_litert_metal_log_handles;
        if (handles && handles->device && handles->command_queue)
            g_litert_metal_log_handles = handles;
    }

    ~ScopedLiteRtMetalLogHandles() {
        g_litert_metal_log_handles = previous;
    }
};

static inline uint64_t metal_uint64_property(void* object, const char* selector_name) {
    if (!object || !selector_name) return 0;

    auto* raw_msg_send = dlsym(RTLD_DEFAULT, "objc_msgSend");
    auto* sel_fn = (void* (*)(const char*))dlsym(RTLD_DEFAULT, "sel_registerName");
    if (!raw_msg_send || !sel_fn) return 0;

    void* property_sel = sel_fn(selector_name);
    void* responds_sel = sel_fn("respondsToSelector:");
    if (!property_sel || !responds_sel) return 0;

    using RespondsFn = bool (*)(void*, void*, void*);
    auto responds_to_selector = reinterpret_cast<RespondsFn>(raw_msg_send);
    if (!responds_to_selector(object, responds_sel, property_sel)) return 0;

    using UInt64MsgFn = unsigned long long (*)(void*, void*);
    auto send_uint64 = reinterpret_cast<UInt64MsgFn>(raw_msg_send);
    return static_cast<uint64_t>(send_uint64(object, property_sel));
}

static inline std::string metal_pointer_string(void* value) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%p", value);
    return std::string(buf);
}

static inline void log_metal_memory_snapshot(const char* label) {
    const auto* handles = g_litert_metal_log_handles;
    if (!handles || !handles->device) return;

    uint64_t current_allocated = metal_uint64_property(handles->device, "currentAllocatedSize");
    uint64_t recommended_max =
        metal_uint64_property(handles->device, "recommendedMaxWorkingSetSize");
    uint64_t max_buffer = metal_uint64_property(handles->device, "maxBufferLength");

    tinylog::logger().info("Metal memory snapshot", {
        {"label", std::string(label ? label : "")},
        {"device", metal_pointer_string(handles->device)},
        {"command_queue", metal_pointer_string(handles->command_queue)},
        {"current_allocated_mb", bytes_to_mb(current_allocated)},
        {"recommended_max_working_set_mb", bytes_to_mb(recommended_max)},
        {"max_buffer_mb", bytes_to_mb(max_buffer)},
    });
}
#else
struct ScopedLiteRtMetalLogHandles {
    explicit ScopedLiteRtMetalLogHandles(const LiteRtExternalMetalHandles*) {}
};

static inline void drain_active_litert_metal_queue() {}
static inline void log_metal_memory_snapshot(const char*) {}
#endif

static inline Platform host_platform() {
#ifdef __ANDROID__
    return Platform::ANDROID_OS;
#elif defined(__linux__)
    return Platform::LINUX_OS;
#else
    return Platform::APPLE_OS;
#endif
}

// Register a TOML string as an opaque option with the given identifier.
static inline bool add_toml_options(LiteRtOptions options, const char* identifier,
                                    const std::string& toml) {
    if (toml.empty()) return true;

    char* copy = new char[toml.size() + 1];
    memcpy(copy, toml.c_str(), toml.size() + 1);

    LiteRtOpaqueOptions opaque = nullptr;
    if (litert_check(LITERT(LiteRtCreateOpaqueOptions)(
                          identifier, copy,
                          [](void* p) { delete[] static_cast<char*>(p); },
                          &opaque),
                      "CreateOpaqueOptions")) {
        LITERT(LiteRtAddOpaqueOptions)(options, opaque);
        return true;
    }
    delete[] copy;
    return false;
}

// Build TOML for each opaque option identifier.
// Returns empty string if the identifier should not be registered.

static inline int resolve_num_threads(int num_threads);
static inline int resolve_gpu_upload_threads(int num_threads);
static inline int resolve_gpu_compile_threads(int num_threads);
static inline int resolve_gemma4_gpu_precision(int requested = -1, int fallback = 2);

static inline int resolve_gemma4_gpu_precision(int requested, int fallback) {
    if (requested >= 0 && requested <= 2)
        return requested;
    return fallback;
}

static inline std::string build_gpu_options(Platform platform,
                                            HardwareTarget hw,
                                            ModelFamily model,
                                            int num_threads,
                                            int gemma4_gpu_precision = -1) {
    if (hw != HardwareTarget::GPU) return "";

    std::string toml;

    auto append_preparation_thread_options = [&]() {
        toml += "num_threads_to_upload = " +
                std::to_string(resolve_gpu_upload_threads(num_threads)) + "\n";
        toml += "num_threads_to_compile = " +
                std::to_string(resolve_gpu_compile_threads(num_threads)) + "\n";
    };

    auto append_shared_llm_state_options = [&]() {
        // Keep KV cache/param_tensor buffers external so delegated signatures
        // share state through our buffers instead of delegate-private storage.
        toml += "external_tensors_mode = false\n";
        toml += "external_tensor_patterns = [\"kv_cache_\", \"param_tensor\"]\n";
        toml += "buffer_storage_tensor_patterns = [\"kv_cache_\", \"param_tensor\"]\n";
        toml += "enable_constant_tensors_sharing = true\n";
        toml += "allow_src_quantized_fc_conv_ops = true\n";
        toml += "num_steps_of_command_buffer_preparations = 2\n";
        append_preparation_thread_options();
    };

    switch (model) {
        case ModelFamily::GEMMA4_VISION_ENCODER:
            // Match LiteRT-LM's vision_litert_compiled_model_executor profile.
            // The vision encoder has no decoder KV-cache/param_tensor state.
            toml += "enable_constant_tensors_sharing = true\n";
            toml += "precision = 2\n";
            switch (platform) {
                case Platform::APPLE_OS:
                    toml += "use_metal_argument_buffers = true\n";
                    toml += "prefer_texture_weights = false\n";
                    break;
                case Platform::ANDROID_OS:
                    toml += "prefer_texture_weights = true\n";
                    break;
                case Platform::LINUX_OS:
                    break;
            }
            append_preparation_thread_options();
            return toml;

        case ModelFamily::GEMMA3_EMBEDDING:
            // Minimal GPU options for single-shot embedding inference.
            toml += "enable_infinite_float_capping = true\n";
            toml += "hint_fully_delegated_to_single_delegate = true\n";
            append_preparation_thread_options();
            switch (platform) {
                case Platform::APPLE_OS:
                    toml += "precision = 2\n";
                    toml += "use_metal_argument_buffers = true\n";
                    toml += "prefer_texture_weights = false\n";
                    break;
                case Platform::ANDROID_OS:
                    toml += "precision = 2\n";
                    toml += "prefer_texture_weights = true\n";
                    break;
                case Platform::LINUX_OS:
                    toml += "precision = 2\n";
                    break;
            }
            return toml;

        case ModelFamily::GEMMA4:
            toml += "enable_infinite_float_capping = true\n";
            append_shared_llm_state_options();
            switch (platform) {
                case Platform::APPLE_OS:
                    toml += "use_metal_argument_buffers = true\n";
                    toml += "prefer_texture_weights = false\n";
                    toml += "precision = " +
                            std::to_string(resolve_gemma4_gpu_precision(gemma4_gpu_precision)) +
                            "\n";
                    toml += "hint_fully_delegated_to_single_delegate = true\n";
                    toml += "convert_weights_on_gpu = true\n";
                    toml += "madvise_original_shared_tensors = true\n";
                    break;
                case Platform::ANDROID_OS:
                    toml += "prefer_texture_weights = true\n";
                    toml += "precision = " +
                            std::to_string(resolve_gemma4_gpu_precision(gemma4_gpu_precision)) +
                            "\n";
                    break;
                case Platform::LINUX_OS:
                    toml += "precision = " +
                            std::to_string(resolve_gemma4_gpu_precision(gemma4_gpu_precision)) +
                            "\n";
                    break;
            }
            return toml;
    }

    return "";
}

static inline int resolve_num_threads(int num_threads) {
    if (num_threads > 0) return num_threads;
    // Match LiteRT-LM's default CPU/XNNPACK thread count.
    return 4;
}

static inline int resolve_gpu_upload_threads(int num_threads) {
    if (num_threads > 0) return num_threads;
    // Match LiteRT-LM's default GPU weight upload thread count.
    return 2;
}

static inline int resolve_gpu_compile_threads(int num_threads) {
    if (num_threads > 0) return num_threads;
    // Match LiteRT-LM's default GPU kernel compilation thread count.
    return 1;
}

static inline const char* model_family_name(ModelFamily m);
static inline std::string escape_toml_string(const std::string& raw);

static inline std::string build_xnnpack_options([[maybe_unused]] Platform platform,
                                                HardwareTarget hw,
                                                [[maybe_unused]] ModelFamily model,
                                                int num_threads,
                                                const char* compilation_cache_dir = nullptr,
                                                bool = false) {
    if (hw != HardwareTarget::CPU) return "";

    std::string toml;
    toml += "num_threads = " + std::to_string(resolve_num_threads(num_threads)) + "\n";
    if (compilation_cache_dir && compilation_cache_dir[0]) {
        std::string cache_file = std::string(compilation_cache_dir) + "/xnnpack_weights.cache";
        toml += "weight_cache_file_path = \"" + escape_toml_string(cache_file) + "\"\n";
    }
    return toml;
}

static inline std::string build_google_tensor_options([[maybe_unused]] Platform platform,
                                                      HardwareTarget hw,
                                                      ModelFamily model) {
    if (hw != HardwareTarget::GPU) return "";
    if (model != ModelFamily::GEMMA4) return "";

    // These flags are parsed by LiteRT's google_tensor option family, not by
    // gpu_options. They mirror LiteRT-LM/prototype settings for Gemma4 INT4
    // compilation and large model handling.
    std::string toml;
    toml += "enable_dynamic_range_quantization = true\n";
    toml += "enable_four_bit_compilation = true\n";
    toml += "enable_large_model_support = true\n";
    return toml;
}

static inline const char* platform_name(Platform p) {
    switch (p) {
        case Platform::APPLE_OS:   return "apple";
        case Platform::ANDROID_OS: return "android";
        case Platform::LINUX_OS:   return "linux";
    }
    return "unknown";
}

static inline const char* hw_target_name(HardwareTarget hw) {
    switch (hw) {
        case HardwareTarget::CPU: return "cpu";
        case HardwareTarget::GPU: return "gpu";
        case HardwareTarget::NPU: return "npu";
    }
    return "unknown";
}

static inline HardwareTarget parse_hardware_target(const char* raw) {
    std::string target = raw && *raw ? std::string(raw) : "cpu";
    if (target == "cpu") return HardwareTarget::CPU;
    if (target == "gpu") return HardwareTarget::GPU;
    if (target == "npu") return HardwareTarget::NPU;
    tinylog::logger().warn("Unknown hardware target; falling back to CPU",
        {{"target", target}});
    return HardwareTarget::CPU;
}

static inline const char* model_family_name(ModelFamily m) {
    switch (m) {
        case ModelFamily::GEMMA4:         return "gemma4";
        case ModelFamily::GEMMA4_VISION_ENCODER: return "gemma4_vision_encoder";
        case ModelFamily::GEMMA3_EMBEDDING: return "gemma3_embedding";
    }
    return "unknown";
}

static inline int hw_accelerator_mask_for_target(HardwareTarget hw) {
    switch (hw) {
        case HardwareTarget::CPU:
            return kLiteRtHwAcceleratorCpu;
        case HardwareTarget::GPU:
            return kLiteRtHwAcceleratorGpu;
        case HardwareTarget::NPU:
            return kLiteRtHwAcceleratorNpu;
    }
    return kLiteRtHwAcceleratorNone;
}

static inline int litert_options_accelerator_mask(HardwareTarget hw, ModelFamily model) {
    if (hw == HardwareTarget::GPU) {
        if (model != ModelFamily::GEMMA4 &&
            model != ModelFamily::GEMMA4_VISION_ENCODER) {
            return kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu;
        }
        return kLiteRtHwAcceleratorGpu;
    }
    if (hw == HardwareTarget::NPU && (
            model == ModelFamily::GEMMA3_EMBEDDING ||
            model == ModelFamily::GEMMA4_VISION_ENCODER))
        return kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu;
    if (hw == HardwareTarget::NPU)
        return kLiteRtHwAcceleratorNpu;
    return kLiteRtHwAcceleratorCpu;
}

// Replace newlines with "; " for single-line log output.
static inline std::string toml_oneline(const std::string& toml) {
    std::string out;
    for (size_t i = 0; i < toml.size(); i++) {
        if (toml[i] == '\n') {
            if (i + 1 < toml.size()) out += "; ";
        } else {
            out += toml[i];
        }
    }
    return out;
}

static inline bool cache_path_safe_char(char c) {
    return (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') ||
           c == '.' || c == '_' || c == '-';
}

static inline std::string sanitize_cache_path_component(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (char c : raw) out += cache_path_safe_char(c) ? c : '_';
    return out.empty() ? "default" : out;
}

static inline std::string escape_toml_string(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (char c : raw) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    return out;
}

static inline std::string stable_hex_hash(const std::string& value) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : value) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }

    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(16);
    for (int shift = 60; shift >= 0; shift -= 4)
        out += digits[(hash >> shift) & 0xF];
    return out;
}

static inline std::string stable_sampled_bytes_hash(const void* data, size_t size) {
    uint64_t hash = 1469598103934665603ULL;
    auto mix = [&](uint64_t value) {
        for (int i = 0; i < 8; i++) {
            hash ^= static_cast<unsigned char>((value >> (i * 8)) & 0xff);
            hash *= 1099511628211ULL;
        }
    };
    auto mix_range = [&](const uint8_t* bytes, size_t count) {
        for (size_t i = 0; i < count; i++) {
            hash ^= bytes[i];
            hash *= 1099511628211ULL;
        }
    };

    mix(static_cast<uint64_t>(size));
    if (data && size > 0) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        constexpr size_t kWindow = 64 * 1024;
        if (size <= kWindow * 3) {
            mix_range(bytes, size);
        } else {
            mix_range(bytes, kWindow);
            size_t middle = (size / 2) - (kWindow / 2);
            mix(static_cast<uint64_t>(middle));
            mix_range(bytes + middle, kWindow);
            size_t tail = size - kWindow;
            mix(static_cast<uint64_t>(tail));
            mix_range(bytes + tail, kWindow);
        }
    }

    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(16);
    for (int shift = 60; shift >= 0; shift -= 4)
        out += digits[(hash >> shift) & 0xF];
    return out;
}

static inline std::string pointer_hex(const void* ptr) {
    uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
    static constexpr char digits[] = "0123456789abcdef";
    std::string out = "0x";
    bool started = false;
    for (int shift = static_cast<int>(sizeof(uintptr_t) * 8) - 4; shift >= 0; shift -= 4) {
        unsigned nibble = static_cast<unsigned>((value >> shift) & 0xF);
        if (nibble != 0 || started || shift == 0) {
            out += digits[nibble];
            started = true;
        }
    }
    return out;
}

static inline std::string model_cache_key(const char* model_path) {
    if (!model_path || !model_path[0]) return "unknown_model";

    std::string path(model_path);
    std::filesystem::path fs_path(path);
    std::string filename = fs_path.filename().string();

    int64_t size = 0;
    int64_t modified = 0;
    struct stat st = {};
    if (stat(model_path, &st) == 0) {
        size = static_cast<int64_t>(st.st_size);
        modified = static_cast<int64_t>(st.st_mtime);
    }

    std::string raw = filename + "_" + std::to_string(size) + "_" +
                      std::to_string(modified) + "_" + stable_hex_hash(path);
    return sanitize_cache_path_component(raw);
}

static inline std::string component_cache_dir(const char* cache_root,
                                              Platform platform,
                                              HardwareTarget hw,
                                              ModelFamily model,
                                              const char* label,
                                              const char* model_path) {
    if (!cache_root || !cache_root[0]) return "";

    std::filesystem::path dir(cache_root);
    dir /= platform_name(platform);
    dir /= hw_target_name(hw);
    dir /= model_family_name(model);
    dir /= model_cache_key(model_path);
    dir /= sanitize_cache_path_component(label ? label : "default");

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        tinylog::logger().warn("LiteRT cache directory unavailable", {
            {"path", dir.string()},
            {"error", ec.message()},
        });
        return "";
    }
    return dir.string();
}

static inline LiteRtOptions create_litert_options(Platform platform, HardwareTarget hw,
                                                   ModelFamily model, int num_threads,
                                                   const char* label = "default",
                                                   const char* model_path = nullptr,
                                                   const char* compilation_cache_dir = nullptr,
                                                   int gemma4_gpu_precision = -1) {
    LiteRtOptions options = nullptr;
    if (!litert_check(LITERT(LiteRtCreateOptions)(&options), "LiteRtCreateOptions"))
        return nullptr;

    LITERT(LiteRtSetOptionsHardwareAccelerators)(
        options, (LiteRtHwAccelerators)litert_options_accelerator_mask(hw, model));

    std::string cache_dir = component_cache_dir(compilation_cache_dir, platform, hw, model, label,
                                                model_path);

    std::string gpu_toml = build_gpu_options(
        platform, hw, model, num_threads, gemma4_gpu_precision);
    if (hw == HardwareTarget::GPU && !cache_dir.empty()) {
        gpu_toml += "serialization_dir = \"" + escape_toml_string(cache_dir) + "\"\n";
        std::string cache_key = model_path
            ? std::filesystem::path(model_path).filename().string()
            : "unknown_model";
        gpu_toml += "model_cache_key = \"" + escape_toml_string(cache_key) + "\"\n";
        gpu_toml += "serialize_program_cache = true\n";
        gpu_toml += "serialize_external_tensors = true\n";
    }
    std::string xnnpack_toml = build_xnnpack_options(platform, hw, model, num_threads,
                                                     cache_dir.empty() ? nullptr : cache_dir.c_str());
    std::string google_tensor_toml = build_google_tensor_options(platform, hw, model);

    // Disable delegate clustering for Gemma4 text decoding (matches LiteRT-LM behavior).
    std::string runtime_toml;
    if (model == ModelFamily::GEMMA4)
        runtime_toml = "disable_delegate_clustering = true\n";

    if (!gpu_toml.empty()) add_toml_options(options, "gpu_options", gpu_toml);
    if (!xnnpack_toml.empty()) add_toml_options(options, "xnnpack", xnnpack_toml);
    if (!google_tensor_toml.empty())
        add_toml_options(options, "google_tensor", google_tensor_toml);
    if (!runtime_toml.empty()) add_toml_options(options, "runtime_options_string", runtime_toml);

    // Log all options in one statement
    std::string summary;
    summary += "LiteRT options for ";
    summary += label;
    summary += " [";
    summary += platform_name(platform);
    summary += "/";
    summary += hw_target_name(hw);
    summary += "/";
    summary += model_family_name(model);
    summary += "]:";
    if (!gpu_toml.empty())
        summary += " gpu_options={" + toml_oneline(gpu_toml) + "}";
    if (!xnnpack_toml.empty())
        summary += " xnnpack={" + toml_oneline(xnnpack_toml) + "}";
    if (!google_tensor_toml.empty())
        summary += " google_tensor={" + toml_oneline(google_tensor_toml) + "}";
    if (!runtime_toml.empty())
        summary += " runtime={" + toml_oneline(runtime_toml) + "}";
    if (gpu_toml.empty() && xnnpack_toml.empty() &&
        google_tensor_toml.empty() && runtime_toml.empty())
        summary += " (no opaque options)";
    tinylog::logger().info(summary);

    return options;
}

// Float32 → Float16 conversion (bit-level, no __fp16 dependency).
static inline uint16_t float_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint16_t frac = (bits >> 13) & 0x03FF;
    if (exp <= 0) return sign;          // underflow → zero
    if (exp >= 31) return sign | 0x7C00; // overflow → infinity
    return sign | (uint16_t)(exp << 10) | frac;
}

// Float32 → BFloat16: just the upper 16 bits.
static inline uint16_t float_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

// Write float32 data as 16-bit values (fp16 or bf16) into a tensor buffer.
static inline bool write_16bit_buf(LiteRtTensorBuffer buf, const float* data, size_t count,
                                   LiteRtElementType elem_type) {
    void* host = nullptr;
    if (LITERT(LiteRtLockTensorBuffer)(buf, &host, kLiteRtTensorBufferLockModeWrite) != kLiteRtStatusOk)
        return false;
    auto* out = static_cast<uint16_t*>(host);
    auto convert = (elem_type == kLiteRtElementTypeBFloat16) ? float_to_bf16 : float_to_fp16;
    for (size_t i = 0; i < count; i++)
        out[i] = convert(data[i]);
    LITERT(LiteRtUnlockTensorBuffer)(buf);
    return true;
}

static inline bool write_float_buf(LiteRtTensorBuffer buf, const float* data, size_t count) {
    size_t buffer_size = 0;
    if (LITERT(LiteRtGetTensorBufferSize)(buf, &buffer_size) == kLiteRtStatusOk &&
        buffer_size < count * sizeof(float)) {
        tinylog::logger().error("tensor buffer too small for float write",
            {{"buffer_bytes", (int64_t)buffer_size},
             {"write_bytes", (int64_t)(count * sizeof(float))}});
        return false;
    }
    void* host = nullptr;
    if (LITERT(LiteRtLockTensorBuffer)(buf, &host, kLiteRtTensorBufferLockModeWrite) != kLiteRtStatusOk)
        return false;
    memcpy(host, data, count * sizeof(float));
    LITERT(LiteRtUnlockTensorBuffer)(buf);
    return true;
}

static inline bool write_int_buf(LiteRtTensorBuffer buf, const int32_t* data, size_t count) {
    size_t buffer_size = 0;
    if (LITERT(LiteRtGetTensorBufferSize)(buf, &buffer_size) == kLiteRtStatusOk &&
        buffer_size < count * sizeof(int32_t)) {
        tinylog::logger().error("tensor buffer too small for int write",
            {{"buffer_bytes", (int64_t)buffer_size},
             {"write_bytes", (int64_t)(count * sizeof(int32_t))}});
        return false;
    }
    void* host = nullptr;
    if (LITERT(LiteRtLockTensorBuffer)(buf, &host, kLiteRtTensorBufferLockModeWrite) != kLiteRtStatusOk)
        return false;
    memcpy(host, data, count * sizeof(int32_t));
    LITERT(LiteRtUnlockTensorBuffer)(buf);
    return true;
}

static inline bool read_float_buf(LiteRtTensorBuffer buf, float* out, size_t count) {
    size_t buffer_size = 0;
    if (LITERT(LiteRtGetTensorBufferSize)(buf, &buffer_size) == kLiteRtStatusOk &&
        buffer_size < count * sizeof(float)) {
        tinylog::logger().error("tensor buffer too small for float read",
            {{"buffer_bytes", (int64_t)buffer_size},
             {"read_bytes", (int64_t)(count * sizeof(float))}});
        return false;
    }
    void* host = nullptr;
    if (LITERT(LiteRtLockTensorBuffer)(buf, &host, kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk)
        return false;
    memcpy(out, host, count * sizeof(float));
    LITERT(LiteRtUnlockTensorBuffer)(buf);
    return true;
}

static inline bool read_u8_buf(LiteRtTensorBuffer buf, uint8_t* out, size_t count) {
    size_t buffer_size = 0;
    if (LITERT(LiteRtGetTensorBufferSize)(buf, &buffer_size) == kLiteRtStatusOk &&
        buffer_size < count) {
        tinylog::logger().error("tensor buffer too small for u8 read",
            {{"buffer_bytes", (int64_t)buffer_size},
             {"read_bytes", (int64_t)count}});
        return false;
    }
    void* host = nullptr;
    if (LITERT(LiteRtLockTensorBuffer)(buf, &host, kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk)
        return false;
    memcpy(out, host, count);
    LITERT(LiteRtUnlockTensorBuffer)(buf);
    return true;
}

static inline bool sync_buffer(LiteRtTensorBuffer buf) {
    void* host = nullptr;
    if (LITERT(LiteRtLockTensorBuffer)(buf, &host, kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk)
        return false;
    LITERT(LiteRtUnlockTensorBuffer)(buf);
    return true;
}

// Clear GPU events from output buffers so they can be reused in subsequent
// Run() calls. LiteRT's Run() rejects output buffers that have stale events
// ("Output buffers cannot have events attached"). LiteRT-LM calls
// Duplicate()+ClearEvent() before every Run(); we clear in-place via the C API.
static inline void clear_buffer_events(LiteRtTensorBuffer* bufs, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (!bufs[i]) continue;
        bool has = false;
        if (LITERT(LiteRtHasTensorBufferEvent)(bufs[i], &has) == kLiteRtStatusOk && has)
            LITERT(LiteRtClearTensorBufferEvent)(bufs[i]);
    }
}

// MARK: - Signature discovery

struct SignatureInfo {
    LiteRtParamIndex sig_index;
    LiteRtSignature sig = nullptr;
    std::string key;
    size_t num_inputs;
    size_t num_outputs;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    int input_index_of(const char* name) const {
        for (size_t i = 0; i < input_names.size(); i++)
            if (input_names[i] == name) return (int)i;
        return -1;
    }
    int output_index_of(const char* name) const {
        for (size_t i = 0; i < output_names.size(); i++)
            if (output_names[i] == name) return (int)i;
        return -1;
    }
};

static inline bool discover_signature(LiteRtModel model, LiteRtParamIndex sig_idx,
                                      SignatureInfo& info) {
    LiteRtSignature sig = nullptr;
    if (!litert_check(LITERT(LiteRtGetModelSignature)(model, sig_idx, &sig),
                      "GetModelSignature")) return false;

    LiteRtParamIndex num_in = 0, num_out = 0;
    LITERT(LiteRtGetNumSignatureInputs)(sig, &num_in);
    LITERT(LiteRtGetNumSignatureOutputs)(sig, &num_out);

    info.sig_index = sig_idx;
    info.sig = sig;
    const char* key = nullptr;
    if (LITERT(LiteRtGetSignatureKey)(sig, &key) == kLiteRtStatusOk && key)
        info.key = key;
    else
        info.key.clear();
    info.num_inputs = num_in;
    info.num_outputs = num_out;
    info.input_names.resize(num_in);
    info.output_names.resize(num_out);

    for (LiteRtParamIndex i = 0; i < num_in; i++) {
        const char* name = nullptr;
        LITERT(LiteRtGetSignatureInputName)(sig, i, &name);
        info.input_names[i] = name;
    }
    for (LiteRtParamIndex i = 0; i < num_out; i++) {
        const char* name = nullptr;
        LITERT(LiteRtGetSignatureOutputName)(sig, i, &name);
        info.output_names[i] = name;
    }
    return true;
}

static inline bool discover_signature_by_key(
        LiteRtModel model,
        const char* signature_key,
        SignatureInfo& info) {
    LiteRtParamIndex num_signatures = 0;
    if (!litert_check(LITERT(LiteRtGetNumModelSignatures)(model, &num_signatures),
                      "GetNumModelSignatures"))
        return false;
    for (LiteRtParamIndex i = 0; i < num_signatures; i++) {
        SignatureInfo candidate;
        if (!discover_signature(model, i, candidate))
            return false;
        if (candidate.key == signature_key) {
            info = std::move(candidate);
            return true;
        }
    }
    tinylog::logger().error("signature key not found",
        {{"signature_key", std::string(signature_key ? signature_key : "")}});
    return false;
}

// MARK: - Tensor type helpers

static inline bool get_input_tensor_type(LiteRtSignature sig, LiteRtParamIndex idx,
                                         LiteRtRankedTensorType* out) {
    LiteRtTensor tensor = nullptr;
    if (!litert_check(LITERT(LiteRtGetSignatureInputTensorByIndex)(sig, idx, &tensor),
                      "GetSignatureInputTensorByIndex")) return false;
    return litert_check(LITERT(LiteRtGetRankedTensorType)(tensor, out), "GetRankedTensorType");
}

static inline bool get_output_tensor_type(LiteRtSignature sig, LiteRtParamIndex idx,
                                          LiteRtRankedTensorType* out) {
    LiteRtTensor tensor = nullptr;
    if (!litert_check(LITERT(LiteRtGetSignatureOutputTensorByIndex)(sig, idx, &tensor),
                      "GetSignatureOutputTensorByIndex")) return false;
    return litert_check(LITERT(LiteRtGetRankedTensorType)(tensor, out), "GetRankedTensorType");
}

// MARK: - Buffer allocation helpers
// These take env + compiled_model directly so they work for any model
// (text decoder, embedder, vision encoder, etc.)

static inline LiteRtTensorBuffer alloc_managed_input(LiteRtEnvironment env,
                                                      LiteRtCompiledModel cm,
                                                      const SignatureInfo& sig_info,
                                                      LiteRtParamIndex input_idx,
                                                      const char* name) {
    LiteRtTensorBufferRequirements reqs = nullptr;
    if (LITERT(LiteRtGetCompiledModelInputBufferRequirements)(
            cm, sig_info.sig_index, input_idx, &reqs) != kLiteRtStatusOk) {
        tinylog::logger().error("GetInputBufferRequirements failed", {{"name", std::string(name)}});
        return nullptr;
    }

    LiteRtRankedTensorType tensor_type = {};
    if (!get_input_tensor_type(sig_info.sig, input_idx, &tensor_type)) return nullptr;

    LiteRtLayout runtime_layout = {};
    if (LITERT(LiteRtGetCompiledModelInputTensorLayout)(
            cm, sig_info.sig_index, input_idx,
            &runtime_layout) == kLiteRtStatusOk) {
        tensor_type.layout = runtime_layout;
    }

    LiteRtTensorBuffer buf = nullptr;
    if (LITERT(LiteRtCreateManagedTensorBufferFromRequirements)(
            env, &tensor_type, reqs, &buf) == kLiteRtStatusOk)
        return buf;

    // Fallback: try each supported buffer type
    int num_types = 0;
    LITERT(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes)(reqs, &num_types);
    size_t buf_size = 0;
    LITERT(LiteRtGetTensorBufferRequirementsBufferSize)(reqs, &buf_size);

    for (int i = 0; i < num_types; i++) {
        LiteRtTensorBufferType btype = kLiteRtTensorBufferTypeUnknown;
        LITERT(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType)(reqs, i, &btype);
        if (LITERT(LiteRtCreateManagedTensorBuffer)(
                env, btype, &tensor_type, buf_size, &buf) == kLiteRtStatusOk)
            return buf;
    }

    tinylog::logger().error("alloc_managed_input failed", {{"name", std::string(name)}});
    return nullptr;
}

static inline LiteRtTensorBuffer alloc_managed_output(LiteRtEnvironment env,
                                                       LiteRtCompiledModel cm,
                                                       const SignatureInfo& sig_info,
                                                       LiteRtParamIndex output_idx,
                                                       const char* name) {
    LiteRtRankedTensorType tensor_type = {};
    if (!get_output_tensor_type(sig_info.sig, output_idx, &tensor_type)) return nullptr;

    {
        std::vector<LiteRtLayout> layouts(sig_info.num_outputs);
        if (LITERT(LiteRtGetCompiledModelOutputTensorLayouts)(
                cm, sig_info.sig_index, sig_info.num_outputs,
                layouts.data(), /*update_allocation=*/true) == kLiteRtStatusOk) {
            tensor_type.layout = layouts[output_idx];
        }
    }

    LiteRtTensorBufferRequirements reqs = nullptr;
    if (LITERT(LiteRtGetCompiledModelOutputBufferRequirements)(
            cm, sig_info.sig_index, output_idx, &reqs) != kLiteRtStatusOk) {
        tinylog::logger().error("GetOutputBufferRequirements failed", {{"name", std::string(name)}});
        return nullptr;
    }

    LiteRtTensorBuffer buf = nullptr;
    if (LITERT(LiteRtCreateManagedTensorBufferFromRequirements)(
            env, &tensor_type, reqs, &buf) == kLiteRtStatusOk)
        return buf;

    int num_types = 0;
    LITERT(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes)(reqs, &num_types);
    size_t buf_size = 0;
    LITERT(LiteRtGetTensorBufferRequirementsBufferSize)(reqs, &buf_size);

    for (int i = 0; i < num_types; i++) {
        LiteRtTensorBufferType btype = kLiteRtTensorBufferTypeUnknown;
        LITERT(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType)(reqs, i, &btype);
        if (LITERT(LiteRtCreateManagedTensorBuffer)(
                env, btype, &tensor_type, buf_size, &buf) == kLiteRtStatusOk)
            return buf;
    }

    tinylog::logger().error("alloc_managed_output failed", {{"name", std::string(name)}});
    return nullptr;
}

static inline void destroy_buffer(LiteRtTensorBuffer& buf) {
    if (buf) { LITERT(LiteRtDestroyTensorBuffer)(buf); buf = nullptr; }
}

struct ScopedTensorBufferDuplicates {
    std::vector<LiteRtTensorBuffer> buffers;

    ~ScopedTensorBufferDuplicates() {
        for (auto* buf : buffers) {
            if (buf) LITERT(LiteRtDestroyTensorBuffer)(buf);
        }
    }

    LiteRtTensorBuffer duplicate_or_original(LiteRtTensorBuffer src, const char* label) {
        if (!src) return nullptr;
        LiteRtStatus status = LITERT(LiteRtDuplicateTensorBuffer)(src);
        if (status == kLiteRtStatusOk) {
            buffers.push_back(src);
            return src;
        }
        tinylog::logger().debug("using original LiteRT tensor buffer",
            {{"name", std::string(label ? label : "")},
             {"duplicate_status", (int64_t)status}});
        return src;
    }
};

// MARK: - Sampling

struct TokenProb {
    int id;
    float logit;
};

static inline int sample_topk_topp(float* logits, int vocab_size,
                                   float temperature, int top_k, float top_p,
                                   std::mt19937& rng) {
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    float inv_temp = 1.0f / temperature;
    int k = std::min(top_k, vocab_size);

    static thread_local std::vector<TokenProb> heap;
    heap.resize(k);
    auto heap_cmp = [](const TokenProb& a, const TokenProb& b) { return a.logit > b.logit; };

    for (int i = 0; i < k; i++)
        heap[i] = {i, logits[i] * inv_temp};
    std::make_heap(heap.begin(), heap.end(), heap_cmp);

    for (int i = k; i < vocab_size; i++) {
        float scaled = logits[i] * inv_temp;
        if (scaled > heap[0].logit) {
            std::pop_heap(heap.begin(), heap.end(), heap_cmp);
            heap[k - 1] = {i, scaled};
            std::push_heap(heap.begin(), heap.end(), heap_cmp);
        }
    }

    std::sort_heap(heap.begin(), heap.end(), heap_cmp);

    float max_logit = heap[0].logit;
    static thread_local std::vector<float> probs;
    probs.resize(k);
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        probs[i] = std::exp(heap[i].logit - max_logit);
        sum += probs[i];
    }
    for (int i = 0; i < k; i++) probs[i] /= sum;

    float cumsum = 0.0f;
    int nucleus_size = k;
    for (int i = 0; i < k; i++) {
        cumsum += probs[i];
        if (cumsum >= top_p) { nucleus_size = i + 1; break; }
    }

    sum = 0.0f;
    for (int i = 0; i < nucleus_size; i++) sum += probs[i];
    for (int i = 0; i < nucleus_size; i++) probs[i] /= sum;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float acc = 0.0f;
    for (int i = 0; i < nucleus_size; i++) {
        acc += probs[i];
        if (acc >= r) return heap[i].id;
    }
    return heap[0].id;
}

static inline void extract_topk(const float* logits, int vocab_size,
                                 int k, std::vector<std::pair<int, float>>& out) {
    out.resize(k);
    struct Cand { float logit; int id; };
    auto cmp = [](const Cand& a, const Cand& b) { return a.logit > b.logit; };
    std::vector<Cand> heap(k);
    for (int i = 0; i < k; i++) heap[i] = {logits[i], i};
    std::make_heap(heap.begin(), heap.end(), cmp);
    for (int i = k; i < vocab_size; i++) {
        if (logits[i] > heap[0].logit) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap[k - 1] = {logits[i], i};
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }
    std::sort_heap(heap.begin(), heap.end(), cmp);
    for (int i = 0; i < k; i++) out[i] = {heap[i].id, heap[i].logit};
}

static inline int constrained_sample(float* logits, int vocab_size,
                                      const std::vector<int>& valid_ids,
                                      float temperature, int top_k, float top_p,
                                      std::mt19937& rng) {
    if (valid_ids.size() == 1) return valid_ids[0];

    std::vector<float> saved(valid_ids.size());
    for (size_t i = 0; i < valid_ids.size(); i++)
        saved[i] = logits[valid_ids[i]];

    std::fill(logits, logits + vocab_size, -std::numeric_limits<float>::infinity());

    for (size_t i = 0; i < valid_ids.size(); i++)
        logits[valid_ids[i]] = saved[i];

    return sample_topk_topp(logits, vocab_size, temperature, top_k, top_p, rng);
}

// ---------------------------------------------------------------------------
// Magic number helpers — detect prime-number placeholder dimensions in
// model signatures and build LiteRtMagicNumberConfigs for the environment.
// ---------------------------------------------------------------------------

static inline bool is_magic_number(int64_t n) {
    if (n < 11) return false;
    if (n % 2 == 0) return false;
    for (int64_t i = 3; i <= n / 2; i += 2)
        if (n % i == 0) return false;
    return true;
}

static inline int64_t default_target_number(int64_t magic) {
    constexpr int64_t kBase = 256;
    if (magic > kBase) return (magic / kBase) * kBase;
    int64_t t = kBase;
    while (t > magic) t /= 2;
    return t;
}

// Scan a decoder model's signatures for prime-number placeholder dimensions
// and build LiteRtMagicNumberConfigs for the environment.
// Caller must free the returned pointer with std::free(). Returns nullptr
// if no magic numbers are found.
static inline LiteRtMagicNumberConfigs* build_magic_number_configs(LiteRtModel decoder_model) {
    LiteRtParamIndex num_sigs = 0;
    LITERT(LiteRtGetNumModelSignatures)(decoder_model, &num_sigs);

    int64_t context_length = 0;
    std::vector<int64_t> prefill_lengths;
    int64_t num_output_candidates = 0;

    for (LiteRtParamIndex i = 0; i < num_sigs; i++) {
        LiteRtSignature sig = nullptr;
        if (LITERT(LiteRtGetModelSignature)(decoder_model, i, &sig) != kLiteRtStatusOk)
            continue;
        const char* key = nullptr;
        LITERT(LiteRtGetSignatureKey)(sig, &key);
        if (!key) continue;
        std::string key_str(key);

        bool is_prefill = (key_str.rfind("prefill", 0) == 0);
        bool is_decode = (key_str.rfind("decode", 0) == 0);
        if (!is_prefill && !is_decode) continue;

        // Scan inputs for mask and pos dimensions
        LiteRtParamIndex n_inputs = 0;
        LITERT(LiteRtGetNumSignatureInputs)(sig, &n_inputs);
        for (LiteRtParamIndex j = 0; j < n_inputs; j++) {
            const char* name = nullptr;
            LITERT(LiteRtGetSignatureInputName)(sig, j, &name);
            if (!name) continue;
            std::string n(name);

            bool is_mask = (n.find("mask") != std::string::npos);
            bool is_pos = (n.find("pos") != std::string::npos);
            if (!is_mask && !is_pos) continue;

            LiteRtRankedTensorType ttype = {};
            if (!get_input_tensor_type(sig, j, &ttype)) continue;
            if (ttype.layout.rank < 1) continue;

            // Last dimension
            int32_t dim = ttype.layout.dimensions[ttype.layout.rank - 1];
            if (!is_magic_number(dim)) continue;

            if (is_mask) {
                if (context_length == 0) context_length = dim;
            } else if (is_pos && is_prefill) {
                prefill_lengths.push_back(dim);
            }
        }

        // Scan decode outputs for logits
        if (is_decode) {
            LiteRtParamIndex n_outputs = 0;
            LITERT(LiteRtGetNumSignatureOutputs)(sig, &n_outputs);
            for (LiteRtParamIndex j = 0; j < n_outputs; j++) {
                const char* name = nullptr;
                LITERT(LiteRtGetSignatureOutputName)(sig, j, &name);
                if (!name) continue;
                if (std::string(name).find("logits") == std::string::npos) continue;

                LiteRtRankedTensorType ttype = {};
                if (!get_output_tensor_type(sig, j, &ttype)) continue;
                if (ttype.layout.rank < 1) continue;

                int32_t dim = ttype.layout.dimensions[0];  // first dimension
                if (is_magic_number(dim) && num_output_candidates == 0)
                    num_output_candidates = dim;
            }
        }
    }

    // Sort and deduplicate prefill lengths
    std::sort(prefill_lengths.begin(), prefill_lengths.end());
    prefill_lengths.erase(std::unique(prefill_lengths.begin(), prefill_lengths.end()),
                          prefill_lengths.end());

    if (context_length == 0 && prefill_lengths.empty() && num_output_candidates == 0) {
        tinylog::logger().debug("magic numbers: none found");
        return nullptr;
    }

    // Build configs
    int num_configs = 0;
    if (context_length > 0) num_configs++;
    if (num_output_candidates > 0) num_configs++;
    num_configs += (int)prefill_lengths.size();

    size_t alloc_size = sizeof(LiteRtMagicNumberConfigs) +
                        num_configs * sizeof(LiteRtMagicNumberConfig);
    auto* configs = static_cast<LiteRtMagicNumberConfigs*>(std::malloc(alloc_size));
    configs->num_configs = num_configs;

    int idx = 0;
    if (context_length > 0) {
        configs->configs[idx].magic_number = context_length;
        configs->configs[idx].target_number = default_target_number(context_length);
        configs->configs[idx].signature_prefix = nullptr;  // all signatures in the text model
        tinylog::logger().debug("magic number config",
            {{"idx", (int64_t)idx}, {"kind", std::string("context_length")},
             {"magic", (int64_t)context_length}, {"target", (int64_t)configs->configs[idx].target_number}});
        idx++;
    }
    if (num_output_candidates > 0) {
        configs->configs[idx].magic_number = num_output_candidates;
        configs->configs[idx].target_number = default_target_number(num_output_candidates);
        configs->configs[idx].signature_prefix = "decode";
        tinylog::logger().debug("magic number config",
            {{"idx", (int64_t)idx}, {"kind", std::string("num_output_candidates")},
             {"magic", (int64_t)num_output_candidates}, {"target", (int64_t)configs->configs[idx].target_number}});
        idx++;
    }
    for (size_t p = 0; p < prefill_lengths.size(); p++) {
        configs->configs[idx].magic_number = prefill_lengths[p];
        configs->configs[idx].target_number = default_target_number(prefill_lengths[p]);
        configs->configs[idx].signature_prefix = "prefill";
        tinylog::logger().debug("magic number config",
            {{"idx", (int64_t)idx}, {"kind", std::string("prefill_length")},
             {"magic", (int64_t)prefill_lengths[p]}, {"target", (int64_t)configs->configs[idx].target_number}});
        idx++;
    }

    return configs;
}

static inline bool same_magic_number_config(const LiteRtMagicNumberConfig& a,
                                            const LiteRtMagicNumberConfig& b) {
    if (a.magic_number != b.magic_number ||
        a.target_number != b.target_number)
        return false;
    if (!a.signature_prefix || !b.signature_prefix)
        return a.signature_prefix == b.signature_prefix;
    return std::strcmp(a.signature_prefix, b.signature_prefix) == 0;
}

static inline LiteRtMagicNumberConfigs* merge_magic_number_configs(
        const LiteRtMagicNumberConfigs* first,
        const LiteRtMagicNumberConfigs* second) {
    std::vector<LiteRtMagicNumberConfig> merged;
    bool conflict = false;
    auto append_unique = [&](const LiteRtMagicNumberConfigs* source) {
        if (!source)
            return;
        for (int64_t i = 0; i < source->num_configs; i++) {
            const auto& candidate = source->configs[i];
            bool exists = false;
            for (const auto& existing : merged) {
                if (same_magic_number_config(existing, candidate)) {
                    exists = true;
                    break;
                }
                const bool same_magic = existing.magic_number == candidate.magic_number;
                const bool same_scope =
                    (!existing.signature_prefix || !candidate.signature_prefix)
                        ? existing.signature_prefix == candidate.signature_prefix
                        : std::strcmp(existing.signature_prefix,
                                      candidate.signature_prefix) == 0;
                if (same_magic && same_scope &&
                    existing.target_number != candidate.target_number) {
                    tinylog::logger().error("conflicting LiteRT magic number config", {
                        {"magic", (int64_t)candidate.magic_number},
                        {"first_target", (int64_t)existing.target_number},
                        {"second_target", (int64_t)candidate.target_number},
                        {"signature_prefix", std::string(candidate.signature_prefix
                            ? candidate.signature_prefix : "<all>")},
                    });
                    conflict = true;
                    return;
                }
            }
            if (!exists)
                merged.push_back(candidate);
        }
    };
    append_unique(first);
    append_unique(second);
    if (conflict)
        return nullptr;
    if (merged.empty())
        return nullptr;

    size_t alloc_size = sizeof(LiteRtMagicNumberConfigs) +
                        merged.size() * sizeof(LiteRtMagicNumberConfig);
    auto* configs = static_cast<LiteRtMagicNumberConfigs*>(std::malloc(alloc_size));
    if (!configs)
        return nullptr;
    configs->num_configs = static_cast<int64_t>(merged.size());
    for (size_t i = 0; i < merged.size(); i++)
        configs->configs[i] = merged[i];
    return configs;
}

// Extract filename from a path (last component after '/')
static inline std::string basename_from_path(const char* path) {
    std::string s(path);
    auto pos = s.rfind('/');
    return (pos != std::string::npos) ? s.substr(pos + 1) : s;
}

// MARK: - LiteRT environment setup

static inline std::string get_litert_lib_path() {
    Dl_info info;
    if (dladdr((void*)LITERT(LiteRtCreateEnvironment), &info) && info.dli_fname)
        return std::string(info.dli_fname);
    return "";
}

static inline std::string get_litert_lib_dir() {
    std::string path = get_litert_lib_path();
    auto pos = path.rfind('/');
    if (pos != std::string::npos) return path.substr(0, pos);
    return "";
}

static inline std::string get_litert_frameworks_dir() {
    std::string lib_dir = get_litert_lib_dir();
    std::string suffix = "/libLiteRt.framework";
    if (lib_dir.size() > suffix.size() &&
        lib_dir.compare(lib_dir.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return lib_dir.substr(0, lib_dir.size() - suffix.size());
    }
    return lib_dir;
}

static inline bool litert_path_exists(const std::string& path) {
    struct stat st = {};
    return stat(path.c_str(), &st) == 0;
}

static inline std::string get_litert_accelerator_plugin_dir() {
#ifdef __APPLE__
    std::string framework_dir = get_litert_frameworks_dir() +
        "/libLiteRtMetalAccelerator.framework";
    if (litert_path_exists(framework_dir + "/libLiteRtMetalAccelerator.dylib"))
        return framework_dir;
#endif
    return get_litert_lib_dir();
}

// Workaround for Metal delegate ARC bug in prebuilt libLiteRtMetalAccelerator.
// The prebuilt dylib can create MTLDevice/MTLCommandQueue with insufficient
// lifetime management, causing a dangling pointer SIGSEGV during Metal delegate
// preparation. Keep a LiteRtSession-owned device/queue alive for the whole
// environment and pass those handles to LiteRT whenever Apple GPU is selected.
//
// See: https://github.com/google-ai-edge/LiteRT/issues/6745
// Remove when prebuilt Metal delegates are built with ARC (check by removing
// this block and verifying GPU init doesn't crash on iOS/macOS).
#ifdef __APPLE__
static inline bool add_metal_environment_options(LiteRtEnvOption* opts,
                                                 int capacity,
                                                 int* num_opts,
                                                 const LiteRtExternalMetalHandles* session_handles) {
    if (!opts || !num_opts || *num_opts + 2 > capacity) return false;
    const LiteRtExternalMetalHandles* handles =
        (session_handles && session_handles->device && session_handles->command_queue)
            ? session_handles
            : nullptr;
    if (!handles) return false;

    opts[*num_opts] = {};
    opts[*num_opts].tag = kLiteRtEnvOptionTagMetalDevice;
    opts[*num_opts].value.type = kLiteRtAnyTypeVoidPtr;
    opts[*num_opts].value.ptr_value = handles->device;
    (*num_opts)++;

    opts[*num_opts] = {};
    opts[*num_opts].tag = kLiteRtEnvOptionTagMetalCommandQueue;
    opts[*num_opts].value.type = kLiteRtAnyTypeVoidPtr;
    opts[*num_opts].value.ptr_value = handles->command_queue;
    (*num_opts)++;

    return true;
}
#endif

static inline LiteRtMagicNumberConfigs* no_magic_number_configs();
static inline LiteRtMagicNumberConfigs* active_environment_magic_configs();
static inline bool copy_magic_number_configs(LiteRtMagicNumberConfigs* dst,
                                             const LiteRtMagicNumberConfigs* src);

static inline bool create_environment(LiteRtEnvironment* env,
                                      HardwareTarget hw,
                                      LiteRtMagicNumberConfigs* magic = nullptr,
                                      int accelerator_mask = kLiteRtHwAcceleratorNone,
                                      const LiteRtExternalMetalHandles* metal_handles = nullptr,
                                      const char* accelerator_runtime_library_dir = nullptr) {
    std::string plugin_dir =
        (accelerator_runtime_library_dir && accelerator_runtime_library_dir[0])
            ? std::string(accelerator_runtime_library_dir)
            : get_litert_accelerator_plugin_dir();
    if (accelerator_mask == kLiteRtHwAcceleratorNone)
        accelerator_mask = hw_accelerator_mask_for_target(hw);

    LiteRtMagicNumberConfigs* environment_magic = active_environment_magic_configs();
    if (!copy_magic_number_configs(
            environment_magic, magic ? magic : no_magic_number_configs())) {
        return false;
    }

    LiteRtEnvOption opts[9];
    constexpr int kMaxEnvironmentOptions = sizeof(opts) / sizeof(opts[0]);
    int num_opts = 0;

    opts[num_opts] = {};
    opts[num_opts].tag = kLiteRtEnvOptionTagMinLoggerSeverity;
    opts[num_opts].value.type = kLiteRtAnyTypeInt;
    opts[num_opts].value.int_value = 1; // kLiteRtLogSeverityInfo
    num_opts++;

    if (!plugin_dir.empty()) {
        opts[num_opts] = {};
        opts[num_opts].tag = kLiteRtEnvOptionTagRuntimeLibraryDir;
        opts[num_opts].value.type = kLiteRtAnyTypeString;
        opts[num_opts].value.str_value = plugin_dir.c_str();
        num_opts++;

        // Runtime NPU artifacts are compiled during conversion, so this
        // environment only needs dispatch. Keep compiler plugins out of the
        // app runtime path unless we intentionally add runtime JIT compilation.
        if (accelerator_mask & kLiteRtHwAcceleratorNpu) {
            opts[num_opts] = {};
            opts[num_opts].tag = kLiteRtEnvOptionTagDispatchLibraryDir;
            opts[num_opts].value.type = kLiteRtAnyTypeString;
            opts[num_opts].value.str_value = plugin_dir.c_str();
            num_opts++;
        }
    }
    // LiteRT's default when this option is absent is to auto-register CPU, GPU,
    // and NPU. Always set it explicitly so the selected role choices determine
    // exactly which accelerators are registered and allowed to probe delegates.
    int auto_register_mask = accelerator_mask;
    opts[num_opts] = {};
    opts[num_opts].tag = kLiteRtEnvOptionTagAutoRegisterAccelerators;
    opts[num_opts].value.type = kLiteRtAnyTypeInt;
    opts[num_opts].value.int_value = auto_register_mask;
    num_opts++;
    opts[num_opts] = {};
    opts[num_opts].tag = kLiteRtEnvOptionTagMagicNumberConfigs;
    opts[num_opts].value.type = kLiteRtAnyTypeVoidPtr;
    opts[num_opts].value.ptr_value = environment_magic;
    num_opts++;

#ifdef __APPLE__
    if (accelerator_mask & kLiteRtHwAcceleratorGpu) {
        if (!add_metal_environment_options(opts, kMaxEnvironmentOptions, &num_opts,
                                           metal_handles)) {
            tinylog::logger().warn("Metal environment handles unavailable; GPU delegate will use defaults");
        }
    }
#endif

    tinylog::logger().info("creating LiteRT environment", {
        {"lib", get_litert_lib_path()},
        {"plugin_dir", plugin_dir},
        {"hw", hw_target_name(hw)},
        {"accelerator_mask", std::to_string(accelerator_mask)},
        {"auto_register_mask", std::to_string(auto_register_mask)},
    });

    if (!litert_check(LITERT(LiteRtCreateEnvironment)(
                          num_opts, num_opts > 0 ? opts : nullptr, env),
                      "LiteRtCreateEnvironment"))
        return false;

    return true;
}

static inline LiteRtMagicNumberConfigs* no_magic_number_configs() {
    // LiteRT has no API to remove a previously-added MagicNumberConfigs option
    // from a process-lifetime environment. Use a deliberately unmatched scoped
    // config before and after compiles that should not receive magic-number
    // rewrites, so stale text-model configs cannot leak into embedding/vision
    // helper models.
    static LiteRtMagicNumberConfigs* configs = [] {
        size_t alloc_size = sizeof(LiteRtMagicNumberConfigs) +
                            sizeof(LiteRtMagicNumberConfig);
        auto* value = static_cast<LiteRtMagicNumberConfigs*>(
            std::malloc(alloc_size));
        if (value) {
            value->num_configs = 1;
            value->configs[0].magic_number = 1;
            value->configs[0].target_number = 1;
            value->configs[0].signature_prefix = "__zephr_no_magic__";
        }
        return value;
    }();
    return configs;
}

static inline LiteRtMagicNumberConfigs* active_environment_magic_configs() {
    constexpr int64_t kMaxScopedMagicConfigs = 32;
    static LiteRtMagicNumberConfigs* configs = [] {
        size_t alloc_size = sizeof(LiteRtMagicNumberConfigs) +
                            kMaxScopedMagicConfigs * sizeof(LiteRtMagicNumberConfig);
        auto* value = static_cast<LiteRtMagicNumberConfigs*>(
            std::malloc(alloc_size));
        if (value) {
            value->num_configs = 0;
        }
        return value;
    }();
    return configs;
}

static inline bool copy_magic_number_configs(LiteRtMagicNumberConfigs* dst,
                                             const LiteRtMagicNumberConfigs* src) {
    if (!dst || !src)
        return false;
    constexpr int64_t kMaxScopedMagicConfigs = 32;
    if (src->num_configs < 0 || src->num_configs > kMaxScopedMagicConfigs) {
        tinylog::logger().error("too many LiteRT magic configs",
            {{"count", (int64_t)src->num_configs},
             {"capacity", (int64_t)kMaxScopedMagicConfigs}});
        return false;
    }
    dst->num_configs = src->num_configs;
    for (int64_t i = 0; i < src->num_configs; i++)
        dst->configs[i] = src->configs[i];
    return true;
}

static inline std::string magic_configs_summary(const LiteRtMagicNumberConfigs* magic) {
    if (!magic)
        return "<null>";
    std::ostringstream out;
    out << "ptr=" << static_cast<const void*>(magic)
        << " count=" << magic->num_configs;
    for (int64_t i = 0; i < magic->num_configs; i++) {
        const auto& cfg = magic->configs[i];
        out << " [" << i
            << " magic=" << cfg.magic_number
            << " target=" << cfg.target_number
            << " prefix=" << (cfg.signature_prefix ? cfg.signature_prefix : "<null>")
            << "]";
    }
    return out.str();
}

static inline void log_magic_configs(const char* label,
                                     const char* phase,
                                     const LiteRtMagicNumberConfigs* magic) {
    const std::string summary = magic_configs_summary(magic);
    tinylog::logger().warn("LiteRT magic configs",
        {{"compile", std::string(label ? label : "<unknown>")},
         {"phase", std::string(phase ? phase : "<unknown>")},
         {"configs", summary}});
}

static inline bool create_compiled_model_with_magic(
        LiteRtEnvironment env,
        LiteRtModel model,
        LiteRtOptions options,
        LiteRtCompiledModel* compiled_model,
        LiteRtMagicNumberConfigs* magic_configs,
        const char* label) {
    LiteRtStatus status = kLiteRtStatusErrorInvalidArgument;
    LiteRtMagicNumberConfigs* expected_magic =
        magic_configs ? magic_configs : no_magic_number_configs();
    LiteRtMagicNumberConfigs* active_magic = active_environment_magic_configs();

    log_magic_configs(label, "before_set", expected_magic);
    if (!copy_magic_number_configs(active_magic, expected_magic))
        return false;
    log_magic_configs(label, "after_set", active_magic);

    status = LITERT(LiteRtCreateCompiledModel)(env, model, options, compiled_model);
    const bool compile_ok = litert_check(status, label);
    const bool clear_ok =
        copy_magic_number_configs(active_magic, no_magic_number_configs());
    log_magic_configs("LiteRtClearMagicNumberConfigs", "after_set", active_magic);
    return compile_ok && clear_ok;
}

// MARK: - Diagnostic / result types

struct TokenSetLogEntry {
    int step;
    int valid_count;
    int sampled_id;
    std::vector<std::pair<int, float>> top_logits;  // (id, logit)
};

struct GenerateResult {
    std::string prompt;                          // full formatted prompt sent to model
    std::string response;
    std::map<std::string, std::string> params;  // only for constrained
    std::vector<TokenSetLogEntry> logs;          // only for constrained
    int decode_steps = 0;
    int prefill_tokens = 0;
    int input_ids_count = 0;
    int64_t tokenize_ms = 0;
    int64_t prefill_ms = 0;
    int64_t decode_ms = 0;
    int64_t first_decode_ms = 0;
    int constrained_target_decode_calls = 0;
    int constrained_verify_batches = 0;
    int constrained_verify_fixed_tokens = 0;
    int constrained_verify_rows = 0;
    int64_t constrained_target_decode_ms = 0;
    int64_t constrained_verify_ms = 0;
    int mtp_cycles = 0;
    bool mtp_trust_verify_kv = false;
    bool mtp_adaptive_enabled = false;
    bool mtp_adaptive_disabled = false;
    int mtp_adaptive_disable_cycle = 0;
    int mtp_adaptive_disable_output_tokens = 0;
    int mtp_max_draft_tokens = 0;
    int mtp_target_decode_calls = 0;
    int mtp_drafter_calls = 0;
    int mtp_verify_calls = 0;
    int mtp_draft_tokens = 0;
    int mtp_accepted_tokens = 0;
    int mtp_rejected_cycles = 0;
    int mtp_rejected_after_prefix_0 = 0;
    int mtp_rejected_after_prefix_1 = 0;
    int mtp_rejected_after_prefix_2 = 0;
    int mtp_full_accept_cycles = 0;
    int mtp_shadow_verify_cycles = 0;
    int mtp_replacement_tokens = 0;
    int mtp_bonus_tokens = 0;
    int mtp_fallback_cycles = 0;
    int mtp_rebuilds = 0;
    int mtp_local_repairs = 0;
    int mtp_local_repair_tokens = 0;
    int64_t mtp_target_decode_ms = 0;
    int64_t mtp_drafter_ms = 0;
    int64_t mtp_verify_ms = 0;
    int64_t mtp_rejection_ms = 0;
    int64_t mtp_rebuild_ms = 0;
    int64_t mtp_local_repair_ms = 0;
    int64_t mtp_target_model_us = 0;
    int64_t mtp_target_logits_read_us = 0;
    int64_t mtp_target_activation_read_us = 0;
    int64_t mtp_target_sample_us = 0;
    int64_t mtp_drafter_model_us = 0;
    int64_t mtp_drafter_logits_read_us = 0;
    int64_t mtp_drafter_activation_read_us = 0;
    int64_t mtp_drafter_sample_us = 0;
    int64_t mtp_verify_model_us = 0;
    int64_t mtp_verify_logits_read_us = 0;
    int64_t mtp_verify_activation_read_us = 0;
    int64_t mtp_verify_sample_us = 0;
    int mtp_target_logits_rows_read = 0;
    int mtp_drafter_logits_rows_read = 0;
    int mtp_verify_logits_rows_read = 0;
    int mtp_target_activation_rows_read = 0;
    int mtp_drafter_activation_rows_read = 0;
    int mtp_verify_activation_rows_read = 0;
};

// MARK: - Causal mask helpers

static inline bool write_causal_attention_mask(LiteRtTensorBuffer mask_buf,
                                               LiteRtElementType fallback_type,
                                               int start_step,
                                               int valid_steps) {
    if (!mask_buf) return false;

    LiteRtRankedTensorType tensor_type = {};
    LiteRtElementType mask_type = fallback_type;
    int rows = std::max(1, valid_steps);
    int context = 0;
    int batch_prefix = 1;
    int heads = 1;
    if (LITERT(LiteRtGetTensorBufferTensorType)(mask_buf, &tensor_type) ==
        kLiteRtStatusOk) {
        mask_type = tensor_type.element_type;
        const int rank = (int)tensor_type.layout.rank;
        if (rank >= 2) {
            context = tensor_type.layout.dimensions[rank - 1];
            int row_axis = 0;
            int row_extent = std::max(1, tensor_type.layout.dimensions[0]);
            for (int i = 0; i < rank - 1; i++) {
                const int dim = std::max(1, tensor_type.layout.dimensions[i]);
                if (valid_steps > 1 && dim == valid_steps) {
                    row_axis = i;
                    row_extent = dim;
                    break;
                }
                if (dim > row_extent) {
                    row_axis = i;
                    row_extent = dim;
                }
            }
            rows = row_extent;
            batch_prefix = 1;
            heads = 1;
            for (int i = 0; i < row_axis; i++)
                batch_prefix *= std::max(1, tensor_type.layout.dimensions[i]);
            for (int i = row_axis + 1; i < rank - 1; i++)
                heads *= std::max(1, tensor_type.layout.dimensions[i]);
        } else if (rank == 1) {
            context = tensor_type.layout.dimensions[0];
        }
    }

    size_t byte_size = 0;
    if (LITERT(LiteRtGetTensorBufferSize)(mask_buf, &byte_size) != kLiteRtStatusOk ||
        byte_size == 0)
        return false;

    size_t elem_size = sizeof(float);
    if (mask_type == kLiteRtElementTypeBool) elem_size = sizeof(uint8_t);
    else if (mask_type == kLiteRtElementTypeFloat16 ||
             mask_type == kLiteRtElementTypeBFloat16) elem_size = sizeof(uint16_t);
    else if (mask_type != kLiteRtElementTypeFloat32) return false;

    const size_t total_elems = byte_size / elem_size;
    if (context <= 0) context = (int)total_elems;
    if (context <= 0) return false;
    if ((size_t)batch_prefix * (size_t)rows *
            (size_t)heads * (size_t)context > total_elems)
        batch_prefix = 1;
    if ((size_t)rows * (size_t)heads * (size_t)context > total_elems)
        rows = (int)(total_elems / ((size_t)heads * (size_t)context));
    if (rows <= 0) return false;

    void* host = nullptr;
    if (LITERT(LiteRtLockTensorBuffer)(mask_buf, &host,
                                       kLiteRtTensorBufferLockModeWrite) != kLiteRtStatusOk)
        return false;

    const int fill_rows = std::min(valid_steps, rows);
    if (mask_type == kLiteRtElementTypeBool) {
        auto* mask = static_cast<uint8_t*>(host);
        memset(mask, 0, byte_size);
        for (int b = 0; b < batch_prefix; b++) {
            for (int q = 0; q < fill_rows; q++) {
                const int attend_count = std::min(context, start_step + q + 1);
                if (attend_count <= 0) continue;
                for (int h = 0; h < heads; h++) {
                    uint8_t* row =
                        mask + (((size_t)b * rows + q) * heads + h) * context;
                    memset(row, 1, (size_t)attend_count);
                }
            }
        }
    } else if (mask_type == kLiteRtElementTypeFloat32) {
        auto* mask = static_cast<float*>(host);
        std::fill(mask, mask + total_elems, MASKED_FP32);
        for (int b = 0; b < batch_prefix; b++) {
            for (int q = 0; q < fill_rows; q++) {
                const int attend_count = std::min(context, start_step + q + 1);
                if (attend_count <= 0) continue;
                for (int h = 0; h < heads; h++) {
                    float* row =
                        mask + (((size_t)b * rows + q) * heads + h) * context;
                    std::fill(row, row + attend_count, ATTEND);
                }
            }
        }
    } else {
        auto* mask = static_cast<uint16_t*>(host);
        auto convert =
            (mask_type == kLiteRtElementTypeBFloat16) ? float_to_bf16 : float_to_fp16;
        const uint16_t masked = convert(MASKED_FP16);
        const uint16_t attend = convert(ATTEND);
        std::fill(mask, mask + total_elems, masked);
        for (int b = 0; b < batch_prefix; b++) {
            for (int q = 0; q < fill_rows; q++) {
                const int attend_count = std::min(context, start_step + q + 1);
                if (attend_count <= 0) continue;
                for (int h = 0; h < heads; h++) {
                    uint16_t* row =
                        mask + (((size_t)b * rows + q) * heads + h) * context;
                    std::fill(row, row + attend_count, attend);
                }
            }
        }
    }

    LITERT(LiteRtUnlockTensorBuffer)(mask_buf);
    return true;
}

// Write a causal attention mask for prefill: rows [0..num_tokens) attend
// causally from pos_offset; padding rows are fully masked.
static inline void write_prefill_mask(LiteRtTensorBuffer mask_buf,
                                      LiteRtElementType mask_type, float masked_value,
                                      int seq_len, int kv_cache_max_len,
                                      int num_tokens, int pos_offset) {
    if (write_causal_attention_mask(mask_buf, mask_type, pos_offset, num_tokens))
        return;

    const int mask_count = seq_len * kv_cache_max_len;
    if (mask_type == kLiteRtElementTypeBool) {
        std::vector<uint8_t> bool_flat(mask_count);
        for (int q = 0; q < seq_len; q++) {
            int row_offset = q * kv_cache_max_len;
            int global_pos = pos_offset + q;
            for (int kv = 0; kv < kv_cache_max_len; kv++)
                bool_flat[row_offset + kv] = (q < num_tokens && kv <= global_pos) ? 1 : 0;
        }
        void* host = nullptr;
        if (LITERT(LiteRtLockTensorBuffer)(mask_buf, &host,
                                           kLiteRtTensorBufferLockModeWrite) == kLiteRtStatusOk) {
            memcpy(host, bool_flat.data(), mask_count);
            LITERT(LiteRtUnlockTensorBuffer)(mask_buf);
        }
    } else {
        static thread_local std::vector<float> mask_vec;
        mask_vec.resize(mask_count);
        float* mask = mask_vec.data();
        for (int q = 0; q < seq_len; q++) {
            int row_offset = q * kv_cache_max_len;
            int global_pos = pos_offset + q;
            for (int kv = 0; kv < kv_cache_max_len; kv++)
                mask[row_offset + kv] = (q < num_tokens && kv <= global_pos)
                                         ? ATTEND : masked_value;
        }
        if (mask_type == kLiteRtElementTypeFloat16 || mask_type == kLiteRtElementTypeBFloat16)
            write_16bit_buf(mask_buf, mask, mask_count, mask_type);
        else
            write_float_buf(mask_buf, mask, mask_count);
    }
}

// Write a causal mask for single-token decode at the given position.
static inline void write_decode_mask(LiteRtTensorBuffer mask_buf,
                                     LiteRtElementType mask_type, float masked_value,
                                     int kv_cache_max_len, int pos) {
    if (write_causal_attention_mask(mask_buf, mask_type, pos, 1))
        return;

    if (mask_type == kLiteRtElementTypeBool) {
        std::vector<uint8_t> bool_mask(kv_cache_max_len);
        for (int kv = 0; kv < kv_cache_max_len; kv++)
            bool_mask[kv] = (kv <= pos) ? 1 : 0;
        void* host = nullptr;
        if (LITERT(LiteRtLockTensorBuffer)(mask_buf, &host,
                                           kLiteRtTensorBufferLockModeWrite) == kLiteRtStatusOk) {
            memcpy(host, bool_mask.data(), kv_cache_max_len);
            LITERT(LiteRtUnlockTensorBuffer)(mask_buf);
        }
    } else {
        static thread_local std::vector<float> mask_vec;
        mask_vec.resize(kv_cache_max_len);
        float* mask = mask_vec.data();
        for (int kv = 0; kv < kv_cache_max_len; kv++)
            mask[kv] = (kv <= pos) ? ATTEND : masked_value;
        if (mask_type == kLiteRtElementTypeFloat16 || mask_type == kLiteRtElementTypeBFloat16)
            write_16bit_buf(mask_buf, mask, kv_cache_max_len, mask_type);
        else
            write_float_buf(mask_buf, mask, kv_cache_max_len);
    }
}
