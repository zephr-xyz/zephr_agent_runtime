// TinyLLM bundle inspection helpers.
//
// Kept in tinyllm C++ instead of nanobind so Python, Swift/C, Android/JNI, and
// host diagnostics can all share the same bundle understanding.

#pragma once

#include "tinyllm_core.hpp"

// MARK: - Metadata types

struct TinyLLMTensorMetadata {
    std::string name;
    std::string element_type;
    std::vector<int32_t> shape;
};

struct TinyLLMSignatureMetadata {
    size_t index = 0;
    std::string key;
    std::vector<TinyLLMTensorMetadata> inputs;
    std::vector<TinyLLMTensorMetadata> outputs;
};

struct TinyLLMBundleSectionMetadata {
    std::string type;
    std::string data_type;
    uint64_t begin_offset = 0;
    uint64_t end_offset = 0;
    size_t byte_size = 0;
    bool model_loaded = false;
    std::vector<TinyLLMSignatureMetadata> signatures;
};

static inline const char* tinyllm_element_type_name(LiteRtElementType type) {
    switch (type) {
        case kLiteRtElementTypeFloat32: return "float32";
        case kLiteRtElementTypeInt32: return "int32";
        case kLiteRtElementTypeInt8: return "int8";
        case kLiteRtElementTypeBool: return "bool";
        case kLiteRtElementTypeFloat16: return "float16";
        case kLiteRtElementTypeBFloat16: return "bfloat16";
        case kLiteRtElementTypeNone: return "none";
        default: return "unknown";
    }
}

static inline TinyLLMTensorMetadata tinyllm_tensor_metadata(
        const char* name,
        const LiteRtRankedTensorType& type) {
    TinyLLMTensorMetadata out;
    out.name = name ? name : "";
    out.element_type = tinyllm_element_type_name(type.element_type);
    for (unsigned int i = 0; i < type.layout.rank; i++)
        out.shape.push_back(type.layout.dimensions[i]);
    return out;
}

static inline std::vector<TinyLLMBundleSectionMetadata>
inspect_litertlm_bundle_metadata(const char* path) {
    std::vector<TinyLLMBundleSectionMetadata> result;

    MappedFile mf;
    if (!mf.open(path))
        throw std::runtime_error(std::string("Failed to open: ") + path);
    if (mf.size < 8 || memcmp(mf.data, "LITERTLM", 8) != 0) {
        mf.close();
        throw std::runtime_error(std::string("Not a LiteRT-LM bundle: ") + path);
    }

    auto sections = parse_litertlm_bundle(mf.data, mf.size);
    result.reserve(sections.size());
    for (const auto& section : sections) {
        TinyLLMBundleSectionMetadata section_info;
        section_info.type = bundle_model_type_name(section.type);
        section_info.data_type = bundle_section_data_type_name(section.data_type);
        section_info.begin_offset = section.begin_offset;
        section_info.end_offset = section.end_offset;
        section_info.byte_size = section.size;

        LiteRtModel model = nullptr;
        if (LITERT(LiteRtCreateModelFromBuffer)(
                nullptr, section.data, section.size, &model) != kLiteRtStatusOk) {
            result.push_back(std::move(section_info));
            continue;
        }
        section_info.model_loaded = true;

        LiteRtParamIndex signature_count = 0;
        LITERT(LiteRtGetNumModelSignatures)(model, &signature_count);
        section_info.signatures.reserve(signature_count);
        for (LiteRtParamIndex sig_index = 0; sig_index < signature_count; sig_index++) {
            LiteRtSignature sig = nullptr;
            if (LITERT(LiteRtGetModelSignature)(model, sig_index, &sig) != kLiteRtStatusOk)
                continue;

            TinyLLMSignatureMetadata sig_info;
            sig_info.index = sig_index;
            const char* key = nullptr;
            LITERT(LiteRtGetSignatureKey)(sig, &key);
            sig_info.key = key ? key : "";

            LiteRtParamIndex input_count = 0;
            LITERT(LiteRtGetNumSignatureInputs)(sig, &input_count);
            sig_info.inputs.reserve(input_count);
            for (LiteRtParamIndex i = 0; i < input_count; i++) {
                const char* name = nullptr;
                LITERT(LiteRtGetSignatureInputName)(sig, i, &name);
                LiteRtRankedTensorType type = {};
                if (get_input_tensor_type(sig, i, &type))
                    sig_info.inputs.push_back(tinyllm_tensor_metadata(name, type));
                else
                    sig_info.inputs.push_back({name ? name : "", "unknown", {}});
            }

            LiteRtParamIndex output_count = 0;
            LITERT(LiteRtGetNumSignatureOutputs)(sig, &output_count);
            sig_info.outputs.reserve(output_count);
            for (LiteRtParamIndex i = 0; i < output_count; i++) {
                const char* name = nullptr;
                LITERT(LiteRtGetSignatureOutputName)(sig, i, &name);
                LiteRtRankedTensorType type = {};
                if (get_output_tensor_type(sig, i, &type))
                    sig_info.outputs.push_back(tinyllm_tensor_metadata(name, type));
                else
                    sig_info.outputs.push_back({name ? name : "", "unknown", {}});
            }

            section_info.signatures.push_back(std::move(sig_info));
        }

        LITERT(LiteRtDestroyModel)(model);
        result.push_back(std::move(section_info));
    }

    mf.close();
    return result;
}
