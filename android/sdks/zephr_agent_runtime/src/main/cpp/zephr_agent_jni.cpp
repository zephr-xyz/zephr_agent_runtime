#include <jni.h>

#include <cstdint>
#include <string>
#include <vector>

#include "zephr_agent_c_api.h"

namespace {

template <typename T>
T* from_handle(jlong handle) {
    return reinterpret_cast<T*>(static_cast<uintptr_t>(handle));
}

jlong to_handle(void* ptr) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(ptr));
}

std::string jstring_to_string(JNIEnv* env, jstring value) {
    if (!value) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) return {};
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

const char* nullable_c_str(const std::string& value, jstring original) {
    return original ? value.c_str() : nullptr;
}

std::vector<std::string> string_array_to_vector(JNIEnv* env, jobjectArray array) {
    std::vector<std::string> out;
    if (!array) return out;
    jsize count = env->GetArrayLength(array);
    out.reserve(count);
    for (jsize i = 0; i < count; i++) {
        auto value = static_cast<jstring>(env->GetObjectArrayElement(array, i));
        out.push_back(jstring_to_string(env, value));
        env->DeleteLocalRef(value);
    }
    return out;
}

std::vector<int> int_array_to_vector(JNIEnv* env, jintArray array) {
    std::vector<int> out;
    if (!array) return out;
    jsize count = env->GetArrayLength(array);
    std::vector<jint> raw(count);
    if (count > 0)
        env->GetIntArrayRegion(array, 0, count, raw.data());
    out.reserve(count);
    for (jint value : raw)
        out.push_back((int)value);
    return out;
}

std::vector<bool> bool_array_to_vector(JNIEnv* env, jbooleanArray array) {
    std::vector<bool> out;
    if (!array) return out;
    jsize count = env->GetArrayLength(array);
    std::vector<jboolean> raw(count);
    if (count > 0)
        env->GetBooleanArrayRegion(array, 0, count, raw.data());
    out.reserve(count);
    for (jboolean value : raw)
        out.push_back(value == JNI_TRUE);
    return out;
}

std::vector<std::string> split_enum_values(const std::string& joined) {
    std::vector<std::string> out;
    if (joined.empty()) return out;
    size_t start = 0;
    while (start <= joined.size()) {
        size_t end = joined.find('\x1f', start);
        if (end == std::string::npos) {
            out.push_back(joined.substr(start));
            break;
        }
        out.push_back(joined.substr(start, end - start));
        start = end + 1;
    }
    return out;
}

struct JniTextStreamCallback {
    JNIEnv* env = nullptr;
    jobject callback = nullptr;
    jmethodID on_text = nullptr;
};

bool emit_jni_text_stream(const char* text, void* user_data) {
    auto* callback = static_cast<JniTextStreamCallback*>(user_data);
    if (!callback || !callback->env || !callback->callback || !callback->on_text)
        return true;
    jstring value = callback->env->NewStringUTF(text ? text : "");
    if (!value)
        return false;
    jboolean keep_going = callback->env->CallBooleanMethod(
        callback->callback,
        callback->on_text,
        value);
    callback->env->DeleteLocalRef(value);
    if (callback->env->ExceptionCheck())
        return false;
    return keep_going == JNI_TRUE;
}

struct ToolSpecBuffers {
    std::vector<std::string> names;
    std::vector<std::string> descriptions;
    std::vector<std::string> param_names;
    std::vector<std::string> param_descriptions;
    std::vector<std::string> param_types;
    std::vector<std::vector<zephr_tool_param_spec_t>> params_by_tool;
    std::vector<std::vector<std::string>> enum_storage;
    std::vector<std::vector<const char*>> enum_ptrs;
    std::vector<zephr_tool_spec_t> specs;
};

ToolSpecBuffers build_tool_specs(
        JNIEnv* env,
        jobjectArray tool_names,
        jobjectArray tool_descriptions,
        jintArray param_tool_indexes,
        jobjectArray param_names,
        jobjectArray param_descriptions,
        jobjectArray param_types,
        jbooleanArray param_required,
        jobjectArray param_enum_values) {
    ToolSpecBuffers buffers;
    buffers.names = string_array_to_vector(env, tool_names);
    buffers.descriptions = string_array_to_vector(env, tool_descriptions);
    auto tool_indexes = int_array_to_vector(env, param_tool_indexes);
    buffers.param_names = string_array_to_vector(env, param_names);
    buffers.param_descriptions = string_array_to_vector(env, param_descriptions);
    buffers.param_types = string_array_to_vector(env, param_types);
    auto param_required_values = bool_array_to_vector(env, param_required);
    auto param_enum_joined_values = string_array_to_vector(env, param_enum_values);

    buffers.params_by_tool.resize(buffers.names.size());
    buffers.enum_storage.resize(buffers.param_names.size());
    buffers.enum_ptrs.resize(buffers.param_names.size());

    const size_t param_count = buffers.param_names.size();
    for (size_t i = 0; i < param_count; i++) {
        int tool_index = i < tool_indexes.size() ? tool_indexes[i] : -1;
        if (tool_index < 0 || tool_index >= (int)buffers.names.size())
            continue;

        if (i < param_enum_joined_values.size())
            buffers.enum_storage[i] = split_enum_values(param_enum_joined_values[i]);
        buffers.enum_ptrs[i].reserve(buffers.enum_storage[i].size());
        for (auto& value : buffers.enum_storage[i])
            buffers.enum_ptrs[i].push_back(value.c_str());

        zephr_tool_param_spec_t param = {};
        param.name = buffers.param_names[i].c_str();
        param.description = i < buffers.param_descriptions.size()
            ? buffers.param_descriptions[i].c_str()
            : "";
        param.type = i < buffers.param_types.size()
            ? buffers.param_types[i].c_str()
            : "STRING";
        param.required = i < param_required_values.size()
            ? param_required_values[i]
            : true;
        param.enum_values = buffers.enum_ptrs[i].empty() ? nullptr : buffers.enum_ptrs[i].data();
        param.enum_value_count = (int)buffers.enum_ptrs[i].size();
        buffers.params_by_tool[tool_index].push_back(param);
    }

    buffers.specs.reserve(buffers.names.size());
    for (size_t i = 0; i < buffers.names.size(); i++) {
        zephr_tool_spec_t tool = {};
        tool.name = buffers.names[i].c_str();
        tool.description = i < buffers.descriptions.size() ? buffers.descriptions[i].c_str() : "";
        tool.params = buffers.params_by_tool[i].empty() ? nullptr : buffers.params_by_tool[i].data();
        tool.param_count = (int)buffers.params_by_tool[i].size();
        buffers.specs.push_back(tool);
    }

    return buffers;
}

jstring new_string(JNIEnv* env, const char* value) {
    return env->NewStringUTF(value ? value : "");
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_createAgent(JNIEnv*, jobject) {
    return to_handle(zephr_agent_create());
}

extern "C" JNIEXPORT void JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_destroyAgent(
        JNIEnv*, jobject, jlong agent_handle) {
    if (auto* agent = from_handle<zephr_agent_s>(agent_handle)) {
        zephr_agent_destroy(agent);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_releaseModels(
        JNIEnv*, jobject, jlong agent_handle) {
    if (auto* agent = from_handle<zephr_agent_s>(agent_handle)) {
        zephr_agent_release_models(agent);
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_initAgent(
        JNIEnv* env,
        jobject,
        jlong agent_handle,
        jstring llm_execution_plan,
        jstring rag_execution_plan,
        jstring vlm_execution_plan,
        jboolean gemma4_runtime_mtp_enabled,
        jint gemma4_gpu_precision,
        jint gemma4_kv_cache_max_tokens,
        jint gemma4_constrained_verify_batch,
        jboolean gemma4_mtp_trust_verify_kv,
        jboolean gemma4_mtp_adaptive_enabled,
        jint gemma4_mtp_adaptive_min_cycles,
        jfloat gemma4_mtp_adaptive_min_saved_per_cycle,
        jboolean gemma4_mtp_trace,
        jboolean diagnostic_gemma4_prefill_by_decode,
        jint diagnostic_gemma4_prefill_max_chunk,
        jboolean diagnostic_gemma4_constrained_verify_trace,
        jint diagnostic_gemma4_constrained_verify_max_accept,
        jint num_threads,
        jstring llm_model_path,
        jstring rag_embedding_path,
        jstring vlm_model_path,
        jstring litert_compilation_cache_dir,
        jstring litert_runtime_library_dir) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    if (!agent) return JNI_FALSE;

    std::string llm = jstring_to_string(env, llm_model_path);
    std::string rag_embedding = jstring_to_string(env, rag_embedding_path);
    std::string vlm = jstring_to_string(env, vlm_model_path);
    std::string cache = jstring_to_string(env, litert_compilation_cache_dir);
    std::string runtime_library_dir = jstring_to_string(env, litert_runtime_library_dir);
    std::string llm_plan = jstring_to_string(env, llm_execution_plan);
    std::string rag_plan = jstring_to_string(env, rag_execution_plan);
    std::string vlm_plan = jstring_to_string(env, vlm_execution_plan);

    zephr_agent_execution_config_t execution = {};
    execution.config_version = 1;
    execution.llm_execution_plan = llm_plan.c_str();
    execution.rag_execution_plan = rag_plan.c_str();
    execution.vlm_execution_plan = vlm_plan.c_str();
    execution.gemma4_runtime.gpu_precision = static_cast<int>(gemma4_gpu_precision);
    execution.gemma4_runtime.kv_cache_max_len =
        static_cast<int>(gemma4_kv_cache_max_tokens);
    execution.gemma4_runtime.constrained_verify_batch =
        static_cast<int>(gemma4_constrained_verify_batch);
    execution.gemma4_runtime.mtp_enabled = gemma4_runtime_mtp_enabled == JNI_TRUE;
    execution.gemma4_runtime.mtp_trust_verify_kv =
        gemma4_mtp_trust_verify_kv == JNI_TRUE;
    execution.gemma4_runtime.mtp_adaptive_enabled =
        gemma4_mtp_adaptive_enabled == JNI_TRUE;
    execution.gemma4_runtime.mtp_adaptive_min_cycles =
        static_cast<int>(gemma4_mtp_adaptive_min_cycles);
    execution.gemma4_runtime.mtp_adaptive_min_saved_per_cycle =
        static_cast<float>(gemma4_mtp_adaptive_min_saved_per_cycle);
    execution.gemma4_runtime.mtp_trace = gemma4_mtp_trace == JNI_TRUE;
    execution.diagnostic_gemma4.prefill_by_decode =
        diagnostic_gemma4_prefill_by_decode == JNI_TRUE;
    execution.diagnostic_gemma4.prefill_max_chunk =
        static_cast<int>(diagnostic_gemma4_prefill_max_chunk);
    execution.diagnostic_gemma4.constrained_verify_trace =
        diagnostic_gemma4_constrained_verify_trace == JNI_TRUE;
    execution.diagnostic_gemma4.constrained_verify_max_accept =
        static_cast<int>(diagnostic_gemma4_constrained_verify_max_accept);

    zephr_agent_set_litert_runtime_library_dir(
        agent,
        nullable_c_str(runtime_library_dir, litert_runtime_library_dir));

    return zephr_agent_init(
        agent,
        execution,
        static_cast<int>(num_threads),
        llm.c_str(),
        nullable_c_str(rag_embedding, rag_embedding_path),
        nullable_c_str(vlm, vlm_model_path),
        nullable_c_str(cache, litert_compilation_cache_dir)
    ) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_textCurrentPosition(
        JNIEnv*, jobject, jlong agent_handle) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    return agent ? zephr_agent_text_current_pos(agent) : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_modelLifecycleTimingCount(
        JNIEnv*, jobject, jlong agent_handle) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    return agent ? zephr_agent_model_lifecycle_timing_count(agent) : 0;
}

extern "C" JNIEXPORT jstring JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_modelLifecycleComponent(
        JNIEnv* env, jobject, jlong agent_handle, jint index) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    return new_string(env, agent ? zephr_agent_model_lifecycle_component(agent, index) : "");
}

extern "C" JNIEXPORT jstring JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_modelLifecycleAction(
        JNIEnv* env, jobject, jlong agent_handle, jint index) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    return new_string(env, agent ? zephr_agent_model_lifecycle_action(agent, index) : "");
}

extern "C" JNIEXPORT jstring JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_modelLifecycleDetail(
        JNIEnv* env, jobject, jlong agent_handle, jint index) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    return new_string(env, agent ? zephr_agent_model_lifecycle_detail(agent, index) : "");
}

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_modelLifecycleDurationMs(
        JNIEnv*, jobject, jlong agent_handle, jint index) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    return agent ? zephr_agent_model_lifecycle_duration_ms(agent, index) : 0;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_modelLifecycleOk(
        JNIEnv*, jobject, jlong agent_handle, jint index) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    return agent && zephr_agent_model_lifecycle_ok(agent, index) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_clearModelTimings(
        JNIEnv*, jobject, jlong agent_handle) {
    if (auto* agent = from_handle<zephr_agent_s>(agent_handle)) {
        zephr_agent_model_lifecycle_clear(agent);
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_embedText(
        JNIEnv* env,
        jobject,
        jlong agent_handle,
        jstring text,
        jstring task_type) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    if (!agent || !text) return 0;

    std::string text_string = jstring_to_string(env, text);
    std::string task_string = jstring_to_string(env, task_type);
    return to_handle(zephr_agent_embed_text(
        agent,
        text_string.c_str(),
        task_string.c_str()
    ));
}

extern "C" JNIEXPORT void JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_destroyEmbeddingResult(
        JNIEnv*, jobject, jlong result_handle) {
    if (auto* result = from_handle<zephr_embedding_result_s>(result_handle)) {
        zephr_embedding_result_destroy(result);
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_embeddingDimension(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_embedding_result_s>(result_handle);
    return result ? zephr_embedding_dimension(result) : 0;
}

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_embeddingDurationMs(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_embedding_result_s>(result_handle);
    return result ? zephr_embedding_duration_ms(result) : 0;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_embeddingData(
        JNIEnv* env,
        jobject,
        jlong result_handle) {
    auto* result = from_handle<zephr_embedding_result_s>(result_handle);
    const int dimension = result ? zephr_embedding_dimension(result) : 0;
    jfloatArray out = env->NewFloatArray(dimension);
    if (!out || dimension <= 0)
        return out;
    const float* values = zephr_embedding_data(result);
    if (!values)
        return out;
    env->SetFloatArrayRegion(out, 0, dimension, values);
    return out;
}

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_describeImageRgb888(
        JNIEnv* env,
        jobject,
        jlong agent_handle,
        jbyteArray rgb,
        jint width,
        jint height,
        jint row_stride,
        jstring prompt,
        jint max_tokens) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    if (!agent || !rgb) return 0;

    jsize size = env->GetArrayLength(rgb);
    std::vector<jbyte> buffer(static_cast<size_t>(size));
    env->GetByteArrayRegion(rgb, 0, size, buffer.data());
    if (env->ExceptionCheck()) return 0;

    std::string prompt_string = jstring_to_string(env, prompt);
    return to_handle(zephr_agent_describe_image_rgb888(
        agent,
        reinterpret_cast<const uint8_t*>(buffer.data()),
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<int>(row_stride),
        prompt_string.c_str(),
        static_cast<int>(max_tokens)
    ));
}

extern "C" JNIEXPORT void JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_destroyVisionResult(
        JNIEnv*, jobject, jlong result_handle) {
    if (auto* result = from_handle<zephr_vlm_result_s>(result_handle)) {
        zephr_vlm_result_destroy(result);
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_vlmResponse(
        JNIEnv* env, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_vlm_result_s>(result_handle);
    return new_string(env, result ? zephr_vlm_response(result) : "");
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_vlmInputPatches(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_vlm_result_s>(result_handle);
    return result ? zephr_vlm_input_patches(result) : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_vlmValidVisionTokens(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_vlm_result_s>(result_handle);
    return result ? zephr_vlm_valid_vision_tokens(result) : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_vlmImageTokenSlots(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_vlm_result_s>(result_handle);
    return result ? zephr_vlm_image_token_slots(result) : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_vlmResizedWidth(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_vlm_result_s>(result_handle);
    return result ? zephr_vlm_resized_width(result) : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_vlmResizedHeight(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_vlm_result_s>(result_handle);
    return result ? zephr_vlm_resized_height(result) : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_vlmPromptTokens(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_vlm_result_s>(result_handle);
    return result ? zephr_vlm_prompt_tokens(result) : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_vlmDecodeSteps(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_vlm_result_s>(result_handle);
    return result ? zephr_vlm_decode_steps(result) : 0;
}

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_vlmFirstDecodeMs(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_vlm_result_s>(result_handle);
    return result ? zephr_vlm_first_decode_ms(result) : 0;
}

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_generateText(
        JNIEnv* env,
        jobject,
        jlong agent_handle,
        jstring user_message,
        jstring system_message,
        jint max_tokens,
        jfloat temperature,
        jint top_k,
        jfloat top_p) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    if (!agent) return 0;

    std::string message = jstring_to_string(env, user_message);
    std::string system = jstring_to_string(env, system_message);
    return to_handle(zephr_agent_generate_text(
        agent,
        message.c_str(),
        nullable_c_str(system, system_message),
        static_cast<int>(max_tokens),
        static_cast<float>(temperature),
        static_cast<int>(top_k),
        static_cast<float>(top_p)
    ));
}

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_generateToolAwareText(
        JNIEnv* env,
        jobject,
        jlong agent_handle,
        jstring user_message,
        jstring system_message,
        jint max_tokens,
        jfloat temperature,
        jint top_k,
        jfloat top_p,
        jint reserve_output_tokens,
        jobjectArray tool_names,
        jobjectArray tool_descriptions,
        jintArray param_tool_indexes,
        jobjectArray param_names,
        jobjectArray param_descriptions,
        jobjectArray param_types,
        jbooleanArray param_required,
        jobjectArray param_enum_values) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    if (!agent) return 0;

    std::string message = jstring_to_string(env, user_message);
    std::string system = jstring_to_string(env, system_message);
    auto names = string_array_to_vector(env, tool_names);
    auto descriptions = string_array_to_vector(env, tool_descriptions);
    auto tool_indexes = int_array_to_vector(env, param_tool_indexes);
    auto param_name_values = string_array_to_vector(env, param_names);
    auto param_description_values = string_array_to_vector(env, param_descriptions);
    auto param_type_values = string_array_to_vector(env, param_types);
    auto param_required_values = bool_array_to_vector(env, param_required);
    auto param_enum_joined_values = string_array_to_vector(env, param_enum_values);

    std::vector<std::vector<zephr_tool_param_spec_t>> params_by_tool(names.size());
    std::vector<std::vector<std::string>> enum_storage(param_name_values.size());
    std::vector<std::vector<const char*>> enum_ptrs(param_name_values.size());

    const size_t param_count = param_name_values.size();
    for (size_t i = 0; i < param_count; i++) {
        int tool_index = i < tool_indexes.size() ? tool_indexes[i] : -1;
        if (tool_index < 0 || tool_index >= (int)names.size())
            continue;

        if (i < param_enum_joined_values.size())
            enum_storage[i] = split_enum_values(param_enum_joined_values[i]);
        enum_ptrs[i].reserve(enum_storage[i].size());
        for (auto& value : enum_storage[i])
            enum_ptrs[i].push_back(value.c_str());

        zephr_tool_param_spec_t param = {};
        param.name = param_name_values[i].c_str();
        param.description = i < param_description_values.size()
            ? param_description_values[i].c_str()
            : "";
        param.type = i < param_type_values.size()
            ? param_type_values[i].c_str()
            : "STRING";
        param.required = i < param_required_values.size()
            ? param_required_values[i]
            : true;
        param.enum_values = enum_ptrs[i].empty() ? nullptr : enum_ptrs[i].data();
        param.enum_value_count = (int)enum_ptrs[i].size();
        params_by_tool[tool_index].push_back(param);
    }

    std::vector<zephr_tool_spec_t> tools;
    tools.reserve(names.size());
    for (size_t i = 0; i < names.size(); i++) {
        zephr_tool_spec_t tool = {};
        tool.name = names[i].c_str();
        tool.description = i < descriptions.size() ? descriptions[i].c_str() : "";
        tool.params = params_by_tool[i].empty() ? nullptr : params_by_tool[i].data();
        tool.param_count = (int)params_by_tool[i].size();
        tools.push_back(tool);
    }

    return to_handle(zephr_agent_generate_tool_aware_text(
        agent,
        message.c_str(),
        nullable_c_str(system, system_message),
        static_cast<int>(max_tokens),
        static_cast<float>(temperature),
        static_cast<int>(top_k),
        static_cast<float>(top_p),
        static_cast<int>(reserve_output_tokens),
        tools.empty() ? nullptr : tools.data(),
        static_cast<int>(tools.size())
    ));
}

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_generateToolAwareTextStreaming(
        JNIEnv* env,
        jobject,
        jlong agent_handle,
        jstring user_message,
        jstring system_message,
        jint max_tokens,
        jfloat temperature,
        jint top_k,
        jfloat top_p,
        jint reserve_output_tokens,
        jobjectArray tool_names,
        jobjectArray tool_descriptions,
        jintArray param_tool_indexes,
        jobjectArray param_names,
        jobjectArray param_descriptions,
        jobjectArray param_types,
        jbooleanArray param_required,
        jobjectArray param_enum_values,
        jobject stream_callback) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    if (!agent) return 0;

    std::string message = jstring_to_string(env, user_message);
    std::string system = jstring_to_string(env, system_message);
    auto names = string_array_to_vector(env, tool_names);
    auto descriptions = string_array_to_vector(env, tool_descriptions);
    auto tool_indexes = int_array_to_vector(env, param_tool_indexes);
    auto param_name_values = string_array_to_vector(env, param_names);
    auto param_description_values = string_array_to_vector(env, param_descriptions);
    auto param_type_values = string_array_to_vector(env, param_types);
    auto param_required_values = bool_array_to_vector(env, param_required);
    auto param_enum_joined_values = string_array_to_vector(env, param_enum_values);

    std::vector<std::vector<zephr_tool_param_spec_t>> params_by_tool(names.size());
    std::vector<std::vector<std::string>> enum_storage(param_name_values.size());
    std::vector<std::vector<const char*>> enum_ptrs(param_name_values.size());

    const size_t param_count = param_name_values.size();
    for (size_t i = 0; i < param_count; i++) {
        int tool_index = i < tool_indexes.size() ? tool_indexes[i] : -1;
        if (tool_index < 0 || tool_index >= (int)names.size())
            continue;

        if (i < param_enum_joined_values.size())
            enum_storage[i] = split_enum_values(param_enum_joined_values[i]);
        enum_ptrs[i].reserve(enum_storage[i].size());
        for (auto& value : enum_storage[i])
            enum_ptrs[i].push_back(value.c_str());

        zephr_tool_param_spec_t param = {};
        param.name = param_name_values[i].c_str();
        param.description = i < param_description_values.size()
            ? param_description_values[i].c_str()
            : "";
        param.type = i < param_type_values.size()
            ? param_type_values[i].c_str()
            : "STRING";
        param.required = i < param_required_values.size()
            ? param_required_values[i]
            : true;
        param.enum_values = enum_ptrs[i].empty() ? nullptr : enum_ptrs[i].data();
        param.enum_value_count = (int)enum_ptrs[i].size();
        params_by_tool[tool_index].push_back(param);
    }

    std::vector<zephr_tool_spec_t> tools;
    tools.reserve(names.size());
    for (size_t i = 0; i < names.size(); i++) {
        zephr_tool_spec_t tool = {};
        tool.name = names[i].c_str();
        tool.description = i < descriptions.size() ? descriptions[i].c_str() : "";
        tool.params = params_by_tool[i].empty() ? nullptr : params_by_tool[i].data();
        tool.param_count = (int)params_by_tool[i].size();
        tools.push_back(tool);
    }

    JniTextStreamCallback callback = {};
    callback.env = env;
    callback.callback = stream_callback;
    if (stream_callback) {
        jclass callback_class = env->GetObjectClass(stream_callback);
        callback.on_text = env->GetMethodID(
            callback_class,
            "onText",
            "(Ljava/lang/String;)Z");
        env->DeleteLocalRef(callback_class);
        if (!callback.on_text)
            return 0;
    }

    return to_handle(zephr_agent_generate_tool_aware_text_stream(
        agent,
        message.c_str(),
        nullable_c_str(system, system_message),
        static_cast<int>(max_tokens),
        static_cast<float>(temperature),
        static_cast<int>(top_k),
        static_cast<float>(top_p),
        static_cast<int>(reserve_output_tokens),
        tools.empty() ? nullptr : tools.data(),
        static_cast<int>(tools.size()),
        stream_callback ? emit_jni_text_stream : nullptr,
        stream_callback ? &callback : nullptr
    ));
}

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_generateToolAwareTextFromPromptStreaming(
        JNIEnv* env,
        jobject,
        jlong agent_handle,
        jstring prompt,
        jint max_tokens,
        jfloat temperature,
        jint top_k,
        jfloat top_p,
        jint reserve_output_tokens,
        jobjectArray tool_names,
        jobjectArray tool_descriptions,
        jintArray param_tool_indexes,
        jobjectArray param_names,
        jobjectArray param_descriptions,
        jobjectArray param_types,
        jbooleanArray param_required,
        jobjectArray param_enum_values,
        jobject stream_callback) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    if (!agent) return 0;

    std::string prompt_text = jstring_to_string(env, prompt);
    auto tools = build_tool_specs(
        env,
        tool_names,
        tool_descriptions,
        param_tool_indexes,
        param_names,
        param_descriptions,
        param_types,
        param_required,
        param_enum_values);

    JniTextStreamCallback callback = {};
    callback.env = env;
    callback.callback = stream_callback;
    if (stream_callback) {
        jclass callback_class = env->GetObjectClass(stream_callback);
        callback.on_text = env->GetMethodID(
            callback_class,
            "onText",
            "(Ljava/lang/String;)Z");
        env->DeleteLocalRef(callback_class);
        if (!callback.on_text)
            return 0;
    }

    return to_handle(zephr_agent_generate_tool_aware_text_from_prompt_stream(
        agent,
        prompt_text.c_str(),
        static_cast<int>(max_tokens),
        static_cast<float>(temperature),
        static_cast<int>(top_k),
        static_cast<float>(top_p),
        static_cast<int>(reserve_output_tokens),
        tools.specs.empty() ? nullptr : tools.specs.data(),
        static_cast<int>(tools.specs.size()),
        stream_callback ? emit_jni_text_stream : nullptr,
        stream_callback ? &callback : nullptr
    ));
}

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_generateTextFromPromptStreaming(
        JNIEnv* env,
        jobject,
        jlong agent_handle,
        jstring prompt,
        jint max_tokens,
        jfloat temperature,
        jint top_k,
        jfloat top_p,
        jobject stream_callback) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    if (!agent) return 0;

    std::string prompt_string = jstring_to_string(env, prompt);
    JniTextStreamCallback callback = {};
    callback.env = env;
    callback.callback = stream_callback;
    if (stream_callback) {
        jclass callback_class = env->GetObjectClass(stream_callback);
        callback.on_text = env->GetMethodID(
            callback_class,
            "onText",
            "(Ljava/lang/String;)Z");
        env->DeleteLocalRef(callback_class);
        if (!callback.on_text)
            return 0;
    }

    return to_handle(zephr_agent_generate_text_from_prompt_stream(
        agent,
        prompt_string.c_str(),
        static_cast<int>(max_tokens),
        static_cast<float>(temperature),
        static_cast<int>(top_k),
        static_cast<float>(top_p),
        stream_callback ? emit_jni_text_stream : nullptr,
        stream_callback ? &callback : nullptr
    ));
}

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_continueAfterToolResponseStreaming(
        JNIEnv* env,
        jobject,
        jlong agent_handle,
        jstring tool_response,
        jint max_tokens,
        jfloat temperature,
        jint top_k,
        jfloat top_p,
        jint reserve_output_tokens,
        jobject stream_callback) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    if (!agent) return 0;

    std::string response = jstring_to_string(env, tool_response);
    JniTextStreamCallback callback = {};
    callback.env = env;
    callback.callback = stream_callback;
    if (stream_callback) {
        jclass callback_class = env->GetObjectClass(stream_callback);
        callback.on_text = env->GetMethodID(
            callback_class,
            "onText",
            "(Ljava/lang/String;)Z");
        env->DeleteLocalRef(callback_class);
        if (!callback.on_text)
            return 0;
    }

    return to_handle(zephr_agent_continue_after_tool_response_stream(
        agent,
        response.c_str(),
        static_cast<int>(max_tokens),
        static_cast<float>(temperature),
        static_cast<int>(top_k),
        static_cast<float>(top_p),
        static_cast<int>(reserve_output_tokens),
        stream_callback ? emit_jni_text_stream : nullptr,
        stream_callback ? &callback : nullptr
    ));
}

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_continueToolAwareTextStreaming(
        JNIEnv* env,
        jobject,
        jlong agent_handle,
        jstring prompt_suffix,
        jint max_tokens,
        jfloat temperature,
        jint top_k,
        jfloat top_p,
        jint reserve_output_tokens,
        jobjectArray tool_names,
        jobjectArray tool_descriptions,
        jintArray param_tool_indexes,
        jobjectArray param_names,
        jobjectArray param_descriptions,
        jobjectArray param_types,
        jbooleanArray param_required,
        jobjectArray param_enum_values,
        jobject stream_callback) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    if (!agent) return 0;

    std::string prompt = jstring_to_string(env, prompt_suffix);
    ToolSpecBuffers tool_buffers = build_tool_specs(
        env,
        tool_names,
        tool_descriptions,
        param_tool_indexes,
        param_names,
        param_descriptions,
        param_types,
        param_required,
        param_enum_values);

    JniTextStreamCallback callback = {};
    callback.env = env;
    callback.callback = stream_callback;
    if (stream_callback) {
        jclass callback_class = env->GetObjectClass(stream_callback);
        callback.on_text = env->GetMethodID(
            callback_class,
            "onText",
            "(Ljava/lang/String;)Z");
        env->DeleteLocalRef(callback_class);
        if (!callback.on_text)
            return 0;
    }

    return to_handle(zephr_agent_continue_tool_aware_text_stream(
        agent,
        prompt.c_str(),
        static_cast<int>(max_tokens),
        static_cast<float>(temperature),
        static_cast<int>(top_k),
        static_cast<float>(top_p),
        static_cast<int>(reserve_output_tokens),
        tool_buffers.specs.data(),
        static_cast<int>(tool_buffers.specs.size()),
        stream_callback ? emit_jni_text_stream : nullptr,
        stream_callback ? &callback : nullptr
    ));
}

extern "C" JNIEXPORT void JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_destroyTextResult(
        JNIEnv*, jobject, jlong result_handle) {
    if (auto* result = from_handle<zephr_text_result_s>(result_handle)) {
        zephr_text_result_destroy(result);
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_textResponse(
        JNIEnv* env, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_text_result_s>(result_handle);
    return new_string(env, result ? zephr_text_response(result) : "");
}

extern "C" JNIEXPORT jstring JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_textPrompt(
        JNIEnv* env, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_text_result_s>(result_handle);
    return new_string(env, result ? zephr_text_prompt(result) : "");
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_textInputTokens(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_text_result_s>(result_handle);
    return result ? zephr_text_input_tokens(result) : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_textPrefillTokens(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_text_result_s>(result_handle);
    return result ? zephr_text_prefill_tokens(result) : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_textDecodeSteps(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_text_result_s>(result_handle);
    return result ? zephr_text_decode_steps(result) : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_textMtpRejectedCycles(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_text_result_s>(result_handle);
    return result ? zephr_text_mtp_rejected_cycles(result) : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_textMtpRejectedAfterPrefix0(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_text_result_s>(result_handle);
    return result ? zephr_text_mtp_rejected_after_prefix_0(result) : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_textMtpRejectedAfterPrefix1(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_text_result_s>(result_handle);
    return result ? zephr_text_mtp_rejected_after_prefix_1(result) : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_textMtpRejectedAfterPrefix2(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_text_result_s>(result_handle);
    return result ? zephr_text_mtp_rejected_after_prefix_2(result) : 0;
}

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_textPrefillMs(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_text_result_s>(result_handle);
    return result ? zephr_text_prefill_ms(result) : 0;
}

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_textDecodeMs(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_text_result_s>(result_handle);
    return result ? zephr_text_decode_ms(result) : 0;
}

extern "C" JNIEXPORT jlong JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_textFirstDecodeMs(
        JNIEnv*, jobject, jlong result_handle) {
    auto* result = from_handle<zephr_text_result_s>(result_handle);
    return result ? zephr_text_first_decode_ms(result) : 0;
}

extern "C" JNIEXPORT jstring JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_collectHeavyJson(
        JNIEnv* env,
        jobject,
        jlong agent_handle,
        jstring prompt,
        jboolean collect_activations,
        jint top_k) {
    auto* agent = from_handle<zephr_agent_s>(agent_handle);
    if (!agent) return new_string(env, "{\"error\":\"missing agent\"}");
    std::string prompt_string = jstring_to_string(env, prompt);
    char* json = zephr_agent_collect_heavy_json(
        agent,
        prompt_string.c_str(),
        collect_activations == JNI_TRUE,
        static_cast<int>(top_k));
    if (!json) return new_string(env, "{\"error\":\"heavy collection returned null\"}");
    jstring out = new_string(env, json);
    zephr_free(json);
    return out;
}

extern "C" JNIEXPORT jstring JNICALL
Java_xyz_zephr_sdks_agent_internal_ZephrAgentRuntimeJniBridge_detectModelFamily(
        JNIEnv* env, jobject, jstring path) {
    std::string path_string = jstring_to_string(env, path);
    char* family = zephr_detect_model_family(path_string.c_str());
    if (!family) return new_string(env, "");
    jstring result = new_string(env, family);
    zephr_free(family);
    return result;
}
