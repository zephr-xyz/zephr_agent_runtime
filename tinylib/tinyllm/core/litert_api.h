// Self-contained LiteRT 2.x C API declarations.
//
// Declares only the types and functions we need for TinyLLM native inference.
// All function names match the actual exported symbols in libLiteRt.so / libLiteRt.dylib.
// Opaque handles are forward-declared struct pointers — ABI-compatible with the library.
//
// Why not copy upstream headers: the source tree uses "Lrt" prefix for some functions
// while the compiled .so uses "LiteRt" prefix. Writing our own avoids version mismatches
// and eliminates the header dependency chain (build_config.h, litert_any.h, etc.).

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// MARK: - Status codes

typedef enum {
    kLiteRtStatusOk = 0,
    kLiteRtStatusErrorInvalidArgument = 1,
    kLiteRtStatusErrorMemoryAllocationFailure = 2,
    kLiteRtStatusErrorRuntimeFailure = 3,
    kLiteRtStatusErrorMissingInputTensor = 4,
    kLiteRtStatusErrorUnsupported = 5,
    kLiteRtStatusErrorNotFound = 6,
    kLiteRtStatusErrorTimeoutExpired = 7,
    kLiteRtStatusErrorWrongVersion = 8,
    kLiteRtStatusErrorUnknown = 9,
    kLiteRtStatusErrorAlreadyExists = 10,
    kLiteRtStatusCancelled = 100,
    kLiteRtStatusErrorFileIO = 500,
    kLiteRtStatusErrorInvalidFlatbuffer = 501,
    kLiteRtStatusErrorDynamicLoading = 502,
    kLiteRtStatusErrorSerialization = 503,
    kLiteRtStatusErrorCompilation = 504,
    kLiteRtStatusErrorIndexOOB = 1000,
    kLiteRtStatusErrorInvalidIrType = 1001,
    kLiteRtStatusErrorInvalidGraphInvariant = 1002,
    kLiteRtStatusErrorGraphModification = 1003,
    kLiteRtStatusErrorInvalidToolConfig = 1500,
    kLiteRtStatusLegalizeNoMatch = 2000,
    kLiteRtStatusErrorInvalidLegalization = 2001,
    kLiteRtStatusPatternNoMatch = 3000,
    kLiteRtStatusInvalidTransformation = 3001,
    kLiteRtStatusErrorUnsupportedRuntimeVersion = 4000,
    kLiteRtStatusErrorUnsupportedCompilerVersion = 4001,
    kLiteRtStatusErrorIncompatibleByteCodeVersion = 4002,
    kLiteRtStatusErrorUnsupportedOpShapeInferer = 5000,
    kLiteRtStatusErrorShapeInferenceFailed = 5001,
} LiteRtStatus;

// MARK: - Opaque handle types (pointer to forward-declared struct)

typedef struct LiteRtEnvironmentT* LiteRtEnvironment;
typedef struct LiteRtModelT* LiteRtModel;
typedef struct LiteRtOptionsT* LiteRtOptions;
typedef struct LiteRtCompiledModelT* LiteRtCompiledModel;
typedef struct LiteRtTensorBufferT* LiteRtTensorBuffer;
typedef struct LiteRtTensorBufferRequirementsT* LiteRtTensorBufferRequirements;
typedef struct LiteRtOpaqueOptionsT* LiteRtOpaqueOptions;
typedef struct LiteRtSignatureT* LiteRtSignature;
typedef struct LiteRtTensorT* LiteRtTensor;
typedef struct LiteRtSubgraphT* LiteRtSubgraph;

typedef size_t LiteRtParamIndex;

// MARK: - Hardware accelerator flags

typedef enum {
    kLiteRtHwAcceleratorNone = 0,
    kLiteRtHwAcceleratorCpu = 1 << 0,
    kLiteRtHwAcceleratorGpu = 1 << 1,
    kLiteRtHwAcceleratorNpu = 1 << 2,
} LiteRtHwAccelerators;

typedef int LiteRtHwAcceleratorSet;

// MARK: - Delegate precision

typedef enum {
    kLiteRtDelegatePrecisionDefault = 0,
    kLiteRtDelegatePrecisionFp16 = 1,
    kLiteRtDelegatePrecisionFp32 = 2,
} LiteRtDelegatePrecision;

// MARK: - Tensor buffer lock mode

typedef enum {
    kLiteRtTensorBufferLockModeRead = 0,
    kLiteRtTensorBufferLockModeWrite = 1,
    kLiteRtTensorBufferLockModeReadWrite = 2,
} LiteRtTensorBufferLockMode;

// MARK: - Layout (ABI-stable struct, must match upstream exactly)

#define LITERT_TENSOR_MAX_RANK 8

typedef struct {
    unsigned int rank : 7;
    bool has_strides : 1;
    int32_t dimensions[LITERT_TENSOR_MAX_RANK];
    uint32_t strides[LITERT_TENSOR_MAX_RANK];
} LiteRtLayout;

// MARK: - Element types

typedef enum {
    kLiteRtElementTypeNone = 0,
    kLiteRtElementTypeFloat32 = 1,
    kLiteRtElementTypeInt32 = 2,
    kLiteRtElementTypeUInt8 = 3,
    kLiteRtElementTypeInt64 = 4,
    kLiteRtElementTypeTfString = 5,
    kLiteRtElementTypeBool = 6,
    kLiteRtElementTypeInt16 = 7,
    kLiteRtElementTypeComplex64 = 8,
    kLiteRtElementTypeInt8 = 9,
    kLiteRtElementTypeFloat16 = 10,
    kLiteRtElementTypeFloat64 = 11,
    kLiteRtElementTypeComplex128 = 12,
    kLiteRtElementTypeUInt64 = 13,
    kLiteRtElementTypeTfResource = 14,
    kLiteRtElementTypeTfVariant = 15,
    kLiteRtElementTypeUInt32 = 16,
    kLiteRtElementTypeUInt16 = 17,
    kLiteRtElementTypeInt4 = 18,
    kLiteRtElementTypeBFloat16 = 19,
    kLiteRtElementTypeInt2 = 20,
} LiteRtElementType;

// MARK: - Ranked tensor type (ABI-stable struct)

typedef struct {
    LiteRtElementType element_type;
    LiteRtLayout layout;
} LiteRtRankedTensorType;

// MARK: - Tensor buffer types

typedef enum {
    kLiteRtTensorBufferTypeUnknown = 0,
    kLiteRtTensorBufferTypeHostMemory = 1,
    kLiteRtTensorBufferTypeAhwb = 2,
    kLiteRtTensorBufferTypeIon = 3,
    kLiteRtTensorBufferTypeDmaBuf = 4,
    kLiteRtTensorBufferTypeFastRpc = 5,
    kLiteRtTensorBufferTypeGlBuffer = 6,
    kLiteRtTensorBufferTypeGlTexture = 7,
    kLiteRtTensorBufferTypeOpenClBuffer = 10,
    kLiteRtTensorBufferTypeOpenClBufferFp16 = 11,
    kLiteRtTensorBufferTypeOpenClTexture = 12,
    kLiteRtTensorBufferTypeOpenClTextureFp16 = 13,
    kLiteRtTensorBufferTypeOpenClBufferPacked = 14,
    kLiteRtTensorBufferTypeOpenClImageBuffer = 15,
    kLiteRtTensorBufferTypeOpenClImageBufferFp16 = 16,
} LiteRtTensorBufferType;

#define LITERT_HOST_MEMORY_BUFFER_ALIGNMENT 64

typedef void (*LiteRtHostMemoryDeallocator)(void* addr);

// MARK: - Environment option (for LiteRtCreateEnvironment)

typedef enum {
    kLiteRtAnyTypeNone = 0,
    kLiteRtAnyTypeBool = 1,
    kLiteRtAnyTypeInt = 2,
    kLiteRtAnyTypeReal = 3,
    kLiteRtAnyTypeString = 8,
    kLiteRtAnyTypeVoidPtr = 9,
} LiteRtAnyType;

typedef struct {
    LiteRtAnyType type;
    union {
        bool bool_value;
        int64_t int_value;
        double real_value;
        const char* str_value;
        const void* ptr_value;
    };
} LiteRtAny;

typedef enum {
    kLiteRtEnvOptionTagCompilerPluginLibraryDir = 0,
    kLiteRtEnvOptionTagDispatchLibraryDir = 1,
    kLiteRtEnvOptionTagMetalDevice = 10,
    kLiteRtEnvOptionTagMetalCommandQueue = 11,
    kLiteRtEnvOptionTagMagicNumberConfigs = 16,
    kLiteRtEnvOptionTagRuntimeLibraryDir = 22,
    kLiteRtEnvOptionTagAutoRegisterAccelerators = 24,
    kLiteRtEnvOptionTagMinLoggerSeverity = 25,
} LiteRtEnvOptionTag;

typedef struct {
    LiteRtEnvOptionTag tag;
    LiteRtAny value;
} LiteRtEnvOption;

// MARK: - Magic number configs (for GPU delegate tensor shape resolution)

typedef struct {
    int64_t magic_number;
    int64_t target_number;
    const char* signature_prefix;  // null = all signatures
} LiteRtMagicNumberConfig;

// Flexible array — allocate with malloc(sizeof header + N * sizeof element)
typedef struct {
    int64_t num_configs;
    LiteRtMagicNumberConfig configs[];
} LiteRtMagicNumberConfigs;

// ===========================================================================
// Function declarations — names match libLiteRt exported symbols
// ===========================================================================

// --- Environment ---
LiteRtStatus LiteRtCreateEnvironment(int num_options,
                                     const LiteRtEnvOption* options,
                                     LiteRtEnvironment* environment);
void LiteRtDestroyEnvironment(LiteRtEnvironment environment);

// --- Model ---
LiteRtStatus LiteRtCreateModelFromFile(LiteRtEnvironment environment,
                                       const char* filename,
                                       LiteRtModel* model);
LiteRtStatus LiteRtCreateModelFromBuffer(LiteRtEnvironment environment,
                                         const void* buffer_addr,
                                         size_t buffer_size,
                                         LiteRtModel* model);
void LiteRtDestroyModel(LiteRtModel model);

// --- Options ---
LiteRtStatus LiteRtCreateOptions(LiteRtOptions* options);
void LiteRtDestroyOptions(LiteRtOptions options);
LiteRtStatus LiteRtSetOptionsHardwareAccelerators(
    LiteRtOptions options, LiteRtHwAcceleratorSet hw);
LiteRtStatus LiteRtAddOpaqueOptions(LiteRtOptions options,
                                    LiteRtOpaqueOptions opaque_options);

// --- Opaque Options ---
LiteRtStatus LiteRtCreateOpaqueOptions(
    const char* payload_identifier, void* payload_data,
    void (*payload_destructor)(void* payload_data),
    LiteRtOpaqueOptions* options);

// --- Compiled Model ---
LiteRtStatus LiteRtCreateCompiledModel(LiteRtEnvironment environment,
                                       LiteRtModel model,
                                       LiteRtOptions compilation_options,
                                       LiteRtCompiledModel* compiled_model);
void LiteRtDestroyCompiledModel(LiteRtCompiledModel compiled_model);

LiteRtStatus LiteRtGetCompiledModelInputBufferRequirements(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex input_index,
    LiteRtTensorBufferRequirements* buffer_requirements);
LiteRtStatus LiteRtGetCompiledModelOutputBufferRequirements(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex output_index,
    LiteRtTensorBufferRequirements* buffer_requirements);

LiteRtStatus LiteRtGetCompiledModelInputTensorLayout(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex input_index, LiteRtLayout* layout);
LiteRtStatus LiteRtGetCompiledModelOutputTensorLayouts(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    size_t num_outputs, LiteRtLayout* layouts, bool update_allocation);

LiteRtStatus LiteRtRunCompiledModel(LiteRtCompiledModel compiled_model,
                                    LiteRtParamIndex signature_index,
                                    size_t num_input_buffers,
                                    LiteRtTensorBuffer* input_buffers,
                                    size_t num_output_buffers,
                                    LiteRtTensorBuffer* output_buffers);

// --- Tensor Buffers ---
LiteRtStatus LiteRtCreateTensorBufferFromHostMemory(
    const LiteRtRankedTensorType* tensor_type, void* host_buffer_addr,
    size_t host_buffer_size, LiteRtHostMemoryDeallocator deallocator,
    LiteRtTensorBuffer* buffer);
LiteRtStatus LiteRtCreateManagedTensorBufferFromRequirements(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements requirements, LiteRtTensorBuffer* buffer);
LiteRtStatus LiteRtLockTensorBuffer(LiteRtTensorBuffer buffer,
                                    void** host_addr,
                                    LiteRtTensorBufferLockMode lock_mode);
LiteRtStatus LiteRtUnlockTensorBuffer(LiteRtTensorBuffer buffer);
LiteRtStatus LiteRtGetTensorBufferHostMemory(LiteRtTensorBuffer buffer,
                                             void** host_memory_addr);
void LiteRtDestroyTensorBuffer(LiteRtTensorBuffer buffer);
LiteRtStatus LiteRtDuplicateTensorBuffer(LiteRtTensorBuffer buffer);
LiteRtStatus LiteRtGetTensorBufferSize(LiteRtTensorBuffer buffer,
                                       size_t* size);
LiteRtStatus LiteRtGetTensorBufferType(LiteRtTensorBuffer buffer,
                                       LiteRtTensorBufferType* buffer_type);
LiteRtStatus LiteRtGetTensorBufferTensorType(LiteRtTensorBuffer buffer,
                                              LiteRtRankedTensorType* tensor_type);

// --- Tensor Buffer Events (GPU synchronization) ---
LiteRtStatus LiteRtHasTensorBufferEvent(LiteRtTensorBuffer tensor_buffer,
                                        bool* has_event);
LiteRtStatus LiteRtClearTensorBufferEvent(LiteRtTensorBuffer tensor_buffer);

// --- Managed Buffer (explicit type) ---
LiteRtStatus LiteRtCreateManagedTensorBuffer(
    LiteRtEnvironment env, LiteRtTensorBufferType buffer_type,
    const LiteRtRankedTensorType* tensor_type, size_t buffer_size,
    LiteRtTensorBuffer* buffer);

// --- Buffer Requirements ---
LiteRtStatus LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
    LiteRtTensorBufferRequirements requirements, int* num_types);
LiteRtStatus LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
    LiteRtTensorBufferRequirements requirements, int type_index,
    LiteRtTensorBufferType* type);
LiteRtStatus LiteRtGetTensorBufferRequirementsBufferSize(
    LiteRtTensorBufferRequirements requirements, size_t* buffer_size);
void LiteRtDestroyTensorBufferRequirements(
    LiteRtTensorBufferRequirements requirements);

// --- Model Signature Query ---
LiteRtStatus LiteRtGetNumModelSignatures(LiteRtModel model,
                                         LiteRtParamIndex* num_signatures);
LiteRtStatus LiteRtGetModelSignature(LiteRtModel model,
                                     LiteRtParamIndex signature_index,
                                     LiteRtSignature* signature);
LiteRtStatus LiteRtGetSignatureKey(LiteRtSignature signature,
                                   const char** signature_key);
LiteRtStatus LiteRtGetNumSignatureInputs(LiteRtSignature signature,
                                         LiteRtParamIndex* num_inputs);
LiteRtStatus LiteRtGetSignatureInputName(LiteRtSignature signature,
                                         LiteRtParamIndex input_idx,
                                         const char** input_name);
LiteRtStatus LiteRtGetNumSignatureOutputs(LiteRtSignature signature,
                                          LiteRtParamIndex* num_outputs);
LiteRtStatus LiteRtGetSignatureOutputName(LiteRtSignature signature,
                                          LiteRtParamIndex output_idx,
                                          const char** output_name);
LiteRtStatus LiteRtGetSignatureInputTensorByIndex(LiteRtSignature signature,
                                                   LiteRtParamIndex input_idx,
                                                   LiteRtTensor* tensor);
LiteRtStatus LiteRtGetSignatureOutputTensorByIndex(LiteRtSignature signature,
                                                    LiteRtParamIndex output_idx,
                                                    LiteRtTensor* tensor);
LiteRtStatus LiteRtGetRankedTensorType(LiteRtTensor tensor,
                                       LiteRtRankedTensorType* ranked_tensor_type);

#ifdef __cplusplus
}  // extern "C"
#endif
