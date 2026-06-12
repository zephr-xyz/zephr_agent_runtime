target_sources(tinyllm
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/core/tinyllm_translation_unit.cpp"
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_LIST_DIR}"
        FILES
            "${CMAKE_CURRENT_LIST_DIR}/core/0_tinyllm_deps.hpp"
            "${CMAKE_CURRENT_LIST_DIR}/core/litert_api.h"
            "${CMAKE_CURRENT_LIST_DIR}/core/tinyllm_bundle_inspect.hpp"
            "${CMAKE_CURRENT_LIST_DIR}/core/tinyllm_core.hpp"
            "${CMAKE_CURRENT_LIST_DIR}/core/tinyllm_engine.hpp"
            "${CMAKE_CURRENT_LIST_DIR}/core/tinyllm_grammar.hpp"
            "${CMAKE_CURRENT_LIST_DIR}/core/tinyllm_inference.hpp"
            "${CMAKE_CURRENT_LIST_DIR}/core/tinyllm_model_gemma4.hpp"
            "${CMAKE_CURRENT_LIST_DIR}/core/tinyllm_rag.hpp"
)
