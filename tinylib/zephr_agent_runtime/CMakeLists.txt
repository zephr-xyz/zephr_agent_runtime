target_sources(zephr_agent_c_api
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_LIST_DIR}"
        FILES
            "${CMAKE_CURRENT_LIST_DIR}/core/0_zephr_agent_deps.hpp"
            "${CMAKE_CURRENT_LIST_DIR}/public/zephr_agent_c_api.h"
)

target_sources(zephr_agent_c_api
    PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/core/zephr_agent_c_api.cpp"
)
