target_sources(tinylog
    INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_LIST_DIR}"
        FILES
            "${CMAKE_CURRENT_LIST_DIR}/tinylog.hpp"
)
