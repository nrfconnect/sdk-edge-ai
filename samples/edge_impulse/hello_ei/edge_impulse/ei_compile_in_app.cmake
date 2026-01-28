#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Edge Impulse library integration - compilation into application binary
#

set(FETCH_CONTENT_NAME edge_impulse)

# Fix for LTO: Add assembler include path for .incbin directives
# When LTO is enabled, the working directory context changes during linking,
# causing relative paths in .incbin directives to fail. Adding the source
# directory as an include path for the assembler resolves this issue.
if(CONFIG_LTO)
  # The Edge Impulse library will be fetched to this location
  set(EI_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/edge_impulse-src")

  # Add the Edge Impulse source directory to assembler include paths
  # This allows .incbin to find files using relative paths
  zephyr_compile_options(-Wa,-I${EI_SOURCE_DIR})
endif()

# Enable C linkage for Edge Impulse library (allows calling from C code)
target_compile_definitions(app PRIVATE
  EI_C_LINKAGE=1
  EIDSP_SIGNAL_C_FN_POINTER=1
)

include(FetchContent)

FetchContent_Declare(
    ${FETCH_CONTENT_NAME}
    DOWNLOAD_EXTRACT_TIMESTAMP True
    URL ${EI_URI_LIST}
    HTTP_HEADER "Accept: application/zip"
      ${EI_API_KEY_HEADER}
)

FetchContent_MakeAvailable(${FETCH_CONTENT_NAME})

# Suppress specific compiler warnings for Edge Impulse SDK source files
set(EI_SUPPRESSED_WARNINGS_FLAGS "-Wno-double-promotion -Wno-unused -Wno-stringop-overread -Wno-sign-compare -Wno-maybe-uninitialized")

# Get all sources from app target and apply flags only to Edge Impulse files
get_target_property(ALL_SOURCES app SOURCES)
foreach(src ${ALL_SOURCES})
    if(src MATCHES ${FETCHCONTENT_BASE_DIR}/${FETCH_CONTENT_NAME})
    set_source_files_properties(${src} PROPERTIES
        COMPILE_FLAGS "${EI_SUPPRESSED_WARNINGS_FLAGS}"
    )
    endif()
endforeach()
