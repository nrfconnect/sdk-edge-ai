#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Shared app-side wired-library step for Edge AI Lab and raw Axon model OTA.

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/model_ota_common.cmake)

# model_ota_add_wired_library(LIB <target> SOURCE <abs .c> ARCHIVE_DIR <abs dir>
#                             [MODEL_SRC <abs>] DESCRIPTION <str>
#                             [DISCARD_SECTIONS <section>...] [DEFINES <d>...]
#                             [INCLUDES <dir>...] [DEPENDS <target>...])
function(model_ota_add_wired_library)
  cmake_parse_arguments(WL ""
    "LIB;SOURCE;ARCHIVE_DIR;MODEL_SRC;DESCRIPTION" "DISCARD_SECTIONS;DEFINES;INCLUDES;DEPENDS"
    ${ARGN})

  if(NOT WL_LIB OR NOT WL_SOURCE OR NOT WL_ARCHIVE_DIR OR NOT WL_DESCRIPTION)
    message(FATAL_ERROR
            "model_ota_add_wired_library requires LIB, SOURCE, ARCHIVE_DIR and DESCRIPTION")
  endif()

  add_library(${WL_LIB} STATIC ${WL_SOURCE})
  set_target_properties(${WL_LIB} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${WL_ARCHIVE_DIR})
  target_link_libraries(${WL_LIB} PRIVATE zephyr_interface)
  add_dependencies(${WL_LIB} zephyr_generated_headers ${WL_DEPENDS})
  target_include_directories(${WL_LIB} PRIVATE ${MODEL_OTA_LIB_DIR}/src ${WL_INCLUDES})
  if(WL_DEFINES)
    target_compile_definitions(${WL_LIB} PRIVATE ${WL_DEFINES})
  endif()
  if(WL_MODEL_SRC)
    set_source_files_properties(${WL_SOURCE}
                                TARGET_DIRECTORY ${WL_LIB}
                                PROPERTIES OBJECT_DEPENDS "${WL_MODEL_SRC}")
  endif()
  target_link_libraries(app PRIVATE ${WL_LIB})

  if(WL_DISCARD_SECTIONS)
    model_ota_discard_register(
      LIB ${WL_LIB}
      DESCRIPTION ${WL_DESCRIPTION}
      SECTIONS ${WL_DISCARD_SECTIONS})
  endif()
endfunction()
