#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Shared app-side wired-library step for Edge AI Lab model OTA.

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/model_ota_common.cmake)

# model_ota_add_wired_library(LIB <target> TEMPLATE <abs .c.in> OUT_SRC <abs .c>
#                             MODEL_SRC <abs> DESCRIPTION <str>
#                             DISCARD_SECTIONS <section>...
#                             [DEFINES <d>...] [INCLUDES <dir>...] [DEPENDS <target>...])
#
# configure_file() inside this helper sees @VAR@ values set by the caller, because CMake function
# scopes chain to the calling scope.
#
# TODO: that scope chaining makes SOLUTION_ID, PARTITION_NODELABEL, NEURONS_CAP, MODEL_SRC_BASENAME,
# AXON_TARGET and AXON_TOKEN implicit inputs of this helper: they are unprefixed names set in the
# caller's scope and never appear at the call site, so a typo substitutes an empty string and a
# collision substitutes the wrong value - in both cases silently, since @ONLY substitution has no
# notion of a required variable. Moving these to explicit DEFINES arguments (see the TODO in
# model_ota_edgeai_neuton_wired.c.in) turns both failures into an #error from the TU itself.
function(model_ota_add_wired_library)
  cmake_parse_arguments(WL ""
    "LIB;TEMPLATE;OUT_SRC;MODEL_SRC;DESCRIPTION" "DISCARD_SECTIONS;DEFINES;INCLUDES;DEPENDS"
    ${ARGN})

  if(NOT WL_LIB OR NOT WL_TEMPLATE OR NOT WL_OUT_SRC OR NOT WL_MODEL_SRC OR NOT WL_DESCRIPTION
     OR NOT WL_DISCARD_SECTIONS)
    message(FATAL_ERROR
            "model_ota_add_wired_library requires LIB, TEMPLATE, OUT_SRC, MODEL_SRC, "
            "DESCRIPTION and DISCARD_SECTIONS")
  endif()

  configure_file(${WL_TEMPLATE} ${WL_OUT_SRC} @ONLY)

  get_filename_component(_wired_dir ${WL_OUT_SRC} DIRECTORY)

  add_library(${WL_LIB} STATIC ${WL_OUT_SRC})
  set_target_properties(${WL_LIB} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${_wired_dir})
  target_link_libraries(${WL_LIB} PRIVATE zephyr_interface)
  add_dependencies(${WL_LIB} zephyr_generated_headers ${WL_DEPENDS})
  target_include_directories(${WL_LIB} PRIVATE ${MODEL_OTA_LIB_DIR}/src ${WL_INCLUDES})
  if(WL_DEFINES)
    target_compile_definitions(${WL_LIB} PRIVATE ${WL_DEFINES})
  endif()
  set_source_files_properties(${WL_OUT_SRC}
                              TARGET_DIRECTORY ${WL_LIB}
                              PROPERTIES OBJECT_DEPENDS "${WL_MODEL_SRC}")
  target_link_libraries(app PRIVATE ${WL_LIB})

  model_ota_discard_register(
    LIB ${WL_LIB}
    DESCRIPTION ${WL_DESCRIPTION}
    SECTIONS ${WL_DISCARD_SECTIONS})
endfunction()
