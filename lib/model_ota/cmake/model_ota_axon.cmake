#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Axon model-only OTA: retain app-owned symbols referenced by partition images.
#
# model_ota_axon_wire(HEADER <compiler-generated nrf_axon_model_*.h>)
#
# Parses the model header at configure time and forces the application link to retain symbols
# (interlayer buffer, packed output, op extensions) whose addresses are injected into the model
# image via PROVIDE() at the second-stage model-image link.

include_guard(GLOBAL)

get_filename_component(MODEL_OTA_ROOT ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
get_filename_component(EDGE_AI_MODULE_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)

set(MODEL_OTA_AXON_FIXUPS_GEN ${EDGE_AI_MODULE_ROOT}/tools/model_ota/gen_axon_model_image_fixups.py)

function(model_ota_axon_wire)
  cmake_parse_arguments(MO "" "HEADER;PERSISTENT_SRC" "" ${ARGN})

  if(NOT MO_HEADER)
    message(FATAL_ERROR "model_ota_axon_wire: HEADER is required")
  endif()

  if(NOT MO_PERSISTENT_SRC)
    get_filename_component(_hdr_name ${MO_HEADER} NAME_WE)
    set(MO_PERSISTENT_SRC ${CMAKE_CURRENT_BINARY_DIR}/${_hdr_name}_persistent.c)
  endif()

  if(NOT EXISTS ${MO_HEADER})
    message(FATAL_ERROR "model_ota_axon_wire: HEADER not found: ${MO_HEADER}")
  endif()

  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${MO_HEADER})

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_AXON_FIXUPS_GEN}
            --header ${MO_HEADER}
            --app-persistent-src ${MO_PERSISTENT_SRC}
    COMMAND_ERROR_IS_FATAL ANY
  )

  if(EXISTS ${MO_PERSISTENT_SRC})
    file(READ ${MO_PERSISTENT_SRC} _persistent_src_content)
    if(_persistent_src_content)
      target_sources(app PRIVATE ${MO_PERSISTENT_SRC})
    endif()
  endif()

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_AXON_FIXUPS_GEN}
            --header ${MO_HEADER}
            --print-symbols
    OUTPUT_VARIABLE model_sym_names_raw
    COMMAND_ERROR_IS_FATAL ANY
  )
  string(STRIP "${model_sym_names_raw}" model_sym_names_raw)
  string(REPLACE "\n" ";" model_sym_names "${model_sym_names_raw}")

  foreach(sym ${model_sym_names})
    if(sym)
      toolchain_ld_force_undefined_symbols(${sym})
    endif()
  endforeach()

  set_property(GLOBAL APPEND PROPERTY model_ota_axon_wired "${MO_HEADER}")
endfunction()
