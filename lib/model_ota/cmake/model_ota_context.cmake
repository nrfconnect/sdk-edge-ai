#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Export model_ota_context.json after the application links.

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/model_ota_common.cmake)

set(MODEL_OTA_CONTEXT_EXPORT ${MODEL_OTA_TOOLS_DIR}/export_model_ota_context.py)

# Released firmware artifacts for out-of-tree model partition builds (see tools/model_ota/README.md):
#   MODEL_OTA_FW_ELF     — shipped zephyr.elf (Axon PROVIDE() symbol resolution)
#   MODEL_OTA_FW_CONTEXT — shipped model_ota_context.json (build-time compat check)
function(model_ota_resolve_fw_vars)
  if(COMMAND zephyr_get)
    zephyr_get(MODEL_OTA_FW_ELF)
    zephyr_get(MODEL_OTA_FW_CONTEXT)
  endif()

  if(DEFINED MODEL_OTA_FW_ELF AND "${MODEL_OTA_FW_ELF}" STREQUAL "")
    unset(MODEL_OTA_FW_ELF CACHE)
  endif()
  if(DEFINED MODEL_OTA_FW_CONTEXT AND "${MODEL_OTA_FW_CONTEXT}" STREQUAL "")
    unset(MODEL_OTA_FW_CONTEXT CACHE)
  endif()

  if(MODEL_OTA_FW_ELF)
    set(MODEL_OTA_FW_ELF "${MODEL_OTA_FW_ELF}" CACHE FILEPATH
        "Released zephyr.elf for out-of-tree Axon model OTA" FORCE)
  endif()
  if(MODEL_OTA_FW_CONTEXT)
    set(MODEL_OTA_FW_CONTEXT "${MODEL_OTA_FW_CONTEXT}" CACHE FILEPATH
        "Released model_ota_context.json for out-of-tree model OTA" FORCE)
  endif()

  if(MODEL_OTA_FW_CONTEXT AND MODEL_OTA_FW_CONTEXT MATCHES "\\.elf$")
    message(FATAL_ERROR
            "MODEL_OTA_FW_CONTEXT must be model_ota_context.json; use MODEL_OTA_FW_ELF for zephyr.elf")
  endif()
  if(MODEL_OTA_FW_ELF AND NOT EXISTS "${MODEL_OTA_FW_ELF}")
    message(FATAL_ERROR "MODEL_OTA_FW_ELF not found: ${MODEL_OTA_FW_ELF}")
  endif()
  if(MODEL_OTA_FW_CONTEXT AND NOT EXISTS "${MODEL_OTA_FW_CONTEXT}")
    message(FATAL_ERROR "MODEL_OTA_FW_CONTEXT not found: ${MODEL_OTA_FW_CONTEXT}")
  endif()
  if(MODEL_OTA_FW_ELF AND NOT MODEL_OTA_FW_CONTEXT)
    message(FATAL_ERROR
            "MODEL_OTA_FW_ELF requires MODEL_OTA_FW_CONTEXT when building Axon partition images out-of-tree")
  endif()
endfunction()

if(COMMAND zephyr_get)
  model_ota_resolve_fw_vars()
endif()

function(model_ota_using_released_fw OUT_VAR)
  if(MODEL_OTA_FW_ELF OR MODEL_OTA_FW_CONTEXT)
    set(${OUT_VAR} TRUE PARENT_SCOPE)
  else()
    set(${OUT_VAR} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(model_ota_context_register_slot)
  cmake_parse_arguments(S ""
    "TARGET;BACKEND;PARTITION_NODELABEL;NAME;NEURONS_CAP;PERSISTENT_VARS_CAP;PACKED_OUTPUT_CAP"
    "" ${ARGN})
  if(NOT S_TARGET OR NOT S_BACKEND OR NOT S_PARTITION_NODELABEL)
    message(FATAL_ERROR "model_ota_context_register_slot requires TARGET, BACKEND, PARTITION_NODELABEL")
  endif()
  if(NOT S_NAME)
    set(S_NAME ${S_TARGET})
  endif()

  dt_nodelabel(_node NODELABEL ${S_PARTITION_NODELABEL} REQUIRED)
  dt_reg_addr(_addr PATH ${_node})
  dt_reg_size(_size PATH ${_node})
  math(EXPR _addr_dec "${_addr}")
  math(EXPR _size_dec "${_size}")

  set(_fields
    "\"target\": \"${S_TARGET}\""
    "\"name\": \"${S_NAME}\""
    "\"backend\": \"${S_BACKEND}\""
    "\"partition_nodelabel\": \"${S_PARTITION_NODELABEL}\""
    "\"partition_addr\": ${_addr_dec}"
    "\"partition_size\": ${_size_dec}"
  )
  if(S_NEURONS_CAP)
    list(APPEND _fields "\"neurons_cap\": ${S_NEURONS_CAP}")
  endif()
  if(DEFINED S_PERSISTENT_VARS_CAP)
    list(APPEND _fields "\"persistent_vars_cap\": ${S_PERSISTENT_VARS_CAP}")
  endif()
  if(DEFINED S_PACKED_OUTPUT_CAP)
    list(APPEND _fields "\"packed_output_cap\": ${S_PACKED_OUTPUT_CAP}")
  endif()

  string(JOIN ", " _slot_json ${_fields})
  set_property(GLOBAL APPEND PROPERTY model_ota_context_slot_json "{${_slot_json}}")
endfunction()

# Per-slot metadata only known once the build runs, merged into model_ota_context.json by
# export_model_ota_context.py. Every backend has at least the contract hash there, since that value
# is read out of a compiled probe object rather than computed at configure time.
function(model_ota_context_register_slot_build)
  cmake_parse_arguments(S "" "SLOT_JSON" "" ${ARGN})
  if(NOT S_SLOT_JSON)
    message(FATAL_ERROR "model_ota_context_register_slot_build requires SLOT_JSON")
  endif()
  set_property(GLOBAL APPEND PROPERTY model_ota_context_slot_build_json ${S_SLOT_JSON})
endfunction()

# Emit context_slot.json, register configure-time and build-time slot metadata.
#
# model_ota_add_context_slot(TARGET <id> BACKEND <axon|neuton> PARTITION_NODELABEL <label>
#                            NAME <str> WORK_DIR <dir> CONTRACT_PROBE <obj>
#                            [CONFIG_HEADER <axon_config.h>] [KEEP_JSON <axon_keep.json>]
#                            [NEURONS_CAP <n>] OUT_SLOT_JSON <var>)
function(model_ota_add_context_slot)
  cmake_parse_arguments(CS ""
    "TARGET;BACKEND;PARTITION_NODELABEL;NAME;WORK_DIR;CONTRACT_PROBE;CONFIG_HEADER;KEEP_JSON;NEURONS_CAP;OUT_SLOT_JSON"
    "" ${ARGN})

  if(NOT CS_TARGET OR NOT CS_BACKEND OR NOT CS_PARTITION_NODELABEL OR NOT CS_WORK_DIR
     OR NOT CS_CONTRACT_PROBE OR NOT CS_OUT_SLOT_JSON)
    message(FATAL_ERROR
            "model_ota_add_context_slot requires TARGET, BACKEND, PARTITION_NODELABEL, "
            "WORK_DIR, CONTRACT_PROBE and OUT_SLOT_JSON")
  endif()
  if(NOT CS_NAME)
    set(CS_NAME ${CS_TARGET})
  endif()

  set(_slot_json ${CS_WORK_DIR}/context_slot.json)
  set(_emit_cmd ${PYTHON_EXECUTABLE} ${MODEL_OTA_CONTEXT_SLOT_TOOL}
               --contract-probe ${CS_CONTRACT_PROBE} --out ${_slot_json})
  if(CS_CONFIG_HEADER)
    list(APPEND _emit_cmd --config ${CS_CONFIG_HEADER})
  endif()
  if(CS_KEEP_JSON)
    list(APPEND _emit_cmd --keep-json ${CS_KEEP_JSON})
  endif()

  add_custom_command(
    OUTPUT ${_slot_json}
    COMMAND ${_emit_cmd}
    DEPENDS ${CS_CONTRACT_PROBE} ${MODEL_OTA_CONTEXT_SLOT_TOOL} ${CS_CONFIG_HEADER} ${CS_KEEP_JSON}
    COMMENT "Emitting OTA context slot metadata (${CS_TARGET})"
    VERBATIM)
  add_custom_target(${CS_TARGET}_contract_slot DEPENDS ${_slot_json})

  model_ota_using_released_fw(_using_released_fw)
  if(CONFIG_MODEL_OTA AND NOT _using_released_fw)
    set(_register_args
      TARGET ${CS_TARGET}
      BACKEND ${CS_BACKEND}
      PARTITION_NODELABEL ${CS_PARTITION_NODELABEL}
      NAME ${CS_NAME})
    if(CS_NEURONS_CAP)
      list(APPEND _register_args NEURONS_CAP ${CS_NEURONS_CAP})
    endif()
    model_ota_context_register_slot(${_register_args})
    model_ota_context_register_slot_build(SLOT_JSON ${_slot_json})
  endif()

  set(${CS_OUT_SLOT_JSON} ${_slot_json} PARENT_SCOPE)
endfunction()

function(model_ota_block_app_build)
  model_ota_using_released_fw(_using_released_fw)
  if(NOT _using_released_fw)
    return()
  endif()

  add_custom_target(model_ota_app_build_blocked
    COMMAND ${CMAKE_COMMAND} -E echo
            "MODEL_OTA_FW_ELF/MODEL_OTA_FW_CONTEXT set: this build tree only produces model partition images; build one with 'cmake --build ${APPLICATION_BINARY_DIR} --target <name>_model_image'"
    COMMAND ${CMAKE_COMMAND} -E false
    VERBATIM)

  add_dependencies(app model_ota_app_build_blocked)
endfunction()

function(model_ota_context_finalize)
  if(NOT CONFIG_MODEL_OTA)
    return()
  endif()

  model_ota_using_released_fw(_using_released_fw)
  if(_using_released_fw)
    model_ota_block_app_build()
    return()
  endif()

  get_property(_slots GLOBAL PROPERTY model_ota_context_slot_json)
  if(NOT _slots)
    return()
  endif()

  get_property(_slot_build_files GLOBAL PROPERTY model_ota_context_slot_build_json)

  set(_manifest ${CMAKE_CURRENT_BINARY_DIR}/model_ota/model_ota_context_manifest.json)
  set(_out ${CMAKE_CURRENT_BINARY_DIR}/model_ota_context.json)
  set(_zephyr_elf ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.elf)
  set(_autoconf ${CMAKE_CURRENT_BINARY_DIR}/zephyr/include/generated/zephyr/autoconf.h)

  string(JOIN ",\n    " _slot_body ${_slots})
  file(WRITE ${_manifest}
"{
  \"slots\": [
    ${_slot_body}
  ]
}
")

  add_custom_command(
    OUTPUT ${_out}
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_CONTEXT_EXPORT}
            --manifest ${_manifest}
            --out ${_out}
            --build-dir ${CMAKE_CURRENT_BINARY_DIR}
            --elf ${_zephyr_elf}
            --autoconf ${_autoconf}
    DEPENDS ${MODEL_OTA_CONTEXT_EXPORT} ${_manifest} ${_zephyr_elf} ${_slot_build_files}
    COMMENT "Exporting model OTA firmware context"
    VERBATIM
  )

  add_custom_target(model_ota_context ALL DEPENDS ${_out})
  add_dependencies(model_ota_context zephyr)
endfunction()
