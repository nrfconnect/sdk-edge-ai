#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Axon model-only OTA: per-model static library + app RAM stubs + symbol retention.
#
# model_ota_axon_wire(HEADER <compiler-generated nrf_axon_model_*.h>
#                     [PERSISTENT_VARS_CAP <cap>] [LIB_NAME <target>])
#
# Packed-output stub size is fixed from the model header (not overridable).

include_guard(GLOBAL)

get_filename_component(MODEL_OTA_ROOT ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
get_filename_component(EDGE_AI_MODULE_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)

set(MODEL_OTA_AXON_FIXUPS_GEN ${EDGE_AI_MODULE_ROOT}/tools/model_ota/gen_axon_model_image_fixups.py)
set(MODEL_OTA_AXON_STUB ${MODEL_OTA_ROOT}/src/model_ota_axon_app_stub.c)

function(model_ota_axon_wire)
  cmake_parse_arguments(MO "" "HEADER;LIB_NAME;PERSISTENT_VARS_CAP" "" ${ARGN})

  if(NOT MO_HEADER)
    message(FATAL_ERROR "model_ota_axon_wire: HEADER is required")
  endif()

  if(NOT EXISTS ${MO_HEADER})
    message(FATAL_ERROR "model_ota_axon_wire: HEADER not found: ${MO_HEADER}")
  endif()

  get_filename_component(model_header_dir ${MO_HEADER} DIRECTORY)
  get_filename_component(model_header_name ${MO_HEADER} NAME)

  string(REGEX REPLACE "^nrf_axon_model_" "" _model_token ${model_header_name})
  string(REGEX REPLACE "\\.h$" "" _model_token ${_model_token})
  string(REGEX REPLACE "_+$" "" _model_token ${_model_token})

  if(NOT MO_LIB_NAME)
    set(MO_LIB_NAME ota_axon_${_model_token})
  endif()

  if(TARGET ${MO_LIB_NAME})
    message(FATAL_ERROR "model_ota_axon_wire: duplicate LIB_NAME/target ${MO_LIB_NAME}")
  endif()

  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${MO_HEADER})

  set(_persistent_required 0)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_AXON_FIXUPS_GEN}
            --header ${MO_HEADER}
            --print-persistent-vars
    OUTPUT_VARIABLE _persistent_raw
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  if(_persistent_raw)
    string(REGEX MATCH "^([^ ]+) ([0-9]+)$" _persistent_match "${_persistent_raw}")
    if(NOT _persistent_match)
      message(FATAL_ERROR "model_ota_axon_wire: bad --print-persistent-vars: '${_persistent_raw}'")
    endif()
    set(_persistent_sym ${CMAKE_MATCH_1})
    set(_persistent_required ${CMAKE_MATCH_2})
  endif()

  if(NOT MO_PERSISTENT_VARS_CAP)
    set(MO_PERSISTENT_VARS_CAP ${_persistent_required})
  elseif(_persistent_required GREATER MO_PERSISTENT_VARS_CAP)
    message(FATAL_ERROR
            "model_ota_axon_wire: PERSISTENT_VARS_CAP ${MO_PERSISTENT_VARS_CAP} < required "
            "${_persistent_required} for ${MO_HEADER}")
  endif()

  set(_packed_required 0)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_AXON_FIXUPS_GEN}
            --header ${MO_HEADER}
            --print-packed-output
    OUTPUT_VARIABLE _packed_raw
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  if(_packed_raw)
    string(REGEX MATCH "^([^ ]+) ([0-9]+)$" _packed_match "${_packed_raw}")
    if(NOT _packed_match)
      message(FATAL_ERROR "model_ota_axon_wire: bad --print-packed-output: '${_packed_raw}'")
    endif()
    set(_packed_sym ${CMAKE_MATCH_1})
    set(_packed_required ${CMAKE_MATCH_2})
  endif()

  if(_packed_required GREATER 0)
    set(_allocate_packed_output ON)
  else()
    set(_allocate_packed_output OFF)
  endif()

  add_library(${MO_LIB_NAME} STATIC ${MODEL_OTA_AXON_STUB})
  target_link_libraries(${MO_LIB_NAME} PRIVATE zephyr_interface)
  add_dependencies(${MO_LIB_NAME} zephyr_generated_headers)
  target_include_directories(${MO_LIB_NAME} PRIVATE ${MODEL_OTA_ROOT}/src ${model_header_dir})
  target_compile_definitions(${MO_LIB_NAME} PRIVATE
                             MODEL_OTA_AXON_WIRED=1
                             MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED=${_persistent_required}
                             MODEL_OTA_AXON_PERSISTENT_VARS_CAP=${MO_PERSISTENT_VARS_CAP}
                             MODEL_OTA_AXON_PACKED_OUTPUT_BYTES=${_packed_required})
  target_compile_options(${MO_LIB_NAME} PRIVATE
                         "-DMODEL_OTA_AXON_HEADER=\"${model_header_name}\"")
  if(_persistent_required GREATER 0)
    target_compile_options(${MO_LIB_NAME} PRIVATE
                           -DMODEL_OTA_AXON_PERSISTENT_VARS_SYM=${_persistent_sym})
  endif()
  if(_allocate_packed_output)
    target_compile_options(${MO_LIB_NAME} PRIVATE
                           -DMODEL_OTA_AXON_PACKED_OUTPUT_SYM=${_packed_sym}
                           -DNRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER=1)
  endif()
  set_source_files_properties(${MODEL_OTA_AXON_STUB}
                              TARGET_DIRECTORY ${MO_LIB_NAME}
                              PROPERTIES OBJECT_DEPENDS "${MO_HEADER}")
  target_link_libraries(app PRIVATE ${MO_LIB_NAME})

  if(_packed_required GREATER 0)
    string(TOUPPER ${_model_token} _model_token_upper)
    target_compile_definitions(app PRIVATE
                               OTA_AXON_${_model_token_upper}_PACKED_OUTPUT_BYTES=${_packed_required})
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
      if(NOT _allocate_packed_output AND sym STREQUAL _packed_sym)
        continue()
      endif()
      toolchain_ld_force_undefined_symbols(${sym})
    endif()
  endforeach()

  set_property(GLOBAL APPEND PROPERTY model_ota_axon_wired
               "${MO_LIB_NAME} (${MO_HEADER}, persistent_cap=${MO_PERSISTENT_VARS_CAP}, "
               "packed_bytes=${_packed_required})")
endfunction()
