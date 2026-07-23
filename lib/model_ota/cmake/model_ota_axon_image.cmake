#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Axon model-only OTA: build a self-contained, linked partition IMAGE using the unified
# @ref model_image_header (params_type == MODEL_IMAGE_PARAMS_AXON).
#
# model_ota_axon_image(TARGET <prefix> HEADER <abs nrf_axon_model_*.h>
#                       PARTITION_NODELABEL <dt-nodelabel> [NAME <str>] [VERSION <x.y.z>])
#
# The model image link runs after zephyr.elf exists so app-owned symbol addresses can be
# injected via PROVIDE(). Each image is a standalone, independently-flashable artifact.

include_guard(GLOBAL)

get_filename_component(MODEL_OTA_ROOT ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
get_filename_component(EDGE_AI_MODULE_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)
set(MODEL_OTA_AXON_CMAKE_DIR ${CMAKE_CURRENT_LIST_DIR})

function(model_ota_axon_image)
  cmake_parse_arguments(MI "" "TARGET;HEADER;PARTITION_NODELABEL;NAME;VERSION" "" ${ARGN})

  if(NOT MI_TARGET OR NOT MI_HEADER OR NOT MI_PARTITION_NODELABEL)
    message(FATAL_ERROR
            "model_ota_axon_image requires TARGET, HEADER and PARTITION_NODELABEL")
  endif()
  if(NOT MI_NAME)
    set(MI_NAME ${MI_TARGET})
  endif()
  if(NOT MI_VERSION)
    set(MI_VERSION "1.0.0")
  endif()

  string(REPLACE "." ";" ver_parts "${MI_VERSION}")
  list(LENGTH ver_parts ver_len)
  list(GET ver_parts 0 ver_major)
  set(ver_minor 0)
  set(ver_patch 0)
  if(ver_len GREATER 1)
    list(GET ver_parts 1 ver_minor)
  endif()
  if(ver_len GREATER 2)
    list(GET ver_parts 2 ver_patch)
  endif()
  math(EXPR ver_u32 "(${ver_major} << 16) | (${ver_minor} << 8) | ${ver_patch}")

  dt_nodelabel(partition_node NODELABEL ${MI_PARTITION_NODELABEL} REQUIRED)
  dt_reg_addr(partition_addr PATH ${partition_node})
  dt_reg_size(partition_size PATH ${partition_node})

  get_filename_component(model_header_dir ${MI_HEADER} DIRECTORY)
  get_filename_component(model_header_name ${MI_HEADER} NAME)

  set(work_dir ${CMAKE_CURRENT_BINARY_DIR}/${MI_TARGET})
  file(MAKE_DIRECTORY ${work_dir})

  set(stub_src          ${MODEL_OTA_ROOT}/src/model_image_stub_axon.c)
  set(stub_body         ${MODEL_OTA_ROOT}/src/model_image_stub_axon_body.h)
  set(axon_model_gen    ${EDGE_AI_MODULE_ROOT}/tools/model_ota/gen_axon_model_image_fixups.py)
  set(extract_syms      ${EDGE_AI_MODULE_ROOT}/tools/model_ota/extract_elf_syms.py)
  set(crc_tool          ${EDGE_AI_MODULE_ROOT}/tools/model_ota/patch_image_crc.py)
  set(validate_tool     ${EDGE_AI_MODULE_ROOT}/tools/model_ota/validate_model_image_layout.py)
  set(defs_header       ${EDGE_AI_MODULE_ROOT}/include/model_ota/model_image.h)
  set(linker_script     ${MODEL_OTA_ROOT}/linker/model_image_neuton.ld)
  set(build_script      ${MODEL_OTA_AXON_CMAKE_DIR}/build_model_image_axon.cmake)
  get_filename_component(build_script ${build_script} ABSOLUTE)
  get_filename_component(axon_model_gen ${axon_model_gen} ABSOLUTE)
  get_filename_component(extract_syms ${extract_syms} ABSOLUTE)
  get_filename_component(crc_tool ${crc_tool} ABSOLUTE)
  get_filename_component(validate_tool ${validate_tool} ABSOLUTE)
  get_filename_component(defs_header ${defs_header} ABSOLUTE)
  get_filename_component(linker_script ${linker_script} ABSOLUTE)
  get_filename_component(stub_src ${stub_src} ABSOLUTE)
  get_filename_component(stub_body ${stub_body} ABSOLUTE)

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${axon_model_gen}
            --header ${MI_HEADER}
            --print-model-symbol
    OUTPUT_VARIABLE model_image_model_sym
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )

  set(_packed_output_bytes 0)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${axon_model_gen}
            --header ${MI_HEADER}
            --print-packed-output
    OUTPUT_VARIABLE _packed_raw
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  if(_packed_raw)
    string(REGEX MATCH "^([^ ]+) ([0-9]+)$" _packed_match "${_packed_raw}")
    if(_packed_match)
      set(_packed_output_bytes ${CMAKE_MATCH_2})
    endif()
  endif()

  set(model_sym_list    ${work_dir}/${MI_TARGET}_model_syms.list)
  set(model_syms_ld     ${work_dir}/${MI_TARGET}_model_syms.ld)
  set(model_image_o     ${work_dir}/${MI_TARGET}_model_image.o)
  set(model_image_elf   ${work_dir}/${MI_TARGET}_model_image.elf)
  set(model_image_bin_raw ${work_dir}/${MI_TARGET}_model_image_raw.bin)
  set(image_bin         ${CMAKE_CURRENT_BINARY_DIR}/${MI_TARGET}_model_image.bin)
  set(image_hex         ${CMAKE_CURRENT_BINARY_DIR}/${MI_TARGET}_model_partition.hex)

  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${MI_HEADER})

  add_custom_command(
    OUTPUT ${model_sym_list}
    COMMAND ${PYTHON_EXECUTABLE} ${axon_model_gen}
            --header ${MI_HEADER}
            --symbols ${model_sym_list}
    DEPENDS ${axon_model_gen} ${MI_HEADER}
    COMMENT "Generating ${MI_TARGET} Axon model image symbol list"
  )

  add_custom_command(
    OUTPUT ${model_syms_ld}
    COMMAND ${PYTHON_EXECUTABLE} ${extract_syms}
            --nm ${CMAKE_NM}
            --elf ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.elf
            --symbols ${model_sym_list}
            --output ${model_syms_ld}
    DEPENDS ${extract_syms}
            ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.elf
            ${model_sym_list}
    COMMENT "Extracting Axon model image symbols from zephyr.elf"
  )

  add_custom_command(
    OUTPUT ${image_bin}
    COMMAND ${CMAKE_COMMAND}
            -DMODEL_IMAGE_STUB_C=${stub_src}
            -DMODEL_HEADER_DIR=${model_header_dir}
            -DMODEL_IMAGE_HEADER=${model_header_name}
            -DMODEL_IMAGE_MODEL_SYM=${model_image_model_sym}
            -DMODEL_IMAGE_O=${model_image_o}
            -DMODEL_IMAGE_ELF=${model_image_elf}
            -DMODEL_IMAGE_BIN_RAW=${model_image_bin_raw}
            -DMODEL_IMAGE_BIN=${image_bin}
            -DSYMS_LINKER_SCRIPT=${model_syms_ld}
            -DMODEL_PARTITION_ADDR=${partition_addr}
            -DNRF_AXON_INTERLAYER_BUFFER_SIZE=${CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE}
            -DLINKER_SCRIPT=${linker_script}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_OBJCOPY=${CMAKE_OBJCOPY}
            -DCMAKE_NM=${CMAKE_NM}
            -DCRC_TOOL=${crc_tool}
            -DVALIDATE_TOOL=${validate_tool}
            -DDEFS_HEADER=${defs_header}
            -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
            -DMODEL_IMAGE_NAME_STR=${MI_NAME}
            -DMODEL_IMAGE_VERSION_U32=${ver_u32}
            -DMODEL_IMAGE_PACKED_OUTPUT_BYTES=${_packed_output_bytes}
            -DINCLUDE_DIR_EDGE_AI=${EDGE_AI_MODULE_ROOT}/include
            -DZEPHYR_BASE=${ZEPHYR_BASE}
            -DZEPHYR_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${build_script}
    DEPENDS ${stub_src} ${stub_body} ${model_syms_ld}
            ${MI_HEADER}
            ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.elf
            ${build_script} ${linker_script} ${crc_tool} ${validate_tool}
    COMMENT "Building Axon model partition image '${MI_NAME}' at ${partition_addr}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT ${image_hex}
    COMMAND ${CMAKE_OBJCOPY} -I binary -O ihex --change-addresses=${partition_addr}
            ${image_bin} ${image_hex}
    DEPENDS ${image_bin}
    COMMENT "Generating ${MI_TARGET} Axon model partition HEX"
  )

  add_custom_target(${MI_TARGET}_model_image ALL DEPENDS ${image_bin} ${image_hex})
endfunction()
