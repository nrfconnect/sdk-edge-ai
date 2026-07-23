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
# After zephyr.elf exists, compiles model_ota_axon_image_stub.c, links it at the partition base with
# model_image.ld plus a generated PROVIDE() fragment for app-owned symbols, then patches CRC,
# validates layout, and emits standalone .bin/.hex artifacts.

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/model_ota_common.cmake)

get_filename_component(MODEL_OTA_ROOT ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
get_filename_component(EDGE_AI_MODULE_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)

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

  model_ota_pack_version("${MI_VERSION}" ver_u32)

  dt_nodelabel(partition_node NODELABEL ${MI_PARTITION_NODELABEL} REQUIRED)
  dt_reg_addr(partition_addr PATH ${partition_node})
  dt_reg_size(partition_size PATH ${partition_node})

  get_filename_component(model_header_dir ${MI_HEADER} DIRECTORY)
  get_filename_component(model_header_name ${MI_HEADER} NAME)

  set(work_dir ${CMAKE_CURRENT_BINARY_DIR}/${MI_TARGET})
  file(MAKE_DIRECTORY ${work_dir})

  set(stub_src          ${MODEL_OTA_ROOT}/src/model_ota_axon_image_stub.c)
  set(axon_model_gen    ${EDGE_AI_MODULE_ROOT}/tools/model_ota/gen_axon_model_image_fixups.py)
  set(extract_syms      ${EDGE_AI_MODULE_ROOT}/tools/model_ota/extract_elf_syms.py)
  set(crc_tool          ${EDGE_AI_MODULE_ROOT}/tools/model_ota/patch_image_crc.py)
  set(validate_tool     ${EDGE_AI_MODULE_ROOT}/tools/model_ota/validate_model_image_layout.py)
  set(defs_header       ${EDGE_AI_MODULE_ROOT}/include/model_ota/model_image.h)
  set(linker_script     ${MODEL_OTA_ROOT}/linker/model_image.ld)

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

  set(model_image_header_c "\"${model_header_name}\"")
  set(model_image_name_str_c "\"${MI_NAME}\"")

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
    OUTPUT ${image_bin} ${image_hex}
    COMMAND ${CMAKE_C_COMPILER}
            -c ${stub_src}
            -o ${model_image_o}
            -DNRF_MODEL_PARTITION_ADDR=${partition_addr}
            -I${model_header_dir}
            -I${EDGE_AI_MODULE_ROOT}/include
            -I${ZEPHYR_BASE}/include
            -I${CMAKE_CURRENT_BINARY_DIR}/zephyr/include/generated
            -include ${CMAKE_CURRENT_BINARY_DIR}/zephyr/include/generated/zephyr/autoconf.h
            -DMODEL_OTA_AXON_IMAGE
            -DMODEL_IMAGE_HEADER=${model_image_header_c}
            -DMODEL_IMAGE_MODEL_SYM=${model_image_model_sym}
            -DMODEL_IMAGE_NAME_STR=${model_image_name_str_c}
            -DMODEL_IMAGE_VERSION_U32=${ver_u32}u
            -DMODEL_IMAGE_PACKED_OUTPUT_BYTES=${_packed_output_bytes}u
            -DNRF_AXON_INTERLAYER_BUFFER_SIZE=${CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE}
    COMMAND ${CMAKE_C_COMPILER}
            -nostdlib -nostartfiles
            -Wl,--gc-sections
            -Wl,--defsym=NRF_MODEL_PARTITION_ADDR=${partition_addr}
            -T ${linker_script}
            -T ${model_syms_ld}
            -o ${model_image_elf}
            ${model_image_o}
    COMMAND ${CMAKE_OBJCOPY} -O binary -j .model_image ${model_image_elf} ${model_image_bin_raw}
    COMMAND ${PYTHON_EXECUTABLE} ${crc_tool} --bin ${model_image_bin_raw} -o ${image_bin}
    COMMAND ${PYTHON_EXECUTABLE} ${validate_tool}
            --nm ${CMAKE_NM}
            --elf ${model_image_elf}
            --bin ${image_bin}
            --partition-addr ${partition_addr}
            --defs-header ${defs_header}
            --model-symbol ${model_image_model_sym}
    COMMAND ${CMAKE_OBJCOPY} -I binary -O ihex --change-addresses=${partition_addr}
            ${image_bin} ${image_hex}
    DEPENDS ${stub_src} ${model_syms_ld}
            ${MI_HEADER}
            ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.elf
            ${linker_script} ${crc_tool} ${validate_tool}
    COMMENT "Building Axon model partition image '${MI_NAME}' at ${partition_addr}"
    VERBATIM
  )

  add_custom_target(${MI_TARGET}_model_image ALL DEPENDS ${image_bin} ${image_hex})
endfunction()
