# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Build and merge an updatable Axon model partition image.
#
# nrf_axon_model_partition_image(
#   TARGET <unique-target-prefix>
#   HEADER <compiler-generated-model.h>
#   PARTITION_NODELABEL <devicetree-nodelabel>
#   [IMAGE_SYMBOL <c-symbol-name>]
# )

include_guard(GLOBAL)

get_filename_component(AXON_MODEL_PARTITION_ROOT ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
get_filename_component(EDGE_AI_MODULE_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../.. ABSOLUTE)

function(nrf_axon_model_partition_image)
  cmake_parse_arguments(ARG "" "TARGET;HEADER;PARTITION_NODELABEL;IMAGE_SYMBOL" "" ${ARGN})

  if(NOT ARG_TARGET OR NOT ARG_HEADER OR NOT ARG_PARTITION_NODELABEL)
    message(FATAL_ERROR "nrf_axon_model_partition_image requires TARGET, HEADER and PARTITION_NODELABEL")
  endif()

  if(NOT ARG_IMAGE_SYMBOL)
    set(ARG_IMAGE_SYMBOL axon_model_partition_image)
  endif()

  set(AXON_MODEL_PARTITION_DIR ${AXON_MODEL_PARTITION_ROOT})
  set(EDGE_AI_MODULE_DIR ${EDGE_AI_MODULE_ROOT})
  set(model_image_stub_c ${AXON_MODEL_PARTITION_DIR}/src/model_image_stub.c)
  set(model_fixups_h ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_model_fixups.h)
  set(model_image_o ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_model_image.o)
  set(model_image_elf ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_model_image.elf)
  set(model_image_bin ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_model_image.bin)
  set(model_image_hex ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_model_partition.hex)
  set(model_sym_list ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_model_syms.list)
  set(model_sym_link_list ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_model_syms.link)
  set(model_syms_h ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_model_syms.h)
  set(model_syms_ld ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_model_syms.ld)
  set(hex_merge_stamp ${CMAKE_CURRENT_BINARY_DIR}/.${ARG_TARGET}_model_hex_merged)

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE}
      ${AXON_MODEL_PARTITION_ROOT}/scripts/gen_axon_model_partition_c.py
      --header ${ARG_HEADER}
      --symbols ${model_sym_link_list}
      --symbols-only
    COMMAND_ERROR_IS_FATAL ANY
  )

  file(READ ${model_sym_link_list} model_sym_link_content)
  string(STRIP "${model_sym_link_content}" model_sym_link_content)
  string(REPLACE "\n" ";" model_sym_names "${model_sym_link_content}")
  foreach(sym ${model_sym_names})
    if(sym)
      toolchain_ld_force_undefined_symbols(${sym})
    endif()
  endforeach()

  dt_nodelabel(partition_node NODELABEL ${ARG_PARTITION_NODELABEL} REQUIRED)
  dt_reg_addr(partition_addr PATH ${partition_node})
  dt_reg_size(partition_size PATH ${partition_node})

  get_filename_component(model_header_dir ${ARG_HEADER} DIRECTORY)

  add_custom_command(
    OUTPUT ${model_fixups_h} ${model_sym_list}
    COMMAND ${PYTHON_EXECUTABLE}
      ${AXON_MODEL_PARTITION_DIR}/scripts/gen_axon_model_partition_c.py
      --header ${ARG_HEADER}
      --fixups-header ${model_fixups_h}
      --symbols ${model_sym_list}
      --use-stub
    DEPENDS
      ${AXON_MODEL_PARTITION_DIR}/scripts/gen_axon_model_partition_c.py
      ${ARG_HEADER}
      ${AXON_MODEL_PARTITION_DIR}/include/axon/nrf_axon_model_partition_defs.h
    COMMENT "Generating ${ARG_TARGET} Axon model partition fixups header"
  )

  add_custom_command(
    OUTPUT ${model_syms_h} ${model_syms_ld}
    COMMAND ${PYTHON_EXECUTABLE}
      ${AXON_MODEL_PARTITION_DIR}/scripts/extract_elf_syms.py
      --nm ${CMAKE_NM}
      --elf ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.elf
      --symbols ${model_sym_list}
      --output ${model_syms_h}
      --linker-script ${model_syms_ld}
    DEPENDS
      ${AXON_MODEL_PARTITION_DIR}/scripts/extract_elf_syms.py
      ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.elf
      ${model_sym_list}
    COMMENT "Extracting Axon model partition symbols from zephyr.elf"
  )

  add_custom_command(
    OUTPUT ${model_image_bin}
    COMMAND ${CMAKE_COMMAND}
      -DMODEL_IMAGE_STUB_C=${model_image_stub_c}
      -DMODEL_FIXUPS_HEADER=${model_fixups_h}
      -DMODEL_HEADER_DIR=${model_header_dir}
      -DMODEL_IMAGE_O=${model_image_o}
      -DMODEL_IMAGE_ELF=${model_image_elf}
      -DMODEL_IMAGE_BIN=${model_image_bin}
      -DSYMS_HEADER=${model_syms_h}
      -DSYMS_LINKER_SCRIPT=${model_syms_ld}
      -DMODEL_PARTITION_ADDR=${partition_addr}
      -DNRF_AXON_INTERLAYER_BUFFER_SIZE=${CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE}
      -DLINKER_SCRIPT=${AXON_MODEL_PARTITION_DIR}/linker/model_image.ld
      -DINCLUDE_DIR_PARTITION=${AXON_MODEL_PARTITION_DIR}/include
      -DINCLUDE_DIR_EDGE_AI=${EDGE_AI_MODULE_DIR}/include
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_OBJCOPY=${CMAKE_OBJCOPY}
      -P ${AXON_MODEL_PARTITION_DIR}/cmake/build_model_image.cmake
    COMMAND ${PYTHON_EXECUTABLE}
      ${AXON_MODEL_PARTITION_DIR}/scripts/report_model_partition_usage.py
      --label ${ARG_PARTITION_NODELABEL}
      --bin ${model_image_bin}
      --region-size ${partition_size}
    DEPENDS
      ${model_image_stub_c}
      ${model_fixups_h}
      ${ARG_HEADER}
      ${model_syms_h}
      ${model_syms_ld}
      ${AXON_MODEL_PARTITION_DIR}/linker/model_image.ld
      ${AXON_MODEL_PARTITION_DIR}/cmake/build_model_image.cmake
      ${AXON_MODEL_PARTITION_DIR}/scripts/report_model_partition_usage.py
    COMMENT "Linking ${ARG_TARGET} Axon model partition image"
    VERBATIM
  )

  add_custom_target(${ARG_TARGET}_model_image ALL DEPENDS ${model_image_bin})

  add_custom_command(
    OUTPUT ${model_image_hex}
    COMMAND ${CMAKE_OBJCOPY}
      -I binary
      -O ihex
      --change-addresses=${partition_addr}
      ${model_image_bin}
      ${model_image_hex}
    DEPENDS ${model_image_bin}
    COMMENT "Generating ${ARG_TARGET} Axon model partition HEX"
  )

  add_custom_target(${ARG_TARGET}_model_partition_hex ALL DEPENDS ${model_image_hex})
  add_dependencies(${ARG_TARGET}_model_partition_hex ${ARG_TARGET}_model_image)

  add_custom_command(
    OUTPUT ${hex_merge_stamp}
    COMMAND ${PYTHON_EXECUTABLE}
      ${ZEPHYR_BASE}/scripts/build/mergehex.py
      -o ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.hex
      ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.hex
      ${model_image_hex}
    COMMAND ${CMAKE_COMMAND} -E touch ${hex_merge_stamp}
    DEPENDS
      ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.elf
      ${model_image_hex}
    COMMENT "Merging ${ARG_TARGET} model partition into zephyr.hex"
    VERBATIM
  )

  add_custom_target(${ARG_TARGET}_hex_merge ALL DEPENDS ${hex_merge_stamp})
  add_dependencies(${ARG_TARGET}_hex_merge ${ARG_TARGET}_model_partition_hex)
endfunction()
