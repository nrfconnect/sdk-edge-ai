#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(hello_axon)

target_sources(app PRIVATE src/main.c)

if(CONFIG_HELLO_AXON_MODEL_IN_PARTITION)
  target_sources(app PRIVATE src/model_partition/model_partition.c)

  set(model_image_c ${CMAKE_CURRENT_BINARY_DIR}/hello_axon_model_image.c)
  set(model_image_o ${CMAKE_CURRENT_BINARY_DIR}/hello_axon_model_image.o)
  set(model_image_elf ${CMAKE_CURRENT_BINARY_DIR}/hello_axon_model_image.elf)
  set(model_image_bin ${CMAKE_CURRENT_BINARY_DIR}/hello_axon_model_image.bin)
  set(model_image_hex ${CMAKE_CURRENT_BINARY_DIR}/hello_axon_model_partition.hex)
  set(interlayer_buffer_addr_file ${CMAKE_CURRENT_BINARY_DIR}/interlayer_buffer.addr)

  dt_nodelabel(axon_model_partition_node NODELABEL axon_model_partition REQUIRED)
  dt_reg_addr(axon_model_partition_addr PATH ${axon_model_partition_node})

  add_custom_command(
    OUTPUT ${model_image_c}
    COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_hello_axon_model_linked_c.py
      --header ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/nrf_axon_model_hello_axon_.h
      --output ${model_image_c}
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_hello_axon_model_linked_c.py
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_hello_axon_model_image.py
      ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/nrf_axon_model_hello_axon_.h
      ${CMAKE_CURRENT_SOURCE_DIR}/src/model_partition/hello_axon_model_image_defs.h
    COMMENT "Generating hello_axon model partition C source"
  )

  add_custom_command(
    OUTPUT ${interlayer_buffer_addr_file}
    COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/scripts/extract_elf_sym_addr.py
      --nm ${CMAKE_NM}
      --elf ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.elf
      --symbol nrf_axon_interlayer_buffer
      --output ${interlayer_buffer_addr_file}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.elf
    COMMENT "Extracting nrf_axon_interlayer_buffer address from zephyr.elf"
  )

  add_custom_command(
    OUTPUT ${model_image_bin}
    COMMAND ${CMAKE_COMMAND}
      -DMODEL_IMAGE_C=${model_image_c}
      -DMODEL_IMAGE_O=${model_image_o}
      -DMODEL_IMAGE_ELF=${model_image_elf}
      -DMODEL_IMAGE_BIN=${model_image_bin}
      -DIL_ADDR_FILE=${interlayer_buffer_addr_file}
      -DMODEL_PARTITION_ADDR=${axon_model_partition_addr}
      -DLINKER_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/model_image/linker.ld
      -DINCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/src/model_partition
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_OBJCOPY=${CMAKE_OBJCOPY}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_model_image.cmake
    DEPENDS
      ${model_image_c}
      ${interlayer_buffer_addr_file}
      ${CMAKE_CURRENT_SOURCE_DIR}/model_image/linker.ld
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_model_image.cmake
    COMMENT "Linking hello_axon model partition image with absolute cmd buffer pointers"
    VERBATIM
  )

  add_custom_target(hello_axon_model_image ALL DEPENDS ${model_image_bin})

  add_custom_command(
    OUTPUT ${model_image_hex}
    COMMAND ${CMAKE_OBJCOPY}
      -I binary
      -O ihex
      --change-addresses=${axon_model_partition_addr}
      ${model_image_bin}
      ${model_image_hex}
    DEPENDS ${model_image_bin}
    COMMENT "Generating hello_axon model partition HEX"
  )

  add_custom_target(hello_axon_model_partition_hex ALL DEPENDS ${model_image_hex})
  add_dependencies(hello_axon_model_partition_hex hello_axon_model_image)

  set(hello_axon_hex_merge_stamp ${CMAKE_CURRENT_BINARY_DIR}/.hello_axon_model_hex_merged)

  add_custom_command(
    OUTPUT ${hello_axon_hex_merge_stamp}
    COMMAND ${PYTHON_EXECUTABLE}
      ${ZEPHYR_BASE}/scripts/build/mergehex.py
      -o ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.hex
      ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.hex
      ${model_image_hex}
    COMMAND ${CMAKE_COMMAND} -E touch ${hello_axon_hex_merge_stamp}
    DEPENDS
      ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.elf
      ${model_image_hex}
    COMMENT "Merging hello_axon model partition into zephyr.hex"
    VERBATIM
  )

  add_custom_target(hello_axon_hex_merge ALL DEPENDS ${hello_axon_hex_merge_stamp})
  add_dependencies(hello_axon_hex_merge hello_axon_model_partition_hex)
endif()
