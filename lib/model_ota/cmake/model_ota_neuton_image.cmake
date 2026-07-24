#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Neuton model-only OTA: build the model as a self-contained, linked partition IMAGE
# (same linked-at-partition-base layout as Axon images).
#
# model_ota_neuton_image(TARGET <prefix> MODEL_SRC <abs nrf_edgeai_user_model.c>
#                        PARTITION_NODELABEL <dt-nodelabel> [NAME <str>] [VERSION <x.y.z>])
#
# adds a target `<prefix>_model_image` (built by default) that:
#
#   1. Compiles lib/model_ota/src/model_ota_neuton_image_stub.c as an OBJECT library with
#      MODEL_OTA_NEUTON_MODEL_SRC set to the model basename. The stub #includes the generated
#      nrf_edgeai_user_model.c and emits the partition header into section .model_image.header.
#   2. Links that object at the partition's flash base (from devicetree) with model_image.ld
#      and --gc-sections, so the header + descriptor + data land in one .model_image section at
#      the base and every intra-image pointer is a correct absolute flash address. All the
#      runtime glue (nrf_edgeai_t, its app-code function pointers, the DSP pipeline, ...) is
#      garbage-collected, which also keeps the link free of undefined application symbols.
#   3. objcopy's .model_image to a raw .bin, patches the header CRC over that binary
#      (patch_image_crc.py), validates the layout (validate_model_image_layout.py), and converts
#      to an addressed .hex. The .hex/.bin are standalone artifacts, deliberately NOT merged into
#      zephyr.hex: the app and each model partition are flashed/updated independently.
#
# Unlike the Axon flow (and unlike a PROVIDE()-from-zephyr.elf design), the image is fully self-
# contained: the one app-owned RAM pointer, params.*.p_neurons, is NOT resolved here. It is left
# for the loader to patch from a caller-owned buffer at load time (see model_image.h). That is
# what lets the three identical file-static `model_neurons_` symbols in the multi_model sample
# coexist - a PROVIDE()-from-ELF approach cannot tell them apart.

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/model_ota_common.cmake)

get_filename_component(MODEL_OTA_ROOT ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
get_filename_component(EDGE_AI_MODULE_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)

function(model_ota_neuton_image)
  cmake_parse_arguments(MI "" "TARGET;MODEL_SRC;PARTITION_NODELABEL;NAME;VERSION" "" ${ARGN})

  if(NOT MI_TARGET OR NOT MI_MODEL_SRC OR NOT MI_PARTITION_NODELABEL)
    message(FATAL_ERROR
            "model_ota_neuton_image requires TARGET, MODEL_SRC and PARTITION_NODELABEL")
  endif()
  if(NOT MI_NAME)
    set(MI_NAME ${MI_TARGET})
  endif()
  if(NOT MI_VERSION)
    set(MI_VERSION "1.0.0")
  endif()

  model_ota_pack_version("${MI_VERSION}" ver_u32)

  # Partition base + size from the mapped-partition devicetree node.
  dt_nodelabel(partition_node NODELABEL ${MI_PARTITION_NODELABEL} REQUIRED)
  dt_reg_addr(partition_addr PATH ${partition_node})
  dt_reg_size(partition_size PATH ${partition_node})

  get_filename_component(model_dir ${MI_MODEL_SRC} DIRECTORY)
  get_filename_component(model_basename ${MI_MODEL_SRC} NAME)

  # Intermediates (ELF, raw bin) live under <target>/; only the flashable .bin and addressed
  # .hex are emitted at the build directory root.
  set(work_dir ${CMAKE_CURRENT_BINARY_DIR}/${MI_TARGET})
  file(MAKE_DIRECTORY ${work_dir})

  set(stub_src       ${MODEL_OTA_ROOT}/src/model_ota_neuton_image_stub.c)
  set(image_elf     ${work_dir}/${MI_TARGET}_model_image.elf)
  set(image_bin_raw ${work_dir}/${MI_TARGET}_model_image_raw.bin)
  set(image_bin     ${CMAKE_CURRENT_BINARY_DIR}/${MI_TARGET}_model_image.bin)
  set(image_hex     ${CMAKE_CURRENT_BINARY_DIR}/${MI_TARGET}_model_partition.hex)

  set(linker_script ${MODEL_OTA_ROOT}/linker/model_image.ld)
  set(crc_tool ${EDGE_AI_MODULE_ROOT}/tools/model_ota/patch_image_crc.py)
  set(validate_tool ${EDGE_AI_MODULE_ROOT}/tools/model_ota/validate_model_image_layout.py)
  set(defs_header ${EDGE_AI_MODULE_ROOT}/include/model_ota/model_image.h)

  set(stub tgt_${MI_TARGET}_model_image_stub)
  add_library(${stub} OBJECT ${stub_src})
  target_link_libraries(${stub} PRIVATE zephyr_interface)
  add_dependencies(${stub} zephyr_generated_headers)
  target_include_directories(${stub} PRIVATE ${model_dir})
  target_compile_options(${stub} PRIVATE -ffunction-sections -fdata-sections)
  target_compile_definitions(${stub} PRIVATE
                             MODEL_OTA_NEUTON_MODEL_SRC=${model_basename}
                             NRF_MODEL_PARTITION_ADDR=${partition_addr}
                             MODEL_IMAGE_NAME_STR=\"${MI_NAME}\"
                             MODEL_IMAGE_VERSION_U32=${ver_u32}u)
  set_source_files_properties(${stub_src}
                              TARGET_DIRECTORY ${stub}
                              PROPERTIES OBJECT_DEPENDS "${MI_MODEL_SRC}")

  add_custom_command(
    OUTPUT ${image_bin} ${image_hex}
    # 1. Link the stub object at the partition base. --gc-sections keeps only what the header
    #    (the sole KEEP root) reaches, dropping the runtime glue and its undefined app symbols.
    COMMAND ${CMAKE_C_COMPILER}
            -nostdlib -nostartfiles
            -Wl,--gc-sections
            -Wl,--defsym=NRF_MODEL_PARTITION_ADDR=${partition_addr}
            -T ${linker_script}
            -o ${image_elf}
            $<TARGET_OBJECTS:${stub}>
    # 2. Raw image bytes (header + descriptor + data), crc32 field still 0.
    COMMAND ${CMAKE_OBJCOPY} -O binary -j .model_image ${image_elf} ${image_bin_raw}
    # 3. Patch the header CRC over the finished binary.
    COMMAND ${PYTHON_EXECUTABLE} ${crc_tool} --bin ${image_bin_raw} -o ${image_bin}
    # 4. Fail the build if the on-flash header disagrees with the linked layout.
    COMMAND ${PYTHON_EXECUTABLE} ${validate_tool}
            --elf ${image_elf} --bin ${image_bin}
            --partition-addr ${partition_addr} --partition-size ${partition_size}
            --defs-header ${defs_header}
    # 5. Addressed hex for flashing the model into its partition, INDEPENDENTLY of the app.
    #    This is a separate, standalone artifact; it is deliberately NOT merged into zephyr.hex,
    #    since model-only OTA means the app image and each model partition are flashed/updated
    #    on their own.
    COMMAND ${CMAKE_OBJCOPY} -I binary -O ihex --change-addresses=${partition_addr}
            ${image_bin} ${image_hex}
    DEPENDS $<TARGET_OBJECTS:${stub}> ${linker_script} ${crc_tool} ${validate_tool}
    COMMENT "Building Neuton model partition image '${MI_NAME}' at ${partition_addr}"
    COMMAND_EXPAND_LISTS
    VERBATIM)

  # Standalone, independently-flashable per-model artifacts at the build dir root:
  #   <TARGET>_model_image.bin, <TARGET>_model_partition.hex
  # Intermediates remain under <TARGET>/.
  add_custom_target(${MI_TARGET}_model_image ALL DEPENDS ${image_bin} ${image_hex})
endfunction()
