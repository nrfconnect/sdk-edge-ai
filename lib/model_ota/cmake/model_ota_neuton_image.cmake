#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Neuton model-only OTA: build the model as a self-contained, linked partition IMAGE
# (same linked-at-partition-base layout as Axon images).
#
# model_ota_neuton_image(TARGET <prefix> SOLUTION_ID <id> MODEL_SRC <abs nrf_edgeai_user_model.c>
#                        PARTITION_NODELABEL <dt-nodelabel>
#                        [NAME <str>] [VERSION <x.y.z>] [NEURONS_CAP <n>])
#
# The neuron cap published in model_ota_context.json comes from the model_ota_neuton_wire() call
# for the same partition; NEURONS_CAP is only needed when the image is built without one.
#
# SOLUTION_ID must be the same value the application passed to model_ota_neuton_wire(): it feeds
# the contract hash (see model_ota_solution_id_hash() in model_ota_common.cmake), so a mismatch
# makes the image report as incompatible rather than producing a subtly wrong one.
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

include(${CMAKE_CURRENT_LIST_DIR}/model_ota_context.cmake)

function(model_ota_neuton_image)
  cmake_parse_arguments(MI ""
    "TARGET;SOLUTION_ID;MODEL_SRC;PARTITION_NODELABEL;NAME;VERSION;NEURONS_CAP" "" ${ARGN})

  if(NOT MI_TARGET OR NOT MI_MODEL_SRC OR NOT MI_PARTITION_NODELABEL)
    message(FATAL_ERROR
            "model_ota_neuton_image requires TARGET, MODEL_SRC and PARTITION_NODELABEL")
  endif()
  if(NOT MI_SOLUTION_ID)
    message(FATAL_ERROR
            "model_ota_neuton_image requires SOLUTION_ID (must match model_ota_neuton_wire)")
  endif()
  model_ota_using_released_fw(_using_released_fw)

  # The cap is the *application's* scratch capacity, recorded in model_ota_context.json so that
  # check_model_compat.py can tell "model outgrew this firmware" from "incompatible model". It is
  # not part of the contract hash, so nothing else would catch it being wrong: the number has to
  # be the buffer model_ota_neuton_wire() allocated, which is why it is taken from the wire rather
  # than repeated here. NEURONS_CAP remains accepted for a wire-less image build, but a value
  # conflicting with the wire is refused rather than silently published.
  get_property(_wired_cap GLOBAL PROPERTY
               model_ota_neuton_wired_cap_${MI_PARTITION_NODELABEL})
  if(_wired_cap)
    if(MI_NEURONS_CAP AND NOT MI_NEURONS_CAP EQUAL _wired_cap)
      message(FATAL_ERROR
              "model_ota_neuton_image(${MI_TARGET}): NEURONS_CAP ${MI_NEURONS_CAP} disagrees with "
              "the application scratch buffer for ${MI_PARTITION_NODELABEL} "
              "(model_ota_neuton_wire MAX_NEURONS ${_wired_cap}); drop NEURONS_CAP or make them "
              "match")
    endif()
    set(MI_NEURONS_CAP ${_wired_cap})
  elseif(NOT MI_NEURONS_CAP AND NOT _using_released_fw)
    message(FATAL_ERROR
            "model_ota_neuton_image(${MI_TARGET}): no model_ota_neuton_wire() for "
            "${MI_PARTITION_NODELABEL}, so NEURONS_CAP is required")
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

  set(work_dir ${CMAKE_CURRENT_BINARY_DIR}/model_ota/${MI_TARGET})
  file(MAKE_DIRECTORY ${work_dir})

  set(stub_src       ${MODEL_OTA_ROOT}/src/model_ota_neuton_image_stub.c)
  set(image_elf     ${work_dir}/${MI_TARGET}_model_image.elf)
  set(image_bin_raw ${work_dir}/${MI_TARGET}_model_image_raw.bin)
  set(image_bin     ${work_dir}/${MI_TARGET}_model_image.bin)
  set(image_hex     ${CMAKE_CURRENT_BINARY_DIR}/${MI_TARGET}_model_partition.hex)

  set(linker_script ${MODEL_OTA_ROOT}/linker/model_image.ld)
  set(crc_tool ${EDGE_AI_MODULE_ROOT}/tools/model_ota/patch_image_crc.py)
  set(validate_tool ${EDGE_AI_MODULE_ROOT}/tools/model_ota/validate_model_image_layout.py)
  set(defs_header ${EDGE_AI_MODULE_ROOT}/include/model_ota/model_image.h)
  set(_compat_tool ${EDGE_AI_MODULE_ROOT}/tools/model_ota/check_model_compat.py)
  set(_context_slot_tool ${EDGE_AI_MODULE_ROOT}/tools/model_ota/emit_contract_slot.py)
  set(_generated_context ${CMAKE_CURRENT_BINARY_DIR}/model_ota_context.json)

  if(MODEL_OTA_FW_CONTEXT)
    set(_compat_context ${MODEL_OTA_FW_CONTEXT})
  else()
    set(_compat_context ${_generated_context})
  endif()

  set(stub tgt_${MI_TARGET}_model_image_stub)
  add_library(${stub} OBJECT ${stub_src})
  target_link_libraries(${stub} PRIVATE zephyr_interface)
  add_dependencies(${stub} zephyr_generated_headers)
  target_include_directories(${stub} PRIVATE ${model_dir})
  target_compile_options(${stub} PRIVATE -ffunction-sections -fdata-sections)

  model_ota_solution_id_hash(${MI_SOLUTION_ID} _solution_id_hash)

  # The contract hash is the compiler's own, read back out of a probe object rather than
  # recomputed on the host, so it only becomes known at build time - hence the slot's
  # contract_hash arrives via context_slot.json instead of model_ota_context_register_slot().
  model_ota_contract_probe(
    OUT_OBJ _contract_probe_o
    WORK_DIR ${work_dir}
    FLAVOR neuton
    IMAGE_BASE ${partition_addr}
    MODEL_SRC ${MI_MODEL_SRC}
    SOLUTION_ID_HASH ${_solution_id_hash})

  set(_context_slot ${work_dir}/context_slot.json)
  add_custom_command(
    OUTPUT ${_context_slot}
    COMMAND ${PYTHON_EXECUTABLE} ${_context_slot_tool}
            --contract-probe ${_contract_probe_o} --out ${_context_slot}
    DEPENDS ${_contract_probe_o} ${_context_slot_tool}
    COMMENT "Emitting Neuton OTA context slot metadata (${MI_TARGET})"
    VERBATIM)
  add_custom_target(${MI_TARGET}_contract_slot DEPENDS ${_context_slot})

  if(CONFIG_MODEL_OTA AND NOT _using_released_fw)
    model_ota_context_register_slot(
      TARGET ${MI_TARGET}
      BACKEND neuton
      PARTITION_NODELABEL ${MI_PARTITION_NODELABEL}
      NAME ${MI_NAME}
      NEURONS_CAP ${MI_NEURONS_CAP})
    model_ota_context_register_slot_build(SLOT_JSON ${_context_slot})
  endif()

  target_compile_definitions(${stub} PRIVATE
                             MODEL_OTA_NEUTON_MODEL_SRC=${model_basename}
                             NRF_MODEL_PARTITION_ADDR=${partition_addr}
                             MODEL_IMAGE_NAME_STR=\"${MI_NAME}\"
                             MODEL_IMAGE_VERSION_U32=${ver_u32}u
                             MODEL_OTA_SOLUTION_ID_HASH=${_solution_id_hash}u)
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
    # 6. Compare the image against the firmware context: the caps are a report, but a contract
    #    hash mismatch fails the build even under --report-only.
    COMMAND ${PYTHON_EXECUTABLE} ${_compat_tool}
            --context ${_compat_context} --image ${image_bin} --slot ${MI_TARGET}
            --report-only
    DEPENDS $<TARGET_OBJECTS:${stub}> ${linker_script} ${crc_tool} ${validate_tool}
            ${_compat_context} ${_compat_tool}
    COMMENT "Building Neuton model partition image '${MI_NAME}' at ${partition_addr}"
    COMMAND_EXPAND_LISTS
    VERBATIM)

  add_custom_target(${MI_TARGET}_model_image ALL DEPENDS ${image_bin} ${image_hex})
  if(TARGET model_ota_context AND NOT _using_released_fw)
    add_dependencies(${MI_TARGET}_model_image model_ota_context)
  endif()
endfunction()
