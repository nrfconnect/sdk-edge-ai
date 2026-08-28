#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# CMake helper for Neuton model-only OTA: per-model static library + payload discard.
#
# model_ota_neuton_wire(SOLUTION_ID <id> MODEL_SRC <abs-path-to-nrf_edgeai_user_model.c>
#                       PARTITION_NODELABEL <dt-nodelabel> MAX_NEURONS <cap>
#                       [LIB_NAME <static-lib-target>])
#
# For each OTA-updatable model:
#
#   1. Generates lib/model_ota/src/model_ota_neuton_wired.c.in into
#      ${CMAKE_CURRENT_BINARY_DIR}/model_ota_neuton_wired_<SOLUTION_ID>.c (partition-load
#      wrapper + #include of the generated model). Builds a dedicated static library (default
#      target ota_neuton_<SOLUTION_ID>) from that file with MODEL_OTA_WIRED and a per-model
#      MODEL_OTA_NEUTON_MAX_NEURONS. Models compiled directly into the app are unaffected
#      and keep compile-time descriptors and payload.
#
#   2. Drops the payload from that library via model_ota_discard_register() (per-section,
#      archive-scoped /DISCARD/ rules for *that* library only).
#
# Partition images use lib/model_ota/src/model_ota_neuton_image_stub.c instead (compile-time
# descriptor + payload kept for the linked image).

include_guard(GLOBAL)

get_filename_component(EDGE_AI_MODULE_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)
get_filename_component(MODEL_OTA_ROOT ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)

include(${CMAKE_CURRENT_LIST_DIR}/model_ota_common.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/model_ota_context.cmake)

# The compiled Neuton model, on top of the shared nrf_edgeai_t parameter arrays.
set(MODEL_OTA_NEUTON_PAYLOAD_SECTIONS
    .rodata.MODEL_WEIGHTS
    .rodata.MODEL_NEURON_ACTIVATION_WEIGHTS
    .rodata.MODEL_NEURON_ACTIVATION_TYPE_MASK
    .rodata.MODEL_NEURONS_LINKS
    .rodata.MODEL_NEURON_INTERNAL_LINKS_NUM
    .rodata.MODEL_NEURON_EXTERNAL_LINKS_NUM
    .rodata.MODEL_OUTPUT_NEURONS_INDICES
    ${MODEL_OTA_EDGEAI_PARAM_SECTIONS}
)

function(model_ota_neuton_wire)
  cmake_parse_arguments(MO "" "SOLUTION_ID;MODEL_SRC;LIB_NAME;MAX_NEURONS;PARTITION_NODELABEL" "" ${ARGN})

  model_ota_using_released_fw(_using_released_fw)
  if(_using_released_fw)
    return()
  endif()

  if(NOT MO_MODEL_SRC)
    message(FATAL_ERROR "model_ota_neuton_wire: MODEL_SRC is required")
  endif()
  if(NOT MO_SOLUTION_ID)
    message(FATAL_ERROR "model_ota_neuton_wire: SOLUTION_ID is required")
  endif()
  if(NOT MO_MAX_NEURONS)
    message(FATAL_ERROR "model_ota_neuton_wire: MAX_NEURONS is required")
  endif()
  if(NOT MO_PARTITION_NODELABEL)
    message(FATAL_ERROR "model_ota_neuton_wire: PARTITION_NODELABEL is required")
  endif()
  if(NOT MO_LIB_NAME)
    set(MO_LIB_NAME ota_neuton_${MO_SOLUTION_ID})
  endif()

  get_filename_component(model_dir ${MO_MODEL_SRC} DIRECTORY)
  get_filename_component(model_basename ${MO_MODEL_SRC} NAME)
  set(wired_tpl ${MODEL_OTA_ROOT}/src/model_ota_neuton_wired.c.in)
  set(wired_src ${CMAKE_CURRENT_BINARY_DIR}/model_ota_neuton_wired_${MO_SOLUTION_ID}.c)

  if(TARGET ${MO_LIB_NAME})
    message(FATAL_ERROR "model_ota_neuton_wire: duplicate LIB_NAME/target ${MO_LIB_NAME}")
  endif()

  set(SOLUTION_ID ${MO_SOLUTION_ID})
  set(PARTITION_NODELABEL ${MO_PARTITION_NODELABEL})
  set(MAX_NEURONS ${MO_MAX_NEURONS})
  set(MODEL_SRC_BASENAME ${model_basename})

  # model_ota_neuton_image() reads this back instead of taking the number again, so the cap
  # published in model_ota_context.json is the buffer this translation unit actually allocates.
  # A published cap larger than the real buffer makes check_model_compat.py call an image
  # compatible that the loader then rejects with MODEL_IMAGE_ERR_NEURONS_BUF_TOO_SMALL.
  set_property(GLOBAL PROPERTY model_ota_neuton_wired_cap_${MO_PARTITION_NODELABEL}
               ${MO_MAX_NEURONS})

  # The contract hash covers the flash base the image is linked at, so the application has to
  # name the same partition the image was built for. The wired translation unit computes the hash
  # itself from the same generated source, so nothing but the base has to be passed in.
  dt_nodelabel(_partition_node NODELABEL ${MO_PARTITION_NODELABEL} REQUIRED)
  dt_reg_addr(_partition_addr PATH ${_partition_node})

  model_ota_solution_id_hash(${MO_SOLUTION_ID} _solution_id_hash)

  configure_file(${wired_tpl} ${wired_src} @ONLY)

  add_library(${MO_LIB_NAME} STATIC ${wired_src})
  target_link_libraries(${MO_LIB_NAME} PRIVATE zephyr_interface)
  target_compile_definitions(${MO_LIB_NAME} PRIVATE
                             NRF_MODEL_PARTITION_ADDR=${_partition_addr}
                             MODEL_OTA_SOLUTION_ID_HASH=${_solution_id_hash}u)
  add_dependencies(${MO_LIB_NAME} zephyr_generated_headers)
  target_include_directories(${MO_LIB_NAME} PRIVATE ${model_dir} ${MODEL_OTA_ROOT}/src)
  set_source_files_properties(${wired_src}
                              TARGET_DIRECTORY ${MO_LIB_NAME}
                              PROPERTIES OBJECT_DEPENDS "${MO_MODEL_SRC}")
  target_link_libraries(app PRIVATE ${MO_LIB_NAME})

  model_ota_discard_register(
    LIB ${MO_LIB_NAME}
    DESCRIPTION
      "solution ${MO_SOLUTION_ID} (${MO_LIB_NAME}, max_neurons=${MO_MAX_NEURONS}) <- ${MO_MODEL_SRC}"
    SECTIONS ${MODEL_OTA_NEUTON_PAYLOAD_SECTIONS})
endfunction()
