#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Edge AI Lab / Neuton-backend model-only OTA: app wiring and partition image from one declaration.
#
# model_ota_edgeai_neuton_model(TARGET <id> SOLUTION_ID <id> MODEL_SRC <abs nrf_edgeai_user_model.c>
#                               PARTITION_NODELABEL <dt-nodelabel> NEURONS_CAP <n>
#                               [NAME <str>] [VERSION <x.y.z>])

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/model_ota_common.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/model_ota_context.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/model_ota_image.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/model_ota_wired.cmake)

set(MODEL_OTA_EDGEAI_NEUTON_IMAGE_STUB
    ${MODEL_OTA_LIB_DIR}/src/model_ota_edgeai_neuton_image_stub.c)
set(MODEL_OTA_EDGEAI_NEUTON_WIRED_TPL
    ${MODEL_OTA_LIB_DIR}/src/model_ota_edgeai_neuton_wired.c.in)

set(MODEL_OTA_EDGEAI_NEUTON_PAYLOAD_SECTIONS
    .rodata.MODEL_WEIGHTS
    .rodata.MODEL_NEURON_ACTIVATION_WEIGHTS
    .rodata.MODEL_NEURON_ACTIVATION_TYPE_MASK
    .rodata.MODEL_NEURONS_LINKS
    .rodata.MODEL_NEURON_INTERNAL_LINKS_NUM
    .rodata.MODEL_NEURON_EXTERNAL_LINKS_NUM
    .rodata.MODEL_OUTPUT_NEURONS_INDICES
    ${MODEL_OTA_EDGEAI_PARAM_SECTIONS}
)

function(model_ota_edgeai_neuton_model)
  cmake_parse_arguments(MI ""
    "TARGET;SOLUTION_ID;MODEL_SRC;PARTITION_NODELABEL;NAME;VERSION;NEURONS_CAP" "" ${ARGN})

  if(NOT MI_TARGET OR NOT MI_SOLUTION_ID OR NOT MI_MODEL_SRC OR NOT MI_PARTITION_NODELABEL
     OR NOT MI_NEURONS_CAP)
    message(FATAL_ERROR
            "model_ota_edgeai_neuton_model requires TARGET, SOLUTION_ID, MODEL_SRC, "
            "PARTITION_NODELABEL and NEURONS_CAP")
  endif()
  if(TARGET ota_edgeai_neuton_${MI_TARGET})
    message(FATAL_ERROR "model_ota_edgeai_neuton_model: duplicate TARGET ${MI_TARGET}")
  endif()
  if(NOT MI_NAME)
    set(MI_NAME ${MI_TARGET})
  endif()
  if(NOT MI_VERSION)
    set(MI_VERSION "1.0.0")
  endif()

  model_ota_pack_version("${MI_VERSION}" _version_u32)
  dt_nodelabel(_partition_node NODELABEL ${MI_PARTITION_NODELABEL} REQUIRED)
  dt_reg_addr(_partition_addr PATH ${_partition_node})
  dt_reg_size(_partition_size PATH ${_partition_node})

  get_filename_component(_model_dir ${MI_MODEL_SRC} DIRECTORY)
  get_filename_component(_model_basename ${MI_MODEL_SRC} NAME)

  set(_work_dir ${CMAKE_CURRENT_BINARY_DIR}/model_ota/${MI_TARGET})
  file(MAKE_DIRECTORY ${_work_dir})

  model_ota_solution_id_hash(${MI_SOLUTION_ID} _solution_id_hash)

  model_ota_contract_probe(
    OUT_OBJ _contract_probe_o
    WORK_DIR ${_work_dir}
    FLAVOR edgeai_neuton
    IMAGE_BASE ${_partition_addr}
    MODEL_SRC ${MI_MODEL_SRC}
    SOLUTION_ID_HASH ${_solution_id_hash})

  model_ota_add_context_slot(
    TARGET ${MI_TARGET}
    BACKEND neuton
    PARTITION_NODELABEL ${MI_PARTITION_NODELABEL}
    NAME ${MI_NAME}
    WORK_DIR ${_work_dir}
    CONTRACT_PROBE ${_contract_probe_o}
    NEURONS_CAP ${MI_NEURONS_CAP}
    OUT_SLOT_JSON _context_slot)

  model_ota_using_released_fw(_using_released_fw)

  if(NOT _using_released_fw)
    set(_wired_lib ota_edgeai_neuton_${MI_TARGET})
    set(_wired_src ${_work_dir}/model_ota_edgeai_neuton_wired_${MI_SOLUTION_ID}.c)

    # TODO: unprefixed template variables, picked up by configure_file() in
    # model_ota_add_wired_library() through CMake scope chaining rather than passed as arguments.
    # All but SOLUTION_ID should become DEFINES below; see model_ota_wired.cmake.
    set(SOLUTION_ID ${MI_SOLUTION_ID})
    set(PARTITION_NODELABEL ${MI_PARTITION_NODELABEL})
    set(NEURONS_CAP ${MI_NEURONS_CAP})
    set(MODEL_SRC_BASENAME ${_model_basename})

    model_ota_add_wired_library(
      LIB ${_wired_lib}
      TEMPLATE ${MODEL_OTA_EDGEAI_NEUTON_WIRED_TPL}
      OUT_SRC ${_wired_src}
      MODEL_SRC ${MI_MODEL_SRC}
      DESCRIPTION
        "solution ${MI_SOLUTION_ID} (${_wired_lib}, neurons_cap=${MI_NEURONS_CAP}) <- ${MI_MODEL_SRC}"
      DISCARD_SECTIONS ${MODEL_OTA_EDGEAI_NEUTON_PAYLOAD_SECTIONS}
      DEFINES
        NRF_MODEL_PARTITION_ADDR=${_partition_addr}
        MODEL_OTA_SOLUTION_ID_HASH=${_solution_id_hash}u
      INCLUDES ${_model_dir})
  endif()

  set(_image_obj tgt_${MI_TARGET}_model_image_stub)
  add_library(${_image_obj} OBJECT ${MODEL_OTA_EDGEAI_NEUTON_IMAGE_STUB})
  target_link_libraries(${_image_obj} PRIVATE zephyr_interface)
  add_dependencies(${_image_obj} zephyr_generated_headers)
  target_include_directories(${_image_obj} PRIVATE ${_model_dir})
  target_compile_options(${_image_obj} PRIVATE -ffunction-sections -fdata-sections)
  target_compile_definitions(${_image_obj} PRIVATE
                             MODEL_OTA_EDGEAI_NEUTON_MODEL_SRC=${_model_basename}
                             NRF_MODEL_PARTITION_ADDR=${_partition_addr}
                             MODEL_IMAGE_NAME_STR=\"${MI_NAME}\"
                             MODEL_IMAGE_VERSION_U32=${_version_u32}u
                             MODEL_OTA_SOLUTION_ID_HASH=${_solution_id_hash}u)
  set_source_files_properties(${MODEL_OTA_EDGEAI_NEUTON_IMAGE_STUB}
                              TARGET_DIRECTORY ${_image_obj}
                              PROPERTIES OBJECT_DEPENDS "${MI_MODEL_SRC}")

  model_ota_add_image(
    TARGET ${MI_TARGET}
    OBJ_LIB ${_image_obj}
    WORK_DIR ${_work_dir}
    PARTITION_ADDR ${_partition_addr}
    PARTITION_SIZE ${_partition_size}
    NAME ${MI_NAME})
endfunction()
