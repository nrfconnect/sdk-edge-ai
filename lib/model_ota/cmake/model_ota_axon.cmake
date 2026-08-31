#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Edge AI Lab / Axon-backend and raw Axon model-only OTA.
#
# model_ota_edgeai_axon_model(TARGET <id> SOLUTION_ID <id> MODEL_SRC <abs nrf_edgeai_user_model.c>
#                             HEADER <abs nrf_edgeai_user_model_axon.h>
#                             PARTITION_NODELABEL <dt-nodelabel>
#                             [NAME <str>] [VERSION <x.y.z>] [PERSISTENT_VARS_CAP <n>]
#                             [MODEL_SYM <sym>] [ALLOCATE_PACKED_OUTPUT])
#
# model_ota_axon_model(TARGET <id> HEADER <nrf_axon_model_*.h> PARTITION_NODELABEL <dt-nodelabel>
#                      [NAME <str>] [VERSION <x.y.z>] [PERSISTENT_VARS_CAP <n>] [MODEL_SYM <sym>]
#                      [ALLOCATE_PACKED_OUTPUT])

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/model_ota_common.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/model_ota_context.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/model_ota_image.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/model_ota_wired.cmake)

set(MODEL_OTA_AXON_PROBE_SRC ${MODEL_OTA_LIB_DIR}/src/model_ota_axon_probe.c)
set(MODEL_OTA_AXON_APP_STUB ${MODEL_OTA_LIB_DIR}/src/model_ota_axon_app_stub.c)
set(MODEL_OTA_AXON_IMAGE_STUB ${MODEL_OTA_LIB_DIR}/src/model_ota_axon_image_stub.c)
set(MODEL_OTA_AXON_KEEP_REFS ${MODEL_OTA_LIB_DIR}/src/model_ota_axon_keep_refs.S)
set(MODEL_OTA_AXON_ELF ${MODEL_OTA_TOOLS_DIR}/axon_elf.py)
set(MODEL_OTA_AXON_WIRED_SRC ${MODEL_OTA_LIB_DIR}/src/model_ota_axon_wired.c)
set(MODEL_OTA_EDGEAI_AXON_WIRED_SRC ${MODEL_OTA_LIB_DIR}/src/model_ota_edgeai_axon_wired.c)

function(model_ota_axon_add_probe OUT_OBJ WORK_DIR HEADER HEADER_NAME HEADER_DIR)
  model_ota_zephyr_c_compile_flags(_zephyr_cflags)

  set(_probe_o ${WORK_DIR}/axon_probe.o)
  set(_probe_d ${WORK_DIR}/axon_probe.d)
  add_custom_command(
    OUTPUT ${_probe_o}
    COMMAND ${CMAKE_C_COMPILER}
            -c ${MODEL_OTA_AXON_PROBE_SRC}
            -o ${_probe_o}
            -MMD -MF ${_probe_d}
            ${_zephyr_cflags}
            -I${MODEL_OTA_LIB_DIR}/src
            -I${HEADER_DIR}
            -I${MODEL_OTA_MODULE_DIR}/include
            -include ${CMAKE_CURRENT_BINARY_DIR}/zephyr/include/generated/zephyr/autoconf.h
            -DMODEL_OTA_AXON_PROBE
            -DMODEL_OTA_AXON_HEADER=\"${HEADER_NAME}\"
            -DNRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER=1
            -DNRF_AXON_INTERLAYER_BUFFER_SIZE=${CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE}
    COMMAND_EXPAND_LISTS
    DEPENDS ${MODEL_OTA_AXON_PROBE_SRC} ${HEADER} zephyr_generated_headers
    DEPFILE ${_probe_d}
    COMMENT "Compiling Axon ELF probe (${HEADER_NAME})"
    VERBATIM
  )
  set(${OUT_OBJ} ${_probe_o} PARENT_SCOPE)
endfunction()

function(_model_ota_axon_slot)
  cmake_parse_arguments(MI "ALLOCATE_PACKED_OUTPUT" "FLAVOR"
    "TARGET;HEADER;PARTITION_NODELABEL;NAME;VERSION;PERSISTENT_VARS_CAP;MODEL_SYM;MODEL_SRC;SOLUTION_ID"
    ${ARGN})

  if(NOT MI_FLAVOR OR NOT MI_TARGET OR NOT MI_HEADER OR NOT MI_PARTITION_NODELABEL)
    message(FATAL_ERROR
            "_model_ota_axon_slot requires FLAVOR, TARGET, HEADER and PARTITION_NODELABEL")
  endif()
  if(MI_FLAVOR STREQUAL "edgeai_axon")
    if(NOT MI_MODEL_SRC OR NOT MI_SOLUTION_ID)
      message(FATAL_ERROR
              "model_ota_edgeai_axon_model requires MODEL_SRC and SOLUTION_ID")
    endif()
  elseif(MI_MODEL_SRC OR MI_SOLUTION_ID)
    message(FATAL_ERROR "model_ota_axon_model does not accept MODEL_SRC or SOLUTION_ID")
  endif()
  if(NOT EXISTS ${MI_HEADER})
    message(FATAL_ERROR "Axon OTA: HEADER not found: ${MI_HEADER}")
  endif()
  if(MI_FLAVOR STREQUAL "edgeai_axon" AND TARGET ota_edgeai_axon_${MI_TARGET})
    message(FATAL_ERROR "duplicate Edge AI Lab / Axon OTA TARGET ${MI_TARGET}")
  endif()
  if(TARGET ota_axon_${MI_TARGET})
    message(FATAL_ERROR "duplicate Axon OTA TARGET ${MI_TARGET}")
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

  get_filename_component(_header_dir ${MI_HEADER} DIRECTORY)
  get_filename_component(_header_name ${MI_HEADER} NAME)

  set(_work_dir ${CMAKE_CURRENT_BINARY_DIR}/model_ota/${MI_TARGET})
  file(MAKE_DIRECTORY ${_work_dir})

  model_ota_axon_add_probe(_probe_o ${_work_dir} ${MI_HEADER} ${_header_name} ${_header_dir})

  set(_private_h ${_work_dir}/axon_config.h)
  set(_public_include_dir ${_work_dir}/include)
  set(_public_h ${_public_include_dir}/model_ota/axon/${MI_TARGET}.h)
  set(_inspect_cmd
    ${PYTHON_EXECUTABLE} ${MODEL_OTA_AXON_ELF} inspect
    --probe ${_probe_o}
    --header-name ${_header_name}
    --model-id ${MI_TARGET}
    --private-header ${_private_h}
    --public-header ${_public_h}
    --partition-addr ${_partition_addr}
  )
  if(MI_FLAVOR STREQUAL "edgeai_axon")
    list(APPEND _inspect_cmd --edgeai)
  endif()
  if(MI_PERSISTENT_VARS_CAP)
    list(APPEND _inspect_cmd --persistent-vars-cap ${MI_PERSISTENT_VARS_CAP})
  endif()
  if(MI_MODEL_SYM)
    list(APPEND _inspect_cmd --model-sym ${MI_MODEL_SYM})
  endif()
  if(MI_ALLOCATE_PACKED_OUTPUT)
    list(APPEND _inspect_cmd --allocate-packed-output)
  endif()

  add_custom_command(
    OUTPUT ${_private_h} ${_public_h}
    COMMAND ${_inspect_cmd}
    DEPENDS ${_probe_o} ${MODEL_OTA_AXON_ELF}
    COMMENT "Inspecting Axon model metadata (${MI_TARGET})"
    VERBATIM
  )

  set(_meta_target ${MI_TARGET}_axon_metadata)
  add_custom_target(${_meta_target} DEPENDS ${_private_h} ${_public_h})

  if(MI_FLAVOR STREQUAL "edgeai_axon")
    model_ota_solution_id_hash(${MI_SOLUTION_ID} _solution_id_hash)
    model_ota_contract_probe(
      OUT_OBJ _contract_probe_o
      WORK_DIR ${_work_dir}
      FLAVOR edgeai_axon
      IMAGE_BASE ${_partition_addr}
      MODEL_SRC ${MI_MODEL_SRC}
      SOLUTION_ID_HASH ${_solution_id_hash})
  else()
    model_ota_contract_probe(
      OUT_OBJ _contract_probe_o
      WORK_DIR ${_work_dir}
      FLAVOR axon
      IMAGE_BASE ${_partition_addr})
  endif()

  model_ota_add_context_slot(
    TARGET ${MI_TARGET}
    BACKEND axon
    PARTITION_NODELABEL ${MI_PARTITION_NODELABEL}
    NAME ${MI_NAME}
    WORK_DIR ${_work_dir}
    CONTRACT_PROBE ${_contract_probe_o}
    CONFIG_HEADER ${_private_h}
    OUT_SLOT_JSON _context_slot)
  add_dependencies(${_meta_target} ${MI_TARGET}_contract_slot)

  model_ota_using_released_fw(_using_released_fw)

  if(MI_FLAVOR STREQUAL "edgeai_axon" AND NOT _using_released_fw)
    get_filename_component(_model_dir ${MI_MODEL_SRC} DIRECTORY)
    get_filename_component(_model_basename ${MI_MODEL_SRC} NAME)

    set(_wired_lib ota_edgeai_axon_${MI_TARGET})
    string(TOUPPER ${MI_TARGET} _axon_token)
    string(REGEX REPLACE "[^A-Z0-9]" "_" _axon_token "${_axon_token}")

    # TODO: MODEL_OTA_AXON_TARGET and MODEL_OTA_AXON_TOKEN only exist because the wired TU includes
    # the token-suffixed public header; force-including ${_private_h} instead removes both.
    model_ota_add_wired_library(
      LIB ${_wired_lib}
      SOURCE ${MODEL_OTA_EDGEAI_AXON_WIRED_SRC}
      ARCHIVE_DIR ${_work_dir}
      MODEL_SRC ${MI_MODEL_SRC}
      DESCRIPTION "solution ${MI_SOLUTION_ID} (${_wired_lib}, axon backend) <- ${MI_MODEL_SRC}"
      DISCARD_SECTIONS ${MODEL_OTA_EDGEAI_PARAM_SECTIONS}
      DEFINES
        MODEL_OTA_EDGEAI_SOLUTION_ID=${MI_SOLUTION_ID}
        MODEL_OTA_EDGEAI_AXON_MODEL_SRC=${_model_basename}
        MODEL_OTA_PARTITION_NODELABEL=${MI_PARTITION_NODELABEL}
        MODEL_OTA_AXON_TARGET=${MI_TARGET}
        MODEL_OTA_AXON_TOKEN=${_axon_token}
        MODEL_OTA_SOLUTION_ID_HASH=${_solution_id_hash}u
      INCLUDES ${_model_dir} ${_public_include_dir}
      DEPENDS ${_meta_target})
  endif()

  if(MI_FLAVOR STREQUAL "axon" AND NOT _using_released_fw)
    set(_wired_lib ota_axon_${MI_TARGET}_wired)
    string(TOUPPER ${MI_TARGET} _axon_token)
    string(REGEX REPLACE "[^A-Z0-9]" "_" _axon_token "${_axon_token}")

    model_ota_add_wired_library(
      LIB ${_wired_lib}
      SOURCE ${MODEL_OTA_AXON_WIRED_SRC}
      ARCHIVE_DIR ${_work_dir}
      DESCRIPTION "raw Axon ${MI_TARGET} (${_wired_lib}) partition loader"
      DEFINES
        MODEL_OTA_AXON_TARGET=${MI_TARGET}
        MODEL_OTA_AXON_TOKEN=${_axon_token}
        MODEL_OTA_PARTITION_NODELABEL=${MI_PARTITION_NODELABEL}
      INCLUDES ${_public_include_dir}
      DEPENDS ${_meta_target})
  endif()

  if(NOT _using_released_fw)
    set(_app_lib ota_axon_${MI_TARGET})
    add_library(${_app_lib} STATIC ${MODEL_OTA_AXON_APP_STUB} ${MODEL_OTA_AXON_KEEP_REFS})
    set_target_properties(${_app_lib} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${_work_dir})
    target_link_libraries(${_app_lib} PRIVATE zephyr_interface)
    target_include_directories(${_app_lib} PRIVATE
                               ${MODEL_OTA_LIB_DIR}/src ${_header_dir}
                               ${MODEL_OTA_MODULE_DIR}/include)
    target_compile_options(${_app_lib} PRIVATE "SHELL:-include \"${_private_h}\"")
    target_compile_definitions(${_app_lib} PRIVATE
                               MODEL_OTA_AXON_KEEP_LABEL=model_ota_axon_keep_${MI_TARGET})
    set_source_files_properties(
      ${MODEL_OTA_AXON_APP_STUB} ${MODEL_OTA_AXON_KEEP_REFS}
      TARGET_DIRECTORY ${_app_lib}
      PROPERTIES OBJECT_DEPENDS "${MI_HEADER};${_private_h}")
    add_dependencies(${_app_lib} ${_meta_target} zephyr_generated_headers)

    target_link_libraries(app PRIVATE ${_app_lib})
    target_include_directories(app PRIVATE ${_public_include_dir})
    add_dependencies(app ${_meta_target})
    toolchain_ld_force_undefined_symbols(model_ota_axon_keep_${MI_TARGET})
  endif()

  set(_image_stub ${MODEL_OTA_AXON_IMAGE_STUB})

  set(_image_obj ${MI_TARGET}_axon_image_obj)
  set(_image_deps "${MI_HEADER};${_private_h}")
  add_library(${_image_obj} OBJECT ${_image_stub})
  target_link_libraries(${_image_obj} PRIVATE zephyr_interface)
  target_include_directories(${_image_obj} PRIVATE
                             ${MODEL_OTA_LIB_DIR}/src ${_header_dir}
                             ${MODEL_OTA_MODULE_DIR}/include)
  target_compile_options(${_image_obj} PRIVATE "SHELL:-include \"${_private_h}\"")
  target_compile_definitions(${_image_obj} PRIVATE
    NRF_MODEL_PARTITION_ADDR=${_partition_addr}
    MODEL_IMAGE_NAME_STR=\"${MI_NAME}\"
    MODEL_IMAGE_VERSION_U32=${_version_u32}u
    NRF_AXON_INTERLAYER_BUFFER_SIZE=${CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE})
  if(MI_FLAVOR STREQUAL "edgeai_axon")
    get_filename_component(_edgeai_model_dir ${MI_MODEL_SRC} DIRECTORY)
    get_filename_component(_edgeai_model_basename ${MI_MODEL_SRC} NAME)
    target_include_directories(${_image_obj} PRIVATE ${_edgeai_model_dir})
    target_compile_definitions(${_image_obj} PRIVATE
      MODEL_OTA_EDGEAI_AXON_MODEL_SRC=${_edgeai_model_basename}
      MODEL_OTA_SOLUTION_ID_HASH=${_solution_id_hash}u)
    list(APPEND _image_deps ${MI_MODEL_SRC})
  endif()
  set_source_files_properties(
    ${_image_stub}
    TARGET_DIRECTORY ${_image_obj}
    PROPERTIES OBJECT_DEPENDS "${_image_deps}")
  add_dependencies(${_image_obj} ${_meta_target} zephyr_generated_headers)

  set(_model_syms_ld ${_work_dir}/${MI_TARGET}_model_syms.ld)
  set(_zephyr_elf ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.elf)

  if(MODEL_OTA_FW_ELF)
    set(_symbol_elf ${MODEL_OTA_FW_ELF})
  else()
    set(_symbol_elf ${_zephyr_elf})
  endif()

  set(_provide_deps ${MODEL_OTA_AXON_ELF} $<TARGET_OBJECTS:${_image_obj}>)
  if(MODEL_OTA_FW_ELF)
    list(APPEND _provide_deps ${MODEL_OTA_FW_ELF})
  else()
    list(APPEND _provide_deps ${_zephyr_elf})
  endif()

  add_custom_command(
    OUTPUT ${_model_syms_ld}
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_AXON_ELF} provide
            --object $<TARGET_OBJECTS:${_image_obj}>
            --elf ${_symbol_elf}
            -o ${_model_syms_ld}
    DEPENDS ${_provide_deps}
    COMMAND_EXPAND_LISTS
    COMMENT "Resolving Axon app symbols from ${_symbol_elf} (${MI_TARGET})"
    VERBATIM)

  set(_exclude_from_all FALSE)
  if(_using_released_fw AND NOT MODEL_OTA_FW_ELF)
    set(_exclude_from_all TRUE)
  endif()

  if(_exclude_from_all)
    model_ota_add_image(
      TARGET ${MI_TARGET}
      OBJ_LIB ${_image_obj}
      WORK_DIR ${_work_dir}
      PARTITION_ADDR ${_partition_addr}
      PARTITION_SIZE ${_partition_size}
      NAME ${MI_NAME}
      LINK_SCRIPTS ${_model_syms_ld}
      VALIDATE_ARGS --params-type 3 --config-header ${_private_h}
      COMPAT_ARGS --elf ${_symbol_elf}
      DEPENDS ${_model_syms_ld} ${_private_h}
      EXCLUDE_FROM_ALL)
  else()
    model_ota_add_image(
      TARGET ${MI_TARGET}
      OBJ_LIB ${_image_obj}
      WORK_DIR ${_work_dir}
      PARTITION_ADDR ${_partition_addr}
      PARTITION_SIZE ${_partition_size}
      NAME ${MI_NAME}
      LINK_SCRIPTS ${_model_syms_ld}
      VALIDATE_ARGS --params-type 3 --config-header ${_private_h}
      COMPAT_ARGS --elf ${_symbol_elf}
      DEPENDS ${_model_syms_ld} ${_private_h})
  endif()
endfunction()

function(model_ota_edgeai_axon_model)
  _model_ota_axon_slot(FLAVOR edgeai_axon ${ARGN})
endfunction()

function(model_ota_axon_model)
  _model_ota_axon_slot(FLAVOR axon ${ARGN})
endfunction()
