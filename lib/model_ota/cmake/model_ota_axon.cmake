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
#                             [MODEL_SYM <sym>] [ALLOCATE_PACKED_OUTPUT]
#                             [UUID_VID <vid>] [UUID_CID <cid>])
#
# model_ota_axon_model(TARGET <id> HEADER <nrf_axon_model_*.h> PARTITION_NODELABEL <dt-nodelabel>
#                      [NAME <str>] [VERSION <x.y.z>] [PERSISTENT_VARS_CAP <n>] [MODEL_SYM <sym>]
#                      [ALLOCATE_PACKED_OUTPUT] [UUID_VID <vid>] [UUID_CID <cid>])
#
# model_ota_axon_bindings(SYMBOLS <symbol>...)
#
# With CONFIG_BOOTLOADER_MCUBOOT, the partition image is also signed as
# <TARGET>_model_mcuboot.signed.{bin,hex}; UUID_VID/UUID_CID go to imgtool --vid/--cid.

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
            -I${MODEL_OTA_LIB_DIR}/include
            -I${HEADER_DIR}
            -I${ZEPHYR_EDGE_AI_MODULE_DIR}/include
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

# PROVIDE() fragment shared by every Axon image link: the firmware's axon_bindings, from the
# generated context in-tree or MODEL_OTA_FW_CONTEXT out-of-tree. Image targets must depend on the
# model_ota_axon_bindings_ld target rather than only on the file, so the fragment has one producer.
function(model_ota_axon_bindings_ld OUT_VAR)
  set(_bindings_ld ${CMAKE_CURRENT_BINARY_DIR}/model_ota/axon_bindings.ld)
  set(${OUT_VAR} ${_bindings_ld} PARENT_SCOPE)
  if(TARGET model_ota_axon_bindings_ld)
    return()
  endif()

  if(MODEL_OTA_FW_CONTEXT)
    set(_symbol_context ${MODEL_OTA_FW_CONTEXT})
  else()
    set(_symbol_context ${CMAKE_CURRENT_BINARY_DIR}/model_ota/model_ota_context.json)
  endif()

  add_custom_command(
    OUTPUT ${_bindings_ld}
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_AXON_ELF} provide
            --context ${_symbol_context}
            -o ${_bindings_ld}
    DEPENDS ${MODEL_OTA_AXON_ELF} ${_symbol_context}
    COMMENT "Resolving Axon app symbols from ${_symbol_context}"
    VERBATIM)
  add_custom_target(model_ota_axon_bindings_ld DEPENDS ${_bindings_ld})
endfunction()

function(_model_ota_axon_slot)
  cmake_parse_arguments(MI "ALLOCATE_PACKED_OUTPUT" "FLAVOR"
    "TARGET;HEADER;PARTITION_NODELABEL;NAME;VERSION;PERSISTENT_VARS_CAP;MODEL_SYM;MODEL_SRC;SOLUTION_ID;UUID_VID;UUID_CID"
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

  model_ota_using_released_fw(_using_released_fw)

  model_ota_pack_version("${MI_VERSION}" _version_u32)
  dt_nodelabel(_partition_node NODELABEL ${MI_PARTITION_NODELABEL} REQUIRED)
  model_ota_claim_partition(${MI_TARGET} ${_partition_node})
  dt_reg_addr(_partition_addr PATH ${_partition_node})
  dt_reg_size(_partition_size PATH ${_partition_node})
  model_ota_image_link_addr(${_partition_addr} _image_link_addr)
  model_ota_model_image_offset(_image_model_offset)

  get_filename_component(_header_dir ${MI_HEADER} DIRECTORY)
  get_filename_component(_header_name ${MI_HEADER} NAME)

  set(_work_dir ${CMAKE_CURRENT_BINARY_DIR}/model_ota/${MI_TARGET})
  file(MAKE_DIRECTORY ${_work_dir})

  set(_shared_include_dir ${CMAKE_CURRENT_BINARY_DIR}/model_ota/include)
  file(MAKE_DIRECTORY ${_shared_include_dir})

  model_ota_axon_add_probe(_probe_o ${_work_dir} ${MI_HEADER} ${_header_name} ${_header_dir})

  set(_private_h ${_work_dir}/model_ota_axon_model_config.h)
  set(_keep_json ${_work_dir}/axon_keep.json)
  set(_public_h ${_shared_include_dir}/model_ota/slots/${MI_TARGET}.h)
  set(_inspect_cmd
    ${PYTHON_EXECUTABLE} ${MODEL_OTA_AXON_ELF} inspect
    --probe ${_probe_o}
    --header-name ${_header_name}
    --model-id ${MI_TARGET}
    --private-header ${_private_h}
    --public-header ${_public_h}
    --keep-json ${_keep_json}
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
    OUTPUT ${_private_h} ${_public_h} ${_keep_json}
    COMMAND ${_inspect_cmd}
    DEPENDS ${_probe_o} ${MODEL_OTA_AXON_ELF}
    COMMENT "Inspecting Axon model metadata (${MI_TARGET})"
    VERBATIM
  )

  set(_meta_target ${MI_TARGET}_axon_metadata)
  add_custom_target(${_meta_target} DEPENDS ${_private_h} ${_public_h} ${_keep_json})
  set_property(GLOBAL APPEND PROPERTY model_ota_axon_keep_json ${_keep_json})

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

  if(MI_FLAVOR STREQUAL "edgeai_axon" AND NOT _using_released_fw)
    get_filename_component(_model_dir ${MI_MODEL_SRC} DIRECTORY)
    get_filename_component(_model_basename ${MI_MODEL_SRC} NAME)

    set(_wired_lib ota_edgeai_axon_${MI_TARGET})

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
        MODEL_OTA_IMAGE_LINK_BASE=${_partition_addr}
        MODEL_OTA_SMP_SLOT_NAME=\"${MI_NAME}\"
        MODEL_OTA_SOLUTION_ID_HASH=${_solution_id_hash}u
      INCLUDES ${_work_dir} ${_model_dir}
      DEPENDS ${_meta_target})
  endif()

  if(MI_FLAVOR STREQUAL "axon" AND NOT _using_released_fw)
    set(_wired_lib ota_axon_${MI_TARGET}_wired)

    model_ota_add_wired_library(
      LIB ${_wired_lib}
      SOURCE ${MODEL_OTA_AXON_WIRED_SRC}
      ARCHIVE_DIR ${_work_dir}
      DESCRIPTION "raw Axon ${MI_TARGET} (${_wired_lib}) partition loader"
      DEFINES
        MODEL_OTA_AXON_TARGET=${MI_TARGET}
        MODEL_OTA_PARTITION_NODELABEL=${MI_PARTITION_NODELABEL}
        MODEL_OTA_IMAGE_LINK_BASE=${_partition_addr}
      INCLUDES ${_work_dir}
      DEPENDS ${_meta_target})
  endif()

  if(NOT _using_released_fw)
    set(_app_lib ota_axon_${MI_TARGET})
    add_library(${_app_lib} OBJECT ${MODEL_OTA_AXON_APP_STUB})
    target_link_libraries(${_app_lib} PRIVATE zephyr_interface)
    target_include_directories(${_app_lib} PRIVATE
                               ${MODEL_OTA_LIB_DIR}/src ${MODEL_OTA_LIB_DIR}/include
                               ${_header_dir} ${ZEPHYR_EDGE_AI_MODULE_DIR}/include)
    target_compile_options(${_app_lib} PRIVATE "SHELL:-include \"${_private_h}\"")
    set_source_files_properties(
      ${MODEL_OTA_AXON_APP_STUB}
      TARGET_DIRECTORY ${_app_lib}
      PROPERTIES OBJECT_DEPENDS "${MI_HEADER};${_private_h}")
    add_dependencies(${_app_lib} ${_meta_target} zephyr_generated_headers)

    target_link_libraries(app PRIVATE ${_app_lib})
    add_dependencies(app ${_meta_target})
  endif()

  set(_image_stub ${MODEL_OTA_AXON_IMAGE_STUB})

  set(_image_obj ${MI_TARGET}_axon_image_obj)
  set(_image_deps "${MI_HEADER};${_private_h}")
  add_library(${_image_obj} OBJECT ${_image_stub})
  target_link_libraries(${_image_obj} PRIVATE zephyr_interface)
  target_include_directories(${_image_obj} PRIVATE
                             ${MODEL_OTA_LIB_DIR}/src ${MODEL_OTA_LIB_DIR}/include
                             ${_header_dir} ${ZEPHYR_EDGE_AI_MODULE_DIR}/include)
  target_compile_options(${_image_obj} PRIVATE "SHELL:-include \"${_private_h}\"")
  target_compile_definitions(${_image_obj} PRIVATE
    MODEL_OTA_IMAGE_LINK_BASE=${_partition_addr}
    MODEL_IMAGE_LINK_ADDR=${_image_link_addr}
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

  model_ota_axon_bindings_ld(_bindings_ld)

  model_ota_add_image(
    TARGET ${MI_TARGET}
    OBJ_LIB ${_image_obj}
    WORK_DIR ${_work_dir}
    PARTITION_ADDR ${_partition_addr}
    PARTITION_SIZE ${_partition_size}
    NAME ${MI_NAME}
    VERSION ${MI_VERSION}
    UUID_VID ${MI_UUID_VID}
    UUID_CID ${MI_UUID_CID}
    LINK_SCRIPTS ${_bindings_ld}
    VALIDATE_ARGS --params-type 3 --config-header ${_private_h}
    DEPENDS ${_bindings_ld} ${_private_h})
  add_dependencies(${MI_TARGET}_model_image model_ota_axon_bindings_ld)
endfunction()

function(model_ota_edgeai_axon_model)
  _model_ota_axon_slot(FLAVOR edgeai_axon ${ARGN})
endfunction()

function(model_ota_axon_model)
  _model_ota_axon_slot(FLAVOR axon ${ARGN})
endfunction()

# Make application symbols bindable by future Axon model images although no model built with
# this firmware references them - typically an op extension a retrained model may start using.
# Each symbol is added to the device-wide binding table (so the loader accepts it and the linker
# keeps it) and to model_ota_context.json's axon_bindings (so an out-of-tree image can link
# against it). Call before model_ota_finalize().
#
# model_ota_axon_bindings(SYMBOLS <symbol>...)
function(model_ota_axon_bindings)
  cmake_parse_arguments(B "" "" "SYMBOLS" ${ARGN})
  if(NOT B_SYMBOLS)
    message(FATAL_ERROR "model_ota_axon_bindings requires SYMBOLS")
  endif()
  get_property(_finalized GLOBAL PROPERTY model_ota_axon_binding_finalized)
  if(_finalized)
    message(FATAL_ERROR "model_ota_axon_bindings must be called before model_ota_finalize()")
  endif()
  set_property(GLOBAL APPEND PROPERTY model_ota_axon_app_bindings ${B_SYMBOLS})
endfunction()

function(model_ota_axon_binding_finalize)
  set_property(GLOBAL PROPERTY model_ota_axon_binding_finalized TRUE)
  model_ota_using_released_fw(_using_released_fw)
  if(_using_released_fw)
    return()
  endif()

  get_property(_keep_json_files GLOBAL PROPERTY model_ota_axon_keep_json)
  if(NOT _keep_json_files)
    return()
  endif()

  get_property(_app_bindings GLOBAL PROPERTY model_ota_axon_app_bindings)
  if(_app_bindings)
    list(REMOVE_DUPLICATES _app_bindings)
    list(SORT _app_bindings)
    list(JOIN _app_bindings "\", \"" _app_bindings_json)
    set(_app_keep_json ${CMAKE_CURRENT_BINARY_DIR}/model_ota/app_axon_keep.json)
    file(CONFIGURE OUTPUT ${_app_keep_json}
         CONTENT "{\"target\": \"app\", \"keep_symbols\": [\"${_app_bindings_json}\"]}\n"
         @ONLY)
    list(APPEND _keep_json_files ${_app_keep_json})
  endif()

  set(_shared_include_dir ${CMAKE_CURRENT_BINARY_DIR}/model_ota/include)
  file(MAKE_DIRECTORY ${_shared_include_dir})
  target_include_directories(app PRIVATE ${_shared_include_dir})

  set(_binding_h ${_shared_include_dir}/model_ota/axon_binding_table.h)
  set(_binding_symbols_json ${CMAKE_CURRENT_BINARY_DIR}/model_ota/axon_binding_symbols.json)
  set(_binding_cmd
    ${PYTHON_EXECUTABLE} ${MODEL_OTA_AXON_ELF} binding-table
    -o ${_binding_h}
    --symbols-json ${_binding_symbols_json}
  )
  foreach(_keep_json ${_keep_json_files})
    list(APPEND _binding_cmd --keep-json ${_keep_json})
  endforeach()
  set_property(GLOBAL PROPERTY model_ota_axon_binding_symbols_json ${_binding_symbols_json})

  add_custom_command(
    OUTPUT ${_binding_h} ${_binding_symbols_json}
    COMMAND ${_binding_cmd}
    DEPENDS ${_keep_json_files} ${MODEL_OTA_AXON_ELF}
    COMMENT "Merging Axon OTA binding table"
    VERBATIM
  )

  set(_binding_lib model_ota_axon_binding)
  if(NOT TARGET ${_binding_lib})
    add_library(${_binding_lib} STATIC ${MODEL_OTA_AXON_KEEP_REFS})
    set_target_properties(${_binding_lib} PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/model_ota)
    target_link_libraries(${_binding_lib} PRIVATE zephyr_interface)
    target_include_directories(${_binding_lib} PRIVATE ${MODEL_OTA_LIB_DIR}/src)
    target_compile_options(${_binding_lib} PRIVATE "SHELL:-include \"${_binding_h}\"")
    set_source_files_properties(
      ${MODEL_OTA_AXON_KEEP_REFS}
      TARGET_DIRECTORY ${_binding_lib}
      PROPERTIES OBJECT_DEPENDS "${_binding_h}")
    add_dependencies(${_binding_lib} zephyr_generated_headers)
    target_link_libraries(app PRIVATE ${_binding_lib})
  endif()

  add_custom_target(model_ota_axon_binding_table DEPENDS ${_binding_h})
  add_dependencies(${_binding_lib} model_ota_axon_binding_table)
endfunction()
