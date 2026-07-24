#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Axon model-only OTA: app wiring and partition image from one model declaration.
#
# model_ota_axon_model(TARGET <id> HEADER <nrf_axon_model_*.h>
#                      PARTITION_NODELABEL <dt-nodelabel>
#                      [NAME <str>] [VERSION <x.y.z>]
#                      [PERSISTENT_VARS_CAP <n>] [MODEL_SYM <symbol>]
#                      [ALLOCATE_PACKED_OUTPUT])
#
# ALLOCATE_PACKED_OUTPUT allocates app-owned RAM for the model's optional
# packed-output buffer and wires it into the linked partition image (the model's
# packed_output_buf field). Without it (the default), the image links with
# packed_output_buf NULL and no app RAM is spent on it.

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/model_ota_common.cmake)

get_filename_component(MODEL_OTA_ROOT ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
get_filename_component(EDGE_AI_MODULE_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)

set(MODEL_OTA_AXON_PROBE_SRC ${MODEL_OTA_ROOT}/src/model_ota_axon_probe.c)
set(MODEL_OTA_AXON_APP_STUB ${MODEL_OTA_ROOT}/src/model_ota_axon_app_stub.c)
set(MODEL_OTA_AXON_IMAGE_STUB ${MODEL_OTA_ROOT}/src/model_ota_axon_image_stub.c)
set(MODEL_OTA_AXON_KEEP_REFS ${MODEL_OTA_ROOT}/src/model_ota_axon_keep_refs.S)
set(MODEL_OTA_AXON_ELF ${EDGE_AI_MODULE_ROOT}/tools/model_ota/axon_elf.py)
set(MODEL_OTA_AXON_LINKER_SCRIPT ${MODEL_OTA_ROOT}/linker/model_image.ld)
set(MODEL_OTA_AXON_CRC_TOOL ${EDGE_AI_MODULE_ROOT}/tools/model_ota/patch_image_crc.py)
set(MODEL_OTA_AXON_VALIDATE_TOOL
    ${EDGE_AI_MODULE_ROOT}/tools/model_ota/validate_model_image_layout.py)
set(MODEL_OTA_IMAGE_DEFS ${EDGE_AI_MODULE_ROOT}/include/model_ota/model_image.h)

function(model_ota_axon_zephyr_c_compile_flags OUT_VAR)
  zephyr_get_include_directories_for_lang(C _inc)
  zephyr_get_system_include_directories_for_lang(C _sys)
  zephyr_get_compile_definitions_for_lang(C _def)
  zephyr_get_compile_options_for_lang(C _opt)
  set(${OUT_VAR} ${_opt} ${_inc} ${_sys} ${_def} PARENT_SCOPE)
endfunction()

function(model_ota_axon_add_probe OUT_OBJ WORK_DIR HEADER HEADER_NAME HEADER_DIR)
  model_ota_axon_zephyr_c_compile_flags(_zephyr_cflags)

  set(_probe_o ${WORK_DIR}/axon_probe.o)
  set(_probe_d ${WORK_DIR}/axon_probe.d)
  add_custom_command(
    OUTPUT ${_probe_o}
    COMMAND ${CMAKE_C_COMPILER}
            -c ${MODEL_OTA_AXON_PROBE_SRC}
            -o ${_probe_o}
            -MMD -MF ${_probe_d}
            ${_zephyr_cflags}
            -I${MODEL_OTA_ROOT}/src
            -I${HEADER_DIR}
            -I${EDGE_AI_MODULE_ROOT}/include
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

function(model_ota_axon_model)
  cmake_parse_arguments(MI "ALLOCATE_PACKED_OUTPUT"
    "TARGET;HEADER;PARTITION_NODELABEL;NAME;VERSION;PERSISTENT_VARS_CAP;MODEL_SYM" "" ${ARGN})

  if(NOT MI_TARGET OR NOT MI_HEADER OR NOT MI_PARTITION_NODELABEL)
    message(FATAL_ERROR
            "model_ota_axon_model requires TARGET, HEADER and PARTITION_NODELABEL")
  endif()
  if(NOT EXISTS ${MI_HEADER})
    message(FATAL_ERROR "model_ota_axon_model: HEADER not found: ${MI_HEADER}")
  endif()
  if(TARGET ota_axon_${MI_TARGET})
    message(FATAL_ERROR "model_ota_axon_model: duplicate TARGET ${MI_TARGET}")
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
  )
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

  set(_app_lib ota_axon_${MI_TARGET})
  add_library(${_app_lib} STATIC ${MODEL_OTA_AXON_APP_STUB} ${MODEL_OTA_AXON_KEEP_REFS})
  target_link_libraries(${_app_lib} PRIVATE zephyr_interface)
  target_include_directories(${_app_lib} PRIVATE
                             ${MODEL_OTA_ROOT}/src ${_header_dir}
                             ${EDGE_AI_MODULE_ROOT}/include)
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

  set(_image_obj ${MI_TARGET}_axon_image_obj)
  add_library(${_image_obj} OBJECT ${MODEL_OTA_AXON_IMAGE_STUB})
  target_link_libraries(${_image_obj} PRIVATE zephyr_interface)
  target_include_directories(${_image_obj} PRIVATE
                             ${MODEL_OTA_ROOT}/src ${_header_dir}
                             ${EDGE_AI_MODULE_ROOT}/include)
  target_compile_options(${_image_obj} PRIVATE "SHELL:-include \"${_private_h}\"")
  target_compile_definitions(${_image_obj} PRIVATE
    NRF_MODEL_PARTITION_ADDR=${_partition_addr}
    MODEL_IMAGE_NAME_STR=\"${MI_NAME}\"
    MODEL_IMAGE_VERSION_U32=${_version_u32}u
    NRF_AXON_INTERLAYER_BUFFER_SIZE=${CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE})
  set_source_files_properties(
    ${MODEL_OTA_AXON_IMAGE_STUB}
    TARGET_DIRECTORY ${_image_obj}
    PROPERTIES OBJECT_DEPENDS "${MI_HEADER};${_private_h}")
  add_dependencies(${_image_obj} ${_meta_target} zephyr_generated_headers)

  set(_model_syms_ld ${_work_dir}/${MI_TARGET}_model_syms.ld)
  set(_image_elf ${_work_dir}/${MI_TARGET}_model_image.elf)
  set(_image_raw ${_work_dir}/${MI_TARGET}_model_image_raw.bin)
  set(_image_bin ${CMAKE_CURRENT_BINARY_DIR}/${MI_TARGET}_model_image.bin)
  set(_image_hex ${CMAKE_CURRENT_BINARY_DIR}/${MI_TARGET}_model_partition.hex)
  set(_zephyr_elf ${CMAKE_CURRENT_BINARY_DIR}/zephyr/zephyr.elf)

  add_custom_command(
    OUTPUT ${_model_syms_ld}
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_AXON_ELF} provide
            --object $<TARGET_OBJECTS:${_image_obj}>
            --elf ${_zephyr_elf}
            -o ${_model_syms_ld}
    DEPENDS ${MODEL_OTA_AXON_ELF} ${_zephyr_elf} $<TARGET_OBJECTS:${_image_obj}>
    COMMAND_EXPAND_LISTS
    COMMENT "Resolving Axon app symbols from zephyr.elf (${MI_TARGET})"
    VERBATIM)

  add_custom_command(
    OUTPUT ${_image_bin} ${_image_hex}
    COMMAND ${CMAKE_C_COMPILER}
            -nostdlib -nostartfiles
            -Wl,--gc-sections
            -Wl,--defsym=NRF_MODEL_PARTITION_ADDR=${_partition_addr}
            -T ${MODEL_OTA_AXON_LINKER_SCRIPT}
            -T ${_model_syms_ld}
            -o ${_image_elf}
            $<TARGET_OBJECTS:${_image_obj}>
    COMMAND ${CMAKE_OBJCOPY} -O binary -j .model_image ${_image_elf} ${_image_raw}
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_AXON_CRC_TOOL}
            --bin ${_image_raw} -o ${_image_bin}
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_AXON_VALIDATE_TOOL}
            --elf ${_image_elf} --bin ${_image_bin}
            --partition-addr ${_partition_addr} --partition-size ${_partition_size}
            --defs-header ${MODEL_OTA_IMAGE_DEFS}
            --params-type 3 --config-header ${_private_h}
    COMMAND ${CMAKE_OBJCOPY} -I binary -O ihex
            --change-addresses=${_partition_addr} ${_image_bin} ${_image_hex}
    DEPENDS $<TARGET_OBJECTS:${_image_obj}> ${_model_syms_ld} ${_private_h}
            ${MODEL_OTA_AXON_LINKER_SCRIPT} ${MODEL_OTA_AXON_CRC_TOOL}
            ${MODEL_OTA_AXON_VALIDATE_TOOL}
    COMMAND_EXPAND_LISTS
    COMMENT "Building Axon model partition image '${MI_NAME}' at ${_partition_addr}"
    VERBATIM)

  add_custom_target(${MI_TARGET}_model_image ALL DEPENDS ${_image_bin} ${_image_hex})
endfunction()
