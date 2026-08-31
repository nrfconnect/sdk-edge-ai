#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Shared partition-image link pipeline for all model_ota backends.

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/model_ota_common.cmake)

# model_ota_add_image(TARGET <prefix> OBJ_LIB <obj-target> WORK_DIR <dir>
#                     PARTITION_ADDR <addr> PARTITION_SIZE <size> NAME <str>
#                     [LINK_SCRIPTS <extra .ld>...]
#                     [VALIDATE_ARGS <arg>...]
#                     [COMPAT_ARGS <arg>...]
#                     [EXCLUDE_FROM_ALL]
#                     [DEPENDS <file>...])
function(model_ota_add_image)
  cmake_parse_arguments(IMG "EXCLUDE_FROM_ALL"
    "TARGET;OBJ_LIB;WORK_DIR;PARTITION_ADDR;PARTITION_SIZE;NAME" "LINK_SCRIPTS;VALIDATE_ARGS;COMPAT_ARGS;DEPENDS"
    ${ARGN})

  if(NOT IMG_TARGET OR NOT IMG_OBJ_LIB OR NOT IMG_WORK_DIR OR NOT IMG_PARTITION_ADDR
     OR NOT IMG_PARTITION_SIZE OR NOT IMG_NAME)
    message(FATAL_ERROR
            "model_ota_add_image requires TARGET, OBJ_LIB, WORK_DIR, PARTITION_ADDR, "
            "PARTITION_SIZE and NAME")
  endif()

  model_ota_using_released_fw(_using_released_fw)

  set(_image_elf ${IMG_WORK_DIR}/${IMG_TARGET}_model_image.elf)
  set(_image_raw ${IMG_WORK_DIR}/${IMG_TARGET}_model_image_raw.bin)
  set(_image_bin ${IMG_WORK_DIR}/${IMG_TARGET}_model_image.bin)
  set(_image_hex ${CMAKE_CURRENT_BINARY_DIR}/${IMG_TARGET}_model_partition.hex)
  set(_generated_context ${CMAKE_CURRENT_BINARY_DIR}/model_ota_context.json)

  if(MODEL_OTA_FW_CONTEXT)
    set(_compat_context ${MODEL_OTA_FW_CONTEXT})
  else()
    set(_compat_context ${_generated_context})
  endif()

  set(_link_scripts "")
  if(IMG_LINK_SCRIPTS)
    foreach(_script ${IMG_LINK_SCRIPTS})
      list(APPEND _link_scripts -T ${_script})
    endforeach()
  endif()

  set(_validate_extra "")
  if(IMG_VALIDATE_ARGS)
    set(_validate_extra ${IMG_VALIDATE_ARGS})
  endif()

  set(_compat_extra "")
  if(IMG_COMPAT_ARGS)
    set(_compat_extra ${IMG_COMPAT_ARGS})
  endif()

  add_custom_command(
    OUTPUT ${_image_bin} ${_image_hex}
    COMMAND ${CMAKE_C_COMPILER}
            -nostdlib -nostartfiles
            -Wl,--gc-sections
            -Wl,--defsym=NRF_MODEL_PARTITION_ADDR=${IMG_PARTITION_ADDR}
            -T ${MODEL_OTA_LINKER_SCRIPT}
            ${_link_scripts}
            -o ${_image_elf}
            $<TARGET_OBJECTS:${IMG_OBJ_LIB}>
    COMMAND ${CMAKE_OBJCOPY} -O binary -j .model_image ${_image_elf} ${_image_raw}
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_CRC_TOOL}
            --bin ${_image_raw} -o ${_image_bin}
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_VALIDATE_TOOL}
            --elf ${_image_elf} --bin ${_image_bin}
            --partition-addr ${IMG_PARTITION_ADDR} --partition-size ${IMG_PARTITION_SIZE}
            --defs-header ${MODEL_OTA_IMAGE_DEFS}
            ${_validate_extra}
    COMMAND ${CMAKE_OBJCOPY} -I binary -O ihex
            --change-addresses=${IMG_PARTITION_ADDR} ${_image_bin} ${_image_hex}
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_COMPAT_TOOL}
            --context ${_compat_context} --image ${_image_bin} --slot ${IMG_TARGET}
            --report-only
            ${_compat_extra}
    DEPENDS $<TARGET_OBJECTS:${IMG_OBJ_LIB}> ${MODEL_OTA_LINKER_SCRIPT}
            ${MODEL_OTA_CRC_TOOL} ${MODEL_OTA_VALIDATE_TOOL} ${_compat_context}
            ${MODEL_OTA_COMPAT_TOOL} ${IMG_DEPENDS}
    COMMAND_EXPAND_LISTS
    COMMENT "Building model partition image '${IMG_NAME}' at ${IMG_PARTITION_ADDR}"
    VERBATIM)

  if(IMG_EXCLUDE_FROM_ALL)
    add_custom_target(${IMG_TARGET}_model_image DEPENDS ${_image_bin} ${_image_hex})
  else()
    add_custom_target(${IMG_TARGET}_model_image ALL DEPENDS ${_image_bin} ${_image_hex})
  endif()
  if(TARGET model_ota_context AND NOT _using_released_fw)
    add_dependencies(${IMG_TARGET}_model_image model_ota_context)
  endif()
endfunction()
