#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Shared partition-image link pipeline for all model_ota backends.

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/model_ota_common.cmake)

# Signs a raw model partition image (.bin) as an MCUboot image, emitting
# <TARGET>_model_mcuboot.signed.{bin,hex} in the build directory root.
#
# model_ota_mcuboot_sign(TARGET <prefix> MODEL_IMAGE_BIN <bin> MODEL_IMAGE_TARGET <target>
#                        PARTITION_ADDR <addr> PARTITION_SIZE <size>
#                        [UUID_VID <vid>] [UUID_CID <cid>] [EXCLUDE_FROM_ALL])
function(model_ota_mcuboot_sign)
  cmake_parse_arguments(SIGN "EXCLUDE_FROM_ALL"
    "TARGET;MODEL_IMAGE_BIN;MODEL_IMAGE_TARGET;PARTITION_ADDR;PARTITION_SIZE;UUID_VID;UUID_CID"
    "" ${ARGN})

  if(NOT SIGN_TARGET OR NOT SIGN_MODEL_IMAGE_BIN OR NOT SIGN_MODEL_IMAGE_TARGET
     OR NOT SIGN_PARTITION_ADDR OR NOT SIGN_PARTITION_SIZE)
    message(FATAL_ERROR
            "model_ota_mcuboot_sign requires TARGET, MODEL_IMAGE_BIN, MODEL_IMAGE_TARGET, "
            "PARTITION_ADDR and PARTITION_SIZE")
  endif()

  if(NOT DEFINED IMGTOOL)
    message(FATAL_ERROR
            "model_ota_mcuboot_sign(${SIGN_TARGET}): IMGTOOL not found (MCUboot module missing?)")
  endif()

  set(_keyfile "${CONFIG_MCUBOOT_SIGNATURE_KEY_FILE}")
  string(CONFIGURE "${_keyfile}" _keyfile)
  if(NOT _keyfile OR NOT EXISTS "${_keyfile}")
    message(FATAL_ERROR
            "model_ota_mcuboot_sign(${SIGN_TARGET}): CONFIG_MCUBOOT_SIGNATURE_KEY_FILE "
            "not set or missing (${_keyfile})")
  endif()

  dt_chosen(_flash_node PROPERTY "zephyr,flash")
  dt_prop(_write_block_size PATH "${_flash_node}" PROPERTY "write-block-size")
  if(NOT _write_block_size)
    set(_write_block_size 16)
  endif()

  set(_out_base ${CMAKE_CURRENT_BINARY_DIR}/${SIGN_TARGET}_model_mcuboot)
  set(_imgtool_hash_arg)
  if(CONFIG_MCUBOOT_BOOTLOADER_USES_SHA512)
    set(_imgtool_hash_arg --sha 512)
  endif()

  set(_imgtool_uuid_args)
  if(SIGN_UUID_VID)
    list(APPEND _imgtool_uuid_args --vid "${SIGN_UUID_VID}")
  endif()
  if(SIGN_UUID_CID)
    list(APPEND _imgtool_uuid_args --cid "${SIGN_UUID_CID}")
  endif()

  set(_imgtool_sign ${PYTHON_EXECUTABLE} ${IMGTOOL} sign
    --version ${CONFIG_MCUBOOT_IMGTOOL_SIGN_VERSION}
    --header-size 32
    --pad-header
    --pad
    ${_imgtool_hash_arg}
    --slot-size ${SIGN_PARTITION_SIZE}
    --align ${_write_block_size}
    --rom-fixed ${SIGN_PARTITION_ADDR}
    ${_imgtool_uuid_args}
    -k "${_keyfile}"
    --overwrite-only
  )

  set(_outputs ${_out_base}.signed.bin ${_out_base}.signed.hex)

  add_custom_command(
    OUTPUT ${_outputs}
    COMMAND ${_imgtool_sign} ${SIGN_MODEL_IMAGE_BIN} ${_out_base}.signed.bin
    COMMAND ${_imgtool_sign} --hex-addr ${SIGN_PARTITION_ADDR}
            ${SIGN_MODEL_IMAGE_BIN} ${_out_base}.signed.hex
    DEPENDS ${SIGN_MODEL_IMAGE_BIN} ${SIGN_MODEL_IMAGE_TARGET}
    COMMENT "model_ota: signing ${SIGN_TARGET} model for MCUboot"
  )

  if(SIGN_EXCLUDE_FROM_ALL)
    add_custom_target(${SIGN_TARGET}_model_mcuboot_signed DEPENDS ${_outputs})
  else()
    add_custom_target(${SIGN_TARGET}_model_mcuboot_signed ALL DEPENDS ${_outputs})
  endif()
endfunction()

# model_ota_add_image(TARGET <prefix> OBJ_LIB <obj-target> WORK_DIR <dir>
#                     PARTITION_ADDR <addr> PARTITION_SIZE <size> NAME <str>
#                     [UUID_VID <vid>] [UUID_CID <cid>]
#                     [LINK_SCRIPTS <extra .ld>...]
#                     [VALIDATE_ARGS <arg>...]
#                     [COMPAT_ARGS <arg>...]
#                     [EXCLUDE_FROM_ALL]
#                     [DEPENDS <file>...])
#
# With CONFIG_BOOTLOADER_MCUBOOT, the image is also signed by model_ota_mcuboot_sign().
function(model_ota_add_image)
  cmake_parse_arguments(IMG "EXCLUDE_FROM_ALL"
    "TARGET;OBJ_LIB;WORK_DIR;PARTITION_ADDR;PARTITION_SIZE;NAME;UUID_VID;UUID_CID"
    "LINK_SCRIPTS;VALIDATE_ARGS;COMPAT_ARGS;DEPENDS"
    ${ARGN})

  if(NOT IMG_TARGET OR NOT IMG_OBJ_LIB OR NOT IMG_WORK_DIR OR NOT IMG_PARTITION_ADDR
     OR NOT IMG_PARTITION_SIZE OR NOT IMG_NAME)
    message(FATAL_ERROR
            "model_ota_add_image requires TARGET, OBJ_LIB, WORK_DIR, PARTITION_ADDR, "
            "PARTITION_SIZE and NAME")
  endif()

  model_ota_image_link_addr(${IMG_PARTITION_ADDR} _image_link_addr)
  model_ota_model_image_offset(_image_model_offset)

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
            -Wl,--defsym=MODEL_IMAGE_LINK_ADDR=${_image_link_addr}
            -T ${MODEL_OTA_LINKER_SCRIPT}
            ${_link_scripts}
            -o ${_image_elf}
            $<TARGET_OBJECTS:${IMG_OBJ_LIB}>
    COMMAND ${CMAKE_OBJCOPY} -O binary -j .model_image ${_image_elf} ${_image_raw}
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_CRC_TOOL}
            --bin ${_image_raw} -o ${_image_bin}
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_VALIDATE_TOOL}
            --elf ${_image_elf} --bin ${_image_bin}
            --image-link-addr ${_image_link_addr} --partition-size ${IMG_PARTITION_SIZE}
            --model-image-offset ${_image_model_offset}
            --defs-header ${MODEL_OTA_IMAGE_DEFS}
            ${_validate_extra}
    COMMAND ${CMAKE_OBJCOPY} -I binary -O ihex
            --change-addresses=${_image_link_addr} ${_image_bin} ${_image_hex}
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_COMPAT_TOOL}
            --context ${_compat_context} --image ${_image_bin} --slot ${IMG_TARGET}
            --report-only
            ${_compat_extra}
    DEPENDS $<TARGET_OBJECTS:${IMG_OBJ_LIB}> ${MODEL_OTA_LINKER_SCRIPT}
            ${MODEL_OTA_CRC_TOOL} ${MODEL_OTA_VALIDATE_TOOL} ${_compat_context}
            ${MODEL_OTA_COMPAT_TOOL} ${IMG_DEPENDS}
    COMMAND_EXPAND_LISTS
    COMMENT "Building model partition image '${IMG_NAME}' at ${_image_link_addr}"
    VERBATIM)

  if(IMG_EXCLUDE_FROM_ALL)
    add_custom_target(${IMG_TARGET}_model_image DEPENDS ${_image_bin} ${_image_hex})
  else()
    add_custom_target(${IMG_TARGET}_model_image ALL DEPENDS ${_image_bin} ${_image_hex})
  endif()
  if(TARGET model_ota_context AND NOT _using_released_fw)
    add_dependencies(${IMG_TARGET}_model_image model_ota_context)
  endif()

  if(CONFIG_BOOTLOADER_MCUBOOT)
    set(_sign_extra "")
    if(IMG_UUID_VID)
      list(APPEND _sign_extra UUID_VID ${IMG_UUID_VID})
    endif()
    if(IMG_UUID_CID)
      list(APPEND _sign_extra UUID_CID ${IMG_UUID_CID})
    endif()
    if(IMG_EXCLUDE_FROM_ALL)
      list(APPEND _sign_extra EXCLUDE_FROM_ALL)
    endif()
    model_ota_mcuboot_sign(
      TARGET ${IMG_TARGET}
      MODEL_IMAGE_BIN ${_image_bin}
      MODEL_IMAGE_TARGET ${IMG_TARGET}_model_image
      PARTITION_ADDR ${IMG_PARTITION_ADDR}
      PARTITION_SIZE ${IMG_PARTITION_SIZE}
      ${_sign_extra})
  endif()
endfunction()
