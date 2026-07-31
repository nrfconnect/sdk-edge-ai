#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Shared helpers for model_ota Neuton/Axon partition-image builds.

include_guard(GLOBAL)

# Offset from partition base to the linked model image payload. With MCUboot, imgtool prepends
# IMAGE_HEADER_SIZE (32) bytes before the raw model image when signing with --pad-header.
set(MODEL_IMAGE_OFFSET_MCUBOOT 32)

function(model_ota_model_image_offset out_var)
  if(CONFIG_BOOTLOADER_MCUBOOT)
    set(${out_var} ${MODEL_IMAGE_OFFSET_MCUBOOT} PARENT_SCOPE)
  else()
    set(${out_var} 0 PARENT_SCOPE)
  endif()
endfunction()

# Link address for a model partition image. When a wrapper sits at the partition base,
# absolute pointers must be linked for partition_base + model_image_offset.
function(model_ota_image_link_addr partition_addr out_var)
  set(link_addr ${partition_addr})
  if(CONFIG_BOOTLOADER_MCUBOOT)
    math(EXPR link_addr "${partition_addr} + ${MODEL_IMAGE_OFFSET_MCUBOOT}")
  endif()
  set(${out_var} ${link_addr} PARENT_SCOPE)
endfunction()

# Parse a Zephyr-style VERSION file (major/minor/patchlevel/tweak/extraversion).
function(model_ota_read_version_file version_file out_tweak_string)
  set(oneValueArgs WITHOUT_TWEAK MAJOR MINOR PATCH TWEAK)
  cmake_parse_arguments(ARG "" "${oneValueArgs}" "" ${ARGN})

  if(NOT EXISTS "${version_file}")
    message(FATAL_ERROR "model_ota_read_version_file: missing ${version_file}")
  endif()

  file(READ "${version_file}" ver)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${version_file}")

  string(REGEX MATCH "VERSION_MAJOR = ([0-9]*)" _ ${ver})
  set(major ${CMAKE_MATCH_1})
  string(REGEX MATCH "VERSION_MINOR = ([0-9]*)" _ ${ver})
  set(minor ${CMAKE_MATCH_1})
  string(REGEX MATCH "PATCHLEVEL = ([0-9]*)" _ ${ver})
  set(patch ${CMAKE_MATCH_1})
  string(REGEX MATCH "VERSION_TWEAK = ([0-9]*)" _ ${ver})
  set(tweak ${CMAKE_MATCH_1})
  string(REGEX MATCH "EXTRAVERSION = ([a-z0-9\\.\\-]*)" _ ${ver})
  set(extra ${CMAKE_MATCH_1})

  set(without_tweak "${major}.${minor}.${patch}")
  if(extra)
    set(without_tweak "${without_tweak}-${extra}")
  endif()
  set(with_tweak "${major}.${minor}.${patch}+${tweak}")

  set(${out_tweak_string} "${with_tweak}" PARENT_SCOPE)
  if(ARG_WITHOUT_TWEAK)
    set(${ARG_WITHOUT_TWEAK} "${without_tweak}" PARENT_SCOPE)
  endif()
  if(ARG_MAJOR)
    set(${ARG_MAJOR} ${major} PARENT_SCOPE)
  endif()
  if(ARG_MINOR)
    set(${ARG_MINOR} ${minor} PARENT_SCOPE)
  endif()
  if(ARG_PATCH)
    set(${ARG_PATCH} ${patch} PARENT_SCOPE)
  endif()
  if(ARG_TWEAK)
    set(${ARG_TWEAK} ${tweak} PARENT_SCOPE)
  endif()
endfunction()

# Pack application version bytes into imgtool's numeric +build field (maj.min.rev+build).
function(model_ota_pack_app_imgtool_build out_var)
  set(oneValueArgs MAJOR MINOR PATCH TWEAK)
  cmake_parse_arguments(ARG "" "${oneValueArgs}" "" ${ARGN})

  if(ARG_MAJOR)
    set(major ${ARG_MAJOR})
    set(minor ${ARG_MINOR})
    set(patch ${ARG_PATCH})
    set(tweak ${ARG_TWEAK})
  else()
    set(major ${APP_VERSION_MAJOR})
    set(minor ${APP_VERSION_MINOR})
    set(patch ${APP_PATCHLEVEL})
    set(tweak ${APP_VERSION_TWEAK})
  endif()

  math(EXPR packed "((${major}) << 24) + ((${minor}) << 16) + ((${patch}) << 8) + (${tweak})")
  set(${out_var} ${packed} PARENT_SCOPE)
endfunction()

# imgtool sign version for a model: model x.y.z with app version encoded in +build.
function(model_ota_mcuboot_sign_version model_major model_minor model_patch app_build out_var)
  set(${out_var} "${model_major}.${model_minor}.${model_patch}+${app_build}" PARENT_SCOPE)
endfunction()

# Pack "x.y.z" into major<<16 | minor<<8 | patch for @ref model_image_header.model_version.
function(model_ota_pack_version version_str out_var)
  string(REPLACE "." ";" ver_parts "${version_str}")
  list(LENGTH ver_parts ver_len)
  list(GET ver_parts 0 ver_major)
  set(ver_minor 0)
  set(ver_patch 0)
  if(ver_len GREATER 1)
    list(GET ver_parts 1 ver_minor)
  endif()
  if(ver_len GREATER 2)
    list(GET ver_parts 2 ver_patch)
  endif()
  math(EXPR ver_u32 "(${ver_major} << 16) | (${ver_minor} << 8) | ${ver_patch}")
  set(${out_var} ${ver_u32} PARENT_SCOPE)
endfunction()
