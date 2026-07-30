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
