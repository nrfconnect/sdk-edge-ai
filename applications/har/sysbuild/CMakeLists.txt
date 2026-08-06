#
# Copyright (c) 2026 Nordic Semiconductor
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

set(SB_APPLICATION_CONFIG_DIR "${CMAKE_CURRENT_LIST_DIR}/../configuration/\${NORMALIZED_BOARD_TARGET}")

if(DEFINED BOARD)
  get_filename_component(APP_CONFIG_BASE "${CMAKE_CURRENT_LIST_DIR}/../configuration" ABSOLUTE)
  string(REPLACE "/" "_" _mcuboot_overlay_board "${BOARD}")

  if(FILE_SUFFIX)
    set(mcuboot_OVERLAY_PATH
        "${APP_CONFIG_BASE}/${_mcuboot_overlay_board}/images/mcuboot/prj_${FILE_SUFFIX}.conf")
    if(EXISTS "${mcuboot_OVERLAY_PATH}")
      set(mcuboot_EXTRA_CONF_FILE "${mcuboot_OVERLAY_PATH}"
      CACHE INTERNAL "MCUboot overlay for FILE_SUFFIX=${FILE_SUFFIX}" FORCE
      )
    else()
      set(mcuboot_EXTRA_CONF_FILE ""
      CACHE INTERNAL "MCUboot overlay for FILE_SUFFIX=${FILE_SUFFIX}" FORCE
      )
    endif()
  endif()

  set(mcuboot_DTS_OVERLAY_PATH
      "${APP_CONFIG_BASE}/${_mcuboot_overlay_board}/images/mcuboot/${_mcuboot_overlay_board}.overlay")
  if(EXISTS "${mcuboot_DTS_OVERLAY_PATH}")
    set(mcuboot_EXTRA_DTC_OVERLAY_FILE "${mcuboot_DTS_OVERLAY_PATH}"
    CACHE INTERNAL "MCUboot DTS overlay for ${BOARD}" FORCE
    )
  endif()
endif()

find_package(Sysbuild REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(sysbuild LANGUAGES)
