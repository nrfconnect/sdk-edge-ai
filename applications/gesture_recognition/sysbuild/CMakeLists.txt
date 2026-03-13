#
# Copyright (c) 2026 Nordic Semiconductor
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

# The application uses the configuration/<board> scheme for configuration files.
set(SB_APPLICATION_CONFIG_DIR "${CMAKE_CURRENT_LIST_DIR}/../configuration/\${NORMALIZED_BOARD_TARGET}")

# Use default mcuboot config; when FILE_SUFFIX is set (e.g. release, release_no_ble),
# add board-specific overlay from configuration/<board>/images/mcuboot if present.
# This avoids duplicating full mcuboot prj.conf per board.
if(FILE_SUFFIX AND DEFINED BOARD)
  get_filename_component(APP_CONFIG_BASE "${CMAKE_CURRENT_LIST_DIR}/../configuration" ABSOLUTE)
  # NORMALIZED_BOARD_TARGET is only set in Zephyr's boards.cmake (per-image); in sysbuild we
  # have BOARD from -DBOARD=, so normalize it here for the overlay path check.
  string(REPLACE "/" "_" _mcuboot_overlay_board "${BOARD}")
  set(mcuboot_OVERLAY_PATH
      "${APP_CONFIG_BASE}/${_mcuboot_overlay_board}/images/mcuboot/prj_${FILE_SUFFIX}.conf")
  if(EXISTS "${mcuboot_OVERLAY_PATH}")
    set(mcuboot_EXTRA_CONF_FILE "${mcuboot_OVERLAY_PATH}"
    CACHE INTERNAL "MCUboot overlay for FILE_SUFFIX=${FILE_SUFFIX}" FORCE
    )
  else()
    # No MCUboot overlay present for this FILE_SUFFIX; leave extra config empty.
    set(mcuboot_EXTRA_CONF_FILE ""
    CACHE INTERNAL "MCUboot overlay for FILE_SUFFIX=${FILE_SUFFIX}" FORCE
    )
  endif()
endif()

find_package(Sysbuild REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(sysbuild LANGUAGES)
