#
# Copyright (c) 2026 Nordic Semiconductor
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

if(DEFINED BOARD)
  string(REPLACE "/" "_" _board "${BOARD}")
  set(_mcuboot_overlay "${CMAKE_CURRENT_LIST_DIR}/../boards/mcuboot_${_board}.overlay")

  if(EXISTS "${_mcuboot_overlay}")
    set(mcuboot_EXTRA_DTC_OVERLAY_FILE "${_mcuboot_overlay}"
    CACHE INTERNAL "MCUboot DTS overlay for ${BOARD}" FORCE
    )
  endif()
endif()

find_package(Sysbuild REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(sysbuild LANGUAGES)
