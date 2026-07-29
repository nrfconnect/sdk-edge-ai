#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

set(APP_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
include(${APP_DIR}/cmake/ww_kws_model_ota.cmake)

set(_ww_kws_mcuboot_overlay
    ${APP_DIR}/sysbuild/mcuboot/boards/nrf54lm20dk_nrf54lm20b_cpuapp.overlay)

ww_kws_model_ota_enabled(WW_KWS_MODEL_OTA)
if(WW_KWS_MODEL_OTA)
  set(mcuboot_EXTRA_DTC_OVERLAY_FILE
      ${APP_DIR}/sysbuild/mcuboot/boards/nrf54lm20dk_nrf54lm20b_cpuapp_model_ota.overlay
      CACHE INTERNAL "ww_kws model OTA MCUboot overlay" FORCE)

  set(SB_EXTRA_CONF_FILE ${APP_DIR}/sysbuild_model_ota.conf
      CACHE STRING "ww_kws model OTA sysbuild options" FORCE)
else()
  set(mcuboot_DTC_OVERLAY_FILE "${_ww_kws_mcuboot_overlay}"
      CACHE INTERNAL "ww_kws MCUboot overlay" FORCE)
endif()

find_package(Sysbuild REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(sysbuild LANGUAGES)
