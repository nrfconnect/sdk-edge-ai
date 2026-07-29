#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

set(APP_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
include(${APP_DIR}/cmake/regression_model_ota.cmake)

regression_model_ota_enabled(REGRESSION_MODEL_OTA)
if(REGRESSION_MODEL_OTA)
  regression_model_ota_board_overlay(_regression_model_ota_overlay)

  set(mcuboot_EXTRA_DTC_OVERLAY_FILE
      ${APP_DIR}/sysbuild/mcuboot/boards/${_regression_model_ota_overlay}
      CACHE INTERNAL "regression model OTA MCUboot overlay" FORCE)

  set(SB_EXTRA_CONF_FILE ${APP_DIR}/sysbuild_model_ota.conf
      CACHE STRING "regression model OTA sysbuild options" FORCE)
endif()

find_package(Sysbuild REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(sysbuild LANGUAGES)
