#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

set(APPLICATION_CONFIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/configuration/\${NORMALIZED_BOARD_TARGET}")

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(har)

set(NN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/nrf_edgeai_generated/${BOARD}")
set(NN_PATH "${NN_SOURCE}/nrf_edgeai_user_model.c")

if(NOT EXISTS "${NN_PATH}")
  message(FATAL_ERROR "Edge AI model file not found for board '${BOARD}'.\n"
    "Expected: ${NN_PATH}\n"
    "Create the model at that path or use a supported board.")
endif()
message(STATUS "Using Edge AI model: ${NN_SOURCE}")

target_include_directories(app PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hw_modules
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ble
  ${NN_SOURCE}
)

target_sources(app PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/inference_postprocessing.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/activity_led.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hw_modules/button/button.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hw_modules/led/led.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hw_modules/sensor/imu/imu.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ble/ble_common.c
  ${NN_PATH}
)

if(CONFIG_BLE_NUS_OUTPUT)
  target_sources(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/ble/nus/ble_nus.c)
endif()
