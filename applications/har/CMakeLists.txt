#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(har)

target_include_directories(app PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hw_modules
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ble
  ${CMAKE_CURRENT_SOURCE_DIR}/src/nrf_edgeai_generated
)

target_sources(app PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/inference_postprocessing.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/activity_led.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hw_modules/button/button.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hw_modules/led/led.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hw_modules/sensor/imu/imu.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ble/ble_common.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/nrf_edgeai_generated/nrf_edgeai_user_model.c
)

if(CONFIG_BLE_NUS_OUTPUT)
  target_sources(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/ble/nus/ble_nus.c)
endif()
