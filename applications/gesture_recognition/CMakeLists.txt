#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

# The application uses the configuration/<board> scheme for configuration files.
set(APPLICATION_CONFIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/configuration/\${NORMALIZED_BOARD_TARGET}")

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(gesture_recognition)

set(NN_BASE "${CMAKE_CURRENT_SOURCE_DIR}/src/nrf_edgeai_generated/${BOARD}")

if(CONFIG_NRF_EDGEAI_GESTURE_RECOGNITION_MODEL_AXON)
  set(NN_SOURCE "${NN_BASE}/Axon")
else()
  # Neuton: use Neuton subdir only for nrf54lm20dk (has both Axon and Neuton)
  if(BOARD STREQUAL "nrf54lm20dk")
    set(NN_SOURCE "${NN_BASE}/Neuton")
  else()
    set(NN_SOURCE "${NN_BASE}")
  endif()
endif()

set(NN_PATH "${NN_SOURCE}/nrf_edgeai_user_model.c")
if(NOT EXISTS "${NN_PATH}")
  message(FATAL_ERROR "Edge AI model file not found for board '${BOARD}' (model dir: ${BOARD_BASE}).\n"
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
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hw_modules/button/button.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hw_modules/led/led.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hw_modules/sensor/imu/imu.c
  ${NN_PATH}
)

if(CONFIG_BLE_MODE_HID)
  target_sources(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/ble/hid/ble_hid.c)
elseif(CONFIG_BLE_MODE_NUS)
  target_sources(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/ble/nus/ble_nus.c)
elseif(CONFIG_BLE_MODE_GATT_CUSTOM)
  target_sources(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/ble/gatt/ble_gatt.c)
endif()
