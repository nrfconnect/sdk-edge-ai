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

# Check if platform-specific neural network model file exists
set(NN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/nrf_edgeai_generated/${BOARD}")
set(NN_PATH "${NN_SOURCE}/nrf_edgeai_user_model.c")
if(EXISTS "${NN_PATH}")
    message(STATUS "Using Edge AI model for board: ${BOARD}")
else()
    message(FATAL_ERROR "Edge AI model file not found for board '${BOARD}'.\n"
        "Expected file: ${NN_PATH}\n"
        "Please create ${NN_PATH} or use one of the supported boards."
    )
endif()

target_include_directories(app PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hw_modules
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ble
  ${NN_SOURCE}
)

target_sources(app PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/inference_postprocessing.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ble/hid/ble_hid.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hw_modules/button/button.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hw_modules/led/led.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hw_modules/sensor/imu/imu.c
  ${NN_PATH}
)
