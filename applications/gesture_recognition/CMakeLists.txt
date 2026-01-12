# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

# The application uses the configuration/<board> scheme for configuration files.
set(APPLICATION_CONFIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/configuration/\${NORMALIZED_BOARD_TARGET}")

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(nrf_edgeai_thingy53)

# Check if platform-specific neural network model file exists
set(NN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/nrf_edgeai_lib/${BOARD}")
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
  ${CMAKE_CURRENT_SOURCE_DIR}/src/bsp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ble
  ${NN_SOURCE}
)

target_sources(app PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/inference_postprocessing.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ble/hid/ble_hid.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/bsp/button/bsp_button.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/bsp/led/bsp_led.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/bsp/sensor/imu/bsp_imu.c
  ${NN_PATH}
)
