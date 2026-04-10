#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Host Axon simulator executable for mcunet_vww_320kb (see README.md).

cmake_minimum_required(VERSION 3.13)
project(mcunet_vww_axon_sim C)

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(NOT_A_ZEPHYR_BUILD 1)
set(CONFIG_NRF_AXON 1)
# >= NRF_AXON_MODEL_MCUNET_VWW_320KB_MAX_IL_BUFFER_USED from the model header
set(CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE 2000000)
set(CONFIG_NRF_AXON_PSUM_BUFFER_SIZE 400000)

get_filename_component(SDK_EDGE_AI_ROOT "${CMAKE_SOURCE_DIR}/../../.." ABSOLUTE)
set(AXON_SIM_ROOT "${SDK_EDGE_AI_ROOT}/axon_simulator")

add_subdirectory("${AXON_SIM_ROOT}" "${CMAKE_BINARY_DIR}/axon_sim_build")

include_directories("${CMAKE_SOURCE_DIR}/../outputs")

add_executable(mcunet_vww_axon_sim mcunet_vww_sim_main.c)
target_compile_definitions(mcunet_vww_axon_sim PRIVATE NOT_A_ZEPHYR_BUILD=1)

find_package(Threads REQUIRED)
target_link_libraries(mcunet_vww_axon_sim PRIVATE
  nrf-axon-platform
  nrf-axon-driver
  nrf-axon-simulator-internal
  nrf-axon-driver-internal
  Threads::Threads
)
