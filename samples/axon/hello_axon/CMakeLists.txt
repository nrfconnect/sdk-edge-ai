#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

# Zephyr CMake project
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(hello_axon)

# Source files
target_sources(app PRIVATE
    src/main.c
)

if(CONFIG_HELLO_AXON_MODEL_OTA)
  include(${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/model_ota/cmake/nrf_axon_model_stub.cmake)
  nrf_axon_model_stub(
    TARGET hello_axon
    MODEL_NAME hello_axon
    HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/nrf_axon_model_hello_axon_.h
    PARTITION_NODELABEL model_partition
  )
endif()
