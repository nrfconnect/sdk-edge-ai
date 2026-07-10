#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(hello_axon)

target_sources(app PRIVATE src/main.c)

if(CONFIG_HELLO_AXON_MODEL_IN_PARTITION)
  include(${CMAKE_CURRENT_LIST_DIR}/../../../lib/axon/model_partition/cmake/nrf_axon_model_partition.cmake)

  nrf_axon_model_partition_image(
    TARGET hello_axon
    HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/nrf_axon_model_hello_axon_.h
    PARTITION_NODELABEL axon_model_partition
  )
endif()
