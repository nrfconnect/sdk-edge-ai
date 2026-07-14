#
# Copyright (c) 2025 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

# Zephyr CMake project
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(nrf_edgeai_regression_sample)

target_include_directories(app PRIVATE include)

if(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_AXON)
  target_sources(app PRIVATE src/model_wiring_axon.c)
  # Needed for model_wiring_axon.c's compiled-in fallback (CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA=n)
  # regardless of whether OTA is actually enabled - harmless if unused.
  target_include_directories(app PRIVATE src/nrf_edgeai_generated/Axon)
  message(STATUS "Using Axon Edge AI model")

  if(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/model_ota/cmake/nrf_axon_model_stub.cmake)
    nrf_axon_model_stub(
      TARGET regression
      MODEL_NAME axon_user_instance_36025
      HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/nrf_edgeai_generated/Axon/nrf_edgeai_user_model_axon.h
      PARTITION_NODELABEL model_partition
    )
  endif()
else()
  target_sources(app PRIVATE src/model_wiring_neuton.c)
  message(STATUS "Using Neuton Edge AI model")
endif()

target_sources(app PRIVATE src/main.c)
