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
  target_sources(app PRIVATE
    src/nrf_edgeai_generated/Axon/nrf_edgeai_user_model.c
  )
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
  target_sources(app PRIVATE
    src/nrf_edgeai_generated/Neuton/nrf_edgeai_user_model.c
  )
  target_include_directories(app PRIVATE src/nrf_edgeai_generated/Neuton)
  message(STATUS "Using Neuton Edge AI model")

  if(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/model_ota/cmake/nrf_neuton_model_package.cmake)
    nrf_neuton_model_package(
      TARGET regression
      MODEL_NAME aq_regression
      MODEL_C ${CMAKE_CURRENT_SOURCE_DIR}/src/nrf_edgeai_generated/Neuton/nrf_edgeai_user_model.c
      PARTITION_NODELABEL model_partition
    )
  endif()
endif()

if(CONFIG_NRF_EDGEAI_REGRESSION_MCUBOOT)
  include(${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/model_ota/cmake/nrf_model_mcuboot_sign.cmake)
  nrf_model_mcuboot_sign(
    TARGET regression
    PKG_BIN ${CMAKE_CURRENT_BINARY_DIR}/regression_model_pkg.bin
    PKG_TARGET regression_model_pkg
    PARTITION_NODELABEL model_partition
    UUID_CID nRF54LM20_regression_model
    UUID_VID nordicsemi.com
  )
endif()

target_sources(app PRIVATE src/main.c)
