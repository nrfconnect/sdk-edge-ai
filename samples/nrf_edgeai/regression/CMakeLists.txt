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

# Each backend's nrf_edgeai_generated/<Backend>/nrf_edgeai_user_model.c is both the model itself
# (what an actual Edge AI Lab export would be, plus a mutable runtime instance so it works
# compiled-in or OTA-loaded without duplicating anything) and, for the CONFIG_MODEL_OTA_AXON /
# CONFIG_MODEL_OTA_NEUTON builds, home to the model_storage-loading logic itself
# (nrf_edgeai_load_user_model_36025() / _90508(), declared alongside the rest of that backend's
# generated API in nrf_edgeai_user_model.h; only main.c's nrf_edgeai_user_model() macro calls
# into it, so it stays backend-agnostic). See
# tools/model_ota/examples/hypothetical_ota_generator_output/ for the design this mirrors, and
# doc/libraries/model_ota.rst for the bigger picture.
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

target_sources(app PRIVATE src/main.c)
