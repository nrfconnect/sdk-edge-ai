#
# Copyright (c) 2025 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

# Zephyr CMake project
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(nrf_edgeai_classification_sample)

# Source files
target_include_directories(app PRIVATE include)
target_sources(app PRIVATE
    src/main.c
    src/nrf_edgeai_generated/nrf_edgeai_user_model.c
)

# See doc/libraries/model_ota.rst for the CONFIG_MODEL_OTA_NEUTON path.
if(CONFIG_NRF_EDGEAI_ANOMALY_MODEL_OTA)
  include(${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/model_ota/cmake/nrf_neuton_model_package.cmake)
  nrf_neuton_model_package(
    TARGET anomaly
    MODEL_NAME gear_anomaly
    MODEL_C ${CMAKE_CURRENT_SOURCE_DIR}/src/nrf_edgeai_generated/nrf_edgeai_user_model.c
    PARTITION_NODELABEL model_partition
  )
endif()
