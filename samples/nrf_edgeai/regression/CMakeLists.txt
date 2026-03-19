#
# Copyright (c) 2025 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

# Zephyr CMake project
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(nrf_edgeai_regression_sample)

# Source files
target_include_directories(app PRIVATE include)

if(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_AXON)
  target_sources(app PRIVATE
    src/nrf_edgeai_generated/Axon/nrf_edgeai_user_model.c)
  message(STATUS "Using Axon Edge AI model")
  target_include_directories(app PRIVATE src/nrf_edgeai_generated/Axon)
else()
  target_sources(app PRIVATE
    src/nrf_edgeai_generated/Neuton/nrf_edgeai_user_model.c)
  message(STATUS "Using Neuton Edge AI model")
  target_include_directories(app PRIVATE src/nrf_edgeai_generated/Neuton)
endif()

target_sources(app PRIVATE src/main.c)
