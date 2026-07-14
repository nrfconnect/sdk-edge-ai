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

if(CONFIG_NRF_EDGEAI_REGRESSION_REFERENCE_BUILD)
  # Never flashed as the app: main.c's reference-build branch directly includes
  # nrf_edgeai_user_model_axon.h to keep the generated Axon model's symbols (and this build's
  # real addresses for them) alive in zephyr.elf for tools/model_ota/package_model_axon.py
  # to introspect. See README.rst.
  message(STATUS "Building Axon reference image for package_model_axon.py")
  target_include_directories(app PRIVATE src/nrf_edgeai_generated/Axon)
elseif(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_AXON)
  target_sources(app PRIVATE src/model_wiring_axon.c)
  # Needed for model_wiring_axon.c's compiled-in fallback (CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA=n)
  # regardless of whether OTA is actually enabled - harmless if unused.
  target_include_directories(app PRIVATE src/nrf_edgeai_generated/Axon)
  message(STATUS "Using Axon Edge AI model")
else()
  target_sources(app PRIVATE src/model_wiring_neuton.c)
  message(STATUS "Using Neuton Edge AI model")
endif()

target_sources(app PRIVATE src/main.c)
