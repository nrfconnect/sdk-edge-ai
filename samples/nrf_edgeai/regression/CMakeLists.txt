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

  if(CONFIG_NRF_EDGEAI_REGRESSION_REFERENCE_BUILD)
    # Never flashed as the app: see model_reference_anchor.c for why this file (and the
    # force-undefined symbol below) exist.
    message(STATUS "Also building the Axon reference model for package_model_axon.py")
    target_sources(app PRIVATE src/model_reference_anchor.c)
    # model_axon_user_instance_36025 is a plain (non-static) global; cmd_buffer_.../
    # axon_model_const_... are only reachable from it via relocations (see
    # model_reference_anchor.c), so forcing just this one symbol undefined is enough to stop
    # --gc-sections from dropping any of the three from this reference-only image.
    toolchain_ld_force_undefined_symbols(model_axon_user_instance_36025)
  endif()
else()
  target_sources(app PRIVATE src/model_wiring_neuton.c)
  message(STATUS "Using Neuton Edge AI model")
endif()

target_sources(app PRIVATE src/main.c)
