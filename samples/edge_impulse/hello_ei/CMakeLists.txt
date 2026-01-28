#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(hello_ei)

# Add application source files
target_sources(app PRIVATE
  src/main.c
)

# Add application include directories
target_include_directories(app PRIVATE
  src/include
)

# Validate that Edge Impulse URI is specified
if(CONFIG_EDGE_IMPULSE_URI STREQUAL "")
  message(FATAL_ERROR "CONFIG_EDGE_IMPULSE_URI must be specified")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/edge_impulse/ei_sdk_utils.cmake)

# Process URI list from configuration (supports URLs and local paths)
process_edge_impulse_uri("${CONFIG_EDGE_IMPULSE_URI}" EI_URI_LIST)

# Get Edge Impulse API key header if specified
get_edge_impulse_api_key(EI_API_KEY_HEADER)

zephyr_compile_definitions(
  EI_C_LINKAGE=1
  EIDSP_SIGNAL_C_FN_POINTER=1
)

# zephyr_compile_options(
#   -Wno-double-promotion
#   -Wno-unused
#   -Wno-sign-compare
#   -Wno-maybe-uninitialized
# )

# Include Edge Impulse library integration
if(CONFIG_EDGE_IMPULSE_COMPILE_IN_TREE)
  message(STATUS "Compiling Edge Impulse library into application binary")
  include(${CMAKE_CURRENT_LIST_DIR}/edge_impulse/ei_compile_in_app.cmake)
else()
  message(STATUS "Building Edge Impulse library as separate static library")
  include(${CMAKE_CURRENT_LIST_DIR}/edge_impulse/ei_standalone_lib.cmake)
endif()
