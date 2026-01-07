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

# Add Edge Impulse library as a separate module
add_subdirectory(edge_impulse)

# Link Edge Impulse library to application
target_link_libraries(app PRIVATE edge_impulse)
