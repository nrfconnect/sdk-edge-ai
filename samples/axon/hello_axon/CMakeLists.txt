#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

# Zephyr CMake project
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(hello_axon)

# Source files
target_sources(app PRIVATE
    src/main.c
)

if(CONFIG_HELLO_AXON_REFERENCE_BUILD)
  target_sources(app PRIVATE src/model_reference_anchor.c)
  # model_hello_axon is a plain (non-static) global; cmd_buffer_hello_axon and
  # axon_model_const_hello_axon are only reachable from it via relocations (see
  # model_reference_anchor.c), so forcing just this one symbol undefined is enough to stop
  # --gc-sections from dropping any of the three from this reference-only image.
  toolchain_ld_force_undefined_symbols(model_hello_axon)
endif()
