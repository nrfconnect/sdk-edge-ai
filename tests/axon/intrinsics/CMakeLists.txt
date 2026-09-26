# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(axon_intrinsics VERSION 0.1.0 LANGUAGES C)

target_sources(app PRIVATE
    src/nrf_axon_app_test_dsp_intrinsics_zephyr.c
    src/nrf_axon_app_test_dsp_intrinsics.c
)
