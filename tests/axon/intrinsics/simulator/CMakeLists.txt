#
# Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
#
# The information contained herein is confidential property of Nordic Semiconductor ASA.
# The use, copying, transfer or disclosure of such information is prohibited except by
# express written agreement with Nordic Semiconductor ASA.
#
cmake_minimum_required(VERSION 3.20.0)

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

project(nrf_axon_app_test_dsp_intrinsics VERSION 0.1.0 LANGUAGES C)

# This is a simulator build.
set(CONFIG_NRF_AXON 1)
set(NOT_A_ZEPHYR_BUILD 1)

set(AXON_ROOT "${CMAKE_SOURCE_DIR}/../../../../axon_simulator")

add_subdirectory(${AXON_ROOT} ..)

set(AXONS_INTRINSICS_BASE_LIBS nrf-axon-nn-compiler-internal nrf-axon-platform nrf-axon-simulator-internal nrf-axon-driver-internal nrf-axon-driver)
if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
    # linux requires explicit link to math lib; windows and zephyr don't.
    set(AXONS_INTRINSICS_PLATFORM_LIBS m)
endif()


add_executable(nrf_axon_app_test_dsp_intrinsics
    ../src/nrf_axon_app_test_dsp_intrinsics.c
)

target_link_libraries(nrf_axon_app_test_dsp_intrinsics
    PRIVATE
        ${AXONS_INTRINSICS_BASE_LIBS}
        ${AXONS_INTRINSICS_PLATFORM_LIBS}
)
target_compile_definitions(nrf_axon_app_test_dsp_intrinsics
    PRIVATE
        NOT_A_ZEPHYR_BUILD=1
        AXON_SIMULATION=1
)
