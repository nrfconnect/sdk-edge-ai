# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(axon_app_test_nn_inference)

target_sources(app PRIVATE 
        src/nrf_axon_app_test_nn_inference_zephyr.c
        src/nrf_axon_app_test_nn_inference.c
    )

# 
# model name is specified through kConfig. However for automated builds the
# environment variable NRF_AXON_MODEL_NAME can override kConfig.
#
if (DEFINED ENV{NRF_AXON_MODEL_NAME})
    message(STATUS "NRF_AXON_MODEL_NAME DEFINED $ENV{NRF_AXON_MODEL_NAME}, OVERRIDES CONFIG_NRF_AXON_MODEL_NAME ${CONFIG_NRF_AXON_MODEL_NAME}")
    zephyr_compile_definitions(
        NRF_AXON_MODEL_NAME=$ENV{NRF_AXON_MODEL_NAME}
    )
else()
    message(STATUS "NRF_AXON_MODEL_NAME NOT DEFINED, USING CONFIG_NRF_AXON_MODEL_NAME ${CONFIG_NRF_AXON_MODEL_NAME}")
    zephyr_compile_definitions(
        NRF_AXON_MODEL_NAME=${CONFIG_NRF_AXON_MODEL_NAME}
    )
endif()
zephyr_include_directories(../compiled_models)

