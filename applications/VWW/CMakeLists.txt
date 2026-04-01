# Copyright (c) 2026 Nordic Semiconductor
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(person_recognition)

list(APPEND DTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/dts)

set(COMPILED_MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/outputs)
set(MCUNET_HDR ${COMPILED_MODELS_DIR}/nrf_axon_model_mcunet_vww_320kb_.h)
if(NOT EXISTS ${MCUNET_HDR})
  message(FATAL_ERROR
    "Compiled mcunet_vww_320kb model not found:\n  ${MCUNET_HDR}\n"
    "Copy mcunetmcunet-320kb-1mb_vww.tflite to models/, then run the Axon compiler "
    "(see README.md and compiler_mcunet_vww_320kb.yaml).")
endif()

target_include_directories(app PRIVATE ${COMPILED_MODELS_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src
			   ${CMAKE_CURRENT_SOURCE_DIR}/src/drivers)
zephyr_compile_definitions(NRF_AXON_MODEL_NAME=mcunet_vww_320kb)

target_sources(app PRIVATE src/main.c src/drivers/arducam_mega.c)
