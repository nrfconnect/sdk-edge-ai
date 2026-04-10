# Copyright (c) 2026 Nordic Semiconductor
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(person_recognition)

list(APPEND DTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/dts)

set(COMPILED_MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/outputs)
set(PERSON_DET_HDR ${COMPILED_MODELS_DIR}/nrf_axon_model_person_det_.h)
if(NOT EXISTS ${PERSON_DET_HDR})
  message(FATAL_ERROR
    "Compiled person_det model not found:\n  ${PERSON_DET_HDR}\n"
    "Build the Axon compiler output for person detection into outputs/ "
    "(see README.md and your person_det compiler YAML).")
endif()

target_include_directories(app PRIVATE ${COMPILED_MODELS_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src
			   ${CMAKE_CURRENT_SOURCE_DIR}/src/drivers)
zephyr_compile_definitions(NRF_AXON_MODEL_NAME=person_det)

target_sources(app PRIVATE src/main.c src/person_det_postprocess.c src/drivers/arducam_mega.c)
