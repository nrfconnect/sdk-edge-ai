# Copyright (c) 2026 Nordic Semiconductor
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(person_det_picture)

# Reuse compiled model + demo JPEG from person_recognition application.
set(PERSON_REC_APP "${CMAKE_CURRENT_SOURCE_DIR}/../person_recognition")
set(COMPILED_MODELS_DIR "${PERSON_REC_APP}/outputs")
set(PERSON_DET_HDR "${COMPILED_MODELS_DIR}/nrf_axon_model_person_det_.h")

if(NOT EXISTS "${PERSON_DET_HDR}")
  message(FATAL_ERROR
    "person_det model header not found:\n  ${PERSON_DET_HDR}\n"
    "Build or copy Axon outputs under person_recognition/outputs/.")
endif()

set(DEMO_JPEG "${CMAKE_CURRENT_SOURCE_DIR}/pictures/demo_2.jpeg")
if(NOT EXISTS "${DEMO_JPEG}")
  message(FATAL_ERROR
    "demo_picture.jpeg not found:\n  ${DEMO_JPEG}")
endif()

set(GEN_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/person_det_demo_input.h")
find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
  COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_demo_input.py"
    --axon-header "${PERSON_DET_HDR}"
    --model-symbol model_person_det
    --jpeg "${DEMO_JPEG}"
    --out "${GEN_HEADER}"
  RESULT_VARIABLE _embed_st
  ERROR_VARIABLE _embed_err
  OUTPUT_VARIABLE _embed_out
)
if(NOT _embed_st EQUAL 0)
  message(FATAL_ERROR
    "embed_demo_input.py failed (install Pillow in build env: pip install Pillow).\n"
    "${_embed_err}${_embed_out}")
endif()

target_include_directories(app PRIVATE
  "${COMPILED_MODELS_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/generated")

zephyr_compile_definitions(NRF_AXON_MODEL_NAME=person_det)

target_sources(app PRIVATE src/main.c src/person_det_postprocess.c)
