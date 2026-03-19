# Copyright (c) 2026 Nordic Semiconductor
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
find_package(Python3 REQUIRED)
project(person_recognition)

# Use compiled virat_mobilenetv2 model from applications/virat_mobilenetv2/outputs
set(COMPILED_MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../virat_mobilenetv2/outputs)
if(NOT EXISTS ${COMPILED_MODELS_DIR}/nrf_axon_model_virat_mobilenetv2_.h)
  message(FATAL_ERROR
    "virat_mobilenetv2 model not found at ${COMPILED_MODELS_DIR}. "
    "Compile virat_mobilenetv2 first (see applications/virat_mobilenetv2/README.md).")
endif()
target_include_directories(app PRIVATE ${COMPILED_MODELS_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
zephyr_compile_definitions(NRF_AXON_MODEL_NAME=virat_mobilenetv2)

# Generate test_images.h (360x640x3 HWC for virat) from pictures
set(GEN_TEST_IMAGES_H ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/test_images.h)
add_custom_command(
  OUTPUT ${GEN_TEST_IMAGES_H}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/generated
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_test_images.py
    --model virat
    --pictures-dir ${CMAKE_CURRENT_SOURCE_DIR}/pictures
    --out ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/test_images.h
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating test_images.h (virat 360x640) from pictures"
)
add_custom_target(gen_test_images ALL DEPENDS ${GEN_TEST_IMAGES_H})
add_dependencies(app gen_test_images)

target_sources(app PRIVATE src/main.c)
