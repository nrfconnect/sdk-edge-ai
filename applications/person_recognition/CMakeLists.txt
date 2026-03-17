# Copyright (c) 2026 Nordic Semiconductor
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
find_package(Python3 REQUIRED)
project(person_recognition)

# Use pre-compiled tinyml_vww model and test vectors from tests/axon/compiled_models
set(COMPILED_MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/axon/compiled_models)
if(NOT EXISTS ${COMPILED_MODELS_DIR}/nrf_axon_model_tinyml_vww_.h)
  message(FATAL_ERROR
    "tinyml_vww model not found at ${COMPILED_MODELS_DIR}. "
    "Ensure sdk-edge-ai/tests/axon/compiled_models contains nrf_axon_model_tinyml_vww_.h and nrf_axon_model_tinyml_vww_test_vectors_.h")
endif()
target_include_directories(app PRIVATE ${COMPILED_MODELS_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
zephyr_compile_definitions(NRF_AXON_MODEL_NAME=tinyml_vww)

# Generate test_images.h from pictures/demo_picture.jpeg, demo_2.jpeg, demo_3.jpeg
set(GEN_TEST_IMAGES_H ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/test_images.h)
add_custom_command(
  OUTPUT ${GEN_TEST_IMAGES_H}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/generated
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_test_images.py
    --pictures-dir ${CMAKE_CURRENT_SOURCE_DIR}/pictures
    --out ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/test_images.h
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating test_images.h from pictures"
)
add_custom_target(gen_test_images ALL DEPENDS ${GEN_TEST_IMAGES_H})
add_dependencies(app gen_test_images)

target_sources(app PRIVATE src/main.c)
