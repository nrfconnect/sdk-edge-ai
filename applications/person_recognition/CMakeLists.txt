# Copyright (c) 2026 Nordic Semiconductor
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
find_package(Python3 REQUIRED)
project(person_recognition)

# OFF = embed demo_picture.jpeg only (less flash). ON = all three demos.
# option(PERSON_RECOGNITION_EMBED_ALL_PICTURES "Embed demo_2 and demo_3 as well as demo_picture" OFF)

set(COMPILED_MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/outputs)
set(MCUNET_HDR ${COMPILED_MODELS_DIR}/nrf_axon_model_mcunet_vww_320kb_.h)
if(NOT EXISTS ${MCUNET_HDR})
  message(FATAL_ERROR
    "Compiled mcunet_vww_320kb model not found:\n  ${MCUNET_HDR}\n"
    "Copy mcunetmcunet-320kb-1mb_vww.tflite to models/, then run the Axon compiler "
    "(see README.md and compiler_mcunet_vww_320kb.yaml).")
endif()

target_include_directories(app PRIVATE ${COMPILED_MODELS_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
zephyr_compile_definitions(NRF_AXON_MODEL_NAME=mcunet_vww_320kb)

# if(PERSON_RECOGNITION_EMBED_ALL_PICTURES)
  set(PERSON_REC_EMBED_IMAGES
    --images demo_picture.jpeg demo_2.jpeg demo_3.jpeg)
# else()
#   set(PERSON_REC_EMBED_IMAGES --images demo_picture.jpeg)
# endif()

set(GEN_TEST_IMAGES_H ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/test_images.h)
add_custom_command(
  OUTPUT ${GEN_TEST_IMAGES_H}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/generated
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_test_images.py
    --axon-header ${MCUNET_HDR}
    --model-symbol model_mcunet_vww_320kb
    --preprocess symmetric_m1_1
    --pictures-dir ${CMAKE_CURRENT_SOURCE_DIR}/pictures
    ${PERSON_REC_EMBED_IMAGES}
    --out ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/test_images.h
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MCUNET_HDR}
  COMMENT "Generating test_images.h from Axon header + pictures"
)
add_custom_target(gen_test_images ALL DEPENDS ${GEN_TEST_IMAGES_H})
add_dependencies(app gen_test_images)

target_sources(app PRIVATE src/main.c)
