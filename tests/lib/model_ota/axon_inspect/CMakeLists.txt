#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(test_model_ota_axon_inspect)

include(${ZEPHYR_EDGE_AI_MODULE_DIR}/lib/model_ota/cmake/backends/axon.cmake)

set(MODEL_HEADER
    ${ZEPHYR_EDGE_AI_MODULE_DIR}/samples/axon/axon_low_power/src/generated/nrf_axon_model_okay_nordic.h)
get_filename_component(MODEL_HEADER_DIR ${MODEL_HEADER} DIRECTORY)
get_filename_component(MODEL_HEADER_NAME ${MODEL_HEADER} NAME)
set(WORK_DIR ${CMAKE_CURRENT_BINARY_DIR}/model_ota/okay_nordic)
file(MAKE_DIRECTORY ${WORK_DIR})

model_ota_axon_add_probe(OUT_OBJ PROBE_OBJ WORK_DIR ${WORK_DIR} HEADER ${MODEL_HEADER})
model_ota_axon_inspect(
  TARGET okay_nordic
  PROBE ${PROBE_OBJ}
  HEADER_NAME ${MODEL_HEADER_NAME}
  WORK_DIR ${WORK_DIR}
)

target_sources(app PRIVATE src/main.c)
target_include_directories(app PRIVATE ${WORK_DIR} ${MODEL_HEADER_DIR})
set_source_files_properties(src/main.c PROPERTIES OBJECT_DEPENDS ${okay_nordic_AXON_PRIVATE_HEADER})
add_dependencies(app okay_nordic_axon_metadata)
