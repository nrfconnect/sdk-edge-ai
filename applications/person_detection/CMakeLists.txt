# Copyright (c) 2026 Nordic Semiconductor
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(person_detection)

list(APPEND DTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/dts)

target_sources(app PRIVATE
  src/main.c
  src/postprocessing.c
  src/drivers/arducam_mega.c
)

if(CONFIG_PERSON_DET_MODEL_IN_PARTITION)
  include(${CMAKE_CURRENT_SOURCE_DIR}/../../lib/axon/model_partition/cmake/nrf_axon_model_partition.cmake)

  nrf_axon_model_partition_image(
    TARGET person_det
    HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/nrf_axon_model_person_det_.h
    PARTITION_NODELABEL axon_model_partition
  )
endif()
