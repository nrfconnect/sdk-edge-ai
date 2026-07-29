#
# Copyright (c) 2025 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/regression_model_ota.cmake)

regression_model_ota_enabled(REGRESSION_MODEL_OTA)
if(REGRESSION_MODEL_OTA)
  regression_model_ota_board_overlay(_regression_model_ota_overlay)
  list(APPEND EXTRA_DTC_OVERLAY_FILE
    ${CMAKE_CURRENT_SOURCE_DIR}/boards/${_regression_model_ota_overlay})
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(nrf_edgeai_regression_sample)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/model_ota/cmake/model_ota_context.cmake)

set(NEUTON_MODEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/nrf_edgeai_generated/Neuton)
set(AXON_MODEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/nrf_edgeai_generated/Axon)
set(NEUTON_MODEL_SRC ${NEUTON_MODEL_DIR}/nrf_edgeai_user_model.c)
set(AXON_MODEL_SRC ${AXON_MODEL_DIR}/nrf_edgeai_user_model.c)
set(AXON_MODEL_HEADER ${AXON_MODEL_DIR}/nrf_edgeai_user_model_axon.h)

target_include_directories(app PRIVATE include)

if(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_AXON)
  if(NOT CONFIG_APP_MODEL_OTA)
    target_sources(app PRIVATE ${AXON_MODEL_SRC})
  endif()
  message(STATUS "Using Axon Edge AI model")
  target_include_directories(app PRIVATE ${AXON_MODEL_DIR})
else()
  if(NOT CONFIG_APP_MODEL_OTA)
    target_sources(app PRIVATE ${NEUTON_MODEL_SRC})
  endif()
  message(STATUS "Using Neuton Edge AI model")
  target_include_directories(app PRIVATE ${NEUTON_MODEL_DIR})
endif()

target_sources(app PRIVATE src/main.c)

if(CONFIG_APP_MODEL_OTA)
  if(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_AXON)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/model_ota/cmake/model_ota_axon_edgeai.cmake)

    model_ota_axon_edgeai_wire(
      SOLUTION_ID 36025
      MODEL_SRC ${AXON_MODEL_SRC}
      HEADER ${AXON_MODEL_HEADER}
      NAME regression
      VERSION 1.0.0
      PARTITION_NODELABEL model_storage
    )
  else()
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/model_ota/cmake/model_ota_neuton.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/model_ota/cmake/model_ota_neuton_image.cmake)

    model_ota_neuton_wire(
      SOLUTION_ID 90508
      MODEL_SRC ${NEUTON_MODEL_SRC}
      MAX_NEURONS 2
    )
    model_ota_neuton_image(
      TARGET regression
      NAME regression
      VERSION 1.0.0
      PARTITION_NODELABEL model_storage
      NEURONS_CAP 2
      MODEL_SRC ${NEUTON_MODEL_SRC}
    )
  endif()

  include(${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/model_ota/cmake/model_ota_mcuboot_sign.cmake)

  model_ota_mcuboot_sign(
    TARGET regression
    MODEL_IMAGE_BIN ${CMAKE_CURRENT_BINARY_DIR}/regression_model_image.bin
    MODEL_IMAGE_TARGET regression_model_image
    PARTITION_NODELABEL model_storage
    UUID_CID nRF54LM20_regression_model
    UUID_VID nordicsemi.com
  )
endif()

target_sources_ifdef(CONFIG_APP_MODEL_OTA app PRIVATE src/model_update.c)

if(CONFIG_APP_MODEL_OTA)
  model_ota_context_finalize()
  target_link_libraries(app PRIVATE MCUBOOT_BOOTUTIL)
endif()
