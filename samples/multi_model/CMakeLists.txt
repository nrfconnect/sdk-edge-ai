#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(nrf_edgeai_multi_model_sample)

set(ANOMALY_MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/../nrf_edgeai/anomaly/src/nrf_edgeai_generated)
set(CLASSIF_MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/../nrf_edgeai/classification/src/nrf_edgeai_generated/Neuton)
set(REGRESS_MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/../nrf_edgeai/regression/src/nrf_edgeai_generated/Neuton)

set(MODEL_SOURCES
  ${ANOMALY_MODEL_DIR}/nrf_edgeai_user_model.c
  ${CLASSIF_MODEL_DIR}/nrf_edgeai_user_model.c
  ${REGRESS_MODEL_DIR}/nrf_edgeai_user_model.c
)

target_include_directories(app PRIVATE include)

target_sources(app PRIVATE
  src/main.c
  src/test_anomaly.c
  src/test_classification.c
  src/test_regression.c
)

# Baked-in models compile directly into the app. OTA-updatable models are linked from dedicated
# static libraries instead (see model_ota_neuton_wire below).
if(NOT CONFIG_MODEL_OTA_NEUTON)
  target_sources(app PRIVATE ${MODEL_SOURCES})
endif()

target_include_directories(app PRIVATE
  ${ANOMALY_MODEL_DIR}
  ${CLASSIF_MODEL_DIR}
  ${REGRESS_MODEL_DIR}
)

if(CONFIG_MODEL_OTA_NEUTON)
  include(${CMAKE_CURRENT_LIST_DIR}/../../lib/model_ota/cmake/model_ota_neuton.cmake)
  model_ota_neuton_wire(SOLUTION_ID 90360 MODEL_SRC ${ANOMALY_MODEL_DIR}/nrf_edgeai_user_model.c
                        MAX_NEURONS 20)
  model_ota_neuton_wire(SOLUTION_ID 90449 MODEL_SRC ${CLASSIF_MODEL_DIR}/nrf_edgeai_user_model.c
                        MAX_NEURONS 80)
  model_ota_neuton_wire(SOLUTION_ID 90508 MODEL_SRC ${REGRESS_MODEL_DIR}/nrf_edgeai_user_model.c
                        MAX_NEURONS 2)

  # Build each model as a self-contained, linked partition IMAGE (Axon-style), placed at its
  # model_storage partition base. The partition header holds a DIRECT pointer to the baked
  # nrf_edgeai_model_neuton_t; the app loads it via model_image_load_neuton(). Partition
  # nodelabels come from boards/*.overlay.
  #
  # Each model image is a SEPARATE, independently-flashable artifact
  # (<name>_model_partition.hex + <name>_model_image.bin under the build dir). They are NOT
  # merged into zephyr.hex: the app and each model partition are flashed/updated on their own.
  include(${CMAKE_CURRENT_LIST_DIR}/../../lib/model_ota/cmake/model_ota_neuton_image.cmake)
  model_ota_neuton_image(TARGET gear_anomaly NAME gear_anomaly VERSION 1.0.0
                         PARTITION_NODELABEL model_anomaly_storage
                         MODEL_SRC ${ANOMALY_MODEL_DIR}/nrf_edgeai_user_model.c)
  model_ota_neuton_image(TARGET gesture_class NAME gesture_class VERSION 1.0.0
                         PARTITION_NODELABEL model_classif_storage
                         MODEL_SRC ${CLASSIF_MODEL_DIR}/nrf_edgeai_user_model.c)
  model_ota_neuton_image(TARGET temp_regress NAME temp_regress VERSION 1.0.0
                         PARTITION_NODELABEL model_regress_storage
                         MODEL_SRC ${REGRESS_MODEL_DIR}/nrf_edgeai_user_model.c)
endif()
