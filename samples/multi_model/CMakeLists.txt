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
set(WW_MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/../../applications/ww_kws/src/ww/nrf_edgeai_generated)
set(PERSON_DET_DIR ${CMAKE_CURRENT_LIST_DIR}/../../applications/person_detection/src)
set(OKAY_NORDIC_DIR ${CMAKE_CURRENT_LIST_DIR}/../axon/axon_low_power/src)

set(NEUTON_MODEL_SOURCES
  ${ANOMALY_MODEL_DIR}/nrf_edgeai_user_model.c
  ${CLASSIF_MODEL_DIR}/nrf_edgeai_user_model.c
  ${REGRESS_MODEL_DIR}/nrf_edgeai_user_model.c
)

set(PERSON_DET_HEADER ${PERSON_DET_DIR}/generated/nrf_axon_model_person_det_.h)
set(OKAY_NORDIC_HEADER ${OKAY_NORDIC_DIR}/generated/nrf_axon_model_okay_nordic.h)

target_include_directories(app PRIVATE include)

target_sources(app PRIVATE
  src/main.c
  src/test_anomaly.c
  src/test_classification.c
  src/test_regression.c
  src/test_wakeword.c
  src/test_person_det.c
  src/test_okay_nordic.c
  ${PERSON_DET_DIR}/postprocessing.c
  ${WW_MODEL_DIR}/nrf_edgeai_user_model.c
)

# Neuton models are partition-loaded when OTA is enabled; wakeword stays baked in.
if(NOT CONFIG_MODEL_OTA_NEUTON)
  target_sources(app PRIVATE ${NEUTON_MODEL_SOURCES})
endif()

target_include_directories(app PRIVATE
  ${ANOMALY_MODEL_DIR}
  ${CLASSIF_MODEL_DIR}
  ${REGRESS_MODEL_DIR}
  ${WW_MODEL_DIR}
  ${PERSON_DET_DIR}
  ${PERSON_DET_DIR}/generated
  ${OKAY_NORDIC_DIR}
  ${OKAY_NORDIC_DIR}/generated
)

if(CONFIG_MODEL_OTA_NEUTON)
  include(${CMAKE_CURRENT_LIST_DIR}/../../lib/model_ota/cmake/model_ota_neuton.cmake)
  model_ota_neuton_wire(SOLUTION_ID 90360 MODEL_SRC ${ANOMALY_MODEL_DIR}/nrf_edgeai_user_model.c
                        MAX_NEURONS 20)
  model_ota_neuton_wire(SOLUTION_ID 90449 MODEL_SRC ${CLASSIF_MODEL_DIR}/nrf_edgeai_user_model.c
                        MAX_NEURONS 80)
  model_ota_neuton_wire(SOLUTION_ID 90508 MODEL_SRC ${REGRESS_MODEL_DIR}/nrf_edgeai_user_model.c
                        MAX_NEURONS 2)

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

if(CONFIG_MODEL_OTA_AXON)
  include(${CMAKE_CURRENT_LIST_DIR}/../../lib/model_ota/cmake/model_ota_axon.cmake)
  # ALLOCATE_PACKED_OUTPUT: keeps this build path (app-owned storage wired into the
  # linked image's packed_output_buf) covered by every OTA build/flash of this sample.
  model_ota_axon_model(TARGET person_det NAME person_det VERSION 1.0.0
                       PARTITION_NODELABEL model_person_det_storage
                       HEADER ${PERSON_DET_HEADER}
                       ALLOCATE_PACKED_OUTPUT)
  model_ota_axon_model(TARGET okay_nordic NAME okay_nordic VERSION 1.0.0
                       PARTITION_NODELABEL model_okay_nordic_storage
                       HEADER ${OKAY_NORDIC_HEADER}
                       PERSISTENT_VARS_CAP 1160)
endif()
