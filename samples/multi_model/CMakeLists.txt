#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(nrf_edgeai_multi_model_sample)

include(${CMAKE_CURRENT_LIST_DIR}/../../lib/model_ota/cmake/model_ota.cmake)

set(ANOMALY_MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/../nrf_edgeai/anomaly/src/nrf_edgeai_generated)
set(CLASSIF_MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/../nrf_edgeai/classification/src/nrf_edgeai_generated/Neuton)
set(CLASSIF_AXON_MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/../nrf_edgeai/classification/src/nrf_edgeai_generated/Axon)
set(REGRESS_MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/../nrf_edgeai/regression/src/nrf_edgeai_generated/Neuton)
set(REGRESS_AXON_MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/../nrf_edgeai/regression/src/nrf_edgeai_generated/Axon)
set(CLASSIF_AXON_MODEL_SRC ${CLASSIF_AXON_MODEL_DIR}/nrf_edgeai_user_model.c)
set(CLASSIF_AXON_MODEL_HEADER ${CLASSIF_AXON_MODEL_DIR}/nrf_edgeai_user_model_axon.h)
set(REGRESS_AXON_MODEL_SRC ${REGRESS_AXON_MODEL_DIR}/nrf_edgeai_user_model.c)
set(REGRESS_AXON_MODEL_HEADER ${REGRESS_AXON_MODEL_DIR}/nrf_edgeai_user_model_axon.h)
set(WW_MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/../../applications/ww_kws/src/ww/nrf_edgeai_generated)
set(WW_MODEL_SRC ${WW_MODEL_DIR}/nrf_edgeai_user_model.c)
set(WW_MODEL_HEADER ${WW_MODEL_DIR}/nrf_edgeai_user_model_axon.h)
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
)

if(CONFIG_MULTI_MODEL_AXON)
  target_sources(app PRIVATE
    src/test_classification_axon.c
    src/test_regression_axon.c
    src/test_wakeword.c
    src/test_person_det.c
    src/test_okay_nordic.c
    ${PERSON_DET_DIR}/postprocessing.c
  )
endif()

# Neuton models are partition-loaded when CONFIG_MODEL_OTA_NEUTON is enabled; compiled directly
# into the app otherwise.
if(NOT CONFIG_MODEL_OTA_NEUTON)
  target_sources(app PRIVATE ${NEUTON_MODEL_SOURCES})
endif()

# Edge AI Lab solutions with an Axon backend are partition-loaded when Axon OTA is enabled.
if(CONFIG_MULTI_MODEL_AXON AND NOT CONFIG_MODEL_OTA_AXON)
  target_sources(app PRIVATE
    ${WW_MODEL_SRC}
    ${CLASSIF_AXON_MODEL_SRC}
    ${REGRESS_AXON_MODEL_SRC}
  )
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
  model_ota_edgeai_neuton_model(TARGET gear_anomaly SOLUTION_ID 90360 VERSION 1.0.0
                                PARTITION_NODELABEL model_anomaly_storage NEURONS_CAP 20
                                MODEL_SRC ${ANOMALY_MODEL_DIR}/nrf_edgeai_user_model.c)
  model_ota_edgeai_neuton_model(TARGET gesture_class SOLUTION_ID 90449 VERSION 1.0.0
                                PARTITION_NODELABEL model_classif_storage NEURONS_CAP 80
                                MODEL_SRC ${CLASSIF_MODEL_DIR}/nrf_edgeai_user_model.c)
  model_ota_edgeai_neuton_model(TARGET temp_regress SOLUTION_ID 90508 VERSION 1.0.0
                                PARTITION_NODELABEL model_regress_storage NEURONS_CAP 2
                                MODEL_SRC ${REGRESS_MODEL_DIR}/nrf_edgeai_user_model.c)
endif()

if(CONFIG_MULTI_MODEL_AXON AND CONFIG_MODEL_OTA_AXON)
  model_ota_edgeai_axon_model(TARGET wakeword SOLUTION_ID 36711 VERSION 1.0.0
                              PARTITION_NODELABEL model_wakeword_storage
                              MODEL_SRC ${WW_MODEL_SRC} HEADER ${WW_MODEL_HEADER}
                              PERSISTENT_VARS_CAP 1160)
  model_ota_edgeai_axon_model(TARGET classif_axon SOLUTION_ID 36237 VERSION 1.0.0
                              PARTITION_NODELABEL model_classif_axon_storage
                              MODEL_SRC ${CLASSIF_AXON_MODEL_SRC}
                              HEADER ${CLASSIF_AXON_MODEL_HEADER})
  model_ota_edgeai_axon_model(TARGET regress_axon SOLUTION_ID 36025 VERSION 1.0.0
                              PARTITION_NODELABEL model_regress_axon_storage
                              MODEL_SRC ${REGRESS_AXON_MODEL_SRC}
                              HEADER ${REGRESS_AXON_MODEL_HEADER})

  # ALLOCATE_PACKED_OUTPUT: keeps this build path (app-owned storage wired into the
  # linked image's packed_output_buf) covered by every OTA build/flash of this sample.
  model_ota_axon_model(TARGET person_det VERSION 1.0.0
                       PARTITION_NODELABEL model_person_det_storage
                       HEADER ${PERSON_DET_HEADER}
                       ALLOCATE_PACKED_OUTPUT)
  model_ota_axon_model(TARGET okay_nordic VERSION 1.0.0
                       PARTITION_NODELABEL model_okay_nordic_storage
                       HEADER ${OKAY_NORDIC_HEADER}
                       PERSISTENT_VARS_CAP 1160)
endif()

if(CONFIG_MODEL_OTA)
  model_ota_context_finalize()
endif()
