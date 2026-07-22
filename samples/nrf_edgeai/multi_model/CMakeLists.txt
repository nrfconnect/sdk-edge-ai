#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(nrf_edgeai_multi_model_sample)

set(ANOMALY_MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/../anomaly/src/nrf_edgeai_generated)
set(CLASSIF_MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/../classification/src/nrf_edgeai_generated/Neuton)
set(REGRESS_MODEL_DIR ${CMAKE_CURRENT_LIST_DIR}/../regression/src/nrf_edgeai_generated/Neuton)

target_include_directories(app PRIVATE include)

target_sources(app PRIVATE
  ${ANOMALY_MODEL_DIR}/nrf_edgeai_user_model.c
  ${CLASSIF_MODEL_DIR}/nrf_edgeai_user_model.c
  ${REGRESS_MODEL_DIR}/nrf_edgeai_user_model.c
  src/main.c
  src/test_anomaly.c
  src/test_classification.c
  src/test_regression.c
)

target_include_directories(app PRIVATE
  ${ANOMALY_MODEL_DIR}
  ${CLASSIF_MODEL_DIR}
  ${REGRESS_MODEL_DIR}
)
