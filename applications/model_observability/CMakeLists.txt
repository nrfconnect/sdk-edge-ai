#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(model_observability)

target_include_directories(app PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/nrf_edgeai_generated
)

target_sources(app PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/accel.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ux_state_manager.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/nrf_edgeai_generated/nrf_edgeai_user_model.c
)

target_sources_ifdef(CONFIG_MODEL_OBSERVABILITY app PRIVATE src/model_obsv.c)
target_sources_ifdef(CONFIG_APP_BLE_NUS app PRIVATE src/ble/ble_nus.c)
target_sources_ifdef(CONFIG_APP_BLE_MDS app PRIVATE src/ble/ble_mds.c)
