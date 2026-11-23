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
)

target_sources(app PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/scenarios.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/model_obsv.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ux_state_manager.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ble_mds.c
)
