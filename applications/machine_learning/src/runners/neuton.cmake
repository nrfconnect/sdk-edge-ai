#
# Copyright (c) 2025 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

set(NEUTON_ARCHIVE  ${APPLICATION_SOURCE_DIR}/models/${CONFIG_ML_APP_ML_RUNNER_NEUTON_ARCHIVE_NAME})
set(NEUTON_DIR      ${CMAKE_BINARY_DIR}/neuton)

if(CONFIG_CPU_CORTEX_M4)
  set(NEUTON_LIBRARY ${NEUTON_DIR}/neuton/lib/libneuton_arm_cortex-m4.a)
elseif(CONFIG_CPU_CORTEX_M33)
  set(NEUTON_LIBRARY ${NEUTON_DIR}/neuton/lib/libneuton_arm_cortex-m33.a)
else()
  message(FATAL_ERROR "CPU not supported now, you can provide Neuton library for specific CPU and modify this CMake file.")
endif()

file(ARCHIVE_EXTRACT
  INPUT       ${NEUTON_ARCHIVE}
  DESTINATION ${NEUTON_DIR}
)

add_library(Neuton INTERFACE)

target_link_libraries(Neuton INTERFACE
  ${NEUTON_LIBRARY}
)

target_include_directories(Neuton INTERFACE
  ${NEUTON_DIR}
  ${NEUTON_DIR}/neuton/include
)

target_link_libraries(app PUBLIC Neuton)
