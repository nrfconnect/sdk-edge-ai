#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(matter_switch)

# Enable GNU STD support.
include(${ZEPHYR_CONNECTEDHOMEIP_MODULE_DIR}/config/nrfconnect/app/enable-gnu-std.cmake)

# NORDIC SDK APP START

# Include all source files that are located in the Matter common directory.
include(${ZEPHYR_NRF_MODULE_DIR}/samples/matter/common/cmake/source_common.cmake)

# Include Data Model utils
include(${ZEPHYR_NRF_MODULE_DIR}/samples/matter/common/cmake/data_model.cmake)

include(${ZEPHYR_NRF_MODULE_DIR}/samples/matter/common/cmake/zap_helpers.cmake)
ncs_get_zap_parent_dir(ZAP_PARENT_DIR)

target_include_directories(app PRIVATE
  src
  src/nrf_edgeai_generated
  src/switch
  src/edgeAI
  ${ZAP_PARENT_DIR}
)

target_compile_options(app PRIVATE -Wno-deprecated-declarations)

target_sources(app PRIVATE
  src/main.cpp
  src/app_task.cpp
  src/nrf_edgeai_task.cpp

  src/switch/switch.cpp

  src/edgeAI/dmic.c
  src/edgeAI/wakeword.c
  src/nrf_edgeai_generated/nrf_edgeai_user_model.c
)

target_sources_ifdef(CONFIG_SWITCH_SHELL app PRIVATE src/switch/shell_commands.cpp)
target_sources_ifdef(CONFIG_GENERIC_SWITCH app PRIVATE src/switch/generic_switch.cpp)
target_sources_ifdef(CONFIG_LIGHT_SWITCH app PRIVATE src/switch/light_switch.cpp)
target_sources_ifdef(CONFIG_LIGHT_SWITCH app PRIVATE ${ZEPHYR_NRF_MODULE_DIR}/samples/matter/common/src/binding/binding_handler.cpp)

ncs_configure_data_model()
