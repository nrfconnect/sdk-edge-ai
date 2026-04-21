#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(matter_ambient_sensing)

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
  src/edgeAI
  ${ZAP_PARENT_DIR}
)

target_compile_options(app PRIVATE -Wno-deprecated-declarations)

target_sources(app PRIVATE
  src/main.cpp
  src/app_task.cpp
  src/nrf_edgeai_task.cpp

  src/ambient_sensing.cpp

  src/edgeAI/dmic.c
)

target_sources_ifdef(CONFIG_AMBIENT_SENSING_MODEL_SNORING app PRIVATE src/models/snoring/nrf_edgeai_user_model.c)
target_sources_ifdef(CONFIG_AMBIENT_SENSING_MODEL_DOG_BARKING app PRIVATE src/models/dog_barking/nrf_edgeai_user_model.c)
target_sources_ifdef(CONFIG_AMBIENT_SENSING_MODEL_CAT_MEOWING app PRIVATE src/models/cat_meowing/nrf_edgeai_user_model.c)
target_sources_ifdef(CONFIG_AMBIENT_SENSING_MODEL_BABY_CRYING app PRIVATE src/models/baby_crying/nrf_edgeai_user_model.c)

ncs_configure_data_model()
