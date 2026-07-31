#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(ww_kws)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../lib/model_ota/cmake/model_ota_context.cmake)

set(WW_MODEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/ww/nrf_edgeai_generated)
set(KWS_MODEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/kws/nrf_edgeai_generated)
set(WW_MODEL_SRC ${WW_MODEL_DIR}/nrf_edgeai_user_model.c)
set(KWS_MODEL_SRC ${KWS_MODEL_DIR}/nrf_edgeai_user_model.c)
set(WW_MODEL_HEADER ${WW_MODEL_DIR}/nrf_edgeai_user_model_axon.h)
set(KWS_MODEL_HEADER ${KWS_MODEL_DIR}/nrf_edgeai_user_model_axon.h)

target_sources(app PRIVATE
  src/main.c

  src/control_output.c
  src/dmic.c
  src/leds.c

  src/ww/wakeword.c

  src/kws/kws.c
)

if(NOT CONFIG_APP_MODEL_OTA)
  target_sources(app PRIVATE
    ${WW_MODEL_SRC}
    ${KWS_MODEL_SRC}
  )
endif()

target_include_directories(app PRIVATE
  ${WW_MODEL_DIR}
  ${KWS_MODEL_DIR}
)

target_sources_ifdef(CONFIG_BT app PRIVATE src/ble/ble_app.c)

if(CONFIG_APP_MODEL_OTA)
  include(${CMAKE_CURRENT_SOURCE_DIR}/../../lib/model_ota/cmake/model_ota_axon_edgeai.cmake)

  model_ota_axon_edgeai_wire(
    SOLUTION_ID 36711
    MODEL_SRC ${WW_MODEL_SRC}
    HEADER ${WW_MODEL_HEADER}
    NAME ww
    VERSION 1.0.0
    PARTITION_NODELABEL model_storage_ww
    PERSISTENT_VARS_CAP 1160
  )
  model_ota_axon_edgeai_wire(
    SOLUTION_ID 36712
    MODEL_SRC ${KWS_MODEL_SRC}
    HEADER ${KWS_MODEL_HEADER}
    NAME kws
    VERSION 1.0.0
    PARTITION_NODELABEL model_storage_kws
    PERSISTENT_VARS_CAP 6648
  )

  include(${CMAKE_CURRENT_SOURCE_DIR}/../../lib/model_ota/cmake/model_ota_mcuboot_sign.cmake)

  model_ota_mcuboot_sign(
    TARGET ww
    MODEL_IMAGE_BIN ${CMAKE_CURRENT_BINARY_DIR}/ww_model_image.bin
    MODEL_IMAGE_TARGET ww_model_image
    PARTITION_NODELABEL model_storage_ww
    UUID_CID ww_kws_model_ww
    UUID_VID nordicsemi.com
  )
  model_ota_mcuboot_sign(
    TARGET kws
    MODEL_IMAGE_BIN ${CMAKE_CURRENT_BINARY_DIR}/kws_model_image.bin
    MODEL_IMAGE_TARGET kws_model_image
    PARTITION_NODELABEL model_storage_kws
    UUID_CID ww_kws_model_kws
    UUID_VID nordicsemi.com
  )

  include(${CMAKE_CURRENT_SOURCE_DIR}/../../lib/model_ota/cmake/model_ota_app_dfu_zip.cmake)
  model_ota_create_dfu_application_zip(
    MODEL ww;kws
    IMAGE_INDEX 1;2
  )
endif()

target_sources_ifdef(CONFIG_APP_MODEL_OTA app PRIVATE src/model_update.c)

if(CONFIG_APP_MODEL_OTA)
  model_ota_context_finalize()
endif()

target_link_libraries(app PRIVATE MCUBOOT_BOOTUTIL)
