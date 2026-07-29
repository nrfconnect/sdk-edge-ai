#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(ww_kws)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../lib/model_ota/cmake/model_ota.cmake)

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

target_sources_ifdef(CONFIG_MODELS_OBSERVABILITY_MDS app PRIVATE src/ble/ble_mds.c)

if(CONFIG_APP_MODEL_OTA)
  model_ota_edgeai_axon_model(
    TARGET ww
    SOLUTION_ID 36711
    MODEL_SRC ${WW_MODEL_SRC}
    HEADER ${WW_MODEL_HEADER}
    NAME ww
    VERSION 1.0.0
    PARTITION_NODELABEL model_storage_ww
    PERSISTENT_VARS_CAP 1160
  )
  model_ota_edgeai_axon_model(
    TARGET kws
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
    MODEL_IMAGE_BIN ${CMAKE_CURRENT_BINARY_DIR}/model_ota/ww/ww_model_image.bin
    MODEL_IMAGE_TARGET ww_model_image
    PARTITION_NODELABEL model_storage_ww
    UUID_CID ww_kws_model_ww
    UUID_VID nordicsemi.com
  )
  model_ota_mcuboot_sign(
    TARGET kws
    MODEL_IMAGE_BIN ${CMAKE_CURRENT_BINARY_DIR}/model_ota/kws/kws_model_image.bin
    MODEL_IMAGE_TARGET kws_model_image
    PARTITION_NODELABEL model_storage_kws
    UUID_CID ww_kws_model_kws
    UUID_VID nordicsemi.com
  )

  target_sources(app PRIVATE src/model_update.c)

  model_ota_finalize()
  target_link_libraries(app PRIVATE MCUBOOT_BOOTUTIL)
endif()
