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

target_sources_ifdef(CONFIG_MODELS_OBSERVABILITY_WW app PRIVATE src/ww/ww_obsv.c)
target_sources_ifdef(CONFIG_MODELS_OBSERVABILITY_KWS app PRIVATE src/kws/kws_obsv.c)
target_sources_ifdef(CONFIG_BT_PERIPHERAL app PRIVATE src/ble/ble_mds.c)
target_sources_ifdef(CONFIG_WW_KWS_MEMFAULT_FOTA app PRIVATE src/memfault_fota.c)

if(CONFIG_APP_MODEL_OTA)
  set(MODEL_VERSION "${APP_VERSION_MAJOR}.${APP_VERSION_MINOR}.${APP_PATCHLEVEL}")
  set(WW_MODEL_ITERATION 1 CACHE STRING "Wakeword model iteration")
  set(KWS_MODEL_ITERATION 1 CACHE STRING "KWS model iteration")

  zephyr_get(WW_MODEL_ITERATION)
  zephyr_get(KWS_MODEL_ITERATION)

  foreach(iteration_var IN ITEMS WW_MODEL_ITERATION KWS_MODEL_ITERATION)
    set(iteration "${${iteration_var}}")
    if(NOT iteration MATCHES "^[0-9]+$")
      message(FATAL_ERROR
        "${iteration_var} must be a non-negative integer, got '${iteration}'")
    endif()
    if(iteration GREATER 4294967295)
      message(FATAL_ERROR
        "${iteration_var} exceeds the MCUboot 32-bit build number limit")
    endif()
  endforeach()

  set(WW_MCUBOOT_SIGN_VERSION "${MODEL_VERSION}+${WW_MODEL_ITERATION}")
  set(KWS_MCUBOOT_SIGN_VERSION "${MODEL_VERSION}+${KWS_MODEL_ITERATION}")
  message(STATUS
    "Model versions: ww=${WW_MCUBOOT_SIGN_VERSION}, kws=${KWS_MCUBOOT_SIGN_VERSION}")

  model_ota_edgeai_axon_model(
    TARGET ww
    SOLUTION_ID 36711
    MODEL_SRC ${WW_MODEL_SRC}
    HEADER ${WW_MODEL_HEADER}
    NAME ww
    VERSION ${MODEL_VERSION}
    SIGN_VERSION ${WW_MCUBOOT_SIGN_VERSION}
    PARTITION_NODELABEL model_storage_ww
    PERSISTENT_VARS_CAP 1160
    UUID_CID ww_kws_model_ww
    UUID_VID nordicsemi.com
  )
  model_ota_edgeai_axon_model(
    TARGET kws
    SOLUTION_ID 36712
    MODEL_SRC ${KWS_MODEL_SRC}
    HEADER ${KWS_MODEL_HEADER}
    NAME kws
    VERSION ${MODEL_VERSION}
    SIGN_VERSION ${KWS_MCUBOOT_SIGN_VERSION}
    PARTITION_NODELABEL model_storage_kws
    PERSISTENT_VARS_CAP 6648
    UUID_CID ww_kws_model_kws
    UUID_VID nordicsemi.com
  )

  include(${CMAKE_CURRENT_SOURCE_DIR}/../../lib/model_ota/cmake/model_ota_app_dfu_zip.cmake)
  model_ota_create_dfu_application_zip(
    MODEL ww;kws
    IMAGE_INDEX 1;2
    MODEL_SIGN_VERSION ${WW_MCUBOOT_SIGN_VERSION};${KWS_MCUBOOT_SIGN_VERSION}
  )

  target_sources(app PRIVATE src/model_update.c)

  model_ota_finalize()
endif()
