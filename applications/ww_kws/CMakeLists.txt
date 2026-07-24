#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(ww_kws)

target_sources(app PRIVATE
  src/main.c

  src/control_output.c
  src/dmic.c
  src/leds.c

  src/ww/wakeword.c
  src/ww/nrf_edgeai_generated/nrf_edgeai_user_model.c

  src/kws/kws.c
  src/kws/nrf_edgeai_generated/nrf_edgeai_user_model.c
)

target_include_directories(app PRIVATE
  src/ww/nrf_edgeai_generated
  src/kws/nrf_edgeai_generated
)

target_sources_ifdef(CONFIG_MODELS_OBSERVABILITY_MDS app PRIVATE src/ble/ble_mds.c)

if(CONFIG_APP_MODEL_OTA)
  # Neither model is compiled in: each is loaded from its own model_storage_{ww,kws} flash
  # partition instead (see nrf_edgeai_user_model.c and doc/libraries/model_ota.rst). The
  # generated headers stay out of the application link entirely - nrf_axon_model_stub() compiles
  # each one only into its own standalone "model stub", used solely to produce that model's
  # package as a build artifact.
  include(${CMAKE_CURRENT_SOURCE_DIR}/../../lib/model_ota/cmake/nrf_axon_model_stub.cmake)

  nrf_axon_model_stub(
    TARGET ww
    MODEL_NAME axon_user_instance_36711
    HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/ww/nrf_edgeai_generated/nrf_edgeai_user_model_axon.h
    PARTITION_NODELABEL model_storage_ww
  )
  nrf_axon_model_stub(
    TARGET kws
    MODEL_NAME axon_user_instance_36712
    HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/kws/nrf_edgeai_generated/nrf_edgeai_user_model_axon.h
    PARTITION_NODELABEL model_storage_kws
  )

  if(CONFIG_APP_MCUBOOT)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../lib/model_ota/cmake/nrf_model_mcuboot_sign.cmake)

    nrf_model_mcuboot_sign(
      TARGET ww
      PKG_BIN ${CMAKE_CURRENT_BINARY_DIR}/ww_model_pkg.bin
      PKG_TARGET ww_model_pkg
      PARTITION_NODELABEL model_storage_ww
      UUID_CID nRF54LM20_ww_kws_model_ww
      UUID_VID nordicsemi.com
    )
    nrf_model_mcuboot_sign(
      TARGET kws
      PKG_BIN ${CMAKE_CURRENT_BINARY_DIR}/kws_model_pkg.bin
      PKG_TARGET kws_model_pkg
      PARTITION_NODELABEL model_storage_kws
      UUID_CID nRF54LM20_ww_kws_model_kws
      UUID_VID nordicsemi.com
    )
  endif()
endif()

target_sources_ifdef(CONFIG_APP_MCUBOOT app PRIVATE src/model_update.c)

if(CONFIG_APP_MCUBOOT)
  target_link_libraries(app PRIVATE MCUBOOT_BOOTUTIL)
endif()
