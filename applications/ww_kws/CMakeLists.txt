#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(ww_kws)

# src/*/model_wiring_axon.c replace src/*/nrf_edgeai_generated/nrf_edgeai_user_model.c (not
# built here at all any more): each is a drop-in equivalent that additionally branches on
# CONFIG_APP_MODEL_OTA internally - see the comment at the top of either file.
target_sources(app PRIVATE
  src/main.c

  src/control_output.c
  src/dmic.c
  src/leds.c

  src/ww/wakeword.c
  src/ww/model_wiring_axon.c

  src/kws/kws.c
  src/kws/model_wiring_axon.c
)

target_sources_ifdef(CONFIG_MODELS_OBSERVABILITY_MDS app PRIVATE src/ble/ble_mds.c)

if(CONFIG_APP_MODEL_OTA)
  # Neither model is compiled in: each is loaded from its own model_storage_{ww,kws} flash
  # partition instead (see src/*/model_wiring_axon.c and doc/libraries/model_ota.rst). The
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
endif()
