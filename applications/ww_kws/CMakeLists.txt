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

target_sources_ifdef(CONFIG_MODELS_OBSERVABILITY_WW app PRIVATE src/ww/ww_obsv.c)
target_sources_ifdef(CONFIG_MODELS_OBSERVABILITY_KWS app PRIVATE src/kws/kws_obsv.c)
target_sources_ifdef(CONFIG_MODELS_OBSERVABILITY_MDS app PRIVATE src/ble/ble_mds.c)
target_sources_ifdef(CONFIG_MODELS_OBSERVABILITY_MEMFAULT_HTTP app PRIVATE src/wifi/memfault_http_obsv.c)

if(CONFIG_WIFI_NRF71_PATCH)
  # Without the patch regions in the devicetree, the Wi-Fi ROM patches are
  # silently left out of the image and the Wi-Fi core never starts.
  dt_nodelabel(umac_patch NODELABEL umac_rom_patch_addr)
  if(NOT umac_patch)
    message(FATAL_ERROR
      "The Wi-Fi transport needs the Wi-Fi ROM patch regions. "
      "Add -DEXTRA_DTC_OVERLAY_FILE=observability_wifi.overlay")
  endif()

  # Place the Axon models above the Wi-Fi ROM patches, page aligned.
  dt_reg_addr(umac_patch_addr PATH ${umac_patch})
  dt_reg_size(umac_patch_size PATH ${umac_patch})
  math(EXPR AXON_MODELS_ADDR
    "((${umac_patch_addr} + ${umac_patch_size} + 0xFFF) / 0x1000) * 0x1000"
    OUTPUT_FORMAT HEXADECIMAL)
  configure_file(axon_models_after_wifi_patch.ld.in
    ${CMAKE_CURRENT_BINARY_DIR}/axon_models_after_wifi_patch.ld @ONLY)
  zephyr_linker_sources(ROM_SECTIONS
    ${CMAKE_CURRENT_BINARY_DIR}/axon_models_after_wifi_patch.ld)
endif()
