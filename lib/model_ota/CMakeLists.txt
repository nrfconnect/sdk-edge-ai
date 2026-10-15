#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/model_ota_common.cmake)

zephyr_library()

zephyr_library_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
zephyr_library_compile_definitions(MODEL_IMAGE_PAYLOAD_OFFSET=${MODEL_IMAGE_OFFSET_MCUBOOT})

zephyr_library_sources(model_image_common.c)
# Both backends carry the nrf_edgeai_t parameter block (a pure Axon model simply leaves it
# zeroed), so the binder is backend-agnostic.
zephyr_library_sources(model_image_edgeai_params.c)
if(CONFIG_NRF_EDGEAI)
  zephyr_library_sources(model_ota_edgeai.c)
  zephyr_link_libraries(-Wl,--wrap=nrf_edgeai_init)
endif()
zephyr_library_sources_ifdef(CONFIG_MODEL_OTA_SMP model_ota_guard.c)
zephyr_library_sources_ifdef(CONFIG_MODEL_OTA_NEUTON model_image_neuton.c)
zephyr_library_sources_ifdef(CONFIG_MODEL_OTA_AXON model_image_axon.c)
zephyr_library_sources_ifdef(CONFIG_MODEL_OTA_SMP model_ota_smp.c)
zephyr_linker_sources_ifdef(CONFIG_MODEL_OTA_SMP SECTIONS linker/model_ota_smp.ld)
zephyr_library_link_libraries_ifdef(CONFIG_MODEL_OTA_SMP MCUBOOT_BOOTUTIL)
