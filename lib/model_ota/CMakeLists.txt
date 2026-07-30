#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

zephyr_library()

zephyr_library_include_directories_ifdef(
  CONFIG_MODEL_OTA_TESTING
  ${ZEPHYR_MCUBOOT_MODULE_DIR}/boot/bootutil/include
  ${ZEPHYR_MCUBOOT_MODULE_DIR}/boot/zephyr/include
)

zephyr_library_sources(model_image_common.c)
# Both backends carry the nrf_edgeai_t parameter block (a pure Axon model simply leaves it
# zeroed), so the binder is backend-agnostic.
zephyr_library_sources(model_image_edgeai_params.c)
zephyr_library_sources_ifdef(CONFIG_MODEL_OTA_NEUTON model_image_neuton.c)
zephyr_library_sources_ifdef(CONFIG_MODEL_OTA_AXON model_image_axon.c)

zephyr_library_link_libraries_ifdef(CONFIG_BOOTLOADER_MCUBOOT MCUBOOT_BOOTUTIL)
