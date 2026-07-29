#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

include(${APP_DIR}/../../lib/model_ota/cmake/model_ota_sysbuild.cmake)

# Model partition flash domains apply only with APP_MODEL_OTA (three MCUboot images).
if(SB_CONFIG_BOOTLOADER_MCUBOOT AND SB_CONFIG_MCUBOOT_UPDATEABLE_IMAGES GREATER_EQUAL 3)
	model_ota_register_provision_hex(
		APP_IMAGE ww_kws
		MODEL_HEX ${CMAKE_BINARY_DIR}/ww_kws/ww_model_mcuboot.signed.hex
		FLASHER_NAME ww_kws_model_ww
		SKIP_PROVISION_MERGE y
	)

	model_ota_register_provision_hex(
		APP_IMAGE ww_kws
		MODEL_HEX ${CMAKE_BINARY_DIR}/ww_kws/kws_model_mcuboot.signed.hex
		FLASHER_NAME ww_kws_model_kws
		SKIP_PROVISION_MERGE y
	)

	model_ota_create_provision_hex(
		APP_IMAGE ww_kws
		MODEL_HEX
			${CMAKE_BINARY_DIR}/ww_kws/ww_model_mcuboot.signed.hex
			${CMAKE_BINARY_DIR}/ww_kws/kws_model_mcuboot.signed.hex
	)
endif()
