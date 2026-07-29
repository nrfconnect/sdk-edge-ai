#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

include(${APP_DIR}/../../lib/model_ota/cmake/model_ota_sysbuild.cmake)

if(SB_CONFIG_APP_MODEL_OTA)
	set_config_bool(${DEFAULT_IMAGE} CONFIG_APP_MODEL_OTA y)
	add_overlay_config(${DEFAULT_IMAGE} ${APP_DIR}/model_ota.conf)
	add_overlay_dts(${DEFAULT_IMAGE}
			${APP_DIR}/boards/nrf54lm20dk_nrf54lm20b_cpuapp_model_ota.overlay)
	add_overlay_dts(mcuboot
			${APP_DIR}/sysbuild/mcuboot/boards/nrf54lm20dk_nrf54lm20b_cpuapp_model_ota.overlay)
	add_overlay_config(mcuboot ${APP_DIR}/sysbuild/mcuboot/model_ota.conf)
else()
	set_config_bool(${DEFAULT_IMAGE} CONFIG_APP_MODEL_OTA n)
	set(mcuboot_DTC_OVERLAY_FILE
	    ${APP_DIR}/sysbuild/mcuboot/boards/nrf54lm20dk_nrf54lm20b_cpuapp.overlay
	    CACHE INTERNAL "ww_kws MCUboot overlay" FORCE)
endif()

if(SB_CONFIG_APP_MODEL_OTA)
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
