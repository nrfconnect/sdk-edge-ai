#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

include(${APP_DIR}/../../../lib/model_ota/cmake/model_ota_sysbuild.cmake)

if(BOARD MATCHES "nrf54lm20a")
	set(_regression_board_suffix nrf54lm20a)
else()
	set(_regression_board_suffix nrf54lm20b)
endif()

if(SB_CONFIG_APP_MODEL_OTA)
	set_config_bool(${DEFAULT_IMAGE} CONFIG_APP_MODEL_OTA y)
	add_overlay_config(${DEFAULT_IMAGE} ${APP_DIR}/model_ota.conf)
	add_overlay_dts(${DEFAULT_IMAGE}
			${APP_DIR}/boards/nrf54lm20dk_${_regression_board_suffix}_cpuapp_model_ota.overlay)
	add_overlay_dts(mcuboot
			${APP_DIR}/sysbuild/mcuboot/boards/nrf54lm20dk_${_regression_board_suffix}_cpuapp_model_ota.overlay)
else()
	set_config_bool(${DEFAULT_IMAGE} CONFIG_APP_MODEL_OTA n)
	set(mcuboot_DTC_OVERLAY_FILE
	    ${APP_DIR}/sysbuild/mcuboot/boards/nrf54lm20dk_${_regression_board_suffix}_cpuapp.overlay
	    CACHE INTERNAL "regression MCUboot overlay" FORCE)
endif()

if(SB_CONFIG_BOOTLOADER_MCUBOOT AND SB_CONFIG_APP_MODEL_OTA)
	model_ota_register_provision_hex(
		APP_IMAGE regression
		MODEL_HEX ${CMAKE_BINARY_DIR}/regression/regression_model_mcuboot.signed.hex
		FLASHER_NAME regression_model
		SKIP_PROVISION_MERGE y
	)

	model_ota_create_provision_hex(
		APP_IMAGE regression
		MODEL_HEX ${CMAKE_BINARY_DIR}/regression/regression_model_mcuboot.signed.hex
	)
endif()
