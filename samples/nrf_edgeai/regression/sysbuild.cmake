#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

if(NOT SB_CONFIG_BOOTLOADER_MCUBOOT OR SB_CONFIG_MCUBOOT_UPDATEABLE_IMAGES LESS 2)
	return()
endif()

if(SB_CONFIG_NRF_EDGEAI_REGRESSION_MODEL_SLOT_DUAL)
	set(_model_layout_overlay ${APP_DIR}/dts/nrf54lm20dk_mcuboot_model_dual.overlay)
	set(regression_EXTRA_DTC_OVERLAY_FILE ${_model_layout_overlay} CACHE INTERNAL "" FORCE)
	set(mcuboot_EXTRA_DTC_OVERLAY_FILE ${_model_layout_overlay} CACHE INTERNAL "" FORCE)
	set_config_bool(${DEFAULT_IMAGE} CONFIG_NRF_EDGEAI_REGRESSION_MODEL_MCUBOOT_DUAL_SLOT y)
else()
	set_config_bool(${DEFAULT_IMAGE} CONFIG_NRF_EDGEAI_REGRESSION_MODEL_MCUBOOT_DUAL_SLOT n)
endif()

include(${APP_DIR}/../../../lib/model_ota/cmake/nrf_model_sysbuild.cmake)

nrf_model_register_provision_hex(
	APP_IMAGE regression
	MODEL_HEX ${CMAKE_BINARY_DIR}/regression/regression_model_mcuboot.signed.hex
)
