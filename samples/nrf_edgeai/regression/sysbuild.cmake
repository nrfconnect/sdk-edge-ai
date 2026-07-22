#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

if(NOT SB_CONFIG_BOOTLOADER_MCUBOOT OR SB_CONFIG_MCUBOOT_UPDATEABLE_IMAGES LESS 2)
	return()
endif()

include(${APP_DIR}/../../../lib/model_ota/cmake/nrf_model_sysbuild.cmake)

nrf_model_sysbuild_provision(
	APP_IMAGE regression
	MODEL_HEX ${CMAKE_BINARY_DIR}/regression/regression_model_mcuboot.signed.hex
)
