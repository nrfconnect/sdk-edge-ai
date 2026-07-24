#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

# Register both model package hex files as sysbuild flash domains so a single
# `west flash` programs the application and both model_storage_* partitions.
# Requires CONFIG_APP_MODEL_OTA=y (the default); with model OTA disabled the
# package hex files are not built and only the ww_kws domain should be flashed.

include(${APP_DIR}/../../lib/model_ota/cmake/nrf_model_sysbuild.cmake)

nrf_model_register_provision_hex(
	APP_IMAGE ww_kws
	MODEL_HEX ${CMAKE_BINARY_DIR}/ww_kws/ww_model_pkg.hex
	FLASHER_NAME ww_kws_model_ww
)

nrf_model_register_provision_hex(
	APP_IMAGE ww_kws
	MODEL_HEX ${CMAKE_BINARY_DIR}/ww_kws/kws_model_pkg.hex
	FLASHER_NAME ww_kws_model_kws
)
