#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

zephyr_library()

zephyr_library_sources_ifdef(CONFIG_MODEL_OTA_NEUTON model_pkg_neuton.c)
zephyr_library_sources_ifdef(CONFIG_MODEL_OTA_AXON model_pkg_axon.c)
