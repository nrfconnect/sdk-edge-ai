#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

zephyr_library()
zephyr_library_include_directories(include)

zephyr_library_sources(loader/model_image_common.c)
zephyr_library_sources_ifdef(CONFIG_MODEL_OTA_AXON loader/model_image_axon.c)
