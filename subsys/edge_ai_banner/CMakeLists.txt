# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

zephyr_library_named(edge_ai_banner)
zephyr_library_sources_ifdef(CONFIG_EDGE_AI_BOOT_BANNER edge_ai_banner.c)
