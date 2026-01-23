#
# Copyright (c) 2025 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

add_subdirectory_ifdef(CONFIG_NRF_AXON axon)

if (CONFIG_NRF_EDGEAI)
    add_subdirectory(nrf_edgeai)
endif(CONFIG_NRF_EDGEAI)
