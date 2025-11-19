#
# Copyright (c) 2025 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

set(SID_LIB_DIR ${ZEPHYR_BASE}/../sdk-edge-ai/lib/nrf_edgeai/cortex-m33)


zephyr_library_link_libraries(
    ${SID_LIB_DIR}/libnrf_edgeai_cortex-m33.a 
)