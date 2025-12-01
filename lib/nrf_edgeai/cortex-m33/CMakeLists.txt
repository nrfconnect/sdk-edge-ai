#
# Copyright (c) 2025 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

set(SID_LIB_DIR ${CMAKE_CURRENT_LIST_DIR})


zephyr_library_link_libraries(
    ${SID_LIB_DIR}/libnrf_edgeai_cortex-m33.a 
)