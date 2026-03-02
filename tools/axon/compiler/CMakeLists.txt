#
# Copyright (c) 2025 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

if (${NOT_A_ZEPHYR_BUILD})
  target_include_directories(nrf-axon-compiler-interface
    INTERFACE
      include
  )
endif()
