#
# Copyright (c) 2026 Nordic Semiconductor
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

# The directory of this file should be added in cmake, when using simulator
if (${NOT_A_ZEPHYR_BUILD})
  add_subdirectory(../drivers/axon ${CMAKE_BINARY_DIR}/drivers_axon)
  add_subdirectory(../lib/axon ${CMAKE_BINARY_DIR}/lib_axon)
endif()
