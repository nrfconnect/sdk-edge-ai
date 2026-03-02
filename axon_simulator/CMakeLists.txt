#
# Copyright (c) 2026 Nordic Semiconductor
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

# The directory of this file should be added in cmake, when using simulator
if (${NOT_A_ZEPHYR_BUILD})
	add_library(nrf-axon-driver-interface INTERFACE)
	target_include_directories(nrf-axon-driver-interface INTERFACE 
		../include
	)
	add_library(nrf-axon-compiler-interface INTERFACE)
	add_library(nrf-axon-platform-interface INTERFACE)
	add_subdirectory(../lib/axon ${CMAKE_BINARY_DIR}/lib_axon)
	add_subdirectory(../drivers/axon ${CMAKE_BINARY_DIR}/drivers_axon)
	add_subdirectory(../tools/axon/compiler ${CMAKE_BINARY_DIR}/tools/axon/compiler)
endif()
