#
# Copyright (c) 2026 Nordic Semiconductor
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

# use policy cmp0079
cmake_minimum_required(VERSION 3.13.0)

if (${NOT_A_ZEPHYR_BUILD})
  add_library(nrf-axon-driver-interface INTERFACE)

  target_include_directories(nrf-axon-driver-interface INTERFACE ../../include)

  target_link_libraries(nrf-axon-driver-interface
    INTERFACE
      nrf-axon-platform-interface
  )

  add_library(nrf-axon-driver STATIC EXCLUDE_FROM_ALL)
  set_target_properties(nrf-axon-driver PROPERTIES LINKER_LANGUAGE C)

  target_compile_definitions(nrf-axon-driver
    PRIVATE
      AXON_SIMULATION=1
  )

  target_sources(nrf-axon-driver
    PRIVATE
      nrf_axon_nn_infer.c
      nrf_axon_nn_infer_test.c
	    nrf_axon_nn_op_extensions.c
  )

  target_link_libraries(nrf-axon-driver
    PUBLIC
      nrf-axon-driver-interface
    PRIVATE
      nrf-axon-platform-interface
  )

else()
	zephyr_sources(
		nrf_axon_nn_infer.c
		nrf_axon_nn_infer_test.c
		nrf_axon_nn_op_extensions.c
	)
endif()
