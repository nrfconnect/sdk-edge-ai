#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# nrf_neuton_model_package(): build-integrated packaging for a Neuton model, producing a
# model_ota package (.bin/.hex) as part of a normal `west build`, mirroring what
# nrf_axon_model_stub.cmake does for Axon (see doc/libraries/model_ota.rst).
#
# Unlike Axon, a Neuton package embeds no addresses at all (see package_model_neuton.py's own
# module comment), so there is no equivalent of the Axon "model stub" second-pass link here: this
# module only needs to run package_model_neuton.py against MODEL_C, with --address/--partition-
# size read from devicetree at configure time (dt_reg_addr()/dt_reg_size()) instead of the
# tool's own --dts option, which requires an already-built zephyr.dts.
#
# Usage (from a sample/application's CMakeLists.txt, after target_sources(app ...)):
#
#   include(${SDK_EDGE_AI_DIR}/lib/model_ota/cmake/nrf_neuton_model_package.cmake)
#   nrf_neuton_model_package(
#     TARGET regression                                              # unique within this build
#     MODEL_NAME aq_regression                                       # embedded in the package header
#     MODEL_C ${SDK_EDGE_AI_DIR}/tools/model_ota/models/regression_v1_generated.c
#     PARTITION_NODELABEL model_partition  # devicetree node the package is built for
#   )

set(MODEL_OTA_TOOLS_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../tools/model_ota)

function(nrf_neuton_model_package)
	cmake_parse_arguments(ARG "" "TARGET;MODEL_NAME;MODEL_C;PARTITION_NODELABEL;VERSION" "" ${ARGN})

	if(NOT ARG_TARGET OR NOT ARG_MODEL_NAME OR NOT ARG_MODEL_C OR NOT ARG_PARTITION_NODELABEL)
		message(FATAL_ERROR
			"nrf_neuton_model_package() requires TARGET, MODEL_NAME, MODEL_C and "
			"PARTITION_NODELABEL")
	endif()
	if(NOT ARG_VERSION)
		set(ARG_VERSION "1.0.0")
	endif()

	dt_nodelabel(partition_path NODELABEL ${ARG_PARTITION_NODELABEL})
	if(NOT partition_path)
		message(FATAL_ERROR
			"nrf_neuton_model_package(${ARG_TARGET}): devicetree has no node labelled "
			"'${ARG_PARTITION_NODELABEL}' - see boards/*.overlay")
	endif()
	dt_reg_addr(partition_addr PATH "${partition_path}")
	dt_reg_size(partition_size PATH "${partition_path}")

	set(pkg_base ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_model_pkg)

	add_custom_command(
		OUTPUT ${pkg_base}.bin ${pkg_base}.hex
		COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_TOOLS_DIR}/package_model_neuton.py
			${ARG_MODEL_C} --name ${ARG_MODEL_NAME} --version ${ARG_VERSION}
			--address ${partition_addr} --partition-size ${partition_size}
			-o ${pkg_base}
		DEPENDS ${ARG_MODEL_C} ${MODEL_OTA_TOOLS_DIR}/package_model_neuton.py
		COMMENT "model_ota: packaging ${ARG_TARGET} (${ARG_MODEL_NAME}, Neuton)"
	)

	add_custom_target(${ARG_TARGET}_model_pkg ALL DEPENDS ${pkg_base}.bin ${pkg_base}.hex)
endfunction()
