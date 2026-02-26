#
# Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
#
# The information contained herein is confidential property of Nordic Semiconductor ASA.
# The use, copying, transfer or disclosure of such information is prohibited except by
# express written agreement with Nordic Semiconductor ASA.
#
cmake_minimum_required(VERSION 3.10.0)

project(nrf_axon_app_test_nn_inference VERSION 0.1.0 LANGUAGES C)
	set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

	set(NOT_A_ZEPHYR_BUILD 1)
	set(CONFIG_NRF_AXON 1)
	set(CONFIG_AXON_INTERLAYER_BUFFER_SIZE 2000000)
	set(CONFIG_AXON_PSUM_BUFFER_SIZE 400000)

	set(AXON_ROOT "${CMAKE_SOURCE_DIR}/../../../../simulator")
	# include(${AXON_ROOT}/CMakeLists.txt)
	add_subdirectory(${AXON_ROOT} ..)

	include_directories(PRIVATE ../../compiled_models)

	# there will be multiple build targets that only vary in which model is included
	# so put all the common stuff into variables
	# nrf-axon-nn-compiler-internal
	set(AXONS_INFER_BASE_LIBS nrf-axon-platform nrf-axon-driver nrf-axon-simulator-internal nrf-axon-driver-internal)
	
	# For command line builds of build target axonpro_app_infer_model, set the model name at the command line with 
	# cmake -D NRF_AXON_MODEL_NAME:STRING=<model_name> -S ..
	# For vs code builds, set the model name explicitly in the "else" clause below
	if (DEFINED NRF_AXON_MODEL_NAME)
		message(STATUS "NRF_AXON_MODEL_NAME DEFINED ${NRF_AXON_MODEL_NAME}")
	else()
		# no model name defined so pick one
		set(NRF_AXON_MODEL_NAME tinyml_kws)
		# set(NRF_AXON_MODEL_NAME tinyml_ic)
		# set(NRF_AXON_MODEL_NAME tinyml_vww)
		# set(NRF_AXON_MODEL_NAME tinyml_ad)
		# set(NRF_AXON_MODEL_NAME tinyml_kws_transposed)

		message(STATUS "NRF_AXON_MODEL_NAME NOT DEFINED, building model ${NRF_AXON_MODEL_NAME}")
	endif()

	add_executable(nrf-axon-app-test-infer ../src/nrf_axon_app_test_nn_inference.c)
	target_compile_definitions(nrf-axon-app-test-infer
		PRIVATE
			NRF_AXON_MODEL_NAME=${NRF_AXON_MODEL_NAME}
			NOT_A_ZEPHYR_BUILD=1
		)

	target_link_libraries(nrf-axon-app-test-infer
		PRIVATE
			${AXONS_INFER_BASE_LIBS}
	)
