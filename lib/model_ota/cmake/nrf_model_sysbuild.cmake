#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Sysbuild helpers for model-only OTA.
#
# nrf_model_register_provision_hex() registers the model package hex as an extra
# sysbuild flash domain (west flash programs it after the bootloader and app)
# and, when MCUboot is enabled, builds a single merged provision hex for
# nrfutil or west flash --hex-file.

function(nrf_model_create_provision_hex)
	cmake_parse_arguments(ARG "" "APP_IMAGE" "MODEL_HEX" ${ARGN})

	if(NOT ARG_APP_IMAGE OR NOT ARG_MODEL_HEX)
		message(FATAL_ERROR
			"${CMAKE_CURRENT_FUNCTION}() requires APP_IMAGE and one or more MODEL_HEX paths")
	endif()

	if(NOT SB_CONFIG_BOOTLOADER_MCUBOOT)
		return()
	endif()

	foreach(model_hex IN LISTS ARG_MODEL_HEX)
		if(NOT IS_ABSOLUTE "${model_hex}")
			message(FATAL_ERROR
				"${CMAKE_CURRENT_FUNCTION}(): MODEL_HEX must be absolute paths")
		endif()
	endforeach()

	set(provision_hex ${CMAKE_BINARY_DIR}/${ARG_APP_IMAGE}_provision.hex)
	set(merge_inputs
		${CMAKE_BINARY_DIR}/mcuboot/zephyr/zephyr.hex
		${CMAKE_BINARY_DIR}/${ARG_APP_IMAGE}/zephyr/zephyr.signed.hex
		${ARG_MODEL_HEX}
	)

	add_custom_command(
		OUTPUT ${provision_hex}
		COMMAND ${PYTHON_EXECUTABLE} ${ZEPHYR_BASE}/scripts/build/mergehex.py
			-o ${provision_hex}
			--overlap replace
			${merge_inputs}
		DEPENDS
			mcuboot_extra_byproducts
			${ARG_APP_IMAGE}_extra_byproducts
			${ARG_APP_IMAGE}
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
		COMMENT "model_ota: merging bootloader, app, and models into ${provision_hex}"
	)

	add_custom_target(${ARG_APP_IMAGE}_provision_hex ALL DEPENDS ${provision_hex})
endfunction()

function(nrf_model_register_provision_hex)
	cmake_parse_arguments(ARG "SKIP_PROVISION_MERGE" "APP_IMAGE;MODEL_HEX;FLASHER_NAME" "" ${ARGN})

	if(NOT ARG_APP_IMAGE OR NOT ARG_MODEL_HEX)
		message(FATAL_ERROR
			"${CMAKE_CURRENT_FUNCTION}() requires APP_IMAGE and MODEL_HEX")
	endif()
	if(NOT ARG_FLASHER_NAME)
		set(ARG_FLASHER_NAME ${ARG_APP_IMAGE}_model)
	endif()
	if(NOT IS_ABSOLUTE "${ARG_MODEL_HEX}")
		message(FATAL_ERROR
			"${CMAKE_CURRENT_FUNCTION}(): MODEL_HEX must be an absolute path")
	endif()

	include(${ZEPHYR_NRF_MODULE_DIR}/sysbuild/image_flasher.cmake)

	set(flasher_image ${ARG_FLASHER_NAME})

	add_image_flasher(
		NAME ${flasher_image}
		HEX_FILE "${ARG_MODEL_HEX}"
		BASE_IMAGE ${ARG_APP_IMAGE}
	)

	sysbuild_add_dependencies(CONFIGURE ${flasher_image} ${ARG_APP_IMAGE})

	if(SB_CONFIG_BOOTLOADER_MCUBOOT)
		sysbuild_add_dependencies(FLASH ${flasher_image} mcuboot ${ARG_APP_IMAGE})
	else()
		sysbuild_add_dependencies(FLASH ${flasher_image} ${ARG_APP_IMAGE})
	endif()

	if(NOT SB_CONFIG_BOOTLOADER_MCUBOOT OR ARG_SKIP_PROVISION_MERGE)
		return()
	endif()

	nrf_model_create_provision_hex(
		APP_IMAGE ${ARG_APP_IMAGE}
		MODEL_HEX ${ARG_MODEL_HEX}
	)
endfunction()
