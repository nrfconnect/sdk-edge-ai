#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Sysbuild helpers for model-only OTA samples that also use MCUboot.
#
# nrf_model_sysbuild_provision() registers the model package hex as an extra
# sysbuild flash domain (west flash programs it after the bootloader and app)
# and, when MCUboot is enabled, builds a single merged provision hex for
# nrfutil or west flash --hex-file.

function(nrf_model_sysbuild_provision)
	cmake_parse_arguments(ARG "" "APP_IMAGE;MODEL_HEX" "" ${ARGN})

	if(NOT ARG_APP_IMAGE OR NOT ARG_MODEL_HEX)
		message(FATAL_ERROR
			"nrf_model_sysbuild_provision() requires APP_IMAGE and MODEL_HEX")
	endif()
	if(NOT IS_ABSOLUTE "${ARG_MODEL_HEX}")
		message(FATAL_ERROR
			"nrf_model_sysbuild_provision(): MODEL_HEX must be an absolute path")
	endif()

	if(SB_CONFIG_PARTITION_MANAGER)
		return()
	endif()

	include(${ZEPHYR_NRF_MODULE_DIR}/sysbuild/image_flasher.cmake)

	set(flasher_image ${ARG_APP_IMAGE}_model)

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

	if(NOT SB_CONFIG_BOOTLOADER_MCUBOOT)
		return()
	endif()

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
		COMMENT "model_ota: merging bootloader, app, and model into ${provision_hex}"
	)

	add_custom_target(${ARG_APP_IMAGE}_provision_hex ALL DEPENDS ${provision_hex})
endfunction()
