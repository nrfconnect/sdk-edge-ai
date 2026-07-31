#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Application-image helper: build dfu_application.zip with the app (image 0) and
# pre-signed model MCUboot images. Invoke from the app CMakeLists when
# CONFIG_APP_MODEL_OTA is enabled (devicetree is available there; sysbuild is too
# early for partition address lookup).

function(model_ota_mcuboot_image_number_to_slot result image secondary)
	if(secondary)
		set(secondary_offset "+ 1")
	else()
		set(secondary_offset "")
	endif()

	math(EXPR slot "${image} * 2 ${secondary_offset}")

	set(${result} ${slot} PARENT_SCOPE)
endfunction()

function(model_ota_create_dfu_application_zip)
	cmake_parse_arguments(ARG "" "" "MODEL;IMAGE_INDEX;MODEL_SIGN_VERSION" ${ARGN})

	if(NOT CONFIG_APP_MODEL_OTA)
		return()
	endif()
	if(NOT ARG_MODEL OR NOT ARG_IMAGE_INDEX)
		message(FATAL_ERROR
			"${CMAKE_CURRENT_FUNCTION}() requires MODEL and IMAGE_INDEX")
	endif()

	list(LENGTH ARG_MODEL model_count)
	list(LENGTH ARG_IMAGE_INDEX index_count)
	if(NOT model_count EQUAL index_count)
		message(FATAL_ERROR
			"${CMAKE_CURRENT_FUNCTION}(): MODEL and IMAGE_INDEX must have the same length")
	endif()
	if(ARG_MODEL_SIGN_VERSION)
		list(LENGTH ARG_MODEL_SIGN_VERSION sign_version_count)
		if(NOT sign_version_count EQUAL model_count)
			message(FATAL_ERROR
				"${CMAKE_CURRENT_FUNCTION}(): MODEL_SIGN_VERSION must match MODEL length")
		endif()
	endif()

	set(sysbuild_dir ${CMAKE_BINARY_DIR}/..)
	set(app_update_name "${CMAKE_PROJECT_NAME}.signed.bin")
	set(app_bin ${CMAKE_BINARY_DIR}/zephyr/${CONFIG_KERNEL_BIN_NAME}.signed.bin)
	set(zip_output ${sysbuild_dir}/dfu_application.zip)

	model_ota_mcuboot_image_number_to_slot(slot_primary 0 n)
	model_ota_mcuboot_image_number_to_slot(slot_secondary 0 y)

	dt_nodelabel(slot0 NODELABEL "slot${slot_primary}_partition")
	dt_reg_addr(load_address PATH "${slot0}")

	math(EXPR slot_primary "${slot_primary} + 1")
	math(EXPR slot_secondary "${slot_secondary} + 1")

	set(script_params
		"${app_update_name}load_address=${load_address}"
		"${app_update_name}image_index=0"
		"${app_update_name}slot_index_primary=${slot_primary}"
		"${app_update_name}slot_index_secondary=${slot_secondary}"
		"${app_update_name}version_MCUBOOT=${CONFIG_MCUBOOT_IMGTOOL_SIGN_VERSION}"
	)

	set(bin_files ${app_bin})
	set(zip_names ${app_update_name})
	set(deps
		${app_bin}
		${CMAKE_BINARY_DIR}/zephyr/${CONFIG_KERNEL_BIN_NAME}.bin
	)

	set(model_idx 0)
	foreach(model IN LISTS ARG_MODEL)
		list(GET ARG_IMAGE_INDEX ${model_idx} image_index)

		set(model_bin ${CMAKE_BINARY_DIR}/${model}_model_mcuboot.signed.bin)
		set(zip_name "${model}_model_mcuboot.signed.bin")

		model_ota_mcuboot_image_number_to_slot(model_slot_primary ${image_index} n)
		model_ota_mcuboot_image_number_to_slot(model_slot_secondary ${image_index} y)
		math(EXPR model_slot_primary "${model_slot_primary} + 1")
		math(EXPR model_slot_secondary "${model_slot_secondary} + 1")

		list(APPEND script_params
			"${zip_name}image_index=${image_index}"
			"${zip_name}slot_index_primary=${model_slot_primary}"
			"${zip_name}slot_index_secondary=${model_slot_secondary}"
		)
		if(ARG_MODEL_SIGN_VERSION)
			list(GET ARG_MODEL_SIGN_VERSION ${model_idx} model_sign_version)
			list(APPEND script_params
				"${zip_name}version_MCUBOOT=${model_sign_version}"
				"${zip_name}app_version_compat=${CONFIG_MCUBOOT_IMGTOOL_SIGN_VERSION}"
			)
		endif()
		list(APPEND bin_files ${model_bin})
		list(APPEND zip_names ${zip_name})
		list(APPEND deps ${model}_model_mcuboot_signed)

		math(EXPR model_idx "${model_idx} + 1")
	endforeach()

	set(meta_argument)
	if(CONFIG_BUILD_OUTPUT_META)
		set(meta_argument
			--meta-info-file ${CMAKE_BINARY_DIR}/zephyr/${CONFIG_KERNEL_BIN_NAME}.meta)
	endif()

	add_custom_command(
		OUTPUT ${zip_output}
		COMMAND ${PYTHON_EXECUTABLE}
			${ZEPHYR_NRF_MODULE_DIR}/scripts/bootloader/generate_zip.py
			--bin-files ${bin_files}
			--zip-names ${zip_names}
			--output ${zip_output}
			--name "${CMAKE_PROJECT_NAME}"
			--format-version 1
			${meta_argument}
			${script_params}
			type=application
			board=${CONFIG_BOARD}
			soc=${CONFIG_SOC}
		DEPENDS ${deps}
		COMMENT "model_ota: packaging dfu_application.zip (app + models)"
		VERBATIM
	)

	add_custom_target(dfu_application_zip ALL DEPENDS ${zip_output})
endfunction()
