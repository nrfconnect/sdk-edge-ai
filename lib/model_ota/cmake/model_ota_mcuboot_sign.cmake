#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Signs a raw model partition image (.bin) as an MCUboot image.

function(model_ota_mcuboot_sign)
	cmake_parse_arguments(ARG ""
		"TARGET;MODEL_IMAGE_BIN;MODEL_IMAGE_TARGET;PARTITION_NODELABEL;UUID_CID;UUID_VID" "" ${ARGN})

	if(NOT ARG_TARGET OR NOT ARG_MODEL_IMAGE_BIN OR NOT ARG_PARTITION_NODELABEL)
		message(FATAL_ERROR
			"${CMAKE_CURRENT_FUNCTION}() requires TARGET, MODEL_IMAGE_BIN, and PARTITION_NODELABEL")
	endif()

	if(NOT DEFINED IMGTOOL)
		message(FATAL_ERROR
			"${CMAKE_CURRENT_FUNCTION}(${ARG_TARGET}): IMGTOOL not found (MCUboot module missing?)")
	endif()

	set(keyfile "${CONFIG_MCUBOOT_SIGNATURE_KEY_FILE}")
	string(CONFIGURE "${keyfile}" keyfile)
	if(NOT keyfile OR NOT EXISTS "${keyfile}")
		message(FATAL_ERROR
			"${CMAKE_CURRENT_FUNCTION}(${ARG_TARGET}): CONFIG_MCUBOOT_SIGNATURE_KEY_FILE "
			"not set or missing (${keyfile})")
	endif()

	dt_nodelabel(slot_path NODELABEL ${ARG_PARTITION_NODELABEL})
	if(NOT slot_path)
		message(FATAL_ERROR
			"${CMAKE_CURRENT_FUNCTION}(${ARG_TARGET}): no devicetree node labelled "
			"'${ARG_PARTITION_NODELABEL}'")
	endif()
	dt_reg_addr(slot_addr PATH "${slot_path}")
	dt_reg_size(slot_size PATH "${slot_path}")

	dt_chosen(flash_node PROPERTY "zephyr,flash")
	dt_prop(write_block_size PATH "${flash_node}" PROPERTY "write-block-size")
	if(NOT write_block_size)
		set(write_block_size 16)
	endif()

	set(out_base ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_model_mcuboot)
	set(imgtool_hash_arg)
	if(CONFIG_MCUBOOT_BOOTLOADER_USES_SHA512)
		set(imgtool_hash_arg --sha 512)
	endif()

	set(imgtool_uuid_args)
	if(ARG_UUID_VID)
		list(APPEND imgtool_uuid_args --vid "${ARG_UUID_VID}")
	endif()
	if(ARG_UUID_CID)
		list(APPEND imgtool_uuid_args --cid "${ARG_UUID_CID}")
	endif()

	set(imgtool_sign_base ${PYTHON_EXECUTABLE} ${IMGTOOL} sign
		--version ${CONFIG_MCUBOOT_IMGTOOL_SIGN_VERSION}
		--header-size 32
		--pad-header
		--pad
		${imgtool_hash_arg}
		--slot-size ${slot_size}
		--align ${write_block_size}
		--rom-fixed ${slot_addr}
		${imgtool_uuid_args}
		-k "${keyfile}"
		--overwrite-only
	)

	set(deps ${ARG_MODEL_IMAGE_BIN})
	if(ARG_MODEL_IMAGE_TARGET)
		list(APPEND deps ${ARG_MODEL_IMAGE_TARGET})
	endif()

	set(outputs ${out_base}.signed.bin ${out_base}.signed.hex)

	add_custom_command(
		OUTPUT ${outputs}
		COMMAND ${imgtool_sign_base} ${ARG_MODEL_IMAGE_BIN} ${out_base}.signed.bin
		COMMAND ${imgtool_sign_base} --hex-addr ${slot_addr}
			${ARG_MODEL_IMAGE_BIN} ${out_base}.signed.hex
		DEPENDS ${deps}
		COMMENT "model_ota: signing ${ARG_TARGET} model for MCUboot"
	)

	add_custom_target(${ARG_TARGET}_model_mcuboot_signed ALL DEPENDS ${outputs})
endfunction()
