#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Signs a raw model_ota package (.bin) as an MCUboot image for slot2 (image 1
# primary). Required when model_storage is an MCUboot updateable image: MCUboot
# validates every configured image at boot and rejects non-MCUboot content.

function(nrf_model_mcuboot_sign)
	cmake_parse_arguments(ARG "PROVISION_CONFIRM"
		"TARGET;PKG_BIN;PKG_TARGET;PARTITION_NODELABEL;UUID_CID;UUID_VID" "" ${ARGN})

	if(NOT ARG_TARGET OR NOT ARG_PKG_BIN OR NOT ARG_PARTITION_NODELABEL OR NOT ARG_UUID_CID OR NOT ARG_UUID_VID)
		message(FATAL_ERROR
			"nrf_model_mcuboot_sign() requires TARGET, PKG_BIN, PARTITION_NODELABEL, UUID_CID, UUID_VID")
	endif()

	if(NOT DEFINED IMGTOOL)
		message(FATAL_ERROR
			"nrf_model_mcuboot_sign(${ARG_TARGET}): IMGTOOL not found (MCUboot module missing?)")
	endif()

	set(keyfile "${CONFIG_MCUBOOT_SIGNATURE_KEY_FILE}")
	string(CONFIGURE "${keyfile}" keyfile)
	if(NOT keyfile OR NOT EXISTS "${keyfile}")
		message(FATAL_ERROR
			"nrf_model_mcuboot_sign(${ARG_TARGET}): CONFIG_MCUBOOT_SIGNATURE_KEY_FILE "
			"not set or missing (${keyfile})")
	endif()

	dt_nodelabel(slot_path NODELABEL ${ARG_PARTITION_NODELABEL})
	if(NOT slot_path)
		message(FATAL_ERROR
			"nrf_model_mcuboot_sign(${ARG_TARGET}): no devicetree node labelled "
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

	set(imgtool_sign_base ${PYTHON_EXECUTABLE} ${IMGTOOL} sign
		--version ${CONFIG_MCUBOOT_IMGTOOL_SIGN_VERSION}
		--header-size 32
		--pad-header
		--pad
		${imgtool_hash_arg}
		--slot-size ${slot_size}
		--align ${write_block_size}
		--rom-fixed ${slot_addr}
		--vid "${ARG_UUID_VID}"
		--cid "${ARG_UUID_CID}"
		-k "${keyfile}"
	)

	set(deps ${ARG_PKG_BIN})
	if(ARG_PKG_TARGET)
		list(APPEND deps ${ARG_PKG_TARGET})
	endif()

	set(outputs ${out_base}.signed.bin ${out_base}.signed.hex)

	if(ARG_PROVISION_CONFIRM)
		# Dual-slot: unconfirmed .bin for SMP OTA; confirmed .hex for first flash to slot2.
		add_custom_command(
			OUTPUT ${outputs}
			COMMAND ${imgtool_sign_base} ${ARG_PKG_BIN} ${out_base}.signed.bin
			COMMAND ${imgtool_sign_base} --confirm --hex-addr ${slot_addr}
				${ARG_PKG_BIN} ${out_base}.signed.hex
			DEPENDS ${deps}
			COMMENT "model_ota: signing ${ARG_TARGET} model for MCUboot slot2 (dual-slot)"
		)
	else()
		# Single-slot: one signed image for provision and SMP (in-place overwrite, no revert).
		add_custom_command(
			OUTPUT ${outputs}
			COMMAND ${imgtool_sign_base} ${ARG_PKG_BIN} ${out_base}.signed.bin
			COMMAND ${imgtool_sign_base} --hex-addr ${slot_addr}
				${ARG_PKG_BIN} ${out_base}.signed.hex
			DEPENDS ${deps}
			COMMENT "model_ota: signing ${ARG_TARGET} model for MCUboot slot2 (single-slot)"
		)
	endif()

	add_custom_target(${ARG_TARGET}_model_mcuboot_signed ALL DEPENDS ${outputs})
endfunction()
