#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# nrf_axon_model_stub(): build-integrated "second-pass link" for an Axon model, producing a
# model_ota package (.bin/.hex) as part of a normal `west build`, with no separate "reference
# build" step. See doc/libraries/model_ota.rst for the full picture; in short:
#
#   1. The application is built normally, without the model's generated header (it loads the
#      model from flash via model_pkg_load_axon() instead) - but every app-owned symbol the
#      model header references (nrf_axon_interlayer_buffer, nrf_axon_nn_op_extension_*,
#      axon_model_<name>_persistent_vars) is kept alive in its ELF even though nothing in the
#      app's own sources references them, via toolchain_ld_force_undefined_symbols().
#   2. Once the application's zephyr.elf exists, this module extracts those symbols' real
#      addresses from it (tools/model_ota/extract_elf_syms.py) and compiles+links a tiny
#      "model stub" - just the model header itself, patched so it does not try to redefine any
#      of them (tools/model_ota/gen_axon_stub_fixups.py) - at the model_storage partition's own
#      address, with those addresses supplied via a generated PROVIDE() linker script.
#   3. tools/model_ota/package_model_axon.py then copies that model stub's own `.model_stub`
#      section verbatim as the package payload - its bytes are already correctly addressed for
#      every pointer field (flash-owned or app-owned), so no relocation is needed - and prepends
#      a small header (whose size the model stub itself reserved space for up front, see
#      model_stub_axon.c/.ld, so package_base lines up with zero host-side arithmetic).
#
# Usage (from a sample/application's CMakeLists.txt, after target_sources(app ...)):
#
#   include(${SDK_EDGE_AI_DIR}/lib/model_ota/cmake/nrf_axon_model_stub.cmake)
#   nrf_axon_model_stub(
#     TARGET hello_axon                    # unique within this build; used for filenames
#     MODEL_NAME hello_axon                # matches the header's model_<name> symbol suffix
#     HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/nrf_axon_model_hello_axon_.h
#     PARTITION_NODELABEL model_partition  # devicetree node the package is built for
#   )
#
# Known simplification: the model stub is compiled with a standalone compiler invocation (this
# module's own add_custom_command, not Zephyr's per-target flag propagation), so it does not
# automatically inherit the application's full compile flags (-mcpu, -mfpu, ...). This is safe
# for nrf_axon_nn_compiled_model_s's own layout (a plain, non-conditionally-compiled AAPCS
# struct - see axon_struct_layout.py's module comment), and package_model_axon.py's existing
# check_struct_size() cross-check catches it regardless if that ever changes. The two Kconfig
# values that *do* affect whether the header even compiles (NRF_AXON_INTERLAYER_BUFFER_SIZE,
# NRF_AXON_PSUM_BUFFER_SIZE - see include/axon/nrf_axon_platform.h) are passed through
# explicitly below.

set(NRF_AXON_MODEL_STUB_DIR ${CMAKE_CURRENT_LIST_DIR}/../stub)
set(MODEL_OTA_TOOLS_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../tools/model_ota)
set(MODEL_OTA_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../include)

function(nrf_axon_model_stub)
	cmake_parse_arguments(ARG "" "TARGET;MODEL_NAME;HEADER;PARTITION_NODELABEL;VERSION" "" ${ARGN})

	if(NOT ARG_TARGET OR NOT ARG_MODEL_NAME OR NOT ARG_HEADER OR NOT ARG_PARTITION_NODELABEL)
		message(FATAL_ERROR
			"nrf_axon_model_stub() requires TARGET, MODEL_NAME, HEADER and "
			"PARTITION_NODELABEL")
	endif()
	if(NOT ARG_VERSION)
		set(ARG_VERSION "1.0.0")
	endif()

	set(work_dir ${CMAKE_CURRENT_BINARY_DIR}/model_stub_${ARG_TARGET})
	file(MAKE_DIRECTORY ${work_dir})

	# --- Step 0: query package_model_axon.py's own header size, so the model stub can
	# reserve exactly that many bytes up front (see model_stub_axon.c/.ld) - keeping the
	# header layout defined in exactly one place (model_pkg.h/package_model_axon.py).
	execute_process(
		COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_TOOLS_DIR}/package_model_axon.py
			--print-header-len
		OUTPUT_VARIABLE header_len_raw
		RESULT_VARIABLE header_len_rc
	)
	if(NOT header_len_rc EQUAL 0)
		message(FATAL_ERROR
			"nrf_axon_model_stub(${ARG_TARGET}): failed to query the package header size")
	endif()
	string(STRIP "${header_len_raw}" header_len)

	# --- Step 1: discover app-owned symbols now (configure time), so they can be force-kept
	# alive in the application link below; a build-time copy is regenerated as well, in case
	# HEADER's content changes without a fresh CMake configure.
	execute_process(
		COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_TOOLS_DIR}/gen_axon_stub_fixups.py
			--header ${ARG_HEADER} --print-symbols
		OUTPUT_VARIABLE app_symbols_raw
		RESULT_VARIABLE gen_fixups_rc
	)
	if(NOT gen_fixups_rc EQUAL 0)
		message(FATAL_ERROR
			"nrf_axon_model_stub(${ARG_TARGET}): failed to inspect ${ARG_HEADER}")
	endif()
	string(STRIP "${app_symbols_raw}" app_symbols_raw)
	string(REPLACE "\n" ";" app_symbols "${app_symbols_raw}")

	foreach(symbol ${app_symbols})
		toolchain_ld_force_undefined_symbols(${symbol})
	endforeach()

	set(symbols_file ${work_dir}/app_symbols.txt)
	set(patched_header ${work_dir}/model_header_patched.h)

	add_custom_command(
		OUTPUT ${symbols_file} ${patched_header}
		COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_TOOLS_DIR}/gen_axon_stub_fixups.py
			--header ${ARG_HEADER}
			--symbols-out ${symbols_file}
			--patched-header-out ${patched_header}
		DEPENDS ${ARG_HEADER} ${MODEL_OTA_TOOLS_DIR}/gen_axon_stub_fixups.py
		COMMENT "model_ota: generating ${ARG_TARGET} model stub fixups"
	)

	# --- Step 2: once the application's own zephyr.elf exists, extract real addresses for
	# those symbols and compile+link the model stub at the model_storage partition's address.
	set(app_elf ${PROJECT_BINARY_DIR}/zephyr/zephyr.elf)
	set(provide_script ${work_dir}/app_symbols.ld)

	add_custom_command(
		OUTPUT ${provide_script}
		COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_TOOLS_DIR}/extract_elf_syms.py
			--nm ${CMAKE_NM} --elf ${app_elf} --symbols ${symbols_file}
			--output ${provide_script}
		DEPENDS ${symbols_file} ${app_elf} ${MODEL_OTA_TOOLS_DIR}/extract_elf_syms.py
		COMMENT "model_ota: extracting ${ARG_TARGET} app symbol addresses"
	)

	dt_nodelabel(partition_path NODELABEL ${ARG_PARTITION_NODELABEL})
	if(NOT partition_path)
		message(FATAL_ERROR
			"nrf_axon_model_stub(${ARG_TARGET}): devicetree has no node labelled "
			"'${ARG_PARTITION_NODELABEL}' - see boards/*.overlay")
	endif()
	dt_reg_addr(partition_addr PATH "${partition_path}")
	dt_reg_size(partition_size PATH "${partition_path}")

	# GNU ld does not merge two independently-provided `-T scriptfile` SECTIONS commands the
	# way one might expect (each unmarked SECTIONS command is meant to fully replace the
	# default script, so passing two leaves the *second* one's - here, the PROVIDE()-only
	# script's - lack of a SECTIONS command falling back to ld's built-in default layout,
	# silently discarding model_stub_axon.ld's placement entirely). Concatenating them into
	# one script file side-steps that ambiguity: a single SECTIONS command followed by
	# top-level PROVIDE() statements is unambiguous.
	set(combined_ld ${work_dir}/model_stub_combined.ld)

	add_custom_command(
		OUTPUT ${combined_ld}
		COMMAND ${CMAKE_COMMAND} -E cat
			${NRF_AXON_MODEL_STUB_DIR}/model_stub_axon.ld ${provide_script}
			> ${combined_ld}
		DEPENDS ${NRF_AXON_MODEL_STUB_DIR}/model_stub_axon.ld ${provide_script}
		COMMENT "model_ota: combining ${ARG_TARGET} model stub linker script"
	)

	set(stub_obj ${work_dir}/${ARG_TARGET}_model_stub.o)
	set(stub_elf ${work_dir}/${ARG_TARGET}_model_stub.elf)

	# -mcpu/-mfloat-abi are deliberately not passed here: every field in
	# nrf_axon_nn_compiled_model_s (see axon_struct_layout.py) is an integer or pointer type,
	# never float/double, so AAPCS lays it out identically regardless of the target's FPU
	# configuration; and this stub contains no executable code at all (model_stub_axon.c
	# defines nothing, it only pulls in a model header's data), so instruction set selection
	# does not apply either.
	add_custom_command(
		OUTPUT ${stub_obj}
		COMMAND ${CMAKE_C_COMPILER}
			-mthumb
			-DNRF_AXON_INTERLAYER_BUFFER_SIZE=${CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE}
			-DNRF_AXON_PSUM_BUFFER_SIZE=${CONFIG_NRF_AXON_PSUM_BUFFER_SIZE}
			-DMODEL_STUB_HEADER_RESERVE_LEN=${header_len}
			-I${MODEL_OTA_INCLUDE_DIR} -I${work_dir}
			-c -o ${stub_obj}
			${NRF_AXON_MODEL_STUB_DIR}/model_stub_axon.c
		DEPENDS ${patched_header}
			${NRF_AXON_MODEL_STUB_DIR}/model_stub_axon.c
		COMMENT "model_ota: compiling ${ARG_TARGET} model stub"
	)

	# Linked with the bare linker (CMAKE_LINKER), not the compiler driver
	# (CMAKE_C_COMPILER): GCC's own spec files inject their own default `-T` linker script
	# (e.g. picolibc.ld) ahead of any user-supplied one, and GNU ld does not fully replace
	# an earlier unmarked SECTIONS command with a later one the way passing a single `-T`
	# would suggest - it silently keeps placing .rodata/.data via the *first* script's
	# rules, discarding combined_ld's placement. The bare linker has no such built-in
	# default to contend with.
	add_custom_command(
		OUTPUT ${stub_elf}
		COMMAND ${CMAKE_LINKER}
			-T ${combined_ld}
			--defsym=MODEL_STUB_ADDR=${partition_addr}
			-o ${stub_elf}
			${stub_obj}
		DEPENDS ${stub_obj} ${combined_ld}
		COMMENT "model_ota: linking ${ARG_TARGET} model stub at ${partition_addr}"
	)

	# --- Step 3: package the model stub's ELF exactly like a reference build's, since its
	# bytes are already correctly addressed for every pointer field.
	set(pkg_base ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_model_pkg)

	add_custom_command(
		OUTPUT ${pkg_base}.bin ${pkg_base}.hex
		COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_TOOLS_DIR}/package_model_axon.py
			--elf ${stub_elf} --model-name ${ARG_MODEL_NAME} --version ${ARG_VERSION}
			--address ${partition_addr} --partition-size ${partition_size}
			-o ${pkg_base}
		DEPENDS ${stub_elf} ${MODEL_OTA_TOOLS_DIR}/package_model_axon.py
		COMMENT "model_ota: packaging ${ARG_TARGET} (${ARG_MODEL_NAME})"
	)

	add_custom_target(${ARG_TARGET}_model_pkg ALL DEPENDS ${pkg_base}.bin ${pkg_base}.hex)
endfunction()
