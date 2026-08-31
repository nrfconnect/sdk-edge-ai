#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Shared helpers for model_ota Neuton/Axon partition-image builds.

include_guard(GLOBAL)

get_filename_component(_model_ota_tools ${CMAKE_CURRENT_LIST_DIR}/../../../tools/model_ota ABSOLUTE)
set(MODEL_OTA_TOOLS_DIR ${_model_ota_tools} CACHE INTERNAL "edge-ai model_ota host tools")

get_filename_component(_model_ota_lib ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
set(MODEL_OTA_LIB_DIR ${_model_ota_lib} CACHE INTERNAL "edge-ai model_ota library sources")

get_filename_component(_model_ota_module ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)
set(MODEL_OTA_MODULE_DIR ${_model_ota_module} CACHE INTERNAL "edge-ai module root")

# Hash of the solution ID. The preprocessor cannot hash a string literal, so the value is
# computed here and passed to the stubs as MODEL_OTA_SOLUTION_ID_HASH (see
# lib/model_ota/src/model_ota_scale_select.h).
#
# The ID comes from the SOLUTION_ID the caller passed, not from parsing EDGEAI_LAB_SOLUTION_ID_STR
# out of the generated source: it is the identity the build was configured for, and every helper on
# both sides of an update already takes it. Since it feeds the contract hash, a SOLUTION_ID that
# does not match the source's own ID makes the image incompatible rather than subtly wrong.
#
# The ID is passed as an argument rather than interpolated into a Python snippet, so a quote or
# backslash in it cannot alter what the interpreter runs; `--` keeps a leading dash an argument.
function(model_ota_solution_id_hash solution_id out_var)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${MODEL_OTA_TOOLS_DIR}/model_contract.py solution-id-hash
            -- "${solution_id}"
    OUTPUT_VARIABLE _hash
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  set(${out_var} ${_hash} PARENT_SCOPE)
endfunction()

# The application's own C flags, for compiling a throwaway probe translation unit with exactly the
# firmware's view of the headers and Kconfig.
function(model_ota_zephyr_c_compile_flags OUT_VAR)
  zephyr_get_include_directories_for_lang(C _inc)
  zephyr_get_system_include_directories_for_lang(C _sys)
  zephyr_get_compile_definitions_for_lang(C _def)
  zephyr_get_compile_options_for_lang(C _opt)
  set(${OUT_VAR} ${_opt} ${_inc} ${_sys} ${_def} PARENT_SCOPE)
endfunction()

# Compile the contract-hash probe for one model slot; see
# lib/model_ota/src/model_ota_contract_probe.c for why the value cannot be computed on the host.
#
# model_ota_contract_probe(OUT_OBJ <var> WORK_DIR <dir> FLAVOR <neuton|axon|axon_edgeai>
#                          IMAGE_BASE <addr> [MODEL_SRC <abs nrf_edgeai_user_model.c>]
#                          [SOLUTION_ID_HASH <u32>])
#
# MODEL_SRC and SOLUTION_ID_HASH go together and are required for the two solution flavours: those
# contracts cover the generated nrf_edgeai_t pipeline, which is only visible with the source in
# scope.
function(model_ota_contract_probe)
  cmake_parse_arguments(CP ""
    "OUT_OBJ;WORK_DIR;FLAVOR;IMAGE_BASE;MODEL_SRC;SOLUTION_ID_HASH" "" ${ARGN})

  if(NOT CP_OUT_OBJ OR NOT CP_WORK_DIR OR NOT CP_FLAVOR OR NOT CP_IMAGE_BASE)
    message(FATAL_ERROR
            "model_ota_contract_probe requires OUT_OBJ, WORK_DIR, FLAVOR and IMAGE_BASE")
  endif()

  if(CP_FLAVOR STREQUAL "neuton")
    set(_flavor_def MODEL_OTA_CONTRACT_PROBE_NEUTON)
  elseif(CP_FLAVOR STREQUAL "axon")
    set(_flavor_def MODEL_OTA_CONTRACT_PROBE_AXON)
  elseif(CP_FLAVOR STREQUAL "axon_edgeai")
    set(_flavor_def MODEL_OTA_CONTRACT_PROBE_AXON_EDGEAI)
  else()
    message(FATAL_ERROR "model_ota_contract_probe: unknown FLAVOR ${CP_FLAVOR}")
  endif()

  if(CP_MODEL_SRC AND NOT CP_SOLUTION_ID_HASH)
    message(FATAL_ERROR "model_ota_contract_probe requires SOLUTION_ID_HASH alongside MODEL_SRC")
  endif()
  if(NOT CP_MODEL_SRC AND NOT CP_FLAVOR STREQUAL "axon")
    message(FATAL_ERROR "model_ota_contract_probe: FLAVOR ${CP_FLAVOR} requires MODEL_SRC")
  endif()

  model_ota_zephyr_c_compile_flags(_zephyr_cflags)

  set(_obj ${CP_WORK_DIR}/contract_probe.o)
  set(_dep ${CP_WORK_DIR}/contract_probe.d)
  set(_src ${MODEL_OTA_LIB_DIR}/src/model_ota_contract_probe.c)
  set(_deps ${_src} zephyr_generated_headers)
  set(_extra_flags "")

  if(CP_MODEL_SRC)
    get_filename_component(_model_dir ${CP_MODEL_SRC} DIRECTORY)
    get_filename_component(_model_basename ${CP_MODEL_SRC} NAME)
    list(APPEND _extra_flags
         -I${_model_dir}
         -DMODEL_OTA_CONTRACT_PROBE_MODEL_SRC=${_model_basename}
         -DMODEL_OTA_SOLUTION_ID_HASH=${CP_SOLUTION_ID_HASH}u)
    list(APPEND _deps ${CP_MODEL_SRC})
  endif()

  file(MAKE_DIRECTORY ${CP_WORK_DIR})
  add_custom_command(
    OUTPUT ${_obj}
    COMMAND ${CMAKE_C_COMPILER}
            -c ${_src}
            -o ${_obj}
            -MMD -MF ${_dep}
            ${_zephyr_cflags}
            -I${MODEL_OTA_LIB_DIR}/src
            -I${MODEL_OTA_MODULE_DIR}/include
            -include ${CMAKE_CURRENT_BINARY_DIR}/zephyr/include/generated/zephyr/autoconf.h
            -D${_flavor_def}
            -DNRF_MODEL_PARTITION_ADDR=${CP_IMAGE_BASE}
            ${_extra_flags}
    DEPENDS ${_deps}
    DEPFILE ${_dep}
    COMMAND_EXPAND_LISTS
    COMMENT "Compiling model OTA contract probe (${CP_FLAVOR}, base ${CP_IMAGE_BASE})"
    VERBATIM
  )
  set(${CP_OUT_OBJ} ${_obj} PARENT_SCOPE)
endfunction()

# Pack "x.y.z" into major<<16 | minor<<8 | patch for @ref model_image_header.model_version.
function(model_ota_pack_version version_str out_var)
  string(REPLACE "." ";" ver_parts "${version_str}")
  list(LENGTH ver_parts ver_len)
  list(GET ver_parts 0 ver_major)
  set(ver_minor 0)
  set(ver_patch 0)
  if(ver_len GREATER 1)
    list(GET ver_parts 1 ver_minor)
  endif()
  if(ver_len GREATER 2)
    list(GET ver_parts 2 ver_patch)
  endif()
  math(EXPR ver_u32 "(${ver_major} << 16) | (${ver_minor} << 8) | ${ver_patch}")
  set(${out_var} ${ver_u32} PARENT_SCOPE)
endfunction()

# The nrf_edgeai_t parameter arrays a model image carries in its edgeai_params block. An OTA-wired
# solution initializes them from the image instead (MODEL_OTA_WIRED zeroes the generated
# initializers), so they are unreferenced in the app and can be dropped from its library.
set(MODEL_OTA_EDGEAI_PARAM_SECTIONS
    .rodata.INPUT_FEATURES_SCALE_MIN
    .rodata.INPUT_FEATURES_SCALE_MAX
    .rodata.EXTRACTED_FEATURES_SCALE_MIN
    .rodata.EXTRACTED_FEATURES_SCALE_MAX
    .rodata.FEATURES_EXTRACTION_ARGUMENTS
    .rodata.MODEL_OUTPUT_SCALE_MIN
    .rodata.MODEL_OUTPUT_SCALE_MAX
    .rodata.MODEL_AVERAGE_EMBEDDING
)

function(model_ota_regenerate_discard_fragment)
  set(frag "${CMAKE_CURRENT_BINARY_DIR}/model_ota/model_ota_discard.ld")

  get_property(wired GLOBAL PROPERTY model_ota_discard_wired)
  string(REPLACE ";" "\n *   " wired_comment "${wired}")

  get_property(discard_lines GLOBAL PROPERTY model_ota_discard_lines)
  string(REPLACE ";" "\n" discard_body "${discard_lines}")

  file(WRITE ${frag}
"/* Auto-generated by model_ota_discard_register(); do not edit.
 * Drops payload rodata from each OTA-wired model static library only (archive-scoped rules).
 * Placed via ROM_SECTIONS (before the generic rodata collector) so GNU ld's first-match rule
 * routes matching sections here. Non-wired models compiled into the app keep their payload.
 * OTA-wired model(s):
 *   ${wired_comment}
 */
/DISCARD/ :
{
${discard_body}
}
")

  get_property(registered GLOBAL PROPERTY model_ota_discard_registered)
  if(NOT registered)
    set_property(GLOBAL PROPERTY model_ota_discard_registered TRUE)
    zephyr_linker_sources(ROM_SECTIONS ${frag})
  endif()
endfunction()

# Drop the given input sections from one OTA-wired model static library.
#
# model_ota_discard_register(LIB <static-lib-target> DESCRIPTION <str> SECTIONS <section>...)
#
# The library must be compiled with -ffunction-sections -fdata-sections so each array lands in
# its own named input section.
function(model_ota_discard_register)
  cmake_parse_arguments(MD "" "LIB;DESCRIPTION" "SECTIONS" ${ARGN})

  if(NOT MD_LIB OR NOT MD_SECTIONS)
    message(FATAL_ERROR "model_ota_discard_register requires LIB and SECTIONS")
  endif()

  target_compile_options(${MD_LIB} PRIVATE -ffunction-sections -fdata-sections)

  set(archive_glob "*lib${MD_LIB}.a")
  foreach(section ${MD_SECTIONS})
    set_property(GLOBAL APPEND PROPERTY model_ota_discard_lines
                 "\t${archive_glob}:*(${section})")
  endforeach()
  set_property(GLOBAL APPEND PROPERTY model_ota_discard_wired "${MD_DESCRIPTION}")

  model_ota_regenerate_discard_fragment()
endfunction()
