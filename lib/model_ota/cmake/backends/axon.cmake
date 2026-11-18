#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

# Axon model metadata helpers:
#   model_ota_axon_add_probe()
#   model_ota_axon_inspect()

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/../common.cmake)

set(MODEL_OTA_AXON_PROBE_SRC ${MODEL_OTA_LIB_DIR}/slot/axon/model_ota_axon_probe.c)
set(MODEL_OTA_AXON_ELF ${MODEL_OTA_TOOLS_DIR}/axon_elf.py)

#
# Compile the ELF probe for a generated Axon model header, with the application's C flags.
#
# Usage:
#   model_ota_axon_add_probe(OUT_OBJ <var> WORK_DIR <dir> HEADER <header>)
#
# OUT_OBJ <var>     : Return variable for the path of the probe object
# WORK_DIR <dir>    : Directory for the probe object and its dependency file
# HEADER <header>   : Absolute path of the generated Axon model header
#
function(model_ota_axon_add_probe)
  set(oneValueArgs OUT_OBJ WORK_DIR HEADER)
  cmake_parse_arguments(AP "" "${oneValueArgs}" "" ${ARGN})

  zephyr_check_no_arguments(model_ota_axon_add_probe ${AP_UNPARSED_ARGUMENTS})
  zephyr_check_arguments_required_all(model_ota_axon_add_probe AP OUT_OBJ WORK_DIR HEADER)

  get_filename_component(header_name ${AP_HEADER} NAME)
  model_ota_zephyr_c_compile_flags(zephyr_cflags)

  set(probe_o ${AP_WORK_DIR}/axon_probe.o)
  set(probe_d ${AP_WORK_DIR}/axon_probe.d)
  add_custom_command(
    OUTPUT ${probe_o}
    COMMAND ${CMAKE_C_COMPILER}
            -c ${MODEL_OTA_AXON_PROBE_SRC}
            -o ${probe_o}
            -MMD -MF ${probe_d}
            ${zephyr_cflags}
            -DMODEL_OTA_AXON_HEADER=\"${AP_HEADER}\"
    COMMAND_EXPAND_LISTS
    DEPENDS ${MODEL_OTA_AXON_PROBE_SRC} ${AP_HEADER}
    DEPFILE ${probe_d}
    COMMENT "Compiling Axon ELF probe (${header_name})"
    VERBATIM
  )
  set(${AP_OUT_OBJ} ${probe_o} PARENT_SCOPE)
endfunction()

#
# Run axon_elf.py inspect on a probe object and generate the model's metadata headers.
#
# Creates the custom target model_ota_axon_metadata_<id> and sets <id>_AXON_PRIVATE_HEADER and
# <id>_AXON_PUBLIC_HEADER in the caller's scope to the generated header paths.
#
# Usage:
#   model_ota_axon_inspect(TARGET <id> PROBE <obj> HEADER_NAME <name> WORK_DIR <dir>
#                          [PERSISTENT_VARS_CAP <n>] [ALLOCATE_PACKED_OUTPUT])
#
# TARGET <id>               : OTA slot ID
# PROBE <obj>               : Probe object from model_ota_axon_add_probe()
# HEADER_NAME <name>        : File name of the Axon model header the probe was compiled from
# WORK_DIR <dir>            : Output directory for the generated headers
# PERSISTENT_VARS_CAP <n>   : Firmware persistent-vars capacity in int32_t entries
#                             (default: what the model requires)
# ALLOCATE_PACKED_OUTPUT    : Firmware owns the model's packed-output buffer
#
function(model_ota_axon_inspect)
  set(options ALLOCATE_PACKED_OUTPUT)
  set(oneValueArgs TARGET PROBE HEADER_NAME WORK_DIR PERSISTENT_VARS_CAP)
  cmake_parse_arguments(AI "${options}" "${oneValueArgs}" "" ${ARGN})

  zephyr_check_arguments_required_all(model_ota_axon_inspect AI TARGET PROBE HEADER_NAME WORK_DIR)
  zephyr_check_no_arguments(model_ota_axon_inspect ${AI_UNPARSED_ARGUMENTS})

  set(private_h ${AI_WORK_DIR}/model_ota_axon_model_config.h)
  set(public_h ${AI_WORK_DIR}/include/model_ota/slots/${AI_TARGET}.h)

  set(inspect_cmd
    ${PYTHON_EXECUTABLE} ${MODEL_OTA_AXON_ELF} inspect
    --probe ${AI_PROBE}
    --header-name ${AI_HEADER_NAME}
    --model-id ${AI_TARGET}
    --private-header ${private_h}
    --public-header ${public_h}
  )
  if(AI_PERSISTENT_VARS_CAP)
    list(APPEND inspect_cmd --persistent-vars-cap ${AI_PERSISTENT_VARS_CAP})
  endif()
  if(AI_ALLOCATE_PACKED_OUTPUT)
    list(APPEND inspect_cmd --allocate-packed-output)
  endif()

  add_custom_command(
    OUTPUT ${private_h} ${public_h}
    COMMAND ${inspect_cmd}
    DEPENDS ${AI_PROBE} ${MODEL_OTA_AXON_ELF}
            ${MODEL_OTA_TOOLS_DIR}/elf_symbols.py ${MODEL_OTA_TOOLS_DIR}/model_contract.py
    COMMENT "Inspecting Axon model metadata (${AI_TARGET})"
    VERBATIM
  )
  add_custom_target(model_ota_axon_metadata_${AI_TARGET} DEPENDS ${private_h} ${public_h})

  set(${AI_TARGET}_AXON_PRIVATE_HEADER ${private_h} PARENT_SCOPE)
  set(${AI_TARGET}_AXON_PUBLIC_HEADER ${public_h} PARENT_SCOPE)
endfunction()
