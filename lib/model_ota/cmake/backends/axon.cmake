#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

# Axon model metadata and firmware-side helpers:
#   model_ota_axon_add_probe()
#   model_ota_axon_inspect()
#   model_ota_axon_add_app_stub()
#   model_ota_axon_add_binding_table()

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/../common.cmake)

set(MODEL_OTA_AXON_PROBE_SRC ${MODEL_OTA_LIB_DIR}/slot/axon/model_ota_axon_probe.c)
set(MODEL_OTA_AXON_SLOT_SRC ${MODEL_OTA_LIB_DIR}/slot/axon/model_ota_axon_slot.c)
set(MODEL_OTA_AXON_APP_STUB_SRC ${MODEL_OTA_LIB_DIR}/slot/axon/model_ota_axon_app_stub.c)
set(MODEL_OTA_AXON_BINDING_TABLE_SRC ${MODEL_OTA_LIB_DIR}/slot/axon/model_ota_axon_binding_table.S)
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
# Creates the custom target model_ota_axon_metadata_<id> and sets <id>_AXON_PRIVATE_HEADER,
# <id>_AXON_PUBLIC_HEADER and <id>_AXON_KEEP_JSON in the caller's scope to the generated file
# paths.
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
  set(keep_json ${AI_WORK_DIR}/axon_keep.json)

  set(inspect_cmd
    ${PYTHON_EXECUTABLE} ${MODEL_OTA_AXON_ELF} inspect
    --probe ${AI_PROBE}
    --header-name ${AI_HEADER_NAME}
    --model-id ${AI_TARGET}
    --private-header ${private_h}
    --public-header ${public_h}
    --keep-json ${keep_json}
  )
  if(AI_PERSISTENT_VARS_CAP)
    list(APPEND inspect_cmd --persistent-vars-cap ${AI_PERSISTENT_VARS_CAP})
  endif()
  if(AI_ALLOCATE_PACKED_OUTPUT)
    list(APPEND inspect_cmd --allocate-packed-output)
  endif()

  add_custom_command(
    OUTPUT ${private_h} ${public_h} ${keep_json}
    COMMAND ${inspect_cmd}
    DEPENDS ${AI_PROBE} ${MODEL_OTA_AXON_ELF}
            ${MODEL_OTA_TOOLS_DIR}/elf_symbols.py ${MODEL_OTA_TOOLS_DIR}/model_contract.py
    COMMENT "Inspecting Axon model metadata (${AI_TARGET})"
    VERBATIM
  )
  add_custom_target(model_ota_axon_metadata_${AI_TARGET}
                    DEPENDS ${private_h} ${public_h} ${keep_json})

  set(${AI_TARGET}_AXON_PRIVATE_HEADER ${private_h} PARENT_SCOPE)
  set(${AI_TARGET}_AXON_PUBLIC_HEADER ${public_h} PARENT_SCOPE)
  set(${AI_TARGET}_AXON_KEEP_JSON ${keep_json} PARENT_SCOPE)
endfunction()

#
# Add the firmware-owned storage of one Axon model slot as the object library
# model_ota_axon_storage_<id>, linked into the application. An object library, so the storage is
# in the link wherever the binding table that references it lands in the archive order.
#
# Usage:
#   model_ota_axon_add_app_stub(TARGET <id> WORK_DIR <dir>)
#
# TARGET <id>               : OTA slot ID, from model_ota_axon_inspect()
# WORK_DIR <dir>            : WORK_DIR passed to model_ota_axon_inspect()
#
function(model_ota_axon_add_app_stub)
  set(oneValueArgs TARGET WORK_DIR)
  cmake_parse_arguments(AS "" "${oneValueArgs}" "" ${ARGN})

  zephyr_check_no_arguments(model_ota_axon_add_app_stub ${AS_UNPARSED_ARGUMENTS})
  zephyr_check_arguments_required_all(model_ota_axon_add_app_stub AS TARGET WORK_DIR)

  set(lib model_ota_axon_storage_${AS_TARGET})

  add_library(${lib} OBJECT ${MODEL_OTA_AXON_APP_STUB_SRC})
  target_link_libraries(${lib} PRIVATE zephyr_interface)
  target_include_directories(${lib} PRIVATE ${AS_WORK_DIR})
  add_dependencies(${lib} zephyr_generated_headers model_ota_axon_metadata_${AS_TARGET})
  target_link_libraries(app PRIVATE ${lib})
endfunction()

#
# Add the firmware binding table shared by all Axon model slots as the static library
# model_ota_axon_binding_table, linked into the application. Call once, after every slot.
#
# Usage:
#   model_ota_axon_add_binding_table(WORK_DIR <dir> KEEP_JSON <file>... [DEPENDS <target>...])
#
# WORK_DIR <dir>            : Output directory for the generated binding table header
# KEEP_JSON <file>...       : <id>_AXON_KEEP_JSON of every slot, from model_ota_axon_inspect()
# DEPENDS <target>...       : Targets that generate the keep JSON files
#
function(model_ota_axon_add_binding_table)
  set(oneValueArgs WORK_DIR)
  set(multiValueArgs KEEP_JSON DEPENDS)
  cmake_parse_arguments(BT "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  zephyr_check_no_arguments(model_ota_axon_add_binding_table ${BT_UNPARSED_ARGUMENTS})
  zephyr_check_arguments_required_all(model_ota_axon_add_binding_table BT WORK_DIR KEEP_JSON)

  set(binding_h ${BT_WORK_DIR}/model_ota_axon_binding_table.h)
  set(binding_cmd
    ${PYTHON_EXECUTABLE} ${MODEL_OTA_AXON_ELF} binding-table
    -o ${binding_h}
  )
  foreach(keep_json ${BT_KEEP_JSON})
    list(APPEND binding_cmd --keep-json ${keep_json})
  endforeach()

  add_custom_command(
    OUTPUT ${binding_h}
    COMMAND ${binding_cmd}
    DEPENDS ${BT_KEEP_JSON} ${MODEL_OTA_AXON_ELF} ${MODEL_OTA_TOOLS_DIR}/model_contract.py
    COMMENT "Generating Axon binding table"
    VERBATIM
  )
  add_custom_target(model_ota_axon_binding_table_header DEPENDS ${binding_h})
  if(BT_DEPENDS)
    add_dependencies(model_ota_axon_binding_table_header ${BT_DEPENDS})
  endif()

  set(lib model_ota_axon_binding_table)

  add_library(${lib} STATIC ${MODEL_OTA_AXON_BINDING_TABLE_SRC})
  target_link_libraries(${lib} PRIVATE zephyr_interface)
  target_include_directories(${lib} PRIVATE ${BT_WORK_DIR})
  add_dependencies(${lib} zephyr_generated_headers model_ota_axon_binding_table_header)
  target_link_libraries(app PRIVATE ${lib})
endfunction()
