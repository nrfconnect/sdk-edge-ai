#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

# Shared helpers for model_ota build steps.

include_guard(GLOBAL)

set(MODEL_OTA_TOOLS_DIR ${ZEPHYR_EDGE_AI_MODULE_DIR}/scripts/model_ota
    CACHE INTERNAL "edge-ai model_ota host tools")
set(MODEL_OTA_LIB_DIR ${ZEPHYR_EDGE_AI_MODULE_DIR}/lib/model_ota
    CACHE INTERNAL "edge-ai model_ota library sources")

# Size of the MCUboot header that precedes the model image in a model partition.
set(MODEL_OTA_IMAGE_OFFSET 32 CACHE INTERNAL "model_ota model image offset in the model partition")

# The application's own C flags, for compiling a throwaway probe translation unit with exactly the
# firmware's view of the headers and Kconfig.
function(model_ota_zephyr_c_compile_flags OUT_VAR)
  zephyr_get_include_directories_for_lang(C inc)
  zephyr_get_system_include_directories_for_lang(C sys)
  zephyr_get_compile_definitions_for_lang(C def)
  zephyr_get_compile_options_for_lang(C opt)
  set(${OUT_VAR} ${opt} ${inc} ${sys} ${def} PARENT_SCOPE)
endfunction()

#
# Add the loader of one model slot as the static library model_ota_slot_<id>, linked into the
# application.
#
# Usage:
#   model_ota_add_slot_library(TARGET <id> SOURCE <src> PARTITION_NODELABEL <label>
#                               [DEFINES <define>...] [INCLUDES <dir>...] [DEPENDS <target>...])
#
# TARGET <id>                  : Model slot ID, passed to the source as MODEL_OTA_SLOT_TARGET
# SOURCE <src>                 : Model slot loader source, compiled once for this model slot
# PARTITION_NODELABEL <label>  : Devicetree node label of the model partition of the model slot
# DEFINES <define>...          : Extra compile definitions
# INCLUDES <dir>...            : Extra include directories
# DEPENDS <target>...          : Targets that generate headers the model slot loader includes
#
function(model_ota_add_slot_library)
  set(oneValueArgs TARGET SOURCE PARTITION_NODELABEL)
  set(multiValueArgs DEFINES INCLUDES DEPENDS)
  cmake_parse_arguments(SL "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  zephyr_check_no_arguments(model_ota_add_slot_library ${SL_UNPARSED_ARGUMENTS})
  zephyr_check_arguments_required_all(model_ota_add_slot_library SL
                                      TARGET SOURCE PARTITION_NODELABEL)

  set(lib model_ota_slot_${SL_TARGET})

  add_library(${lib} STATIC ${SL_SOURCE})
  target_link_libraries(${lib} PRIVATE zephyr_interface)
  target_include_directories(${lib} PRIVATE
    ${MODEL_OTA_LIB_DIR}/include
    ${SL_INCLUDES}
  )
  target_compile_definitions(${lib} PRIVATE
    MODEL_OTA_SLOT_TARGET=${SL_TARGET}
    MODEL_OTA_PARTITION_NODELABEL=${SL_PARTITION_NODELABEL}
    MODEL_OTA_IMAGE_OFFSET=${MODEL_OTA_IMAGE_OFFSET}
    ${SL_DEFINES}
  )
  add_dependencies(${lib} zephyr_generated_headers ${SL_DEPENDS})
  target_link_libraries(app PRIVATE ${lib})
endfunction()
