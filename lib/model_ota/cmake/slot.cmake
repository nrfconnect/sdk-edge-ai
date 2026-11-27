#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

# Model slot loader build steps.

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/common.cmake)

#
# Add the loader of one model slot as the static library model_ota_slot_<id> and linked into the
# model_ota library.
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
  set_target_properties(${lib} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/model_ota/${SL_TARGET}
  )
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
  target_link_libraries(model_ota PRIVATE ${lib})
endfunction()
