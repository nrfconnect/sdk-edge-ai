#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

include_guard(GLOBAL)

macro(model_ota_check_required_all prefix)
  foreach(required ${ARGN})
    if(NOT DEFINED ${prefix}_${required} OR "${${prefix}_${required}}" STREQUAL "")
      message(FATAL_ERROR
              "${CMAKE_CURRENT_FUNCTION}() missing a required argument: ${required}")
    endif()
  endforeach()
endmacro()

macro(model_ota_check_required_multi prefix arg)
  if(NOT ${prefix}_${arg})
    message(FATAL_ERROR
            "${CMAKE_CURRENT_FUNCTION}() missing a required argument: ${arg}")
  endif()
endmacro()

macro(model_ota_check_unparsed prefix)
  if(${prefix}_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
            "${CMAKE_CURRENT_FUNCTION}() given unknown arguments: "
            "${${prefix}_UNPARSED_ARGUMENTS}")
  endif()
endmacro()
