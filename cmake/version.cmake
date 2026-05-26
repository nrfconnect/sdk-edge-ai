#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Generates edge_ai_version.h for C code using Zephyr's canonical
# gen_version_h.cmake script.  edge-ai/VERSION is the single source of truth;
# the same file is parsed by Kconfig (via orsource) and by
# scripts/version_handling/get_version.py (for documentation).
#
# The generated header is placed at:
#   ${PROJECT_BINARY_DIR}/include/generated/edge_ai_version.h
#
# Include it in application code with:
#   #include <edge_ai_version.h>
#
# Exported macros (PREFIX = EDGE_AI):
#   EDGE_AIVERSION            - combined 32-bit version (major<<24 | minor<<16 | patch<<8 | tweak)
#   EDGE_AI_VERSION_NUMBER    - 24-bit version number (major<<16 | minor<<8 | patch)
#   EDGE_AI_VERSION_MAJOR
#   EDGE_AI_VERSION_MINOR
#   EDGE_AI_PATCHLEVEL
#   EDGE_AI_VERSION_TWEAK
#   EDGE_AI_VERSION_STRING    - e.g. "2.1.0"
#   EDGE_AI_VERSION_EXTENDED_STRING - e.g. "2.1.0+0"
#   EDGE_AI_BUILD_VERSION     - git describe string (set at build time)

set(_edge_ai_dir ${ZEPHYR_CURRENT_MODULE_DIR})
set(_edge_ai_version_file ${_edge_ai_dir}/VERSION)

# Regenerate the header whenever a new commit is made in the edge-ai repo.
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --absolute-git-dir
    WORKING_DIRECTORY ${_edge_ai_dir}
    OUTPUT_VARIABLE _edge_ai_git_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _rc
  )
  if(_rc EQUAL 0 AND EXISTS "${_edge_ai_git_dir}/index")
    set(_edge_ai_git_dependency ${_edge_ai_git_dir}/index)
  endif()
  unset(_edge_ai_git_dir)
  unset(_rc)
endif()

add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/include/generated/edge_ai_version.h
  COMMAND ${CMAKE_COMMAND}
    -DZEPHYR_BASE=${ZEPHYR_BASE}
    -DOUT_FILE=${PROJECT_BINARY_DIR}/include/generated/edge_ai_version.h
    -DVERSION_TYPE=EDGE_AI
    -DVERSION_FILE=${_edge_ai_version_file}
    -P ${ZEPHYR_BASE}/cmake/gen_version_h.cmake
  DEPENDS ${_edge_ai_version_file} ${_edge_ai_git_dependency}
)

add_custom_target(
  edge_ai_version_h
  DEPENDS ${PROJECT_BINARY_DIR}/include/generated/edge_ai_version.h
)

add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/include/generated/edge_ai_commit.h
  COMMAND ${CMAKE_COMMAND}
    -DZEPHYR_BASE=${ZEPHYR_BASE}
    -DNRF_DIR=${ZEPHYR_NRF_MODULE_DIR}
    -DOUT_FILE=${PROJECT_BINARY_DIR}/include/generated/edge_ai_commit.h
    -DCOMMIT_TYPE=EDGE_AI
    -DCOMMIT_PATH=${_edge_ai_dir}
    -P ${ZEPHYR_NRF_MODULE_DIR}/cmake/gen_commit_h.cmake
  DEPENDS ${_edge_ai_git_dependency}
)

add_custom_target(
  edge_ai_commit_h
  DEPENDS ${PROJECT_BINARY_DIR}/include/generated/edge_ai_commit.h
)

# Ensure both headers are ready before version_h (and therefore zephyr_interface).
add_dependencies(version_h edge_ai_version_h edge_ai_commit_h)

unset(_edge_ai_dir)
unset(_edge_ai_version_file)
unset(_edge_ai_git_dependency)
