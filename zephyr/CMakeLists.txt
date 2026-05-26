#
# Copyright (c) 2025 Nordic Semiconductor
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

# Generate edge_ai_version.h from edge-ai/VERSION using Zephyr's canonical
# gen_version_h.cmake.  Must be included before any target that may need the
# generated header (version_h pulls in edge_ai_version_h automatically).
include(${ZEPHYR_CURRENT_MODULE_DIR}/cmake/version.cmake)

zephyr_include_directories(${ZEPHYR_CURRENT_MODULE_DIR}/include)

add_subdirectory(${ZEPHYR_CURRENT_MODULE_DIR}/drivers drivers)
add_subdirectory(${ZEPHYR_CURRENT_MODULE_DIR}/lib lib)
add_subdirectory(${ZEPHYR_CURRENT_MODULE_DIR}/subsys subsys)
