#
# Copyright (c) 2025 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
if(CONFIG_CPU_CORTEX_M4)
    message(STATUS "Building for Cortex-M4 CPU")
    add_subdirectory(cortex-m4)
elseif(CONFIG_CPU_CORTEX_M33)
    message(STATUS "Building for Cortex-M33 CPU")
    add_subdirectory(cortex-m33)
else()
    message(FATAL_ERROR "Edge AI is not supported on the selected CPU architecture.")
endif()