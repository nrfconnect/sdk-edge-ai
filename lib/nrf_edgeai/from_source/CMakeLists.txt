#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Build libnrf_edgeai from edge-ai-srcx sources (for development and model-OTA tests).

if(NOT NRF_EDGEAI_SRC_PATH)
	set(NRF_EDGEAI_SRC_PATH "$ENV{NRF_EDGEAI_SRC_PATH}")
endif()

if(NOT NRF_EDGEAI_SRC_PATH OR NOT IS_DIRECTORY "${NRF_EDGEAI_SRC_PATH}/lib/source")
	message(FATAL_ERROR
		"CONFIG_NRF_EDGEAI_FROM_SOURCE requires NRF_EDGEAI_SRC_PATH (CMake variable or "
		"environment variable) pointing at an edge-ai-srcx checkout.")
endif()

file(GLOB_RECURSE NRF_EDGEAI_SRCS
	"${NRF_EDGEAI_SRC_PATH}/lib/source/**/*.c"
)

zephyr_library()
zephyr_library_sources(${NRF_EDGEAI_SRCS})

zephyr_library_include_directories(BEFORE
	${NRF_EDGEAI_SRC_PATH}/lib/include
	${NRF_EDGEAI_SRC_PATH}/deps/third_party/kissfft
)

zephyr_library_compile_definitions(NRF_DSP_ENABLE_RUNTIME_CHECKS=1)
