#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(test_nrf_edgeai_obsv_memfault)

zephyr_include_directories(config)

# Private header for the Memfault transport, exposed only to this test.
set(OBSV_MFLT_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/nrf_edgeai_obsv_memfault)
target_include_directories(app PRIVATE ${OBSV_MFLT_LIB})

# Memfault SDK include path — needed for CDR types without enabling the full
# Memfault Kconfig (which would compile arch-specific panics/coredump sources
# incompatible with native_sim). Only the header path is required; all
# Memfault function implementations are stubbed in src/memfault_mock.c.
zephyr_include_directories(${ZEPHYR_MEMFAULT_FIRMWARE_SDK_MODULE_DIR}/components/include)

# CONFIG_NRF_EDGEAI_OBSV_EXTRA_ENCODE_BYTES is only defined when
# CONFIG_NRF_EDGEAI_OBSV_ENCODE=y. Provide a zero default so the size macros
# in nrf_edgeai_obsv_encode_sizes.h compile when the encoder is stubbed out.
zephyr_compile_definitions(CONFIG_NRF_EDGEAI_OBSV_EXTRA_ENCODE_BYTES=0)

target_sources(app PRIVATE
  src/main.c
  src/memfault_mock.c
  src/obsv_mock.c
)
