#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(test_nrf_edgeai_obsv_memfault)

zephyr_include_directories(config)

# Memfault SDK headers are required for MemfaultMetricId. Implementations used
# by the heartbeat adapter are stubbed in src/memfault_mock.c.
zephyr_include_directories(${ZEPHYR_MEMFAULT_FIRMWARE_SDK_MODULE_DIR}/components/include)

target_sources(app PRIVATE
  src/heartbeat.c
  src/memfault_mock.c
)
