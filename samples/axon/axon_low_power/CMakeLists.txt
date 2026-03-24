#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(axon_low_power)

target_include_directories(app PRIVATE src/instrumentation)

target_sources(app PRIVATE
    src/main.c
    src/mel_test_vector.c
    src/instrumentation/instrumentation.c
)

target_sources_ifdef(CONFIG_AXON_LOW_POWER_GPIO_TRACING app PRIVATE
    src/instrumentation/gpio_trace.c
)

target_sources_ifdef(CONFIG_AXON_LOW_POWER_LED_INDICATION app PRIVATE
    src/instrumentation/led.c
)
