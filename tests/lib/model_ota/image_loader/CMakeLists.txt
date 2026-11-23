#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(test_model_ota_image_loader)

# The Axon driver needs the NPU, so the loader is built here against a faked model validation.
target_sources(app PRIVATE
  src/test_model_image_common.c
  src/test_model_image_axon.c
  ${ZEPHYR_EDGE_AI_MODULE_DIR}/lib/model_ota/loader/model_image_axon.c
)
target_include_directories(app PRIVATE ${ZEPHYR_EDGE_AI_MODULE_DIR}/lib/model_ota/include)
