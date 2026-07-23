/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Axon model partition-image stub (one translation unit, compiled once per model image).
 *
 * model_ota_axon_image() compiles this file with -DMODEL_OTA_AXON_IMAGE, -DMODEL_IMAGE_HEADER,
 * and -DMODEL_IMAGE_MODEL_SYM, then links the result at the partition base. App-owned pointer
 * fields are resolved from zephyr.elf via a generated PROVIDE() linker fragment.
 */

#include <stddef.h>
#include <stdint.h>
#include <assert.h>

#define NRF_AXON_MODEL_APP_STORAGE extern

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>

#ifndef NRF_MODEL_PARTITION_ADDR
#error "NRF_MODEL_PARTITION_ADDR must be defined when linking the Axon model image"
#endif

#ifndef MODEL_IMAGE_HEADER
#error "MODEL_IMAGE_HEADER must be defined when building the Axon model image stub"
#endif

#include MODEL_IMAGE_HEADER

#include "model_image_stub_axon_body.h"
