/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * OTA-wired Axon model app-side stub (one translation unit per static library).
 */

#ifndef MODEL_OTA_AXON_HEADER
#error "MODEL_OTA_AXON_HEADER must be defined by model_ota_axon_wire()"
#endif

#ifndef MODEL_OTA_AXON_WIRED
#error "MODEL_OTA_AXON_WIRED must be defined by model_ota_axon_wire()"
#endif

#define NRF_AXON_MODEL_APP_STORAGE extern

#include <assert.h>
#include <model_ota/model_image.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_driver.h>

#include MODEL_OTA_AXON_HEADER
#include "model_ota_axon_app_stub_body.h"
