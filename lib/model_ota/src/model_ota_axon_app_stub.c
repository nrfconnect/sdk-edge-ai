/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * OTA-wired Axon model app-side stub (one translation unit per static library).
 *
 * model_ota_axon_wire() sets MODEL_OTA_AXON_WIRED and buffer macros from the model header,
 * includes the compiler-generated model, then allocates app-owned RAM the model references.
 */

#include "model_ota_stub_macros.h"

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

#ifndef MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED
#define MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED 0
#endif

#ifndef MODEL_OTA_AXON_PERSISTENT_VARS_CAP
#define MODEL_OTA_AXON_PERSISTENT_VARS_CAP MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED
#endif

#ifndef MODEL_OTA_AXON_PACKED_OUTPUT_BYTES
#define MODEL_OTA_AXON_PACKED_OUTPUT_BYTES 0
#endif

#if (MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED > 0)
#ifndef MODEL_OTA_AXON_PERSISTENT_VARS_SYM
#error "MODEL_OTA_AXON_PERSISTENT_VARS_SYM must be set when the model uses persistent vars"
#endif

#if (MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED > MODEL_OTA_AXON_PERSISTENT_VARS_CAP)
#error "MODEL_OTA_AXON_PERSISTENT_VARS_CAP too small for wired Axon model"
#endif

int32_t MODEL_OTA_AXON_PERSISTENT_VARS_SYM[MODEL_OTA_AXON_PERSISTENT_VARS_CAP];
#endif

#if (MODEL_OTA_AXON_PACKED_OUTPUT_BYTES > 0)
#ifndef MODEL_OTA_AXON_PACKED_OUTPUT_SYM
#error "MODEL_OTA_AXON_PACKED_OUTPUT_SYM must be set when the model uses a packed output buffer"
#endif

uint32_t MODEL_OTA_AXON_PACKED_OUTPUT_SYM[MODEL_OTA_AXON_PACKED_OUTPUT_BYTES / sizeof(uint32_t)];
#endif
