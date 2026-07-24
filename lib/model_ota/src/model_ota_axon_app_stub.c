/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Fixed app-side stub for an OTA-wired Axon model.
 *
 * model_ota_axon_model() force-includes a private header generated from the
 * model probe. It identifies the generated model header and the app-owned
 * storage that must remain in the application image.
 */

#include "model_ota_stub_macros.h"

#if !defined(MODEL_OTA_AXON_CONFIG_VERSION) || (MODEL_OTA_AXON_CONFIG_VERSION != 1)
#error "Unsupported or missing Axon OTA configuration"
#endif

#ifndef MODEL_OTA_AXON_HEADER
#error "MODEL_OTA_AXON_HEADER is missing"
#endif

#if (MODEL_OTA_AXON_PACKED_OUTPUT_BYTES > 0) && MODEL_OTA_AXON_PACKED_OUTPUT_ALLOC
#define NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER 1
#endif

#define NRF_AXON_MODEL_APP_STORAGE extern

#include <assert.h>
#include <model_ota/model_image.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_driver.h>

#include MODEL_OTA_AXON_HEADER

#if (MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED > 0)
#ifndef MODEL_OTA_AXON_PERSISTENT_VARS_SYM
#error "MODEL_OTA_AXON_PERSISTENT_VARS_SYM must be set when the model uses persistent vars"
#endif

#if (MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED > MODEL_OTA_AXON_PERSISTENT_VARS_CAP)
#error "MODEL_OTA_AXON_PERSISTENT_VARS_CAP too small for wired Axon model"
#endif

int32_t MODEL_OTA_AXON_PERSISTENT_VARS_SYM[MODEL_OTA_AXON_PERSISTENT_VARS_CAP];
#endif

#if (MODEL_OTA_AXON_PACKED_OUTPUT_BYTES > 0) && MODEL_OTA_AXON_PACKED_OUTPUT_ALLOC
/*
 * Opt-in (model_ota_axon_model(ALLOCATE_PACKED_OUTPUT)): app-owned storage for the
 * model's packed-output buffer, kept alive via model_ota_axon_keep_refs.S and wired
 * into the linked partition image via the generated PROVIDE() linker fragment.
 */
#ifndef MODEL_OTA_AXON_PACKED_OUTPUT_SYM
#error "MODEL_OTA_AXON_PACKED_OUTPUT_SYM must be set when the model uses a packed output buffer"
#endif

uint32_t MODEL_OTA_AXON_PACKED_OUTPUT_SYM[MODEL_OTA_AXON_PACKED_OUTPUT_BYTES / sizeof(uint32_t)];
#else
/*
 * Default: no app-side packed-output buffer. The OTA partition image links without
 * NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER, so its model struct's packed_output_buf
 * is NULL. Callers that need a packed decode buffer size it themselves using the
 * generated MODEL_OTA_AXON_<ID>_PACKED_OUTPUT_BYTES macro.
 */
#endif
