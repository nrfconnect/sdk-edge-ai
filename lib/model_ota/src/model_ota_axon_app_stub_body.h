/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * App-owned Axon RAM referenced by OTA-wired models. Included at the bottom of
 * model_ota_axon_app_stub.c after MODEL_OTA_AXON_HEADER pulls in the model.
 *
 * CMake sets MODEL_OTA_AXON_* from the model header and PERSISTENT_VARS_CAP from
 * model_ota_axon_wire() (defaults to required, like MODEL_OTA_NEUTON_MAX_NEURONS).
 */

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
