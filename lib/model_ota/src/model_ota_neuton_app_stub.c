/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * OTA-wired Neuton model app-side stub (one translation unit per static library).
 *
 * model_ota_neuton_wire() sets MODEL_OTA_NEUTON_WIRED, MODEL_OTA_NEUTON_MAX_NEURONS and
 * MODEL_OTA_NEUTON_MODEL_SRC, includes model_image.h, then #includes the generated model
 * source. Payload arrays land in named input sections discarded from the app image by
 * archive-scoped linker rules.
 */

#include "model_ota_stub_macros.h"

#ifndef MODEL_OTA_NEUTON_MODEL_SRC
#error "MODEL_OTA_NEUTON_MODEL_SRC must be defined by model_ota_neuton_wire()"
#endif

#ifndef MODEL_OTA_NEUTON_MAX_NEURONS
#error "MODEL_OTA_NEUTON_MAX_NEURONS must be defined by model_ota_neuton_wire()"
#endif

#ifndef MODEL_OTA_NEUTON_WIRED
#error "MODEL_OTA_NEUTON_WIRED must be defined by model_ota_neuton_wire()"
#endif

#include <model_ota/model_image.h>
#include <zephyr/sys/util.h>

#include MODEL_OTA_STUB_STR(MODEL_OTA_NEUTON_MODEL_SRC)
