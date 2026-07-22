/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * OTA-wired Neuton model app-side stub (one translation unit per static library).
 *
 * model_ota_neuton_wire() compiles this file instead of nrf_edgeai_user_model.c directly:
 * sets MODEL_OTA_NEUTON_WIRED, MODEL_OTA_NEUTON_MAX_NEURONS and MODEL_OTA_NEUTON_MODEL_SRC,
 * includes model_ota/model_image.h, then #includes the generated model source (Axon-style).
 */

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

#define MODEL_OTA_NEUTON_XSTR(s) #s
#define MODEL_OTA_NEUTON_STR(s)  MODEL_OTA_NEUTON_XSTR(s)

#include MODEL_OTA_NEUTON_STR(MODEL_OTA_NEUTON_MODEL_SRC)
