/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Neuton model partition-image stub (one translation unit, compiled once per model image).
 *
 * model_ota_neuton_image() sets MODEL_OTA_NEUTON_MODEL_SRC to the model source basename and
 * adds that model's directory to the include path, then compiles this file as an OBJECT library
 * (same pattern as tests/axon/inference/src/nrf_axon_app_test_nn_inference.c + AXON_MODEL_FILE_NAME).
 *
 * MODEL_OTA_NEUTON_WIRED is not set here, so the included nrf_edgeai_user_model.c emits the
 * compile-time model_instance_ descriptor and payload arrays; model_image_stub_body.h then emits
 * the partition header that roots the gc-sections link.
 */

#ifndef MODEL_OTA_NEUTON_MODEL_SRC
#error "MODEL_OTA_NEUTON_MODEL_SRC must be defined by model_ota_neuton_image()"
#endif

#define MODEL_OTA_NEUTON_XSTR(s) #s
#define MODEL_OTA_NEUTON_STR(s)  MODEL_OTA_NEUTON_XSTR(s)

#include MODEL_OTA_NEUTON_STR(MODEL_OTA_NEUTON_MODEL_SRC)
#include "model_image_stub_body.h"
