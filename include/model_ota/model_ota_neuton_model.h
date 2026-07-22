/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef MODEL_OTA_NEUTON_MODEL_H_
#define MODEL_OTA_NEUTON_MODEL_H_

/**
 * @file
 * @brief Per-model compile-time switches for OTA-wired Neuton generated sources.
 *
 * These are CMake compile definitions, not Kconfig options:
 *
 * - @ref MODEL_OTA_NEUTON_WIRED — set to 1 by model_ota_neuton_app_stub.c when the model is
 *   built into an OTA static library (runtime load path, payload discarded by the linker).
 *   Undefined when the model is baked into the app or compiled for a partition image stub.
 *
 * - @ref MODEL_OTA_NEUTON_MAX_NEURONS — neuron scratch buffer capacity for that wired model;
 *   passed per model by model_ota_neuton_wire(MAX_NEURONS ...).
 *
 * - @ref MODEL_OTA_NEUTON_MODEL_SRC — basename of the generated nrf_edgeai_user_model.c to
 *   #include from a stub translation unit.
 */

#endif /* MODEL_OTA_NEUTON_MODEL_H_ */
