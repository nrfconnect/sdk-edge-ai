/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef MODEL_OTA_MODEL_OTA_NEUTON_H_
#define MODEL_OTA_MODEL_OTA_NEUTON_H_

/**
 * @file
 * @brief Neuton model-only OTA helpers for app-side wired models.
 *
 * Wired models are built from model_ota_neuton_wired.c.in (see model_ota_neuton.cmake).
 * Generated nrf_edgeai_user_model.c stays agnostic: it honors the hooks below when a wired
 * translation unit defines them before #include.
 */

#include <stdint.h>

#include <nrf_edgeai/nrf_edgeai.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Declare nrf_edgeai_load_user_model_<solution_id>() from a wired static library. */
#define MODEL_OTA_NEUTON_LOAD_DECL(solution_id)                                                \
	nrf_edgeai_t *nrf_edgeai_load_user_model_##solution_id(uint8_t fa_id,                    \
							       const uint8_t *partition_addr)

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_OTA_NEUTON_H_ */
