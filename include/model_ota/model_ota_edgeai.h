/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef MODEL_OTA_MODEL_OTA_EDGEAI_H_
#define MODEL_OTA_MODEL_OTA_EDGEAI_H_

/**
 * @file
 * @brief Model-only OTA helpers for Edge AI Lab solutions (Neuton or Axon backend).
 *
 * Wired models are built from model_ota_edgeai_neuton_wired.c or model_ota_edgeai_axon_wired.c.
 * Generated nrf_edgeai_user_model.c stays agnostic: it honors
 * the MODEL_OTA_WIRED hook when a wired translation unit defines it before #include.
 */

#include <model_ota/model_image.h>

#include <nrf_edgeai/nrf_edgeai.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MODEL_OTA_EDGEAI_LOAD_DECL_(solution_id)                                                   \
	enum model_image_result nrf_edgeai_load_user_model_##solution_id(nrf_edgeai_t **out)

/** Declare nrf_edgeai_load_user_model_<solution_id>() from a wired static library. */
#define MODEL_OTA_EDGEAI_LOAD_DECL(solution_id) MODEL_OTA_EDGEAI_LOAD_DECL_(solution_id)

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_OTA_EDGEAI_H_ */
