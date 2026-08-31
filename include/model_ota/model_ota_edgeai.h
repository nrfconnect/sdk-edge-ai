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
 * Wired models are built from model_ota_edgeai_neuton_wired.c.in or
 * model_ota_edgeai_axon_wired.c.in. Generated nrf_edgeai_user_model.c stays agnostic: it honors
 * the MODEL_OTA_WIRED hook when a wired translation unit defines it before #include.
 */

#include <stddef.h>
#include <stdint.h>

#include <nrf_edgeai/nrf_edgeai.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Declare nrf_edgeai_load_user_model_<solution_id>() from a wired static library. */
#define MODEL_OTA_EDGEAI_LOAD_DECL(solution_id)                                               \
	nrf_edgeai_t *nrf_edgeai_load_user_model_##solution_id(const uint8_t *partition_addr, \
							       size_t partition_size)

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_OTA_EDGEAI_H_ */
