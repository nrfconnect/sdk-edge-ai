/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef KWS_MODEL_WIRING_H_
#define KWS_MODEL_WIRING_H_

#include <nrf_edgeai/nrf_edgeai.h>

/**
 * @brief Get the keyword spotting model to run inference against.
 *
 * With CONFIG_APP_MODEL_OTA=y (the default), this loads and validates a model package from
 * the model_storage_kws flash partition (see doc/libraries/model_ota.rst).
 *
 * With CONFIG_APP_MODEL_OTA=n, the model is compiled directly into the application image
 * instead: this always returns the same pointer to it, and never NULL.
 *
 * @return Pointer to a ready-to-use nrf_edgeai_t on success, or NULL if model_storage_kws does
 *         not currently hold a valid package (CONFIG_APP_MODEL_OTA=y only).
 */
nrf_edgeai_t *kws_model_ota_load(void);

#endif /* KWS_MODEL_WIRING_H_ */
