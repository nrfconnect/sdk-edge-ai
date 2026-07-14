/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MODEL_WIRING_H_
#define MODEL_WIRING_H_

#include <nrf_edgeai/nrf_edgeai.h>

/**
 * @brief Get the regression model to run inference against.
 *
 * Implemented by exactly one of model_wiring_neuton.c / model_wiring_axon.c, selected at build
 * time by the NRF_EDGEAI_REGRESSION_MODEL Kconfig choice, so main.c stays backend-agnostic.
 *
 * With CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA=y (the default), this loads and validates a
 * model package from the model_storage flash partition on every call.
 *
 * With CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA=n, the model is compiled directly into the
 * application image instead: this always returns the same pointer to it, and never NULL.
 *
 * @return Pointer to a ready-to-use nrf_edgeai_t on success, or NULL if model_storage does not
 *         currently hold a valid package for this backend (CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA=y
 *         only).
 */
nrf_edgeai_t *model_ota_load(void);

#endif /* MODEL_WIRING_H_ */
