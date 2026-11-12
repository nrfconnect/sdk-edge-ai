/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @file model_obsv.h
 * @defgroup model_obsv Model observability
 * @{
 * @ingroup app_mo
 *
 * nRF Edge AI observability for the application model: output metrics
 * (class predictions distribution, transition matrix, model certainty
 * descriptor) published as a Memfault CDR.
 *
 * With @c CONFIG_MODEL_OBSERVABILITY disabled, the functions are empty stubs.
 */

#ifndef __MODEL_OBSV_H__
#define __MODEL_OBSV_H__

#include <zephyr/sys/util.h>

#include <nrf_edgeai/nrf_edgeai.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create the observability context and metrics, bind the Memfault transport
 *
 * @param model Initialized Edge AI model, used to read the model metadata
 *
 * @return 0 on success, negative errno on failure
 */
int model_obsv_init(nrf_edgeai_t *model);

/**
 * @brief Feed one class-probability vector to the observability metrics
 *
 * @param probs Probability per model output class, in model output order
 *
 * @return 0 on success, negative errno on failure
 */
int model_obsv_update_probs(const flt32_t *probs);

#ifdef __cplusplus
}
#endif

#endif /* __MODEL_OBSV_H__ */

/**
 * @}
 */
