/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @file model_obsv.h
 * @defgroup sample_mo_model_obsv Model observability
 * @{
 * @ingroup sample_mo
 *
 * nRF Edge AI observability for the sample: output metrics
 * (class predictions distribution, transition matrix, model certainty
 * descriptor) published as a Memfault CDR.
 */

#ifndef __MODEL_OBSV_H__
#define __MODEL_OBSV_H__

#include <stdint.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_core.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create the observability context and metrics, bind the Memfault transport
 *
 * @param info Model metadata embedded in every CDR
 *
 * @return 0 on success, negative errno on failure
 */
int model_obsv_init(const nrf_edgeai_obsv_model_info_t *info);

/**
 * @brief Feed one class-probability vector to the observability metrics
 *
 * @param probs Probability per class, in @ref activity_class order
 *
 * @return 0 on success, negative errno on failure
 */
int model_obsv_update_probs(const float *probs);

/**
 * @brief Get the number of probability vectors fed since the last collect
 *
 * The collect that stages a CDR resets the context, so the count drops to 0
 * once the fed vectors are encoded into a CDR.
 *
 * @return Number of vectors fed since the last reset
 */
uint32_t model_obsv_num_inferences(void);

#ifdef __cplusplus
}
#endif

#endif /* __MODEL_OBSV_H__ */

/**
 * @}
 */
