/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef KWS_OBSV_H_
#define KWS_OBSV_H_

#include <stdint.h>

#include <nrf_edgeai/nrf_edgeai.h>

/** @brief Create the observability context and metrics for the keyword spotting model.
 *
 * @param[in] model Initialized Edge AI model, used to read the model metadata.
 *
 * @return 0 on success, or a negative errno code on failure.
 */
int kws_obsv_init(nrf_edgeai_t *model);

/** @brief Feed one mel feature vector to the input-feature metrics.
 *
 * The vector is dropped and an error is logged if @p n is not the expected
 * feature count.
 *
 * @param[in] feats Mel feature vector, @p n floats.
 * @param[in] n     Number of features in @p feats.
 */
void kws_obsv_update_features(const float *feats, uint16_t n);

/** @brief Feed one class-probability vector to the probability metrics.
 *
 * @param[in] probs Probability per model output class, in label order. Must hold
 *                  MODEL_USER_LABEL_COUNT probability values from the KWS model.
 */
void kws_obsv_update_probs(const float *probs);

#endif /* KWS_OBSV_H_ */
