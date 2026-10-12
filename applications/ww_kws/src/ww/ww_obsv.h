/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef WW_OBSV_H_
#define WW_OBSV_H_

#include <stdint.h>

#include <nrf_edgeai/nrf_edgeai.h>

/** @brief Create the observability context and metrics for the wakeword model.
 *
 * The single-output model is advertised as two classes, [absent, present].
 *
 * @param[in] model Initialized Edge AI model, used to read the model metadata.
 *
 * @return 0 on success, or a negative errno code on failure.
 */
int ww_obsv_init(nrf_edgeai_t *model);

/** @brief Feed one mel feature vector to the input-feature metrics.
 *
 * The vector is dropped and an error is logged if @p n is not the expected
 * feature count.
 *
 * @param[in] feats Mel feature vector, @p n floats.
 * @param[in] n     Number of features in @p feats.
 */
void ww_obsv_update_features(const float *feats, uint16_t n);

/** @brief Feed the wakeword score to the probability metrics.
 *
 * The score is expanded into the two-class vector [1 - p, p] before it is
 * passed to the metrics.
 *
 * @param[in] p Wakeword probability in the range [0, 1].
 */
void ww_obsv_update_probs(float p);

#endif /* WW_OBSV_H_ */
