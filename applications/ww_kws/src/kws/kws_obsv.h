/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef KWS_OBSV_H_
#define KWS_OBSV_H_

#include <stdint.h>

#include <nrf_edgeai/nrf_edgeai.h>

/** @brief Create the observability context and metrics for the keyword spotting model. */
int kws_obsv_init(nrf_edgeai_t *model);

/** @brief Feed one mel feature vector to the input-feature metrics. */
void kws_obsv_update_features(const float *feats, uint16_t n);

/** @brief Feed one class-probability vector to the probability metrics. */
void kws_obsv_update_probs(const float *probs);

#endif /* KWS_OBSV_H_ */
