/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef WW_OBSV_H_
#define WW_OBSV_H_

#include <stdint.h>

#include <nrf_edgeai/nrf_edgeai.h>

/** @brief Create the observability context and metrics for the wakeword model. */
int ww_obsv_init(nrf_edgeai_t *model);

/** @brief Feed one mel feature vector to the input-feature metrics. */
void ww_obsv_update_features(const float *feats, uint16_t n);

/** @brief Feed the wakeword score to the probability metrics. */
void ww_obsv_update_probs(float p);

#endif /* WW_OBSV_H_ */
