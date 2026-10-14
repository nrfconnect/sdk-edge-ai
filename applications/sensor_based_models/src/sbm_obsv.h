/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef SBM_OBSV_H_
#define SBM_OBSV_H_

#include <nrf_edgeai/nrf_edgeai.h>

/** @brief Initialize output observability for the sensor-based model.
 *
 * @param[in] model Initialized Edge AI model used to read model metadata.
 *
 * @return 0 on success, or a negative errno code on failure.
 */
int sbm_obsv_init(nrf_edgeai_t *model);

/** @brief Feed one class-probability vector to the output metrics.
 *
 * @param[in] probs Probability per model output class, in label order.
 */
void sbm_obsv_update_probs(const float *probs);

#endif /* SBM_OBSV_H_ */
