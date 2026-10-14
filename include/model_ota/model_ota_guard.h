/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MODEL_OTA_GUARD_H_
#define MODEL_OTA_GUARD_H_

/**
 * @file
 * @brief Hold a model OTA reader for direct Axon or model-metadata access.
 *
 * OTA-wired Edge AI inference is guarded automatically. Use these calls for
 * other model-linked work (direct ``nrf_axon_nn_model_infer_*``, quantization
 * fields, async inference until the completion callback returns).
 * Requires ``CONFIG_MODEL_OTA_SMP``. See :ref:`lib_model_ota`.
 */

#include <stdbool.h>

/**
 * @brief Snapshot of whether model access is currently permitted.
 *
 * Use for application policy only (for example skipping work or driving an LED).
 * This is not synchronization: an upload may block the guard right after this
 * returns true. Hold model_ota_guard_acquire() around any actual model access.
 *
 * @return True when model access and inference are permitted.
 */
bool model_ota_guard_inference_permitted(void);

/**
 * @retval 0 Access permitted.
 * @retval -EBUSY Guard blocked after a model update started.
 */
int model_ota_guard_acquire(void);

void model_ota_guard_release(void);

#endif /* MODEL_OTA_GUARD_H_ */
