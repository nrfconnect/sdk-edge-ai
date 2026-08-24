/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MODEL_OTA_GUARD_H_
#define MODEL_OTA_GUARD_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/**
 * @file
 * @brief Global inference guard for model OTA.
 *
 * A single **device-wide** state and reader count gate all OTA-managed model
 * access. Any model update blocks inference on every OTA-managed context until
 * device reset. Inference is permitted in READY.
 * ``model_ota_guard_begin_update()`` waits for in-flight readers and sets
 * BLOCKED. Once model flash is modified, only device reset returns to READY;
 * an update that is given up on before touching flash releases the guard with
 * ``model_ota_guard_abort_update()``.
 *
 * OTA-wired Edge AI solutions set ``nrf_edgeai_t.is_ota_managed``; the runtime
 * checks that flag and calls ``nrf_edgeai_guard_inference_session_*()`` only
 * for those contexts (through propagate; decode uses RAM and app-side metadata
 * only).
 *
 * For any other model access — direct Axon driver calls, reading quantization
 * fields from a loaded model image, etc. — call ``model_ota_guard_acquire()``
 * and ``model_ota_guard_release()`` in pairs. For async Axon inference, hold
 * the reader from before ``nrf_axon_nn_model_infer_async()`` until the
 * completion callback returns.
 */

/** Guard region lifecycle states. */
enum model_ota_guard_state {
	MODEL_OTA_GUARD_READY = 0,
	MODEL_OTA_GUARD_BLOCKED,
};

/**
 * @brief Hold the global guard reader while accessing model-linked data.
 *
 * Use before direct Axon driver calls, reads of model flash metadata (e.g.
 * quantization coefficients), or other model access outside
 * ``nrf_edgeai_run_inference()``.
 *
 * @retval 0 Access permitted.
 * @retval -EBUSY Guard blocked after a model update started.
 */
int model_ota_guard_acquire(void);

/** Release a reader held by model_ota_guard_acquire(). */
void model_ota_guard_release(void);

/** @return True when model access and inference are permitted. */
bool model_ota_guard_inference_permitted(void);

/**
 * @brief Block model access device-wide for an SMP upload.
 *
 * Sets BLOCKED, then waits up to ``CONFIG_MODEL_OTA_GUARD_DRAIN_TIMEOUT_MS``
 * for in-flight readers to finish.
 *
 * On timeout the guard is left BLOCKED and owned by the caller: new readers are
 * refused, so the in-flight ones drain while the caller waits for the rejected
 * operation to be retried. The caller must then either take the guard with
 * ``model_ota_guard_retry_update()`` or hand it back with
 * ``model_ota_guard_abort_update()`` - nothing else returns the guard to READY.
 *
 * @retval 0 Readers drained; model access stays blocked until device reset.
 * @retval -EBUSY Readers did not drain; guard left BLOCKED for the caller.
 */
int model_ota_guard_begin_update(void);

/**
 * @brief Re-check drain for a guard already blocked by begin_update().
 *
 * Does not wait: readers have been refused since the failed
 * ``model_ota_guard_begin_update()``, so any that were in flight have either
 * finished by now or are stuck.
 *
 * @retval 0 Readers drained; model access stays blocked until device reset.
 * @retval -EBUSY Readers still in flight; guard left BLOCKED for the caller.
 * @retval -EINVAL Guard is not blocked.
 */
int model_ota_guard_retry_update(void);

/**
 * @brief Return a blocked guard to READY.
 *
 * Only valid while no model flash was modified, that is after
 * ``model_ota_guard_begin_update()`` or ``model_ota_guard_retry_update()``
 * returned -EBUSY and the update was given up on. Inference resumes.
 */
void model_ota_guard_abort_update(void);

#endif /* MODEL_OTA_GUARD_H_ */
