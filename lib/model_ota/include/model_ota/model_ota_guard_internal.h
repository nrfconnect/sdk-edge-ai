/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MODEL_OTA_MODEL_OTA_GUARD_INTERNAL_H_
#define MODEL_OTA_MODEL_OTA_GUARD_INTERNAL_H_

/**
 * @file
 * @brief SMP upload and runtime integration for the model OTA guard (library internal).
 *
 * Applications hold explicit readers with :file:`include/model_ota/model_ota_guard.h`.
 */

/**
 * @brief Block model access device-wide for an SMP upload.
 *
 * @retval 0 Readers drained; model access stays blocked until device reset.
 * @retval -EBUSY Readers did not drain; guard left BLOCKED for the caller.
 */
int model_ota_guard_begin_update(void);

/**
 * @brief Re-check drain for a guard already blocked by begin_update().
 *
 * @retval 0 Readers drained; model access stays blocked until device reset.
 * @retval -EBUSY Readers still in flight; guard left BLOCKED for the caller.
 * @retval -EINVAL Guard is not blocked.
 */
int model_ota_guard_retry_update(void);

/** Return a blocked guard to READY when no model flash was modified. */
void model_ota_guard_abort_update(void);

#endif /* MODEL_OTA_MODEL_OTA_GUARD_INTERNAL_H_ */
