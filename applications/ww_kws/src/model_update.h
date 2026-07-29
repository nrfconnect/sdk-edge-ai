/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#include <stdbool.h>

/** MCUboot updateable image index for the wakeword model. */
#define WW_MODEL_IMAGE_INDEX 1

/** MCUboot updateable image index for the keyword-spotting model. */
#define KWS_MODEL_IMAGE_INDEX 2

/**
 * Register SMP upload hooks that pause inference during model DFU.
 *
 * @retval 0 on success.
 * @retval Negative errno from model_ota_smp_init() on failure.
 */
int model_update_init(void);

/** True while a WW model SMP upload is active or pending reset. */
bool model_update_blocks_ww_inference(void);

/** True while a KWS model SMP upload is active or pending reset. */
bool model_update_blocks_kws_inference(void);

/** True after any model SMP upload finished and a reset is required. */
bool model_update_is_pending_reset(void);
