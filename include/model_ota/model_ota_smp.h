/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MODEL_OTA_SMP_H_
#define MODEL_OTA_SMP_H_

/**
 * @file
 * @brief Model SMP upload policy hooks (optional callback, pending-reset query).
 *
 * Inference blocking is handled inside the model OTA library. Requires
 * ``CONFIG_MODEL_OTA_SMP``. See :ref:`lib_model_ota` for the upload flow.
 */

#include <stdbool.h>

/** @return True after a model upload finished and reset is required. */
bool model_ota_smp_is_pending_reset(void);

typedef void (*model_ota_smp_upload_notify_cb)(bool active);

void model_ota_smp_set_upload_notify_cb(model_ota_smp_upload_notify_cb cb);

#endif /* MODEL_OTA_SMP_H_ */
