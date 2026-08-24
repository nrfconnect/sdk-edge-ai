/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MODEL_UPDATE_H_
#define MODEL_UPDATE_H_

#include <stdbool.h>

int model_update_init(void);

bool model_update_is_pending_reset(void);

#endif /* MODEL_UPDATE_H_ */
