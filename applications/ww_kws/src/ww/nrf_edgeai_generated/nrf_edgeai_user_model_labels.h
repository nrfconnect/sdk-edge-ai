/* 2026-07-07T12:01:02.530970 */

/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef _NRF_EDGEAI_USER_MODEL_LABELS_H_
#define _NRF_EDGEAI_USER_MODEL_LABELS_H_

#include <nrf_edgeai/nrf_edgeai_ctypes.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum nrf_edgeai_user_label_e {
	MODEL_USER_LABEL_OKAY_NORDIC,

	MODEL_USER_LABEL_COUNT
} nrf_edgeai_user_label_t;

static const char *NRF_EDGEAI_USER_LABELS_NAME[] = {"okay nordic"};

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_USER_MODEL_LABELS_H_ */
