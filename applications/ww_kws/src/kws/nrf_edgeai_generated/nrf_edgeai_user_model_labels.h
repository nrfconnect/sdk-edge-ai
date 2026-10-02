/* 2026-07-07T13:24:00.439200 */

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
	MODEL_USER_LABEL_OTHER,
	MODEL_USER_LABEL_SILENCE,
	MODEL_USER_LABEL_DOWN,
	MODEL_USER_LABEL_GO,
	MODEL_USER_LABEL_LEFT,
	MODEL_USER_LABEL_NO,
	MODEL_USER_LABEL_OFF,
	MODEL_USER_LABEL_ON,
	MODEL_USER_LABEL_RIGHT,
	MODEL_USER_LABEL_STOP,
	MODEL_USER_LABEL_UP,
	MODEL_USER_LABEL_YES,

	MODEL_USER_LABEL_COUNT
} nrf_edgeai_user_label_t;

static const char *NRF_EDGEAI_USER_LABELS_NAME[] = {
	"OTHER", "SILENCE", "down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"};

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_USER_MODEL_LABELS_H_ */
