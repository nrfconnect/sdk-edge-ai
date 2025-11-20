/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _NRF_EDGEAI_USER_MODEL_H_
#define _NRF_EDGEAI_USER_MODEL_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus
extern "C" {
#endif

nrf_edgeai_t* nrf_edgeai_user_model(void);
uint32_t      nrf_edgeai_user_model_size(void);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_USER_MODEL_H_ */