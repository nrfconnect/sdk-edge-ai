/* 2025-12-03T11:23:19.065745 */

/*
* Copyright (c) 2021 Nordic Semiconductor ASA
* SPDX-License-Identifier: Apache-2.0
*/

#ifndef _NRF_EDGEAI_USER_TYPES_H_
#define _NRF_EDGEAI_USER_TYPES_H_

#include <nrf_edgeai/nrf_edgeai_ctypes.h>

#ifdef   __cplusplus
extern "C"
{
#endif

typedef int16_t nrf_user_input_t;
typedef int32_t nrf_user_feature_t;
typedef flt32_t nrf_user_output_t;
typedef uint16_t nrf_user_coeff_t;
typedef int16_t nrf_user_weight_t;
typedef nrf_user_coeff_t nrf_user_neuron_t;

#ifdef   __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_USER_TYPES_H_ */
