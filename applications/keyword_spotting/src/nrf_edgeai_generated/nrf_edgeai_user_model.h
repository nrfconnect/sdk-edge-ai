/* 2026-01-29T20:26:36.117161 */

/*
* Copyright (c) 2021 Nordic Semiconductor ASA
* SPDX-License-Identifier: Apache-2.0
*/

#ifndef _NRF_EDGEAI_USER_MODEL_H_
#define _NRF_EDGEAI_USER_MODEL_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/** 
 * @brief Get pointer to the Nordic Edge AI Lab model instance (@ref nrf_edgeai_t).
 */
nrf_edgeai_t* nrf_edgeai_user_model_wakeword(void);
/** 
 * @brief Get size FLASH/ROM size of the Nordic Edge AI Neuton model. 
 * 
 * @warning This function is only valid for Neuton models. You can check the model type
 *          by inspecting the model context type field: p_edgeai->model.type == NRF_EDGEAI_MODEL_NEUTON
 * @return Size in bytes of the Neuton model.
 */
uint32_t nrf_edgeai_user_model_neuton_size_wakeword(void);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_USER_MODEL_H_ */
