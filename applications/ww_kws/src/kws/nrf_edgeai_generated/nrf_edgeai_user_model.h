/* 2026-07-07T13:24:00.439127 */

/*
* Copyright (c) 2026 Nordic Semiconductor ASA
* SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
*/

#ifndef _NRF_EDGEAI_USER_MODEL_36712_H_
#define _NRF_EDGEAI_USER_MODEL_36712_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get pointer to the Nordic Edge AI Lab model instance (@ref nrf_edgeai_t).
 */
nrf_edgeai_t* nrf_edgeai_user_model_36712(void);
/**
 * @brief Get size FLASH/ROM size of the Nordic Edge AI model.
 *
 * @return Size in bytes of the model.
 */
uint32_t nrf_edgeai_user_model_size_36712(void);

/**
 * @brief Alias for the Nordic Edge AI Lab user model API name.
 */
#ifndef nrf_edgeai_user_model
#define nrf_edgeai_user_model nrf_edgeai_user_model_36712
#endif

/**
 * @brief Alias for the Nordic Edge AI Lab user model size API name.
 */
#ifndef nrf_edgeai_user_model_size
#define nrf_edgeai_user_model_size nrf_edgeai_user_model_size_36712
#endif

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_USER_MODEL_36712_H_ */
