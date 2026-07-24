/* 2026-07-07T12:01:02.530929 */

/*
* Copyright (c) 2026 Nordic Semiconductor ASA
* SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
*/

#ifndef _NRF_EDGEAI_USER_MODEL_36711_H_
#define _NRF_EDGEAI_USER_MODEL_36711_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(CONFIG_APP_MODEL_OTA)
/**
 * @brief Load the model from a package in flash and get its instance (@ref nrf_edgeai_t).
 *
 * @param fa_id           Flash area ID of the partition to load from.
 * @param partition_addr  Base address of that same partition.
 * @return Pointer to a ready-to-use nrf_edgeai_t, or NULL if the load failed.
 */
nrf_edgeai_t *nrf_edgeai_load_user_model_36711(uint8_t fa_id, const uint8_t *partition_addr);

/**
 * @brief Get the current model instance (@ref nrf_edgeai_t) as-is, without loading anything.
 *
 * Only valid to call after a successful nrf_edgeai_load_user_model_36711() - use that instead
 * unless a load already happened and only the pointer is needed again.
 */
nrf_edgeai_t *nrf_edgeai_user_model_36711(void);
#else
/**
 * @brief Get pointer to the Nordic Edge AI Lab model instance (@ref nrf_edgeai_t).
 */
nrf_edgeai_t *nrf_edgeai_user_model_36711(void);
#endif
/**
 * @brief Get size FLASH/ROM size of the Nordic Edge AI model.
 *
 * @return Size in bytes of the model.
 */
uint32_t nrf_edgeai_user_model_size_36711(void);

/**
 * @brief Alias for the Nordic Edge AI Lab user model API name: the load function when OTA is
 * enabled (ww_init() loads from flash at boot), the plain accessor otherwise.
 */
#ifndef nrf_edgeai_user_model
#if defined(CONFIG_APP_MODEL_OTA)
#define nrf_edgeai_user_model nrf_edgeai_load_user_model_36711
#else
#define nrf_edgeai_user_model nrf_edgeai_user_model_36711
#endif
#endif

/**
 * @brief Alias for the Nordic Edge AI Lab user model size API name.
 */
#ifndef nrf_edgeai_user_model_size
#define nrf_edgeai_user_model_size nrf_edgeai_user_model_size_36711
#endif

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_USER_MODEL_36711_H_ */
