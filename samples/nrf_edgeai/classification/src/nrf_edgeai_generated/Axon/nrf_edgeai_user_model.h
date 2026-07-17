/* 2026-05-20T15:18:33.270485 */

/*
* Copyright (c) 2026 Nordic Semiconductor ASA
* SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
*/

#ifndef _NRF_EDGEAI_USER_MODEL_36237_H_
#define _NRF_EDGEAI_USER_MODEL_36237_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(CONFIG_MODEL_OTA_AXON)
/**
 * @brief Load the model from a package in flash and get its instance (@ref nrf_edgeai_t).
 *
 * @param fa_id           Flash area ID of the partition to load from.
 * @param partition_addr  Base address of that same partition.
 * @return Pointer to a ready-to-use nrf_edgeai_t, or NULL if the load failed.
 */
nrf_edgeai_t *nrf_edgeai_load_user_model_36237(uint8_t fa_id, const uint8_t *partition_addr);

/**
 * @brief Get the current model instance (@ref nrf_edgeai_t) as-is, without loading anything.
 *
 * Only valid to call after a successful nrf_edgeai_load_user_model_36237() - use that instead
 * unless a load already happened and only the pointer is needed again.
 */
nrf_edgeai_t *nrf_edgeai_user_model_36237(void);
#else
/**
 * @brief Get pointer to the Nordic Edge AI Lab model instance (@ref nrf_edgeai_t).
 */
nrf_edgeai_t* nrf_edgeai_user_model_36237(void);
#endif
/**
 * @brief Get size FLASH/ROM size of the Nordic Edge AI Neuton model.
 *
 * @warning This function is only valid for Neuton models. You can check the model type
 *          by inspecting the model context type field: p_edgeai->model.type == NRF_EDGEAI_MODEL_NEUTON
 * @return Size in bytes of the Neuton model.
 */
uint32_t nrf_edgeai_user_model_neuton_size_36237(void);

/**
 * @brief Alias for the Nordic Edge AI Lab user model API name: the load function when OTA is
 * enabled (main.c needs a load on every iteration), the plain accessor otherwise.
 */
#ifndef nrf_edgeai_user_model
#if defined(CONFIG_MODEL_OTA_AXON)
#define nrf_edgeai_user_model nrf_edgeai_load_user_model_36237
#else
#define nrf_edgeai_user_model nrf_edgeai_user_model_36237
#endif
#endif

/**
 * @brief Alias for the Nordic Edge AI Lab user model neuton size API name.
 */
#ifndef nrf_edgeai_user_model_neuton_size
#define nrf_edgeai_user_model_neuton_size nrf_edgeai_user_model_neuton_size_36237
#endif

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_USER_MODEL_36237_H_ */
