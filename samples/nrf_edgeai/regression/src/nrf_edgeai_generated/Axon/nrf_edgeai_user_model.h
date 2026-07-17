/* 2026-05-20T16:18:07.908538 */

/*
* Copyright (c) 2026 Nordic Semiconductor ASA
* SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
*/

#ifndef _NRF_EDGEAI_USER_MODEL_36025_H_
#define _NRF_EDGEAI_USER_MODEL_36025_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus 
extern "C" {
#endif

#if defined(CONFIG_MODEL_OTA_AXON)
/**
 * @brief Load the model from a package in flash and get its instance (@ref nrf_edgeai_t).
 *
 * Non-generated addition, implemented in nrf_edgeai_user_model.c (this directory): fa_id/
 * partition_addr are passed straight through to model_pkg_load_axon(), so this can be pointed
 * at any partition holding an Axon package for this solution ID - including more than one
 * model instance sharing this same generated code, each backed by its own partition.
 *
 * @param fa_id           Flash area ID of the partition to load from (e.g.
 *                        PARTITION_ID(model_partition)).
 * @param partition_addr  Base address of that same partition (e.g.
 *                        PARTITION_ADDRESS(model_partition)).
 * @return Pointer to a ready-to-use nrf_edgeai_t on success, or NULL if that partition does
 *         not currently hold a valid Axon package for this solution ID.
 */
nrf_edgeai_t *nrf_edgeai_load_user_model_36025(uint8_t fa_id, const uint8_t *partition_addr);

/**
 * @brief Get the current model instance (@ref nrf_edgeai_t) as-is, without loading anything.
 *
 * Only valid to call after a successful nrf_edgeai_load_user_model_36025() - use that instead
 * unless a load already happened and only the pointer is needed again.
 */
nrf_edgeai_t *nrf_edgeai_user_model_36025(void);
#else
/** 
 * @brief Get pointer to the Nordic Edge AI Lab model instance (@ref nrf_edgeai_t).
 */
nrf_edgeai_t* nrf_edgeai_user_model_36025(void);
#endif
/** 
 * @brief Get size FLASH/ROM size of the Nordic Edge AI Neuton model. 
 * 
 * @warning This function is only valid for Neuton models. You can check the model type
 *          by inspecting the model context type field: p_edgeai->model.type == NRF_EDGEAI_MODEL_NEUTON
 * @return Size in bytes of the Neuton model.
 */ 
uint32_t nrf_edgeai_user_model_neuton_size_36025(void);

/** 
 * @brief Alias for the Nordic Edge AI Lab user model API name: the load function when OTA is
 * enabled (main.c needs a load on every iteration), the plain accessor otherwise.
 */
#ifndef nrf_edgeai_user_model
#if defined(CONFIG_MODEL_OTA_AXON)
#define nrf_edgeai_user_model nrf_edgeai_load_user_model_36025
#else
#define nrf_edgeai_user_model nrf_edgeai_user_model_36025
#endif
#endif

/** 
 * @brief Alias for the Nordic Edge AI Lab user model neuton size API name.
 */
#ifndef nrf_edgeai_user_model_neuton_size
#define nrf_edgeai_user_model_neuton_size nrf_edgeai_user_model_neuton_size_36025
#endif

#ifdef __cplusplus 
}
#endif 

#endif /* _NRF_EDGEAI_USER_MODEL_36025_H_ */
