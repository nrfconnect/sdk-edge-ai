/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef _NRF_EDGEAI_PRIVATE_INTERFACES_RUN_INFERENCE_H_
#define _NRF_EDGEAI_PRIVATE_INTERFACES_RUN_INFERENCE_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/** 
 * @brief Initialize Neuton neural network inference engine
 * 
 * @param[in,out] p_edgeai Pointer to the Edge AI solution context
 *                         (@ref nrf_edgeai_t).
 * @return Operation status code @ref nrf_edgeai_err_t
 */
nrf_edgeai_err_t nrf_edgeai_init_inference_neuton(nrf_edgeai_t* p_edgeai);

/**
 * @defgroup neuton_inference Neuton inference API
 * @brief Run neural network inference for Neuton models.
 * 
 * These functions perform a forward pass (inference) of the neural network
 * using Neuton models with different numeric representations for activations, and neuron buffers:
 * - 8-bit quantized
 * - 16-bit quantized
 * - 32-bit floating point
 *
 * The input source (raw input or extracted features) is selected based on
 * model meta information, and computes the output by propagating features through all neurons.
 *
 * @param[in,out] p_edgeai Pointer to the Edge AI solution context
 *                         (@ref nrf_edgeai_t).
 * 
 *  @return Operation status code @ref nrf_edgeai_err_t
 */

/** @ingroup neuton_inference */
nrf_edgeai_err_t nrf_edgeai_run_inference_neuton_q8(nrf_edgeai_t* p_edgeai);

/** @ingroup neuton_inference */
nrf_edgeai_err_t nrf_edgeai_run_inference_neuton_q16(nrf_edgeai_t* p_edgeai);

/** @ingroup neuton_inference */
nrf_edgeai_err_t nrf_edgeai_run_inference_neuton_f32(nrf_edgeai_t* p_edgeai);

/** 
 * @brief Initialize Axon neural network inference engine
 * 
 * @param[in,out] p_edgeai Pointer to the Edge AI solution context
 *                         (@ref nrf_edgeai_t).
 * @return Operation status code @ref nrf_edgeai_err_t
 */
nrf_edgeai_err_t nrf_edgeai_init_inference_axon(nrf_edgeai_t* p_edgeai);

/** 
 * @brief Run neural network inference for Axon model
 * 
 * @param[in,out] p_edgeai Pointer to the Edge AI solution context
 *                         (@ref nrf_edgeai_t).
 * 
 * @return Operation status code @ref nrf_edgeai_err_t
 */
nrf_edgeai_err_t nrf_edgeai_run_inference_axon(nrf_edgeai_t* p_edgeai);

/** 
 * @brief Run neural network inference for Axon model using specialized audio mel features pipeline
 * 
 * @param[in,out] p_edgeai Pointer to the Edge AI solution context
 *                         (@ref nrf_edgeai_t).
 * 
 * @return Operation status code @ref nrf_edgeai_err_t
 */
nrf_edgeai_err_t nrf_edgeai_run_inference_axon_audiomels(nrf_edgeai_t* p_edgeai);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_PRIVATE_INTERFACES_RUN_INFERENCE_H_ */