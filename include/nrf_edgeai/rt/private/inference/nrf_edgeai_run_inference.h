/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_EDGEAI_PRIVATE_INTERFACES_RUN_INFERENCE_H_
#define _NRF_EDGEAI_PRIVATE_INTERFACES_RUN_INFERENCE_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Run neural network inference for 8-bit quantized model.
 *
 * This function performs a forward pass (inference) of the neural network using 8-bit quantized weights,
 * activations, and neuron buffers. The function selects the input source (raw input or extracted features)
 * based on the model meta information and computes the output by propagating data through all layers.
 *
 * @param[in,out] p_edgeai Pointer to the neural network context structure (@ref nrf_edgeai_t).
 */
void nrf_edgeai_run_model_inference_q8(nrf_edgeai_t* p_edgeai);

/**
 * @brief Run neural network inference for 16-bit quantized model.
 *
 * This function performs a forward pass (inference) of the neural network using 16-bit quantized weights,
 * activations, and neuron buffers. The function selects the input source (raw input or extracted features)
 * based on the model meta information and computes the output by propagating data through all layers.
 *
 * @param[in,out] p_edgeai Pointer to the neural network context structure (@ref nrf_edgeai_t).
 */
void nrf_edgeai_run_model_inference_q16(nrf_edgeai_t* p_edgeai);

/**
 * @brief Run neural network inference for 32-bit floating point model.
 *
 * This function performs a forward pass (inference) of the neural network using 32-bit floating point weights,
 * activations, and neuron buffers. The function selects the input source (raw input or extracted features)
 * based on the model meta information and computes the output by propagating data through all layers.
 *
 * @param[in,out] p_edgeai Pointer to the neural network context structure (@ref nrf_edgeai_t).
 */
void nrf_edgeai_run_model_inference_f32(nrf_edgeai_t* p_edgeai);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_PRIVATE_INTERFACES_RUN_INFERENCE_H_ */