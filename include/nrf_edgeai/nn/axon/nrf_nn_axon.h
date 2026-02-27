/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
/**
 *
 * @defgroup nrf_nn_axon Axon Neural Network processing library
 * @{
 * @ingroup nrf_nn
 *
 *
 */
#ifndef _NRF_NN_AXON_H_
#define _NRF_NN_AXON_H_

#include <nrf_edgeai/nn/platform/nrf_nn_platform_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Axon NN model type, forward declaration */
struct nrf_axon_nn_compiled_model_tag_s;
typedef struct nrf_axon_nn_compiled_model_tag_s nrf_nn_axon_model_t;

/**
 * @brief Initialize the Axon NN inference engine.
 * 
 * @param[in] p_nn Pointer to Axon NN model.
 * 
 * @return Operational status code @ref nrf_axon_result_e
 */
int8_t nrf_nn_axon_init(const nrf_nn_axon_model_t* p_nn);

/**
 * @brief Deinitialize the Axon NN inference engine and release any resources.
 * 
 * @param[in] p_nn Pointer to Axon NN model.
 * 
 * @return Operational status code @ref nrf_axon_result_e
 */
int8_t nrf_nn_axon_deinit(const nrf_nn_axon_model_t* p_nn);

/**
 * @brief Initialize all the persistent var buffers 
 * in a streaming-style model (with VarHandle/ReadVariable/AssignVariable). 
 * Should be called at the start of each streaming session. Harmless to call for non-streaming style models.
 * 
 * @note Called internally by nrf_nn_axon_init, but can be called separately if persistent vars need to be re-initialized without re-initializing the whole model (e.g. between streaming sessions).
 * 
 * @param[in] p_nn Pointer to Axon NN model.
 * 
 * @return Operational status code @ref nrf_axon_result_e
 */
int8_t nrf_nn_axon_init_persistent_vars(const nrf_nn_axon_model_t* p_nn);

/** 
 * @brief Scale floating point input features to quantized int8 input tensor format.
 * 
 * @param[in] p_nn Pointer to Axon NN model.
 * @param[in] p_input_features Pointer to floating point input features.
 * @param[in] input_features_num Number of input features.
 * @param[out] p_scaled_features Pointer to buffer where scaled int8 features will be stored.
 */
void nrf_nn_axon_scale_inputs(const nrf_nn_axon_model_t* p_nn,
                              const flt32_t*             p_input_features,
                              size32_t                   input_features_num,
                              int8_t*                    p_scaled_features);

/** 
 * @brief Run inference on an Axon NN model.
 * 
 * @param[in] p_nn Pointer to initialized Axon NN model.
 * @param[in] p_input_features Pointer to input features to be used for inference.
 * @param[out] p_output_buffer Pointer to buffer where output will be stored.
 * 
 * @return Operational status code @ref nrf_axon_result_e
 */
int8_t nrf_nn_axon_run_inference(const nrf_nn_axon_model_t* p_nn,
                                 const int8_t*              p_input_features,
                                 int8_t*                    p_output_buffer);

/** 
 * @brief Dequantize the outputs from an Axon NN model.
 * 
 * @param[in] p_nn Pointer to initialized Axon NN model.
 * @param[in] p_output_buffer Pointer to buffer containing quantized outputs.
 * @param[in] outputs_num Number of outputs to dequantize.
 * @param[out] p_dequantized_outputs Pointer to buffer where dequantized outputs will be stored.
 */
void nrf_nn_axon_dequantize_outputs(const nrf_nn_axon_model_t* p_nn,
                                    const int8_t*              p_output_buffer,
                                    uint16_t                   outputs_num,
                                    flt32_t*                   p_dequantized_outputs);
#ifdef __cplusplus
}
#endif

#endif /* _NRF_NN_AXON_H_ */

/**
 * @}
 */