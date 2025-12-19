/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_nn_neuton Neuton Neural Network processing library
 * @{
 * @ingroup nrf_nn
 *
 *
 */
#ifndef _NRF_NN_NEUTON_H_
#define _NRF_NN_NEUTON_H_

#include <nrf_edgeai/nn/platform/nrf_nn_platform_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Neuton model meta information */
typedef struct nrf_nn_neuton_meta_s
{
    const uint16_t* p_neuron_internal_links_num; /**< Number of internal links for each neuron */
    const uint16_t* p_neuron_external_links_num; /**< Number of external links for each neuron */
    const uint16_t* p_output_neurons_indices;    /**< Indices of output neurons */
    const uint16_t* p_neuron_links;              /**< Neuron links array */
    const uint8_t*  p_neuron_act_type_mask;      /**< Activation function type mask for neurons */
    const uint16_t  outputs_num;                 /**< Number of model outputs */
    const uint16_t  neurons_num;                 /**< Number of neurons in the model */
    const uint32_t  weights_num;                 /**< Number of weights in the model */
} nrf_nn_neuton_meta_t;

/** Neuton model structure for 8-bit quantized model */
typedef struct nrf_nn_neuton_model_q8_s
{
    nrf_nn_neuton_meta_t meta;          /**< Model meta information */
    const int8_t*        p_weights;     /**< Pointer to weights array */
    const uint8_t*       p_act_weights; /**< Pointer to activation function weights */
    uint8_t*             p_neurons;     /**< Pointer to neurons buffer */
} nrf_nn_neuton_model_q8_t;

/** Neuton model structure for 16-bit quantized model */
typedef struct nrf_nn_neuton_model_q16_s
{
    nrf_nn_neuton_meta_t meta;          /**< Model meta information */
    const int16_t*       p_weights;     /**< Pointer to weights array */
    const uint16_t*      p_act_weights; /**< Pointer to activation function weights */
    uint16_t*            p_neurons;     /**< Pointer to neurons buffer */
} nrf_nn_neuton_model_q16_t;

/** Neuton model structure for 32-bit floating point model */
typedef struct nrf_nn_neuton_model_f32_s
{
    nrf_nn_neuton_meta_t meta;          /**< Model meta information */
    const flt32_t*       p_weights;     /**< Pointer to weights array */
    const flt32_t*       p_act_weights; /**< Pointer to activation function weights */
    flt32_t*             p_neurons;     /**< Pointer to neurons buffer */
} nrf_nn_neuton_model_f32_t;

/** @brief Run inference for 8-bit quantized Neuton model, results of inferece are stored in p_model->p_neurons
 * 
 * @param[in] p_model     Pointer to Neuton model structure @ref nrf_nn_neuton_model_q8_t
 * @param[in] p_inputs    Pointer to model input features array
 * @param[in] inputs_num  Number of model input features
 */
void nrf_nn_neuton_run_inference_q8(nrf_nn_neuton_model_q8_t* p_model,
                                    const uint8_t*            p_inputs,
                                    const uint16_t            inputs_num);

/** @brief Run inference for 16-bit quantized Neuton model, results of inferece are stored in p_model->p_neurons
 * 
 * @param[in] p_model     Pointer to Neuton model structure @ref nrf_nn_neuton_model_q16_t
 * @param[in] p_inputs    Pointer to model input features array
 * @param[in] inputs_num  Number of model input features
 */
void nrf_nn_neuton_run_inference_q16(nrf_nn_neuton_model_q16_t* p_model,
                                     const uint16_t*            p_inputs,
                                     const uint16_t             inputs_num);

/** @brief Run inference for 32-bit floating point Neuton model, results of inferece are stored in p_model->p_neurons
 *  
 * @param[in] p_model     Pointer to Neuton model structure @ref nrf_nn_neuton_model_f32_t
 * @param[in] p_inputs    Pointer to model input features array
 * @param[in] inputs_num  Number of model input features
 */
void nrf_nn_neuton_run_inference_f32(nrf_nn_neuton_model_f32_t* p_model,
                                     const flt32_t*             p_inputs,
                                     const uint16_t             inputs_num);

/** @brief Run inference with raw arguments for 32-bit floating point Neuton model, results of inferece are stored in p_neurons
 *
 * @param[in, out] p_neurons                  Pointer to neurons buffer
 * @param[in]     neurons_num                 Number of neurons in the model
 * @param[in]     p_neuron_links              Pointer to neuron links array
 * @param[in]     p_neuron_internal_links_num Pointer to number of internal links for each neuron
 * @param[in]     p_neuron_external_links_num Pointer to number of external links for each neuron
 * @param[in]     p_neuron_act_type_mask      Pointer to activation function type mask for neurons
 * @param[in]     p_weights                   Pointer to weights array 
 * @param[in]     p_act_weights               Pointer to activation function weights
 * @param[in]     p_neurons_inputs            Pointer to model input features array
 * @param[in]     neurons_inputs_num         Number of model input features
 */
void nrf_nn_neuton_run_inference_raw_f32(flt32_t*        p_neurons,
                                         const uint32_t  neurons_num,
                                         const uint16_t* p_neuron_links,
                                         const uint16_t* p_neuron_internal_links_num,
                                         const uint16_t* p_neuron_external_links_num,
                                         const uint8_t*  p_neuron_act_type_mask,
                                         const flt32_t*  p_weights,
                                         const flt32_t*  p_act_weights,
                                         const flt32_t*  p_neurons_inputs,
                                         const uint16_t  neurons_inputs_num);

/** @brief Run inference with raw arguments for 8-bit quantized Neuton model, results of inferece are stored in p_neurons
 *
 * @param[in, out] p_neurons                  Pointer to neurons buffer
 * @param[in]     neurons_num                 Number of neurons in the model
 * @param[in]     p_neuron_links              Pointer to neuron links array
 * @param[in]     p_neuron_internal_links_num Pointer to number of internal links for each neuron
 * @param[in]     p_neuron_external_links_num Pointer to number of external links for each neuron
 * @param[in]     p_neuron_act_type_mask      Pointer to activation function type mask for neurons
 * @param[in]     p_weights                   Pointer to weights array 
 * @param[in]     p_act_weights               Pointer to activation function weights
 * @param[in]     p_neurons_inputs            Pointer to model input features array
 * @param[in]     neurons_inputs_num         Number of model input features
 */
void nrf_nn_neuton_run_inference_raw_q8(uint8_t*        p_neurons,
                                        const uint32_t  neurons_num,
                                        const uint16_t* p_neuron_links,
                                        const uint16_t* p_neuron_internal_links_num,
                                        const uint16_t* p_neuron_external_links_num,
                                        const uint8_t*  p_neuron_act_type_mask,
                                        const int8_t*   p_weights,
                                        const uint8_t*  p_act_weights,
                                        const uint8_t*  p_neurons_inputs,
                                        const uint16_t  neurons_inputs_num);

/** @brief Run inference with raw arguments for 16-bit quantized Neuton model, results of inferece are stored in p_neurons
 *
 * @param[in, out] p_neurons                  Pointer to neurons buffer
 * @param[in]     neurons_num                 Number of neurons in the model
 * @param[in]     p_neuron_links              Pointer to neuron links array
 * @param[in]     p_neuron_internal_links_num Pointer to number of internal links for each neuron
 * @param[in]     p_neuron_external_links_num Pointer to number of external links for each neuron
 * @param[in]     p_neuron_act_type_mask      Pointer to activation function type mask for neurons
 * @param[in]     p_weights                   Pointer to weights array 
 * @param[in]     p_act_weights               Pointer to activation function weights
 * @param[in]     p_neurons_inputs            Pointer to model input features array
 * @param[in]     neurons_inputs_num         Number of model input features
 */
void nrf_nn_neuton_run_inference_raw_q16(uint16_t*       p_neurons,
                                         const uint32_t  neurons_num,
                                         const uint16_t* p_neuron_links,
                                         const uint16_t* p_neuron_internal_links_num,
                                         const uint16_t* p_neuron_external_links_num,
                                         const uint8_t*  p_neuron_act_type_mask,
                                         const int16_t*  p_weights,
                                         const uint16_t* p_act_weights,
                                         const uint16_t* p_neurons_inputs,
                                         const uint16_t  neurons_inputs_num);
#ifdef __cplusplus
}
#endif

#endif /* _NRF_NN_NEUTON_H_ */

/**
 * @}
 */