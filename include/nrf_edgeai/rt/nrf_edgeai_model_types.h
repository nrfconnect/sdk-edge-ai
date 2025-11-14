/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_EDGEAI_MODEL_TYPES_H_
#define _NRF_EDGEAI_MODEL_TYPES_H_

#include <nrf_edgeai/nrf_edgeai_ctypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief NRF Edge AI model task types
 */

/** For preprocessor */
#define __NRF_EDGEAI_TASK_MULT_CLASS        0
#define __NRF_EDGEAI_TASK_BIN_CLASS         1
#define __NRF_EDGEAI_TASK_REGRESSION        2
#define __NRF_EDGEAI_TASK_ANOMALY_DETECTION 3

typedef enum nrf_edgeai_model_task_e
{
    /**< Multiclass classification task */
    NRF_EDGEAI_TASK_MULT_CLASS = __NRF_EDGEAI_TASK_MULT_CLASS,
    /**< Binary classification task */
    NRF_EDGEAI_TASK_BIN_CLASS = __NRF_EDGEAI_TASK_BIN_CLASS,
    /**< Regression task */
    NRF_EDGEAI_TASK_REGRESSION = __NRF_EDGEAI_TASK_REGRESSION,
    /**< Anomaly Detection task */
    NRF_EDGEAI_TASK_ANOMALY_DETECTION = __NRF_EDGEAI_TASK_ANOMALY_DETECTION
} nrf_edgeai_model_task_t;

/**
 * @brief Model input usage type
 */
typedef union nrf_edgeai_model_uses_as_input_u
{
    struct
    {
        bool input     : 1; /**< Use raw input features */
        bool extracted : 1; /**< Use extracted features */
    } features;
    uint8_t all; /**< All usage flags as bitmask */
} nrf_edgeai_model_uses_as_input_t;

/**
 * @brief Model parameters for 32-bit floating point precision
 */
typedef struct nrf_edgeai_model_params_f32_s
{
    const flt32_t* p_weights;     /**< Pointer to weights array */
    const flt32_t* p_act_weights; /**< Pointer to activation function weights */
    flt32_t*       p_neurons;     /**< Pointer to neurons buffer */
} nrf_edgeai_model_params_f32_t;

/**
 * @brief Model parameters for 16-bit quantized precision
 */
typedef struct nrf_edgeai_model_params_q16_s
{
    const int16_t*  p_weights;     /**< Pointer to weights array */
    const uint16_t* p_act_weights; /**< Pointer to activation function weights */
    uint16_t*       p_neurons;     /**< Pointer to neurons buffer */
} nrf_edgeai_model_params_q16_t;

/**
 * @brief Model parameters for 8-bit quantized precision
 */
typedef struct nrf_edgeai_model_params_q8_s
{
    const int8_t*  p_weights;     /**< Pointer to weights array */
    const uint8_t* p_act_weights; /**< Pointer to activation function weights */
    uint8_t*       p_neurons;     /**< Pointer to neurons buffer */
} nrf_edgeai_model_params_q8_t;

/**
 * @brief Union of model parameters for all supported types
 */
typedef union nrf_edgeai_model_params_u
{
    nrf_edgeai_model_params_q8_t  q8;  /**< 8-bit quantized parameters */
    nrf_edgeai_model_params_q16_t q16; /**< 16-bit quantized parameters */
    nrf_edgeai_model_params_f32_t f32; /**< 32-bit floating point parameters */
} nrf_edgeai_model_params_t;

/**
 * @brief Model meta information structure
 */
typedef struct nrf_edgeai_model_meta_s
{
    /**< Number of internal links for each neuron */
    const uint16_t* p_neuron_internal_links_num;
    /**< Number of external links for each neuron */
    const uint16_t* p_neuron_external_links_num;

    const uint16_t* p_output_neurons_indices; /**< Indices of output neurons */
    const uint16_t* p_neuron_links;           /**< Neuron links array */
    const uint8_t*  p_neuron_act_type_mask;   /**< Activation function type mask for neurons */
    const uint16_t  outputs_num;              /**< Number of model outputs */
    const uint16_t  neurons_num;              /**< Number of neurons in the model */
    const uint32_t  weights_num;              /**< Number of weights in the model */
    const nrf_edgeai_model_task_t task;       /**< Model task type */

    const nrf_edgeai_model_uses_as_input_t uses_as_input; /**< Model input usage type */
} nrf_edgeai_model_meta_t;

/**
 * @brief Model output structure
 */
typedef struct nrf_edgeai_model_output_s
{
    union
    {
        flt32_t*  p_f32;  /**< Pointer to 32-bit float output buffer */
        uint8_t*  p_q8;   /**< Pointer to 8-bit quantized output buffer */
        uint16_t* p_q16;  /**< Pointer to 16-bit quantized output buffer */
        void*     p_void; /**< Generic pointer to output buffer */
    } memory;             /**< Union of output buffer pointers */
    uint16_t num;         /**< Number of output values */
} nrf_edgeai_model_output_t;

/**
 * @brief Model context structure
 */
typedef struct nrf_edgeai_model_s
{
    nrf_edgeai_model_meta_t   meta;   /**< Model meta information */
    nrf_edgeai_model_params_t params; /**< Model parameters */
    nrf_edgeai_model_output_t output; /**< Model output */
} nrf_edgeai_model_t;

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_MODEL_TYPES_H_ */