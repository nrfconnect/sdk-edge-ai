/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef _NRF_EDGEAI_MODEL_TYPES_H_
#define _NRF_EDGEAI_MODEL_TYPES_H_

#include <nrf_edgeai/nrf_edgeai_ctypes.h>
#include <nrf_edgeai/nn/axon/nrf_nn_axon.h>
#include <nrf_edgeai/nn/neuton/nrf_nn_neuton.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief nRF Edge AI model task types
 */

/** For preprocessor */
#define __NRF_EDGEAI_TASK_MULT_CLASS        0
#define __NRF_EDGEAI_TASK_BIN_CLASS         1
#define __NRF_EDGEAI_TASK_REGRESSION        2
#define __NRF_EDGEAI_TASK_ANOMALY_DETECTION 3

#define __NRF_EDGEAI_MODEL_NEUTON 0
#define __NRF_EDGEAI_MODEL_AXON   1

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
 * @brief Model parameters for 32-bit floating point, 8-bit quantized, 
 *          16-bit quantized precision
 */
typedef nrf_nn_neuton_model_params_f32_t nrf_edgeai_model_neuton_params_f32_t;
typedef nrf_nn_neuton_model_params_q16_t nrf_edgeai_model_neuton_params_q16_t;
typedef nrf_nn_neuton_model_params_q8_t  nrf_edgeai_model_neuton_params_q8_t;

/**
 * @brief Union of model parameters for all supported types
 */
typedef union nrf_edgeai_model_neuton_params_u
{
    nrf_edgeai_model_neuton_params_q8_t  q8;  /**< 8-bit quantized parameters */
    nrf_edgeai_model_neuton_params_q16_t q16; /**< 16-bit quantized parameters */
    nrf_edgeai_model_neuton_params_f32_t f32; /**< 32-bit floating point parameters */
} nrf_edgeai_model_neuton_params_t;

/**
 * @brief Model meta information structure
 */
typedef nrf_nn_neuton_model_meta_t nrf_edgeai_model_neuton_meta_t;

/**
 * @brief Model context structure
 */
typedef struct nrf_edgeai_model_neuton_s
{
    const nrf_edgeai_model_neuton_meta_t   meta;   /**< Model meta information */
    const nrf_edgeai_model_neuton_params_t params; /**< Model parameters */
} nrf_edgeai_model_neuton_t;

/**
 * @brief nRF Edge AI Axon model type
 */
typedef nrf_nn_axon_model_t nrf_edgeai_model_axon_t;

/**
 * @brief nRF Edge AI model types
 */
typedef enum nrf_edgeai_model_type_e
{
    NRF_EDGEAI_MODEL_NEUTON = __NRF_EDGEAI_MODEL_NEUTON,
    NRF_EDGEAI_MODEL_AXON   = __NRF_EDGEAI_MODEL_AXON,
} nrf_edgeai_model_type_t;

/**
 * @brief Union of model instances for all supported types
 */
typedef union nrf_edgeai_model_instance_u
{
    const nrf_edgeai_model_neuton_t* p_neuton; /**< Neuton model instance */
    const nrf_edgeai_model_axon_t*   p_axon;   /**< Axon model instance */
    const void*                      p_void;   /**< Generic pointer to model instance */
} nrf_edgeai_model_instance_t;

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
 * @brief nRF Edge AI model structure
 */
typedef struct nrf_edgeai_model_s
{
    nrf_edgeai_model_type_t          type;          /**< Model type */
    nrf_edgeai_model_task_t          task;          /**< Model task */
    nrf_edgeai_model_instance_t      instance;      /**< Model instance */
    nrf_edgeai_model_output_t        output;        /**< Model output */
    nrf_edgeai_model_uses_as_input_t uses_as_input; /**< Model input usage type */
} nrf_edgeai_model_t;

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_MODEL_TYPES_H_ */