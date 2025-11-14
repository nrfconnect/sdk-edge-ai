/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_EDGEAI_OUTPUT_TYPES_H_
#define _NRF_EDGEAI_OUTPUT_TYPES_H_

#include <nrf_edgeai/nrf_edgeai_ctypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Decoded output structure for classification tasks
 */
typedef struct nrf_edgeai_decoded_output_classif_s
{
    uint16_t predicted_class; /**< Predicted class index */
    uint16_t num_classes;     /**< Number of classes in the model */

    /**
     * @brief Pointer to probabilities for each class
     */
    union
    {
        const flt32_t*  p_f32; /**< Probabilities for f32 models or after dequantization */
        const uint8_t*  p_q8;  /**< Probabilities for q8 models */
        const uint16_t* p_q16; /**< Probabilities for q16 models */
    } probabilities;
} nrf_edgeai_decoded_output_classif_t;

/**
 * @brief Decoded output structure for regression tasks
 */
typedef struct nrf_edgeai_decoded_output_regress_s
{
    uint16_t       outputs_num; /**< Number of regression outputs */
    const flt32_t* p_outputs;   /**< Pointer to regression output values */

    /**
     * @brief Meta information for regression output (internal use only)
     */
    struct
    {
        const flt32_t* p_scale_min; /**< Pointer to minimum scale values */
        const flt32_t* p_scale_max; /**< Pointer to maximum scale values */
    } meta;
} nrf_edgeai_decoded_output_regress_t;

/**
 * @brief Decoded output structure for anomaly detection tasks
 */
typedef struct nrf_edgeai_decoded_output_anomaly_s
{
    flt32_t score; /**< Anomaly score value */

    /**
     * @brief Meta information for regression-anomaly output (internal use only)
     */
    struct
    {
        const flt32_t* p_scale_min;         /**< Pointer to minimum scale values */
        const flt32_t* p_scale_max;         /**< Pointer to maximum scale values */
        const flt32_t* p_average_embedding; /**< Pointer to average embedding values */
    } meta;
} nrf_edgeai_decoded_output_anomaly_t;

/**
 * @brief Union of decoded output structures for all supported model tasks
 */
typedef union nrf_edgeai_decoded_output_u
{
    nrf_edgeai_decoded_output_classif_t classif;    /**< Decoded output for classification */
    nrf_edgeai_decoded_output_regress_t regression; /**< Decoded output for regression */
    nrf_edgeai_decoded_output_anomaly_t anomaly;    /**< Decoded output for anomaly detection */
} nrf_edgeai_decoded_output_t;

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_OUTPUT_TYPES_H_ */