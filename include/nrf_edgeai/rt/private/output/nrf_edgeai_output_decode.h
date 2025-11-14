/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_EDGEAI_PRIVATE_INTERFACES_OUTPUT_DECODE_H_
#define _NRF_EDGEAI_PRIVATE_INTERFACES_OUTPUT_DECODE_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Decode neural network outputs for classification tasks.
 *
 * These functions process the raw model outputs for classification tasks, converting them into
 * predicted class indices and class probabilities. The decoding is performed according to the
 * output data type (f32, q8, or q16).
 *
 * @param[in] p_model_output      Pointer to model output structure @ref nrf_edgeai_model_t
 * @param[out] p_decoded_output   Pointer to decoded output structure @ref nrf_edgeai_decoded_output_t
 *
 * @note
 * - Use nrf_edgeai_output_decode_classification_f32() for 32-bit floating point outputs.
 * - Use nrf_edgeai_output_decode_classification_q8() for 8-bit quantized outputs.
 * - Use nrf_edgeai_output_decode_classification_q16() for 16-bit quantized outputs.
 */
void nrf_edgeai_output_decode_classification_f32(nrf_edgeai_model_output_t*   p_model_output,
                                                 nrf_edgeai_decoded_output_t* p_decoded_output);
void nrf_edgeai_output_decode_classification_q8(nrf_edgeai_model_output_t*   p_model_output,
                                                nrf_edgeai_decoded_output_t* p_decoded_output);
void nrf_edgeai_output_decode_classification_q16(nrf_edgeai_model_output_t*   p_model_output,
                                                 nrf_edgeai_decoded_output_t* p_decoded_output);

/**
 * @brief Decode neural network outputs for regression tasks.
 *
 * This function processes the raw model outputs for regression tasks, converting them into
 * final regression values.
 *
 * @param[in] p_model_output      Pointer to model output structure @ref nrf_edgeai_model_t
 * @param[out] p_decoded_output   Pointer to decoded output structure @ref nrf_edgeai_decoded_output_t
 *
 * @note
 * - Use nrf_edgeai_output_decode_regression_f32() for 32-bit floating point outputs.
 */
void nrf_edgeai_output_decode_regression_f32(nrf_edgeai_model_output_t*   p_model_output,
                                             nrf_edgeai_decoded_output_t* p_decoded_output);

/**
 * @brief Decode neural network outputs for anomaly detection tasks.
 *
 * This function processes the raw model outputs for anomaly detection tasks, converting them into
 * final anomaly score value.
 *
 * @param[in] p_model_output      Pointer to model output structure @ref nrf_edgeai_model_t
 * @param[out] p_decoded_output   Pointer to decoded output structure @ref nrf_edgeai_decoded_output_t
 *
 * @note
 * - Use nrf_edgeai_output_decode_anomaly_f32() for 32-bit floating point outputs.
 */
void nrf_edgeai_output_decode_anomaly_f32(nrf_edgeai_model_output_t*   p_model_output,
                                          nrf_edgeai_decoded_output_t* p_decoded_output);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_PRIVATE_INTERFACES_OUTPUT_DECODE_H_ */