/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_EDGEAI_PRIVATE_INTERFACES_OUTPUT_PROPAGATE_H_
#define _NRF_EDGEAI_PRIVATE_INTERFACES_OUTPUT_PROPAGATE_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Propagate neural network outputs to the model output structure.
 *
 * These functions transfer the output neuron values from the internal buffer (of the corresponding data type)
 * to the model output structure. They may also apply any required post-processing such as activation or scaling.
 *
 * @param[in, out] p_model    Pointer to model context @ref nrf_edgeai_model_t
 *
 * @note
 * - Use nrf_edgeai_output_propagate_q8() for 8-bit quantized models.
 * - Use nrf_edgeai_output_propagate_q16() for 16-bit quantized models.
 * - Use nrf_edgeai_output_propagate_f32() for 32-bit floating point models.
 */
void nrf_edgeai_output_propagate_q8(nrf_edgeai_model_t* p_model);
void nrf_edgeai_output_propagate_q16(nrf_edgeai_model_t* p_model);
void nrf_edgeai_output_propagate_f32(nrf_edgeai_model_t* p_model);

/**
 * @brief Dequantize neural network outputs to 32-bit floating point.
 *
 * These functions convert the output values from quantized format (8-bit or 16-bit) to 32-bit floating point format,
 * using the scaling parameters defined in the model context.
 *
 * @param[in, out] p_model    Pointer to model context @ref nrf_edgeai_model_t
 *
 * @note
 * - Use nrf_edgeai_output_dequantize_q8_f32() for 8-bit quantized outputs.
 * - Use nrf_edgeai_output_dequantize_q16_f32() for 16-bit quantized outputs.
 */
void nrf_edgeai_output_dequantize_q8_f32(nrf_edgeai_model_t* p_model);
void nrf_edgeai_output_dequantize_q16_f32(nrf_edgeai_model_t* p_model);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_PRIVATE_INTERFACES_OUTPUT_PROPAGATE_H_ */