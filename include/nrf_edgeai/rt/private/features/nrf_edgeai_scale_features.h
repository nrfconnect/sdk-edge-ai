/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef _NRF_EDGEAI_PRIVATE_INTERFACES_SCALE_FEATURES_H_
#define _NRF_EDGEAI_PRIVATE_INTERFACES_SCALE_FEATURES_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @def NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE
 * @brief Macro to declare feature scaling interface functions for various data types and scaling modes.
 *
 * All declared functions have the following prototype:
 * @code
 * nrf_edgeai_err_t nrf_edgeai_scale_features_<mode>_<input_type>_<output_type>(nrf_edgeai_input_t* p_input, nrf_edgeai_dsp_pipeline_t* p_dsp);
 * @endcode
 * where:
 *   - \<mode\>        : input_vector, input_window, or dsp
 *   - \<input_type\>  : i8, i16, or f32 (input data type)
 *   - \<output_type\> : q8, q16, or f32 (output/target data type)
 */
#define NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(interface_name)                                 \
    nrf_edgeai_err_t nrf_edgeai_scale_features_##interface_name(nrf_edgeai_input_t*        p_input, \
                                                                nrf_edgeai_dsp_pipeline_t* p_dsp)

/**
 * @brief Scale a vector of input features to the target data type.
 *
 * These functions perform scaling of a single vector of input features from the original data type
 * (int8, int16, or float32) to the target quantized or floating-point type (q8, q16, or f32).
 * Scaling is performed using per-feature min/max values provided in the input context.
 *
 * @param[in, out] p_input        Pointer to the input processing context @ref nrf_edgeai_input_t
 * @param[in, out] p_dsp          Pointer to the DSP pipeline context @ref nrf_edgeai_dsp_pipeline_t
 * @return Status code indicating success or error.
 *
 * @note Example functions: nrf_edgeai_scale_features_input_vector_i8_q8, nrf_edgeai_scale_features_input_vector_f32_q16, etc.
 */
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_vector_i8_q8);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_vector_i8_q16);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_vector_i8_f32);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_vector_i16_q8);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_vector_i16_q16);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_vector_i16_f32);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_vector_f32_q8);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_vector_f32_q16);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_vector_f32_f32);

/**
 * @brief Scale a window (matrix) of input features to the target data type.
 *
 * These functions perform scaling of a window (matrix) of input features, where each row or column
 * represents a feature over a time window, from the original data type to the target type.
 * Scaling is performed using per-feature min/max values and supports selective scaling based on feature masks.
 *
 * @param[in, out] p_input        Pointer to the input processing context @ref nrf_edgeai_input_t
 * @param[in, out] p_dsp          Pointer to the DSP pipeline context @ref nrf_edgeai_dsp_pipeline_t
 * @return Status code indicating success or error.
 *
 * @note Example functions: nrf_edgeai_scale_features_input_window_i8_q8, nrf_edgeai_scale_features_input_window_f32_f32, etc.
 */
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_window_i8_q8);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_window_i8_q16);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_window_i8_f32);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_window_i16_q8);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_window_i16_q16);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_window_i16_f32);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_window_f32_q8);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_window_f32_q16);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(input_window_f32_f32);

/**
 * @brief Scale DSP features from input data.
 *
 * These functions perform scaling of digital signal processing (DSP) features (such as statistical or spectral features)
 * from the input data.
 * The scaling is performed according to the configuration in the neural network context.
 *
 * @param[in, out] p_input        Pointer to the input processing context @ref nrf_edgeai_input_t
 * @param[in, out] p_dsp          Pointer to the DSP pipeline context @ref nrf_edgeai_dsp_pipeline_t
 * @return Status code indicating success or error.
 *
 * @note Example functions: nrf_edgeai_scale_features_dsp_i8_q8, nrf_edgeai_scale_features_dsp_f32_f32, etc.
 */
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(dsp_i8_q8);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(dsp_i8_q16);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(dsp_i8_f32);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(dsp_i16_q8);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(dsp_i16_q16);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(dsp_i16_f32);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(dsp_f32_q8);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(dsp_f32_q16);
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(dsp_f32_f32);

// For models that do not require feature scaling, an empty implementation is provided.
NRF_EDGEAI_DECLARE_SCALE_FEATURES_INTERFACE(empty);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_PRIVATE_INTERFACES_SCALE_FEATURES_H_ */