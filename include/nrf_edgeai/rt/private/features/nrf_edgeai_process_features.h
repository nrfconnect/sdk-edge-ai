/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_EDGEAI_PRIVATE_INTERFACES_PROCESS_FEATURES_H_
#define _NRF_EDGEAI_PRIVATE_INTERFACES_PROCESS_FEATURES_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @def NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE
 * @brief Macro to declare feature processing interface functions for various data types and processing modes.
 *
 * All declared functions have the following prototype:
 * @code
 * nrf_edgeai_err_t nrf_edgeai_process_features_<mode>_<input_type>_<output_type>(nrf_edgeai_t* p_edgeai);
 * @endcode
 * where:
 *   - <mode>        : scale_vector, scale_window, or dsp
 *   - <input_type>  : i8, i16, or f32 (input data type)
 *   - <output_type> : q8, q16, or f32 (output/target data type)
 */
#define NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(interface_name) \
    nrf_edgeai_err_t nrf_edgeai_process_features_##interface_name(    \
        nrf_edgeai_input_t*        p_input,                           \
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
 * @note Example functions: nrf_edgeai_process_features_scale_vector_i8_q8, nrf_edgeai_process_features_scale_vector_f32_q16, etc.
 */
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_vector_i8_q8);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_vector_i8_q16);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_vector_i8_f32);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_vector_i16_q8);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_vector_i16_q16);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_vector_i16_f32);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_vector_f32_q8);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_vector_f32_q16);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_vector_f32_f32);

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
 * @note Example functions: nrf_edgeai_process_features_scale_window_i8_q8, nrf_edgeai_process_features_scale_window_f32_f32, etc.
 */
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_window_i8_q8);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_window_i8_q16);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_window_i8_f32);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_window_i16_q8);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_window_i16_q16);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_window_i16_f32);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_window_f32_q8);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_window_f32_q16);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(scale_window_f32_f32);

/**
 * @brief Extract and scale DSP features from input data.
 *
 * These functions perform digital signal processing (DSP) feature extraction (such as statistical or spectral features)
 * from the input data and then scale the extracted features to the target data type.
 * The extraction and scaling are performed according to the configuration in the neural network context.
 *
 * @param[in, out] p_input        Pointer to the input processing context @ref nrf_edgeai_input_t
 * @param[in, out] p_dsp          Pointer to the DSP pipeline context @ref nrf_edgeai_dsp_pipeline_t
 * @return Status code indicating success or error.
 *
 * @note Example functions: nrf_edgeai_process_features_dsp_i8_q8, nrf_edgeai_process_features_dsp_f32_f32, etc.
 */
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(dsp_i8_q8);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(dsp_i8_q16);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(dsp_i8_f32);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(dsp_i16_q8);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(dsp_i16_q16);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(dsp_i16_f32);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(dsp_f32_q8);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(dsp_f32_q16);
NRF_EDGEAI_DECLARE_PROCESS_FEATURES_INTERFACE(dsp_f32_f32);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_PRIVATE_INTERFACES_PROCESS_FEATURES_H_ */