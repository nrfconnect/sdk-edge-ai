/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_EDGEAI_PRIVATE_INTERFACES_FEATURES_COMMON_H_
#define _NRF_EDGEAI_PRIVATE_INTERFACES_FEATURES_COMMON_H_

#include <nrf_edgeai/rt/nrf_edgeai_dsp_pipeline_types.h>

/**
 * @def NRF_EDGEAI_DECLARE_FEATURE_FUNCTION
 * @brief Macro to declare time-domain DSP feature extraction functions for various data types.
 *
 * All declared functions have the following prototype:
 * @code
 * size32_t nrf_edgeai_feature_<feature_name>(
 *     <input_type>*                    p_input,
 *     size32_t                      num,
 *     <feature_type>*                  p_features,
 *     nrf_edgeai_features_mask_t        feature_mask,
 *     void*                            p_pipeline_ctx,
 *     nrf_edgeai_feature_get_arg_cb_t   get_argument,
 *     void*                            p_argument_ctx
 * );
 * @endcode
 * where:
 *   - <feature_name>  : Name of the feature (e.g., mean_i8, std_f32, etc.)
 *   - <input_type>    : Input data type (int8_t, int16_t, flt32_t)
 *   - <feature_type>  : Output feature data type (int16_t, int32_t, flt32_t)
 *
 * These functions extract one or more time-domain features from the input signal buffer,
 * storing the results in the output feature buffer. The feature_mask parameter allows
 * selective extraction of features. The pipeline context and argument callback provide
 * additional configuration or arguments for feature extraction.
 *
 * @param[in]  p_input         Pointer to the input signal buffer.
 * @param[in]  num             Number of input samples.
 * @param[out] p_features      Pointer to the output feature buffer.
 * @param[in]  feature_mask    Mask specifying which features to extract.
 * @param[in]  p_pipeline_ctx  Pointer to the pipeline context (may be NULL).
 * @param[in]  get_argument    Callback to get feature-specific arguments (may be NULL).
 * @param[in]  p_argument_ctx  Pointer to argument context for the callback (may be NULL).
 * @return     Number of features extracted and written to p_features.
 *
 * @note
 * - Use NRF_EDGEAI_DECLARE_FEATURE_FUNCTION_I8, NRF_EDGEAI_DECLARE_FEATURE_FUNCTION_I16, or NRF_EDGEAI_DECLARE_FEATURE_FUNCTION_F32
 *   to declare feature functions for int8, int16, or float32 input types, respectively.
 *
 * @example
 * // Declaration for mean feature extraction on int8 input:
 * NRF_EDGEAI_DECLARE_FEATURE_FUNCTION_I8(mean_i8);
 *
 * // Expands to:
 * size32_t nrf_edgeai_feature_mean_i8(
 *     int8_t*                    p_input,
 *     size32_t                     num,
 *     int16_t*                   p_features,
 *     nrf_edgeai_features_mask_t       feature_mask,
 *     void*                           p_pipeline_ctx,
 *     nrf_edgeai_feature_get_arg_cb_t  get_argument,
 *     void*                           p_argument_ctx
 * );
 */

#define NRF_EDGEAI_DECLARE_FEATURE_FUNCTION(feature_name, input_type, feature_type)            \
    size32_t nrf_edgeai_feature_##feature_name(input_type*                     p_input,        \
                                               size32_t                        num,            \
                                               feature_type*                   p_features,     \
                                               nrf_edgeai_features_mask_t      feature_mask,   \
                                               void*                           p_pipeline_ctx, \
                                               nrf_edgeai_feature_get_arg_cb_t get_argument,   \
                                               void*                           p_argument_ctx)

#define NRF_EDGEAI_DECLARE_FEATURE_FUNCTION_I8(feature_name) \
    NRF_EDGEAI_DECLARE_FEATURE_FUNCTION(feature_name, int8_t, int16_t)

#define NRF_EDGEAI_DECLARE_FEATURE_FUNCTION_I16(feature_name) \
    NRF_EDGEAI_DECLARE_FEATURE_FUNCTION(feature_name, int16_t, int32_t)

#define NRF_EDGEAI_DECLARE_FEATURE_FUNCTION_F32(feature_name) \
    NRF_EDGEAI_DECLARE_FEATURE_FUNCTION(feature_name, flt32_t, flt32_t)

#endif /* _NRF_EDGEAI_PRIVATE_INTERFACES_FEATURES_COMMON_H_ */