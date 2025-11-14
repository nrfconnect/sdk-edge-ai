/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_window_flatsliding Flatten and sliding by columns window 
 * @{
 * @ingroup nrf_dsp_utils
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_UTILS_WINDOW_FLATTEN_SLIDING_H_
#define _NRF_DSP_UTILS_WINDOW_FLATTEN_SLIDING_H_

#include "nrf_dsp_window_flatten.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nrf_dsp_window_flatsliding_s
{
    /** Flatten window context */
    nrf_dsp_window_flatten_t flatten;

    /** Window shift in samples, by which the window will be shifted when it is filled. 
     * Must be less or equal @flatten.max_samples_num
     */
    uint16_t window_shift;

    /** Flag if window should be shifted by window_shift on the next window feed */
    bool is_shift_pending;
} nrf_dsp_window_flatsliding_t;

/**
 * @brief Initialize flatten by colums sliding window context type 
 * 
 * @param[out]   p_ctx               Pointer to the window context which will be initialized
 * @param[in]    p_window            Pointer to window buffer
 * @param[in]    max_samples_num     Maximum number of feature samples in window buffer, feature sample = 1 * uniq_features_num
 * @param[in]    uniq_features_num   Number of unique features in feature sample, e.g feature sample = {x, y, z}, then uniq_features_num = 3
 * @param[in]    window_shift        Window shift in samples by which the window will be shifted when it is filled. Must be less or equal @max_samples_num
 * 
 * @return Operation status code @ref neuton_status_t
 */
int8_t nrf_dsp_window_flatsliding_init(nrf_dsp_window_flatsliding_t* p_ctx,
                                       void*                         p_window,
                                       const uint16_t                max_samples_num,
                                       const size_t                  uniq_features_num,
                                       const uint16_t                window_shift);

/**
 * @brief Feeds f32 samples to the floating-point flatten by colums window,
 *        with ordering by feature columns, e.g p_input = [x,y,z,x1,y1,z1] -> p_window = [x,x1,.., y,y1,.., z,z1,..]
 * 
 * @note If window was filled, next call will shift window to the right for each unique feature in window by p_ctx->window_shift
 * 
 * @param[in, out]  p_ctx       Pointer to the window context, should be initialized first @ref nrf_dsp_window_flatsliding_init
 * @param[in]       p_input     Pointer to the feature sample buffer
 * @param[in]       num         Number of samples in the input buffer, num = buffer_size / p_ctx->uniq_features_num
 *  
 * @return uint16_t Number of remaining samples to collect window
 */
uint16_t nrf_dsp_window_flatsliding_feed_f32(nrf_dsp_window_flatsliding_t* p_ctx,
                                             const flt32_t*                p_input,
                                             uint16_t                      num);

/**
 * @brief Feeds INT8 samples to the INT8 flatten by colums window,
 *        with ordering by feature columns, e.g p_input = [x,y,z,x1,y1,z1] -> p_window = [x,x1,.., y,y1,.., z,z1,..]
 * 
 * @param[in, out]  p_ctx       Pointer to the window context, should be initialized first @ref nrf_dsp_window_flatsliding_init
 * @param[in]       p_input     Pointer to the feature sample buffer
 * @param[in]       num         Number of samples in the input buffer, num = buffer_size / p_ctx->uniq_features_num
 *  
 * @return uint16_t Number of remaining samples to collect window
 */
uint16_t nrf_dsp_window_flatsliding_feed_i8(nrf_dsp_window_flatsliding_t* p_ctx,
                                            const int8_t*                 p_input,
                                            uint16_t                      num);

/**
 * @brief Feeds INT16 samples to the INT16 flatten by colums window,
 *        with ordering by feature columns, e.g p_input = [x,y,z,x1,y1,z1] -> p_window = [x,x1,.., y,y1,.., z,z1,..]
 * 
 * @param[in, out]  p_ctx       Pointer to the window context, should be initialized first @ref nrf_dsp_window_flatsliding_init
 * @param[in]       p_input     Pointer to the feature sample buffer
 * @param[in]       num         Number of samples in the input buffer, num = buffer_size / p_ctx->uniq_features_num
 *  
 * @return uint16_t Number of remaining samples to collect window
 */
uint16_t nrf_dsp_window_flatsliding_feed_i16(nrf_dsp_window_flatsliding_t* p_ctx,
                                             const int16_t*                p_input,
                                             uint16_t                      num);

/**
 * @brief Feeds f32 samples to the floating-point flatten by colums window,
 *        with ordering by feature columns, e.g p_input = [x,y,z,x1,y1,z1] -> p_window = [x,x1,.., y,y1,.., z,z1,..],
 *        usage of inputs column is determined by p_input_usage_mask where each bit represents each unique feature
 * 
 * @param[in, out]  p_ctx           Pointer to the window context, should be initialized first @ref nrf_dsp_window_flatsliding_init
 * @param[in]       p_input         Pointer to the feature sample buffer
 * @param[in]       num             Number of samples in the input buffer, num = buffer_size / p_ctx->uniq_features_num
 * @param[in]       p_usage_mask    Input features usage mask
 *  
 * @return uint16_t Number of remaining samples to collect window
 */
uint16_t nrf_dsp_window_flatsliding_feed_bymask_f32(nrf_dsp_window_flatsliding_t* p_ctx,
                                                    const flt32_t*                p_input,
                                                    uint16_t                      num,
                                                    const uint8_t*                p_usage_mask);

/**
 * @brief Feeds INT8 samples to the floating-point flatten by colums window,
 *        with ordering by feature columns, e.g p_input = [x,y,z,x1,y1,z1] -> p_window = [x,x1,.., y,y1,.., z,z1,..],
 *        usage of inputs column is determined by p_input_usage_mask where each bit represents each unique feature
 * 
 * @param[in, out]  p_ctx           Pointer to the window context, should be initialized first @ref nrf_dsp_window_flatsliding_init
 * @param[in]       p_input         Pointer to the feature sample buffer
 * @param[in]       num             Number of samples in the input buffer, num = buffer_size / p_ctx->uniq_features_num
 * @param[in]       p_usage_mask    Input features usage mask
 *  
 * @return uint16_t Number of remaining samples to collect window
 */
uint16_t nrf_dsp_window_flatsliding_feed_bymask_i8(nrf_dsp_window_flatsliding_t* p_ctx,
                                                   const int8_t*                 p_input,
                                                   uint16_t                      num,
                                                   const uint8_t*                p_usage_mask);

/**
 * @brief Feeds INT16 samples to the floating-point flatten by colums window,
 *        with ordering by feature columns, e.g p_input = [x,y,z,x1,y1,z1] -> p_window = [x,x1,.., y,y1,.., z,z1,..],
 *        usage of inputs column is determined by p_input_usage_mask where each bit represents each unique feature
 * 
 * @param[in, out]  p_ctx           Pointer to the window context, should be initialized first @ref nrf_dsp_window_flatsliding_init
 * @param[in]       p_input         Pointer to the feature sample buffer
 * @param[in]       num             Number of samples in the input buffer, num = buffer_size / p_ctx->uniq_features_num
 * @param[in]       p_usage_mask    Input features usage mask
 *  
 * @return uint16_t Number of remaining samples to collect window
 */
uint16_t nrf_dsp_window_flatsliding_feed_bymask_i16(nrf_dsp_window_flatsliding_t* p_ctx,
                                                    const int16_t*                p_input,
                                                    uint16_t                      num,
                                                    const uint8_t*                p_usage_mask);
#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_UTILS_WINDOW_FLATTEN_SLIDING_H_ */

/**
 * @}
 */
