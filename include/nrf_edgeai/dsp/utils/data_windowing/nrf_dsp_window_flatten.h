/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_window_flatten Flatten by columns window
 * @{
 * @ingroup nrf_dsp_utils
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_UTILS_WINDOW_FLATTEN_H_
#define _NRF_DSP_UTILS_WINDOW_FLATTEN_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Flatten by columns window context
 * 
 */
typedef struct nrf_dsp_window_flatten_s
{
    /** Pointer to window buffer, 
     * nrf_dsp_window_flatten_feed_xx() using pointer according it's type 
     */
    union
    {
        int8_t*  i8;
        int16_t* i16;
        int32_t* i32;
        flt32_t* f32;
        void*    generic;
    } p_window;

    /** Maximum number of feature samples in window buffer */
    uint16_t max_samples_num;

    /** Current sample number */
    uint16_t current_sample;

    /** Number of unique features in feature sample */
    uint16_t uniq_features_num;

    /** Number of unique features actually collected in windows, 
     * equal to uniq_features_num by default,
     * only changed in @ref nrf_dsp_window_flatten_feed_bymask 
     */
    uint16_t uniq_features_collected;
} nrf_dsp_window_flatten_t;

/**
 * @brief Initialize flatten by columns window context 
 * 
 * @param[out]      p_ctx               Pointer to the window context which will be initialized
 * @param[in]       p_window            Pointer to window buffer
 * @param[in]       max_samples_num     Maximum number of feature samples in window buffer, feature sample = 1 * uniq_features_num
 * @param[in]       uniq_features_num   Number of unique features in feature sample, e.g feature sample = {x, y, z}, then uniq_features_num = 3
 * 
 * @return Operation status code: -1 - Null pointer error, -2 - Invalid argument error, 0 - Success
 */
int8_t nrf_dsp_window_flatten_init(nrf_dsp_window_flatten_t* p_ctx,
                                   void*                     p_window,
                                   const uint16_t            max_samples_num,
                                   const size_t              uniq_features_num);

/**
 * @brief Feeds f32 samples to the floating-point flatten by columns window,
 *        with ordering by feature columns, e.g p_input = [x,y,z,x1,y1,z1] -> p_window = [x,x1,.., y,y1,.., z,z1,..]
 * 
 * @param[in, out]  p_ctx       Pointer to the window context, should be initialized first @ref nrf_dsp_window_flatten_init
 * @param[in]       p_input     Pointer to the feature sample buffer
 * @param[in]       num         Number of samples in the input buffer, num = buffer_size / p_ctx->uniq_features_num
 *  
 * @return uint16_t Number of remaining samples to collect window
 */
uint16_t nrf_dsp_window_flatten_feed_f32(nrf_dsp_window_flatten_t* p_ctx,
                                         const flt32_t*            p_input,
                                         uint16_t                  num);

/**
 * @brief Feeds INT8 samples to the INT8 flatten by columns window,
 *        with ordering by feature columns, e.g p_input = [x,y,z,x1,y1,z1] -> p_window = [x,x1,.., y,y1,.., z,z1,..]
 * 
 * @param[in, out]  p_ctx       Pointer to the window context, should be initialized first @ref nrf_dsp_window_flatten_init
 * @param[in]       p_input     Pointer to the feature sample buffer
 * @param[in]       num         Number of samples in the input buffer, num = buffer_size / p_ctx->uniq_features_num
 *  
 * @return uint16_t Number of remaining samples to collect window
 */
uint16_t nrf_dsp_window_flatten_feed_i8(nrf_dsp_window_flatten_t* p_ctx,
                                        const int8_t*             p_input,
                                        uint16_t                  num);

/**
 * @brief Feeds INT16 samples to the INT16 flatten by columns window,
 *        with ordering by feature columns, e.g p_input = [x,y,z,x1,y1,z1] -> p_window = [x,x1,.., y,y1,.., z,z1,..]
 * 
 * @param[in, out]  p_ctx       Pointer to the window context, should be initialized first @ref nrf_dsp_window_flatten_init
 * @param[in]       p_input     Pointer to the feature sample buffer
 * @param[in]       num         Number of samples in the input buffer, num = buffer_size / p_ctx->uniq_features_num
 *  
 * @return uint16_t Number of remaining samples to collect window
 */
uint16_t nrf_dsp_window_flatten_feed_i16(nrf_dsp_window_flatten_t* p_ctx,
                                         const int16_t*            p_input,
                                         uint16_t                  num);

/**
 * @brief Feeds f32 samples to the floating-point flatten by columns window,
 *        with ordering by feature columns, e.g p_input = [x,y,z,x1,y1,z1] -> p_window = [x,x1,.., y,y1,.., z,z1,..],
 *        usage of inputs column is determined by p_input_usage_mask where each bit represents each unique feature
 * 
 * @param[in, out]  p_ctx           Pointer to the window context, should be initialized first @ref nrf_dsp_window_flatten_init
 * @param[in]       p_input         Pointer to the feature sample buffer
 * @param[in]       num             Number of samples in the input buffer, num = buffer_size / p_ctx->uniq_features_num
 * @param[in]       p_usage_mask    Input features usage mask
 *  
 * @return uint16_t Number of remaining samples to collect window
 */
uint16_t nrf_dsp_window_flatten_feed_bymask_f32(nrf_dsp_window_flatten_t* p_ctx,
                                                const flt32_t*            p_input,
                                                uint16_t                  num,
                                                const uint8_t*            p_usage_mask);

/**
 * @brief Feeds INT8 samples to the INT8 flatten by columns window,
 *        with ordering by feature columns, e.g p_input = [x,y,z,x1,y1,z1] -> p_window = [x,x1,.., y,y1,.., z,z1,..],
 *        usage of inputs column is determined by p_input_usage_mask where each bit represents each unique feature
 * 
 * @param[in, out]  p_ctx           Pointer to the window context, should be initialized first @ref nrf_dsp_window_flatten_init
 * @param[in]       p_input         Pointer to the feature sample buffer
 * @param[in]       num             Number of samples in the input buffer, num = buffer_size / p_ctx->uniq_features_num
 * @param[in]       p_usage_mask    Input features usage mask
 *  
 * @return uint16_t Number of remaining samples to collect window
 */
uint16_t nrf_dsp_window_flatten_feed_bymask_i8(nrf_dsp_window_flatten_t* p_ctx,
                                               const int8_t*             p_input,
                                               uint16_t                  num,
                                               const uint8_t*            p_usage_mask);

/**
 * @brief Feeds INT16 samples to the INT8 flatten by columns window,
 *        with ordering by feature columns, e.g p_input = [x,y,z,x1,y1,z1] -> p_window = [x,x1,.., y,y1,.., z,z1,..],
 *        usage of inputs column is determined by p_input_usage_mask where each bit represents each unique feature
 * 
 * @param[in, out]  p_ctx           Pointer to the window context, should be initialized first @ref nrf_dsp_window_flatten_init
 * @param[in]       p_input         Pointer to the feature sample buffer
 * @param[in]       num             Number of samples in the input buffer, num = buffer_size / p_ctx->uniq_features_num
 * @param[in]       p_usage_mask    Input features usage mask
 *  
 * @return uint16_t Number of remaining samples to collect window
 */
uint16_t nrf_dsp_window_flatten_feed_bymask_i16(nrf_dsp_window_flatten_t* p_ctx,
                                                const int16_t*            p_input,
                                                uint16_t                  num,
                                                const uint8_t*            p_usage_mask);
#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_UTILS_WINDOW_FLATTEN_H_ */

/**
 * @}
 */
