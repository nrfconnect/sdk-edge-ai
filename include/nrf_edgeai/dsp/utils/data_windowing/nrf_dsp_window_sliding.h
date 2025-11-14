/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_window_sliding Sliding window
 * @{
 * @ingroup nrf_dsp_utils
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_UTILS_WINDOW_SLIDING_H_
#define _NRF_DSP_UTILS_WINDOW_SLIDING_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Sliding window callback function. Called when window buffer is ready
 * 
 * @param p_window          Pointer to window buffer
 * @param window_size       Window buffer size in samples
 * @param p_user_ctx        Pointer to user context
 */
typedef void (*nrf_dsp_window_sliding_cb_t)(void* p_window, uint16_t window_size, void* p_user_ctx);

/**
 * @brief Sliding window context block
 */
typedef struct nrf_dsp_window_sliding_s
{
    /** Pointer to window buffer */
    void* p_window;

    /** On window ready user callback */
    nrf_dsp_window_sliding_cb_t on_ready_cb;

    /** Current window buffer fill */
    uint32_t current_fill;

    /** Sample size in bytes */
    uint16_t sample_size;

    /** Window buffer size in samples */
    uint16_t window_size;

    /** Window hop in samples. Must be less or equal @window_size*/
    uint16_t window_hop;

    /** Pointer to user context. Can be accessed in callback function */
    void* p_user_ctx;
} nrf_dsp_window_sliding_t;

/**
 * @brief Initialize sliding window context
 * 
 * @param[out]  p_ctx         Context to be initialized 
 * @param[in]   p_window      Pointer to window buffer
 * @param[in]   window_size   Window buffer size in samples
 * @param[in]   sample_size   Sample size in bytes 
 * @param[in]   window_hop    Window hop in samples. Must be less or equal @window_size
 * @param[in]   cb            On window ready user callback
 * @param[in]   p_user_ctx    Pointer to user context
 *  
 * @return Operation status code: -1 - Null pointer error, -2 - Invalid argument error, 0 - Success
 */
int8_t nrf_dsp_window_sliding_init(nrf_dsp_window_sliding_t*   p_ctx,
                                   void*                       p_window,
                                   uint16_t                    window_size,
                                   uint16_t                    sample_size,
                                   uint16_t                    window_hop,
                                   nrf_dsp_window_sliding_cb_t cb,
                                   void*                       p_user_ctx);

/**
 * @brief Feeds samples to the sliding window
 * 
 * @param[in, out]  p_ctx         Sliding window context, should be initialized first @ref nrf_dsp_window_sliding_init
 * @param[in]       p_samples     Input data samples
 * @param[in]       samples_num   Number of data samples
 */
void nrf_dsp_window_sliding_feed(nrf_dsp_window_sliding_t* p_ctx,
                                 void*                     p_samples,
                                 uint16_t                  samples_num);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_UTILS_WINDOW_SLIDING_H_ */

/**
 * @}
 */
