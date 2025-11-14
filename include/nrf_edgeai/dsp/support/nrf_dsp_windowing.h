/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_windowing Windowing functions
 * @{
 * @ingroup nrf_dsp_support
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_SUPPORT_WINDOWING_H_
#define _NRF_DSP_SUPPORT_WINDOWING_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize Hanning window
 * 
 * @param[out]  p_window        Pointer to window for initialization 
 * @param[in]   window_size     Window size in samples
 */
void nrf_dsp_window_hanning_f32(flt32_t* p_window, uint16_t window_size);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_SUPPORT_WINDOWING_H_ */

/**
 * @}
 */
