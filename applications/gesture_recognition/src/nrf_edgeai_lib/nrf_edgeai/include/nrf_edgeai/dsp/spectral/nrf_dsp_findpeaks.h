/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_spectral_findpeaks Find Peaks
 * @{
 * @ingroup nrf_dsp_spectral
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_SPECTRAL_FINDPEAKS_H_
#define _NRF_DSP_SPECTRAL_FINDPEAKS_H_

#include <nrf_edgeai/dsp/platform/nrf_dsp_platform.h>
#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Finds peaks indexes in the floating-point vector
 * 
 * @param[in] p_input           Pointer to the input vector
 * @param[in] num               Number of samples in the input vector
 * @param[in] min_peak_height   Minimum peak height to find
 * @param[in] min_peak_distance Minimum distance between peaks 
 * @param[out] peaks_indices    Pointer to the output array with founded highest peak indicies, 
 *                              if peaks corresponding search coniditions (min_peak_height, min_peak_distance) was not found, 
 *                              -1 will be written as output
 * @param[in] peaks_indices_num Number of highest peaks to find
 */
void nrf_dsp_findpeaks_f32(flt32_t  p_input[],
                           uint16_t num,
                           flt32_t  min_peak_height,
                           uint16_t min_peak_distance,
                           int16_t  peaks_indices[],
                           uint16_t peaks_indices_num);

/**
 * @brief Finds peaks indexes in the INT8 fixed-point
 * 
 * @param[in] p_input           Pointer to the input vector
 * @param[in] num               Number of samples in the input vector
 * @param[in] min_peak_height   Minimum peak height to find
 * @param[in] min_peak_distance Minimum distance between peaks 
 * @param[out] peaks_indices    Pointer to the output array with founded highest peak indicies, 
 *                              if peaks corresponding search coniditions (min_peak_height, min_peak_distance) was not found, 
 *                              -1 will be written as output
 * @param[in] peaks_indices_num Number of highest peaks to find
 */
void nrf_dsp_findpeaks_i8(int8_t   p_input[],
                          uint16_t num,
                          int8_t   min_peak_height,
                          uint16_t min_peak_distance,
                          int16_t  peaks_indices[],
                          uint16_t peaks_indices_num);

/**
 * @brief Finds peaks indexes in the INT16 fixed-point
 * 
 * @param[in] p_input           Pointer to the input vector
 * @param[in] num               Number of samples in the input vector
 * @param[in] min_peak_height   Minimum peak height to find
 * @param[in] min_peak_distance Minimum distance between peaks 
 * @param[out] peaks_indices    Pointer to the output array with founded highest peak indicies, 
 *                              if peaks corresponding search coniditions (min_peak_height, min_peak_distance) was not found, 
 *                              -1 will be written as output
 * @param[in] peaks_indices_num Number of highest peaks to find
 */
void nrf_dsp_findpeaks_i16(int16_t  p_input[],
                           uint16_t num,
                           int16_t  min_peak_height,
                           uint16_t min_peak_distance,
                           int16_t  peaks_indices[],
                           uint16_t peaks_indices_num);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_SPECTRAL_FINDPEAKS_H_ */

/**
 * @}
 */
