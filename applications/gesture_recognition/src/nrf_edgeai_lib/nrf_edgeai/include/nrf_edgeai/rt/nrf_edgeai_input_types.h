/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_EDGEAI_INPUT_TYPES_H_
#define _NRF_EDGEAI_INPUT_TYPES_H_

#include <nrf_edgeai/nrf_edgeai_ctypes.h>
#include <nrf_edgeai/dsp/utils/data_windowing/nrf_dsp_data_windowing.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Supported input data types for neural network
 */
typedef enum nrf_edgeai_input_type_e
{
    NRF_EDGEAI_INPUT_I8  = 1, /**< 8-bit signed integer input */
    NRF_EDGEAI_INPUT_I16 = 2, /**< 16-bit signed integer input */
    NRF_EDGEAI_INPUT_F32 = 4, /**< 32-bit floating point input */
} nrf_edgeai_input_type_t;

/**
 * @brief Scaling parameters for 8-bit integer input
 */
typedef struct nrf_edgeai_input_scale_i8_s
{
    const int8_t* p_min; /**< Pointer to minimum value for scaling */
    const int8_t* p_max; /**< Pointer to maximum value for scaling */
} nrf_edgeai_input_scale_i8_t;

/**
 * @brief Scaling parameters for 16-bit integer input
 */
typedef struct nrf_edgeai_input_scale_i16_s
{
    const int16_t* p_min; /**< Pointer to minimum value for scaling */
    const int16_t* p_max; /**< Pointer to maximum value for scaling */
} nrf_edgeai_input_scale_i16_t;

/**
 * @brief Scaling parameters for 32-bit floating point input
 */
typedef struct nrf_edgeai_input_scale_f32_s
{
    const flt32_t* p_min; /**< Pointer to minimum value for scaling */
    const flt32_t* p_max; /**< Pointer to maximum value for scaling */
} nrf_edgeai_input_scale_f32_t;

/**
 * @brief Union of input scaling parameters for all supported types
 */
typedef union nrf_edgeai_input_scale_u
{
    nrf_edgeai_input_scale_i8_t  i8;  /**< Scaling for 8-bit integer input */
    nrf_edgeai_input_scale_i16_t i16; /**< Scaling for 16-bit integer input */
    nrf_edgeai_input_scale_f32_t f32; /**< Scaling for 32-bit float input */
} nrf_edgeai_input_scale_t;

/**
 * @brief Union of window context types for input data
 */
typedef union nrf_edgeai_window_ctx_u
{
    nrf_dsp_window_flatsliding_t sliding;  /**< Sliding window context */
    nrf_dsp_window_flatten_t     discrete; /**< Discrete window context */
} nrf_edgeai_window_ctx_t;

/**
 * @brief Input features processing context for neural network
 */
typedef struct nrf_edgeai_input_s
{
    nrf_edgeai_input_scale_t scale; /**< Input scaling parameters */

    const uint8_t* p_used_for_lags_mask; /**< Mask for features used for lags */
    const uint8_t* p_usage_mask;         /**< Mask for features usage */

    const nrf_edgeai_input_type_t type; /**< Input data type */
    /**
     * @brief Number of unique input features (axes) in input window,
     *        e.g. for Accel_X, Accel_Y, Accel_Z the num = 3
     */
    const uint16_t unique_num;
    const uint16_t unique_num_used;   /**< Number of unique features used */
    const uint16_t unique_scales_num; /**< Number of unique scales */
    const uint16_t window_size;       /**< Size of input window in samples */
    const uint16_t window_shift;      /**< Shift of input window, while sliding */
    const uint8_t  subwindow_num;     /**< Number of subwindows */
    /**
     * @brief Pointer to the input data buffer
     */
    union
    {
        void*    p_void; /**< Generic pointer to window memory */
        int8_t*  p_i8;   /**< Pointer to 8-bit integer window memory */
        int16_t* p_i16;  /**< Pointer to 16-bit integer window memory */
        flt32_t* p_f32;  /**< Pointer to 32-bit float window memory */
    } window_memory;

    nrf_edgeai_window_ctx_t* p_window_ctx; /**< Pointer to window context */
} nrf_edgeai_input_t;

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_INPUT_TYPES_H_ */