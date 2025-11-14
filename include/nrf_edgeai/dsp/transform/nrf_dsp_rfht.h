/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_transform_rfht Fast Hartley Transform Real
 * @{
 * @ingroup nrf_dsp_transform
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_TRANSFORM_REAL_FHT_H_
#define _NRF_DSP_TRANSFORM_REAL_FHT_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Fractional bits used for fractional part of the value */
#define NRF_DSP_FRACT_BITS_7  7
#define NRF_DSP_FRACT_BITS_15 15

/**
 * @brief Instance structure for the RFHT/RIFHT functions.
 */
typedef struct nrf_dsp_rfht_instance_s
{
    /**  Pointer to sin() table, with size of window_size / 4 elements */
    void* p_sin_table;

    /**  Pointer to tg() table, with size of window_size / 4 elements */
    void* p_tg_table;

    /**  Pointer to index reordering array, with size of window_size elements */
    uint16_t* p_rev_bit_index;

    /**  Pointer to RFHT result buffer, with size of window_size elements */
    void* p_window;

    /** Window buffer size to store result of FHT, should be powers of two in value (128/256/512/etc) */
    uint16_t window_size;

    /** Cached value of log2(window_size) */
    uint16_t log_n;
} nrf_dsp_rfht_instance_t;

/**
 * @brief Initialize instance structure for the RFHT/RIFHT functions
 * 
 * @param[out]    p_inst                Pointer to instance structure to initialize
 * @param[in]     p_sin_table           Pointer to sin() table, with size of window_size / 4 elements
 * @param[in]     p_tg_table            Pointer to tg() table, with size of window_size / 4 elements
 * @param[in]     p_rev_bit_index       Pointer to index reordering array, with size of window_size elements
 * @param[in]     p_window              Pointer to RFHT result buffer, with size of window_size elements
 * @param[in]     window_size           Window buffer size to store result of FHT, should be powers of two in value (128/256/512/etc)
 * @param[in]     log_n                 Value of log2(window_size)
 * 
 * @return 0 on success, otherwise error
 */
int8_t nrf_dsp_rfht_init(nrf_dsp_rfht_instance_t* p_inst,
                         const void*              p_sin_table,
                         const void*              p_tg_table,
                         const uint16_t*          p_rev_bit_index,
                         void*                    p_window,
                         const uint16_t           window_size,
                         const uint16_t           log_n);

/**
 * @brief Perform RFHT on the floating-point input vector
 * @note p_inst->p_window should be point to the input data vector, RFHT is performed in-place
 * 
 * @param[in] p_inst     Pointer to the RFHT instance
 */
void nrf_dsp_rfht_f32(const nrf_dsp_rfht_instance_t* p_inst);

/**
 * @brief Perform RFHT on the INT16 input vector
 * @note p_inst->p_window should be point to the input data vector, RFHT is performed in-place
 * 
 * @param[in] p_inst     Pointer to the RFHT instance
 * 
 * @return Number of fraction bit for for the fractional part of the value in f32 format
 */
uint16_t nrf_dsp_rfht_i16(const nrf_dsp_rfht_instance_t* p_inst);

/**
 * @brief Perform RFHT on the INT8 input vector
 * @note p_inst->p_window should be point to the input data vector, RFHT is performed in-place
 * 
 * @param[in] p_inst     Pointer to the RFHT instance
 * 
* @return Number of fraction bit for for the fractional part of the value in f32 format
 */
uint16_t nrf_dsp_rfht_i8(const nrf_dsp_rfht_instance_t* p_inst);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_TRANSFORM_REAL_FHT_H_ */

/**
 * @}
 */
