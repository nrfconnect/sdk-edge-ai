/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_DSP_MEMORY_OPERATIONS_H_
#define _NRF_DSP_MEMORY_OPERATIONS_H_

#include "nrf_dsp_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Read 4 INT8 values as INT32 value, from INT8 vector
 *
 * @param[in] p_input   Pointer to the input vector
 *
 * @return  int32_t  4x INT8 values
 */
__NRF_DSP_STATIC_FORCEINLINE int32_t read_i8x4(int8_t const* p_input)
{
    return ((int32_t)(p_input[3] & 0x0FF) << 24) | ((int32_t)(p_input[2] & 0x0FF) << 16) |
           ((int32_t)(p_input[1] & 0x0FF) << 8) | (p_input[0] & 0x0FF);
}

/**
 * @brief Read 2 INT16 values as INT32 value, from INT16 vector
 *
 * @param[in] p_input   Pointer to the input vector
 *
 * @return  int32_t  2x INT16 values
 */
__NRF_DSP_STATIC_FORCEINLINE int32_t read_i16x2(int16_t const* p_input)
{
    return ((int32_t)p_input[1] << 16) | (p_input[0] & 0x0FFFF);
}

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_MEMORY_OPERATIONS_H_ */