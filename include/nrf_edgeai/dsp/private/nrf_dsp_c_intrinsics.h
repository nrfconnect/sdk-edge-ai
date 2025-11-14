/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_DSP_C_INTRINSICS_H_
#define _NRF_DSP_C_INTRINSICS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>
#include "nrf_dsp_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

#define __NRF_DSP_CLZ  __CLZ
#define __NRF_DSP_SSAT __SSAT
#define __NRF_DSP_USAT __USAT
#define __NRF_DSP_ROR  __ROR

/**
  * @brief   Count leading zeros
  * @param [in]  value  Value to count the leading zeros
  * @return             number of leading zeros in value
  */
__NRF_DSP_STATIC_FORCEINLINE uint8_t __CLZ(uint32_t data)
{
    if (data == 0U) { return 32U; }

    uint32_t count = 0U;
    uint32_t mask  = 0x80000000U;

    while ((data & mask) == 0U)
    {
        count += 1U;
        mask = mask >> 1U;
    }
    return count;
}

/**
  * @brief   Signed Saturate
  * @details Saturates a signed value.
  * @param [in]  value  Value to be saturated
  * @param [in]  sat  Bit position to saturate to (1..32)
  * @return             Saturated value
  */
__NRF_DSP_STATIC_FORCEINLINE int32_t __SSAT(int32_t val, uint32_t sat)
{
    if ((sat >= 1U) && (sat <= 32U))
    {
        const int32_t max = (int32_t)((1U << (sat - 1U)) - 1U);
        const int32_t min = -1 - max;
        if (val > max)
            return max;
        else if (val < min)
            return min;
    }
    return val;
}

/**
 * @brief Signed Saturate for integer 16-bit
 */
#define __SSAT16(x) CLIP_MINMAX((x), NRF_DSP_INT16_MIN, NRF_DSP_INT16_MAX)

/**
  * @brief   Unsigned Saturate
  * @details Saturates an unsigned value.
  * @param [in]  value  Value to be saturated
  * @param [in]  sat  Bit position to saturate to (0..31)
  * @return             Saturated value
  */
__NRF_DSP_STATIC_FORCEINLINE uint32_t __USAT(int32_t val, uint32_t sat)
{
    if (sat <= 31U)
    {
        const uint32_t max = ((1U << sat) - 1U);
        if (val > (int32_t)max)
            return max;
        else if (val < 0)
            return 0U;
    }
    return (uint32_t)val;
}

/**
 * @brief   Rotate Right in unsigned value (32 bit)
 * @details Rotate Right (immediate) provides the value of the contents of a register rotated by a variable number of bits.
 * @param [in]    op1  Value to rotate
 * @param [in]    op2  Number of Bits to rotate
 * @return               Rotated value
 */
__NRF_DSP_STATIC_FORCEINLINE uint32_t __ROR(uint32_t op1, uint32_t op2)
{
    op2 %= 32U;
    return (op2 == 0U) ? op1 : ((op1 >> op2) | (op1 << (32U - op2)));
}

/**
  * @brief Clips INT64 to INT32 values.
  */
__NRF_DSP_STATIC_FORCEINLINE int32_t clip_i64_to_i32(int64_t x)
{
    return ((int32_t)(x >> 32) != ((int32_t)x >> 31)) ? ((0x7FFFFFFF ^ ((int32_t)(x >> 63)))) :
                                                        (int32_t)x;
}

/**
 * @brief Clips INT64 to INT16 values.
 */
__NRF_DSP_STATIC_FORCEINLINE int16_t clip_i64_to_i16(int64_t x)
{
    return ((int32_t)(x >> 32) != ((int32_t)x >> 31)) ? ((0x7FFF ^ ((int16_t)(x >> 63)))) :
                                                        (int16_t)(x >> 15);
}

/**
 * @brief Clips INT32 to INT8 values.
 */
__NRF_DSP_STATIC_FORCEINLINE int8_t clip_i32_to_i8(int32_t x)
{
    return ((int32_t)(x >> 24) != ((int32_t)x >> 23)) ? ((0x7F ^ ((int8_t)(x >> 31)))) : (int8_t)x;
}

/**
 * @brief Clips INT32 to INT16 values.
 */
__NRF_DSP_STATIC_FORCEINLINE int16_t clip_i32_to_i16(int32_t x)
{
    return ((int32_t)(x >> 16) != ((int32_t)x >> 15)) ? ((0x7FFF ^ ((int16_t)(x >> 31)))) :
                                                        (int16_t)x;
}

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_C_INTRINSICS_H_ */
