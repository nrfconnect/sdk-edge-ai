/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_DSP_DEFINITIONS_H_
#define _NRF_DSP_DEFINITIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>
#include <nrf_edgeai/dsp/platform/nrf_dsp_platform.h>

#define NRF_DSP_INT8_MAX           ((int8_t)(0x7F))
#define NRF_DSP_INT8_MIN           ((int8_t)(0x80))
#define NRF_DSP_INT8_ABSMAX        ((int8_t)(0x7F))
#define NRF_DSP_INT8_ABSMIN        ((int8_t)0)
#define NRF_DSP_INT8_QFACTOR       (127)
#define NRF_DSP_INT8_SIGN_BIT      ((int8_t)1 << 7)
#define NRF_DSP_INT8_CHECK_SIGN(x) (bool)((x) & NRF_DSP_INT8_SIGN_BIT)

#define NRF_DSP_INT16_MAX           ((int16_t)(0x7FFF))
#define NRF_DSP_INT16_MIN           ((int16_t)(0x8000))
#define NRF_DSP_INT16_ABSMAX        ((int16_t)(0x7FFF))
#define NRF_DSP_INT16_ABSMIN        ((int16_t)0)
#define NRF_DSP_INT16_QFACTOR       ((int16_t)32767)
#define NRF_DSP_INT16_SIGN_BIT      ((int16_t)1 << 15)
#define NRF_DSP_INT16_CHECK_SIGN(x) (bool)((x) & NRF_DSP_INT16_SIGN_BIT)

#define NRF_DSP_UINT16_MAX ((uint16_t)(0xFFFF))
#define NRF_DSP_UINT16_MIN ((uint16_t)(0x0000))

#define NRF_DSP_INT32_MAX           ((int32_t)(0x7FFFFFFFL))
#define NRF_DSP_INT32_MIN           ((int32_t)(0x80000000L))
#define NRF_DSP_INT32_ABSMAX        ((int32_t)(0x7FFFFFFFL))
#define NRF_DSP_INT32_ABSMIN        ((int32_t)0)
#define NRF_DSP_INT32_QFACTOR       ((int32_t)2147483648)
#define NRF_DSP_INT32_SIGN_BIT      ((int32_t)1 << 31)
#define NRF_DSP_INT32_CHECK_SIGN(x) (bool)((x) & NRF_DSP_INT32_SIGN_BIT)

// smallest such that 1.0+NRF_DSP_F32_EPSILON != 1.0
#define NRF_DSP_F32_EPSILON       ((flt32_t)1.192092896e-07F)
#define NRF_DSP_F32_MAX           ((flt32_t)3.402823466e+38F)
#define NRF_DSP_F32_MIN           -NRF_DSP_F32_MAX
#define NRF_DSP_F32_NP_MIN        ((flt32_t)1.175494351e-38F)  // min normalized positive value
#define NRF_DSP_F32_CHECK_SIGN(x) (bool)(signbit(x))

#define NRF_DSP_USE_LOOPUNROLL    (0)
#define NRF_DSP_USE_MATH_ROUNDING (0)

#define NRF_DSP_FLOAT_TO_INT_FACTOR      (100U)
#define NRF_DSP_PERCENTAGE_TO_INT_FACTOR (1000U)

#define CONCAT2x(a, b) a##_##b
#define CONCAT2(a, b)  CONCAT2x(a, b)

#define CONCAT3x(a, b, c) a##_##b##_##c
#define CONCAT3(a, b, c)  CONCAT3x(a, b, c)

#define CONCAT4x(a, b, c, d) a##_##b##_##c##_##d
#define CONCAT4(a, b, c, d)  CONCAT4x(a, b, c, d)

#define FUNCTION_NAME(name, postfix) CONCAT2(name, postfix)

#define CLIP_MINMAX(x, min, max) (((x) > max) ? max : (((x) < min) ? min : (x)))

#define IS_POWER_OF_TWO(A) (((A) != 0) && ((((A) - 1) & (A)) == 0))

#ifndef M_PI
#define M_PI (float)3.14159265358979323846f  // pi
#endif

#endif /* _NRF_DSP_DEFINITIONS_H_ */