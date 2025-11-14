/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_f24 Float 24
 * @{
 * @ingroup nrf_dsp_support
 *
 * @brief Custom numeric library for integer calculation with high correlation to float
 *
 */
#ifndef _NRF_DSP_SUPPORT_FLOAT24_H_
#define _NRF_DSP_SUPPORT_FLOAT24_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>
#include <nrf_edgeai/dsp/platform/nrf_dsp_platform.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NRF_DSP_F24_SIGNED   0x40
#define NRF_DSP_F24_UNSIGNED 0
#define NRF_DSP_F24_SIGN(x)  ((uint8_t)(x.exps & NRF_DSP_F24_SIGNED))

// ///////////////////////////////////////////////////////////////////////////
/** Short name macro */
#define f24_init    nrf_dsp_f24_init
#define i16_to_f24  nrf_dsp_i16_to_f24
#define i8_to_f24   nrf_dsp_i8_to_f24
#define f24_to_i32  nrf_dsp_f24_to_i32
#define f24_mul     nrf_dsp_f24_mul
#define f24_add     nrf_dsp_f24_add
#define f24_addx    nrf_dsp_f24_addx
#define f24_div     nrf_dsp_f24_div
#define f24_sqrt    nrf_dsp_f24_sqrt
#define f24_get_exp nrf_dsp_f24_get_exp

// ///////////////////////////////////////////////////////////////////////////

/**
 * @brief Container for Float24 representation
 * 
 */
typedef struct nrf_dsp_f24_s
{
    /** Mantissa*/
    uint16_t man;

    /** Exponenta with sign */
    uint8_t exps;
} nrf_dsp_f24_t;

/**
 * @brief Convert INT16 to Float24
 */
nrf_dsp_f24_t nrf_dsp_i16_to_f24(int16_t x);

/**
 * @brief Convert INT8 to Float24
 */
nrf_dsp_f24_t nrf_dsp_i8_to_f24(int8_t x);

/**
 * @brief Convert Float24 to INT32 value
 * 
 * @param[in] x                 Float24 value to convert
 * @param[in] required_exp      Required exponent bit shift, affect fractional part accuracy
 * 
 * @return int32_t Converted INT32 value
 */
int32_t nrf_dsp_f24_to_i32(nrf_dsp_f24_t x, int8_t required_exp);

/**
 * @brief Float24 multiplication (x * y) -> z
 */
nrf_dsp_f24_t nrf_dsp_f24_mul(nrf_dsp_f24_t x, nrf_dsp_f24_t y);

/**
 * @brief Float24 addition (x + y) -> z
 */
nrf_dsp_f24_t nrf_dsp_f24_add(nrf_dsp_f24_t x, nrf_dsp_f24_t y);

/**
 * @brief Float24 division (x / y) -> z
 */
nrf_dsp_f24_t nrf_dsp_f24_div(nrf_dsp_f24_t x, nrf_dsp_f24_t y);

/**
 * @brief Float24 square root (sqrt(x)) -> z
 */
nrf_dsp_f24_t nrf_dsp_f24_sqrt(nrf_dsp_f24_t x);

/**
 * @brief Initialize Float24 value from metainformaton
 * 
 * @param[in] mantissa      Mantissa
 * @param[in] exp           Exponent
 * @param[in] sign          Sign byte: nrf_dsp_f24_SIGNED or nrf_dsp_f24_UNSIGNED
 * 
 * @return nrf_dsp_f24_t Float24 value
 */
nrf_dsp_f24_t nrf_dsp_f24_init(uint16_t mantissa, int8_t exp, uint8_t sign);

/**
 * @brief Extended Float24 addition function (x + y) -> z
 * @warning To perform addition, the mantissa and exponents of X and Y must be bit aligned
 * 
 * @param[in] x_man         Mantissa of first argument X
 * @param[in] x_exp         Exponent of first argument X
 * @param[in] x_sign        Sign byte of first argument X
 * @param[in] y_man         Mantissa of second argument Y
 * @param[in] y_exp         Exponent of second argument Y
 * @param[in] y_sign        Sign byte of second argument Y
 * 
 * @return nrf_dsp_f24_t Float24 result of addition
 */
nrf_dsp_f24_t nrf_dsp_f24_addx(uint16_t x_man,
                               int8_t   x_exp,
                               uint8_t  x_sign,
                               uint16_t y_man,
                               int8_t   y_exp,
                               uint8_t  y_sign);

/**
 * @brief Get a pointer to an internal utility array [0-255] to help figure out
 *        how many bits is occupied in byte
 */
const uint8_t* nrf_dsp_f24_get_highbit_helper(void);

/**
 * @brief Get exponent of Float24 value
 */
__NRF_DSP_STATIC_FORCEINLINE
int8_t nrf_dsp_f24_get_exp(nrf_dsp_f24_t x)
{
    return (int8_t)((x.exps & 0x80) ? -(x.exps & 0x3F) : (x.exps & 0x3F));
}

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_SUPPORT_FLOAT24_H_ */

/**
 * @}
 */
