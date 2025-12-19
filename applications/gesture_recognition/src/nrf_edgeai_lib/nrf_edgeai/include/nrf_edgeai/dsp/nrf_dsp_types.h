/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_DSP_TYPES_H_
#define _NRF_DSP_TYPES_H_

#include <nrf_edgeai/dsp/platform/nrf_dsp_platform_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Statistics context flags that indicating the presence of values ​​to use
 * 
 */
typedef union nrf_dsp_stat_ctx_flags_u
{
    struct
    {
        bool sum    : 1;
        bool tss    : 1;
        bool var    : 1;
        bool abssum : 1;
    } is;
    uint8_t all;
} nrf_dsp_stat_ctx_flags_t;

/**
 * @brief Spectral context flags that indicating the presence of values ​​to use
 * 
 */
typedef union nrf_dsp_spectral_ctx_flags_u
{
    struct
    {
        bool magnitude_sum : 1;
        bool centroid      : 1;
    } is;
    uint8_t all;
} nrf_dsp_spectral_ctx_flags_t;

/**
 * @brief Macro definitions for setting flags of @ref nrf_dsp_stat_ctx_flags_t
 * 
 */
#define NRF_DSP_STAT_CTX_EMPTY       (0)
#define NRF_DSP_STAT_CTX_SUM_FLAG    (1 << 0)
#define NRF_DSP_STAT_CTX_TSS_FLAG    (1 << 1)
#define NRF_DSP_STAT_CTX_VAR_FLAG    (1 << 2)
#define NRF_DSP_STAT_CTX_ABSSUM_FLAG (1 << 4)

#define NRF_DSP_STAT_CTX_SUM_TSS_FLAGS (NRF_DSP_STAT_CTX_SUM_FLAG | NRF_DSP_STAT_CTX_TSS_FLAG)
#define NRF_DSP_STAT_CTX_SUM_VAR_FLAGS (NRF_DSP_STAT_CTX_SUM_FLAG | NRF_DSP_STAT_CTX_VAR_FLAG)
#define NRF_DSP_STAT_CTX_SUM_TSS_VAR_FLAGS \
    (NRF_DSP_STAT_CTX_SUM_TSS_FLAGS | NRF_DSP_STAT_CTX_VAR_FLAG)
#define NRF_DSP_STAT_CTX_TSS_VAR_FLAGS (NRF_DSP_STAT_CTX_TSS_FLAG | NRF_DSP_STAT_CTX_VAR_FLAG)

/**
 * @brief Reset utility macro for @ref nrf_dsp_stat_ctx_f32_t context structure
 * 
 */
#define NRF_DSP_STAT_CTX_RESET(ctx) ctx.flags.all = NRF_DSP_STAT_CTX_EMPTY

/**
 * @brief Macro definitions for setting flags of @ref nrf_dsp_spectral_ctx_flags_t
 * 
 */
#define NRF_DSP_SPECTRAL_CTX_EMPTY         (0)
#define NRF_DSP_SPECTRAL_CTX_MAG_SUM_FLAG  (1 << 0)
#define NRF_DSP_SPECTRAL_CTX_CENTROID_FLAG (1 << 1)
/**
 * @brief Reset utility macro for @ref nrf_dsp_spectral_ctx_flags_t context structure
 * 
 */
#define NRF_DSP_SPECTRAL_CTX_RESET(ctx) ctx.flags.all = NRF_DSP_STAT_CTX_EMPTY

/**
 * @brief  Floating-point statistics reusable context, in which is used to store derived or intermediate 
 *          metrics to simplify / speed up the calculation of dependent metrics
 */
typedef struct nrf_dsp_stat_ctx_f32_s
{
    struct
    {
        /* Absolute Sum of vector elements */
        flt32_t abssum;
        /* Sum of vector elements */
        flt32_t sum;
        /* Total sum of squares of vector elements */
        flt32_t tss;
        /* Variance of vector elements */
        flt32_t var;
    } value;

    nrf_dsp_stat_ctx_flags_t flags;
} nrf_dsp_stat_ctx_f32_t;

/**
 * @brief Floating-point spectral reusable context, in which is used to store derived or intermediate
 *        metrics to simplify / speed up the calculation of dependent metrics
 * 
 */
typedef struct nrf_dsp_spectral_ctx_f32_s
{

    struct
    {
        /* Magnitude sum of spectrum */
        flt32_t magnitude_sum;
        /* Spectral centroid  */
        flt32_t centroid;
    } value;

    nrf_dsp_spectral_ctx_flags_t flags;
} nrf_dsp_spectral_ctx_f32_t;

/**
 * @brief  INT8 statistics reusable context, in which is used to store derived or intermediate 
 *          metrics to simplify / speed up the calculation of dependent metrics
 */
typedef struct nrf_dsp_stat_ctx_i8_s
{
    struct
    {
        /* Absolute Sum of vector elements */
        uint32_t abssum;
        /* Sum of vector elements */
        int32_t sum;
        /* Total sum of squares of vector elements */
        uint32_t tss;
        /* Variance of vector elements */
        uint16_t var;
    } value;

    nrf_dsp_stat_ctx_flags_t flags;
} nrf_dsp_stat_ctx_i8_t;

/**
 * @brief  INT16 statistics reusable context, in which is used to store derived or intermediate 
 *          metrics to simplify / speed up the calculation of dependent metrics
 */
typedef struct nrf_dsp_stat_ctx_i16_s
{
    struct
    {
        /* Absolute Sum of vector elements */
        uint32_t abssum;
        /* Sum of vector elements */
        int32_t sum;
        /* Total sum of squares of vector elements */
        uint64_t tss;
        /* Variance of vector elements */
        uint32_t var;
    } value;

    nrf_dsp_stat_ctx_flags_t flags;
} nrf_dsp_stat_ctx_i16_t;

/**
 * @brief INT16-fixed-point spectral reusable context, in which is used to store derived or intermediate
 *        metrics to simplify / speed up the calculation of dependent metrics
 * 
 */
typedef struct nrf_dsp_spectral_ctx_i16_s
{

    struct
    {
        /* Magnitude sum of spectrum */
        uint32_t magnitude_sum;
        /* Spectral centroid  */
        int16_t centroid;
    } value;

    nrf_dsp_spectral_ctx_flags_t flags;
} nrf_dsp_spectral_ctx_i16_t;

/**
 * @brief  INT32 statistics reusable context, in which is used to store derived or intermediate 
 *          metrics to simplify / speed up the calculation of dependent metrics
 */
typedef struct nrf_dsp_stat_ctx_i32_s
{
    struct
    {
        /* Absolute Sum of vector elements */
        uint64_t abssum;
        /* Sum of vector elements */
        int64_t sum;
        /* Total sum of squares of vector elements */
        uint64_t tss;
        /* Variance of vector elements */
        uint64_t var;
    } value;

    nrf_dsp_stat_ctx_flags_t flags;
} nrf_dsp_stat_ctx_i32_t;

/**
 * @brief Sigma(STD) factor for calculation sigma related features, can be positive or negative
 * 
 */
typedef enum nrf_dsp_sigma_factor_e
{
    /** Positive Sigma factors */
    NRF_DSP_SIGMA_FACTOR_P_1 = 1,
    NRF_DSP_SIGMA_FACTOR_P_2 = 2,
    NRF_DSP_SIGMA_FACTOR_P_3 = 3,

    /** Negative Sigma factors */
    NRF_DSP_SIGMA_FACTOR_N_1 = -1,
    NRF_DSP_SIGMA_FACTOR_N_2 = -2,
    NRF_DSP_SIGMA_FACTOR_N_3 = -3,
} nrf_dsp_sigma_factor_t;

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_TYPES_H_ */