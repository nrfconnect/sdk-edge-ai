/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_EDGEAI_DSP_PIPELINE_TYPES_H_
#define _NRF_EDGEAI_DSP_PIPELINE_TYPES_H_

#include <nrf_edgeai/nrf_edgeai_ctypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Possible NN features domain for feature extraction
 */
typedef enum nrf_edgeai_features_domain_e
{
    NRF_EDGEAI_FEATURES_DOMAIN_TIME, /**< Time-domain features */
    NRF_EDGEAI_FEATURES_DOMAIN_FREQ, /**< Frequency-domain features */
} nrf_edgeai_features_domain_t;

/**
 * @brief Time-domain features ID, should be kept in the same order as in nrf_edgeai_features_stat_mask_t
 */
typedef enum nrf_edgeai_feature_timedomain_e
{
    NRF_EDGEAI_FEATURE_MIN = 0,       /**< Minimum */
    NRF_EDGEAI_FEATURE_MAX,           /**< Maximum */
    NRF_EDGEAI_FEATURE_RANGE,         /**< Range */
    NRF_EDGEAI_FEATURE_MEAN,          /**< Mean */
    NRF_EDGEAI_FEATURE_MAD,           /**< Mean Absolute Deviation */
    NRF_EDGEAI_FEATURE_SKEW,          /**< Skewness */
    NRF_EDGEAI_FEATURE_KUR,           /**< Kurtosis */
    NRF_EDGEAI_FEATURE_STD,           /**< Standard Deviation */
    NRF_EDGEAI_FEATURE_RMS,           /**< Root Mean Square */
    NRF_EDGEAI_FEATURE_MCR,           /**< Mean-crossing Rate */
    NRF_EDGEAI_FEATURE_ZCR,           /**< Zero-crossing Rate */
    NRF_EDGEAI_FEATURE_TCR,           /**< Threshold-crossing Rate */
    NRF_EDGEAI_FEATURE_P2P_LF,        /**< Peak-to-Peak Low Frequency */
    NRF_EDGEAI_FEATURE_P2P_HF,        /**< Peak-to-Peak High Frequency */
    NRF_EDGEAI_FEATURE_ABSMEAN,       /**< Absolute mean */
    NRF_EDGEAI_FEATURE_AMDF,          /**< Average magnitude difference */
    NRF_EDGEAI_FEATURE_PSCR,          /**< Positive sigma crossing rate */
    NRF_EDGEAI_FEATURE_NSCR,          /**< Negative sigma crossing rate */
    NRF_EDGEAI_FEATURE_PSOZ,          /**< Percentage of signal over zero */
    NRF_EDGEAI_FEATURE_PSOM,          /**< Percentage of signal over mean */
    NRF_EDGEAI_FEATURE_PSOS,          /**< Percentage of signal over sigma */
    NRF_EDGEAI_FEATURE_CREST,         /**< Crest factor */
    NRF_EDGEAI_FEATURE_RMDS,          /**< Root Mean Difference Square */
    NRF_EDGEAI_FEATURE_AUTOCORR,      /**< Autocorrelation */
    NRF_EDGEAI_FEATURE_HJ_MOBILITY,   /**< Hjorth Mobility */
    NRF_EDGEAI_FEATURE_HJ_COMPLEXITY, /**< Hjorth Complexity */
    NRF_EDGEAI_FEATURE_LR_SLOPE,      /**< Linear regression Slope */
    NRF_EDGEAI_FEATURE_LR_INTERCEPT,  /**< Linear regression Intercept */

    NRF_EDGEAI_FEATURE_cnt, /**< Number of time-domain features */
} nrf_edgeai_feature_timedomain_t;

/** Bit position in time-domain feature extraction mask */
#define NRF_EDGEAI_FEATURE_BIT_MIN           (1U << NRF_EDGEAI_FEATURE_MIN)
#define NRF_EDGEAI_FEATURE_BIT_MAX           (1U << NRF_EDGEAI_FEATURE_MAX)
#define NRF_EDGEAI_FEATURE_BIT_RANGE         (1U << NRF_EDGEAI_FEATURE_RANGE)
#define NRF_EDGEAI_FEATURE_BIT_MEAN          (1U << NRF_EDGEAI_FEATURE_MEAN)
#define NRF_EDGEAI_FEATURE_BIT_RMS           (1U << NRF_EDGEAI_FEATURE_RMS)
#define NRF_EDGEAI_FEATURE_BIT_STD           (1U << NRF_EDGEAI_FEATURE_STD)
#define NRF_EDGEAI_FEATURE_BIT_ZCR           (1U << NRF_EDGEAI_FEATURE_ZCR)
#define NRF_EDGEAI_FEATURE_BIT_MCR           (1U << NRF_EDGEAI_FEATURE_MCR)
#define NRF_EDGEAI_FEATURE_BIT_TCR           (1U << NRF_EDGEAI_FEATURE_TCR)
#define NRF_EDGEAI_FEATURE_BIT_MAD           (1U << NRF_EDGEAI_FEATURE_MAD)
#define NRF_EDGEAI_FEATURE_BIT_SKEW          (1U << NRF_EDGEAI_FEATURE_SKEW)
#define NRF_EDGEAI_FEATURE_BIT_KUR           (1U << NRF_EDGEAI_FEATURE_KUR)
#define NRF_EDGEAI_FEATURE_BIT_P2P_LF        (1U << NRF_EDGEAI_FEATURE_P2P_LF)
#define NRF_EDGEAI_FEATURE_BIT_P2P_HF        (1U << NRF_EDGEAI_FEATURE_P2P_HF)
#define NRF_EDGEAI_FEATURE_BIT_ABSMEAN       (1U << NRF_EDGEAI_FEATURE_ABSMEAN)
#define NRF_EDGEAI_FEATURE_BIT_AMDF          (1U << NRF_EDGEAI_FEATURE_AMDF)
#define NRF_EDGEAI_FEATURE_BIT_POSITIVE_SCR  (1U << NRF_EDGEAI_FEATURE_PSCR)
#define NRF_EDGEAI_FEATURE_BIT_NEGATIVE_SCR  (1U << NRF_EDGEAI_FEATURE_NSCR)
#define NRF_EDGEAI_FEATURE_BIT_PSOZ          (1U << NRF_EDGEAI_FEATURE_PSOZ)
#define NRF_EDGEAI_FEATURE_BIT_PSOM          (1U << NRF_EDGEAI_FEATURE_PSOM)
#define NRF_EDGEAI_FEATURE_BIT_PSOS          (1U << NRF_EDGEAI_FEATURE_PSOS)
#define NRF_EDGEAI_FEATURE_BIT_CREST         (1U << NRF_EDGEAI_FEATURE_CREST)
#define NRF_EDGEAI_FEATURE_BIT_RMDS          (1U << NRF_EDGEAI_FEATURE_RMDS)
#define NRF_EDGEAI_FEATURE_BIT_AUTOCORR      (1U << NRF_EDGEAI_FEATURE_AUTOCORR)
#define NRF_EDGEAI_FEATURE_BIT_HJ_MOBILITY   (1U << NRF_EDGEAI_FEATURE_HJ_MOBILITY)
#define NRF_EDGEAI_FEATURE_BIT_HJ_COMPLEXITY (1U << NRF_EDGEAI_FEATURE_HJ_COMPLEXITY)
#define NRF_EDGEAI_FEATURE_BIT_LR_SLOPE      (1U << NRF_EDGEAI_FEATURE_LR_SLOPE)
#define NRF_EDGEAI_FEATURE_BIT_LR_INTERCEPT  (1U << NRF_EDGEAI_FEATURE_LR_INTERCEPT)

/**
 * @brief Frequency-domain features ID, should be kept in the same order as in nrf_edgeai_features_spectral_mask_t
 */
typedef enum nrf_edgeai_feature_freqdomain_e
{
    NRF_EDGEAI_FEATURE_DOM_FREQS = 0,  /**< Dominant frequencies in spectrum */
    NRF_EDGEAI_FEATURE_DOM_FREQS_AMPL, /**< Amplitudes of dominant frequencies in spectrum */
    NRF_EDGEAI_FEATURE_DOM_FREQS_THD,  /**< Total harmonic distortion of dominant frequencies */
    NRF_EDGEAI_FEATURE_DOM_FREQS_SNR,  /**< Signal-to-noise ratio of dominant frequencies */
    NRF_EDGEAI_FEATURE_DOM_FREQS_MEAN_DIST, /**< Mean distance between dominant frequencies peaks */
    NRF_EDGEAI_FEATURE_FER_LOWMID,          /**< Frequencies energy ratio low/mid */
    NRF_EDGEAI_FEATURE_FER_MIDHIGH,         /**< Frequencies energy ratio mid/high */
    NRF_EDGEAI_FEATURE_FER_LOWHIGH,         /**< Frequencies energy ratio low/high */
    NRF_EDGEAI_FEATURE_SPECTRAL_RMS,        /**< Spectral RMS */
    NRF_EDGEAI_FEATURE_SPECTRAL_CREST,      /**< Spectral Crest */
    NRF_EDGEAI_FEATURE_SPECTRAL_CENTROID,   /**< Spectral Centroid */
    NRF_EDGEAI_FEATURE_SPECTRAL_SPREAD,     /**< Spectral Spread */
    NRF_EDGEAI_FEATURE_SPECTRUM_BINS,       /**< Amplitude spectrum raw bins */
} nrf_edgeai_feature_freqdomain_t;

/** Bit position in frequency-domain feature extraction mask */
#define NRF_EDGEAI_FEATURE_BIT_DOM_FREQS           (1U << NRF_EDGEAI_FEATURE_DOM_FREQS)
#define NRF_EDGEAI_FEATURE_BIT_DOM_FREQS_AMPL      (1U << NRF_EDGEAI_FEATURE_DOM_FREQS_AMPL)
#define NRF_EDGEAI_FEATURE_BIT_DOM_FREQS_THD       (1U << NRF_EDGEAI_FEATURE_DOM_FREQS_THD)
#define NRF_EDGEAI_FEATURE_BIT_DOM_FREQS_SNR       (1U << NRF_EDGEAI_FEATURE_DOM_FREQS_SNR)
#define NRF_EDGEAI_FEATURE_BIT_DOM_FREQS_MEAN_DIST (1U << NRF_EDGEAI_FEATURE_DOM_FREQS_MEAN_DIST)
#define NRF_EDGEAI_FEATURE_BIT_FER_LOWMID          (1U << NRF_EDGEAI_FEATURE_FER_LOWMID)
#define NRF_EDGEAI_FEATURE_BIT_FER_MIDHIGH         (1U << NRF_EDGEAI_FEATURE_FER_MIDHIGH)
#define NRF_EDGEAI_FEATURE_BIT_FER_LOWHIGH         (1U << NRF_EDGEAI_FEATURE_FER_LOWHIGH)
#define NRF_EDGEAI_FEATURE_BIT_SPECTRAL_RMS        (1U << NRF_EDGEAI_FEATURE_SPECTRAL_RMS)
#define NRF_EDGEAI_FEATURE_BIT_SPECTRAL_CREST      (1U << NRF_EDGEAI_FEATURE_SPECTRAL_CREST)
#define NRF_EDGEAI_FEATURE_BIT_SPECTRAL_CENTROID   (1U << NRF_EDGEAI_FEATURE_SPECTRAL_CENTROID)
#define NRF_EDGEAI_FEATURE_BIT_SPECTRAL_SPREAD     (1U << NRF_EDGEAI_FEATURE_SPECTRAL_SPREAD)
#define NRF_EDGEAI_FEATURE_BIT_SPECTRUM_BINS       (1U << NRF_EDGEAI_FEATURE_SPECTRUM_BINS)

/**
 * @brief Time-domain features mask type for feature extracting API
 */
typedef union nrf_edgeai_features_timedomain_mask_u
{
    struct
    {
        bool min          : 1; /**< Minimum */
        bool max          : 1; /**< Maximum */
        bool range        : 1; /**< Range */
        bool mean         : 1; /**< Mean */
        bool mad          : 1; /**< Mean Absolute Deviation */
        bool skew         : 1; /**< Skewness */
        bool kur          : 1; /**< Kurtosis */
        bool std          : 1; /**< Standard Deviation */
        bool rms          : 1; /**< Root Mean Square */
        bool mcr          : 1; /**< Mean-crossing Rate */
        bool zcr          : 1; /**< Zero-crossing Rate */
        bool tcr          : 1; /**< Threshold-crossing Rate */
        bool p2p_lf       : 1; /**< Peak-to-Peak Low Frequency */
        bool p2p_hf       : 1; /**< Peak-to-Peak High Frequency */
        bool absmean      : 1; /**< Absolute mean */
        bool amdf         : 1; /**< Average magnitude difference */
        bool p_scr        : 1; /**< Positive sigma crossing rate */
        bool n_scr        : 1; /**< Negative sigma crossing rate */
        bool psoz         : 1; /**< Percentage of signal over zero */
        bool psom         : 1; /**< Percentage of signal over mean */
        bool psos         : 1; /**< Percentage of signal over sigma */
        bool crest        : 1; /**< Crest factor */
        bool rmds         : 1; /**< Root Mean Difference Square */
        bool autocorr     : 1; /**< Autocorrelation */
        bool hjorth_m     : 1; /**< Hjorth Mobility */
        bool hjorth_c     : 1; /**< Hjorth Complexity */
        bool lr_slope     : 1; /**< Linear regression Slope */
        bool lr_intercept : 1; /**< Linear regression Intercept */
    } is;
    uint32_t all; /**< All time-domain features as bitmask */
} nrf_edgeai_features_timedomain_mask_t;

/**
 * @brief Frequency-domain features mask type for feature extracting API
 */
typedef union nrf_edgeai_features_freqdomain_mask_u
{
    struct
    {
        bool dom_freqs        : 1; /**< Dominant frequencies indicies in spectrum */
        bool dom_freqs_ampl   : 1; /**< Dominant frequencies amplitude values in spectrum */
        bool dom_freqs_thd    : 1; /**< Dominant frequencies total harmonic distortion */
        bool dom_freqs_snr    : 1; /**< Dominant frequencies SNR */
        bool dom_freqs_mean_d : 1; /**< Dominant frequencies mean distance */
        bool fer_lowmid       : 1; /**< Frequencies energy ratio Low / Mid */
        bool fer_midhigh      : 1; /**< Frequencies energy ratio Mid / High */
        bool fer_lowhigh      : 1; /**< Frequencies energy ratio Low / High */
        bool rms              : 1; /**< Spectral RMS */
        bool crest            : 1; /**< Spectral Crest */
        bool centroid         : 1; /**< Spectral Centriod */
        bool spread           : 1; /**< Spectral Spread */
        bool ampl_spectrum    : 1; /**< Amplitude spectrum N first bins */
    } is;
    uint16_t all; /**< All frequency-domain features as bitmask */
} nrf_edgeai_features_freqdomain_mask_t;

/**
 * @brief Mask for all avaliable features for feature extraction
 */
typedef union nrf_edgeai_features_mask_u
{
    struct
    {
        nrf_edgeai_features_freqdomain_mask_t freq; /**< Frequency-domain features mask */
        uint16_t dummy;                             /**< Dummy 2 bytes for proper memory aligment */
        nrf_edgeai_features_timedomain_mask_t time; /**< Time-domain features mask */
    } domain;
    uint64_t all; /**< All features as bitmask */
} nrf_edgeai_features_mask_t;

/**
 * @brief Callback for getting feature argument while feature extraction
 *
 * @param[in] domain               Features domain (time/freq)
 * @param[in] feature_id           Feature id for which argument is needed
 * @param[in] args_num             Numbers of arguments to get
 * @param[in] p_argument_ctx       Argument user context
 * @param[out] p_output_arguments  Pointer to store feature argument
 */
typedef void (*nrf_edgeai_feature_get_arg_cb_t)(nrf_edgeai_features_domain_t domain,
                                                uint8_t                      feature_id,
                                                uint8_t                      args_num,
                                                void*                        p_argument_ctx,
                                                void*                        p_output_arguments);

/**
 * @brief Pipeline function type for 8-bit integer input
 */
typedef size32_t (*nrf_edgeai_features_pipeline_func_i8_t)(
    int8_t*                          p_input,
    size32_t                         num,
    int16_t*                         p_features,
    const nrf_edgeai_features_mask_t feature_mask,
    void*                            p_ctx,
    nrf_edgeai_feature_get_arg_cb_t  get_argument,
    void*                            p_argument_ctx);

/**
 * @brief Pipeline function type for 16-bit integer input
 */
typedef size32_t (*nrf_edgeai_features_pipeline_func_i16_t)(
    int16_t*                         p_input,
    size32_t                         num,
    int32_t*                         p_features,
    const nrf_edgeai_features_mask_t feature_mask,
    void*                            p_ctx,
    nrf_edgeai_feature_get_arg_cb_t  get_argument,
    void*                            p_argument_ctx);

/**
 * @brief Pipeline function type for 32-bit float input
 */
typedef size32_t (*nrf_edgeai_features_pipeline_func_f32_t)(
    flt32_t*                         p_input,
    size32_t                         num,
    flt32_t*                         p_features,
    const nrf_edgeai_features_mask_t feature_mask,
    void*                            p_ctx,
    nrf_edgeai_feature_get_arg_cb_t  get_argument,
    void*                            p_argument_ctx);

/**
 * @brief Union of pipeline function pointers for all supported input types
 */
typedef union nrf_edgeai_features_pipeline_u
{
    const void* p_void; /**< Generic pointer */
    const nrf_edgeai_features_pipeline_func_i8_t*
        p_array_i8; /**< Array of 8-bit pipeline functions */
    const nrf_edgeai_features_pipeline_func_i16_t*
        p_array_i16; /**< Array of 16-bit pipeline functions */
    const nrf_edgeai_features_pipeline_func_f32_t*
        p_array_f32; /**< Array of 32-bit pipeline functions */
} nrf_edgeai_features_pipeline_t;

/**
 * @brief Features pipeline context structure
 */
typedef struct nrf_edgeai_features_pipeline_ctx_s
{
    const nrf_edgeai_features_pipeline_t functions;     /**< Pipeline functions union */
    const size32_t                       functions_num; /**< Number of pipeline functions */
    void*                                p_ctx;         /**< User context for pipeline functions */
} nrf_edgeai_features_pipeline_ctx_t;

/**
 * @brief Features meta information for 16-bit integer features
 */
typedef struct nrf_edgeai_features_meta_i16_s
{
    const int16_t* p_min;       /**< Pointer to minimum values */
    const int16_t* p_max;       /**< Pointer to maximum values */
    const int8_t*  p_arguments; /**< Pointer to feature arguments */
} nrf_edgeai_features_meta_i16_t;

/**
 * @brief Features meta information for 32-bit integer features
 */
typedef struct nrf_edgeai_features_meta_i32_s
{
    const int32_t* p_min;       /**< Pointer to minimum values */
    const int32_t* p_max;       /**< Pointer to maximum values */
    const int16_t* p_arguments; /**< Pointer to feature arguments */
} nrf_edgeai_features_meta_i32_t;

/**
 * @brief Features meta information for 32-bit float features
 */
typedef struct nrf_edgeai_features_meta_f32_s
{
    const flt32_t* p_min;       /**< Pointer to minimum values */
    const flt32_t* p_max;       /**< Pointer to maximum values */
    const flt32_t* p_arguments; /**< Pointer to feature arguments */
} nrf_edgeai_features_meta_f32_t;

/**
 * @brief Union of features meta information for all supported types
 */
typedef union nrf_edgeai_features_meta_u
{
    nrf_edgeai_features_meta_i16_t i16; /**< Meta for 16-bit integer features */
    nrf_edgeai_features_meta_i32_t i32; /**< Meta for 32-bit integer features */
    nrf_edgeai_features_meta_f32_t f32; /**< Meta for 32-bit float features */
} nrf_edgeai_features_meta_t;

/**
 * @brief FFT context frequency-domain features extraction for 32-bit float input
 */
typedef struct nrf_edgeai_features_freq_fft_ctx_f32_s
{
    flt32_t*        p_rfft_buffer;         /**< Pointer to FFT working buffer */
    const flt32_t*  p_rfft_twiddle_table;  /**< Pointer to RFFT twiddle factors table */
    const flt32_t*  p_cfft_twiddle_table;  /**< Pointer to CFFT twiddle factors table */
    const uint16_t* p_cfft_bitrev_table;   /**< Pointer to bit-reversal table */
    const uint16_t  cfft_bitrev_table_len; /**< Length of bit-reversal table */
    const uint16_t  rfft_len;              /**< Length of RFFT */
} nrf_edgeai_features_freq_fft_ctx_f32_t;

/**
 * @brief FFT context frequency-domain features extraction for 16-bit integer input
 */
typedef struct nrf_edgeai_features_freq_fft_ctx_i16_s
{
    int16_t*        p_rfft_buffer;         /**< Pointer to FFT working buffer */
    const int16_t*  p_rfft_twiddle_table;  /**< Pointer to RFFT twiddle factors table */
    const int16_t*  p_cfft_twiddle_table;  /**< Pointer to CFFT twiddle factors table */
    const uint16_t* p_cfft_bitrev_table;   /**< Pointer to bit-reversal table */
    const uint16_t  cfft_bitrev_table_len; /**< Length of bit-reversal table */
    const uint16_t  rfft_len;              /**< Length of RFFT */
} nrf_edgeai_features_freq_fft_ctx_i16_t;

/**
 * @brief Union of FFT contexts for all supported input types
 */
typedef union nrf_edgeai_features_freq_fft_ctx_u
{
    nrf_edgeai_features_freq_fft_ctx_f32_t f32; /**< FFT context for 32-bit float input */
    nrf_edgeai_features_freq_fft_ctx_i16_t i16; /**< FFT context for 16-bit integer input */
} nrf_edgeai_features_freq_fft_ctx_t;

/**
 * @brief DSP feature extraction context structure
 */
typedef struct nrf_edgeai_dsp_feature_extraction_s
{
    union
    {
        void*    p_void; /**< Generic pointer to extracted features memory */
        int16_t* p_i16;  /**< Pointer to 16-bit extracted features memory */
        int32_t* p_i32;  /**< Pointer to 32-bit extracted features memory */
        flt32_t* p_f32;  /**< Pointer to 32-bit float extracted features memory */
    } extracted_memory;

    nrf_edgeai_features_meta_t meta; /**< Features meta information */

    const uint16_t overall_num; /**< Overall number of extracted features */

    const uint16_t masks_num; /**< Number of features masks in array */

    const nrf_edgeai_features_mask_t* p_masks; /**< Pointer to array of features mask */

    /**< Pointer to time-domain pipeline context */
    const nrf_edgeai_features_pipeline_ctx_t* p_timedomain_pipeline;

    /**< Pointer to freq-domain pipeline context */
    const nrf_edgeai_features_pipeline_ctx_t* p_freqdomain_pipeline;
} nrf_edgeai_dsp_feature_extraction_t;

/**
 * @brief DSP pipeline context structure
 */
typedef struct nrf_edgeai_dsp_pipeline_s
{
    nrf_edgeai_dsp_feature_extraction_t features; /**< DSP feature extraction context */
} nrf_edgeai_dsp_pipeline_t;

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_DSP_PIPELINE_TYPES_H_ */