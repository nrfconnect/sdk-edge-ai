/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nrf_edgeai_user_model.h"
#include "nrf_edgeai_user_types.h"

#include <nrf_edgeai/rt/private/nrf_edgeai_interfaces.h>
#include <nrf_edgeai/nrf_edgeai_platform.h>

//////////////////////////////////////////////////////////////////////////////

#define EDGEAI_LAB_SOLUTION_ID_STR      "90360"
#define EDGEAI_RUNTIME_VERSION_COMBINED 0x00000001

//////////////////////////////////////////////////////////////////////////////
#define INPUT_TYPE f32

/** User input features type */
#define INPUT_FEATURE_DATA_TYPE NRF_EDGEAI_INPUT_F32

/** Number of unique features in the original input sample */
#define INPUT_UNIQ_FEATURES_NUM 2

/** Number of unique features actually used by NN from the original input sample */
#define INPUT_UNIQ_FEATURES_USED_NUM 2

/** Number of input feature samples that should be collected in the input window
 *  feature_sample = 1 * INPUT_UNIQ_FEATURES_NUM
 */
#define INPUT_WINDOW_SIZE 128

/** Number of input feature samples on that the input window is shifted */
#define INPUT_WINDOW_SHIFT 128

/** Number of subwindows in input feature window,
* the SUBWINDOW_SIZE = INPUT_WINDOW_SIZE / INPUT_SUBWINDOW_NUM
* if the window size is not divisible by the number of subwindows without a remainder,
* the remainder is added to the last subwindow size */
#define INPUT_SUBWINDOW_NUM 0

#define INPUT_UNIQUE_SCALES_NUM \
    (sizeof(INPUT_FEATURES_SCALE_MIN) / sizeof(INPUT_FEATURES_SCALE_MIN[0]))

//////////////////////////////////////////////////////////////////////////////
#define MODEL_NEURONS_NUM 20
#define MODEL_WEIGHTS_NUM 211
#define MODEL_OUTPUTS_NUM 10
#define MODEL_TASK        3
#define MODEL_PARAMS_TYPE q16
#define MODEL_REORDERING  0

#define MODEL_USES_AS_INPUT_INPUT_FEATURES 0
#define MODEL_USES_AS_INPUT_DSP_FEATURES   1
#define MODEL_USES_AS_INPUT_MASK \
    ((MODEL_USES_AS_INPUT_INPUT_FEATURES << 0) | (MODEL_USES_AS_INPUT_DSP_FEATURES << 1))

//////////////////////////////////////////////////////////////////////////////
/** Defines input(also used for LAG) features MIN scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MIN[] = { 2.4513564, 2.3787556 };

/** Defines input(also used for LAG) features MAX scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MAX[] = { 2.5927811, 2.4932418 };

/** Defines which unique features from the input data will be used/collected,
 *  one bit for one unique feature, starting from LSB
 */
#define INPUT_FEATURES_USAGE_MASK NULL

/** Defines which unique input features is used for LAG features processing,
 *  one bit for one unique feature, starting from LSB
 */
#define INPUT_FEATURES_USED_FOR_LAGS_MASK NULL

//////////////////////////////////////////////////////////////////////////////
/** Input feature buffer element size, 
 * if quantization of model is bigger than input features size in bits, 
 * the size of input buffer should aligned to nrf_user_neuron_t */
#define INPUT_TYPE_SIZE                                                                  \
    ((sizeof(nrf_user_input_t) > sizeof(nrf_user_neuron_t)) ? sizeof(nrf_user_input_t) : \
                                                              sizeof(nrf_user_neuron_t))

/** Input features window size in bytes to allocate statically */
#define INPUT_WINDOW_BUFFER_SIZE_BYTES \
    (INPUT_WINDOW_SIZE * INPUT_UNIQ_FEATURES_NUM * INPUT_TYPE_SIZE)

static uint8_t input_window_[INPUT_WINDOW_BUFFER_SIZE_BYTES] __NRF_EDGEAI_ALIGNED;

#define INPUT_WINDOW_MEMORY &input_window_[0]

static nrf_edgeai_window_ctx_t input_window_ctx_;
#define P_INPUT_WINDOW_CTX &input_window_ctx_

//////////////////////////////////////////////////////////////////////////////
/** The maximum number of extracted features that user used for all unique input features */
#define EXTRACTED_FEATURES_NUM 100

#define EXTRACTED_FEATURES_META_TYPE f32

/** DSP feature buffer element size,
 * if quantization of model is bigger than DSP features size in bits,
 * the size of extracted DSP features buffer should aligned to nrf_user_neuron_t */
#define EXTRACTED_FEATURE_SIZE_BYTES                                                         \
    ((sizeof(nrf_user_feature_t) > sizeof(nrf_user_neuron_t)) ? sizeof(nrf_user_feature_t) : \
                                                                sizeof(nrf_user_neuron_t))

/** Size of extracted features buffer in bytes */
#define EXTRACTED_FEATURES_BUFFER_SIZE_BYTES (EXTRACTED_FEATURES_NUM * EXTRACTED_FEATURE_SIZE_BYTES)

/** Defines feature extraction masks used as nrf_edgeai_features_mask_t,
 *  64 bit for one unique input feature, @ref nrf_edgeai_features_mask_t to see bitmask
 */

/** Defines feature extraction masks used as nrf_edgeai_features_mask_t,
 *  64 bit for one unique input feature, @ref nrf_edgeai_features_mask_t to see bitmask
 */

static const uint64_t FEATURES_EXTRACTION_MASK[] = { 0x3ffffff00000fff, 0x3ffffff00000fff };

/** Defines arguments used while feature extraction
 */

/** Defines arguments used while feature extraction
 */
static const nrf_user_input_t FEATURES_EXTRACTION_ARGUMENTS[] = { 0, 4, 4, 2, 2, 2, 2,
                                                                  0, 4, 4, 2, 2, 2, 2 };

/** Defines extracted features MIN scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MIN[] = {
    2.4513564,  2.5229721,  0.0037780,  2.5171421, 0.0006703, -1.1318706, -0.8314173, 0.0008326,
    2.5171454,  0.3149606,  0.0000000,  0.0000000, 0.0018883, 0.0030801,  2.5171421,  0.0008226,
    0.0000000,  0.0000000,  1.0000000,  0.4062500, 0.0000000, 1.0006109,  0.0114511,  -0.3330964,
    1.0897368,  1.0457573,  1.0000000,  1.0000000, 1.0000000, 1.0000000,  1.0000000,  0.0218001,
    0.0189585,  0.0172489,  0.0145591,  0.0143178, 0.0000000, 0.0000000,  0.0000000,  0.0000000,
    0.0000000,  0.2996616,  2.0000000,  0.2043949, 0.0466924, 0.0891880,  0.0094301,  1.7584587,
    26.3579922, 16.1412811, 2.3787556,  2.4321387, 0.0039420, 2.4259148,  0.0006063,  -1.2133886,
    -0.7785149, 0.0007721,  2.4259200,  0.2204724, 0.0000000, 0.0000000,  0.0017657,  0.0027924,
    2.4259148,  0.0007553,  0.0000000,  0.0000000, 1.0000000, 0.4296875,  0.0000000,  1.0007271,
    0.0107538,  -0.3089918, 0.8414398,  1.0700141, 1.0000000, 1.0000000,  1.0000000,  1.0000000,
    1.0000000,  0.0212893,  0.0178190,  0.0146569, 0.0128636, 0.0120153,  0.0000000,  0.0000000,
    0.0000000,  0.0000000,  0.0000000,  0.2498299, 3.0000000, 0.1599509,  0.0531547,  0.1782835,
    0.0087450,  1.7724997,  23.6686268, 15.9895439
};

/** Defines extracted features MAX scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MAX[] = {
    2.5200155,  2.5927811,  0.1246705,  2.5234764,  0.0161919,  0.9820696,  3.9926000,  0.0204250,
    2.5234842,  0.7795275,  0.0000000,  0.0000000,  0.0410223,  0.1225357,  2.5234764,  0.0285017,
    0.1102362,  0.1102362,  1.0000000,  0.5781250,  0.0546875,  1.0279365,  0.4104524,  0.5154383,
    1.8258654,  1.4977585,  61.0000000, 62.0000000, 62.0000000, 62.0000000, 62.0000000, 0.9358254,
    0.6779823,  0.5883865,  0.5266715,  0.4964337,  1.4449730,  2.3301661,  3.7930861,  5.4245057,
    4.0417528,  1.9504315,  48.0000000, 6.3891587,  4.7852931,  4.8026490,  0.2319150,  5.3385010,
    40.6795044, 20.8609238, 2.4290178,  2.4932418,  0.0977321,  2.4334788,  0.0125451,  0.8930452,
    4.8162580,  0.0155131,  2.4334841,  0.6850393,  0.0000000,  0.0000000,  0.0344934,  0.1046307,
    2.4334788,  0.0192994,  0.1102362,  0.1102362,  1.0000000,  0.6093750,  0.0546875,  1.0250756,
    0.2861258,  0.4948431,  1.7271173,  1.7972425,  61.0000000, 62.0000000, 62.0000000, 62.0000000,
    62.0000000, 0.5665466,  0.4413168,  0.3787947,  0.3317127,  0.2942180,  1.6734345,  2.9116232,
    3.2026401,  3.9170854,  5.8657188,  2.0856731,  46.0000000, 7.6323743,  5.4191923,  14.8948355,
    0.1755164,  5.6097703,  37.7644272, 20.3906918
};

/** Memory allocation to store extracted features during DSP pipeline */
static uint8_t
    extracted_features_buffer_[EXTRACTED_FEATURES_BUFFER_SIZE_BYTES] __NRF_EDGEAI_ALIGNED;

/** Timedomain features processing context  */
#define P_TIMEDOMAIN_FEATURES_CTX NULL
/** Timedomain features in feature extraction pipeline  */
static const nrf_edgeai_features_pipeline_func_f32_t timedomain_features_[] = {
    nrf_edgeai_feature_utility_tss_sum_f32,
    nrf_edgeai_feature_min_max_range_f32,
    nrf_edgeai_feature_mean_f32,
    nrf_edgeai_feature_mad_f32,
    nrf_edgeai_feature_skew_kur_f32,
    nrf_edgeai_feature_std_f32,
    nrf_edgeai_feature_rms_f32,
    nrf_edgeai_feature_mcr_f32,
    nrf_edgeai_feature_zcr_f32,
    nrf_edgeai_feature_tcr_f32,
    nrf_edgeai_feature_p2p_lf_hf_f32,
    nrf_edgeai_feature_absmean_f32,
    nrf_edgeai_feature_amdf_f32,
    nrf_edgeai_feature_pscr_f32,
    nrf_edgeai_feature_nscr_f32,
    nrf_edgeai_feature_psoz_f32,
    nrf_edgeai_feature_psom_f32,
    nrf_edgeai_feature_psos_f32,
    nrf_edgeai_feature_crest_f32,
    nrf_edgeai_feature_rmds_f32,
    nrf_edgeai_feature_autocorr_f32,
    nrf_edgeai_feature_hjorth_f32
};

static const nrf_edgeai_features_pipeline_ctx_t timedomain_pipeline_ = {
    .functions_num    = sizeof(timedomain_features_) / sizeof(timedomain_features_[0]),
    .functions.p_void = timedomain_features_,
    .p_ctx            = P_TIMEDOMAIN_FEATURES_CTX,
};
#define P_TIMEDOMAIN_PIPELINE &timedomain_pipeline_

/** DSP Amplitude spectrum and complex RFFT length */
#define DSP_AMPLITUDE_SPECTRUM_LEN 64
#define DSP_RFFT_LEN               (DSP_AMPLITUDE_SPECTRUM_LEN * 2)

/** Defines DSP Complex FFT reverse bit index table length
 */
#define DSP_CFFT_BITREV_INDEX_TABLE_LEN 56

/** Defines DSP Complex FFT reverse bit index table
 */
static const uint16_t DSP_CFFT_BITREV_INDEX_TABLE[] = {
    8,   64,  16,  128, 24,  192, 32,  256, 40,  320, 48,  384, 56,  448, 80,  136, 88,  200, 96,
    264, 104, 328, 112, 392, 120, 456, 152, 208, 160, 272, 168, 336, 176, 400, 184, 464, 224, 280,
    232, 344, 240, 408, 248, 472, 296, 352, 304, 416, 312, 480, 368, 424, 376, 488, 440, 496
};

/** Defines DSP Real FFT twiddle factors table
 */
static const nrf_user_input_t DSP_RFFT_TWIDDLE_FACTORS[] = {
    0.0000000, 1.0000000,  0.0490677, 0.9987954,  0.0980171, 0.9951847,  0.1467305, 0.9891765,
    0.1950903, 0.9807853,  0.2429802, 0.9700313,  0.2902847, 0.9569404,  0.3368899, 0.9415441,
    0.3826834, 0.9238795,  0.4275551, 0.9039893,  0.4713967, 0.8819213,  0.5141028, 0.8577286,
    0.5555702, 0.8314696,  0.5956993, 0.8032075,  0.6343933, 0.7730104,  0.6715590, 0.7409511,
    0.7071068, 0.7071068,  0.7409511, 0.6715590,  0.7730104, 0.6343933,  0.8032075, 0.5956993,
    0.8314696, 0.5555702,  0.8577286, 0.5141028,  0.8819213, 0.4713967,  0.9039893, 0.4275551,
    0.9238795, 0.3826834,  0.9415441, 0.3368899,  0.9569404, 0.2902847,  0.9700313, 0.2429802,
    0.9807853, 0.1950903,  0.9891765, 0.1467305,  0.9951847, 0.0980171,  0.9987954, 0.0490677,
    1.0000000, 0.0000000,  0.9987954, -0.0490677, 0.9951847, -0.0980171, 0.9891765, -0.1467305,
    0.9807853, -0.1950903, 0.9700313, -0.2429802, 0.9569404, -0.2902847, 0.9415441, -0.3368899,
    0.9238795, -0.3826834, 0.9039893, -0.4275551, 0.8819213, -0.4713967, 0.8577286, -0.5141028,
    0.8314696, -0.5555702, 0.8032075, -0.5956993, 0.7730104, -0.6343933, 0.7409511, -0.6715590,
    0.7071068, -0.7071068, 0.6715590, -0.7409511, 0.6343933, -0.7730104, 0.5956993, -0.8032075,
    0.5555702, -0.8314696, 0.5141028, -0.8577286, 0.4713967, -0.8819213, 0.4275551, -0.9039893,
    0.3826834, -0.9238795, 0.3368899, -0.9415441, 0.2902847, -0.9569404, 0.2429802, -0.9700313,
    0.1950903, -0.9807853, 0.1467305, -0.9891765, 0.0980171, -0.9951847, 0.0490677, -0.9987954
};

/** Defines DSP Complex FFT twiddle factors table
 */
static const nrf_user_input_t DSP_CFFT_TWIDDLE_FACTORS[] = {
    1.0000000,  0.0000000,  0.9951847,  0.0980171,  0.9807853,  0.1950903,  0.9569404,  0.2902847,
    0.9238795,  0.3826834,  0.8819213,  0.4713967,  0.8314696,  0.5555702,  0.7730104,  0.6343933,
    0.7071068,  0.7071068,  0.6343933,  0.7730104,  0.5555702,  0.8314696,  0.4713967,  0.8819213,
    0.3826834,  0.9238795,  0.2902847,  0.9569404,  0.1950903,  0.9807853,  0.0980171,  0.9951847,
    0.0000000,  1.0000000,  -0.0980171, 0.9951847,  -0.1950903, 0.9807853,  -0.2902847, 0.9569404,
    -0.3826834, 0.9238795,  -0.4713967, 0.8819213,  -0.5555702, 0.8314696,  -0.6343933, 0.7730104,
    -0.7071068, 0.7071068,  -0.7730104, 0.6343933,  -0.8314696, 0.5555702,  -0.8819213, 0.4713967,
    -0.9238795, 0.3826834,  -0.9569404, 0.2902847,  -0.9807853, 0.1950903,  -0.9951847, 0.0980171,
    -1.0000000, 0.0000000,  -0.9951847, -0.0980171, -0.9807853, -0.1950903, -0.9569404, -0.2902847,
    -0.9238795, -0.3826834, -0.8819213, -0.4713967, -0.8314696, -0.5555702, -0.7730104, -0.6343933,
    -0.7071068, -0.7071068, -0.6343933, -0.7730104, -0.5555702, -0.8314696, -0.4713967, -0.8819213,
    -0.3826834, -0.9238795, -0.2902847, -0.9569404, -0.1950903, -0.9807853, -0.0980171, -0.9951847,
    -0.0000000, -1.0000000, 0.0980171,  -0.9951847, 0.1950903,  -0.9807853, 0.2902847,  -0.9569404,
    0.3826834,  -0.9238795, 0.4713967,  -0.8819213, 0.5555702,  -0.8314696, 0.6343933,  -0.7730104,
    0.7071068,  -0.7071068, 0.7730104,  -0.6343933, 0.8314696,  -0.5555702, 0.8819213,  -0.4713967,
    0.9238795,  -0.3826834, 0.9569404,  -0.2902847, 0.9807853,  -0.1950903, 0.9951847,  -0.0980171
};
/** DSP FFT context */
static nrf_edgeai_features_freq_fft_ctx_t dsp_fft_ctx_ = {
    .INPUT_TYPE = {
        .p_rfft_buffer          = NULL,
        .p_rfft_twiddle_table   = DSP_RFFT_TWIDDLE_FACTORS,
        .p_cfft_twiddle_table   = DSP_CFFT_TWIDDLE_FACTORS,
        .p_cfft_bitrev_table    = DSP_CFFT_BITREV_INDEX_TABLE,
        .cfft_bitrev_table_len  = DSP_CFFT_BITREV_INDEX_TABLE_LEN,
        .rfft_len               = DSP_RFFT_LEN,
    }
};

/** Frequency domain features processing context  */
#define P_FREQDOMAIN_FEATURES_CTX &dsp_fft_ctx_
/** Frequency domain features in feature extraction pipeline  */
static const nrf_edgeai_features_pipeline_func_f32_t freqdomain_features_[] = {
    nrf_edgeai_feature_utility_rfft_128_f32,    nrf_edgeai_feature_dom_freqs_features_f32,
    nrf_edgeai_feature_freqs_energy_ratios_f32, nrf_edgeai_feature_spectral_rms_f32,
    nrf_edgeai_feature_spectral_crest_f32,      nrf_edgeai_feature_spectral_centroid_f32,
    nrf_edgeai_feature_spectral_spread_f32
};

static const nrf_edgeai_features_pipeline_ctx_t freqdomain_pipeline_ = {
    .functions_num    = sizeof(freqdomain_features_) / sizeof(freqdomain_features_[0]),
    .functions.p_void = freqdomain_features_,
    .p_ctx            = P_FREQDOMAIN_FEATURES_CTX,
};
#define P_FREQDOMAIN_PIPELINE &freqdomain_pipeline_

static nrf_edgeai_dsp_pipeline_t dsp_pipeline_ = { 
   .features = {  
       .p_masks = (nrf_edgeai_features_mask_t*)FEATURES_EXTRACTION_MASK, 
       .extracted_memory.p_void = extracted_features_buffer_, 
       .overall_num = EXTRACTED_FEATURES_NUM, 
       .masks_num = sizeof(FEATURES_EXTRACTION_MASK) / sizeof(FEATURES_EXTRACTION_MASK[0]), 

       .p_timedomain_pipeline = P_TIMEDOMAIN_PIPELINE, 
       .p_freqdomain_pipeline = P_FREQDOMAIN_PIPELINE, 

       .meta.EXTRACTED_FEATURES_META_TYPE = { 
           .p_min = EXTRACTED_FEATURES_SCALE_MIN, 
           .p_max = EXTRACTED_FEATURES_SCALE_MAX, 
       .p_arguments = FEATURES_EXTRACTION_ARGUMENTS, 
       },
   }, 
};

#define P_DSP_PIPELINE &dsp_pipeline_

//////////////////////////////////////////////////////////////////////////////

static const nrf_user_weight_t MODEL_WEIGHTS[] = {
    19384,  13458, 12946,  -10935, 4073,   7624,   -10468, -20496, 13551,  19344,  -8687,  9265,
    7429,   3960,  15993,  17053,  14222,  -16857, 32767,  -28014, -2986,  -3030,  -1444,  -7643,
    -3380,  -256,  1728,   -3311,  -22282, -2124,  -3860,  12260,  1572,   28106,  -28903, 11614,
    11340,  -9203, 5417,   -20119, 6079,   -12556, 8589,   -6903,  1069,   -12470, 25709,  9285,
    9768,   23964, 7498,   -5265,  -5885,  2051,   6985,   9787,   -12018, 14645,  -7890,  -4278,
    -6904,  19820, 6381,   14491,  13338,  8435,   -11555, -3778,  4826,   -11017, -13905, 9813,
    6502,   5133,  2862,   -13771, -9468,  1625,   8614,   -2169,  -12716, -12537, 8201,   8818,
    13425,  -878,  7506,   32765,  -15485, 1707,   6285,   -1467,  1603,   4554,   -4442,  -1931,
    -6138,  22678, -4980,  16820,  -24653, 17654,  8155,   -25720, -9174,  -6493,  7610,   -13576,
    13136,  21558, -9833,  6783,   -10684, 18021,  12646,  -4111,  10324,  5176,   15092,  32765,
    -21173, -3161, 3795,   1730,   -4489,  -4976,  -2926,  -18691, 736,    3875,   -3396,  25119,
    20249,  -9503, -2216,  3645,   -977,   -8129,  -5223,  -4104,  -6913,  -2380,  -4577,  5847,
    5207,   -7014, 8709,   -7339,  -3128,  -20025, -958,   -488,   5849,   -7489,  -5215,  688,
    2244,   28051, 10047,  -3170,  -1918,  -11582, -7330,  6348,   6799,   -17800, 30531,  11622,
    6562,   2507,  7716,   -957,   -4445,  -700,   10546,  -13747, -7045,  7964,   -10586, -2517,
    10660,  1369,  -3757,  -2130,  -7113,  1746,   8059,   3799,   -4956,  13494,  -6228,  -1075,
    13431,  -5192, -20154, 2371,   4454,   356,    8892,   5984,   2945,   2782,   -2642,  6941,
    5245,   8902,  5666,   3428,   -5365,  23056,  -11928
};

static const uint16_t MODEL_NEURONS_LINKS[] = {
    5,   16,  26, 27, 28, 29, 41,  49, 56,  70, 77, 78,  79, 80,  87, 91,  98,  100, 0,   100,
    3,   16,  20, 29, 36, 49, 74,  80, 85,  88, 93, 94,  99, 100, 2,  100, 5,   31,  36,  38,
    39,  41,  42, 43, 44, 48, 51,  58, 69,  70, 76, 77,  80, 83,  85, 99,  100, 4,   100, 0,
    0,   3,   6,  9,  16, 23, 24,  26, 27,  30, 32, 36,  41, 45,  53, 55,  76,  77,  80,  85,
    88,  90,  92, 95, 97, 98, 100, 6,  100, 26, 28, 29,  30, 37,  47, 53,  56,  63,  94,  100,
    8,   100, 0,  8,  20, 25, 29,  36, 40,  41, 44, 70,  77, 78,  80, 92,  93,  97,  100, 10,
    100, 4,   16, 26, 28, 46, 56,  57, 79,  81, 86, 100, 12, 100, 12, 0,   1,   5,   16,  21,
    25,  28,  29, 30, 47, 49, 50,  56, 58,  65, 69, 74,  77, 86,  87, 93,  97,  100, 14,  100,
    0,   2,   3,  6,  9,  16, 24,  25, 26,  28, 29, 30,  36, 39,  40, 42,  44,  51,  63,  71,
    73,  76,  78, 79, 80, 82, 84,  85, 87,  89, 94, 100, 16, 100, 2,  6,   23,  26,  27,  28,
    36,  40,  44, 53, 69, 73, 76,  78, 100, 18, 100
};

static const uint16_t MODEL_NEURON_INTERNAL_LINKS_NUM[] = { 0,   19,  20,  35,  36,  58,  60,
                                                            88,  89,  101, 104, 120, 122, 133,
                                                            135, 159, 161, 193, 196, 210 };

static const uint16_t MODEL_NEURON_EXTERNAL_LINKS_NUM[] = { 18,  20,  34,  36,  57,  59,  87,
                                                            89,  100, 102, 119, 121, 132, 134,
                                                            158, 160, 192, 194, 209, 211 };

static const nrf_user_coeff_t MODEL_NEURON_ACTIVATION_WEIGHTS[] = {
    0, 2938, 0, 6752, 0, 7942, 0, 3507, 0, 10238, 0, 2882, 0, 12015, 0, 15057, 0, 8338, 0, 6266
};

static const uint8_t MODEL_NEURON_ACTIVATION_TYPE_MASK[] = { 0x55, 0x55, 0x5 };

static const uint16_t MODEL_OUTPUT_NEURONS_INDICES[] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19 };

static const nrf_user_output_t MODEL_OUTPUT_SCALE_MIN[] = { 0.8250794, 0.8159970, 0.8708660,
                                                            0.4854683, 0.5862076, 0.4753624,
                                                            0.4090292, 0.4397500, 0.7855731,
                                                            0.7961165 };

static const nrf_user_output_t MODEL_OUTPUT_SCALE_MAX[] = { 0.8351415, 0.8306280, 0.8806544,
                                                            0.4993975, 0.6120152, 0.4950075,
                                                            0.4409296, 0.4589339, 0.7969485,
                                                            0.8136284 };

static const nrf_user_output_t MODEL_AVERAGE_EMBEDDING[] = { 0.8292735, 0.8226876, 0.8751104,
                                                             0.4926738, 0.5999025, 0.4841458,
                                                             0.4264121, 0.4509522, 0.7913507,
                                                             0.8032691 };

#define NN_DECODED_OUTPUT_INIT                                       \
    .anomaly = {                                                     \
        .score = 0.f,                                                \
        .meta  = { .p_scale_min         = MODEL_OUTPUT_SCALE_MIN,    \
                  .p_scale_max         = MODEL_OUTPUT_SCALE_MAX,    \
                  .p_average_embedding = MODEL_AVERAGE_EMBEDDING }, \
    }

//////////////////////////////////////////////////////////////////////////////
#define NN_INPUT_SETUP_INTERFACE       nrf_edgeai_input_setup_discrete_window
#define NN_INPUT_FEED_INTERFACE        nrf_edgeai_input_feed_discrete_window_f32
#define NN_PROCESS_FEATURES_INTERFACE  nrf_edgeai_process_features_dsp_f32_q16
#define NN_RUN_INFERENCE_INTERFACE     nrf_edgeai_run_model_inference_q16
#define NN_PROPAGATE_OUTPUTS_INTERFACE nrf_edgeai_output_dequantize_q16_f32
#define NN_DECODE_OUTPUTS_INTERFACE    nrf_edgeai_output_decode_anomaly_f32

//////////////////////////////////////////////////////////////////////////////

static nrf_user_neuron_t model_neurons_[MODEL_NEURONS_NUM];
static nrf_user_output_t model_outputs_[MODEL_OUTPUTS_NUM];

//////////////////////////////////////////////////////////////////////////////

static nrf_edgeai_t nrf_edgeai_ = {
    ///
    .metadata.p_solution_id     = EDGEAI_LAB_SOLUTION_ID_STR,
    .metadata.version.combined  = EDGEAI_RUNTIME_VERSION_COMBINED,
    ///   
    .input.p_used_for_lags_mask = INPUT_FEATURES_USED_FOR_LAGS_MASK,
    .input.p_usage_mask         = INPUT_FEATURES_USAGE_MASK,
    .input.type                 = INPUT_FEATURE_DATA_TYPE,
    .input.unique_num           = INPUT_UNIQ_FEATURES_NUM,
    .input.unique_num_used      = INPUT_UNIQ_FEATURES_USED_NUM,
    .input.unique_scales_num    = INPUT_UNIQUE_SCALES_NUM,
    .input.window_size          = INPUT_WINDOW_SIZE,
    .input.window_shift         = INPUT_WINDOW_SHIFT,
    .input.subwindow_num        = INPUT_SUBWINDOW_NUM,
    .input.window_memory.p_void = INPUT_WINDOW_MEMORY,
    .input.p_window_ctx         = P_INPUT_WINDOW_CTX,

    .input.scale.INPUT_TYPE = {
        .p_min = INPUT_FEATURES_SCALE_MIN,
        .p_max = INPUT_FEATURES_SCALE_MAX,
    }, 
    ///
    .p_dsp = P_DSP_PIPELINE,
    ///
    .model.meta.p_neuron_internal_links_num = MODEL_NEURON_INTERNAL_LINKS_NUM,
    .model.meta.p_neuron_external_links_num = MODEL_NEURON_EXTERNAL_LINKS_NUM,
    .model.meta.p_output_neurons_indices    = MODEL_OUTPUT_NEURONS_INDICES,
    .model.meta.p_neuron_links              = MODEL_NEURONS_LINKS,
    .model.meta.p_neuron_act_type_mask      = MODEL_NEURON_ACTIVATION_TYPE_MASK,
    .model.meta.outputs_num                 = MODEL_OUTPUTS_NUM,
    .model.meta.neurons_num                 = MODEL_NEURONS_NUM,
    .model.meta.weights_num                 = MODEL_WEIGHTS_NUM,
    .model.meta.task                        = MODEL_TASK,
    .model.meta.uses_as_input.all           = MODEL_USES_AS_INPUT_MASK,

    .model.params.MODEL_PARAMS_TYPE = {
        .p_weights      = MODEL_WEIGHTS,
        .p_act_weights = MODEL_NEURON_ACTIVATION_WEIGHTS,
        .p_neurons      = model_neurons_,
    },

    .model.output.memory.p_void = model_outputs_,
    .model.output.num = MODEL_OUTPUTS_NUM,
    ///
    .interfaces.input_setup = NN_INPUT_SETUP_INTERFACE,
    .interfaces.feed_inputs = NN_INPUT_FEED_INTERFACE,
    .interfaces.process_features = NN_PROCESS_FEATURES_INTERFACE,
    .interfaces.run_inference = NN_RUN_INFERENCE_INTERFACE,
    .interfaces.propagate_outputs = NN_PROPAGATE_OUTPUTS_INTERFACE,
    .interfaces.decode_outputs = NN_DECODE_OUTPUTS_INTERFACE,
    ///
    .decoded_output = { NN_DECODED_OUTPUT_INIT },
};

//////////////////////////////////////////////////////////////////////////////

nrf_edgeai_t* nrf_edgeai_user_model(void)
{
    return &nrf_edgeai_;
}

//////////////////////////////////////////////////////////////////////////////

uint32_t nrf_edgeai_user_model_size(void)
{
    uint32_t model_meta_size =
        (sizeof(MODEL_WEIGHTS) + sizeof(MODEL_NEURONS_LINKS) +
         sizeof(MODEL_NEURON_EXTERNAL_LINKS_NUM) + sizeof(MODEL_NEURON_INTERNAL_LINKS_NUM) +
         sizeof(MODEL_NEURON_ACTIVATION_WEIGHTS) + sizeof(MODEL_NEURON_ACTIVATION_TYPE_MASK) +
         sizeof(MODEL_OUTPUT_NEURONS_INDICES));

#if MODEL_TASK == __NRF_EDGEAI_TASK_ANOMALY_DETECTION
    model_meta_size += sizeof(MODEL_AVERAGE_EMBEDDING) + sizeof(MODEL_OUTPUT_SCALE_MIN) +
                       sizeof(MODEL_OUTPUT_SCALE_MAX);
#endif

#if MODEL_TASK == __NRF_EDGEAI_TASK_REGRESSION
    model_meta_size += sizeof(MODEL_OUTPUT_SCALE_MIN) + sizeof(MODEL_OUTPUT_SCALE_MAX);
#endif

    return model_meta_size;
}
