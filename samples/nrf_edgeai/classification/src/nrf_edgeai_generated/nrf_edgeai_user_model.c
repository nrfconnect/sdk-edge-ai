/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nrf_edgeai_user_model.h"
#include "nrf_edgeai_user_types.h"

#include <nrf_edgeai/rt/private/nrf_edgeai_interfaces.h>
#include <nrf_edgeai/nrf_edgeai_platform.h>

//////////////////////////////////////////////////////////////////////////////

#define EDGEAI_LAB_SOLUTION_ID_STR      "90449"
#define EDGEAI_RUNTIME_VERSION_COMBINED 0x00000001

//////////////////////////////////////////////////////////////////////////////
#define INPUT_TYPE f32

/** User input features type */
#define INPUT_FEATURE_DATA_TYPE NRF_EDGEAI_INPUT_F32

/** Number of unique features in the original input sample */
#define INPUT_UNIQ_FEATURES_NUM 1

/** Number of unique features actually used by NN from the original input sample */
#define INPUT_UNIQ_FEATURES_USED_NUM 1

/** Number of input feature samples that should be collected in the input window
 *  feature_sample = 1 * INPUT_UNIQ_FEATURES_NUM
 */
#define INPUT_WINDOW_SIZE 50

/** Number of input feature samples on that the input window is shifted */
#define INPUT_WINDOW_SHIFT 50

/** Number of subwindows in input feature window,
* the SUBWINDOW_SIZE = INPUT_WINDOW_SIZE / INPUT_SUBWINDOW_NUM
* if the window size is not divisible by the number of subwindows without a remainder,
* the remainder is added to the last subwindow size */
#define INPUT_SUBWINDOW_NUM 0

#define INPUT_UNIQUE_SCALES_NUM \
    (sizeof(INPUT_FEATURES_SCALE_MIN) / sizeof(INPUT_FEATURES_SCALE_MIN[0]))

//////////////////////////////////////////////////////////////////////////////
#define MODEL_NEURONS_NUM 80
#define MODEL_WEIGHTS_NUM 795
#define MODEL_OUTPUTS_NUM 7
#define MODEL_TASK        0
#define MODEL_PARAMS_TYPE f32
#define MODEL_REORDERING  0

#define MODEL_USES_AS_INPUT_INPUT_FEATURES 0
#define MODEL_USES_AS_INPUT_DSP_FEATURES   1
#define MODEL_USES_AS_INPUT_MASK \
    ((MODEL_USES_AS_INPUT_INPUT_FEATURES << 0) | (MODEL_USES_AS_INPUT_DSP_FEATURES << 1))

//////////////////////////////////////////////////////////////////////////////
/** Defines input(also used for LAG) features MIN scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MIN[] = { 6.2637229 };

/** Defines input(also used for LAG) features MAX scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MAX[] = { 232004.5625000 };

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
#define EXTRACTED_FEATURES_NUM 11

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

static const uint64_t FEATURES_EXTRACTION_MASK[] = { 0x308c39c00000000 };
/** Defines arguments used while feature extraction
 */

/** Defines arguments used while feature extraction
 */
#define FEATURES_EXTRACTION_ARGUMENTS NULL

/** Defines extracted features MIN scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MIN[] = {
    6.2608643,  45.0701027, 1.2371691, 1.5077032, 51.9344368, 0.0204082,
    45.0701027, 1.7665554,  0.0200000, 0.2883697, 0.9643899
};

/** Defines extracted features MAX scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MAX[] = {
    231012.2343750, 5638.1899414, 9054.6552734, 32338.0703125, 32825.9062500, 0.8775510,
    5638.1899414,   9455.0820312, 0.9600000,    1.8917454,     3.9208295
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
    nrf_edgeai_feature_std_f32,
    nrf_edgeai_feature_rms_f32,
    nrf_edgeai_feature_mcr_f32,
    nrf_edgeai_feature_absmean_f32,
    nrf_edgeai_feature_amdf_f32,
    nrf_edgeai_feature_psom_f32,
    nrf_edgeai_feature_hjorth_f32
};

static const nrf_edgeai_features_pipeline_ctx_t timedomain_pipeline_ = {
    .functions_num    = sizeof(timedomain_features_) / sizeof(timedomain_features_[0]),
    .functions.p_void = timedomain_features_,
    .p_ctx            = P_TIMEDOMAIN_FEATURES_CTX,
};
#define P_TIMEDOMAIN_PIPELINE &timedomain_pipeline_

#define P_FREQDOMAIN_PIPELINE NULL

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
    1.0000000,  1.0000000,  1.0000000,  1.0000000,  -0.1773506, 0.1025528,  -0.5000000, 0.5000000,
    -1.0000000, -1.0000000, 0.5000000,  -1.0000000, 0.5000000,  -0.9842999, 0.7284356,  0.3760768,
    0.8240758,  0.5000000,  -0.5000000, 0.4416595,  0.1892119,  -1.0000000, 0.7269202,  0.7727900,
    0.2117875,  -0.3944074, -0.6242056, 1.0000000,  0.9990233,  1.0000000,  0.2950771,  -1.0000000,
    0.8125002,  0.1011844,  0.3460517,  0.5238172,  -0.3342050, 0.5000000,  0.8271456,  -0.5000000,
    -0.3121242, -0.9999983, 0.4790727,  -0.3215830, 1.0000000,  1.0000000,  -0.0788430, 0.0320328,
    0.9112340,  -0.8985478, -0.2160975, 0.0412205,  -0.1201425, -0.0140920, -1.0000000, -0.1068797,
    -1.0000000, -1.0000000, -0.2275100, -1.0000000, -0.0991692, 0.2183208,  -1.0000000, 0.3037523,
    0.9999999,  -0.0063237, -0.5000000, -1.0000000, 0.0158322,  0.2096808,  -0.8191613, -0.7625924,
    -0.4706602, 0.2274858,  -1.0000000, 0.4826190,  0.6958395,  0.0175143,  -0.7934636, 0.9115231,
    -0.1267306, 0.8750000,  -0.8124532, 0.2500370,  -1.0000000, 1.0000000,  0.2243235,  -0.2303281,
    -0.8843036, -1.0000000, 0.5303499,  -0.5000000, -0.8171355, 0.4194192,  0.1758527,  0.1075206,
    1.0000000,  -0.9778006, -0.0146906, -0.2822467, 0.5788659,  -0.8214819, -1.0000000, -0.6362679,
    -0.4778717, 0.7313876,  -1.0000000, 0.7500000,  -1.0000000, -1.0000000, -1.0000000, -1.0000000,
    0.0305578,  -0.3000741, 0.0632161,  -0.9627988, -0.0550063, 1.0000000,  0.6250000,  -0.9816962,
    1.0000000,  0.0629631,  0.0232007,  0.0252995,  1.0000000,  -0.9999996, -1.0000000, -0.7103697,
    -1.0000000, 0.1444723,  -0.0620008, 0.0713461,  -0.9421639, 0.2342067,  1.0000000,  0.9687500,
    1.0000000,  1.0000000,  -0.4770938, 0.7961850,  -1.0000000, 0.5116186,  -1.0000000, 0.2930011,
    -0.2179618, -0.8369197, 0.4976150,  0.9780653,  -0.9988567, 1.0000000,  -0.3387719, -0.4549440,
    0.5286108,  0.2058240,  -0.7941926, -0.1281076, -0.0238827, -0.2488818, 0.8871075,  -1.0000000,
    0.1988315,  -0.5625000, 0.0203431,  0.3144327,  -0.1604952, 0.0028135,  0.3691303,  0.0139924,
    -0.9375000, -0.6011256, -1.0000000, 1.0000000,  -0.0415764, 0.5000000,  0.0133260,  0.0075792,
    0.8626838,  -0.6901675, -0.9541095, 1.0000000,  -0.4187873, 0.3525417,  -0.3749979, 0.0008210,
    -0.3270886, 1.0000000,  -0.2417055, -1.0000000, -0.9159633, 0.9589844,  0.8218943,  -0.6807768,
    0.4894831,  -0.3278691, -0.7923374, -0.7438374, 0.0011508,  0.2595338,  0.9806232,  0.6219366,
    -0.1184394, 1.0000000,  1.0000000,  -1.0000000, 0.2635160,  0.9999999,  0.2258377,  -0.1310730,
    -0.4143657, -0.0182059, 0.2489015,  0.3289461,  0.0528807,  0.7812316,  -0.0340433, -1.0000000,
    1.0000000,  -0.9999999, -0.7090486, -0.2328943, 0.0067852,  0.0020000,  0.0146502,  0.2354937,
    0.0576414,  -1.0000000, 0.0328556,  -1.0000000, -0.9678286, 0.0074041,  0.0046363,  0.4014447,
    -0.4892308, 1.0000000,  -1.0000000, 0.0020955,  0.8739697,  -0.5955151, -1.0000000, -1.0000000,
    -0.3183214, -0.3844346, -0.6172144, 1.0000000,  1.0000000,  1.0000000,  -1.0000000, -1.0000000,
    0.9687500,  0.0149751,  -0.6619177, -0.5927357, -1.0000000, 0.6862462,  -0.7551440, -1.0000000,
    -0.5820354, 0.0013487,  0.0007687,  -0.0595816, 0.4193061,  0.0002352,  -0.2129818, 1.0000000,
    0.9060213,  0.0064873,  -0.1749456, 0.0057176,  -0.1891191, 0.0010626,  0.0012318,  0.1855356,
    -0.1742892, -0.1911259, 0.8187619,  0.5569951,  0.0781906,  0.9164071,  1.0000000,  0.9999999,
    0.9999998,  1.0000000,  1.0000000,  1.0000000,  -0.5000000, -0.2058733, -0.4947093, -0.4198906,
    0.2645468,  -0.6688218, -0.1151882, 0.0094565,  -0.9438559, -1.0000000, -0.4957004, -1.0000000,
    -1.0000000, 0.8581368,  0.0577298,  0.7823514,  -0.7923394, -0.7465191, -0.7602159, 1.0000000,
    0.2269868,  -0.7174520, -0.9131186, 0.7754206,  0.8102347,  -0.0900926, -1.0000000, -1.0000000,
    -0.0663935, 0.9380340,  -0.1111779, -1.0000000, 0.9180208,  -0.8751062, -0.8704910, 0.2691562,
    -0.9226307, 0.8868504,  0.0866064,  -1.0000000, -1.0000000, 0.9166111,  -1.0000000, -1.0000000,
    -1.0000000, 0.5000000,  0.6841391,  0.0557042,  0.6265390,  -1.0000000, 0.3022797,  0.4375000,
    0.9011333,  0.6029691,  -0.9375000, 0.1145151,  -0.7303995, 1.0000000,  -0.6866471, -0.9146206,
    1.0000000,  0.0910903,  0.8218440,  0.0854587,  -0.6708930, -0.3826108, 0.3768105,  -0.1569728,
    -0.5631879, -0.6417785, 0.7055346,  -0.0697696, -0.7037482, -0.0774234, 0.9184486,  -0.8750000,
    -0.7500000, 0.7094673,  0.1385086,  -0.5990146, -0.6743139, -0.2374120, 0.3218946,  0.3178399,
    0.6557505,  0.7500000,  -0.8750000, -0.0988207, -0.6250000, -0.5000000, -0.7058358, 1.0000000,
    -1.0000000, -0.4003033, -0.8121694, 0.2209916,  0.8878376,  -1.0000000, -0.5571169, 0.8750000,
    -0.3059891, -1.0000000, -0.1937073, 0.9062500,  0.5067989,  -1.0000000, -0.0500376, -1.0000000,
    0.2819609,  0.2048353,  -1.0000000, -1.0000000, -1.0000000, 0.0307720,  0.5000000,  0.0135049,
    0.0328295,  -0.1182518, 0.0237122,  0.1048499,  1.0000000,  0.5000000,  -0.8951370, -1.0000000,
    0.0546942,  0.5725595,  -1.0000000, -0.5069597, 1.0000000,  -1.0000000, 0.5000000,  -1.0000000,
    -0.3437901, 1.0000000,  -0.0714142, 0.0404937,  0.7772402,  0.5000000,  -1.0000000, 1.0000000,
    -1.0000000, 0.0501351,  -0.7496489, -0.2431490, -1.0000000, 0.9836172,  -0.3214822, -0.8731251,
    0.6795083,  -0.7672634, 0.5397292,  1.0000000,  -0.0507064, -0.2635911, 0.1636506,  -1.0000000,
    -0.9999977, -1.0000000, -0.6250000, -1.0000000, -1.0000000, -0.7883164, 0.8381751,  -0.4636860,
    0.6793420,  0.1680345,  -0.1725015, 1.0000000,  -0.8582326, 1.0000000,  -0.6950718, 0.0058796,
    -0.8750000, 0.5000000,  -1.0000000, 0.8750000,  0.5372663,  -0.5000000, 0.0011561,  -0.3243369,
    -0.2341914, 0.2984449,  -0.2328784, 1.0000000,  -0.2534707, 1.0000000,  -0.1001693, 0.0100845,
    -0.2275345, -1.0000000, -0.9997966, 0.8116736,  0.0261347,  -0.2848238, -0.8914250, 0.8631911,
    0.0066426,  -0.1144011, -0.5000000, 0.1330238,  0.1555669,  -0.0796103, -0.8248091, 0.0883630,
    -1.0000000, -1.0000000, -0.4237557, 0.6095541,  -1.0000000, 0.3136213,  0.9346626,  -0.4176250,
    1.0000000,  -0.6605196, -0.9979347, -0.0999720, 0.7560310,  0.0822129,  -0.3990116, 0.0168902,
    0.0324038,  0.0005763,  -0.0720171, -0.7615088, -0.8910491, -1.0000000, 0.7644322,  0.2823682,
    1.0000000,  -0.8750000, 0.0025036,  0.2198258,  -0.0243208, -0.6301464, 1.0000000,  -1.0000000,
    0.0069873,  -0.2103111, -0.8536816, 0.5130774,  0.1147240,  -0.7724609, 0.5000000,  0.1235306,
    -0.7876673, 1.0000000,  -0.4880137, 0.0482299,  0.0057061,  -0.2014823, 1.0000000,  -0.2541978,
    -0.5937765, -1.0000000, 1.0000000,  -0.4016438, 0.7292826,  0.8893278,  -0.0776985, -1.0000000,
    -0.5000000, 0.0176350,  0.0068114,  -0.5030549, 0.7500000,  -0.4006400, 1.0000000,  -0.8815008,
    1.0000000,  1.0000000,  -1.0000000, 0.7914675,  0.0097839,  0.2750760,  -0.2699279, 0.3750000,
    0.5808296,  1.0000000,  -1.0000000, -0.2474433, 0.5000000,  0.0006630,  -0.9999964, 0.3863725,
    0.3421431,  1.0000000,  0.7163745,  1.0000000,  0.7578177,  -1.0000000, -0.1630361, -0.5396950,
    1.0000000,  -0.2500011, -0.4488296, -0.3418921, -0.9061809, 0.4519985,  -0.8712118, -0.5000000,
    -0.9740979, -0.9419683, 1.0000000,  -1.0000000, 0.8731297,  0.9327782,  0.2231661,  -0.3764215,
    0.0027068,  -0.4701780, -0.0113462, -0.7776670, -0.8676758, -1.0000000, 0.3764222,  0.5637386,
    -0.9765625, 0.3876542,  -0.0338921, -0.7339121, 0.0777751,  0.0418566,  -0.1447571, 0.0746477,
    -1.0000000, 0.6297495,  1.0000000,  1.0000000,  0.9794444,  0.5018972,  0.9449025,  -1.0000000,
    0.8069298,  -0.4147857, 0.0011893,  0.3125000,  1.0000000,  -0.3059373, 1.0000000,  0.3423465,
    -0.0445676, 0.0011535,  0.0036323,  -0.1055557, -0.4136157, 0.5478845,  -0.4001391, 0.0627365,
    -0.3004450, -0.0811784, -0.1324017, -1.0000000, 0.9062500,  0.0105411,  -1.0000000, -1.0000000,
    0.0512320,  -0.0170938, 0.0966434,  0.1462382,  -0.6958621, -1.0000000, 0.5183185,  0.2930206,
    0.0635680,  -0.1754706, -1.0000000, -1.0000000, 1.0000000,  -1.0000000, 1.0000000,  -1.0000000,
    0.6874999,  0.0000000,  -0.9721296, 1.0000000,  0.6503391,  -1.0000000, 1.0000000,  0.8340515,
    0.9921875,  -0.4448150, 0.8699652,  -0.3868237, 0.9999999,  0.3509361,  0.9597781,  -0.6318178,
    -0.2038928, -0.3116137, -0.0473055, -1.0000000, -0.6720163, 1.0000000,  0.5561935,  0.0415795,
    0.0022340,  0.0062639,  -0.9211799, -0.6280386, -0.5293612, -0.5245395, 0.3216876,  0.4690169,
    0.1701325,  0.0032964,  -1.0000000, 1.0000000,  -1.0000000, 1.0000000,  1.0000000,  -1.0000000,
    1.0000000,  0.9999994,  -1.0000000, -0.6250000, 0.7082075,  -1.0000000, 1.0000000,  -1.0000000,
    0.8780289,  -1.0000000, -1.0000000, 0.0000000,  -0.8011363, -0.2118544, -0.8750000, -1.0000000,
    0.1239833,  0.1408712,  0.8333189,  0.2958576,  1.0000000,  -0.5000000, 0.6870841,  -0.9687500,
    0.0109187,  0.4139082,  -1.0000000, -0.9062499, 0.8750000,  -0.9788048, 0.4999979,  -1.0000000,
    1.0000000,  -1.0000000, -1.0000000, -1.0000000, 1.0000000,  -1.0000000, 1.0000000,  1.0000000,
    -1.0000000, 1.0000000,  -0.3771273, 0.2819206,  -0.7601624, 0.3273069,  -0.0931032, 0.6240888,
    -0.0771301, -0.2997367, 0.0304659,  -0.4907390, -1.0000000, -1.0000000, -1.0000000, 1.0000000,
    -1.0000000, 0.9843750,  -1.0000000, 0.3398438,  0.2059470,  -0.2569979, 0.0035238,  -0.3992916,
    0.4049694,  0.7991645,  -1.0000000, -0.9674876, -0.4136197, 0.0126753,  -1.0000000, 0.7578125,
    -1.0000000, -1.0000000, -1.0000000, 0.7187499,  -1.0000000, 0.4063349,  0.0108147,  -0.4436664,
    0.0550596,  0.0026405,  -0.9609630, 0.9980469,  0.0004531,  0.0016569,  -0.6707963, -1.0000000,
    -1.0000000, 1.0000000,  -0.7151117, 0.5000000,  -1.0000000, -1.0000000, 1.0000000,  1.0000000,
    1.0000000,  1.0000000,  0.1757144
};

static const uint16_t MODEL_NEURONS_LINKS[] = {
    0,  2,  3,  7,  9,  11, 0,  0,  1,  2,  4,  6,  11, 0,  1,  0,  1,  4,  7,  8,  11, 1,  11, 0,
    1,  2,  3,  0,  1,  2,  5,  7,  9,  10, 11, 1,  2,  0,  1,  2,  5,  7,  11, 0,  0,  3,  8,  9,
    11, 0,  1,  2,  4,  5,  0,  1,  2,  3,  4,  7,  8,  11, 1,  2,  3,  5,  0,  3,  7,  8,  11, 0,
    2,  4,  8,  0,  2,  4,  7,  8,  11, 1,  3,  11, 3,  10, 11, 0,  1,  2,  6,  9,  0,  7,  9,  11,
    1,  3,  4,  5,  7,  1,  2,  4,  6,  11, 0,  7,  0,  2,  3,  7,  11, 0,  1,  14, 1,  2,  3,  4,
    7,  8,  11, 1,  7,  0,  2,  4,  7,  8,  11, 0,  1,  2,  7,  9,  14, 16, 0,  1,  2,  4,  8,  9,
    11, 2,  5,  6,  8,  9,  12, 0,  2,  9,  11, 2,  5,  6,  8,  18, 0,  4,  5,  7,  8,  9,  11, 5,
    13, 16, 0,  2,  4,  7,  9,  11, 2,  6,  12, 18, 9,  11, 1,  2,  5,  7,  9,  13, 14, 16, 18, 20,
    0,  1,  2,  4,  9,  11, 0,  5,  15, 17, 22, 0,  1,  2,  8,  9,  11, 1,  3,  5,  9,  15, 21, 23,
    0,  2,  7,  11, 2,  6,  9,  18, 23, 0,  1,  2,  7,  11, 2,  20, 23, 0,  7,  11, 2,  5,  14, 26,
    1,  11, 5,  17, 19, 25, 26, 0,  1,  11, 2,  12, 18, 19, 24, 26, 0,  2,  9,  10, 11, 2,  24, 26,
    0,  1,  2,  5,  7,  8,  11, 0,  1,  2,  5,  6,  9,  13, 16, 17, 22, 23, 28, 30, 0,  1,  4,  6,
    8,  11, 2,  4,  6,  22, 28, 31, 0,  1,  8,  11, 0,  2,  9,  30, 31, 32, 1,  11, 6,  9,  26, 30,
    31, 32, 33, 0,  1,  11, 0,  9,  13, 14, 16, 24, 31, 32, 33, 0,  1,  4,  10, 11, 13, 0,  1,  4,
    9,  10, 11, 0,  2,  8,  12, 18, 19, 20, 28, 31, 0,  2,  3,  7,  8,  10, 11, 2,  6,  18, 37, 0,
    2,  11, 6,  12, 13, 18, 23, 33, 38, 0,  7,  11, 6,  31, 35, 0,  1,  4,  6,  10, 11, 2,  12, 19,
    31, 38, 0,  2,  11, 15, 24, 30, 31, 32, 0,  2,  3,  4,  7,  8,  11, 1,  5,  15, 20, 25, 26, 30,
    31, 32, 37, 40, 42, 0,  2,  3,  4,  7,  8,  11, 2,  32, 40, 0,  2,  7,  8,  11, 2,  26, 31, 0,
    2,  8,  11, 1,  2,  5,  13, 20, 24, 26, 31, 38, 0,  1,  4,  5,  9,  10, 11, 35, 0,  4,  10, 11,
    13, 25, 26, 28, 43, 3,  11, 0,  13, 16, 20, 26, 42, 47, 4,  11, 16, 20, 26, 30, 38, 47, 48, 4,
    11, 4,  6,  12, 13, 23, 38, 41, 45, 48, 0,  11, 20, 24, 26, 31, 35, 42, 48, 51, 0,  2,  7,  8,
    11, 4,  15, 24, 42, 48, 51, 52, 0,  7,  11, 7,  24, 26, 48, 0,  11, 26, 31, 32, 35, 43, 46, 47,
    48, 53, 7,  9,  11, 13, 25, 30, 31, 34, 35, 43, 47, 49, 1,  7,  10, 11, 5,  25, 26, 27, 47, 55,
    56, 2,  3,  7,  11, 5,  15, 30, 47, 56, 0,  5,  7,  11, 0,  1,  4,  6,  21, 22, 30, 32, 40, 51,
    53, 2,  4,  9,  11, 0,  21, 23, 42, 43, 48, 51, 52, 8,  10, 11, 0,  1,  5,  14, 16, 22, 42, 51,
    52, 54, 58, 0,  1,  4,  10, 11, 3,  22, 53, 58, 4,  11, 16, 23, 58, 10, 11, 47, 53, 55, 56, 3,
    4,  9,  11, 20, 24, 32, 35, 39, 49, 55, 56, 64, 0,  1,  2,  3,  9,  11, 5,  12, 18, 41, 2,  8,
    9,  11, 1,  18, 19, 21, 29, 41, 66, 11, 5,  6,  16, 21, 22, 24, 29, 32, 40, 51, 53, 58, 64, 1,
    9,  11, 13, 22, 55, 56, 57, 1,  11, 5,  13, 22, 25, 32, 35, 55, 4,  11, 0,  7,  15, 16, 17, 20,
    22, 27, 35, 36, 46, 49, 50, 64, 65, 69, 70, 11, 14, 15, 24, 32, 38, 51, 52, 54, 64, 65, 0,  7,
    11, 6,  14, 23, 24, 25, 26, 28, 30, 42, 43, 44, 45, 48, 52, 53, 54, 72, 11, 0,  16, 20, 21, 22,
    72, 4,  11, 4,  12, 59, 60, 61, 62, 63, 68, 74, 11, 7,  12, 13, 34, 38, 41, 45, 3,  11, 2,  9,
    37, 38, 39, 51, 76, 11, 5,  25, 26, 47, 50, 56, 5,  11, 5,  13, 31, 32, 33, 34, 40, 55, 56, 57,
    58, 78, 11
};

static const uint16_t MODEL_NEURON_INTERNAL_LINKS_NUM[] = {
    0,   7,   15,  22,  27,  37,  44,  54,  66,  75,  83,  86,  92,  101, 108, 116,
    125, 138, 151, 160, 170, 180, 192, 203, 216, 225, 233, 240, 247, 256, 264, 284,
    296, 306, 315, 327, 333, 348, 359, 369, 375, 386, 394, 413, 423, 431, 444, 452,
    461, 470, 479, 490, 500, 512, 519, 530, 542, 553, 562, 577, 589, 603, 612, 617,
    623, 636, 646, 657, 671, 679, 688, 707, 718, 738, 745, 756, 764, 773, 780, 794
};

static const uint16_t MODEL_NEURON_EXTERNAL_LINKS_NUM[] = {
    6,   13,  21,  23,  35,  43,  49,  62,  71,  81,  84,  87,  96,  106, 113, 123,
    131, 145, 155, 167, 176, 182, 198, 209, 220, 230, 236, 242, 250, 261, 271, 290,
    300, 308, 318, 332, 339, 355, 362, 372, 381, 389, 401, 420, 428, 435, 451, 456,
    463, 472, 481, 492, 505, 515, 521, 533, 546, 557, 566, 581, 592, 608, 614, 619,
    627, 642, 650, 658, 674, 681, 690, 708, 721, 739, 747, 757, 766, 774, 782, 795
};

static const nrf_user_coeff_t MODEL_NEURON_ACTIVATION_WEIGHTS[] = {
    40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000, 28.1546803,
    40.0000000, 40.0000000, 40.0000000, 40.0000000, 28.0245857, 23.5921097, 40.0000000, 37.5062370,
    37.5062370, 37.5062370, 40.0000000, 40.0000000, 37.5062370, 40.0000000, 37.5062370, 40.0000000,
    40.0000000, 40.0000000, 40.0000000, 37.5062370, 40.0000000, 40.0000000, 40.0000000, 33.9899216,
    27.9798431, 37.2731209, 37.2731209, 37.5062370, 37.5062370, 40.0000000, 40.0000000, 40.0000000,
    37.2731209, 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000, 37.5062370, 37.2731209,
    40.0000000, 37.5062370, 37.5062370, 40.0000000, 40.0000000, 40.0000000, 40.0000000, 37.2731209,
    40.0000000, 40.0000000, 40.0000000, 35.8122940, 40.0000000, 40.0000000, 40.0000000, 40.0000000,
    37.5062370, 37.5062370, 40.0000000, 40.0000000, 40.0000000, 37.5062370, 37.5062370, 37.5062370,
    40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000
};

static const uint8_t MODEL_NEURON_ACTIVATION_TYPE_MASK[] = { 0xff, 0xf7, 0xff, 0xff, 0xff,
                                                             0xff, 0xff, 0xff, 0x77, 0x55 };

static const uint16_t MODEL_OUTPUT_NEURONS_INDICES[] = { 71, 67, 77, 11, 75, 79, 73 };

#define NN_DECODED_OUTPUT_INIT                \
    .classif = {                              \
        .predicted_class = 0,                 \
        .num_classes     = MODEL_OUTPUTS_NUM, \
    }

//////////////////////////////////////////////////////////////////////////////
#define NN_INPUT_SETUP_INTERFACE       nrf_edgeai_input_setup_discrete_window
#define NN_INPUT_FEED_INTERFACE        nrf_edgeai_input_feed_discrete_window_f32
#define NN_PROCESS_FEATURES_INTERFACE  nrf_edgeai_process_features_dsp_f32_f32
#define NN_RUN_INFERENCE_INTERFACE     nrf_edgeai_run_model_inference_f32
#define NN_PROPAGATE_OUTPUTS_INTERFACE nrf_edgeai_output_propagate_f32
#define NN_DECODE_OUTPUTS_INTERFACE    nrf_edgeai_output_decode_classification_f32

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
