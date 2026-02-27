#include "nrf_edgeai_user_model.h"
#include "nrf_edgeai_user_types.h"

#include <nrf_edgeai/rt/private/nrf_edgeai_interfaces.h>
#include <nrf_edgeai/nrf_edgeai_platform.h>

//////////////////////////////////////////////////////////////////////////////

#define EDGEAI_LAB_SOLUTION_ID_STR	"91278"
#define EDGEAI_RUNTIME_VERSION_COMBINED 0x00000202

//////////////////////////////////////////////////////////////////////////////
#define INPUT_TYPE i16

/** User input features type */
#define INPUT_FEATURE_DATA_TYPE NRF_EDGEAI_INPUT_I16

/** Number of unique features in the original input sample */
#define INPUT_UNIQ_FEATURES_NUM 6

/** Number of unique features actually used by NN from the original input sample */
#define INPUT_UNIQ_FEATURES_USED_NUM 6

/** Number of input feature samples that should be collected in the input window
 *  feature_sample = 1 * INPUT_UNIQ_FEATURES_NUM
 */
#define INPUT_WINDOW_SIZE 99

/** Number of input feature samples on that the input window is shifted */
#define INPUT_WINDOW_SHIFT 33

/** Number of subwindows in input feature window,
 * the SUBWINDOW_SIZE = INPUT_WINDOW_SIZE / INPUT_SUBWINDOW_NUM
 * if the window size is not divisible by the number of subwindows without a remainder,
 * the remainder is added to the last subwindow size */
#define INPUT_SUBWINDOW_NUM 0

#define INPUT_UNIQUE_SCALES_NUM                                                                    \
	(sizeof(INPUT_FEATURES_SCALE_MIN) / sizeof(INPUT_FEATURES_SCALE_MIN[0]))

//////////////////////////////////////////////////////////////////////////////
/** Defines input(also used for LAG) features MIN scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MIN[] = {-32765, -32755, -32768,
							    -17453, -17453, -17453};

/** Defines input(also used for LAG) features MAX scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MAX[] = {32754, 32763, 32763,
							    17453, 17453, 17453};

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
#define INPUT_TYPE_SIZE                                                                            \
	((sizeof(nrf_user_input_t) > sizeof(nrf_user_neuron_t)) ? sizeof(nrf_user_input_t)         \
								: sizeof(nrf_user_neuron_t))

/** Input features window size in bytes to allocate statically */
#define INPUT_WINDOW_BUFFER_SIZE_BYTES                                                             \
	(INPUT_WINDOW_SIZE * INPUT_UNIQ_FEATURES_NUM * INPUT_TYPE_SIZE)

static uint8_t input_window_[INPUT_WINDOW_BUFFER_SIZE_BYTES] __NRF_EDGEAI_ALIGNED;

#define INPUT_WINDOW_MEMORY &input_window_[0]

static nrf_edgeai_window_ctx_t input_window_ctx_;
#define P_INPUT_WINDOW_CTX &input_window_ctx_

//////////////////////////////////////////////////////////////////////////////
/** The maximum number of extracted features that user used for all unique input features */
#define EXTRACTED_FEATURES_NUM 85

#define EXTRACTED_FEATURES_META_TYPE i32

/** DSP feature buffer element size,
 * if quantization of model is bigger than DSP features size in bits,
 * the size of extracted DSP features buffer should aligned to nrf_user_neuron_t */
#define EXTRACTED_FEATURE_SIZE_BYTES                                                               \
	((sizeof(nrf_user_feature_t) > sizeof(nrf_user_neuron_t)) ? sizeof(nrf_user_feature_t)     \
								  : sizeof(nrf_user_neuron_t))

/** Size of extracted features buffer in bytes */
#define EXTRACTED_FEATURES_BUFFER_SIZE_BYTES (EXTRACTED_FEATURES_NUM * EXTRACTED_FEATURE_SIZE_BYTES)

/** Defines feature extraction masks used as nrf_edgeai_features_mask_t,
 *  64 bit for one unique input feature, @ref nrf_edgeai_features_mask_t to see bitmask
 */

static const uint64_t FEATURES_EXTRACTION_MASK[] = {0x5dc79b00000000, 0x5cc79b00000000,
						    0x4dc79b00000000, 0x5dc79b00000000,
						    0x5cc79b00000000, 0x5cc59b00000000};

/** Defines arguments used while feature extraction
 */
static const nrf_user_input_t FEATURES_EXTRACTION_ARGUMENTS[] = {1, 1, 1, 1, 1, 1, 1, 1};

/** Defines extracted features MIN scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MIN[] = {
	-32765, -8389, -14578, 20, 30,	81, 10, 0,   61, 14,  0,   0,	121,	0,	172,
	-32755, -7858, -13763, 23, 38,	89, 10, 0,   73, 15,  0,   70,	0,	224,	-32768,
	-9223,	-9723, 32,     55, 237, 10, 0,	197, 24, 0,   0,   161, 305,	-17453, -2164,
	-5657,	6,     9,      9,  10,	0,  7,	4,   0,	 0,   121, 0,	60,	-17453, -120,
	-3762,	5,     8,      8,  10,	0,  6,	2,   0,	 101, 0,   35,	-17453, -1800,	-7437,
	2,	6,     7,      0,  4,	2,  0,	111, 0,	 35};

/** Defines extracted features MAX scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MAX[] = {
	9279, 32754, 13671, 26554, 26986, 27123, 520,	459,   26856, 15962, 397,   1000,  858,
	404,  65490, 9192,  32763, 11883, 15605, 18527, 18919, 428,   438,   16826, 18296, 1000,
	909,  333,   64858, 9759,  32763, 12830, 18826, 20438, 20695, 469,   489,   18823, 19239,
	346,  1000,  909,   65508, 62,	  17453, 6005,	9845,  11612, 11617, 357,   357,   9871,
	7798, 265,   1000,  888,   292,	  64749, 124,	17453, 5320,  11136, 12349, 12378, 255,
	275,  11054, 8366,  1000,  888,	  323,	 63823, 500,   17453, 4392,  8564,  10164, 10913,
	295,  9436,  5215,  1000,  898,	  303,	 52460};

/** Memory allocation to store extracted features during DSP pipeline */
static uint8_t
	extracted_features_buffer_[EXTRACTED_FEATURES_BUFFER_SIZE_BYTES] __NRF_EDGEAI_ALIGNED;

/** Timedomain features processing context  */
#define P_TIMEDOMAIN_FEATURES_CTX NULL
/** Timedomain features in feature extraction pipeline  */
static const nrf_edgeai_features_pipeline_func_i16_t timedomain_features_[] = {
	nrf_edgeai_feature_utility_tss_sum_i16,
	nrf_edgeai_feature_min_max_range_i16,
	nrf_edgeai_feature_mean_i16,
	nrf_edgeai_feature_mad_i16,
	nrf_edgeai_feature_std_i16,
	nrf_edgeai_feature_rms_i16,
	nrf_edgeai_feature_mcr_i16,
	nrf_edgeai_feature_zcr_i16,
	nrf_edgeai_feature_absmean_i16,
	nrf_edgeai_feature_amdf_i16,
	nrf_edgeai_feature_pscr_i16,
	nrf_edgeai_feature_psoz_i16,
	nrf_edgeai_feature_psom_i16,
	nrf_edgeai_feature_psos_i16,
	nrf_edgeai_feature_rmds_i16};

static const nrf_edgeai_features_pipeline_ctx_t timedomain_pipeline_ = {
	.functions_num = sizeof(timedomain_features_) / sizeof(timedomain_features_[0]),
	.functions.p_void = timedomain_features_,
	.p_ctx = P_TIMEDOMAIN_FEATURES_CTX,
};
#define P_TIMEDOMAIN_PIPELINE &timedomain_pipeline_

#define P_FREQDOMAIN_PIPELINE NULL

#define P_CUSTOMDOMAIN_PIPELINE NULL

static nrf_edgeai_dsp_pipeline_t dsp_pipeline_ = {
	.features =
		{
			.p_masks = (nrf_edgeai_features_mask_t *)FEATURES_EXTRACTION_MASK,
			.buffer.p_void = extracted_features_buffer_,
			.overall_num = EXTRACTED_FEATURES_NUM,
			.masks_num = sizeof(FEATURES_EXTRACTION_MASK) /
				     sizeof(FEATURES_EXTRACTION_MASK[0]),

			.p_timedomain_pipeline = P_TIMEDOMAIN_PIPELINE,
			.p_freqdomain_pipeline = P_FREQDOMAIN_PIPELINE,
			.p_customdomain_pipeline = P_CUSTOMDOMAIN_PIPELINE,

			.meta.EXTRACTED_FEATURES_META_TYPE =
				{
					.p_min = EXTRACTED_FEATURES_SCALE_MIN,
					.p_max = EXTRACTED_FEATURES_SCALE_MAX,
					.p_arguments = FEATURES_EXTRACTION_ARGUMENTS,
				},
		},
};

#define P_DSP_PIPELINE &dsp_pipeline_

//////////////////////////////////////////////////////////////////////////////

#define MODEL_TYPE	  __NRF_EDGEAI_MODEL_NEUTON
#define MODEL_TASK	  0
#define MODEL_OUTPUTS_NUM 8

#if MODEL_TYPE == __NRF_EDGEAI_MODEL_AXON
#include <drivers/axon/nrf_axon_nn_infer.h>
#include <axon/nrf_axon_platform.h>
#include "nrf_edgeai_user_model_axon.h"
#define P_MODEL_INSTANCE &model_instance_
#else // MODEL_TYPE == __NRF_EDGEAI_MODEL_NEUTON
#define P_MODEL_INSTANCE &model_instance_
#endif

//////////////////////////////////////////////////////////////////////////////

#define MODEL_NEURONS_NUM 45
#define MODEL_WEIGHTS_NUM 501
#define MODEL_PARAMS_TYPE q16
#define MODEL_REORDERING  1

#define MODEL_USES_AS_INPUT_INPUT_FEATURES 0
#define MODEL_USES_AS_INPUT_DSP_FEATURES   1
#define MODEL_USES_AS_INPUT_MASK                                                                   \
	((MODEL_USES_AS_INPUT_INPUT_FEATURES << 0) | (MODEL_USES_AS_INPUT_DSP_FEATURES << 1))

static const nrf_user_weight_t MODEL_WEIGHTS[] = {
	32763,	32763,	32766,	31159,	-24506, -31629, 32759,	-11103, 32752,	32766,	32766,
	32758,	30553,	12801,	30416,	15818,	8422,	-13803, -31884, 12816,	-3832,	31059,
	7105,	27018,	-19441, 14354,	-3353,	-24971, 12899,	15182,	23972,	6138,	-3732,
	-22094, -13867, 27958,	26661,	1380,	-4392,	-2277,	-32766, -11924, -12745, 8254,
	11427,	14260,	4930,	-28394, 31623,	3481,	5789,	11676,	-4307,	-32502, -3102,
	8905,	17172,	-13972, 18991,	-15356, 10617,	-26989, -2687,	3115,	-22941, -2286,
	-4246,	-11257, 24934,	-9916,	14245,	-7900,	-14821, 11285,	-6315,	22555,	-31269,
	6358,	-24517, 10703,	6778,	-32766, -22526, 24182,	-32766, -19994, 30950,	-2255,
	-5365,	-20793, -2821,	32761,	32766,	20543,	-13632, 32755,	-10701, -6632,	6351,
	-13346, -17700, -4677,	-2211,	9378,	16012,	32764,	23040,	1589,	4751,	-29216,
	11080,	-15594, 21374,	-7777,	-9533,	-2024,	-8602,	-30694, 28821,	-32765, -15068,
	-32766, -11910, -32766, 16383,	-760,	3565,	3554,	-721,	30465,	-32548, 27994,
	-11300, 19100,	8912,	-5268,	-4353,	3386,	-21489, -26324, -25134, 939,	-5650,
	12739,	11968,	32765,	-32768, -9830,	-25522, -5731,	31319,	15137,	-31859, -29177,
	32641,	32767,	-32768, -18026, 29576,	19942,	-32406, -32766, -12147, 10933,	-9801,
	29828,	-24993, -22067, 32766,	-31295, -32766, -5866,	7087,	9576,	-5642,	-19314,
	32763,	-6095,	-1889,	-6166,	-2810,	2909,	32759,	32749,	3636,	-27189, 6736,
	-13592, 24801,	10451,	7098,	-7735,	-10753, -20383, 32765,	-32768, -10086, 20509,
	32760,	-32768, -17130, -10587, -4570,	16407,	-10338, 4945,	8257,	-30427, 5310,
	-6981,	11747,	7610,	-11269, -16151, 3424,	8559,	16675,	-32740, 21738,	-638,
	2168,	26593,	-31352, 14333,	-16933, -6008,	10838,	-15235, 20582,	25519,	-32768,
	-23357, 32748,	32493,	28562,	-29100, -24243, -28673, -32768, 11151,	-24457, -32768,
	-32766, -26575, 16184,	-2020,	2497,	198,	16087,	-1701,	-7860,	6438,	-23028,
	4196,	32272,	6402,	-6819,	17844,	3372,	32763,	6113,	-654,	-637,	-3124,
	3683,	-381,	-32766, 32766,	32687,	1425,	-5788,	-32766, -11282, 26960,	-32766,
	-14591, 10080,	-20006, -4686,	21685,	32765,	8252,	12598,	-28682, -16577, -21724,
	-13623, -5118,	14738,	-32715, -28678, -1386,	32334,	-9893,	-32766, -27239, -30142,
	-3976,	10695,	8500,	10813,	-25676, -29525, 6410,	18690,	-29945, -18429, -13105,
	32764,	-31770, 2533,	-20284, 27635,	-31230, -8850,	14529,	-32542, -3782,	-6150,
	-32422, -6243,	-4215,	-9839,	-554,	-18871, 4548,	15475,	-3529,	-31587, -32766,
	-22998, 15758,	16567,	24722,	-25608, 13477,	14535,	27815,	-14438, 29216,	-32766,
	-32768, 14025,	32767,	-32768, -21671, 32765,	32766,	-29789, -32768, 32765,	32765,
	-29923, -9431,	987,	-10314, 18385,	-4214,	-32765, 32763,	-11852, -1841,	-3112,
	1426,	13069,	11990,	-6170,	-6677,	6674,	2690,	-5582,	-9188,	-12218, 1877,
	-3395,	18171,	-4465,	-32766, 8164,	30717,	-19068, -16579, -32387, -6776,	-26566,
	-9226,	6810,	4607,	31815,	19465,	1659,	-11120, -32766, -32766, 23815,	3019,
	-6127,	-32766, -26690, -31570, -28622, 28671,	16383,	20051,	-32768, 24700,	20665,
	-32766, -29726, -32768, -25118, -6071,	-4739,	-32768, 32765,	32766,	-32766, -28673,
	-32766, -3546,	32766,	18090,	-32768, 13279,	-32768, 32767,	1643,	-687,	6802,
	-32766, 32766,	4201,	-23052, -6335,	15094,	20748,	-16062, -28470, -9685,	-32766,
	-126,	1987,	5699,	3247,	-4416,	925,	6776,	12892,	26932,	-15671, -16385,
	-32768, -8049,	-5264,	-1184,	32760,	-13149, -32762, -1348,	-30019, -32766, -32768,
	3499,	31276,	-32766, -2124,	-8930,	4632,	4632,	-16629, 32767,	-32768, 32766,
	-32768, 32765,	192,	4172,	32766,	-14776, -19654, 376,	-2814,	7584,	2278,
	7962,	-32768, -32766, -6092,	-27148, -16385, 32766,	-32237, 32766,	-32768, 25204,
	-32768, 29816,	-21720, 32765,	-32768, 22244};

static const uint16_t MODEL_NEURONS_LINKS[] = {
	3,  4,	9,  14, 28, 29, 48, 52, 62, 67, 71, 84, 85, 32, 34, 40, 50, 57, 67, 76, 85, 3,	4,
	5,  19, 23, 29, 31, 33, 51, 72, 73, 75, 79, 81, 85, 1,	2,  1,	3,  6,	8,  13, 16, 17, 18,
	26, 32, 38, 41, 48, 49, 52, 66, 69, 72, 73, 74, 75, 76, 79, 81, 83, 84, 85, 12, 18, 29, 32,
	39, 40, 42, 54, 59, 60, 66, 77, 78, 79, 85, 0,	5,  23, 42, 71, 79, 85, 11, 27, 31, 32, 35,
	45, 54, 61, 65, 68, 83, 85, 4,	6,  11, 16, 32, 35, 45, 54, 56, 58, 60, 70, 85, 1,  3,	15,
	23, 29, 33, 45, 67, 85, 0,  5,	14, 18, 27, 52, 85, 2,	3,  7,	15, 16, 23, 43, 44, 72, 74,
	85, 2,	3,  41, 69, 73, 85, 3,	11, 85, 4,  4,	8,  23, 59, 60, 85, 4,	13, 85, 5,  8,	9,
	9,  20, 23, 29, 31, 34, 57, 80, 85, 6,	2,  7,	31, 45, 59, 62, 65, 85, 1,  6,	7,  13, 16,
	11, 35, 41, 43, 45, 47, 54, 70, 74, 85, 7,  17, 85, 5,	9,  15, 0,  10, 14, 17, 23, 25, 29,
	31, 34, 35, 53, 55, 58, 66, 82, 85, 4,	17, 50, 55, 85, 12, 16, 20, 23, 25, 55, 85, 2,	7,
	11, 4,	18, 71, 77, 85, 2,  10, 22, 85, 1,  8,	15, 16, 2,  19, 20, 23, 31, 37, 45, 50, 52,
	56, 67, 71, 78, 84, 85, 16, 22, 28, 45, 50, 71, 85, 8,	19, 20, 22, 28, 31, 34, 50, 52, 57,
	71, 85, 1,  8,	9,  19, 0,  3,	15, 16, 20, 22, 29, 66, 71, 85, 1,  7,	8,  15, 2,  8,	15,
	16, 17, 19, 23, 29, 30, 37, 48, 50, 51, 61, 64, 85, 7,	28, 0,	7,  9,	17, 34, 36, 41, 42,
	46, 54, 55, 75, 85, 13, 28, 3,	4,  9,	17, 38, 40, 41, 54, 68, 85, 15, 19, 20, 23, 25, 55,
	85, 5,	15, 19, 20, 21, 31, 85, 2,  8,	13, 15, 19, 24, 27, 28, 7,  12, 16, 17, 24, 28, 37,
	38, 41, 55, 65, 67, 75, 81, 85, 5,  8,	19, 28, 33, 7,	21, 23, 40, 41, 61, 78, 79, 85, 0,
	19, 27, 14, 15, 20, 27, 29, 52, 57, 67, 76, 80, 85, 0,	1,  19, 35, 9,	14, 16, 21, 28, 29,
	46, 52, 66, 67, 71, 77, 79, 85, 0,  9,	35, 36, 85, 6,	7,  16, 24, 33, 54, 63, 68, 74, 85,
	6,  7,	38, 13, 31, 33, 54, 63, 85, 4,	6,  7,	16, 25, 39, 8,	11, 31, 44, 45, 65, 85, 19,
	21, 24, 25, 39, 40, 21, 44, 78, 85, 6,	16, 38, 39, 40, 41, 85, 5,  27, 28, 3,	11, 28, 34,
	35, 42, 52, 61, 85, 1,	8,  24, 25, 26, 27, 28, 29, 30, 33, 34, 43, 85};

static const uint16_t MODEL_NEURON_INTERNAL_LINKS_NUM[] = {
	0,   13,  21,  38,  65,	 81,  87,  101, 113, 122, 131, 141, 147, 149, 157,
	161, 171, 184, 196, 200, 217, 221, 231, 239, 244, 260, 267, 282, 296, 314,
	329, 341, 352, 361, 381, 393, 408, 426, 431, 440, 452, 465, 475, 479, 500};

static const uint16_t MODEL_NEURON_EXTERNAL_LINKS_NUM[] = {
	13,  21,  36,  65,  80,	 87,  99,  112, 121, 128, 139, 145, 148, 155, 158,
	170, 179, 194, 197, 216, 221, 228, 236, 240, 259, 266, 278, 292, 312, 327,
	339, 346, 353, 376, 390, 404, 422, 427, 437, 446, 459, 469, 476, 488, 501};

static const nrf_user_coeff_t MODEL_NEURON_ACTIVATION_WEIGHTS[] = {
	0,     0,     0,     0,	    0,	   0,	 0,	0,     6144,  64512, 0,	    0,
	40960, 6144,  37510, 64512, 62464, 2048, 39426, 64512, 64512, 64512, 64512, 40959,
	6144,  6144,  6144,  6144,  6144,  6144, 6144,	64512, 40959, 6144,  64512, 64512,
	64512, 40959, 0,     0,	    0,	   0,	 40960, 64512, 40959};

static const uint8_t MODEL_NEURON_ACTIVATION_TYPE_MASK[] = {0xff, 0xaf, 0x7b, 0xff, 0xde, 0xb};

static const uint16_t MODEL_OUTPUT_NEURONS_INDICES[] = {37, 44, 23, 12, 14, 32, 42, 18};

#define NN_DECODED_OUTPUT_INIT                                                                     \
	.classif = {                                                                               \
		.predicted_class = 0,                                                              \
		.num_classes = MODEL_OUTPUTS_NUM,                                                  \
	}

/** Model neurons activations buffer */
static nrf_user_neuron_t model_neurons_[MODEL_NEURONS_NUM];

/** Neuton model instance */
static const nrf_edgeai_model_neuton_t model_instance_ = {
	///
	.meta.p_neuron_internal_links_num = MODEL_NEURON_INTERNAL_LINKS_NUM,
	.meta.p_neuron_external_links_num = MODEL_NEURON_EXTERNAL_LINKS_NUM,
	.meta.p_output_neurons_indices = MODEL_OUTPUT_NEURONS_INDICES,
	.meta.p_neuron_links = MODEL_NEURONS_LINKS,
	.meta.p_neuron_act_type_mask = MODEL_NEURON_ACTIVATION_TYPE_MASK,
	.meta.outputs_num = MODEL_OUTPUTS_NUM,
	.meta.neurons_num = MODEL_NEURONS_NUM,
	.meta.weights_num = MODEL_WEIGHTS_NUM,
	///
	.params.MODEL_PARAMS_TYPE =
		{
			.p_weights = MODEL_WEIGHTS,
			.p_act_weights = MODEL_NEURON_ACTIVATION_WEIGHTS,
			.p_neurons = model_neurons_,
		},
};

//////////////////////////////////////////////////////////////////////////////
#define NN_INPUT_INIT_INTERFACE	       nrf_edgeai_input_init_sliding_window
#define NN_INPUT_FEED_INTERFACE	       nrf_edgeai_input_feed_sliding_window_i16
#define NN_PROCESS_FEATURES_INTERFACE  nrf_edgeai_process_features_dsp_i16_q16
#define NN_INIT_INFERENCE_INTERFACE    nrf_edgeai_init_inference_neuton
#define NN_RUN_INFERENCE_INTERFACE     nrf_edgeai_run_inference_neuton_q16
#define NN_PROPAGATE_OUTPUTS_INTERFACE nrf_edgeai_output_dequantize_neuton_q16_f32
#define NN_DECODE_OUTPUTS_INTERFACE    nrf_edgeai_output_decode_classification_f32

//////////////////////////////////////////////////////////////////////////////

static nrf_user_output_t model_outputs_[MODEL_OUTPUTS_NUM];

//////////////////////////////////////////////////////////////////////////////

static nrf_edgeai_t nrf_edgeai_ = {
	///
	.metadata.p_solution_id = EDGEAI_LAB_SOLUTION_ID_STR,
	.metadata.version.combined = EDGEAI_RUNTIME_VERSION_COMBINED,
	///
	.input.p_used_for_lags_mask = INPUT_FEATURES_USED_FOR_LAGS_MASK,
	.input.p_usage_mask = INPUT_FEATURES_USAGE_MASK,
	.input.type = INPUT_FEATURE_DATA_TYPE,
	.input.unique_num = INPUT_UNIQ_FEATURES_NUM,
	.input.unique_num_used = INPUT_UNIQ_FEATURES_USED_NUM,
	.input.unique_scales_num = INPUT_UNIQUE_SCALES_NUM,
	.input.window_size = INPUT_WINDOW_SIZE,
	.input.window_shift = INPUT_WINDOW_SHIFT,
	.input.subwindow_num = INPUT_SUBWINDOW_NUM,
	.input.window_memory.p_void = INPUT_WINDOW_MEMORY,
	.input.p_window_ctx = P_INPUT_WINDOW_CTX,

	.input.scale.INPUT_TYPE =
		{
			.p_min = INPUT_FEATURES_SCALE_MIN,
			.p_max = INPUT_FEATURES_SCALE_MAX,
		},
	///
	.p_dsp = P_DSP_PIPELINE,
	///
	.model.type = MODEL_TYPE,
	.model.task = MODEL_TASK,
	.model.instance.p_void = P_MODEL_INSTANCE,
	.model.output.memory.p_void = model_outputs_,
	.model.output.num = MODEL_OUTPUTS_NUM,
	.model.uses_as_input.all = MODEL_USES_AS_INPUT_MASK,
	///
	.interfaces.input_init = NN_INPUT_INIT_INTERFACE,
	.interfaces.feed_inputs = NN_INPUT_FEED_INTERFACE,
	.interfaces.process_features = NN_PROCESS_FEATURES_INTERFACE,
	.interfaces.init_inference = NN_INIT_INFERENCE_INTERFACE,
	.interfaces.run_inference = NN_RUN_INFERENCE_INTERFACE,
	.interfaces.propagate_outputs = NN_PROPAGATE_OUTPUTS_INTERFACE,
	.interfaces.decode_outputs = NN_DECODE_OUTPUTS_INTERFACE,
	///
	.decoded_output = {NN_DECODED_OUTPUT_INIT},
};

//////////////////////////////////////////////////////////////////////////////

nrf_edgeai_t *nrf_edgeai_user_model_91278(void)
{
	return &nrf_edgeai_;
}

//////////////////////////////////////////////////////////////////////////////

uint32_t nrf_edgeai_user_model_neuton_size_91278(void)
{
	uint32_t model_meta_size =
		(sizeof(MODEL_WEIGHTS) + sizeof(MODEL_NEURONS_LINKS) +
		 sizeof(MODEL_NEURON_EXTERNAL_LINKS_NUM) + sizeof(MODEL_NEURON_INTERNAL_LINKS_NUM) +
		 sizeof(MODEL_NEURON_ACTIVATION_WEIGHTS) +
		 sizeof(MODEL_NEURON_ACTIVATION_TYPE_MASK) + sizeof(MODEL_OUTPUT_NEURONS_INDICES));

#if MODEL_TASK == __NRF_EDGEAI_TASK_ANOMALY_DETECTION
	model_meta_size += sizeof(MODEL_AVERAGE_EMBEDDING) + sizeof(MODEL_OUTPUT_SCALE_MIN) +
			   sizeof(MODEL_OUTPUT_SCALE_MAX);
#endif

#if MODEL_TASK == __NRF_EDGEAI_TASK_REGRESSION
	model_meta_size += sizeof(MODEL_OUTPUT_SCALE_MIN) + sizeof(MODEL_OUTPUT_SCALE_MAX);
#endif

	return model_meta_size;
}
