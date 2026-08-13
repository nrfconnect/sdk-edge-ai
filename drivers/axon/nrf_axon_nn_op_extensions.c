/*
* Copyright (c) 2025-26 Nordic Semiconductor ASA
*
* SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
*/

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "axon/nrf_axon_platform.h"
#include "drivers/axon/nrf_axon_nn_op_extensions.h"
#include "drivers/axon/nrf_axon_nn_infer.h"
#include "drivers/axon/nrf_axon_dsp_intrinsics.h"

static inline void axon_saturate_i8(float value, int8_t *output_ptr, uint32_t output_offset)
{
	*(output_ptr+output_offset) = (int8_t)(value > (float)INT8_MAX ?
		INT8_MAX : value < (float)INT8_MIN ? INT8_MIN : (int8_t)value);
}
static inline void axon_saturate_i16(float value, int16_t *output_ptr, uint32_t output_offset)
{
	*(output_ptr+output_offset) = (int16_t)(value > (float)INT16_MAX ?
		INT16_MAX : value < (float)INT16_MIN ? INT16_MIN : (int16_t)value);
}
static inline void axon_saturate_i32(float value, int32_t *output_ptr, uint32_t output_offset)
{
	*(output_ptr+output_offset) = (int32_t)((double)value > (double)INT32_MAX ?
		INT32_MAX : (double)value < (double)INT32_MIN ? INT32_MIN : (int32_t)value);
}

/**
 * @brief
 * Implements neural net operator softmax as a software operation that can be
 * embedded in an axon command buffer.
 * @param argc number of arguments in argv. Must be 2
 * @param args down cast to a *nrf_axon_nn_op_extension_base1_args_s
 */
nrf_axon_result_e nrf_axon_nn_op_extension_softmax(uint16_t argc,
	NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args)
{
#define MAX_EXP_INPUT (31182) /*the maximum input to exp before it saturates */
	if (((argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)) <
		sizeof(nrf_axon_nn_op_extension_base1_args_s)) || (args == NULL)) {
		return NRF_AXON_RESULT_FAILURE;
	}
	nrf_axon_nn_op_extension_base1_args_s *base1_args =
		(nrf_axon_nn_op_extension_base1_args_s *)args;
	/*
	 * softmax is calculated independently across all channels of each output surface location.
	 * Input is 32bit so it is effectively packed.
	 */
	uint16_t area = base1_args->remaining_args.height * base1_args->remaining_args.width;
	int32_t *input_ptr = (int32_t *)base1_args->ptr_args.input;

	/* 1st step is to normalize input of each surface element to the maximum channel value */
	for (uint16_t area_ndx = 0; area_ndx < area; area_ndx++, input_ptr++) {
		int32_t max_value = *input_ptr; /* channel 0 starts off as the max */
		int32_t *channel_ptr = input_ptr + area;

		for (uint16_t channel_ndx = 1;
			channel_ndx < base1_args->remaining_args.channel_cnt;
			channel_ndx++, channel_ptr += area) {
			if (*channel_ptr > max_value) {
				max_value = *channel_ptr;
			}
		}
		/* found the max value for this surface ndx, now normalize.
		 * The Q11.12 value saturates at 2048. ln(2048) = 7.61283103,
		 * converted to q11.12 is 31182.
		 * Set the maximum value to this, and adjust all the other values accordingly.
		 */
		int32_t offset = max_value - MAX_EXP_INPUT;

		channel_ptr = input_ptr;
		for (uint16_t channel_ndx = 0;
			channel_ndx < base1_args->remaining_args.channel_cnt;
			channel_ndx++, channel_ptr += area) {
			*channel_ptr -= offset;
		}
	}

	/* run exp() on everything */
	input_ptr = (int32_t *)base1_args->ptr_args.input; /* reset input ptr to beginning */
	uint16_t total_elements = area * base1_args->remaining_args.channel_cnt;

	for (uint16_t done_so_far = 0;
		done_so_far < total_elements;
		done_so_far += 512, input_ptr += 512) {
		uint16_t done_this_time = total_elements-done_so_far;

		if (done_this_time > 512) {
			done_this_time = 512;
		} else	if (done_this_time < 4) { /* minimum length of exp op is 4. */
			memset(input_ptr + done_this_time, 0,
				(4 - done_this_time) * sizeof(*input_ptr));
			done_this_time = 4;
		} else if (done_this_time & 1) { /* exp op needs even length */
			*(input_ptr + done_this_time) = 0;
			done_this_time++;
		}
		/* MUST ALWAYS KEEP THE RESERVATION WHEN EXECUING AN INTRISIC
		 * WITHIN A OP EXTENSION!
		 */
		nrf_axon_exp_11p12(input_ptr, input_ptr, done_this_time,
			NRF_AXON_SYNC_MODE_BLOCKING_POLLING, true);
	}
	float scaling_multiplier = (float)base1_args->remaining_args.output_multiplier /
		(float)(1 << base1_args->remaining_args.output_rounding);
	/* go back through and normalize results to 1. */
	input_ptr = (int32_t *)base1_args->ptr_args.input; /* reset input ptr to beginning */
	for (uint16_t area_ndx = 0; area_ndx < area; area_ndx++, input_ptr++) {
		uint64_t sum = 0;
		int32_t *channel_ptr = input_ptr;
		/* 1st sum the results */
		for (uint16_t channel_ndx = 0;
			channel_ndx < base1_args->remaining_args.channel_cnt;
			channel_ndx++, channel_ptr += area) {
			sum += *channel_ptr;
		}

		if (sum == 0) {
			/*
			 * Ideally we should not have a sum value of zero as we are offsetting
			 * the input to the exponents in the first software step, in the off-case
			 * we do get all zeros, the actual values are going to be very small
			 * floating values which when quantized are equal to -128. setting sum=1
			 * prevents undefined results w/o affecting the final results
			 */
			sum = 1;
		}

		/* now go back and divide by the sum and quantize */
		channel_ptr = input_ptr;
		for (uint16_t channel_ndx = 0;
			channel_ndx < base1_args->remaining_args.channel_cnt;
			channel_ndx++, channel_ptr += area) {
			/*
			 * @FIXME!! SEE ABOUT SAVING THE *channel_ptr/sum term to input,
			 * then using axon to quantize saturate.
			 * Also, the below doesn't handle negative saturation.
			 */
			float temp = (*channel_ptr) * scaling_multiplier;

			temp /= sum;
			temp = roundf(temp);
			temp += base1_args->remaining_args.output_zeropoint;
			switch (base1_args->remaining_args.output_bytewidth) {
			case 1:
				axon_saturate_i8(temp, (int8_t *)base1_args->ptr_args.output,
				area_ndx + (area * channel_ndx));
				break;
			case 2:
				axon_saturate_i16(temp, (int16_t *)base1_args->ptr_args.output,
				area_ndx + (area * channel_ndx));
				break;
			case 4:
				axon_saturate_i32(temp, (int32_t *)base1_args->ptr_args.output,
				area_ndx + (area * channel_ndx));
				break;
			}
		}
	}
	return NRF_AXON_RESULT_SUCCESS;
}

#define SIGMOID_TANH_OPTIMIZED 1

#if SIGMOID_TANH_OPTIMIZED
/**
 * LUT for sigmoid for inputs 0 to 8 stored as q3.12.
 * 256 values spaced by 128.
 * - shift input right by 7 to get table index.
 * - inputs > 8 (q3.12 8 << 12) are saturated to 1.
 * - inputs < 0 are 1 - the positive value.
 * For LUT index less than 150, interpolation is done by:
 * - Bitwise AND the input with 0x7f
 * - multiply the result with sigmoid_interp_multiplier[lut_ndx]
 * - right shift by 7.
 * - Add to the lut entry.
 */
static const uint16_t sigmoid_lut[256] = {8192,8320,8448,8576,8703,8831,8958,9084,9211,9336,9462,9586,9710,9833,9956,10078,10198,10318,10437,10555,10672,10788,10902,11015,11128,11239,11348,11457,11564,11669,11773,11876,11978,12078,12176,12273,12369,12463,12555,12646,12735,12823,12909,12994,13077,13159,13239,13318,13395,13471,13545,13617,13689,13758,13826,13893,13958,14022,14085,14146,14206,14264,14321,14377,14431,14484,14536,14587,14636,14684,14731,14777,14822,14865,14908,14949,14990,15029,15067,15105,15141,15177,15211,15245,15277,15309,15340,15370,15400,15428,15456,15483,15509,15535,15559,15584,15607,15630,15652,15673,15694,15715,15735,15754,15772,15791,15808,15825,15842,15858,15874,15889,15904,15918,15932,15946,15959,15971,15984,15996,16008,16019,16030,16041,16051,16061,16071,16080,16089,16098,16107,16115,16123,16131,16139,16146,16154,16161,16167,16174,16180,16187,16193,16198,16204,16209,16215,16220,16225,16230,16234,16239,16243,16248,16252,16256,16260,16264,16267,16271,16274,16278,16281,16284,16287,16290,16293,16296,16298,16301,16304,16306,16308,16311,16313,16315,16317,16319,16321,16323,16325,16327,16329,16330,16332,16334,16335,16337,16338,16340,16341,16342,16343,16345,16346,16347,16348,16349,16350,16351,16352,16353,16354,16355,16356,16357,16358,16359,16359,16360,16361,16362,16362,16363,16364,16364,16365,16365,16366,16367,16367,16368,16368,16369,16369,16370,16370,16370,16371,16371,16372,16372,16372,16373,16373,16373,16374,16374,16374,16375,16375,16375,16375,16376,16376,16376,16376,16377,16377,16377,16377,16378,16378,16378,16378,16378,};
static const uint8_t sigmoid_interp_multiplier[] = {128,128,128,127,128,127,126,127,125,126,124,124,123,123,122,120,120,119,118,117,116,114,113,113,111,109,109,107,105,104,103,102,100,98,97,96,94,92,91,89,88,86,85,83,82,80,79,77,76,74,72,72,69,68,67,65,64,63,61,60,58,57,56,54,53,52,51,49,48,47,46,45,43,43,41,41,39,38,38,36,36,34,34,32,32,31,30,30,28,28,27,26,26,24,25,23,23,22,21,21,21,20,19,18,19,17,17,17,16,16,15,15,14,14,14,13,12,13,12,12,11,11,11,10,10,10,9,9,9,9,8,8,8,8,7,8,7,6,7,6,7,6,5,6,5,6,5,5,5,4,5,4,5,4,4,4,4,3,4,3,4,3,3,3,3,3,3,2,3,3,2,2,3,2,2,2,2,2,2,2,2,2,1,2,2,1,2,1,2,1,1,1,2,};
/* entries in the interpolation multiply table are left shifted this ammount */
#define SIGMOID_INTERP_SHIFT 7

/* input values are shifted right this much to create the lut table index */
#define SIGMOID_LUT_SHIFT (NRF_AXON_SIG_TANH_INPUT16_RADIX - 5)
/* any input value greater than this is saturated to 0 or 1 */
#define SIGMOID_LUT_MAX_INPUT (8 << NRF_AXON_SIG_TANH_INPUT16_RADIX)
/* lut table indexes greater than this do not do interpolation because the difference is < 1 */
#define SIGMOID_INTER_MAX_NDX sizeof(sigmoid_interp_multiplier)
/* input values are and'ed with this to create the interpolation value to multiply */
#define SIGMOID_INTERP_MASK ((1 << SIGMOID_LUT_SHIFT) - 1)

/**
 *
 */
static inline int16_t sigmoid_optimized(int16_t in_data)
{
	int32_t in_data_abs =  in_data < 0 ? -in_data : in_data;
	int32_t out_data;
	if (in_data_abs >= SIGMOID_LUT_MAX_INPUT) {
		out_data = sigmoid_lut[255];
	} else {
		uint8_t lut_ndx = in_data_abs >> SIGMOID_LUT_SHIFT;
		out_data = sigmoid_lut[lut_ndx];
		if (lut_ndx < SIGMOID_INTER_MAX_NDX) {
			// interpolate intermediate points
			out_data += ((in_data_abs & SIGMOID_INTERP_MASK) * sigmoid_interp_multiplier[lut_ndx]) >> SIGMOID_INTERP_SHIFT;
		}
	}
	if (in_data < 0) {
		out_data = (1 << NRF_AXON_SIG_TANH_OUTPUT16_RADIX) - out_data;
	}
	return (int16_t)out_data;
}
#else
static inline float sigmoid_float(int16_t in_data)
{
	float scratch = in_data;
	/*
		* sigmoid(x) = 1/(1+exp(-x))
		*/
	scratch /= (float)(1 << NRF_AXON_SIG_TANH_INPUT16_RADIX); /* input is q.12, convert to float */
	scratch = (float)exp(-scratch); /* now have exp(x) */
	scratch = 1/(1+scratch); /* have float sigmoid(x) */
	return scratch;
}
#endif
/*
 * @brief
 * Implements neural net operator sigmoid as a software operation that can be
 * embedded in an axon command buffer. This is the base operation that can operate in legacy
 * mode (pre 1.1.0) where output is packed, and v2 mode where output is not masked.
 * @param argc number of arguments in argv. Must be 2
 * @param args downcast to a *nrf_axon_nn_op_extension_base1_args_s
 * @param packed_output if true, output will be written in packed format. if false, each row will start on a 32bit boundary.
 */
static nrf_axon_result_e nrf_axon_nn_op_extension_sigmoid_base(uint16_t argc,
	NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args, bool packed_output)
{
	if (((argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)) <
		sizeof(nrf_axon_nn_op_extension_base1_args_s)) || (args == NULL)) {
		return NRF_AXON_RESULT_FAILURE;
	}
	nrf_axon_nn_op_extension_base1_args_s *base1_args =
		(nrf_axon_nn_op_extension_base1_args_s *)args;
	/*
	 * Sigmoid is calculated 1 for 1 for all input. Input is int16, q3.12 format
	 * So iterate through channels/rows/columns
	 */
	/* unpacked input rows always start on a 32bit boundary. */
	uint8_t input_extra_stride =
		(!base1_args->remaining_args.input_is_packed &&
		base1_args->remaining_args.width & 1) ?
		1 : 0;
	uint8_t output_extra_stride =
		packed_output || (base1_args->remaining_args.output_bytewidth == 4) ?
		0 : (4 - (base1_args->remaining_args.width & 3)) & 3;

	int16_t *input_ptr = (int16_t *)base1_args->ptr_args.input;
	union {
		int8_t *i8;
		int16_t *i16;
		int32_t *i32;
		NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE value;
	} output_ptr;
	output_ptr.value =
		(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)base1_args->ptr_args.output;
	for (uint16_t channel_ndx = 0;
		channel_ndx < base1_args->remaining_args.channel_cnt;
		channel_ndx++) {
		for (uint16_t row_ndx = 0;
			row_ndx < base1_args->remaining_args.height;
			row_ndx++) {
			for (uint16_t col_ndx = 0;
				col_ndx < base1_args->remaining_args.width;
				col_ndx++, input_ptr++) {
				int16_t in_data = *input_ptr;
#if SIGMOID_TANH_OPTIMIZED
				int16_t out_data = sigmoid_optimized(in_data);
#if 0
				int16_t output_from_float = (int16_t)(sigmoid_float(in_data) * (1 << NRF_AXON_SIG_TANH_OUTPUT16_RADIX));
				int16_t error = abs(output_from_float - out_data);
				if (error > 2) {
					printf("Error! output_from_float (%d) != out_data (%d)\n", output_from_float, out_data);
				}
#endif

				switch (base1_args->remaining_args.output_bytewidth) {
				case 1: /* quantized output. scales between 0 and 1. */
					out_data = (out_data >> (NRF_AXON_SIG_TANH_OUTPUT16_RADIX - 8)) - 128;
					if (out_data > 127) {
						out_data = 127;
					}

					*output_ptr.i8 = (int8_t)out_data;
					output_ptr.i8++;
					break;
				case 2: /* q1.14 output */
					*output_ptr.i16 = out_data;
					output_ptr.i16++;
					break;
				case 4: /* q1.30 output */
					*output_ptr.i32 = out_data << (NRF_AXON_SIG_TANH_OUTPUT16_RADIX + 6);
					output_ptr.i32++;
					break;
				default:
					nrf_axon_platform_printf(
					"Axon NN: Invalid Sigmoid bytewidth %d\n",
					base1_args->remaining_args.output_bytewidth);
					return -1;
				}
#else
				float out_data = sigmoid_float(in_data);
				switch (base1_args->remaining_args.output_bytewidth) {
				case 1: /* quantized output. scales between 0 and 1. */
					out_data = (float)round(out_data * 256.0f) - 128;
					axon_saturate_i8(out_data, output_ptr.i8, 0);
					output_ptr.i8++;
					break;
				case 2: /* q1.14 output */
					*output_ptr.i16 = (int16_t)(out_data * (1 << NRF_AXON_SIG_TANH_OUTPUT16_RADIX));
					output_ptr.i16++;
					break;
				case 4: /* q1.30 output */
					*output_ptr.i32 = (int32_t)(out_data * (1<<30));
					output_ptr.i32++;
					break;
				default:
					nrf_axon_platform_printf(
					"Axon NN: Invalid Sigmoid bytewidth %d\n",
					base1_args->remaining_args.output_bytewidth);
					return -1;
				}
#endif
			}
			input_ptr += input_extra_stride;
			output_ptr.value += output_extra_stride;
		}
	}
	return NRF_AXON_RESULT_SUCCESS;
}

/**
 * sigmoid version used by compiler versions before 1.1.0.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_sigmoid(uint16_t argc,
	NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args)
{
	return nrf_axon_nn_op_extension_sigmoid_base(argc, args, false);
}
/**
 * sigmoid version used by compiler versions 1.1.0 and later
 */
nrf_axon_result_e nrf_axon_nn_op_extension_sigmoid_v2(uint16_t argc,
	NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args)
{
	return nrf_axon_nn_op_extension_sigmoid_base(argc, args, true);
}
/**
 * sigmoid version used when input dequantization is needed.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_sigmoid_dequantize_input(uint16_t argc,
	NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args)
{
	nrf_axon_nn_op_extension_sig_tanh_args_s *sig_tanh_args =
		(nrf_axon_nn_op_extension_sig_tanh_args_s *)args;
	/* 1st dequantize */
	if (sig_tanh_args->remaining_args.input_byte_width == 1) {
		/* dequantize 8bit input */
		/*
		 * MUST ALWAYS KEEP THE RESERVATION WHEN EXECUING AN INTRISIC
		 * WITHIN A OP EXTENSION!
		 */
		nrf_axon_axpb_2d_8_16(sig_tanh_args->ptr_args.base1.input,
			sig_tanh_args->remaining_args.input_dequant_multiplier,
			-sig_tanh_args->remaining_args.input_dequant_zero_pt *
				sig_tanh_args->remaining_args.input_dequant_multiplier,
			(int16_t *)sig_tanh_args->ptr_args.scratch,
			sig_tanh_args->remaining_args.base1.height,
			sig_tanh_args->remaining_args.base1.width,
			sig_tanh_args->remaining_args.base1.input_is_packed,
			sig_tanh_args->remaining_args.input_dequant_rounding_bits,
			NRF_AXON_SYNC_MODE_BLOCKING_POLLING, true);
	} else {
		/* dequantize 16bit input */
		nrf_axon_axpb_2d_16_16((int16_t *)sig_tanh_args->ptr_args.base1.input,
			sig_tanh_args->remaining_args.input_dequant_multiplier,
			-sig_tanh_args->remaining_args.input_dequant_zero_pt *
				sig_tanh_args->remaining_args.input_dequant_multiplier,
			(int16_t *)sig_tanh_args->ptr_args.scratch,
			sig_tanh_args->remaining_args.base1.height,
			sig_tanh_args->remaining_args.base1.width,
			sig_tanh_args->remaining_args.base1.input_is_packed,
			sig_tanh_args->remaining_args.input_dequant_rounding_bits,
			NRF_AXON_SYNC_MODE_BLOCKING_POLLING, true);
	}
	/*
	 * copy params to a nrf_axon_nn_op_extension_base1_args_s
	 */
	nrf_axon_nn_op_extension_base1_args_s base1_args;

	memcpy(&base1_args.ptr_args, &sig_tanh_args->ptr_args.base1,
		sizeof(sig_tanh_args->ptr_args.base1));
	/* scratch is the input */
	base1_args.ptr_args.input = sig_tanh_args->ptr_args.scratch;
	memcpy(&base1_args.remaining_args, &sig_tanh_args->remaining_args.base1,
		sizeof(sig_tanh_args->remaining_args.base1));
	return nrf_axon_nn_op_extension_sigmoid_base(
		sizeof(base1_args) / sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE),
		(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *)&base1_args, true);
}

#if SIGMOID_TANH_OPTIMIZED

/**
 * LUT for tanh for inputs 0 to 4 stored as q1.14.
 * 256 values spaced by 64.
 * - shift input right by 6 to get table index.
 * - inputs > 4 (q3.12 4 << 12) are saturated to 1.
 * - input < 0 are -1 * the positive value.
 * For LUT index less than ~180, interpolation is done by:
 * - Bitwise AND the input with 0x3f
 * - multiply the result with tanh_interp_multiplier[lut_ndx]
 * - right shift by 6.
 * - Add to the lut entry.
 */
static const uint16_t tanh_lut[256] = {0,256,512,767,1023,1277,1532,1785,2037,2289,2539,2789,3036,3283,3528,3771,4013,4252,4490,4726,4960,5191,5420,5647,5871,6093,6312,6529,6743,6954,7163,7369,7571,7771,7968,8162,8353,8541,8726,8908,9087,9262,9435,9604,9771,9934,10095,10252,10406,10557,10706,10851,10993,11132,11269,11402,11533,11661,11785,11908,12027,12144,12258,12369,12478,12584,12688,12789,12888,12984,13078,13170,13260,13347,13432,13515,13595,13674,13751,13826,13898,13969,14038,14105,14171,14234,14296,14356,14415,14472,14528,14582,14634,14685,14735,14783,14830,14876,14920,14963,15005,15046,15085,15124,15161,15197,15232,15267,15300,15332,15363,15394,15423,15452,15480,15507,15533,15559,15584,15608,15631,15654,15676,15697,15718,15738,15757,15776,15795,15812,15830,15847,15863,15879,15894,15909,15923,15937,15951,15964,15977,15989,16001,16013,16024,16035,16046,16056,16066,16076,16085,16094,16103,16112,16120,16128,16136,16143,16151,16158,16165,16171,16178,16184,16190,16196,16202,16208,16213,16218,16223,16228,16233,16238,16242,16246,16251,16255,16259,16263,16266,16270,16273,16277,16280,16283,16286,16289,16292,16295,16298,16300,16303,16305,16308,16310,16312,16315,16317,16319,16321,16323,16325,16327,16328,16330,16332,16333,16335,16336,16338,16339,16341,16342,16343,16344,16346,16347,16348,16349,16350,16351,16352,16353,16354,16355,16356,16357,16358,16358,16359,16360,16361,16361,16362,16363,16363,16364,16365,16365,16366,16366,16367,16368,16368,16369,16369,16369,16370,16370,16371,16371,16372,16372,16372,16373,};
static const uint8_t tanh_interp_multiplier[] = {128,128,127,128,127,127,126,126,126,125,125,123,123,122,121,121,119,119,118,117,115,114,113,112,111,109,108,107,105,104,103,101,100,98,97,95,94,92,91,89,87,86,84,83,81,80,78,77,75,74,72,71,69,68,66,65,64,62,61,59,58,57,55,54,53,52,50,49,48,47,46,45,43,42,41,40,39,38,37,36,35,34,33,33,31,31,30,29,28,28,27,26,25,25,24,23,23,22,21,21,20,19,19,18,18,17,17,16,16,15,15,14,14,14,13,13,13,12,12,11,11,11,10,10,10,9,9,9,8,9,8,8,8,7,7,7,7,7,6,6,6,6,6,5,5,5,5,5,5,4,4,4,4,4,4,4,3,4,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,1,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,1,0,1,};
#define TANH_INTERP_SHIFT 6
/* lut table indexes greater than this do not do interpolation because the difference is < 1 */
#define TANH_INTER_MAX_NDX sizeof(tanh_interp_multiplier)
/* any input value greater than this is saturated to +/-1 */
#define TANH_LUT_MAX_INPUT (4 << NRF_AXON_SIG_TANH_INPUT16_RADIX)
/* input 3.12 values are shifted right this much to create the lut table index */
#define TANH_LUT_SHIFT (NRF_AXON_SIG_TANH_INPUT16_RADIX - 6)
/* input 3.12 values are and'ed with this to create the interpolation value */
#define TANH_INTERP_MASK ((1 << TANH_LUT_SHIFT) - 1)

/**
 *
 */
static inline int16_t tanh_optimized(int16_t in_data)
{
	int32_t in_data_abs =  in_data < 0 ? -in_data : in_data;
	int32_t out_data;
	if (in_data_abs >= TANH_LUT_MAX_INPUT) {
		out_data = tanh_lut[255];
	} else {
		uint8_t lut_ndx = in_data_abs >> TANH_LUT_SHIFT;
		out_data = tanh_lut[lut_ndx];
		if (lut_ndx < TANH_INTER_MAX_NDX) {
			// interpolate intermediate points
			out_data += ((in_data_abs & TANH_INTERP_MASK) * tanh_interp_multiplier[lut_ndx]) >> TANH_INTERP_SHIFT;
		}
	}
	if (in_data < 0) {
		out_data *= -1;
	}
	return (int16_t)out_data;
}
#else
static inline float tanh_float(int16_t data)
{
	float out_data = data;
	/**
	 * tanh(x) = (exp(2x)-1)/(exp(2x)+1)
	 */
	/* input is q.12, multiply by 2 and convert to float */
	out_data /= (float)(1<<(NRF_AXON_SIG_TANH_INPUT16_RADIX - 1));
	out_data = expf(out_data); /* now have exp(2x) */
	out_data = (out_data - 1)/(out_data+1); /* have float tanh(x) */
	return out_data;
}
#endif
/**
 * @brief
 * Implements neural net operator tanh as a software operation that can be
 * embedded in an axon command buffer. This is the base operation that can operate in legacy mode (pre 1.1.0) where
 * output is packed, and v2 mode where output is not packed.
 * @param argc number of arguments in argv. Must be 2
 * @param args down cast to a *nrf_axon_nn_op_extension_base1_args_s
 * @param packed_output if true, output will be written in packed format. if false, each row will start on a 32bit boundary.
 */
static nrf_axon_result_e nrf_axon_nn_op_extension_tanh_base(uint16_t argc,
	NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args, bool packed_output)
{
	if (((argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)) <
		sizeof(nrf_axon_nn_op_extension_base1_args_s)) || (args == NULL)) {
		return NRF_AXON_RESULT_FAILURE;
	}
	nrf_axon_nn_op_extension_base1_args_s *base1_args =
		(nrf_axon_nn_op_extension_base1_args_s *)args;
	/*
	 * Sigmoid is calculated 1 for 1 for all input. Input is int16, q3.12 format
	 * So iterate through channels/rows/columns
	 */
	/* unpacked input rows always start on a 32bit boundary. */
	uint8_t input_extra_stride =
	(!base1_args->remaining_args.input_is_packed &&
		base1_args->remaining_args.width & 1) ? 1 : 0;
	uint8_t output_extra_stride = packed_output ||
		base1_args->remaining_args.output_bytewidth == 4 ?
		0 : (4 - (base1_args->remaining_args.width & 3)) & 3;

	int16_t *input_ptr = (int16_t *)base1_args->ptr_args.input;
	union {
		int8_t *i8;
		int16_t *i16;
		int32_t *i32;
		NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE value;
	} output_ptr;
	output_ptr.value = (NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
	base1_args->ptr_args.output;
	for (uint16_t channel_ndx = 0;
		channel_ndx < base1_args->remaining_args.channel_cnt;
		channel_ndx++) {
		for (uint16_t row_ndx = 0;
			row_ndx < base1_args->remaining_args.height;
			row_ndx++) {
			for (uint16_t col_ndx = 0;
				col_ndx < base1_args->remaining_args.width;
				col_ndx++, input_ptr++) {
#if SIGMOID_TANH_OPTIMIZED
				int16_t out_data = tanh_optimized(*input_ptr);
#if 0
				int16_t output_from_float = (int16_t)((tanh_float(*input_ptr)) * (1 << NRF_AXON_SIG_TANH_OUTPUT16_RADIX));
				int16_t error = abs(output_from_float - out_data);
				if (error > 2) {
					printf("Error! output_from_float (*%d) != out_data (%d)\n", output_from_float, out_data);
				}
#endif

				switch (base1_args->remaining_args.output_bytewidth) {
				case 1: /* quantized output. scales between -1 and 1. */
					out_data = (out_data >> (NRF_AXON_SIG_TANH_OUTPUT16_RADIX - 7));
					if (out_data > 127) {
						out_data = 127;
					}
					*output_ptr.i8 = (int8_t)out_data;
					output_ptr.i8++;
					break;
				case 2: /* q1.14 output */
					*output_ptr.i16 = out_data;
					output_ptr.i16++;
					break;
				case 4: /* q1.30 output */
					*output_ptr.i32 = out_data << (NRF_AXON_SIG_TANH_OUTPUT16_RADIX + 6);
					output_ptr.i32++;
					break;
				default:
					nrf_axon_platform_printf(
					"Axon NN: Invalid Sigmoid bytewidth %d\n",
					base1_args->remaining_args.output_bytewidth);
					return -1;
				}
#else
				float out_data = tanh_float(*input_ptr);
				switch (base1_args->remaining_args.output_bytewidth) {
				case 1: /* quantized output. scales between -1 and 1. */
					out_data = roundf(out_data * 128.0f); /* quantized */
					axon_saturate_i8(out_data, output_ptr.i8, 0);
					output_ptr.i8++;
					break;
				case 2: /* q1.14 output */
					*output_ptr.i16 = (int16_t)(out_data * (1 << NRF_AXON_SIG_TANH_OUTPUT16_RADIX));
					output_ptr.i16++;
					break;
				case 4: /* q1.30 output */
					*output_ptr.i32 = (int32_t)(out_data * (1<<30));
					output_ptr.i32++;
					break;
				default:
					nrf_axon_platform_printf(
					"Axon NN: Invalid Tanh bytewidth %d\n",
						base1_args->remaining_args.output_bytewidth);
					return -1;
				}
#endif
			}
			input_ptr += input_extra_stride;
			output_ptr.value += output_extra_stride;
		}
	}
	return NRF_AXON_RESULT_SUCCESS;
}
/**
 * tanh version used by compiler versions before 1.1.0.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_tanh(uint16_t argc,
	NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args)
{
	return nrf_axon_nn_op_extension_tanh_base(argc, args, false);
}
/**
 * tanh version used by compiler versions 1.1.0 and later
 */
nrf_axon_result_e nrf_axon_nn_op_extension_tanh_v2(uint16_t argc,
	NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args)
{
	return nrf_axon_nn_op_extension_tanh_base(argc, args, true);
}
/**
 * sigmoid version used when input dequantization is needed.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_tanh_dequantize_input(uint16_t argc,
	NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args)
{
	nrf_axon_nn_op_extension_sig_tanh_args_s *sig_tanh_args =
		(nrf_axon_nn_op_extension_sig_tanh_args_s *)args;
	/* 1st dequantize */
	if (sig_tanh_args->remaining_args.input_byte_width == 1) {
		/* dequantize 8bit input */
		/*
		 * MUST ALWAYS KEEP THE RESERVATION WHEN EXECUING AN INTRISIC
		 * WITHIN A OP EXTENSION!
		 */
		nrf_axon_axpb_2d_8_16(sig_tanh_args->ptr_args.base1.input,
			sig_tanh_args->remaining_args.input_dequant_multiplier,
			-sig_tanh_args->remaining_args.input_dequant_zero_pt *
				sig_tanh_args->remaining_args.input_dequant_multiplier,
			(int16_t *)sig_tanh_args->ptr_args.scratch,
			sig_tanh_args->remaining_args.base1.height,
			sig_tanh_args->remaining_args.base1.width,
			sig_tanh_args->remaining_args.base1.input_is_packed,
			sig_tanh_args->remaining_args.input_dequant_rounding_bits,
			NRF_AXON_SYNC_MODE_BLOCKING_POLLING, true);
	} else {
		/* dequantize 16bit input */
		nrf_axon_axpb_2d_16_16((int16_t *)sig_tanh_args->ptr_args.base1.input,
			sig_tanh_args->remaining_args.input_dequant_multiplier,
			-sig_tanh_args->remaining_args.input_dequant_zero_pt *
				sig_tanh_args->remaining_args.input_dequant_multiplier,
			(int16_t *)sig_tanh_args->ptr_args.scratch,
			sig_tanh_args->remaining_args.base1.height,
			sig_tanh_args->remaining_args.base1.width,
			sig_tanh_args->remaining_args.base1.input_is_packed,
			sig_tanh_args->remaining_args.input_dequant_rounding_bits,
			NRF_AXON_SYNC_MODE_BLOCKING_POLLING, true);
	}
	/*
	 * copy params to a nrf_axon_nn_op_extension_base1_args_s
	 */
	nrf_axon_nn_op_extension_base1_args_s base1_args;

	memcpy(&base1_args.ptr_args, &sig_tanh_args->ptr_args.base1,
		sizeof(sig_tanh_args->ptr_args.base1));
	/* scratch is the input */
	base1_args.ptr_args.input = sig_tanh_args->ptr_args.scratch;
	memcpy(&base1_args.remaining_args, &sig_tanh_args->remaining_args.base1,
		sizeof(sig_tanh_args->remaining_args.base1));
	return nrf_axon_nn_op_extension_tanh_base(
		sizeof(base1_args) / sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE),
		(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *)&base1_args, true);
}


nrf_axon_result_e nrf_axon_nn_op_extension_reshape(uint16_t argc,
	NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args)
{
	if (((argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)) <
		sizeof(nrf_axon_nn_op_extension_base2_args_s)) || (args == NULL)) {
		return NRF_AXON_RESULT_FAILURE;
	}
	nrf_axon_nn_op_extension_base2_args_s *base2_args =
		(nrf_axon_nn_op_extension_base2_args_s *)args;

	int input_stride = base2_args->remaining_args.input_stride;
	int output_stride = base2_args->remaining_args.output_width;

	for (uint16_t chan_ndx = 0;
		chan_ndx < base2_args->remaining_args.output_channel_cnt;
		chan_ndx++) { /*channel */

		for (uint16_t row_ndx = 0;
			row_ndx < base2_args->remaining_args.output_height;
			row_ndx++) { /* height */

			for (uint16_t col_ndx = 0;
				col_ndx < base2_args->remaining_args.output_width;
				col_ndx++) { /*width */

				int xtf = row_ndx * (base2_args->remaining_args.output_width *
					base2_args->remaining_args.output_channel_cnt) +
					(col_ndx * base2_args->remaining_args.output_channel_cnt) +
					chan_ndx;

				int h_prime = xtf / (base2_args->remaining_args.input_width *
					base2_args->remaining_args.input_channel_cnt);
				int rem =  xtf - (h_prime *
					(base2_args->remaining_args.input_width *
					base2_args->remaining_args.input_channel_cnt));
				int w_prime = rem / base2_args->remaining_args.input_channel_cnt;
				int c_prime = rem -
					(w_prime * base2_args->remaining_args.input_channel_cnt);

				int old_idx = c_prime *
					(base2_args->remaining_args.input_height * input_stride) +
					(h_prime * input_stride) + w_prime;

				int new_idx = chan_ndx *
					(base2_args->remaining_args.output_height * output_stride) +
					(row_ndx * output_stride) + col_ndx;

				((int8_t *)base2_args->ptr_args.output)[new_idx] =
					((int8_t *)base2_args->ptr_args.input)[old_idx];
			}
		}
	}
	return NRF_AXON_RESULT_SUCCESS;
}

nrf_axon_result_e nrf_axon_nn_op_extension_reshape_from_axon(uint16_t argc,
	NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args)
{
	if (((argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)) <
		sizeof(nrf_axon_nn_op_extension_base2_args_s)) || (args == NULL)) {
		return NRF_AXON_RESULT_FAILURE;
	}
	nrf_axon_nn_op_extension_base2_args_s *base2_args =
		(nrf_axon_nn_op_extension_base2_args_s *)args;
	unsigned int ax_width_tf_ch = base2_args->remaining_args.input_width;
	/* iterate through axon space, mapping to tf space */
	unsigned int ax_stride = base2_args->remaining_args.input_stride;
	unsigned int height = base2_args->remaining_args.input_height;
	unsigned int ax_ch_tf_width = base2_args->remaining_args.input_channel_cnt;
	unsigned int ax_extra_stride = ax_stride - ax_width_tf_ch;

	unsigned int axon_offset = 0;
	unsigned int tf_offset;

	for (uint16_t chan_ndx = 0;
		chan_ndx < ax_ch_tf_width;
		chan_ndx++) { /*channel */
		tf_offset = chan_ndx;
		for (uint16_t row_ndx = 0;
			row_ndx < height;
			row_ndx++) { /* height */

			for (uint16_t col_ndx = 0;
				col_ndx < ax_width_tf_ch;
				col_ndx++) { /*width */

				((int8_t *)base2_args->ptr_args.output)[tf_offset] =
					((int8_t *)base2_args->ptr_args.input)[axon_offset];
				tf_offset += ax_ch_tf_width;
				axon_offset++;
			}
			axon_offset += ax_extra_stride;
		}
	}
	return NRF_AXON_RESULT_SUCCESS;
}

static inline int32_t get_nearest_neighbor(
	const int input_value,
	const int32_t input_size,
	const int32_t output_size,
	const bool align_corners,
	const bool half_pixel_centers)
{
#define MY_MIN(a, b) ((a) > (b) ? (b) : (a))
#define MY_MAX(a, b) ((a) > (b) ? (a) : (b))
	const float scale =
		(align_corners && (output_size > 1)) ?
		(input_size - 1) / (float)(output_size - 1) :
		input_size / (float)(output_size);
	const float offset = half_pixel_centers ? 0.5f : 0.0f;
	int32_t output_value = MY_MIN(
		input_size - 1,
		align_corners ?
		(int32_t)(roundf((input_value + offset) * scale)) :
		(int32_t)((input_value + offset) * scale));
	if (half_pixel_centers) {
		output_value = MY_MAX(0, output_value);
	}
	return output_value;
}


nrf_axon_result_e nrf_axon_nn_op_extension_resize_nearest_neighbor(
	uint16_t argc,
	NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args)
{
	if (((argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)) <
		sizeof(nrf_axon_nn_op_extension_base2_args_s)) || (args == NULL)) {
		return NRF_AXON_RESULT_FAILURE;
	}
	nrf_axon_nn_op_extension_resize_nearest_neighbor_args_s *resize_nearest_neighbor_args =
		(nrf_axon_nn_op_extension_resize_nearest_neighbor_args_s *)args;

	int input_stride = resize_nearest_neighbor_args->remaining_args.input_stride;
	int output_stride = resize_nearest_neighbor_args->remaining_args.output_width;
	int input_z_stride =
		input_stride * resize_nearest_neighbor_args->remaining_args.input_height;
	int output_z_stride =
		output_stride * resize_nearest_neighbor_args->remaining_args.output_height;
	/* iterate across the output surface. */
	for (uint16_t row_ndx = 0;
		row_ndx < resize_nearest_neighbor_args->remaining_args.output_height;
		row_ndx++) { /* height */
		int32_t in_row_ndx = get_nearest_neighbor(row_ndx,
			resize_nearest_neighbor_args->remaining_args.input_height,
			resize_nearest_neighbor_args->remaining_args.output_height,
			resize_nearest_neighbor_args->remaining_args.align_corners,
			resize_nearest_neighbor_args->remaining_args.half_pixel_centers);

		for (uint16_t col_ndx = 0;
			col_ndx < resize_nearest_neighbor_args->remaining_args.output_width;
			col_ndx++) { /* width */
			int32_t in_col_ndx = get_nearest_neighbor(col_ndx,
				resize_nearest_neighbor_args->remaining_args.input_width,
				resize_nearest_neighbor_args->remaining_args.output_width,
				resize_nearest_neighbor_args->remaining_args.align_corners,
				resize_nearest_neighbor_args->remaining_args.half_pixel_centers);
			/* now propagate the input to the output across all channels. */
			int32_t input_offset = (in_row_ndx *
				resize_nearest_neighbor_args->remaining_args.input_stride) +
				in_col_ndx;
			int32_t output_offset = row_ndx * output_stride + col_ndx;

			for (uint16_t chan_ndx = 0;
				chan_ndx <
				resize_nearest_neighbor_args->remaining_args.output_channel_cnt;
				chan_ndx++) { /* channel*/
				resize_nearest_neighbor_args->ptr_args.output[output_offset] =
					resize_nearest_neighbor_args->ptr_args.input[input_offset];
				input_offset += input_z_stride;
				output_offset += output_z_stride;
			}
		}
	}
	return NRF_AXON_RESULT_SUCCESS;
}
