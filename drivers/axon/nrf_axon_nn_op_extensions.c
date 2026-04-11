/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "axon/nrf_axon_platform.h"
#include "drivers/axon/nrf_axon_nn_op_extensions.h"
#include "drivers/axon/nrf_axon_nn_infer.h"
#include "drivers/axon/nrf_axon_dsp_intrinsics.h"

static inline void axon_saturate_i8(float value, int8_t *output_ptr, unsigned output_offset)
{
	*(output_ptr+output_offset) = (int8_t)(value > INT8_MAX ?
	    INT8_MAX : value < INT8_MIN ? INT8_MIN : value);
}
static inline void axon_saturate_i16(float value, int16_t *output_ptr, unsigned output_offset)
{
	*(output_ptr+output_offset) = (int16_t)(value > INT16_MAX ?
	    INT16_MAX : value < INT16_MIN ? INT16_MIN : value);
}
static inline void axon_saturate_i32(float value, int32_t *output_ptr, unsigned output_offset)
{
	*(output_ptr+output_offset) = (int32_t)(value > INT32_MAX ?
	    INT32_MAX : value < INT32_MIN ? INT32_MIN : value);
}

/**
 * @brief
 * Implements neural net operator softmax as a software operation that can be
 * embedded in an axon command buffer.
 * @param argc number of arguments in argv. Must be 2
 * @param args down cast to a *nrf_axon_nn_op_extension_base1_args_s
 */
nrf_axon_result_e nrf_axon_nn_op_extension_softmax(uint16_t argc,
                      NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args)
{
#define MAX_EXP_INPUT (31182) //the maximum input to exp before it saturates
	if (((argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)) <
	     sizeof(nrf_axon_nn_op_extension_base1_args_s)) || (args==NULL)) {
		return NRF_AXON_RESULT_FAILURE;
	}
	nrf_axon_nn_op_extension_base1_args_s *base1_args =
	    (nrf_axon_nn_op_extension_base1_args_s*)args;
	/**
	 * softmax is calculated independently across all channels of each output surface location.
	 * Input is 32bit so it is effectively packed.
	 */
	uint16_t area = base1_args->remaining_args.height * base1_args->remaining_args.width;
	int32_t *input_ptr = (int32_t*)base1_args->ptr_args.input;

	// 1st step is to normalize input of each surface element to the maximum channel value
	for (uint16_t area_ndx = 0; area_ndx < area; area_ndx++, input_ptr++) {
		int32_t max_value = *input_ptr; // channel 0 starts off as the max
		int32_t *channel_ptr = input_ptr + area;
		for (uint16_t channel_ndx = 1;
		     channel_ndx < base1_args->remaining_args.channel_cnt;
		     channel_ndx++, channel_ptr += area) {
			if (*channel_ptr > max_value) {
				max_value = *channel_ptr;
			}
		}
		// found the max value for this surface ndx, now normalize.
		/* The Q11.12 value saturates at 2048. ln(2048) = 7.61283103, converted to q11.12 is	31182.
			Set the maximum value to this, and adjust all the other values accordingly. */
		int32_t offset = max_value - MAX_EXP_INPUT;
		channel_ptr = input_ptr;
		for (uint16_t channel_ndx = 0;
		     channel_ndx < base1_args->remaining_args.channel_cnt;
			 channel_ndx++, channel_ptr += area) {
			*channel_ptr -= offset;
		}
	}

	// run exp() on everything
	input_ptr = (int32_t*)base1_args->ptr_args.input; // reset input ptr to beginning
	uint16_t total_elements = area * base1_args->remaining_args.channel_cnt;
	for (uint16_t done_so_far=0;
	     done_so_far < total_elements;
		 done_so_far += 512, input_ptr +=512) {
		uint16_t done_this_time = total_elements-done_so_far;
		if (done_this_time > 512) {
			done_this_time = 512;
		} else	if(done_this_time < 4) { // minimum length of exp op is 4.
			memset(input_ptr + done_this_time, 0, (4 - done_this_time) * sizeof(*input_ptr));
			done_this_time = 4;
		} else if (done_this_time & 1) { // exp op needs even length
			*(input_ptr + done_this_time) = 0;
			done_this_time++; //
		}
		// MUST ALWAYS KEEP THE RESERVATION WHEN EXECUING AN INTRISIC WITHIN A OP EXTENSION!
		axon_exp_11p12(input_ptr, input_ptr, done_this_time,
		    NRF_AXON_SYNC_MODE_BLOCKING_POLLING, true);
	}
  float scaling_multiplier = (float)base1_args->remaining_args.output_multiplier /
          (float)(1 << base1_args->remaining_args.output_rounding);
	// go back through and normalize results to 1.
	input_ptr = (int32_t*)base1_args->ptr_args.input; // reset input ptr to beginning
	for (uint16_t area_ndx = 0; area_ndx < area; area_ndx++, input_ptr++) {
		uint64_t sum = 0;
		int32_t *channel_ptr = input_ptr;
		// 1st sum the results
		for (uint16_t channel_ndx = 0;
		     channel_ndx < base1_args->remaining_args.channel_cnt;
		     channel_ndx++, channel_ptr += area) {
			sum += *channel_ptr;
		}
		

		if(sum==0) {
			//Ideally we should not have a sum value of zero as we are offsetting the input to the exponents in the first software step,
			//in the off-case we do get all zeros, the actual values are going to be very small floating values which when quantized are equal to -128
			//setting sum=1 prevents undefined results w/o affecting the final results
			sum = 1;
		}

		// now go back and divide by the sum and quantize
		channel_ptr = input_ptr;
		for (uint16_t channel_ndx = 0;
		     channel_ndx < base1_args->remaining_args.channel_cnt;
			 channel_ndx++, channel_ptr += area) {
			/**
			 * @FIXME!! SEE ABOUT SAVING THE *channel_ptr/sum term to input, then using axon to quantize saturate.
			 * Also, the below doesn't handle negative saturation.
			 */

			float temp = (*channel_ptr) * scaling_multiplier;
			temp /= sum;
			temp = roundf(temp);
			temp += base1_args->remaining_args.output_zeropoint;
			switch (base1_args->remaining_args.output_bytewidth) {
			case 1:
				axon_saturate_i8(temp, (int8_t*)base1_args->ptr_args.output,
				    area_ndx + (area * channel_ndx));
				break;
			case 2:
				axon_saturate_i16(temp, (int16_t*)base1_args->ptr_args.output,
				    area_ndx + (area * channel_ndx));
				break;
			case 4:
				axon_saturate_i32(temp, (int32_t*)base1_args->ptr_args.output,
				    area_ndx + (area * channel_ndx));
				break;
			}
		}
	}
	return NRF_AXON_RESULT_SUCCESS;
}

/**
 * @brief
 * Implements neural net operator sigmoid as a software operation that can be
 * embedded in an axon command buffer. This is the base operation that can operate in legacy mode (pre 1.1.0) where
 * output is packed, and v2 mode where output is not masked.
 * @param argc number of arguments in argv. Must be 2
 * @param args down cast to a *nrf_axon_nn_op_extension_base1_args_s
 * @param packed_output if true, output will be written in packed format. if false, each row will start on a 32bit boundary.
 */
static nrf_axon_result_e nrf_axon_nn_op_extension_sigmoid_base(uint16_t argc,
          NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args, bool packed_output)
{
	if (((argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)) <
	     sizeof(nrf_axon_nn_op_extension_base1_args_s)) || (args==NULL)) {
		return NRF_AXON_RESULT_FAILURE;
	}
	nrf_axon_nn_op_extension_base1_args_s *base1_args =
	     (nrf_axon_nn_op_extension_base1_args_s*)args;
	/**
	 * Sigmoid is calculated 1 for 1 for all input. Input is int16, q3.12 format
	 * So iterate through channels/rows/columns
	 */
	// unpacked input rows always start on a 32bit boundary.
	uint8_t input_extra_stride =
	    (!base1_args->remaining_args.input_is_packed &&
	     base1_args->remaining_args.width & 1) ?
	    1 : 0;
	uint8_t output_extra_stride =
	           packed_output || (base1_args->remaining_args.output_bytewidth == 4) ?
			   0 : (4 - (base1_args->remaining_args.width & 3)) & 3;

	int16_t *input_ptr = (int16_t*)base1_args->ptr_args.input;
	union {
		int8_t *i8;
		int32_t *i32;
		NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE value;
	} output_ptr;
	output_ptr.value = (NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
	                   base1_args->ptr_args.output;
	for (uint16_t channel_ndx = 0;
	     channel_ndx < base1_args->remaining_args.channel_cnt;
		 channel_ndx++) {
		for (uint16_t row_ndx=0;
		     row_ndx < base1_args->remaining_args.height;
			 row_ndx++) {
			for (uint16_t col_ndx=0;
			     col_ndx < base1_args->remaining_args.width;
				 col_ndx++, input_ptr++) {
				float scratch = *input_ptr;
				/**
				 * sigmoid(x) = 1/(1+exp(-x))
				 */
				scratch /= (float)(1<<12); // input is q.12, convert to float
				scratch = (float)exp(-scratch); // now have exp(x)
				scratch = 1/(1+scratch); // have float sigmoid(x)
				switch (base1_args->remaining_args.output_bytewidth) {
					case 1: // quantized output. scales between 0 and 1.
						scratch = (float)round(scratch * 256.0f) - 128; // quantized
						axon_saturate_i8(scratch, output_ptr.i8, 0);
						output_ptr.i8++;
						break;
					case 4: // q1.30 output
						*output_ptr.i32 = (int32_t)(scratch * (1<<30));
						output_ptr.i32++;
						break;
					default:
						nrf_axon_platform_printf(
						    "Axon NN: Invalid Sigmoid bytewidth %d\n",
						    base1_args->remaining_args.output_bytewidth);
						return -1;
				}
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
                     NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args)
{
	return nrf_axon_nn_op_extension_sigmoid_base(argc, args, false);
}
/**
 * sigmoid version used by compiler versions 1.1.0 and later
 */
nrf_axon_result_e nrf_axon_nn_op_extension_sigmoid_v2(uint16_t argc,
                    NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args)
{
	return nrf_axon_nn_op_extension_sigmoid_base(argc, args, true);
}

/**
 * @brief
 * Implements neural net operator tanh as a software operation that can be
 * embedded in an axon command buffer. This is the base operation that can operate in legacy mode (pre 1.1.0) where
 * output is packed, and v2 mode where output is not masked.
 * @param argc number of arguments in argv. Must be 2
 * @param args down cast to a *nrf_axon_nn_op_extension_base1_args_s
 * @param packed_output if true, output will be written in packed format. if false, each row will start on a 32bit boundary.
 */
static nrf_axon_result_e nrf_axon_nn_op_extension_tanh_base(uint16_t argc,
                           NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args, bool packed_output)
{
	if (((argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)) <
	     sizeof(nrf_axon_nn_op_extension_base1_args_s)) || (args==NULL)) {
		return NRF_AXON_RESULT_FAILURE;
	}
	nrf_axon_nn_op_extension_base1_args_s *base1_args =
	         (nrf_axon_nn_op_extension_base1_args_s*)args;
	/**
	 * Sigmoid is calculated 1 for 1 for all input. Input is int16, q3.12 format
	 * So iterate through channels/rows/columns
	 */
	// unpacked input rows always start on a 32bit boundary.
	uint8_t input_extra_stride =
	       (!base1_args->remaining_args.input_is_packed &&
		    base1_args->remaining_args.width & 1) ? 1 : 0;
	uint8_t output_extra_stride = packed_output ||
	                              base1_args->remaining_args.output_bytewidth == 4 ?
	                              0 : (4 - (base1_args->remaining_args.width & 3)) & 3;

	int16_t *input_ptr = (int16_t*)base1_args->ptr_args.input;
	union {
		int8_t *i8;
		int32_t *i32;
		NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE value;
	} output_ptr;
	output_ptr.value = (NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
	                   base1_args->ptr_args.output;
	for (uint16_t channel_ndx = 0;
	     channel_ndx < base1_args->remaining_args.channel_cnt;
	     channel_ndx++) {
		for (uint16_t row_ndx=0;
		     row_ndx < base1_args->remaining_args.height;
			 row_ndx++) {
			for (uint16_t col_ndx=0;
			     col_ndx < base1_args->remaining_args.width;
				 col_ndx++, input_ptr++) {
				float scratch = *input_ptr;
				/**
				 * tanh(x) = (exp(2x)-1)/(exp(2x)+1)
				 */

				scratch /= (float)(1<<11); // input is q.12, multiply by 2 and convert to float
				scratch = expf(scratch); // now have exp(2x)
				scratch = (scratch - 1)/(scratch+1); // have float tanh(x)
				switch (base1_args->remaining_args.output_bytewidth) {
					case 1: // quantized output. scales between -1 and 1.
						scratch = roundf(scratch * 128.0f); // quantized
						axon_saturate_i8(scratch, output_ptr.i8, 0);
						output_ptr.i8++;
						break;
					case 4: // q1.30 output
						*output_ptr.i32 = (int32_t)(scratch * (1<<30));
						output_ptr.i32++;
						break;
					default:
						nrf_axon_platform_printf(
						    "Axon NN: Invalid Tanh bytewidth %d\n",
							base1_args->remaining_args.output_bytewidth);
						return -1;
				}
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
                       NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args)
{
	return nrf_axon_nn_op_extension_tanh_base(argc, args, false);
}
/**
 * tanh version used by compiler versions 1.1.0 and later
 */
nrf_axon_result_e nrf_axon_nn_op_extension_tanh_v2(uint16_t argc,
                      NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args)
{
	return nrf_axon_nn_op_extension_tanh_base(argc, args, true);
}

nrf_axon_result_e nrf_axon_nn_op_extension_reshape(uint16_t argc,
                      NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args)
{
	if (((argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)) <
	     sizeof(nrf_axon_nn_op_extension_base2_args_s)) || (args==NULL)) {
		return NRF_AXON_RESULT_FAILURE;
	}
	nrf_axon_nn_op_extension_base2_args_s *base2_args =
	               (nrf_axon_nn_op_extension_base2_args_s*)args;

	int input_stride = base2_args->remaining_args.input_stride;
	int output_stride = base2_args->remaining_args.output_width;

	for (uint16_t chan_ndx = 0;
	     chan_ndx < base2_args->remaining_args.output_channel_cnt;
		 chan_ndx++) { //channel

		for (uint16_t row_ndx=0;
		     row_ndx < base2_args->remaining_args.output_height;
			 row_ndx++) { // height

			for (uint16_t col_ndx=0;
			     col_ndx < base2_args->remaining_args.output_width;
				 col_ndx++) { //width

				int xtf = row_ndx * (base2_args->remaining_args.output_width *
				           base2_args->remaining_args.output_channel_cnt)
				        + col_ndx * base2_args->remaining_args.output_channel_cnt
				        + chan_ndx;

				int h_prime = xtf / (base2_args->remaining_args.input_width *
				               base2_args->remaining_args.input_channel_cnt);
				int rem =  xtf - (h_prime *
				        (base2_args->remaining_args.input_width *
						 base2_args->remaining_args.input_channel_cnt));
				int w_prime = rem / base2_args->remaining_args.input_channel_cnt;
				int c_prime = rem -
				     w_prime * base2_args->remaining_args.input_channel_cnt;

				int old_idx = c_prime *
				             (base2_args->remaining_args.input_height * input_stride)
				        + h_prime * input_stride
				        + w_prime;

				int new_idx = chan_ndx *
				             (base2_args->remaining_args.output_height * output_stride)
				        + row_ndx * output_stride
				        + col_ndx;

				((int8_t*)base2_args->ptr_args.output)[new_idx] =
				          ((int8_t*)base2_args->ptr_args.input)[old_idx];
			}
		}
	}
	return NRF_AXON_RESULT_SUCCESS;
}

static inline int32_t get_nearest_neighbor(const int input_value,
                                  const int32_t input_size,
                                  const int32_t output_size,
                                  const bool align_corners,
                                  const bool half_pixel_centers) {
#define MY_MIN(a,b) (a>b ? b : a)
#define MY_MAX(a,b) (a>b ? a : b)
	const float scale =
	            (align_corners && output_size > 1) ?
	            (input_size - 1) / (float)(output_size - 1) :
	            input_size / (float)(output_size);
	const float offset =
	            half_pixel_centers ? 0.5f : 0.0f;
	            int32_t output_value = MY_MIN(
	              align_corners ?
	              (int32_t)(roundf((input_value + offset) * scale)) :
	              (int32_t)((input_value + offset) * scale), input_size - 1);
	if (half_pixel_centers) {
		output_value = MY_MAX(0, output_value);
	}
	return output_value;
}

nrf_axon_result_e nrf_axon_nn_op_extension_resize_nearest_neighbor(uint16_t argc,
                     NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args)
{
	if (((argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)) <
	     sizeof(nrf_axon_nn_op_extension_base2_args_s)) || (args==NULL)) {
		return NRF_AXON_RESULT_FAILURE;
	}
	nrf_axon_nn_op_extension_resize_nearest_neighbor_args_s
	    *resize_nearest_neighbor_args =
		(nrf_axon_nn_op_extension_resize_nearest_neighbor_args_s*)args;

	int input_stride = resize_nearest_neighbor_args->remaining_args.input_stride;
	int output_stride = resize_nearest_neighbor_args->remaining_args.output_width;
	int input_z_stride = input_stride *
	       resize_nearest_neighbor_args->remaining_args.input_height;
	int output_z_stride = output_stride *
	       resize_nearest_neighbor_args->remaining_args.output_height;

// iterate across the output surface.
	for (uint16_t row_ndx=0;
	     row_ndx < resize_nearest_neighbor_args->remaining_args.output_height;
	     row_ndx++) { // height

		int32_t in_row_ndx = get_nearest_neighbor(row_ndx,
		        resize_nearest_neighbor_args->remaining_args.input_height,
		        resize_nearest_neighbor_args->remaining_args.output_height,
		        resize_nearest_neighbor_args->remaining_args.align_corners, //align_corners,
		        resize_nearest_neighbor_args->remaining_args.half_pixel_centers); //half_pixel_centers

		for (uint16_t col_ndx=0;
		     col_ndx < resize_nearest_neighbor_args->remaining_args.output_width;
		     col_ndx++) { //width
			
			int32_t in_col_ndx = get_nearest_neighbor(col_ndx,
			       resize_nearest_neighbor_args->remaining_args.input_width,
			       resize_nearest_neighbor_args->remaining_args.output_width,
			       resize_nearest_neighbor_args->remaining_args.align_corners, //align_corners,
			       resize_nearest_neighbor_args->remaining_args.half_pixel_centers); //half_pixel_centers
			// now propagate the input to the output across all channels.
			int32_t input_offset = in_row_ndx *
			                       resize_nearest_neighbor_args->remaining_args.input_stride +
			                       in_col_ndx;
			int32_t output_offset = row_ndx * output_stride + col_ndx;
			for (uint16_t chan_ndx = 0;
			     chan_ndx < resize_nearest_neighbor_args->remaining_args.output_channel_cnt;
			     chan_ndx++) { //channel
				resize_nearest_neighbor_args->ptr_args.output[output_offset] =
				    resize_nearest_neighbor_args->ptr_args.input[input_offset];
				input_offset += input_z_stride;
				output_offset += output_z_stride;
			}
		}
	}
  return NRF_AXON_RESULT_SUCCESS;
}
