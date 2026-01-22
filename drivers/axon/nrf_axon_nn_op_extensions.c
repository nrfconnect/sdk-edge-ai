/**
 *          Copyright (c) 2020-2022, Atlazo Inc.
 *          All rights reserved.
 *
 *          Licensed under the Apache License, Version 2.0 (the "License");
 *          you may not use this file except in compliance with the License.
 *          You may obtain a copy of the License at
 *
 *              http://www.apache.org/licenses/LICENSE-2.0
 *
 *          Unless required by applicable law or agreed to in writing, software
 *          distributed under the License is distributed on an "AS IS" BASIS,
 *          WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *          See the License for the specific language governing permissions and
 *          limitations under the License.
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "nrf_axon_platform.h"
#include "nrf_axon_nn_op_extensions.h"
#include "nrf_axon_nn_infer.h"
#include "nrf_axon_dsp_intrinsics.h"

/**
 * @brief
 * Implements neural net operator softmax as a software operation that can be
 * embedded in an axon command buffer.
 * @param argc number of arguments in argv. Must be 2
 * @param args down cast to a *nrf_axon_nn_op_extension_base1_args_s
 */
nrf_axon_result_e nrf_axon_nn_op_extension_softmax(uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args)
{
 #define MAX_EXP_INPUT (31182) //the maximum input to exp before it saturates
 if (((argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)) < sizeof(nrf_axon_nn_op_extension_base1_args_s)) || (args==NULL)) {
    return NRF_AXON_RESULT_FAILURE;
  }
  nrf_axon_nn_op_extension_base1_args_s *base1_args = (nrf_axon_nn_op_extension_base1_args_s*)args;
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
    for (uint16_t channel_ndx = 1; channel_ndx < base1_args->remaining_args.channel_cnt; channel_ndx++, channel_ptr += area) {
      if (*channel_ptr > max_value) {
        max_value = *channel_ptr;
      }
    }
    // found the max value for this surface ndx, now normalize.
    /* The Q11.12 value saturates at 2048. ln(2048) = 7.61283103, converted to q11.12 is  31182.
      Set the maximum value to this, and adjust all the other values accordingly. */
    int32_t offset = max_value - MAX_EXP_INPUT;
    channel_ptr = input_ptr;
    for (uint16_t channel_ndx = 0; channel_ndx < base1_args->remaining_args.channel_cnt; channel_ndx++, channel_ptr += area) {
      *channel_ptr -= offset;
    }
  }

  // run exp() on everything
  input_ptr = (int32_t*)base1_args->ptr_args.input; // reset input ptr to beginning
  uint16_t total_elements = area * base1_args->remaining_args.channel_cnt;
  for (uint16_t done_so_far=0; done_so_far < total_elements; done_so_far += 512, input_ptr +=512) {
    uint16_t done_this_time = total_elements-done_so_far;
    if (done_this_time > 512) {
      done_this_time = 512;
    } else  if(done_this_time < 4) { // minimum length of exp op is 4.
      *(input_ptr+3) = 0;
      if (done_this_time < 3) {
        *(input_ptr+2) = 0;
        if (done_this_time < 2) {
          *(input_ptr+1) = 0;
        }
      }
      done_this_time = 4; 
    } else if (done_this_time & 1) { // exp op needs even length
      *(input_ptr + done_this_time) = 0;
      done_this_time++; // 
    }
    // MUST ALWAYS KEEP THE RESERVATION WHEN EXECUING AN INTRISIC WITHIN A OP EXTENSION!
    axon_exp_11p12(input_ptr, input_ptr, done_this_time, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, true);
  }

  // go back through and normalize results to 1.
  input_ptr = (int32_t*)base1_args->ptr_args.input; // reset input ptr to beginning
  for (uint16_t area_ndx = 0; area_ndx < area; area_ndx++, input_ptr++) {
    uint64_t sum = 0;
    int32_t *channel_ptr = input_ptr;
    // 1st sum the results
    for (uint16_t channel_ndx = 0; channel_ndx < base1_args->remaining_args.channel_cnt; channel_ndx++, channel_ptr += area) {
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
    for (uint16_t channel_ndx = 0; channel_ndx < base1_args->remaining_args.channel_cnt; channel_ndx++, channel_ptr += area) {
      /**
       * @FIXME!! SEE ABOUT SAVING THE *channel_ptr/sum term to input, then using axon to quantize saturate.
       * Also, the below doesn't handle negative saturation. 
       */

      int64_t temp = (int64_t) (*channel_ptr) * (int64_t)base1_args->remaining_args.output_multiplier;  
      temp /= sum;
      temp += ((int64_t)base1_args->remaining_args.output_zeropoint << base1_args->remaining_args.output_rounding);
      temp >>= base1_args->remaining_args.output_rounding;
      switch (base1_args->remaining_args.output_bytewidth) {
      case 1:
        (*((int8_t*)(base1_args->ptr_args.output)+(area_ndx + (area * channel_ndx)))) = (int8_t)(temp  > INT8_MAX ? INT8_MAX : temp < INT8_MIN ? INT8_MIN : temp);
        break;
      case 2:       
        (*((int16_t*)(base1_args->ptr_args.output)+(area_ndx + (area * channel_ndx)))) = (int16_t)(temp  > INT16_MAX ? INT16_MAX : temp < INT16_MIN ? INT16_MIN : temp);
        break;
      case 4:
        (*((int32_t*)(base1_args->ptr_args.output)+(area_ndx + (area * channel_ndx)))) = (int32_t)(temp  > INT32_MAX ? INT32_MAX : temp < INT32_MIN ? INT32_MIN : temp);
        break;
      }      
    }
  }
  return NRF_AXON_RESULT_SUCCESS;
}

/**
 * @brief
 * Implements neural net operator sigmoid as a software operation that can be
 * embedded in an axon command buffer.
 * @param argc number of arguments in argv. Must be 2
 * @param args down cast to a *nrf_axon_nn_op_extension_base1_args_s
 */
nrf_axon_result_e nrf_axon_nn_op_extension_sigmoid(uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args)
{
 if (((argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)) < sizeof(nrf_axon_nn_op_extension_base1_args_s)) || (args==NULL)) {
    return NRF_AXON_RESULT_FAILURE;
  }
  nrf_axon_nn_op_extension_base1_args_s *base1_args = (nrf_axon_nn_op_extension_base1_args_s*)args;
  /**
   * Sigmoid is calculated 1 for 1 for all input. Input is int16, q3.12 format
   * So iterate through channels/rows/columns
   */
  // unpacked input rows always start on a 32bit boundary.
  uint8_t input_extra_stride = (!base1_args->remaining_args.input_is_packed && base1_args->remaining_args.width & 1) ? 1 : 0; 
  uint8_t output_extra_stride = base1_args->remaining_args.output_bytewidth == 4 ? 0 : (4 - base1_args->remaining_args.width & 3) & 3;

  int16_t *input_ptr = (int16_t*)base1_args->ptr_args.input;
  union {
    int8_t *i8;
    int32_t *i32;
    NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE value;
  } output_ptr;
  output_ptr.value = (NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)base1_args->ptr_args.output;
  for (uint16_t channel_ndx = 0; channel_ndx < base1_args->remaining_args.channel_cnt; channel_ndx++) {
    for (uint16_t row_ndx=0; row_ndx < base1_args->remaining_args.height; row_ndx++) {
      for (uint16_t col_ndx=0;col_ndx < base1_args->remaining_args.width; col_ndx++, input_ptr++) {
        float scratch = *input_ptr;
        /**
         * sigmoid(x) = 1/(1+exp(-x))
         */
        scratch /=  (float)(1<<12); // input is q.12, convert to float
        scratch = (float)exp(-scratch); // now have exp(x)
        scratch = 1/(1+scratch); // have float sigmoid(x)
        switch (base1_args->remaining_args.output_bytewidth) {
          case 1: // quantized output. scales between 0 and 1.
            scratch = (float)round(scratch * 256.0) - 128; // quantized
            *output_ptr.i8 = scratch > 127 ? 127: scratch < -128 ? -128 : (int8_t)scratch; // saturated
            output_ptr.i8++;
            break;
          case 4: // q1.30 output
            *output_ptr.i32 = (int32_t)(scratch * (1<<30));
            output_ptr.i32++;
            break;
          default:
            nrf_axon_platform_printf("Axon NN: Invalid Sigmoid bytewidth %d\n", base1_args->remaining_args.output_bytewidth);
            return -1;
        }
      }
      input_ptr += input_extra_stride;
      output_ptr.value += output_extra_stride;
    }
  }
  return NRF_AXON_RESULT_SUCCESS;
}

nrf_axon_result_e nrf_axon_nn_op_extension_tanh(uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args)
{
 if (((argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)) < sizeof(nrf_axon_nn_op_extension_base1_args_s)) || (args==NULL)) {
    return NRF_AXON_RESULT_FAILURE;
  }
  nrf_axon_nn_op_extension_base1_args_s *base1_args = (nrf_axon_nn_op_extension_base1_args_s*)args;
  /**
   * Sigmoid is calculated 1 for 1 for all input. Input is int16, q3.12 format
   * So iterate through channels/rows/columns
   */
  // unpacked input rows always start on a 32bit boundary.
  uint8_t input_extra_stride = (!base1_args->remaining_args.input_is_packed && base1_args->remaining_args.width & 1) ? 1 : 0; 
  uint8_t output_extra_stride = base1_args->remaining_args.output_bytewidth == 4 ? 0 : (4 - base1_args->remaining_args.width & 3) & 3;

  int16_t *input_ptr = (int16_t*)base1_args->ptr_args.input;
  union {
    int8_t *i8;
    int32_t *i32;
    NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE value;
  } output_ptr;
  output_ptr.value = (NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)base1_args->ptr_args.output;
  for (uint16_t channel_ndx = 0; channel_ndx < base1_args->remaining_args.channel_cnt; channel_ndx++) {
    for (uint16_t row_ndx=0; row_ndx < base1_args->remaining_args.height; row_ndx++) {
      for (uint16_t col_ndx=0;col_ndx < base1_args->remaining_args.width; col_ndx++, input_ptr++) {
        float scratch = *input_ptr;
        /**
         * tanh(x) = (exp(2x)-1)/(exp(2x)+1)
         */

        scratch /=  (float)(1<<11); // input is q.12, multiply by 2 and convert to float
        scratch = expf(scratch); // now have exp(2x)
        scratch = (scratch - 1)/(scratch+1); // have float tanh(x)
        switch (base1_args->remaining_args.output_bytewidth) {
          case 1: // quantized output. scales between -1 and 1.
            scratch = roundf(scratch * 128.0f); // quantized
            *output_ptr.i8 = scratch > 127 ? 127: scratch < -128 ? -128 : (int8_t)scratch; // saturated
            output_ptr.i8++;
            break;
          case 4: // q1.30 output
            *output_ptr.i32 = (int32_t)(scratch * (1<<30));
            output_ptr.i32++;
            break;
          default:
            nrf_axon_platform_printf("Axon NN: Invalid Sigmoid bytewidth %d\n", base1_args->remaining_args.output_bytewidth);
            return -1;
        }
      }
      input_ptr += input_extra_stride;
      output_ptr.value += output_extra_stride;
    }
  }
  return NRF_AXON_RESULT_SUCCESS;
}
