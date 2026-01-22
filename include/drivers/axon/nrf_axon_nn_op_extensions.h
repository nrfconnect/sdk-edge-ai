/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include <stdbool.h>
#include "nrf_axon_driver.h"

/**
 * @brief Function prototype for all operation extension functions.
 */
typedef nrf_axon_result_e(*axon_op_extension_func)(uint16_t arg_size, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args);


/**
 * Op Extension Base 1 profile argumnents.
 * input and output dimensions are the same.
 * input/output are  unpacked (rows are padded to 4byte boundaries)
 * input/output data storage is data[channel_cnt][height][ceil(width,4)]
 * input bytewidth is 4 (32bit), implicitly q11.12
 * output bytewidth depends on quantization_enabled.
 */
typedef struct  {
  struct {
    int8_t *input;  /*< to compile properly, all pointers must be declared 1st, consecutively */
    int8_t *output; /*< to compile properly, all pointers must be declared 1st, consecutively */
  } ptr_args;
  struct {
    union {
      NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer0;
      uint32_t output_multiplier; /*< quantization => output * output_multiplier >> output_rounding + output_zeropoint */
    };
    union {
      NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer1;
      struct {
        uint16_t height;
        uint16_t width;
      };
    };
    union {
      NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer2;
      struct {
        uint16_t channel_cnt;
        uint8_t output_rounding; /*< quantization is ((output * output_multiplier) >> output_rounding) + output_zeropoint */
        int8_t output_zeropoint; /*< quantization is output * output_multiplier >> output_rounding + output_zeropoint */
      };
    };
    union {
      NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer3;
      struct {
        uint8_t output_bytewidth; /*< can be 1 or 4*/
        bool quantization_enabled; /*< 8bt quantized if true, q11.12 if false*/
        bool input_is_packed;
      };
    };
  } remaining_args;
} nrf_axon_nn_op_extension_base1_args_s;

/**
 * Implements softmax operator.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_softmax(uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args);

/**
 * Implements sigmoid operator.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_sigmoid(uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args);

/**
 * implements tanh operator
 */
nrf_axon_result_e nrf_axon_nn_op_extension_tanh(uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args);

#ifdef __cplusplus
} // extern "C" {
#endif

