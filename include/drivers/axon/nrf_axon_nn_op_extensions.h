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
 * 
 * These functions are encoded by the nn compiler into the command buffer. 
 * They are invoked by the axon driver during model inference, using the paramters embedded in the
 * command buffer.
 */
typedef nrf_axon_result_e(*axon_op_extension_func)(uint16_t arg_size, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args);


/**
 * @brief Structure compiled into the command buffer then passed to some CPU op extension functions
 * 
 * @note To maintain compatibilty with simulator builds, the parameters are in elements of NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE.
 * This is 64bits on the simulator, 32bits on Nordic MCUs.
 * By organizing this as a series of unions, the upper 32bits of the parameters remain 0s unless the arguement is a pointer.
 * 
 * Op Extension Base 1 profile arguments. The following conditions must be met for an op extension to use
 * this structure to pass its parameters. (If the conditions aren't met, a new structure needs to be defined.)
 * - A single input vector.
 * - input and output dimensions are the same.
 * - output is packed (rows are padded to 4byte boundaries)
 * - input/output data storage is data[channel_cnt][height][ceil(width,4)]
 * - input bytewidth is 4 (32bit), implicitly q11.12
 * - output bytewidth depends on quantization_enabled.
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
        uint16_t height;          /*< height in elements of the input and output */
        uint16_t width;           /*< width in elements of the input and output */
      };
    };
    union {
      NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer2;
      struct {
        uint16_t channel_cnt;    /*< number of channels in elements of the input and output */
        uint8_t output_rounding; /*< quantization is ((output * output_multiplier) >> output_rounding) + output_zeropoint */
        int8_t output_zeropoint; /*< quantization is output * output_multiplier >> output_rounding + output_zeropoint */
      };
    };
    union {
      NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer3;
      struct {
        uint8_t output_bytewidth;   /*< can be 1 or 4*/
        bool quantization_enabled;  /*< 8bt quantized if true, q11.12 if false*/
        bool input_is_packed;       /*< If true, input rows begin on the following byte from the previous row. If false, rows are aligned to 4byte boundaries. */
      };
    };
  } remaining_args;
} nrf_axon_nn_op_extension_base1_args_s;

/**
 * @brief Implements softmax operator.
 * 
 * @param[in] argc number of elements in args. Must equal sizeof(nrf_axon_nn_op_extension_base1_args_s)/sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
 * @param[in] args up-casted nrf_axon_nn_op_extension_base1_args_s, with parameters to the function.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_softmax(uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args);

/**
 * @brief Implements sigmoid operator.
 * 
 * @param[in] argc number of elements in args. Must equal sizeof(nrf_axon_nn_op_extension_base1_args_s)/sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
 * @param[in] args up-casted nrf_axon_nn_op_extension_base1_args_s, with parameters to the function.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_sigmoid(uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args);

/**
 * @brief Implements tanh operator.
 * 
 * @param[in] argc number of elements in args. Must equal sizeof(nrf_axon_nn_op_extension_base1_args_s)/sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
 * @param[in] args up-casted nrf_axon_nn_op_extension_base1_args_s, with parameters to the function.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_tanh(uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args);

/**
 * @brief Implements reshape operator.
 * 
 * @param[in] argc number of elements in args. Must equal sizeof(nrf_axon_nn_op_extension_base1_args_s)/sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
 * @param[in] args up-casted nrf_axon_nn_op_extension_base1_args_s, with parameters to the function.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_reshape(uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* args);


#ifdef __cplusplus
} // extern "C" {
#endif

