/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
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
typedef nrf_axon_result_e
	(*axon_op_extension_func)
		(uint16_t arg_size, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args);

/**
 * @brief Structure compiled into the command buffer then passed to some CPU op extension functions
 *
 * @note To maintain compatibility with simulator builds, the parameters are in elements of
 * NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE.
 * This is 64bits on the simulator, 32bits on Nordic MCUs.
 * By organizing this as a series of unions, the upper 32bits of the parameters remain 0s unless
 * the argument is a pointer.
 *
 * Op Extension Base 1 profile arguments. The following conditions must be met for an op extension
 * to use this structure to pass its parameters. (If the conditions aren't met, a new structure
 * needs to be defined.)
 * - A single input vector.
 * - input and output dimensions are the same.
 * - output is packed (rows are padded to 4byte boundaries)
 * - input/output data storage is data[channel_cnt][height][ceil(width,4)]
 * - input bytewidth is 4 (32bit), implicitly q11.12
 * - output bytewidth depends on quantization_enabled.
 */
typedef struct  {
	struct {
		/**< to compile properly, all pointers must be declared 1st, consecutively */
		int8_t *input;
		/**< to compile properly, all pointers must be declared 1st, consecutively */
		int8_t *output;
	} ptr_args;
	struct {
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer0;
			/**< quantization => (output * output_multiplier) >>
			 * output_rounding + output_zeropoint
			 */
			/**< quantization output multiplier */
			uint32_t output_multiplier;
		};
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer1;
			struct {
				/**< height in elements of the input and output */
				uint16_t height;
				/**< width in elements of the input and output */
				uint16_t width;
			};
		};
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer2;
			struct {
				/**< number of channels in elements of the input and output */
				uint16_t channel_cnt;
				/**< quantization output rounding  in bits */
				uint8_t output_rounding;
				/**< quantization output zero point */
				int8_t output_zeropoint;
			};
		};
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer3;
			struct {
				/**< can be 1 or 4*/
				uint8_t output_bytewidth;
				/**< 8bt quantized if true, q11.12 if false*/
				bool quantization_enabled;
				/**< If true, input rows begin on the following byte from the
				 * previous row. If false, rows are aligned to 4byte boundaries.
				 */
				bool input_is_packed;
			};
		};
	} remaining_args;
} nrf_axon_nn_op_extension_base1_args_s;


/**
 * @brief
 * Structure compiled into the command buffer then passed to some CPU op extension functions
 *
 * @note
 * To maintain compatibility with simulator builds, the parameters are in elements of
 * NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE.
 * This is 64bits on the simulator, 32bits on Nordic MCUs.
 * By organizing this as a series of unions, the upper 32bits of the parameters remain 0s unless
 * the argument is a pointer.
 *
 * Op Extension Base 2 profile arguments. The following conditions must be met for an op extension
 * to use this structure to pass its parameters. (If the conditions aren't met, a new structure
 * needs to be defined.)
 * - A single input vector.
 * - input and output dimensions are not the same.
 * - no quantization involved in the oepration.
 */
typedef struct  {
	struct {
		/**< to compile properly, all pointers must be declared 1st, consecutively */
		/**< pointer to the input of this layer */
		int8_t *input;
		/**< pointer to the output of this layer */
		int8_t *output;
	} ptr_args;
	struct {
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer0;
			struct {
				/*< height in elements of the input */
				uint16_t input_height;
				/*< width in elements of the input */
				uint16_t input_width;
			};
		};
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer1;
			struct {
				/*< height in elements of the output */
				uint16_t output_height;
				/*< width in elements of the output */
				uint16_t output_width;
			};
		};
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer2;
			struct {
				/*< channel count in elements of the input */
				uint16_t input_channel_cnt;
				/*< channel count in elements of the output */
				uint16_t output_channel_cnt;
			};
		};
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer3;
			struct {
				/**< Distance in bytes between the start each input row. */
				uint16_t input_stride;
			};
		};
	} remaining_args;
} nrf_axon_nn_op_extension_base2_args_s;


/**
 * @brief Implements softmax operator.
 *
 * @param[in] argc number of elements in args. Must equal
 * sizeof(nrf_axon_nn_op_extension_base1_args_s)/sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
 * @param[in] args up-casted nrf_axon_nn_op_extension_base1_args_s, with parameters to the function.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_softmax(
	uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args);

/**
 * @brief Implements sigmoid operator.
 *
 * Note: legacy version that produces packed output for compiler versions below 1.1.0
 *
 * @param[in] argc number of elements in args. Must equal
 *            sizeof(nrf_axon_nn_op_extension_base1_args_s)/
 *            sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
 * @param[in] args up-casted nrf_axon_nn_op_extension_base1_args_s, with parameters to the function.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_sigmoid(
	uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args);

/**
 * @brief Implements sigmoid operator.
 *
 * Note: Updated version that produces packed output for compiler versions 1.1.0 and later.
 *
 * @param[in] argc number of elements in args. Must equal
 *            sizeof(nrf_axon_nn_op_extension_base1_args_s)/
 *            sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
 * @param[in] args up-casted nrf_axon_nn_op_extension_base1_args_s, with parameters to the function.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_sigmoid_v2(
	uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args);

/**
 * @brief Implements tanh operator.
 *
 * Note: legacy version that produces packed output for compiler versions below 1.1.0
 *
 * @param[in] argc number of elements in args. Must equal
 *            sizeof(nrf_axon_nn_op_extension_base1_args_s)/
 *            sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
 * @param[in] args up-casted nrf_axon_nn_op_extension_base1_args_s, with parameters to the function.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_tanh(
	uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args);

/**
 * @brief Implements tanh operator.
 *
 * Note: Updated  version that produces packed output for compiler versions 1.1.0 and later.
 *
 * @param[in] argc number of elements in args. Must equal
 *            sizeof(nrf_axon_nn_op_extension_base1_args_s)/
 *            sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
 * @param[in] args up-casted nrf_axon_nn_op_extension_base1_args_s, with parameters to the function.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_tanh_v2(
	uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args);

/**
 * @brief Implements reshape operator.
 *
 * @param[in] argc number of elements in args. Must equal
 *            sizeof(nrf_axon_nn_op_extension_base2_args_s)/
 *            sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
 * @param[in] args up-casted nrf_axon_nn_op_extension_base2_args_s, with parameters to the function.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_reshape(
	uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args);



/**
 * @brief
 * Structure compiled into the command buffer then passed to the CPU op extension function
 * resize_nearest_neighbor
 *
 * @note
 * To maintain compatibility with simulator builds, the parameters are in elements of
 * NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE.
 * This is 64bits on the simulator, 32bits on Nordic MCUs.
 * By organizing this as a series of unions, the upper 32bits of the parameters remain 0s unless
 * the argument is a pointer.
 */
typedef struct  {
	struct {
		/**< to compile properly, all pointers must be declared 1st, consecutively */
		/**< pointer to input. */
		int8_t *input;
		/**< pointer to output */
		int8_t *output;
	} ptr_args;
	struct {
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer0;
			struct {
				/**< height in elements of the input */
				uint16_t input_height;
				/**< width in elements of the input */
				uint16_t input_width;
			};
		};
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer1;
			struct {
				/**< height in elements of the output */
				uint16_t output_height;
				/**< width in elements of the output */
				uint16_t output_width;
			};
		};
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer2;
			struct {
				/**< channel count in elements of the input */
				uint16_t input_channel_cnt;
				/**< channel count in elements of the output */
				uint16_t output_channel_cnt;
			};
		};
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer3;
			struct {
				/**< Distance in bytes between the start each input row. */
				uint16_t input_stride;
				bool align_corners;
				bool half_pixel_centers;
			};
		};
	} remaining_args;
} nrf_axon_nn_op_extension_resize_nearest_neighbor_args_s;

/**
 * @brief Implements resize_nearest_neighbor operator.
 *
 * @param[in] argc number of elements in args. Must equal
 *            sizeof(nrf_axon_nn_op_extension_resize_nearest_neighbor_args_s)/
 *            sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
 * @param[in] args up-casted nrf_axon_nn_op_extension_resize_nearest_neighbor_args_s,
 *            with parameters to the function.
 */
nrf_axon_result_e nrf_axon_nn_op_extension_resize_nearest_neighbor(
	uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args);



/**
 * @brief
 * Structure compiled into the command buffer then passed to the CPU op extension functions
 * space_to_batcnh and batch_to_space
 *
 * @note
 * To maintain compatibility with simulator builds, the parameters are in elements of
 * NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE.
 * This is 64bits on the simulator, 32bits on Nordic MCUs.
 * By organizing this as a series of unions, the upper 32bits of the parameters remain 0s unless
 * the argument is a pointer.
 */
typedef struct  {
	struct {
		/**< to compile properly, all pointers must be declared 1st, consecutively */
		/**< pointer to spatial buffer */
		int8_t *space;
		/**< pointer to batch buffer */
		int8_t *batch;
	} ptr_args;
	struct {
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer0;
			struct {
				/**< height in elements of the input */
				uint16_t space_height;
				/**< width in elements of the input */
				uint16_t space_width;
			};
		};
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer1;
			struct {
				/**< height in elements of the output */
				uint16_t batch_height;
				/**< width in elements of the output */
				uint16_t batch_width;
			};
		};
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer2;
			struct {
				/**< channel count in elements */
				uint16_t channel_cnt;
				/**< pad value for space_to_batch */
				uint16_t zero_point;
			};
		};
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer3;
			struct {
				/**
				 * Distance in bytes between the start each input row.
				 * Applies to the space or the batch depending on which is
				 * the input. Output stride will always be the output width.
				 */
				uint16_t input_stride;
				/**< number of batches in the input */
				uint16_t batch_cnt;
			};
		};
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE spacer4;
			struct {
				/**
				 * Padding on the spatial buffer. Gets added for
				 * space_to_batch, cropped for batch_to_space
				 */
				uint8_t pad_top;
				uint8_t pad_bottom;
				uint8_t pad_left;
				uint8_t pad_right;
			};
		};
		union {
			NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE space5;
			struct {
				uint16_t block_height;
				uint16_t block_width;
			};
		};
	} remaining_args;
} nrf_axon_nn_op_extension_space_to_from_batch_args_s;

/**
 * @brief Implements space_to_batch operator.
 *
 * @param[in] argc number of elements in args. Must equal
 *            sizeof(nrf_axon_nn_op_extension_batch_to_space_args_s)/
 *            sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
 * @param[in] args up-casted nrf_axon_nn_op_extension_batch_to_space_args_s,
 *            with parameters to the function.
 */
nrf_axon_result_e nrf_axon_nn_cpu_op_space_to_batch(
	uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args);

/**
 * @brief Implements batch_to_space operator.
 *
 * @param[in] argc number of elements in args. Must equal
 *            sizeof(nrf_axon_nn_op_extension_batch_to_space_args_s)/
 *            sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)
 * @param[in] args up-casted nrf_axon_nn_op_extension_batch_to_space_args_s,
 *            with parameters to the function.
 */
nrf_axon_result_e nrf_axon_nn_cpu_op_batch_to_space(
	uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args);

#ifdef __cplusplus
} /* extern "C" { */
#endif
