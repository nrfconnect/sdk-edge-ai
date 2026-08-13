/*
 * Copyright (c) 2025-2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/*
 * File used by the cross compiler to generate the binary file.
 * The file acts as an interface file for the cross compiler and has the
 * common definitions used by the axon compiler and the tflite cross compiler
 *
 * This file will be referenced using its direct path by the cross compiler.
 * The file needs to satisfy the following conditions to ensure that it is
 * valid and can be used an interface header file
 * Condition 1 : The file should not have any #include directories
 * Condition 2 : Macros defined in the file should have fixed integer values
 *               for example #define AXONPRO_MACRO (1) and not
 *               #define AXONPRO_MACRO(X) (X<<2)
 * Condition 3 : All the typedefs should be contained within this file and
 *               shold not have references outside the file
 * Condition 4 : It should not contain any #ifdef macros
 */
#pragma once

typedef enum {
	/* Compiler failed: persistent buffer too small for the model. */
	NRF_AXON_COMPILER_RESULT_PERSISTENT_BUFFER_TOO_SMALL = -243,
	/* Compiler failed due to invalid batch count parameters. */
	NRF_AXON_COMPILER_RESULT_INVALID_BATCH_CNT = -242,
	/* Compiler failed due to invalid dilation parameters. */
	NRF_AXON_COMPILER_RESULT_INVALID_DILATION           = -242,
	/* Compiler failed due to invalid rounding parameters. */
	NRF_AXON_COMPILER_RESULT_INVALID_ROUNDING           = -241,
	/* Compiler failed due to input height/width/depth violation.*/
	NRF_AXON_COMPILER_RESULT_INVALID_INPUT_SIZE         = -240,
	/* Compiler failed due to filter height/width/depth parameter violation. */
	NRF_AXON_COMPILER_RESULT_INVALID_FILTER_SIZE        = -239,
	/* Compiler failed due to invalid padding parameters. */
	NRF_AXON_COMPILER_RESULT_INVALID_PADDING            = -238,
	/* Compiler failed due to invalid stride parameters. */
	NRF_AXON_COMPILER_RESULT_INVALID_STRIDE             = -236,
	/* Compiler failed due to too many or not enough inputs to a layer. */
	NRF_AXON_COMPILER_RESULT_INVALID_INPUT              = -235,
	/* Compiler failed due to an invalid number filters (output channels) */
	NRF_AXON_COMPILER_RESULT_INVALID_FILTER_COUNT       = -234,
	/* Compiler failed due to internal, oversized PSUM buffer being too small for the model. */
	NRF_AXON_COMPILER_RESULT_PSUM_BUFFER_TOO_SMALL      = -232,
	/*
	 * Compiler failed due to specified activation function not being a member of
	 * nrf_axon_nn_activation_function_e.
	 */
	NRF_AXON_COMPILER_RESULT_INVALID_ACTIVATION_FUNCTION = -228,
	/* Compiler failed during simulator test inference. */
	NRF_AXON_COMPILER_RESULT_INFERENCE_FAILED           = -227,
	/* Compiler failed initializing the simulator for test inference. */
	NRF_AXON_COMPILER_RESULT_SIMULATOR_INIT_FAILED      = -226,
	/* Compiler failed initializing model for simulator test inference. */
	NRF_AXON_COMPILER_RESULT_MODEL_INIT_FAILED          = -225,
	/*
	 * Compiler failed due the model containing an address to a symbol that was not
	 * added to the symbol list.
	 */
	NRF_AXON_COMPILER_RESULT_UNRESOLVED_ADDRESS         = -224,
	/* Compiler failed due to internal error. */
	NRF_AXON_COMPILER_RESULT_CORRUPT_CMD_BUFFER         = -223,
	/* Compiler failed due to oversized interlayer buffer being too small for the model. */
	NRF_AXON_COMPILER_RESULT_INTERLAYER_BUFFER_TOO_SMALL = -222,
	/* Compiler failed due to number of declared buffers exceeding the internal limit. */
	NRF_AXON_COMPILER_RESULT_TOO_MANY_DECLARED_BUFFERS  = -221,
	/* compiler could not allocate enough memory */
	NRF_AXON_COMPILER_RESULT_MALLOC_FAILED              = -220,
	/*
	 * Error occurred in compiler due to invalid bin file. Version mismatch
	 * or file corruption.
	 */
	NRF_AXON_COMPILER_RESULT_INVALID_BIN_FILE           = -219,
	/* Error occurred in compiler while performing file i/o */
	NRF_AXON_COMPILER_RESULT_FILE_IO_FAILED             = -218,
	/*
	 * Expanding the weights to account for unpacked input exceeds internal buffer
	 * allocated.
	 */
	NRF_AXON_COMPILER_RESULT_CANNOT_GROW_FC_WEIGHTS     = -217,
	/* A reshape between layers occurred that would corrupt the compute output */
	NRF_AXON_COMPILER_RESULT_UNHANDLED_RESHAPE          = -216,
	/* Compiler failed due to output height/width/depth parameter violation. */
	NRF_AXON_COMPILER_RESULT_INVALID_OUTPUT_SIZE        = -215,
	/* Compiler failed due to invalid operation. */
	NRF_AXON_COMPILER_RESULT_INVALID_OPERATION          = -214,
	/* Compiler failed due to invalid byte width on input/filter/output for an operation. */
	NRF_AXON_COMPILER_RESULT_INVALID_BYTE_WIDTH         = -212,
	/* Compiler failed due to a NULL pointer passed to a low-level function. */
	NRF_AXON_COMPILER_RESULT_NULL_PTR                   = -208,
	/* Compiler failed due to internal command buffer overflow. */
	NRF_AXON_COMPILER_RESULT_CMD_BUF_TOO_SMALL          = -207,
	/* generic failure code, check the log message */
	NRF_AXON_COMPILER_RESULT_FAILURE                    = -1,
	/* success */
	NRF_AXON_COMPILER_RESULT_SUCCESS                    = 0,
} nrf_axon_compiler_result_e;


/**
 * @brief Enum used for bytewidth fields
 */
typedef enum {
	NRF_AXON_NN_BYTEWIDTH_1 = 1, /**< 8bits */
	NRF_AXON_NN_BYTEWIDTH_2 = 2, /**< 16bits */
	NRF_AXON_NN_BYTEWIDTH_4 = 4, /**< 32bits */
} nrf_axon_nn_byte_width_e;

/**
 * @brief Supported "fused" activation functions
 */
typedef enum {
	/* no activation function */
	NRF_AXON_NN_ACTIVATION_FUNCTION_DISABLED,
	/*
	 * ReLU. Note that ReLU6 is mapped to ReLU because quantization causes
	 * saturation at 6.
	 */
	NRF_AXON_NN_ACTIVATION_FUNCTION_RELU,
	/*
	 * Produces 32bit, q11.12 output, in preparation for a subsequent softmax operation
	 * that will perform exp() on the input.
	 */
	NRF_AXON_NN_ACTIVATION_FUNCTION_PREPARE_SOFTMAX,
	/* leaky ReLU*/
	NRF_AXON_NN_ACTIVATION_FUNCTION_LEAKY_RELU,
} nrf_axon_nn_activation_function_e;

/**
 * Operations supported by axonpro compiler.
*/
typedef enum {
	NRF_AXON_NN_OP_FULLY_CONNECTED,
	NRF_AXON_NN_OP_CONV2D,
	NRF_AXON_NN_OP_DEPTHWISE_CONV2D,
	NRF_AXON_NN_OP_POINTWISE_CONV2D,
	NRF_AXON_NN_OP_AVERAGE_POOLING,
	NRF_AXON_NN_OP_MAX_POOLING,
	NRF_AXON_NN_OP_ADD2,
	/* special handling for channel padding. */
	NRF_AXON_NN_OP_CHANNEL_PADDING,
	NRF_AXON_NN_OP_PERSISTENT_VAR,
	NRF_AXON_NN_OP_CONCATENATE,
	NRF_AXON_NN_OP_STRIDED_SLICE,
	NRF_AXON_NN_OP_MULTIPLY,
	NRF_AXON_NN_OP_UNIDIRECTIONAL_SEQUENCE_LSTM,
	NRF_AXON_NN_OP_MEAN,
	NRF_AXON_NN_OP_PACK,
	/* Operations implemented as extensions on the CPU*/
	NRF_AXON_NN_OP_FIRST_EXTENSION = 100,
	NRF_AXON_NN_OP_SOFTMAX = NRF_AXON_NN_OP_FIRST_EXTENSION,
	NRF_AXON_NN_OP_SIGMOID,
	NRF_AXON_NN_OP_TANH,
	/*
	 * handles reshapes that require physical movement of data due to difference
	 * between axon and tflite data organization
	 */
	NRF_AXON_NN_OP_RESHAPE,
	NRF_AXON_NN_OP_SPACE_TO_BATCH,
	NRF_AXON_NN_OP_BATCH_TO_SPACE,
	NRF_AXON_NN_OP_RESIZE_NEAREST_NEIGHBOR,
} nrf_axon_nn_op_e;

/**
 * @brief Likely to be removed in a future release.
 */
typedef enum{
	NRF_AXON_NN_CONV2D_SETTING_LOCAL_PSUM,
	NRF_AXON_NN_CONV2D_SETTING_INPUT_INNER_LOOP,
	NRF_AXON_NN_CONV2D_SETTING_INPUT_OUTER_LOOP,
} nrf_axon_nn_conv2d_setting_e;

/**
 * @brief Option for using a dedicated PSUM buffer or using the interlayer buffer for PSUM.
 *
 * PSUM buffer is used for scratch memory by operatoins.
*/
typedef enum {
	/* interlayer buffer is used as psum buffer */
	NRF_AXON_NN_PSUM_BUFFER_PLACEMENT_INTERLAYER_BUFFER,
	/* separate psum buffer. */
	NRF_AXON_NN_PSUM_BUFFER_PLACEMENT_DEDICATED_MEM,
} nrf_axon_nn_psum_buffer_placement_e;

/**
 * @brief To be removed...
 */
typedef enum{
	NRF_AXON_NN_COMPILER_LOG_LEVEL_NOT_USED = 0,
} nrf_axon_nn_compiler_log_level_e;

/**
 * @brief enum for tensor axis.
 * Used by operations like mean, concatenate, strided slice.
 */
typedef enum {
	NRF_AXON_NN_AXIS_CHANNEL,
	NRF_AXON_NN_AXIS_HEIGHT,
	NRF_AXON_NN_AXIS_WIDTH,
	NRF_AXON_NN_AXIS_COUNT,
} nrf_axon_nn_axis_e;

/**
 * @brief describes tensor dimensions
 */
typedef struct {
	uint16_t height;
	uint16_t width;
	uint16_t channel_cnt;
	nrf_axon_nn_byte_width_e byte_width;
	uint16_t batch_cnt;   /**< introduced 1.2.9 */
} nrf_axon_nn_compiler_model_layer_dimensions_s;

/**
 * @brief describes input address offset as a result of split.
 * If a split operation is on the most significant axis with value > 1, the
 * split input can be used for the output. This structure informs
 * the consumer of the split output the offset from the input for
 * the start of its data.
 */
typedef struct {
	/** one of nrf_axon_nn_axis_e */
	uint8_t axis;
	/** offset in elements on the split axis to the start of this layer's input. */
	uint16_t offset;
} nrf_axon_nn_compiler_model_input_split_offset_s;

/**
 * @brief Block shape parameters for space_to_batch and batch_to_space.
 */
typedef struct {
	uint16_t height_size;
	uint16_t width_size;
} nrf_axon_nn_compiler_model_block_shape;
/**
 * @brief parameters for strided slice operation
 */
typedef struct {
	uint32_t begin[NRF_AXON_NN_AXIS_COUNT];
	uint32_t end[NRF_AXON_NN_AXIS_COUNT];
	uint32_t strides[NRF_AXON_NN_AXIS_COUNT];
} nrf_axon_nn_compiler_strided_slice_parameters_s;

/**
 * @brief Interface structure between nn compiler executor and shared library.
 *
 * The executor creates a binary file with an instance of this structure per model layer. Pointers
 * in this structure are populated as offsets within the binary file to the relevant data.
 */
#define NRF_AXON_NN_MAX_LAYER_INPUTS 4
typedef struct {
	/* number of inputs to layer. */
	uint8_t input_id_cnt;
	/*
	 * layer ids of the inputs (input_id is the index into the array of
	 * nrf_axon_nn_model_layer_desc_s). 1st inputid_cnt entries are valid, negative ID
	 * indicates external input.
	 */
	int16_t input_ids[NRF_AXON_NN_MAX_LAYER_INPUTS];
	/* operation this layer performs */
	nrf_axon_nn_op_e nn_operation;
	/* dimensions of the inputs */
	nrf_axon_nn_compiler_model_layer_dimensions_s
		input_dimensions[NRF_AXON_NN_MAX_LAYER_INPUTS];
	/* offsets from the input node's output for this nodes input. */
	nrf_axon_nn_compiler_model_input_split_offset_s
		input_split_offsets[NRF_AXON_NN_MAX_LAYER_INPUTS];
	union {
		/* dimensions of the filter */
		nrf_axon_nn_compiler_model_layer_dimensions_s filter_dimensions;
		/* block shape for batch to/from space */
		nrf_axon_nn_compiler_model_block_shape block_shape;
	};
	/* dimensions of the output. */
	nrf_axon_nn_compiler_model_layer_dimensions_s output_dimensions;
	/* For concatenate only. one of nrf_axon_nn_axis_e */
	uint8_t concatenate_axis;
	/* stride in the width dimension. */
	uint8_t stride_x;
	/* stride in the height dimension. */
	uint8_t stride_y;
	/* filter dilation in the width dimension. */
	uint8_t dilation_x;
	/* filter dilation in the height dimension. */
	uint8_t dilation_y;
	/* input zero point for quantization */
	int8_t input_zero_point;
	/* To support multiple outputs, need output quantization parameters */
	uint32_t output_dequant_multiplier;
	/* To support multiple outputs, need output quantization parameters */
	uint8_t output_dequant_shift;
	/* output zero point for quantization. */
	int8_t output_zero_point;
	/* Bias vector that includes the sum of the weights/filters * input_zero_point */
	union {
		/* populated w/ an offset into the bin file by the executor... */
		uint64_t offset;
		/*...replaced with a resolved pointer by the compiler. */
		int32_t *ptr;
	} bias_prime;
	/*
	 * Quantization multiplier term vector that combines input*filter/weight
	 * scaling factors
	 */
	union {
		/* populated w/ an offset into the bin file by the executor... */
		uint64_t offset;
		/*...replaced with a resolved pointer by the compiler. */
		int32_t *ptr;
	} output_multipliers;
	/* Quantization rounding term vector. */
	union {
		/* populated w/ an offset into the bin file by the executor... */
		uint64_t offset;
		/*...replaced with a resolved pointer by the compiler. */
		int8_t *ptr;
	} scale_shifts;
	/*
	 * 0 if no output quantization, 1 if scale_shift is the same for all channels,
	 * else = output_dimensions.channel_cnt
	 */
	uint16_t scale_shift_cnt;
	/* fused activationn function for layer. */
	nrf_axon_nn_activation_function_e activation_function;
	union {
		 /* convolution, max-pooling, and block_shape padding. */
		struct {
			uint8_t pad_left;
			uint8_t pad_right;
			uint8_t pad_top;
			uint8_t pad_bottom;
		};
		uint32_t unused;
	};
	/* filter/weights/strided_slice parameters. */
	union {
		/* populated w/ an offset into the bin file by the executor... */
		uint64_t offset;
		/*...replaced with a resolved pointer by the compiler. */
		int8_t *ptr;
		nrf_axon_nn_compiler_strided_slice_parameters_s *ss_ptr;
	} filter;
	/*
	 * CPU operations (op extensions) can have additional attributes.
	 * Specifies how large the cpu_op_additional_attributes vector is.
	 */
	uint32_t cpu_op_additional_attributes_count;
	/* Additional CPU operation attributes. */
	union {
		/* populated w/ an offset into the bin file by the executor... */
		uint64_t offset;
		/*...replaced with a resolved pointer by the compiler. */
		int8_t *ptr;
	} cpu_op_additional_attributes;
} nrf_axon_nn_model_layer_desc_s;

/**
 * @brief Additional information conveyed in the bin file from the executor to the compiler.
 */
typedef struct{
	/*
	 * interlayer buffer size limit. If the model requires more than specified,
	 * will display a warning.
	 */
	uint32_t interlayer_buffer_size;
	/*
	 * psum buffer size limit. If the model requires more than specified, will display
	 * a warning.
	 */
	uint32_t psum_buffer_size;
	/* number of test vectors to write to the test_vectors header file. */
	uint32_t header_file_test_vector_cnt;
	/* unused. to be removed */
	nrf_axon_nn_conv2d_setting_e convolution_2d_setting;
	/* unused. to be removed */
	nrf_axon_nn_compiler_log_level_e log_level;
	/* specifies of psum buffer should be part of the interlayer buffer or not. */
	nrf_axon_nn_psum_buffer_placement_e psum_buffer_placement;
} nrf_axon_nn_model_compilation_options_s;

/**
 * @brief structure for conveying the location/size of variable length parameters in the file.
 */
typedef struct {
	uint32_t offset;  /**< offset into the file of the item. */
	uint32_t length;  /**< length (in bytes) of the item. */
} nrf_axon_nn_model_bin_item_s;

/**
 * @brief structure to hold model input/output quantization/dequantization parameters
 *
 * quantized input = (float input) * mult / 2^round + zero_point
 * float output = ((quantized input)-zero_point) *  / 2^^round
 * float = (quantized-zero_point) * 2^round/mult
 */
typedef struct {
	uint32_t mult;
	uint8_t round;
	int8_t zero_point;
} nrf_axon_nn_model_quant_parameters_s;

/**
 * @brief meta data for the model included in the binary file.
 */
typedef struct {
	/*
	 * name of the model as specified to the executor. All file and symbol names
	 * incorporate this name.
	 */
	nrf_axon_nn_model_bin_item_s model_name;
	/* optional text labels for a single dimension output classification model. */
	nrf_axon_nn_model_bin_item_s model_labels;
	/* number of layers (nodes) in the model. */
	uint32_t model_layer_cnt;
	/*
	 * quantization for the model input. All other layers can infer the input quantization
	 * from the output quantization of their inputs.
	 */
	nrf_axon_nn_model_quant_parameters_s input_quant;
	/* dequantization parameters for the model output. */
	nrf_axon_nn_model_quant_parameters_s output_dequant;
} nrf_axon_nn_model_meta_info_s;

/**
 * @brief binary file header structure.
 *
 * This describes where the various components of the model are located in the binary file.
 * It resides at the very beginning of the file.
 */
typedef struct {
	/* expected to be the string "AXON_INTERMEDIATE_REPRESENTATION_FILE" */
	nrf_axon_nn_model_bin_item_s title;
	/* location/size of the bin file version. */
	nrf_axon_nn_model_bin_item_s version;
	/* location/size of nrf_axon_nn_model_meta_info_s */
	nrf_axon_nn_model_bin_item_s meta;
	/* location/size of nrf_axon_nn_model_layer_desc_s[model_layer_cnt] */
	nrf_axon_nn_model_bin_item_s layers;
	/* location size of model constants in the bin file. */
	nrf_axon_nn_model_bin_item_s consts;
	/* location size of nrf_axon_nn_model_compilation_options_s in the bin file. */
	nrf_axon_nn_model_bin_item_s compilation_option;
} nrf_axon_nn_model_desc_hdr_s;

/**
 * @brief structure populated  by the compiler and returned to the executor.
 */
typedef struct {
	/* total size of the model constants (filters/weights/quantization parameters, etc.) */
	uint32_t model_const_buffer_size;
	/* minimum size of the interlayer buffer for model to run. */
	uint32_t interlayer_buffer_size;
	/* minimum size of the psum buffer for model to run. */
	uint32_t psum_buffer_size;
	/* Size of the compiled model code. */
	uint32_t cmd_buffer_size;
	/* Performance estimate for full model inference, in units of system clock cycles. */
	uint32_t profiling_ticks;
} nrf_axon_nn_compiler_return_s;
