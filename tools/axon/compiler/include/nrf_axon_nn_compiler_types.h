/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */

/*
  File used by the cross compiler to generate the binary file.
  The file acts as an interface file for the cross compiler and has the common definitions used by the axon compiler and the tflite cross compiler
  
  This file will be referenced using its direct path by the cross compiler.
  The file needs to satisfy the following conditions to ensure that it is valid and can be used an interface header file 
  Condition 1 : The file should not have any #include directories
  Condition 2 : Macros defined in the file should have fixed integer values
                for example #define AXONPRO_MACRO (1) and not #define AXONPRO_MACRO(X) (X<<2)
  Condition 3 : All the typedefs should be contained within this file and shold not have references outside the file
  Condition 4 : It should not contain any #ifdef macros

*/
#pragma once

typedef enum {
  NRF_AXON_COMPILER_RESULT_INVALID_PADDING            = -238, /**< Compiler failed due to invalid padding parameters. */
  NRF_AXON_COMPILER_RESULT_INVALID_STRIDE             = -236, /**< Compiler failed due to invalid stride parameters. */
  NRF_AXON_COMPILER_RESULT_INVALID_INPUT              = -235, /**< Compiler failed due to too many or not enough inputs to a layer. */
  NRF_AXON_COMPILER_RESULT_INVALID_FILTER_COUNT       = -234, /**< Compiler failed due to an invalid number filters (output channels) */
  NRF_AXON_COMPILER_RESULT_PSUM_BUFFER_TOO_SMALL      = -232, /**< Compiler failed due to internal, oversized PSUM buffer being too small for the model. */
  NRF_AXON_COMPILER_RESULT_INVALID_ACTIVATION_FUNCTION= -228, /**< Compiler failed due to specified activation function not being a member of nrf_axon_nn_activation_function_e. */
  NRF_AXON_COMPILER_RESULT_INFERENCE_FAILED           = -227, /**< Compiler failed during simulator test inference. */
  NRF_AXON_COMPILER_RESULT_SIMULATOR_INIT_FAILED      = -226, /**< Compiler failed initializing the simulator for test inference. */
  NRF_AXON_COMPILER_RESULT_MODEL_INIT_FAILED          = -225, /**< Compiler failed initializing model for simulator test inference. */
  NRF_AXON_COMPILER_RESULT_UNRESOLVED_ADDRESS         = -224, /**< Compiler failed due the model containing an address to a symbol that was not added to the symbol list. */
  NRF_AXON_COMPILER_RESULT_CORRUPT_CMD_BUFFER         = -223, /**< Compiler failed due to internal error. */
  NRF_AXON_COMPILER_RESULT_INTERLAYER_BUFFER_TOO_SMALL= -222,/**< Compiler failed due to oversized interlayer buffer being too small for the model. */
  NRF_AXON_COMPILER_RESULT_TOO_MANY_DECLARED_BUFFERS  = -221, /**< Compiler failed due to number of declared buffers exceeding the internal limit. */
  NRF_AXON_COMPILER_RESULT_MALLOC_FAILED              = -220, /**< compiler could not allocate enough memory */
  NRF_AXON_COMPILER_RESULT_INVALID_BIN_FILE           = -219, /**< Error occurred in compiler due to invalid bin file. Version mismatch or file corruption. */
  NRF_AXON_COMPILER_RESULT_FILE_IO_FAILED             = -218, /**< Error occurred in compiler while performing file i/o */
  NRF_AXON_COMPILER_RESULT_CANNOT_GROW_FC_WEIGHTS     = -217, /**< Expanding the weights to account for unpacked input exceeds internal buffer allocated. */
  NRF_AXON_COMPILER_RESULT_UNHANDLED_RESHAPE          = -216, /**< A reshape between layers occurred that would corrupt the compute output */
  NRF_AXON_COMPILER_RESULT_INVALID_INPUT_SIZE         = -240, /**< Compiler failed due to input height/width/depth violation.*/
  NRF_AXON_COMPILER_RESULT_INVALID_FILTER_SIZE        = -239, /**< Compiler failed due to filter height/width/depth parameter violation. */
  NRF_AXON_COMPILER_RESULT_INVALID_OUTPUT_SIZE        = -215, /**< Compiler failed due to output height/width/depth parameter violation. */
  NRF_AXON_COMPILER_RESULT_INVALID_OPERATION          = -214, /**< Compiler failed due to invalid operation. */
  NRF_AXON_COMPILER_RESULT_INVALID_BYTE_WIDTH         = -212, /**< Compiler failed due to invalid byte width on input/filter/output for an operation. */
  NRF_AXON_COMPILER_RESULT_NULL_PTR                   = -208, /**< Compiler failed due to a NULL pointer passed to a low-level function. */
  NRF_AXON_COMPILER_RESULT_CMD_BUF_TOO_SMALL          = -207, /**< Compiler failed due to internal command buffer overflow. */
  NRF_AXON_COMPILER_RESULT_FAILURE                    = -1,   /**< generic failure code */
  NRF_AXON_COMPILER_RESULT_SUCCESS                    = 0,    /**< success */
} nrf_axon_compiler_result_e;


typedef enum {
  NRF_AXON_NN_BYTEWIDTH_1 = 1,
  NRF_AXON_NN_BYTEWIDTH_2 = 2,
  NRF_AXON_NN_BYTEWIDTH_4 = 4,
} nrf_axon_nn_byte_width_e;

typedef enum {
  NRF_AXON_NN_ACTIVATION_FUNCTION_DISABLED,
  NRF_AXON_NN_ACTIVATION_FUNCTION_RELU,
  NRF_AXON_NN_ACTIVATION_FUNCTION_PREPARE_SOFTMAX,
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
  NRF_AXON_NN_OP_ADD2, // adds 2 quantized input vectors
  NRF_AXON_NN_OP_CHANNEL_PADDING,
  NRF_AXON_NN_OP_PERSISTENT_VAR,
  NRF_AXON_NN_OP_CONCATENATE,
  NRF_AXON_NN_OP_STRIDED_SLICE,
  NRF_AXON_NN_OP_MULTIPLY,
  NRF_AXON_NN_OP_MEAN,
  NRF_AXON_NN_OP_FIRST_EXTENSION=100,//
  NRF_AXON_NN_OP_SOFTMAX=NRF_AXON_NN_OP_FIRST_EXTENSION, // Softmax implemented as an op extension
  NRF_AXON_NN_OP_SIGMOID, 
  NRF_AXON_NN_OP_TANH,
} nrf_axon_nn_op_e;

typedef enum{
  NRF_AXON_NN_CONV2D_SETTING_LOCAL_PSUM,
  NRF_AXON_NN_CONV2D_SETTING_INPUT_INNER_LOOP, //(lower psum memory requirement)
  NRF_AXON_NN_CONV2D_SETTING_INPUT_OUTER_LOOP, //(higher psum memory requirement)
} nrf_axon_nn_conv2d_setting_e;

/*
 * Below only applies for 2D Convolutions not using NRF_AXON_NN_CONV2D_SETTING_LOCAL_PSUM
*/
typedef enum {
  NRF_AXON_NN_PSUM_BUFFER_PLACEMENT_INTERLAYER_BUFFER, // interlayer buffer is used as psum buffer
  NRF_AXON_NN_PSUM_BUFFER_PLACEMENT_DEDICATED_MEM, // separate psum buffer.
} nrf_axon_nn_psum_buffer_placement_e;

typedef enum{ //Matching values with python logger object
  NRF_AXON_NN_COMPILER_LOG_LEVEL_NOT_USED=0,
} nrf_axon_nn_compiler_log_level_e;

typedef enum { //tensor axis. ignore batch/filter_cnt
  NRF_AXON_NN_AXIS_CHANNEL,
  NRF_AXON_NN_AXIS_HEIGHT,
  NRF_AXON_NN_AXIS_WIDTH,
  NRF_AXON_NN_AXIS_COUNT,
} nrf_axon_nn_axis_e;

typedef struct {
  uint16_t height;
  uint16_t width;
  uint16_t channel_cnt;
  nrf_axon_nn_byte_width_e byte_width;
} nrf_axon_nn_compiler_model_layer_dimensions_s;

typedef struct {
  uint32_t begin[NRF_AXON_NN_AXIS_COUNT];
  uint32_t end[NRF_AXON_NN_AXIS_COUNT];
  uint32_t strides[NRF_AXON_NN_AXIS_COUNT];
} nrf_axon_nn_compiler_strided_slice_parameters_s;

/**
 * Interface structure between nn compiler executor and shared library.
 * 
 * The executor creates a binary file with an instance of this structure per model layer. Pointers
 * in this structure are populated as offsets within the binary file to the relevant data.
 * 
 * The compiled shared library reads in the binary, then resolves all the offsets to absolute addresses.
 * This structure is then passed to the layer compilation function.
 */
#define NRF_AXON_NN_MAX_LAYER_INPUTS 4
typedef struct {
  uint8_t input_id_cnt; // number of inputs to layer.
  int16_t input_ids[NRF_AXON_NN_MAX_LAYER_INPUTS];  // layer ids of the inputs. 1st inputid_cnt entries are valid, negative ID indicates external input.
  nrf_axon_nn_op_e nn_operation;
  nrf_axon_nn_compiler_model_layer_dimensions_s input_dimensions[NRF_AXON_NN_MAX_LAYER_INPUTS];
  nrf_axon_nn_compiler_model_layer_dimensions_s filter_dimensions;
  nrf_axon_nn_compiler_model_layer_dimensions_s output_dimensions;
  uint8_t concatenate_axis; // one of nrf_axon_nn_axis_e
  uint8_t stride_x;
  uint8_t stride_y;
  uint8_t dilation_x;
  uint8_t dilation_y;
  int8_t input_zero_point;
  int8_t output_zero_point;
  union {
    uint64_t offset; // populated w/ an offset into the bin file by the executor...
    int32_t* ptr; // ...replaced with a resolved pointer by the compiler
  } bias_prime;
  union {
    uint64_t offset; // populated w/ an offset into the bin file by the executor...
    int32_t* ptr; // ...replaced with a resolved pointer by the compiler
  } output_multipliers;
    union {
    uint64_t offset; // populated w/ an offset into the bin file by the executor...
    int8_t* ptr; // ...replaced with a resolved pointer by the compiler
  } scale_shifts;
  uint16_t scale_shift_cnt; // 0 if no output quantization, 1 if scale_shift is the same for all channels, else axonpro_model_layer_dimensions.channel_cnt
  nrf_axon_nn_activation_function_e activation_function;
  union {
    struct { // populated for conv and depthwise conv
      uint8_t pad_left;
      uint8_t pad_right;
      uint8_t pad_top;
      uint8_t pad_bottom;
    };
    uint32_t unused;
  };
  union {
    uint64_t offset; // populated w/ an offset into the bin file by the executor...
    int8_t* ptr; // ...replaced with a resolved pointer by the compiler. Note: see filter_dimensions.byte_width to determine the size of filter elements.
    nrf_axon_nn_compiler_strided_slice_parameters_s *ss_ptr; // for strided slice layers only.
  } filter;
  uint32_t cpu_op_additional_attributes_count;
  union {
    uint64_t offset; // populated w/ an offset into the bin file by the executor...
    int8_t* ptr; // ...replaced with a resolved pointer by the compiler
  } cpu_op_additional_attributes;
} nrf_axon_nn_model_layer_desc_s;

typedef struct{
  uint32_t interlayer_buffer_size;
  uint32_t psum_buffer_size;
  uint32_t header_file_test_vector_cnt;
  nrf_axon_nn_conv2d_setting_e convolution_2d_setting;
  nrf_axon_nn_compiler_log_level_e log_level;
  nrf_axon_nn_psum_buffer_placement_e psum_buffer_placement;
} nrf_axon_nn_model_compilation_options_s;//NOTE : Must be padded if not ending at 32 bit boundary!

typedef struct {
  uint32_t offset;
  uint32_t length;
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
} nrf_axon_nn_model_quant_paramters_s;

typedef struct {
  nrf_axon_nn_model_bin_item_s model_name;  
  nrf_axon_nn_model_bin_item_s model_labels;
  uint32_t model_layer_cnt;
  nrf_axon_nn_model_quant_paramters_s input_quant;
  nrf_axon_nn_model_quant_paramters_s output_dequant;
} nrf_axon_nn_model_meta_info_s;

typedef struct {
  nrf_axon_nn_model_bin_item_s title;
  nrf_axon_nn_model_bin_item_s version;
  nrf_axon_nn_model_bin_item_s meta;
  nrf_axon_nn_model_bin_item_s layers;  
  nrf_axon_nn_model_bin_item_s consts;
  nrf_axon_nn_model_bin_item_s compilation_option;
} nrf_axon_nn_model_desc_hdr_s;

typedef struct {
  uint32_t model_const_buffer_size;
  uint32_t interlayer_buffer_size;
  uint32_t psum_buffer_size;
  uint32_t cmd_buffer_size;
  uint32_t profiling_ticks;
} nrf_axon_nn_compiler_return_s;