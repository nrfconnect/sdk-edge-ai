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
#include <stdarg.h>
#include <stdint.h>
#include "nrf_axon_driver.h"


/**
 * describes the dimensions of an input or output of the model.
 */
typedef struct {
  uint16_t height; 
  uint16_t width;
  uint16_t channel_cnt;
  uint8_t byte_width;
} nrf_axon_nn_model_layer_dimensions_s;

typedef struct {
  int8_t *ptr;         // location of the input
  nrf_axon_nn_model_layer_dimensions_s dimensions;
  uint32_t quant_mult; // quant_input = (float_input * quant_mult)/2^^quant_round + quant_zp. quant_mult/(1>>quant_round) is the inverse quantization scaling factor.
  uint16_t stride; // distance (in bytes) between the start of each row of input.
  uint8_t quant_round; // quantization rounding (in bits).
  int8_t quant_zp; // quantization zero point
  bool is_external;    // populated externally or internal to model
} nrf_axon_nn_compiled_model_input_s;

/**
 * persistent variables are feedback elements in a model. They are placed in a dedicated buffer
 * rather than the interlayer buffer so that they persist between inferences of the model, and
 * are not corrupted by other models.
 */
typedef struct {
  int8_t *buf_ptr;
  uint32_t buf_size; // in bytes
  int32_t initial_value; // value to initialize buffer to
  uint8_t byte_width;
} nrf_axon_nn_model_persistent_var_s;

/**
 * compiled model structure output by the nn compiler.
*/
#define NRF_AXON_NN_MAX_MODEL_INPUTS 2
typedef struct {
  uint32_t compiler_version;        /**< version of the compiler that generated the model. bits 23:16 => major, bits 15:8 => minor, bits 8:0 => patch */
  const char* model_name;
  const char** labels;
  nrf_axon_nn_compiled_model_input_s inputs[NRF_AXON_NN_MAX_MODEL_INPUTS];
  uint8_t input_cnt;                 /**< number of valid inputs[] */
  int8_t external_input_ndx;         /**< index into inputs[] of the external input (the 1 that needs to be populated before inference). */
  int8_t *output_ptr;                /**< location in interlayer buffer that model output resides, unpacked*/
  int8_t *packed_output_buf;         /**< optional dedicated output that output can be packed and copied to */
  uint32_t interlayer_buffer_needed; /* amount of interlayer buffer used by this model. Must be checked against available buffer size. */
  uint32_t psum_buffer_needed; // amount of psum buffer used by this model. Must be checked against available buffer size.
  const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* cmd_buffer_ptr;
  const void *model_const_ptr;  // location of all the model constants
  uint32_t model_const_size; // size of the model constants, in bytes.
  uint32_t cmd_buffer_len;
  struct {
    int8_t *buf_ptr;  // persistent vars are stored in a contiguous block starting here.
    uint32_t buf_size; // total combined size of persistent vars
    const nrf_axon_nn_model_persistent_var_s *vars; // list of info for each persistent var
    uint16_t count;   // number of persistent vars.
  } persistent_vars; 
  nrf_axon_nn_model_layer_dimensions_s output_dimensions;
  
  uint32_t output_dequant_mult; /**< float_output = (quant_output - output_dequant_zp) * output_dequant_mult)/2^^output_dequant_round. output_dequant_mult)/(1>>output_dequant_round) is the output quantization scaling factor. */
  uint8_t output_dequant_round;
  int8_t output_dequant_zp;

  uint16_t output_stride; /* length in bytes of the distance between the start of rows in the output. Ceil(width * bytewdith, 4) */
  bool is_layer_model; /* layer model is a subset of the full model. If true, this can be treated as a nrf_axon_nn_compiled_model_layer_s.*/
} nrf_axon_nn_compiled_model_s;

/**
 * @brief returns the index of the model's input that is external.
 * 
 * The 1st layer in a model can have multiple inputs; one that is internal (ie, VarHandle for streaming models), and
 * at 1 that is external. This will find which input needs to be populated explicitly before inference begins.
 */
int8_t nrf_axon_nn_model_1st_external_input_ndx(const nrf_axon_nn_compiled_model_s* the_model);

const nrf_axon_nn_compiled_model_input_s *nrf_axon_nn_model_1st_external_input(const nrf_axon_nn_compiled_model_s* the_model);


/**
 * @brief Sanity check of a compiled model.
 * 
 * Verifies basic validity of the compiled model. 
 * Verifies that interlayer and psum buffers are large enough to accomodate the model.
 * Should be called at start-up for each model in the application.
 * 
 * @note
 * If asynchronous inference is performed this function is called as part of nrf_axon_nn_model_async_init.
 * 
 * @param[in] compiled_model The compiled model to check.
 * @retval 0 on success or a negative error code. 
 */
nrf_axon_result_e nrf_axon_nn_model_validate(const nrf_axon_nn_compiled_model_s *compiled_model);


/**
 * @brief 
 * Initialize all the persistent var buffers in a streaming-style model (with VarHandle/ReadVariable/AssignVariable). 
 * Should be called at the start of each streaming session.
 * Harmless to call for non-streaming style models.
 * @param[in] the_model Model instance initialized through nrf_axon_nn_model_init.
 * @retval 0 on success or a negative error code.
 */
int nrf_axon_nn_model_init_vars(const nrf_axon_nn_compiled_model_s *compiled_model);

/**
 * @brief Blocking inference function of a compiled model.
 *  
 * - Reserves Axon for its exclusive access (using nrf_axon_platform_reserve_for_user()).
 * - Copies the input_vector to its location in the interlayer buffer. This input_vector is assumed to be packed in memory. (optional)
 * - Performs the inference.
 * - copies and packs unpacked output from interlayer buffer to output_buffer (optional)
 * - Frees Axon for other users.
 * - Returns to user.
 * 
 * If input_vector is NULL, it is assumed that the input is already placed in the interlayer buffer.
 * 
 * if output_buffer is NULL, results can be retrieved directly from compiled_model.output_ptr or by using functions
 * nrf_axon_nn_get_classification() and nrf_axon_nn_copy_output_to_packed_buffer(). 
 * 
 * Passing either input_vector and/or output_buffer as NULL is only safe if no other threads 
 * are using Axon and there are no asynchronous users. 
 * The interlayer buffer is not locked prior to entry to nor after return from this function.
 * 
 * Cannot be called from interrupt context.
 * 
 * @param[in] compiled_model The compiled model to perform inference on.
 * @param[in] input_vector Packed input that should be copied to the model's input_ptr prior to inference. Can be NULL.
 * @param[out] output_buffer Buffer to hold the packed output result. Can be NULL.
 * @retval 0 on success, are a negative error code.
 */
nrf_axon_result_e nrf_axon_nn_model_infer_sync(
  const nrf_axon_nn_compiled_model_s *compiled_model,
  const int8_t* input_vector,
  int8_t* output_buffer);

/**
 * 
 */
typedef enum {
  kAxonnnInferStatusIdle,
  kAxonnnInferStatusInferring,
  kAxonnnInferStatusInferenceComplete,
} nrf_axon_nn_async_inference_status_e;

/**
 * @brief Combines the compiled model info with other structures to support asynchronous inferencing.
 */
typedef struct {
  const nrf_axon_nn_compiled_model_s *compiled_model;  /**< user populates this with the pointer to the compiled model. */
  nrf_axon_cmd_buffer_info_s cmd_buf_info;          /**< populated by calling nrf_axon_nn_model_init, and managed by the driver.*/
  nrf_axon_queued_cmd_info_wrapper_s queued_cmd_buf_wrapper; /**< populated by and managed by the driver. */
  void (*inference_callback)(nrf_axon_result_e result, void* callback_context); /**< user callback function. For asynchronous mode only. */
  void* callback_context;                           /**< Passthrough parameter to the user callback function. For asynchronous mode only. */
  nrf_axon_nn_async_inference_status_e infer_status;           /**< Indicates completion status of infernce. For asynchronous mode only. Use nrf_axon_nn_get_model_async_infer_status to read value. */
} nrf_axon_nn_model_async_inference_wrapper_s;

/**
 * @brief Initialize a model for asynchronous inference.
 * Calls nrf_axon_nn_model_validate then binds the model wrapper structure to its compiled model and performs some data initialization.
 * Model is ready to be inferred upon completion.
 * Called once per model at start-up.
 * 
 * @param[out] the_model Allocated nrf_axon_nn_model_inference_wrapper_s instance in static (non-stack) memory that will be passed to inference functions.
 * @param[in] compiled_model Pointer to compiled model that will be bound to the_model.
 * @retval 0 on success or a negative error code.
*/
nrf_axon_result_e nrf_axon_nn_model_async_init(
  nrf_axon_nn_model_async_inference_wrapper_s *model_wrapper, 
  const nrf_axon_nn_compiled_model_s *compiled_model);

/**
 * @brief Returns the model inference status of an asynchronous inference.
 */
nrf_axon_nn_async_inference_status_e nrf_axon_nn_get_model_async_infer_status(const nrf_axon_nn_model_async_inference_wrapper_s* model_wrapper);

/*
 * Starts an asynchronous inference on the provided vector.
 * In asynchonous mode, models are inferred in a separate thread, one after another.
 * User provides input vector and output buffer information as the interlayer buffer 
 * is used by all models, so only when it is this model's turn to execute can its
 * input be populated from the input_vector.
 * 
 * Upon completion of inference the next job is queued before the user callback is invoked,
 * so the results have to be copied by the driver to the output_buffer before invoking the user callback.
 * 
 * axon will insert padding bytes at the end of each row of output up to the next 32bit boundary. 
 * These padding bytes will be stripped out from the output_buffer copy. The difference between
 * output_width_in_bytes and output_stride is the amount of padding at the end of each row.
 */
nrf_axon_result_e nrf_axon_nn_model_infer_async(
  nrf_axon_nn_model_async_inference_wrapper_s* model_wrapper,
  const int8_t* input_vector, 
  int8_t *output_buffer,
  void (*inference_callback)(nrf_axon_result_e result, void* callback_context), // user call-back function
  void* callback_context);// provided by the inference caller to be passed to the inference_callback() blindly

/**
 * Gets the inference results for a classification model.
 * Returns the infernce index, populates optional parameters label, score, and profiling_ticks if not NULL.
 * Only valid for single dimension, classification type models.
 */
int16_t nrf_axon_nn_get_classification(
  const nrf_axon_nn_compiled_model_s *compiled_model, 
  const int8_t *packed_output, 
  const char** label, int32_t* score);

/**
 * @brief Copies model input from input_vector to the location in the interlayer buffer the model expects.
 * 
 * It is not recommended for users to invoke this function directly. The inference APIs handle copying the input vector
 * in a safe manner that do not risk corrupting the current or future inferences.
 * 
 * This function can be called safely in synchronous mode if Axon is reserved by the user (exactly what the synchronous inference
 * function does). This function cannot be called safely in asynchronous inference mode unless the caller knows a-priori that
 * no other inferences are occurring or will occur (ie, a simple one model system).
 */
nrf_axon_result_e nrf_axon_nn_populate_input_vector(const nrf_axon_nn_compiled_model_s *compiled_model, const int8_t* input_vector);

/**
 * @brief Copies and packs the model inference output from the common interlayer buffer to the users dedicated buffer to_buffer.
 * 
 * It is not recommended for users to invoke this function directly. The inference APIs handle copying the output results
 * in a safe manner that do not risk corrupting the current or future inferences.
 * 
 * This function cannot be called safely in synchronous or asynchronous inference modes unless the caller knows a-priori that
 * no other inferences are occurring or will occur (ie, a simple one model system).
*/
void nrf_axon_nn_copy_output_to_packed_buffer(
  const nrf_axon_nn_compiled_model_s *compiled_model, 
  void *to_buffer);

#ifdef __cplusplus
} // extern "C" {
#endif

