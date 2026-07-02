/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
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

/**
 * Describes 1 input to a model.
 */
typedef struct {
	/**< location of the input during inference. */
	int8_t *ptr;
	/**< dimensions of the input */
	nrf_axon_nn_model_layer_dimensions_s dimensions;
	/**< quantization parameter (multiplier):
	 * quantized_input = (float_input * quant_mult)/ 2^^quant_round + quant_zp.
	 * quant_mult/(1>>quant_round) is the inverse quantization scaling factor.
	 */
	uint32_t quant_mult;
	/**< distance (in bytes) between the start of each row of input. */
	uint16_t stride;
	/**< quantization parameter (rounding bits) */
	uint8_t quant_round;
	/**< quantization parameter (zero point) */
	int8_t quant_zp;
	/**< populated externally or internal to model (from another layer or persistent variable)
	 */
	bool is_external;
} nrf_axon_nn_compiled_model_input_s;

/**
 * persistent variables are feedback elements in a model. They are placed in a dedicated buffer
 * rather than the interlayer buffer so that they persist between inferences of the model, and
 * are not corrupted by other models.
 */
typedef struct {
	/**< pointer to the buffer. The buffer is declared by the compiled model. */
	int8_t *buf_ptr;
	/**< size in bytes of the buffer. */
	uint32_t buf_size;
	/**< buffer initialization value (quantization zero point for the input) */
	int32_t initial_value;
	/**< data width in bytes of the persistent variable. */
	uint8_t byte_width;
} nrf_axon_nn_model_persistent_var_s;

/**
 * Beginning with compiler version 1.1.1, multiple output tensors are supported.
 *
 * This structure describes their dimensions and dequantizaton parameters.
 * If a dedicated buffer is allocated to hold the outputs, the packed outputs will be stored
 * in order of declaration. Each node's output will begin on a 32bit boundary, so there can be up to
 * 3bytes of padding between the end of the prior node's output and the beginning of the next node's
 * output.
 */
typedef struct nrf_axon_compiled_model_output_tag_s {
	/**< address in memory where this output is placed in its raw form. */
	int8_t *ptr;
	nrf_axon_nn_model_layer_dimensions_s dimensions;
	/**< dequantization :
	 * float_output = (quant_output - output_dequant_zp) * output_dequant_mult /
	 *                2^^output_dequant_round
	 * output_dequant_mult)/(1>>output_dequant_round) is the output quantization
	 * scaling factor.
	 */
	/**< dequantization multiplier. */
	uint32_t dequant_mult;
	/**< dequantization rounding bits. */
	uint8_t dequant_round;
	/**< dequantization zero point. */
	int8_t dequant_zp;
	/**< length in bytes of the distance between the start of rows in the unpacked
	 * output in the interlayer byffer.
	 */
	uint16_t stride;
} nrf_axon_compiled_model_output_s;

/**
 * Compiled model structure output by the nn compiler.
 * Most of the fields are consumed by driver APIs; users do not need to access them directly.
*/
#define NRF_AXON_NN_MAX_MODEL_INPUTS 2
typedef struct nrf_axon_nn_compiled_model_tag_s  {
	/**< version of the compiler that generated the model. bits 23:16 => major,
	 * bits 15:8 => minor, bits 8:0 => patch
	 */
	uint32_t compiler_version;
	/**< name of the model provided by user at compilation time.*/
	const char *model_name;
	/**< optional list of text labels the correspond to classification indices.
	 * Applies to single dimension classification models only.
	 */
	const char **labels;
	/**< Inputs to the model. */
	nrf_axon_nn_compiled_model_input_s inputs[NRF_AXON_NN_MAX_MODEL_INPUTS];
	/**< number of valid inputs[] */
	uint8_t input_cnt;
	/**< index into inputs[] of the external input (the 1 that needs to
	 * be populated before inference).
	 */
	int8_t external_input_ndx;
	/**< location in interlayer buffer that model output resides, unpacked,
	 * for output ndx 0.
	 */
	int8_t *output_ptr;
	/**< optional dedicated buffer that output can be packed and copied to by the driver,
	 * after inference completes.
	 */
	int8_t *packed_output_buf;
	/**< amount of interlayer buffer used by this model. Is checked against available buffer
	 * size by nrf_axon_nn_model_validate().
	 */
	uint32_t interlayer_buffer_needed;
	/**< amount of psum buffer used by this model. Must be checked against available
	 * buffer size by nrf_axon_nn_model_validate().
	 */
	uint32_t psum_buffer_needed;
	/**< compiled axon machine code for the model. */
	const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *cmd_buffer_ptr;
	/**< location in memory of all the model constants. */
	const void *model_const_ptr;
	/**< size of the model constants, in bytes. */
	uint32_t model_const_size;
	/**< length of the command buffer, in elements. */
	uint32_t cmd_buffer_len;
	/** listing of persistent variables that need to be initialized one time
	 * per session by calling nrf_axon_nn_model_init_vars().
	 */
	struct {
		/**< persistent vars are stored in a contiguous block starting here. */
		int8_t *buf_ptr;
		/**<  total combined size of persistent vars in bytes. */
		uint32_t buf_size;
		/**<  list of info for each persistent var */
		const nrf_axon_nn_model_persistent_var_s *vars;
		/**<  number of persistent vars. */
		uint16_t count;
	} persistent_vars;
	/**< Describes the dimensions of the model output. */
	nrf_axon_nn_model_layer_dimensions_s output_dimensions;

	/**< dequantization :
	 * float_output = (quant_output - output_dequant_zp) * output_dequant_mult /
	 *                 2^^output_dequant_round
	 * output_dequant_mult)/(1>>output_dequant_round) is the output quantization
	 * scaling factor.
	 */
	/**< dequantization multiplier for output ndx 0. */
	uint32_t output_dequant_mult;
	/**< dequantization rounding bits for output ndx 0. */
	uint8_t output_dequant_round;
	/**< dequantization zero point for output ndx 0. */
	int8_t output_dequant_zp;
	/**< length in bytes of the distance between the start of rows in the unpacked output
	 * in the interlayer byffer for output ndx 0.
	 */
	uint16_t output_stride;
	/**< layer model is a superset of the full model. If true, this can be treated as a
	 * nrf_axon_nn_compiled_model_layer_s.
	 */
	bool is_layer_model;
	/**
	 * introduced with compiler version 1.2.0, multiple output tensors are supported.
	 * Verify compiler_version >= 0x10101 before accessing these fields.
	 */
	/**< number of additional model outputs pointed to by extra_outputs */
	uint16_t extra_output_cnt;
	/**< pointer to the additional outputs of the model */
	const nrf_axon_compiled_model_output_s *extra_outputs;
	/**< model requires this version of the driver or later to execute properly. */
	uint32_t min_driver_version_required;
} nrf_axon_nn_compiled_model_s;

#define NRF_AXON_COMPILED_MODEL_EXTRA_OUTPUT_CNT(compiled_model) \
	(compiled_model->compiler_version >= 0x10200 ? compiled_model->extra_output_cnt : 0)

/**
 * @brief returns the nrf_axon_nn_compiled_model_input_s instance of the model's input that is
 * external.
 *
 * The 1st layer in a model can have multiple inputs; one that is internal (ie, VarHandle for
 * streaming models), and 1 that is external. This will find which input needs to be populated
 * explicitly before inference begins.
 *
 * @param[in] the_model Model to find the external input for.
 * @retval index into the_model->inputs[] for the model's external input.
 */
int8_t nrf_axon_nn_model_1st_external_input_ndx(
	const nrf_axon_nn_compiled_model_s *the_model);

/**
 * @brief returns the index of the model's input that is external.
 *
 * The 1st layer in a model can have multiple inputs; one that is internal (ie, VarHandle for
 * streaming models), and 1 that is external. This will find which input needs to be populated
 * explicitly before inference begins.
 *
 * @param[in] the_model Model to find the external input for.
 * @retval instance in the_model->inputs[] of the model's external input.
 */
const nrf_axon_nn_compiled_model_input_s *nrf_axon_nn_model_1st_external_input(
	const nrf_axon_nn_compiled_model_s *the_model);

/**
 * @brief Sanity check of a compiled model.
 *
 * Verifies basic validity of the compiled model.
 * Verifies that interlayer and psum buffers are large enough to accomodate the model.
 * Should be called at start-up for each model in the application.
 *
 * @note
 * If asynchronous inference is performed this function is called as part of
 * nrf_axon_nn_model_async_init().
 *
 * @param[in] compiled_model The compiled model to check.
 * @retval 0 on success or a negative error code.
 */
nrf_axon_result_e nrf_axon_nn_model_validate(
	const nrf_axon_nn_compiled_model_s *compiled_model);


/**
 * @brief
 * Initialize all the persistent var buffers in a streaming-style model
 * (with VarHandle/ReadVariable/AssignVariable).
 * Should be called at the start of each streaming session.
 * Harmless to call for non-streaming style models.
 * @param[in] compiled_model Model to operate on.
 * @retval 0 on success or a negative error code.
 */
int nrf_axon_nn_model_init_vars(
	const nrf_axon_nn_compiled_model_s *compiled_model);

/**
 * @brief Blocking inference function of a compiled model.
 *
 * - Reserves Axon for its exclusive access (using nrf_axon_platform_reserve_for_user()).
 * - Copies the input_vector to its location in the interlayer buffer. This input_vector is assumed
 *   to be packed in memory.
 * - Performs the inference.
 * - copies and packs unpacked output from interlayer buffer to output_buffer (optional)
 * - Frees Axon for other users.
 * - Returns to user.
 *
 * If input_vector is NULL, the user has to copy the model input to the correct location in the
 * interlayer buffer (specified by the compiled_model). In a dynamic system where there are other
 * threads utilizing axon and/or asynchronous inference is occurring, the user must 1st call
 * nrf_axon_platform_reserve_for_user() prior to accessing the interlayer_buffer, then invoke this
 * function.
 *
 * Similarly, output_buffer cannot be NULL in a dynamic system as there is no way to retain control
 * of axon after inference completes.
 *
 * Cannot be called from interrupt context.
 *
 * @param[in] compiled_model The compiled model to perform inference on that has been validated by
 *            calling nrf_axon_nn_model_validate(compiled_model).
 * @param[in] input_vector Packed input that is copied to the model's input_ptr prior to inference.
 *            Can be NULL (see above).
 * @param[out] output_buffer Buffer to hold the packed output result. Can be NULL (see above).
 * @retval 0 on success, are a negative error code.
 */
nrf_axon_result_e nrf_axon_nn_model_infer_sync(
	const nrf_axon_nn_compiled_model_s *compiled_model,
	const int8_t *input_vector,
	int8_t *output_buffer);


/**
 * @brief Asynchronous inference states
 *
 * These states only apply to asynchronous inference.
 */
typedef enum {
	/**< Initial async inference state, set by nrf_axon_nn_model_async_init(). */
	NRF_AXON_NN_ASYNC_INFERENCE_STATUS_IDLE,
	/**< Model has been submitted to the async inference queue. */
	NRF_AXON_NN_ASYNC_INFERENCE_STATUS_ACTIVE,
	/**< Model inference has completed. */
	NRF_AXON_NN_ASYNC_INFERENCE_STATUS_COMPLETE,
} nrf_axon_nn_async_inference_status_e;

/**
 * @brief Combines the compiled model info with other structures to support asynchronous
 * inferencing.
 * Users should not access these fields directly; they are managed by the driver.
 */
typedef struct {
	/**< Reference to the compiled model structure. */
	const nrf_axon_nn_compiled_model_s *compiled_model;
	/**< populated by calling nrf_axon_nn_model_init. */
	nrf_axon_cmd_buffer_info_s cmd_buf_info;
	/**< populated by and managed by the driver. */
	nrf_axon_queued_cmd_info_wrapper_s queued_cmd_buf_wrapper;
	/**< user callback function.*/
	void (*inference_callback)(nrf_axon_result_e result, void *callback_context);
	/**< Passthrough parameter to the user callback function. */
	void *callback_context;
	/**< populated and managed by the driver. dedicated buffer outside the interlayer buffer for
	 * storing model output.
	 */
	int8_t *output_buffer;
	/**< Indicates completion status of infernce. Use nrf_axon_nn_get_model_async_infer_status
	 * to read value.
	 */
	nrf_axon_nn_async_inference_status_e infer_status;
} nrf_axon_nn_model_async_inference_wrapper_s;

/**
 * @brief Initialize a model for asynchronous inference.
 * Calls nrf_axon_nn_model_validate then binds the model wrapper structure to its compiled model
 * and performs some data initialization.
 * Model is ready to be inferred upon completion.
 * Called once per model at start-up.
 *
 * @param[out] the_model Allocated nrf_axon_nn_model_inference_wrapper_s instance in static
 *             (non-stack) memory that will be passed to inference functions.
 * @param[in] compiled_model Pointer to compiled model that will be bound to the_model.
 * @retval 0 on success or a negative error code.
*/
nrf_axon_result_e nrf_axon_nn_model_async_init(
	nrf_axon_nn_model_async_inference_wrapper_s *model_wrapper,
	const nrf_axon_nn_compiled_model_s *compiled_model);

/**
 * @brief Returns the model inference status of an asynchronous inference.
 *
 * @param[in] model_wrapper model that has been initialized via nrf_axon_nn_model_async_init()
 * @retval enum indicating current status.
 */
nrf_axon_nn_async_inference_status_e nrf_axon_nn_get_model_async_infer_status(
	const nrf_axon_nn_model_async_inference_wrapper_s *model_wrapper);

/*
 * @brief Starts an asynchronous inference on the provided model.

 * In asynchonous mode, models are inferred in a separate thread, one after another.
 * User provides input vector and output buffer information as the interlayer buffer
 * is used by all models, so only when it is this model's turn to execute can its
 * input be populated from the input_vector.
 *
 * Upon completion of inference the next job is queued before the user callback is invoked,
 * so the results have to be copied by the driver to the output_buffer before invoking the
 * user callback.
 *
 * @param[in] model_wrapper Model to run inference on, initialized via a one-time call to nrf_axon_nn_model_async_init.
 * @param[in] input_vector Input to run inference on. It is not consumed immediately so has to be in memory that is valid as long as inference is occurring.
 * @param[in] output_buffer buffer to copy inference results to.
 * @param[in] inference_callback Function to invoke when inference has completed.
 * @param[in] callback_context Opaque pointer provided to inference_callback.
 * @retval[0] Inference successfully queued.
 * @retval[NRF_AXON_RESULT_NOT_FINISHED] Model is still busy with an ealier inference
 * @retval[<0] Error code.
 */
nrf_axon_result_e nrf_axon_nn_model_infer_async(
	nrf_axon_nn_model_async_inference_wrapper_s *model_wrapper,
	const int8_t *input_vector,
	int8_t *output_buffer,
	void (*inference_callback)(nrf_axon_result_e result, void *callback_context),
	void *callback_context);

/**
 * @brief Gets the inference results for a classification model.
 *
 * Returns the inference index, and populates optional parameters label, score of the highest
 * scoring class.
 * Only valid for single dimension, classification type models.
 *
 * @param[in] compiled_model Model to get results for.
 * @param[in] packed_output Location of model output. If NULL, output is looked for in the
 *            interlayer buffer. This is not safe in dynamic systems.
 * @param[out] label Text of the classification label (if labels were provided in the compiled model).
 * @param[out] score Score of the highest scoring classification.
 * @retval Index of highest scoring classification.
 */
int16_t nrf_axon_nn_get_classification(
	const nrf_axon_nn_compiled_model_s *compiled_model,
	const int8_t *packed_output,
	const char **label, int32_t *score);

/**
 * @brief Copies model input from input_vector to the location in the interlayer buffer the model
 * expects.
 *
 * It is not recommended for users to invoke this function directly. The inference APIs handle
 * copying the input vector in a safe manner that do not risk corrupting the current or future
 * inferences.
 *
 * This function can be called safely in synchronous mode if Axon is reserved by the user (exactly
 * what the synchronous inference function does). This function cannot be called safely in
 * asynchronous inference mode unless the caller knows a priori that no other inferences are
 * occurring or will occur (ie, a simple one model system).
 *
 * @param[in] compiled_model model to copy input for.
 * @param[in] input_vector vector to copy
 * @retval 0 on success, or a negative error code. Note: errors due to multiple users of axon are
 *         not detected.
 */
nrf_axon_result_e nrf_axon_nn_populate_input_vector(
	const nrf_axon_nn_compiled_model_s *compiled_model,
	const int8_t *input_vector);

/**
 * @brief Returns the offset into the packed output buffer filled for the start of a particular
 *        output node index.
 *
 * Output node ndx 0 corresponds to the output described in the main compiled_model structures.
 * Output nodes ndx 1 and greater are correspond to extra_outputs[].
 * Each node's output data is stored in packed format (ie, rows do not have to start on a 32bit
 * boundary).
 * However, each node's output data will start on a 32bit boundary.
 * @param[in] compiled_model Model to find the output offset for.
 * @param[in] output_ndx index of the output to find the storage offset for.
 * @retval Offset into the output buffer to the start of the output for this output index.
 */
int nrf_axon_nn_offset_to_output_ndx(
	const nrf_axon_nn_compiled_model_s *compiled_model,
	uint8_t output_ndx);

/**
 * @brief Copies and packs the model inference output from the common interlayer buffer to the
 *        users dedicated buffer to_buffer.
 *
 * It is not recommended for users to invoke this function directly. The inference APIs handle
 * copying the output results in a safe manner that do not risk corrupting the current or future
 * inferences.
 *
 * This function cannot be called safely in synchronous or asynchronous inference modes unless the
 * caller knows a-priori that no other inferences are occurring or will occur (ie, a simple one
 * model system).
 * @param[in] compiled_model model that just completed inference to copy output from.
 * @param[out] to_buffer allocated buffer to copy output to. Must be sized to store all the outputs
 *             in the model, with each output starting on the next 32bit boundary after the prior
 *             one.
*/
void nrf_axon_nn_copy_output_to_packed_buffer(
	const nrf_axon_nn_compiled_model_s *compiled_model,
	void *to_buffer);

#ifdef __cplusplus
} /* extern "C" { */
#endif
