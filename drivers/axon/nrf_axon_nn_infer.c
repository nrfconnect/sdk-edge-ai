/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "axon/nrf_axon_platform.h"
#include "drivers/axon/nrf_axon_nn_infer.h"

nrf_axon_nn_async_inference_status_e nrf_axon_nn_get_model_infer_status(
	const nrf_axon_nn_model_async_inference_wrapper_s * model_wrapper)
{
	return model_wrapper->infer_status;
}

/*
* callback function invoked by the driver in the interrupt context when the model inferency
* has completed. context is the model handle.
*/
static void classify_complete_callback(nrf_axon_result_e result, void *callback_context)
{
	nrf_axon_nn_model_async_inference_wrapper_s *model_wrapper =
		(nrf_axon_nn_model_async_inference_wrapper_s *)callback_context;

	/* invoke the caller's callback. */
	model_wrapper->infer_status = NRF_AXON_NN_ASYNC_INFERENCE_STATUS_COMPLETE;
	if (NULL != model_wrapper->inference_callback) {
		model_wrapper->inference_callback(result, model_wrapper->callback_context);
	}
}

nrf_axon_result_e nrf_axon_nn_populate_input_vectors(
	const nrf_axon_nn_compiled_model_s *compiled_model)
{
	if (compiled_model->input_vector_list == NULL) {
		return NRF_AXON_RESULT_SUCCESS;
	}
	for (int16_t input_ndx = 0; input_ndx < compiled_model->input_cnt; input_ndx++) {
		if (compiled_model->input_vector_list[input_ndx] == NULL) {
			continue;
		}
		const nrf_axon_nn_compiled_model_input_s *the_input =
			&compiled_model->inputs[input_ndx];
		uint32_t input_size = the_input->stride * the_input->dimensions.height *
			the_input->dimensions.channel_cnt * the_input->dimensions.batch_cnt;

		memcpy(the_input->ptr, compiled_model->input_vector_list[input_ndx], input_size);
	}

	return NRF_AXON_RESULT_SUCCESS;
}

/**
 * In async mode, this is the callback to copy the input vectors to the to the interlayer buffer
 * immediately prior to inference. The axon mutex is owned at this time, preventing other queued
 * jobs from starting, so this needs to be as quick as possible.
 */
static void copy_input_callback(void *callback_context)
{
	nrf_axon_nn_model_async_inference_wrapper_s *model_wrapper =
		(nrf_axon_nn_model_async_inference_wrapper_s *)callback_context;
	nrf_axon_nn_populate_input_vectors(model_wrapper->compiled_model);
}

/**
 * In async mode, this is the callback to copy the results from the interlayer buffer
 * to the user's buffer. The axon mutex is owned at this time, preventing other queued
 * jobs from starting, so this needs to be as quick as possible.
 */
static void copy_result_callback(void *callback_context)
{
	nrf_axon_nn_model_async_inference_wrapper_s *model_wrapper =
		(nrf_axon_nn_model_async_inference_wrapper_s *)callback_context;
	nrf_axon_nn_copy_output_to_packed_buffer(
		model_wrapper->compiled_model, model_wrapper->output_buffer);
}

/*
 * A starts an asynchronous inference on the provided vector.
 * If the vector is NULL, assumption is that input buffer is already populated.
 */
nrf_axon_result_e nrf_axon_nn_model_infer_async_multi_inputs(
	nrf_axon_nn_model_async_inference_wrapper_s *model_wrapper,
	const int8_t **input_vector_list,
	int8_t *output_buffer,
	void (*inference_callback)(nrf_axon_result_e result, void *callback_context),
	void *callback_context)
{
	nrf_axon_result_e result;

	if (model_wrapper->infer_status == NRF_AXON_NN_ASYNC_INFERENCE_STATUS_ACTIVE) {
		/* model still busy w/ a prior inference */
		return NRF_AXON_RESULT_NOT_FINISHED;
	}

	model_wrapper->infer_status = NRF_AXON_NN_ASYNC_INFERENCE_STATUS_ACTIVE;

	model_wrapper->queued_cmd_buf_wrapper.cmd_buf_info = &model_wrapper->cmd_buf_info;
	model_wrapper->queued_cmd_buf_wrapper.callback_context = (void *)model_wrapper;
	/* register our own handler for the driver callback */
	model_wrapper->queued_cmd_buf_wrapper.callback_function = classify_complete_callback;
	model_wrapper->output_buffer = output_buffer;
	/* also record the user's callback */
	model_wrapper->inference_callback = inference_callback;
	model_wrapper->callback_context = callback_context;

	/* give the driver the input and output buffer locations and sizes*/
	if (input_vector_list == NULL) {
		/* advanced option. inputs are already populated. user knows that axon is idle. */
		model_wrapper->queued_cmd_buf_wrapper.copy_input_function = NULL;
	} else {
		if (input_vector_list != model_wrapper->compiled_model->input_vector_list) {
			/* copy it over */
			memcpy(model_wrapper->compiled_model->input_vector_list, input_vector_list,
			sizeof(input_vector_list) * model_wrapper->compiled_model->input_cnt);
		}
		model_wrapper->queued_cmd_buf_wrapper.copy_input_function = copy_input_callback;
	}
	model_wrapper->queued_cmd_buf_wrapper.copy_result_function =
		output_buffer == NULL ? NULL : copy_result_callback;
	/* and submit! */
	result = nrf_axon_queue_cmd_buf(&model_wrapper->queued_cmd_buf_wrapper);
	return result;
}

nrf_axon_result_e nrf_axon_nn_model_infer_async(
	nrf_axon_nn_model_async_inference_wrapper_s *model_wrapper,
	const int8_t *input_vector,
	int8_t *output_buffer,
	void (*inference_callback)(nrf_axon_result_e result, void *callback_context),
	void *callback_context)
{
	return nrf_axon_nn_model_infer_async_multi_inputs(model_wrapper, &input_vector,
		output_buffer, inference_callback, callback_context);
}

nrf_axon_result_e nrf_axon_nn_model_infer_sync_multi_inputs(
	const nrf_axon_nn_compiled_model_s *compiled_model,
	int8_t *output_buffer)
{
	nrf_axon_result_e result;

	nrf_axon_cmd_buffer_info_s cmd_buf_info;
	/* bind the command buffer to a wrapper struct */
	nrf_axon_init_command_buffer_info(&cmd_buf_info,
		compiled_model->cmd_buffer_ptr, compiled_model->cmd_buffer_len);

	/*
	 * The input vector is copied to the common interlayer buffer. Make sure no axon
	 * operations are active before copying to the buffer.
	 */
	if (!nrf_axon_platform_reserve_for_user()) {
		return NRF_AXON_RESULT_MUTEX_FAILED; /* should never happen! */
	}
	result = nrf_axon_nn_populate_input_vectors(compiled_model);
	if (result < 0) {
		nrf_axon_platform_free_reservation_from_user();
		return result;
	}

	/*
	 * Synchronous inference but do not free the axon reservation because need to
	 * pull the results 1st.
	 */
	result = nrf_axon_run_cmd_buf_sync(&cmd_buf_info,
		NRF_AXON_SYNC_MODE_BLOCKING_EVENT, false);

	if (NULL != output_buffer) {
		nrf_axon_nn_copy_output_to_packed_buffer(compiled_model, output_buffer);
	}

	nrf_axon_platform_free_reservation_from_user();
	return result;
}

nrf_axon_result_e nrf_axon_nn_model_infer_sync(
	const nrf_axon_nn_compiled_model_s *compiled_model,
	const int8_t *input_vector,
	int8_t *output_buffer)
{
	compiled_model->input_vector_list[0] = input_vector;
	return nrf_axon_nn_model_infer_sync_multi_inputs(compiled_model, output_buffer);
}

static uint16_t findmax8(const int8_t *buffer, const nrf_axon_nn_model_layer_dimensions_s *dim_ptr,
	uint16_t stride, int32_t *score)
{
	int32_t max_value = *buffer;
	uint16_t max_value_ndx = 0;
	uint16_t extra_stride = stride - dim_ptr->byte_width*dim_ptr->width;

	for (uint16_t ch_ndx = 0; ch_ndx < dim_ptr->channel_cnt; ch_ndx++) {
		for (uint16_t height_ndx = 0; height_ndx < dim_ptr->height; height_ndx++) {

			for (uint16_t width_ndx = 0; width_ndx < dim_ptr->width; width_ndx++) {

				if (*buffer > max_value) {
					max_value = *buffer;
					max_value_ndx = width_ndx +
						dim_ptr->width*height_ndx +
						dim_ptr->width*dim_ptr->height * ch_ndx;
				}
				buffer++;
			}
			buffer += extra_stride;
		}
	}
	if (score != NULL) {
		*score = max_value;
	}
	return max_value_ndx;
}

static uint16_t findmax16(const int16_t *buffer,
	const nrf_axon_nn_model_layer_dimensions_s *dim_ptr,
	uint16_t stride, int32_t *score)
{
	int32_t max_value = *buffer;
	uint16_t max_value_ndx = 0;
	uint16_t extra_stride = stride - dim_ptr->byte_width*dim_ptr->width;

	for (uint16_t ch_ndx = 0; ch_ndx < dim_ptr->channel_cnt; ch_ndx++) {
		for (uint16_t height_ndx = 0; height_ndx < dim_ptr->height; height_ndx++) {
			for (uint16_t width_ndx = 0; width_ndx < dim_ptr->width; width_ndx++) {
				if (*buffer > max_value) {
					max_value = *buffer;
					max_value_ndx = width_ndx + dim_ptr->width*height_ndx +
					dim_ptr->width*dim_ptr->height * ch_ndx;
				}
				buffer++;
			}
			buffer = (int16_t *)((int8_t *)buffer + extra_stride);
		}
	}
	if (score != NULL) {
		*score = max_value;
	}
	return max_value_ndx;
}

static uint16_t findmax32(const int32_t *buffer,
	const nrf_axon_nn_model_layer_dimensions_s *dim_ptr,
	uint16_t stride, int32_t *score)
{
	int32_t max_value = *buffer;
	uint16_t max_value_ndx = 0;
	uint16_t extra_stride = stride - dim_ptr->byte_width * dim_ptr->width;

	for (uint16_t ch_ndx = 0; ch_ndx < dim_ptr->channel_cnt; ch_ndx++) {
		for (uint16_t height_ndx = 0; height_ndx < dim_ptr->height; height_ndx++) {
			for (uint16_t width_ndx = 0; width_ndx < dim_ptr->width; width_ndx++) {
				if (*buffer > max_value) {
					max_value = *buffer;
					max_value_ndx = width_ndx + dim_ptr->width * height_ndx +
						dim_ptr->width*dim_ptr->height * ch_ndx;
				}
				buffer++;
			}
			buffer = (int32_t *)((int8_t *)buffer + extra_stride);
		}
	}
	if (score != NULL) {
		*score = max_value;
	}
	return max_value_ndx;
}

int nrf_axon_nn_offset_to_output_ndx(const nrf_axon_nn_compiled_model_s *compiled_model,
	uint8_t output_ndx)
{
	/* short circuit the most likely path */
	if (output_ndx == 0) {
		return 0;
	}

	/* prevent out of range. */
	if (output_ndx >= (compiled_model->output_cnt)) {
		return -1; /* illegal index */
	}
	return compiled_model->outputs[output_ndx].packed_buffer_offset;
}

/**
 * Copies the model's output from the interlayer buffer to the user-provided output_buffer.
 * output structure is int<bytewidth*8> [channel_cnt][height][width]
 * Result will be packed (output_stride = output_bytewidth * output_width)
 */
static void copy_unpacked_to_packed_buffer(const nrf_axon_nn_model_layer_dimensions_s *dimensions,
	uint16_t stride, const int8_t *from_buffer, int8_t *to_buffer)
{
	uint16_t row_width_in_bytes = dimensions->byte_width * dimensions->width;

	for (uint16_t ch_ndx = 0; ch_ndx < dimensions->channel_cnt; ch_ndx++) {
		for (uint16_t height_ndx = 0; height_ndx < dimensions->height; height_ndx++) {
			memcpy(to_buffer, from_buffer, row_width_in_bytes);
			from_buffer += stride;
			to_buffer += row_width_in_bytes;
		}
	}
}

void nrf_axon_nn_copy_output_to_packed_buffer(
	const nrf_axon_nn_compiled_model_s *compiled_model, int8_t *to_buffer)
{
	for (int output_ndx = 0;
		output_ndx < compiled_model->output_cnt;
		output_ndx++) {
		copy_unpacked_to_packed_buffer(
			&compiled_model->outputs[output_ndx].dimensions,
			compiled_model->outputs[output_ndx].stride,
			compiled_model->outputs[output_ndx].ptr,
			(int8_t *)to_buffer +
			compiled_model->outputs[output_ndx].packed_buffer_offset);
	}
}
/*
 * find the maxvalue in io_buffer and return its index
 * legacy function for single output models.
 */
int16_t nrf_axon_nn_get_classification(const nrf_axon_nn_compiled_model_s *compiled_model,
	const int8_t *packed_output, const char **label, int32_t *score)
{
	const nrf_axon_nn_model_layer_dimensions_s *output_dim_ptr =
		&compiled_model->outputs[0].dimensions;
	const int8_t *output;
	uint16_t output_stride;

	if (packed_output != NULL) {
		/* user provided the packed output in a separate buffer */
		output = packed_output;
		output_stride = output_dim_ptr->width * output_dim_ptr->byte_width;
	} else {
		/* extract the output from the interlayer buffer, could be unpacked. */
		output_stride = compiled_model->outputs[0].stride;
		output = (int8_t *)compiled_model->outputs[0].ptr;
	}
	static const char *label_not_applicable = "N/A";

	if (label != NULL) {
		*label = label_not_applicable;
	}
	int16_t result;

	switch (compiled_model->outputs[0].dimensions.byte_width) {
	case 1:
		result = findmax8((int8_t *)output, output_dim_ptr, output_stride, score);
		break;
	case 2:
		result = findmax16((int16_t *)output, output_dim_ptr, output_stride, score);
		break;
	case 4:
		result = findmax32((int32_t *)output, output_dim_ptr, output_stride, score);
		break;
	default:
		/* shouldn't happen. */
		return NRF_AXON_RESULT_FAILURE;
	}
	if (label != NULL) {
		if (compiled_model->labels != NULL) {
			*label = compiled_model->labels[result];
		} else {
			*label = NULL;
		}
	}
	return result;
}

static inline void init_persistent_var_i8(const nrf_axon_nn_model_persistent_var_s *persistent_var)
{
	memset(persistent_var->buf_ptr, persistent_var->initial_value, persistent_var->buf_size);
}
static inline void init_persistent_var_i16(const nrf_axon_nn_model_persistent_var_s *persistent_var)
{
	uint32_t length = persistent_var->buf_size >> 1;
	int16_t *buff_i16 = (int16_t *)persistent_var->buf_ptr;

	while (length--) {
		*buff_i16++ = (int16_t)persistent_var->initial_value;
	}
}
/**
 * initialize all the persistent var buffers.
 */
int nrf_axon_nn_model_init_vars(const nrf_axon_nn_compiled_model_s *compiled_model)
{
	for (uint16_t var_ndx = 0; var_ndx < compiled_model->persistent_vars.count; var_ndx++) {
		switch (compiled_model->persistent_vars.vars[var_ndx].byte_width) {
		case 1: /* 1 byte, use memset */
			init_persistent_var_i8(&compiled_model->persistent_vars.vars[var_ndx]);
			break;
		case 2: /* 2bytes, fill with shorts */
			init_persistent_var_i16(&compiled_model->persistent_vars.vars[var_ndx]);
			break;
		default:
			return -1; /* unsupported. */
		}
	}

	return 0;
}

nrf_axon_result_e nrf_axon_nn_model_validate(const nrf_axon_nn_compiled_model_s *compiled_model)
{
	if (NULL == compiled_model) {
		return NRF_AXON_RESULT_INVALID_MODEL;
	}

	if ((NULL == compiled_model->cmd_buffer_ptr) ||
		(0 == compiled_model->cmd_buffer_len)) {
		return NRF_AXON_RESULT_INVALID_CMD_BUF;
	}
	if (compiled_model->interlayer_buffer_needed) {/* need the interlayer buffer */
#if NRF_AXON_INTERLAYER_BUFFER_SIZE
		if (sizeof(nrf_axon_interlayer_buffer) < compiled_model->interlayer_buffer_needed) {
			nrf_axon_platform_printf(
				"validate model %s failed! interlayer buffer too small! Allocated %d, need %d\n",
				compiled_model->model_name, sizeof(nrf_axon_interlayer_buffer),
				compiled_model->interlayer_buffer_needed);
			return NRF_AXON_RESULT_BUFFER_TOO_SMALL;
		}
#else
		nrf_axon_platform_printf(
			"validate model %s failed! interlayer buffer not defined! add "
			"CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE=%d to the prj.conf file!\n",
			compiled_model->model_name, compiled_model->interlayer_buffer_needed);
		return NRF_AXON_RESULT_BUFFER_TOO_SMALL;
#endif
	}

	if (compiled_model->psum_buffer_needed) {/* need the interlayer buffer */
#if NRF_AXON_PSUM_BUFFER_SIZE
		if (sizeof(nrf_axon_psum_buffer) < compiled_model->psum_buffer_needed) {
			nrf_axon_platform_printf(
				"validate model %s failed! psum buffer too small! Allocated %d, need %d\n",
				compiled_model->model_name, sizeof(nrf_axon_psum_buffer),
				compiled_model->psum_buffer_needed);
			return NRF_AXON_RESULT_BUFFER_TOO_SMALL;
		}
#else
		nrf_axon_platform_printf("validate model %s failed! psum buffer not defined!"
			"add CONFIG_NRF_AXON_PSUM_BUFFER_SIZE=%d to the prj.conf file!\n",
			compiled_model->model_name, compiled_model->psum_buffer_needed);
		return NRF_AXON_RESULT_BUFFER_TOO_SMALL;
#endif
	}

	if (compiled_model->min_driver_version_required >
		NRF_AXON_VERSION) {
		nrf_axon_platform_printf("validate model %s failed!\nCurrent driver "
			"version 0x%x is lower than version required by the model 0x%x\n",
			compiled_model->model_name, NRF_AXON_VERSION,
			compiled_model->min_driver_version_required);
		return NRF_AXON_RESULT_DRIVER_VERSION_TOO_OLD;
	}
	return NRF_AXON_RESULT_SUCCESS;
}

nrf_axon_result_e nrf_axon_nn_model_async_init(
	nrf_axon_nn_model_async_inference_wrapper_s *the_model,
	const nrf_axon_nn_compiled_model_s *compiled_model)
{
	nrf_axon_result_e result = nrf_axon_nn_model_validate(compiled_model);

	if (result < 0) {
		return result;
	}

	if (NULL == the_model) {
		return NRF_AXON_RESULT_NULL_BUFFER;
	}
	memset(the_model, 0, sizeof(*the_model));
	/* attach the two... */
	the_model->compiled_model = compiled_model;
	/* init the cmd buffer */
	nrf_axon_init_command_buffer_info(&the_model->cmd_buf_info,
		compiled_model->cmd_buffer_ptr,
		compiled_model->cmd_buffer_len);
	return 0;
}
