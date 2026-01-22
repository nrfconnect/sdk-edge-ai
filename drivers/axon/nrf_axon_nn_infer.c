/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */

 #include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "nrf_axon_platform.h"
#include "nrf_axon_nn_infer.h"

nrf_axon_nn_async_inference_status_e nrf_axon_nn_get_model_infer_status(const nrf_axon_nn_model_async_inference_wrapper_s* model_wrapper) {
  return model_wrapper->infer_status;
}

int8_t nrf_axon_nn_model_1st_external_input_ndx(const nrf_axon_nn_compiled_model_s* the_model) {
  for (uint8_t input_ndx=0; input_ndx < the_model->input_cnt; input_ndx++) {
    if (the_model->inputs[input_ndx].is_external) {
      return (int8_t)input_ndx;
    }
  }
  return -1;
}
const nrf_axon_nn_compiled_model_input_s *nrf_axon_nn_model_1st_external_input(const nrf_axon_nn_compiled_model_s* the_model) {
  int8_t input_ndx = nrf_axon_nn_model_1st_external_input_ndx(the_model);
  return input_ndx < 0 ? NULL : &the_model->inputs[input_ndx];
}

/*
* callback function invoked by the driver in the interrupt context when the model inferency has completed
* context is the model handle.
*/
static void classify_complete_callback(nrf_axon_result_e result, void* callback_context) {
  nrf_axon_nn_model_async_inference_wrapper_s* model_wrapper = (nrf_axon_nn_model_async_inference_wrapper_s*)callback_context;

  // invoke the caller's callback.
  model_wrapper->infer_status = kAxonnnInferStatusInferenceComplete;
  if (NULL != model_wrapper->inference_callback) {
    model_wrapper->inference_callback(result, model_wrapper->callback_context);
  }
}

nrf_axon_result_e nrf_axon_nn_populate_input_vector(const nrf_axon_nn_compiled_model_s *compiled_model, const int8_t* input_vector)
{
  if (input_vector==NULL) {
    return NRF_AXON_RESULT_SUCCESS;
  }
  if (compiled_model->external_input_ndx < 0) {
    // no external input!
    nrf_axon_platform_printf("ERROR! MODEL LACKS AN EXTERNAL INPUT!\n");
    return NRF_AXON_RESULT_INVALID_MODEL;
  }
  const nrf_axon_nn_compiled_model_input_s *the_input = &compiled_model->inputs[compiled_model->external_input_ndx];
  uint32_t input_size = the_input->stride * the_input->dimensions.height * 
    the_input->dimensions.channel_cnt;
  memcpy(the_input->ptr, input_vector, input_size);
  
  return NRF_AXON_RESULT_SUCCESS;
}
/*
 * A starts an asynchronous inference on the provided vector.
 * If the vector is NULL, assumption is that input buffer is already populated.
 */
nrf_axon_result_e nrf_axon_nn_model_infer_async(
  nrf_axon_nn_model_async_inference_wrapper_s* model_wrapper,
  const int8_t* input_vector, 
  int8_t *output_buffer,
  void (*inference_callback)(nrf_axon_result_e result, void* callback_context), // user call-back function
  void* callback_context)// provided by the inference caller to be passed to the inference_callback() blindly
{
  nrf_axon_result_e result;
  const nrf_axon_nn_compiled_model_s *compiled_model = model_wrapper->compiled_model;
  
  if (model_wrapper->infer_status == kAxonnnInferStatusInferring) {
    return NRF_AXON_RESULT_NOT_FINISHED; // model still busy w/ a prior inference
  }

  model_wrapper->infer_status = kAxonnnInferStatusInferring;
  if (0 > (result = nrf_axon_nn_populate_input_vector(compiled_model, input_vector))) {
    return result;
  }

  model_wrapper->queued_cmd_buf_wrapper.cmd_buf_info = &model_wrapper->cmd_buf_info;
  model_wrapper->queued_cmd_buf_wrapper.callback_context = (void *)model_wrapper;
  // register our own handler for the driver callback
  model_wrapper->queued_cmd_buf_wrapper.callback_function = classify_complete_callback; 
  // also record the user's callback
  model_wrapper->inference_callback = inference_callback;
  model_wrapper->callback_context = callback_context;

    /* give the driver the input and ouput buffer locations and sizes*/
  if (input_vector == NULL) {
    // advanced option. inputs are already populated. user knows that axon is idle.
    model_wrapper->queued_cmd_buf_wrapper.input_buffer = NULL;
    model_wrapper->queued_cmd_buf_wrapper.input_size = 0;
    model_wrapper->queued_cmd_buf_wrapper.input_vector = NULL;
  } else {
    const nrf_axon_nn_compiled_model_input_s *model_input = nrf_axon_nn_model_1st_external_input(compiled_model);
    if (NULL == model_input) {
      nrf_axon_platform_printf("ERROR: no external input to model %s\n", compiled_model->model_name);
      return NRF_AXON_RESULT_FAILURE;
    }
    model_wrapper->queued_cmd_buf_wrapper.input_buffer = model_input->ptr;
    model_wrapper->queued_cmd_buf_wrapper.input_size = model_input->dimensions.byte_width*model_input->dimensions.channel_cnt*model_input->dimensions.height*model_input->dimensions.width;
    model_wrapper->queued_cmd_buf_wrapper.input_vector = input_vector;
  }
  model_wrapper->queued_cmd_buf_wrapper.output_buffer = output_buffer;
  model_wrapper->queued_cmd_buf_wrapper.tmp_output_buffer = model_wrapper->compiled_model->output_ptr;
  model_wrapper->queued_cmd_buf_wrapper.output_stride = model_wrapper->compiled_model->output_stride;
  model_wrapper->queued_cmd_buf_wrapper.output_width_in_bytes = model_wrapper->compiled_model->output_dimensions.byte_width * model_wrapper->compiled_model->output_dimensions.width;
  model_wrapper->queued_cmd_buf_wrapper.output_buffer_packed_size = model_wrapper->queued_cmd_buf_wrapper.output_width_in_bytes * 
                  model_wrapper->compiled_model->output_dimensions.height * model_wrapper->compiled_model->output_dimensions.channel_cnt;
  // and submit!
  result = nrf_axon_queue_cmd_buf(&model_wrapper->queued_cmd_buf_wrapper);
  return result;
}

/*
*/
nrf_axon_result_e nrf_axon_nn_model_infer_sync(
  const nrf_axon_nn_compiled_model_s *compiled_model, 
  const int8_t* input_vector, 
  int8_t *output_buffer) 
{
  nrf_axon_result_e result;

  nrf_axon_cmd_buffer_info_s cmd_buf_info;
  /* bind the command buffer to a wrapper struct */
  nrf_axon_init_command_buffer_info(&cmd_buf_info, compiled_model->cmd_buffer_ptr, compiled_model->cmd_buffer_len);

  /*
   The input vector is copied to the common interlayer buffer. Make sure no axon operations are 
   active before copying to the buffer.
   */
  if (!nrf_axon_platform_reserve_for_user()) {
    return NRF_AXON_RESULT_MUTEX_FAILED; // should never happen!
  }
  if (0 > (result = nrf_axon_nn_populate_input_vector(compiled_model, input_vector))) {
    nrf_axon_platform_free_reservation_from_user();
    return result;
  }

  /*
    Synchronous inference but do not free the axon reservation because need to pull the results 1st.
  */
  result = nrf_axon_run_cmd_buf_sync(&cmd_buf_info, NRF_AXON_SYNC_MODE_BLOCKING_EVENT, false);

  if (NULL != output_buffer) {
    nrf_axon_nn_copy_output_to_packed_buffer(compiled_model, output_buffer);
  }

  nrf_axon_platform_free_reservation_from_user();
  return result;
}

static uint16_t findmax8(const int8_t* buffer, const nrf_axon_nn_model_layer_dimensions_s* dim_ptr, uint16_t stride, int32_t* score) {
  int32_t max_value = *buffer;
  uint16_t max_value_ndx = 0;
  uint16_t extra_stride = stride - dim_ptr->byte_width*dim_ptr->width;
  for (uint16_t ch_ndx = 0; ch_ndx < dim_ptr->channel_cnt; ch_ndx++) {
    for (uint16_t height_ndx = 0; height_ndx < dim_ptr->height; height_ndx++) {
      for (uint16_t width_ndx = 0; width_ndx < dim_ptr->width; width_ndx++) {
        if (*buffer > max_value) {
          max_value = *buffer;
          max_value_ndx = width_ndx + dim_ptr->width*height_ndx + dim_ptr->width*dim_ptr->height*ch_ndx;
        }
        buffer++;
      }
      buffer += extra_stride;
    }
  }
  if (NULL != score) {
    *score = max_value;
  }
  return max_value_ndx;
}

static uint16_t findmax16(const int16_t* buffer, const nrf_axon_nn_model_layer_dimensions_s* dim_ptr, uint16_t stride, int32_t* score) {
  int32_t max_value = *buffer;
  uint16_t max_value_ndx = 0;
  uint16_t extra_stride = stride - dim_ptr->byte_width*dim_ptr->width;
  for (uint16_t ch_ndx = 0; ch_ndx < dim_ptr->channel_cnt; ch_ndx++) {
    for (uint16_t height_ndx = 0; height_ndx < dim_ptr->height; height_ndx++) {
      for (uint16_t width_ndx = 0; width_ndx < dim_ptr->width; width_ndx++) {
        if (*buffer > max_value) {
          max_value = *buffer;
          max_value_ndx = width_ndx + dim_ptr->width*height_ndx + dim_ptr->width*dim_ptr->height*ch_ndx;
        }
        buffer++;
      }
      buffer = (int16_t*)((int8_t*)buffer + extra_stride);
    }
  }
  if (NULL != score) {
    *score = max_value;
  }
  return max_value_ndx;
}

static uint16_t findmax32(const int32_t* buffer, const nrf_axon_nn_model_layer_dimensions_s* dim_ptr, uint16_t stride, int32_t *score) {
  int32_t max_value = *buffer;
  uint16_t max_value_ndx = 0;
  uint16_t extra_stride = stride - dim_ptr->byte_width*dim_ptr->width;
  for (uint16_t ch_ndx = 0; ch_ndx < dim_ptr->channel_cnt; ch_ndx++) {
    for (uint16_t height_ndx = 0; height_ndx < dim_ptr->height; height_ndx++) {
      for (uint16_t width_ndx = 0; width_ndx < dim_ptr->width; width_ndx++) {
        if (*buffer > max_value) {
          max_value = *buffer;
          max_value_ndx = width_ndx + dim_ptr->width*height_ndx + dim_ptr->width*dim_ptr->height*ch_ndx;
        }
        buffer++;
      }
      buffer = (int32_t*)((int8_t*)buffer + extra_stride);
    }
  }
  if (NULL != score) {
    *score = max_value;
  }
  return max_value_ndx;
}


/**
 * Copies the model's output from the interlayer buffer to the user-provided output_buffer.
 * output structure is int<bytewidth*8> [channel_cnt][height][width]
 * Result will be packed (output_stride = output_bytewidth * output_width)
 */
void nrf_axon_nn_copy_output_to_packed_buffer(const nrf_axon_nn_compiled_model_s *compiled_model, void *to_buffer)
{
  int8_t *from_buffer = compiled_model->output_ptr;
  int8_t *to_buffer_int8 = to_buffer;
  uint16_t row_width_in_bytes = compiled_model->output_dimensions.byte_width*compiled_model->output_dimensions.width;
  for (uint16_t ch_ndx = 0; ch_ndx < compiled_model->output_dimensions.channel_cnt; ch_ndx++) {
    for (uint16_t height_ndx = 0; height_ndx < compiled_model->output_dimensions.height; height_ndx++) {
      memcpy(to_buffer_int8, from_buffer, row_width_in_bytes);
      from_buffer += compiled_model->output_stride;
      to_buffer_int8 += row_width_in_bytes;
    }
  }

}

/*
 * find the maxvalue in io_buffer and return its index
 */
int16_t nrf_axon_nn_get_classification(const nrf_axon_nn_compiled_model_s *compiled_model, const int8_t *packed_output, const char** label, int32_t* score) {

  const nrf_axon_nn_model_layer_dimensions_s* output_dim_ptr = &compiled_model->output_dimensions;
  const int8_t *output;
  uint16_t output_stride;
  if (NULL != packed_output) {
    // user provided the packed output in a separate buffer
    output = packed_output;
    output_stride = output_dim_ptr->width * output_dim_ptr->byte_width;
  } else {
    // extract the output from the interlayer buffer, could be unpacked.
    output_stride = compiled_model->output_stride;
    output = (int8_t *)compiled_model->output_ptr;
  }
  static const char* label_not_applicable = "N/A";
  if (NULL != label) {
    *label = label_not_applicable;
  }
  int16_t result;

  switch (compiled_model->output_dimensions.byte_width) {
  case 1:
    result = findmax8((int8_t *)output, output_dim_ptr, output_stride, score);
    break;
  case 2:
    result = findmax16((int16_t*)output, output_dim_ptr, output_stride, score);
    break;
  case 4:
    result = findmax32((int32_t*)output, output_dim_ptr, output_stride, score);
    break;
  default:
    // shouldn't happen. 
    return NRF_AXON_RESULT_FAILURE; 
  }
  if (NULL != label) {
    if (NULL != compiled_model->labels) {
      *label = compiled_model->labels[result];
    } else {
      *label = NULL;
    }
  }
  return result;
}

/**
 * initialize all the persistent var buffers.
 */
int nrf_axon_nn_model_init_vars(const nrf_axon_nn_compiled_model_s *compiled_model) {
  for (uint16_t var_ndx=0;var_ndx < compiled_model->persistent_vars.count;var_ndx++) {
    switch (compiled_model->persistent_vars.vars[var_ndx].byte_width) {
      case 1: // 1 byte, use memset
        memset(compiled_model->persistent_vars.vars[var_ndx].buf_ptr, compiled_model->persistent_vars.vars[var_ndx].initial_value, compiled_model->persistent_vars.vars[var_ndx].buf_size);
        break;
      case 2: {// 2bytes, fill with shorts
        unsigned length = compiled_model->persistent_vars.vars[var_ndx].buf_size >> 1;
        int16_t *buff_i16 = (int16_t*)compiled_model->persistent_vars.vars[var_ndx].buf_ptr;
        while (length--) {
          *buff_i16++= (int16_t)compiled_model->persistent_vars.vars[var_ndx].initial_value;
        } 
        break;
      }
      default:
        return -1; // unsupported.
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
      (0==compiled_model->cmd_buffer_len)) {
    return NRF_AXON_RESULT_INVALID_CMD_BUF;
  }
  if (compiled_model->interlayer_buffer_needed) {// need the interlayer buffer
#if NRF_AXON_INTERLAYER_BUFFER_SIZE
  if (sizeof(nrf_axon_interlayer_buffer) < compiled_model->interlayer_buffer_needed) {
    nrf_axon_platform_printf("init model %s failed! interlayer buffer too small! Allocated %d, need %d\n", compiled_model->model_name, sizeof(nrf_axon_interlayer_buffer), compiled_model->interlayer_buffer_needed);
    return NRF_AXON_RESULT_BUFFER_TOO_SMALL;
  }
#else
    nrf_axon_platform_printf("init model %s failed! interlayer buffer not defined! add CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE=%d to the prj.conf file!\n", compiled_model->model_name, compiled_model->interlayer_buffer_needed);
    return NRF_AXON_RESULT_BUFFER_TOO_SMALL;
#endif
  }

  if (compiled_model->psum_buffer_needed) {// need the interlayer buffer
#if NRF_AXON_PSUM_BUFFER_SIZE
  if (sizeof(nrf_axon_psum_buffer) < compiled_model->psum_buffer_needed) {
    nrf_axon_platform_printf("init model %s failed! psum buffer too small! Allocated %d, need %d\n", compiled_model->model_name, sizeof(nrf_axon_psum_buffer), compiled_model->psum_buffer_needed);
    return NRF_AXON_RESULT_BUFFER_TOO_SMALL;
  }
#else
    nrf_axon_platform_printf("init model %s failed! psum buffer not defined! add CONFIG_NRF_AXON_PSUM_BUFFER_SIZE=%d to the prj.conf file!\n", compiled_model->model_name, compiled_model->psum_buffer_needed);
    return NRF_AXON_RESULT_BUFFER_TOO_SMALL;
#endif
  }
  return NRF_AXON_RESULT_SUCCESS;
}

nrf_axon_result_e nrf_axon_nn_model_async_init(nrf_axon_nn_model_async_inference_wrapper_s *the_model, const nrf_axon_nn_compiled_model_s *compiled_model) 
{
  nrf_axon_result_e result = nrf_axon_nn_model_validate(compiled_model);
  if (result < 0) {
    return result;
  }
  
  if (NULL == the_model) {
    return NRF_AXON_RESULT_NULL_BUFFER;
  }
  memset(the_model, 0, sizeof(*the_model));
  

  // attach the two...
  the_model->compiled_model = compiled_model;
  // init the cmd buffer
  nrf_axon_init_command_buffer_info(&the_model->cmd_buf_info, compiled_model->cmd_buffer_ptr, compiled_model->cmd_buffer_len);
  return 0;
}
