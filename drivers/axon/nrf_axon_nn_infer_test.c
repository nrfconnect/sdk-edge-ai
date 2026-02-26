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
#include <stdlib.h>
#include "drivers/axon/nrf_axon_nn_infer_test.h"
#include "axon/nrf_axon_platform.h"
#include "axon/nrf_axon_logging.h"
#if AXON_SIMULATION    
# include "axon/nrf_axon_platform_simulator.h"
#endif

static void unpack_vector32(int32_t* unpacked, const int32_t* packed, const nrf_axon_nn_model_layer_dimensions_s* packed_dimensions, uint16_t unpacked_stride) {
  uint16_t extra_stride = (unpacked_stride -  (packed_dimensions->width * packed_dimensions->byte_width)) / packed_dimensions->byte_width;
  for (int ch_ndx = 0; ch_ndx < packed_dimensions->channel_cnt; ch_ndx++) {
    for (int height_ndx = 0; height_ndx < packed_dimensions->height; height_ndx++) {
      for (int width_ndx = 0; width_ndx < packed_dimensions->width; width_ndx++) {
        // accumulate the mis-match count in result.
        *unpacked = *packed;
        unpacked++;
        packed++;
      }
      // expected_output is packed, outpout isn't, so advance the output_ndx past the padding space
      unpacked += extra_stride;
    }
  }
}

static void unpack_vector16(int16_t* unpacked, const int16_t* packed, const nrf_axon_nn_model_layer_dimensions_s* packed_dimensions, uint16_t unpacked_stride) {
  uint16_t extra_stride = (unpacked_stride - (packed_dimensions->width * packed_dimensions->byte_width)) / packed_dimensions->byte_width;
  for (int ch_ndx = 0; ch_ndx < packed_dimensions->channel_cnt; ch_ndx++) {
    for (int height_ndx = 0; height_ndx < packed_dimensions->height; height_ndx++) {
      for (int width_ndx = 0; width_ndx < packed_dimensions->width; width_ndx++) {
        // accumulate the mis-match count in result.
        *unpacked = *packed;
        unpacked++;
        packed++;
      }
      // expected_output is packed, output isn't, so advance the output_ndx past the padding space
      unpacked += extra_stride;
    }
  }
}

static void unpack_vector8(int8_t* unpacked, const int8_t* packed, const nrf_axon_nn_model_layer_dimensions_s* packed_dimensions, uint16_t unpacked_stride) {

  uint16_t extra_stride = unpacked_stride -
    (packed_dimensions->width * packed_dimensions->byte_width);

  for (int ch_ndx = 0; ch_ndx < packed_dimensions->channel_cnt; ch_ndx++) {
    for (int height_ndx = 0; height_ndx < packed_dimensions->height; height_ndx++) {
      for (int width_ndx = 0; width_ndx < packed_dimensions->width; width_ndx++) {
        // accumulate the mis-match count in result.
        *unpacked = *packed;
        unpacked += packed_dimensions->byte_width;
        packed++;
      }
      // expected_output is packed, outpout isn't, so advance the output_ndx past the padding space
      unpacked += extra_stride;
    }
  }
}

static void unpack_vector(void* unpacked, const void* packed, const nrf_axon_nn_model_layer_dimensions_s* packed_dimensions, uint16_t unpacked_stride) {
  if ((NULL==unpacked) || (NULL==packed)){
    return;
  }
  switch (packed_dimensions->byte_width) {
    case 1: unpack_vector8( (int8_t*) unpacked, (int8_t*) packed, packed_dimensions, unpacked_stride); break;
    case 2: unpack_vector16((int16_t*) unpacked, (int16_t*) packed, packed_dimensions, unpacked_stride); break;
    case 4: unpack_vector32((int32_t*) unpacked, (int32_t*) packed, packed_dimensions, unpacked_stride); break;
  }
}
#define CEIL4(number) ((((number)+3)>>2)<<2)

/*
* Compares the potentially unpacked output to the packed expected_output.
*/
static int test_compare_results(const nrf_axon_nn_compiled_model_s* compiled_model, const int8_t *packed_output, const int8_t *expected_output) {
  int max_error = 0;
  int max_error_cnt = 0;
  int success_cnt = 0;

  uint16_t extra_stride;
  const int8_t *output; 
  if (packed_output != NULL) {
    /* Caller provided the output in unpacked form; no extra stride */
    output = packed_output;
    extra_stride = 0;
  } else {
    /* extract the output from the interlayer buffer. It will probably be un-packed so need extra stride. */
    output = compiled_model->output_ptr;
    extra_stride = compiled_model->output_stride - 
                  (compiled_model->output_dimensions.width * compiled_model->output_dimensions.byte_width);
  }
  int result = 0;

  for (int ch_ndx = 0; ch_ndx < compiled_model->output_dimensions.channel_cnt; ch_ndx++) {
    for (int height_ndx = 0; height_ndx < compiled_model->output_dimensions.height; height_ndx++) {
      for (int width_ndx = 0; width_ndx < compiled_model->output_dimensions.width; width_ndx++) {
        // accumulate the mis-match count in result.
        int32_t output_val;
        int32_t expected_val;
        switch (compiled_model->output_dimensions.byte_width) {
        case 1: 
          output_val = *(int8_t*)output; 
          expected_val = *(int8_t*)expected_output;
          break;
        case 2: 
          output_val = *(int16_t*)output; 
          expected_val = *(int16_t*)expected_output;
          break;
        case 4: 
          output_val = *(int32_t*)output; 
          expected_val = *(int32_t*)expected_output;
          break;
        }

        int error = abs(output_val - expected_val);
        if (!error) {
          // nrf_axon_platform_printf("matched: %d == %d @ch %d, row %d, col %d\n", output_val, expected_val, ch_ndx, height_ndx, width_ndx);
          success_cnt++;
        } else {
          result++;
          // if this was the 1st error, display its location.
          if (result < 15) {
            nrf_axon_platform_printf("output mismatch: %d != %d @ch %d, row %d, col %d\n", output_val, expected_val, ch_ndx, height_ndx, width_ndx);
          } 
          if (error > max_error) {
            max_error = error;
            max_error_cnt = 1;
          } else if (error == max_error) {
            max_error_cnt++;
          }
        }
        output += compiled_model->output_dimensions.byte_width;
        expected_output += compiled_model->output_dimensions.byte_width;
      }
      // expected_output is packed, outpout isn't, so advance the output_ndx past the padding space
      output += extra_stride;
    }
  }
  // all done, print the results
  if (result) {
    nrf_axon_platform_printf("output matched %d, mismatched %d, max error %d occurred %d times.\n", success_cnt, result, max_error, max_error_cnt);
  } else {
    nrf_axon_platform_printf("output bit exact!\n");
  }
  return result;
}

int nrf_axon_print_test_inference_results(const nrf_axon_nn_compiled_model_s *compiled_model, int8_t *output_ptr, uint32_t profiling_ticks) {
  int label_ndx;
  const char* label;
  int32_t score;
  

  if (compiled_model->is_layer_model) {
    label_ndx = -1;
    label = "N/A";
  } else {
    label_ndx = nrf_axon_nn_get_classification(compiled_model, output_ptr, &label, &score);
    if (label==NULL) {
      label = "(null)";
    }
  }
  nrf_axon_platform_printf("model %s inference: ndx %d, label %s, score %d, profiling ticks %u\n", compiled_model->model_name, label_ndx, label, score, profiling_ticks);
  return label_ndx;
}

/**
 * Runs an individual layer model generated by the compiler.
 * Injects the input(s) into the interlayer buffer because the input_buffer parameter to nrf_axon_nn_model_infer_sync
 * only supports a single external input which is by definition packed.
 * 
 * Note: this function does not follow buffer access best practices; it accesses the output directly from
 * the interlayer buffer. However, it is understood that this is test code for testing individual layers,
 * and there are no other axon users.
 */
static int run_layer_test_vector(const nrf_axon_nn_compiled_model_s* compiled_model, const int8_t *packed_input, const int8_t *packed_input1, const int8_t *expected_output) 
{

  /* Inner layers expect inputs to be unpacked because outputs are always unpacked. However, test vectors are stored in packed format,
     so need to be unpacked and inserted into the interlayer buffer.*/
  nrf_axon_platform_reserve_for_user();
  unpack_vector(compiled_model->inputs[0].ptr, packed_input, &compiled_model->inputs[0].dimensions, compiled_model->inputs[0].stride); // since we've already populated the input buffer.
  unpack_vector(compiled_model->inputs[1].ptr, packed_input1, &compiled_model->inputs[1].dimensions, compiled_model->inputs[1].stride); // since we've already populated the input buffer.

  // do the inference. don't provide the input vector as it has already been transferred to the proper location
#if AXON_SIMULATION    
    nrf_axon_simulator_perfmodel_enable();
    nrf_axon_simulator_perfmodel_init();
#endif  
  uint32_t profiling_ticks = nrf_axon_platform_get_ticks();

  nrf_axon_platform_set_profiling_gpio();

  nrf_axon_result_e result = nrf_axon_nn_model_infer_sync(compiled_model, NULL, 0);

  nrf_axon_platform_clear_profiling_gpio();
  profiling_ticks = nrf_axon_platform_get_ticks() - profiling_ticks;
  if (0 > result) {
    return result;
  }
  if(0 < test_compare_results(compiled_model, NULL, expected_output)){
	  result = NRF_AXON_RESULT_FAILURE;
  }
#if AXON_SIMULATION    
    profiling_ticks = (uint32_t)nrf_axon_simulator_perfmodel_get_cycles();
    nrf_axon_simulator_perfmodel_disable();
#endif    
  nrf_axon_platform_printf("profiling ticks %u\n", profiling_ticks);

  return result;
}

/**
 * This function accepts a complete model input that has a single, external input, and performs a synchronous(blocking) inference.
 * This function only works for complete models that expect a single, packed external input vector. Inner layer models are not supported.
 */
static int run_test_vector_sync(const nrf_axon_nn_compiled_model_s* compiled_model, const int8_t *packed_input, const int8_t *expected_output) 
{
  // do the inference. don't provide the input vector as it has already been transferred to the proper location
  uint32_t profiling_ticks = nrf_axon_platform_get_ticks();

  nrf_axon_platform_set_profiling_gpio();

  /*
   Note: by supplying output_buffer==NULL the output results will remain in the interlayer buffer after the axon reservation is freed.
   This is safe as long as there are guaranteed to be no other threads using Axon. 
   This test code can't guarantee that the locally declared output_buffer is sized properly for the output. 
   */
  int8_t *packed_output_ptr = compiled_model->packed_output_buf != NULL ?  compiled_model->packed_output_buf :  // use the output buffer provisioned by the compiler.
          NULL;                                                                              // leave output in the interlayer buffer.
  nrf_axon_result_e result = nrf_axon_nn_model_infer_sync(compiled_model, packed_input, packed_output_ptr);
                                                                            

  /*
    inference is complete but cannot release axon until output has either been consumed or transfered to another buffer.
  */
  nrf_axon_platform_clear_profiling_gpio();
#if AXON_SIMULATION    
  profiling_ticks = (uint32_t) nrf_axon_simulator_perfmodel_get_cycles();
#else
  profiling_ticks = nrf_axon_platform_get_ticks() - profiling_ticks;
#endif    

  if (0 > result) {
    return result;
  }

  /* returns the # of mismatches. */
  result = test_compare_results(compiled_model, packed_output_ptr, expected_output); /* returns the number of mismatches */
  nrf_axon_print_test_inference_results(compiled_model, packed_output_ptr, profiling_ticks);
  
  result = result != 0 ? NRF_AXON_RESULT_FAILURE : NRF_AXON_RESULT_SUCCESS;
  return result;
}

static void test_vector_async_inference_callback(nrf_axon_result_e result, void *callback_context)
{
  nrf_axon_platform_generate_user_event();  
}
/**
 * This function accepts a complete model input that has a single external input, and performs a non-blocking inference 
 * (even though it just stalls while waiting for a result), and returns the inferred label index.
 * This function only works for complete models that expect a single, packed external input vector. Inner layer models are not supported.
 * 
 * In asynchronous mode the driver needs to copy the input to the interlayer buffer only when the model is about to run so as to not
 * corrupt any on-going axon operations. Similarly, the output needs to be copied from the interlayer buffer at model completion before
 * the next model is run. By supplying output_buffer to nrf_axon_nn_model_infer_async(), the driver will populate it with the output results.
 * 
 * Note that the driver will start the next job in the queue before invoking the current job's callback. So it isn't safe to
 * retrieve the output in the callback function.
 */

static int run_test_vector_async(nrf_axon_nn_model_async_inference_wrapper_s* model_wrapper, const int8_t *packed_input, const int8_t *expected_output) 
{
  const nrf_axon_nn_compiled_model_s* compiled_model = model_wrapper->compiled_model;

  uint32_t profiling_ticks = nrf_axon_platform_get_ticks();

  nrf_axon_platform_set_profiling_gpio();
  /* 
     set the input vector length to -1 to skip the length validation since we know the vector length is correct.
     Don't know a-priori if the local output buffer is sufficient to hold the output results. If it is, have
     the driver copy the output to it. This method guarantees that the output will not be overwritten by another inference
     if there are multiple users of axons. 
  */
  int8_t *packed_output_ptr = compiled_model->packed_output_buf != NULL ?  compiled_model->packed_output_buf :  // use the output buffer provisioned by the compiler.
          NULL;                                                                              // leave output in the interlayer buffer.

  nrf_axon_result_e result = nrf_axon_nn_model_infer_async(model_wrapper, packed_input, packed_output_ptr,                                                                            // leave output in the interlayer buffer.
            test_vector_async_inference_callback, NULL);
  
  nrf_axon_platform_wait_for_user_event();

  nrf_axon_platform_clear_profiling_gpio();
  profiling_ticks = nrf_axon_platform_get_ticks() - profiling_ticks;
  if (0 > result) {
    return result;
  }

  if(0 < test_compare_results(model_wrapper->compiled_model, packed_output_ptr, expected_output)){
    result = NRF_AXON_RESULT_FAILURE;
  }
#if AXON_SIMULATION    
    profiling_ticks = (uint32_t)nrf_axon_simulator_perfmodel_get_cycles();
#endif    
  // get the label. 
  nrf_axon_print_test_inference_results(model_wrapper->compiled_model, packed_output_ptr, profiling_ticks);

  return result;
}

int nrf_axon_nn_run_test_vectors(const nrf_axon_nn_compiled_model_s **compiled_full_models,
      const char* test_group_name, uint16_t models_count,
      const nrf_axon_nn_compiled_model_layer_s **compiled_1_layer_models[],
      uint16_t *model_layers_count,
      const nrf_axon_nn_model_test_info_s *test_vectors) 
{
  static nrf_axon_nn_model_async_inference_wrapper_s the_model_wrapper;

  nrf_axon_platform_clear_profiling_gpio(); // start in low position
  
  const char* test_name;
  if (test_group_name) {
    test_name = test_group_name;
  } else {
    test_name = test_vectors->test_name;
  }

  uint32_t test_case_cnt = 0;
  uint16_t model_ndx;
  for (model_ndx = 0; model_ndx < models_count; model_ndx++) {
    test_case_cnt += test_vectors[model_ndx].full_model_vector_count+
                  (model_layers_count==NULL ? 0 : model_layers_count[model_ndx]);
  }
  nrf_axon_platform_printf("\r\nTEST:\t%s\tCASE COUNT\t%d\n", test_name, test_case_cnt);

  uint32_t test_case_ndx = 0;
  uint32_t test_pass_cnt = 0;
  uint32_t test_fail_cnt = 0;
  for (model_ndx = 0; model_ndx < models_count; model_ndx++) {
    const nrf_axon_nn_compiled_model_s *this_full_model = compiled_full_models[model_ndx];
    
    // bind the full model for async operation since every other vector will be in async mode.
    int result = nrf_axon_nn_model_async_init(&the_model_wrapper, this_full_model);
    if (result < 0) {
      return result;
    }

    // init the persistent variables 1 time only.
    nrf_axon_nn_model_init_vars(this_full_model);
    
    /*
    * 1st loop. Process compiled-in full model test vectors
    */
  int vector_ndx=0;
  #if 1 /* set this to 0 to skip full model vectors and only execute single layer models */
    if (this_full_model->external_input_ndx < 0) {
      nrf_axon_platform_printf("ERROR! model lacks an external input!\n");
      return -1;
    }

    /* Will alternate between async and sync inference, just to exercise both paths in this test. */
    bool use_async_inference = false;
    for (; vector_ndx < test_vectors->full_model_vector_count; vector_ndx++, use_async_inference = !use_async_inference) {
      nrf_axon_platform_printf("\r\nTEST:\t%s\tSTART CASE NO\t%d\n", test_vectors[model_ndx].test_name, test_case_ndx);
      nrf_axon_platform_printf("Test inference %s vector %d FULL MODEL %s mode\n", 
        this_full_model->model_name, vector_ndx, 
        use_async_inference ? "async" : "sync");
        const int8_t *input_ptr = test_vectors[model_ndx].full_model_input_vectors[vector_ndx];

#if AXON_SIMULATION   
        if (vector_ndx == 0) {
          nrf_axon_simulator_perfmodel_enable();
          nrf_axon_simulator_perfmodel_init();
        } else {
          nrf_axon_simulator_perfmodel_disable();
        }
#endif

      if (use_async_inference) {
        result = run_test_vector_async(&the_model_wrapper, input_ptr, test_vectors[model_ndx].full_model_expected_output_vectors[vector_ndx]);
      } else {
        result = run_test_vector_sync(the_model_wrapper.compiled_model, input_ptr, test_vectors[model_ndx].full_model_expected_output_vectors[vector_ndx]);
      }
      if(NRF_AXON_RESULT_SUCCESS == result){
          nrf_axon_platform_printf("\r\nTEST:\t%s\tCASE NO\t%d\tRESULT:\t%s\n", test_vectors[model_ndx].test_name, test_case_ndx, "PASS");
          test_pass_cnt++;
        } else {
          nrf_axon_platform_printf("\r\nTEST:\t%s\tCASE NO\t%d\tRESULT:\t%s\n", test_vectors[model_ndx].test_name, test_case_ndx, "FAIL");
          test_fail_cnt++;
        }
        test_case_ndx++;
    } // for loop to run the full model on all vectors
#endif // #if 1, block to disable full model inference.

    /* 
    * next loop. process layer-by-layer models.
    */
    if ((model_layers_count != NULL) && (compiled_1_layer_models != NULL)) {
#if AXON_SIMULATION    
      nrf_axon_simulator_perfmodel_enable();
#endif 
      for (int layer_ndx=0; layer_ndx < model_layers_count[model_ndx];layer_ndx++) {
        // make sure this layer is populated
        if (NULL==compiled_1_layer_models[model_ndx][layer_ndx]) {
          continue;
        }
        // layers are only supported in sync mode, so just validate them; don't need to bind them.
        if (0 > (result=nrf_axon_nn_model_validate(&compiled_1_layer_models[model_ndx][layer_ndx]->base))) {
          return result;
        }
        nrf_axon_platform_printf("\r\nTEST:\t%s\tSTART CASE NO\t%d\n", test_vectors[model_ndx].test_name, test_case_ndx);
        printf("\nTest inference %s vector %d layer %d\n", 
          compiled_1_layer_models[model_ndx][layer_ndx]->base.model_name, 0, layer_ndx);  
        
        const int8_t *packed_input = compiled_1_layer_models[model_ndx][layer_ndx]->input0_layer_ndx < 0 ? 
            test_vectors[model_ndx].full_model_input_vectors[0] :  // negative code indicates external inpout
              test_vectors[model_ndx].layer_vectors[compiled_1_layer_models[model_ndx][layer_ndx]->input0_layer_ndx];
        const int8_t *packed_input1 = compiled_1_layer_models[model_ndx][layer_ndx]->base.input_cnt < 2 ? 
              NULL : // only 1 input
                compiled_1_layer_models[model_ndx][layer_ndx]->input1_layer_ndx < 0 ? 
                  test_vectors[model_ndx].full_model_input_vectors[0] :  // negative code indicates external inpout
                    test_vectors[model_ndx].layer_vectors[compiled_1_layer_models[model_ndx][layer_ndx]->input1_layer_ndx];

#if AXON_SIMULATION    
        nrf_axon_simulator_perfmodel_init();
#endif 

        if(NRF_AXON_RESULT_SUCCESS == run_layer_test_vector(&compiled_1_layer_models[model_ndx][layer_ndx]->base, packed_input, packed_input1, 
                                      test_vectors[model_ndx].layer_vectors[compiled_1_layer_models[model_ndx][layer_ndx]->layer_ndx])) {
          nrf_axon_platform_printf("\r\nTEST:\t%s\tCASE NO\t%d\tRESULT:\t%s\n", test_vectors[model_ndx].test_name, test_case_ndx, "PASS");
          test_pass_cnt++;
        } else {
          nrf_axon_platform_printf("\r\nTEST:\t%s\tCASE NO\t%d\tRESULT:\t%s\n", test_vectors[model_ndx].test_name,test_case_ndx++, "FAIL");
          test_fail_cnt++;
        }
        test_case_ndx++;
        vector_ndx++;
      } // for layer_ndx
    } 
  } // outer loop through all th models
 
  nrf_axon_platform_printf("\r\nTEST:\t%s\tCOMPLETE\tPASS COUNT\t%d\tFAIL COUNT\t%d\n", test_name, test_pass_cnt, test_fail_cnt);

  return 0;
}
/*
*/
void nrf_axon_nn_populate_model_test_info_s(
  nrf_axon_nn_model_test_info_s *the_struct, // structure to populate
  const char* test_name,
  const int8_t** full_model_input_vectors,  // test vectors for layer 1 (full model)
  const int8_t** full_model_expected_output_vectors, // expected output vectors, one for each input vector (full model)
  uint16_t full_model_vector_count, // number of full model test/expected_output vector pairs.
  const int8_t** layer_vectors,  // individual layer outputs. 
  uint16_t layer_cnt) // number of elements in layer_vectors 
{
  the_struct->test_name = test_name;
  the_struct->full_model_input_vectors = full_model_input_vectors;  // test vectors for layer 1 (full model)
  the_struct->full_model_expected_output_vectors = full_model_expected_output_vectors; // expected output vectors, one for each input vector (full model)
  the_struct->full_model_vector_count = full_model_vector_count; // number of full model test/expected_output vector pairs.
  the_struct->layer_vectors = layer_vectors;  // individual layer outputs. for each n, layer_models[n] input is layer_vectors[n-1] (except n=0, input is full_model_input_vectors[0]), expected output is layer_vectors[n]
  the_struct->layer_cnt = layer_cnt; // number of elements in layer_models and layer_vectors
}
