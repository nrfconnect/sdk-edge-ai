/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "axon/nrf_axon_platform.h"
#include "drivers/axon/nrf_axon_driver.h"
#include "drivers/axon/nrf_axon_nn_infer.h"
#include "drivers/axon/nrf_axon_nn_infer_test.h"
#include "axon/nrf_axon_stringization.h"

/**
 * Set this to 0 to perform a build w/o the test vector header file to verify the build process and see
 * the image size without the test vectors included. 
 * 
 * Set this one to include the test vector header file and perform inference.
 */
#define INCLUDE_VECTORS 1

/**
 * Set this to use the predefined packed output buffer.
 */
#define NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER 1
/**
 * Set this to 1 to enable the bare minimum test vectors; a single end-to-end vector.
 * Minimizes the image size while still performing an infernce. 
 * If 0, all the end-to-end test vectors in the header file will by inferenced.
 */
#define AXON_MINIMUM_TEST_VECTORS 0

/**
 * Set this to 1 to include single layer test vectors. The model will perform inference on
 * each layer separately. Single layer vectors can be large and potentially not fit in the device's memory.
 * The number of layers to include can be controlled by setting the start and stop layers.
 */
#define AXON_LAYER_TEST_VECTORS (1 && INCLUDE_VECTORS)
#define AXON_LAYER_TEST_START_LAYER 0
#define AXON_LAYER_TEST_STOP_LAYER  1000

/*
* Create the model include header file name and structure from 
* the model name.
*/
#define AXON_MODEL_FILE_NAME_ROOT nrf_axon_model_
#define AXON_MODEL_LAYERS_FILE_NAME_ROOT AXON_MODEL_FILE_NAME_ROOT
#define AXON_MODEL_TEST_VECTORS_FILE_NAME_ROOT AXON_MODEL_FILE_NAME_ROOT
#define AXON_MODEL_TEST_VECTORS_FILE_NAME_END _test_vectors_.h
#define AXON_MODEL_LAYERS_FILE_NAME_TAIL _layers_.h
#define AXON_MODEL_DOT_H _.h

#define AXON_MODEL_FILE_NAME STRINGIZE_3_CONCAT(AXON_MODEL_FILE_NAME_ROOT, NRF_AXON_MODEL_NAME, AXON_MODEL_DOT_H)
#define AXON_MODEL_FILE_LAYERS_NAME STRINGIZE_3_CONCAT(AXON_MODEL_LAYERS_FILE_NAME_ROOT, NRF_AXON_MODEL_NAME, AXON_MODEL_LAYERS_FILE_NAME_TAIL)
#define AXON_MODEL_TEST_VECTORS_FILE_NAME STRINGIZE_3_CONCAT(AXON_MODEL_TEST_VECTORS_FILE_NAME_ROOT, NRF_AXON_MODEL_NAME, AXON_MODEL_TEST_VECTORS_FILE_NAME_END)

// generate structure name model_<model_name>
#define THE_REAL_MODEL_STRUCT_NAME(model_name) model_##model_name
#define THE_MODEL_STRUCT_NAME(model_name) THE_REAL_MODEL_STRUCT_NAME(model_name)

// generate structure name model_<model_name>_layer_list
#define THE_REAL_MODEL_LAYERS_STRUCT_NAME(model_name) model_##model_name##_layer_list
#define THE_MODEL_LAYERS_STRUCT_NAME(model_name) THE_REAL_MODEL_LAYERS_STRUCT_NAME(model_name)

// generate structure name model_<model_name>_test_vectors
#define THE_REAL_MODEL_TEST_VECTORS_STRUCT_NAME(model_name) model_##model_name##_test_vectors
#define THE_MODEL_TEST_VECTORS_STRUCT_NAME(model_name) THE_REAL_MODEL_TEST_VECTORS_STRUCT_NAME(model_name)

// generate structure name <model_name>_input_test_vectors
#define THE_REAL_TEST_INPUT_VECTORS_LIST_NAME(model_name) model_name##_input_test_vectors
#define THE_TEST_INPUT_VECTORS_LIST_NAME(model_name) THE_REAL_TEST_INPUT_VECTORS_LIST_NAME(model_name)

// generate structure name <model_name>_expected_output_test_vectors
#define THE_REAL_expected_output_vectors_NAME(model_name) model_name##_expected_output_vectors
#define THE_expected_output_vectors_NAME(model_name) THE_REAL_expected_output_vectors_NAME(model_name)

// generate structure name <model_name>_layer_vectors
#define THE_REAL_layer_vectors_NAME(model_name) model_name##_layer_expected_output_vectors
#define THE_layer_vectors_NAME(model_name) THE_REAL_layer_vectors_NAME(model_name)

#define THE_TEST_NAME_ROOT test_nn_inference_

nrf_axon_nn_compiled_model_s const *the_full_model_static_info[1];
nrf_axon_nn_compiled_model_layer_s const **the_model_layers_static_info[1] = {NULL};
uint16_t model_layers_count[] = {0};
nrf_axon_nn_model_test_info_s the_test_vectors[] = {0};

#include AXON_MODEL_FILE_NAME
#if INCLUDE_VECTORS
# include AXON_MODEL_TEST_VECTORS_FILE_NAME
# if AXON_LAYER_TEST_VECTORS 
#   include AXON_MODEL_FILE_LAYERS_NAME
# endif
#endif
int AxonnnModelPrepare() {
  the_full_model_static_info[0] = &THE_MODEL_STRUCT_NAME(NRF_AXON_MODEL_NAME);
#if INCLUDE_VECTORS
  nrf_axon_nn_populate_model_test_info_s(
    &the_test_vectors[0], // structure to populate
    STRINGIZE_2_CONCAT(THE_TEST_NAME_ROOT, NRF_AXON_MODEL_NAME),
    (const int8_t**)THE_TEST_INPUT_VECTORS_LIST_NAME(NRF_AXON_MODEL_NAME),  // test vectors for full model
    (const int8_t**)THE_expected_output_vectors_NAME(NRF_AXON_MODEL_NAME), // expected output vectors, one for each input vector (full model)
    sizeof(THE_expected_output_vectors_NAME(NRF_AXON_MODEL_NAME))/sizeof(*THE_expected_output_vectors_NAME(NRF_AXON_MODEL_NAME)), // number of full model test/expected_output vector pairs.
    (const int8_t**)THE_layer_vectors_NAME(NRF_AXON_MODEL_NAME),  // individual layer outputs. for each n, layer_models[n] input is layer_vectors[n-1] (except n=0, input is full_model_input_vectors[0]), expected output is layer_vectors[n]
    sizeof(THE_layer_vectors_NAME(NRF_AXON_MODEL_NAME))/sizeof(*THE_layer_vectors_NAME(NRF_AXON_MODEL_NAME))); // number of elements in layer_vectors
# if AXON_LAYER_TEST_VECTORS 
  the_model_layers_static_info[0] = THE_MODEL_LAYERS_STRUCT_NAME(NRF_AXON_MODEL_NAME);
  model_layers_count[0] = sizeof(THE_MODEL_LAYERS_STRUCT_NAME(NRF_AXON_MODEL_NAME)) / sizeof(*THE_MODEL_LAYERS_STRUCT_NAME(NRF_AXON_MODEL_NAME));
# endif
#endif

  return 0;
}


void base_inference_main(void)
{
  nrf_axon_platform_printf("Start Platform!\n");

  nrf_axon_result_e result = nrf_axon_platform_init();
  
  if (result != NRF_AXON_RESULT_SUCCESS) {
    nrf_axon_platform_printf("axon_platform_init failed!\n");
  }

  nrf_axon_platform_printf("Prepare and run Axon!\n");

  if (0 > (result = AxonnnModelPrepare())) {
    return; // failed!
  } //AxonnnPrepare();

  // run the internal test vectors
  nrf_axon_nn_run_test_vectors(the_full_model_static_info,
      NULL, 1, 
      the_model_layers_static_info,
      model_layers_count,
      the_test_vectors); 

  nrf_axon_platform_printf("test_nn_inference complete!\n");
  nrf_axon_platform_close();
}
#if (NOT_A_ZEPHYR_BUILD)
int main(int argc, char* argv[]) 
{ 
  base_inference_main();
}
#endif