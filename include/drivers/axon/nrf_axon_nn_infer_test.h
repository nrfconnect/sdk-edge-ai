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
#include "nrf_axon_nn_infer.h"

/**
 * Test extensions to nrf_axon_nn_compiled_model_test_s to support 
 * model layer testing.
 */
typedef struct {
  nrf_axon_nn_compiled_model_s base;
  uint16_t model_layer_cnt;
  uint8_t test_start_layer;
  uint8_t test_stop_layer;
  int8_t test_input_layer;
  int8_t test_input1_layer;
} nrf_axon_nn_compiled_model_layer_s;

typedef struct {
  const char* test_name;
  const int8_t** full_model_input_vectors;  // test vectors for layer 1 (full model)
  const int8_t** full_model_expected_output_vectors; // expected output vectors, one for each input vector (full model)
  uint16_t full_model_vector_count; // number of full model test/expected_output vector pairs.
  const int8_t** layer_expected_output_vectors;  // individual layer outputs. 
  uint16_t layer_cnt; // number of elements in layer_models and layer_vectors
  int8_t test_vector_expected_output_byte_width;
} nrf_axon_nn_model_test_info_s;

void nrf_axon_nn_populate_model_test_info_s(
  nrf_axon_nn_model_test_info_s *the_struct, // structure to populate
  const char* test_name,
  const int8_t** full_model_input_vectors,  // test vectors for layer 1 (full model)
  const int8_t** full_model_expected_output_vectors, // expected output vectors, one for each input vector (full model)
  uint16_t full_model_vector_count, // number of full model test/expected_output vector pairs.
  const int8_t** layer_vectors,  // individual layer outputs. for each n, layer_models[n] input is layer_vectors[n-1] (except n=0, input is full_model_input_vectors[0]), expected output is layer_vectors[n]
  uint16_t layer_cnt, // number of elements in layer_models and layer_vectors
  int8_t test_vector_expected_output_byte_width);

int nrf_axon_nn_run_test_vectors(const nrf_axon_nn_compiled_model_s **compiled_full_models,
      char* test_group_name, uint16_t models_count,
      const nrf_axon_nn_compiled_model_layer_s **compiled_1_layer_models[],
      uint16_t *model_layers_count,
      const nrf_axon_nn_model_test_info_s *test_vectors);

int nrf_axon_print_test_inference_results(const nrf_axon_nn_compiled_model_s *compiled_model, int8_t *output_ptr, uint32_t profiling_ticks);

#ifdef __cplusplus
} // extern "C" {
#endif

