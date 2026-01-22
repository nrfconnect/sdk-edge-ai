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

#include "stdint.h"
#include "stdarg.h"
extern uint8_t ML_LOGGING_DISABLE_PRINT;
/*
 * This functions will print a vector of the associated type and specified length out the
 * debug port. The printed format is:
 * <type of vector_ptr> <name>[<count>]=\r\n
 * {<vector_ptr[0]>,<vector_ptr[1]>, <...>,<vector_ptr[count-1]>}\r\n
 * For example
 * int32_t my_int32_vector[10]=
 * {1,-1,3,0,7,-1000,-10,8,9,10}
 */
void print_int64_vector(const char *name, const int64_t *vector_ptr, uint32_t count, uint8_t stride);
void print_hex64_vector(const char *name, const int64_t *vector_ptr, uint32_t count, uint8_t stride);
void print_float_vector(const char *name, const float *vector_ptr, uint32_t count, uint8_t stride);
void print_hex32_vector(const char *name, const uint32_t *vector_ptr, uint32_t count, uint8_t stride);
void print_int32_vector(const char *name, const int32_t *vector_ptr, uint32_t count, uint8_t stride);
void print_int16_vector(const char *name, const int16_t *vector_ptr, uint32_t count, uint8_t stride);
void print_int16_circ_buffer(const char *name, const int16_t *vector_ptr, uint32_t count, uint8_t stride, uint32_t start_offset);
void print_int8_vector(const char *name, const int8_t *vector_ptr, uint32_t count);
void PrintVector(const char *name, const uint8_t *vector_ptr, uint32_t count, uint8_t element_size);

int verify_vectors(const char *msg, const int32_t *output, const int32_t* expected_output, uint32_t count, uint32_t margin);
int verify_vectors_8(const char *msg, const int8_t *output, const int8_t* expected_output, uint32_t count, uint32_t margin);
int verify_vectors_16(const char *msg, const int16_t *output, const int16_t* expected_output, uint32_t count, uint32_t margin);
int verify_scalar_output_stride(const char *msg, const int32_t *output, const int32_t expected_output, uint32_t count, uint32_t margin, uint8_t extra_output_stride);
int verify_vectors_output_stride(const char *msg, const int32_t *output, const int32_t* expected_output, uint32_t count, uint32_t margin, uint8_t extra_output_stride);

void nrf_axon_platform_set_profiling_gpio();
void nrf_axon_platform_clear_profiling_gpio();


#ifdef __cplusplus
}
#endif
