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
/**
 * @brief group of functions for printing vectors of different formats out the console.
 * The printed format is:
 * <type of vector_ptr> <name>[<count>]=\r\n
 * {<vector_ptr[0]>,<vector_ptr[1]>, <...>,<vector_ptr[count-1]>}\r\n
 * For example
 * int32_t my_int32_vector[10]=
 * {1,-1,3,0,7,-1000,-10,8,9,10}
 * 
 * @param[in] name Symbol name of the vector.
 * @param[in] vector_ptr pointer to the vector to print.
 * @param[in] count Number of elements in vector_ptr to print.
 * @param[in] stride Number of elements to moveby after each print; ie, 1 means to print every element, 2 skips every other element.
 */
/**
 * @brief prints an int64 vector in decimal.
 */
void nrf_axon_print_int64_vector(const char *name, const int64_t *vector_ptr, uint32_t count, uint8_t stride);
/**
 * @brief prints an int64 vector in hexidecimal.
 */
void nrf_axon_print_hex64_vector(const char *name, const int64_t *vector_ptr, uint32_t count, uint8_t stride);
/**
 * @brief prints an float32 vector in decimal.
 */
void nrf_axon_print_float_vector(const char *name, const float *vector_ptr, uint32_t count, uint8_t stride);

/**
 * @brief prints an int32 vector in hexidecimal.
 */
void nrf_axon_print_hex32_vector(const char *name, const uint32_t *vector_ptr, uint32_t count, uint8_t stride);
/**
 * @brief prints an int32 vector in decimal.
 */
void nrf_axon_print_int32_vector(const char *name, const int32_t *vector_ptr, uint32_t count, uint8_t stride);
/**
 * @brief prints an int16 vector in decimal.
 */
void nrf_axon_print_int16_vector(const char *name, const int16_t *vector_ptr, uint32_t count, uint8_t stride);
/**
 * @brief prints an int32 circular vector in decimal.
 * @param[in] start_offset is the 1st element printed. Wraps to element 0 after count-start_offset
 */
void nrf_axon_print_int16_circ_buffer(const char *name, const int16_t *vector_ptr, uint32_t count, uint8_t stride, uint32_t start_offset);
/**
 * @brief prints an int8 vector in decimal.
 */
void nrf_axon_print_int8_vector(const char *name, const int8_t *vector_ptr, uint32_t count);

/**
 * @brief prints an vector in decimal. 
 * @param[in] element_size determines the type of the input (4=int32, 2=int16, 1=int8)
 */
void nrf_axon_print_vector(const char *name, const uint8_t *vector_ptr, uint32_t count, uint8_t element_size);

/**
 * @brief Group of funcitons for comparing 2 vectors
 * 
 * Prints a message in the format:
 * verify <msg>...
 * Logs an error whenever abs(output[ndx] - expected_output[ndx]) > margin.
 * 
 * @param[in] msg Initial message to display before starting the comparison.
 * @param[in] output Test values.
 * @param[in] expected_output Values output should match.
 * @param[in] count Number of elements to compare.
 * @param[in] margin Maximum tolerated difference betwee output[ndx] and expected_output[ndx]. 0 means exact match.
 * @retval Number of mismatches.
 */
int nrf_axon_verify_vectors(const char *msg, const int32_t *output, const int32_t* expected_output, uint32_t count, uint32_t margin);
int nrf_axon_verify_vectors_8(const char *msg, const int8_t *output, const int8_t* expected_output, uint32_t count, uint32_t margin);
int nrf_axon_verify_vectors_16(const char *msg, const int16_t *output, const int16_t* expected_output, uint32_t count, uint32_t margin);
int nrf_axon_verify_scalar_output_stride(const char *msg, const int32_t *output, const int32_t expected_output, uint32_t count, uint32_t margin, uint8_t extra_output_stride);
int nrf_axon_verify_vectors_output_stride(const char *msg, const int32_t *output, const int32_t* expected_output, uint32_t count, uint32_t margin, uint8_t extra_output_stride);

/**
 * @brief Controls a GPIO that has been designated as the Axon profiling GPIO in the board.dts file.
 */
void nrf_axon_platform_set_profiling_gpio();
void nrf_axon_platform_clear_profiling_gpio();


#ifdef __cplusplus
}
#endif
