/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "nrf_axon_logging.h"
#include "nrf_axon_platform.h"


void print_float_vector(const char *name, const float *vector_ptr, uint32_t count, uint8_t stride) {
  char printbuffer[20]; // sized big enough for longest integer.
  nrf_axon_platform_printf("float ");;

  nrf_axon_platform_printf(name);
  snprintf(printbuffer, 20, "[%d] = {\r\n", count);
  nrf_axon_platform_printf(printbuffer);
  while (count--) {
    snprintf(printbuffer, 20, "%f,", *vector_ptr);
    vector_ptr+=stride;
    nrf_axon_platform_printf(printbuffer);
  }
  nrf_axon_platform_printf("\r\n}\r\n");
}

void print_int32_vector(const char *name, const int32_t *vector_ptr, uint32_t count, uint8_t stride) {
  char printbuffer[20]; // sized big enough for longest integer.

  nrf_axon_platform_printf("int32_t ");;

  nrf_axon_platform_printf(name);
  snprintf(printbuffer, 20, "[%d] = {\r\n", count);
  nrf_axon_platform_printf(printbuffer);
  while (count--) {
    snprintf(printbuffer, 20, "%d,", *vector_ptr);
    vector_ptr+=stride;
    nrf_axon_platform_printf(printbuffer);
  }
  nrf_axon_platform_printf("\r\n}\r\n");
}
void print_hex32_vector(const char *name, const uint32_t *vector_ptr, uint32_t count, uint8_t stride) {
  char printbuffer[20]; // sized big enough for longest integer.

  nrf_axon_platform_printf("int32_t ");;

  nrf_axon_platform_printf(name);
  snprintf(printbuffer, 20, "[%d] = {\r\n", count);
  nrf_axon_platform_printf(printbuffer);
  while (count--) {
    snprintf(printbuffer, 20, "0x%x,", *vector_ptr);
    vector_ptr+=stride;
    nrf_axon_platform_printf(printbuffer);
  }
  nrf_axon_platform_printf("\r\n}\r\n");
}


void print_int16_vector(const char *name, const int16_t *vector_ptr, uint32_t count, uint8_t stride) {
  print_int16_circ_buffer(name, vector_ptr, count, stride, 0);
}
void print_int16_circ_buffer(const char *name, const int16_t *vector_ptr, uint32_t count, uint8_t stride, uint32_t start_index) {
  char printbuffer[20]; // sized big enough for longest integer.
  uint32_t sample_ndx;

  nrf_axon_platform_printf("int16_t ");
  nrf_axon_platform_printf(name);
  snprintf(printbuffer, 20, "[%d] = {\r\n", count);
  nrf_axon_platform_printf(printbuffer);
  start_index *= stride;
  for (sample_ndx=0; sample_ndx<count; sample_ndx++) {
    snprintf(printbuffer, 20, "%d,", vector_ptr[start_index]);
    nrf_axon_platform_printf(printbuffer);
    start_index+=stride;
    if (start_index >= count * stride) {
      start_index=0;
    }
  }
  nrf_axon_platform_printf("\r\n}\r\n");

}

void print_int8_vector(const char *name, const int8_t *vector_ptr, uint32_t count) {
  char printbuffer[20]; // sized big enough for longest integer.
  nrf_axon_platform_printf("int8_t ");

  nrf_axon_platform_printf(name);
  snprintf(printbuffer, 20, "[%d] = {\r\n", count);
  nrf_axon_platform_printf(printbuffer);
  while (count--) {
    snprintf(printbuffer, 20, "%d,", *vector_ptr++);
    nrf_axon_platform_printf(printbuffer);
  }
  nrf_axon_platform_printf("\r\n}\r\n");
}


/*
 * utility for printing a vector to the debug console.
 */
void PrintVector(const char *name, const uint8_t *vector_ptr, uint32_t count, uint8_t element_size) {
  char printbuffer[20]; // sized big enough for longest integer.
  int32_t element_value = 0;
  switch (element_size) {
  case 1: nrf_axon_platform_printf("int8_t "); break;
  case 2: nrf_axon_platform_printf("int16_t "); break;
  case 4: nrf_axon_platform_printf("int32_t "); break;
  default: return;
  }

  nrf_axon_platform_printf(name);
  nrf_axon_platform_printf(" = {\r\n");
  while (count--) {
    memcpy(&element_value, vector_ptr, element_size);
    snprintf(printbuffer, 20, "%d,", element_value);
    nrf_axon_platform_printf(printbuffer);
    vector_ptr+=element_size;
  }
  nrf_axon_platform_printf("\r\n}\r\n");
}

void print_int64_vector(const char *name, const int64_t *vector_ptr, uint32_t count, uint8_t stride) {
  char printbuffer[20]; // sized big enough for longest integer.

  nrf_axon_platform_printf("int64_t ");;

  nrf_axon_platform_printf(name);
  snprintf(printbuffer, 20, "[%d] = {\r\n", count);
  nrf_axon_platform_printf(printbuffer);
  while (count--) {
    snprintf(printbuffer, 20, "%I64d,", *vector_ptr);
    vector_ptr+=stride;
    nrf_axon_platform_printf(printbuffer);
  }
  nrf_axon_platform_printf("\r\n}\r\n");
}
void print_hex64_vector(const char *name, const int64_t *vector_ptr, uint32_t count, uint8_t stride) {
  char printbuffer[20]; // sized big enough for longest integer.

  nrf_axon_platform_printf("int64_t ");;

  nrf_axon_platform_printf(name);
  snprintf(printbuffer, 20, "[%d] = {\r\n", count);
  nrf_axon_platform_printf(printbuffer);
  while (count--) {
    snprintf(printbuffer, 20, "0x%I64x,", *vector_ptr);
    vector_ptr+=stride;
    nrf_axon_platform_printf(printbuffer);
  }
  nrf_axon_platform_printf("\r\n}\r\n");
}
