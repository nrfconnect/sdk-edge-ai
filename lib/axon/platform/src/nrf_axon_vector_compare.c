/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include "axon/nrf_axon_platform.h"
#if (NOT_A_ZEPHYR_BUILD)
#include "axon/nrf_axon_platform_simulator.h"
#endif

#define axon_abs_u32(x) ((uint32_t)((int32_t)x<0?-1*(int32_t)x:x))



int nrf_axon_verify_scalar_output_stride(const char *msg, const int32_t *output, int32_t expected_output, uint32_t count, uint32_t margin, uint8_t extra_output_stride) {
  int err_cnt = 0;
  int error_dif;

  nrf_axon_platform_printf("Verify %s... ", msg);

  for (unsigned i = 0; i < count; i++) {
    error_dif = output[i * (1+extra_output_stride)] - expected_output;
    if (axon_abs_u32(error_dif) > margin) {
      nrf_axon_platform_printf(" error @ %d: dif %d not less than %d: got 0x%.8lx, expected 0x%.8lx\r\n", i, error_dif, margin, output[i], expected_output);
      err_cnt++;
    }
  }
  // return error in case of mismatch
  if (err_cnt > 0) {
    nrf_axon_platform_printf(" FAILED!: %d mismatches!\r\n", err_cnt);
    return (-1);
  } else {
    nrf_axon_platform_printf(" PASSED!\r\n");
    return (0);
  }

}

/*
 * Verifies output[] == expected_output[] with each value being within margin of each other.
 */
int nrf_axon_verify_vectors_output_stride(const char *msg, const int32_t *output, const int32_t* expected_output, uint32_t count, uint32_t margin, uint8_t extra_output_stride) {
  int err_cnt = 0;
  int error_dif;

  nrf_axon_platform_printf("Verify %s... ", msg);

  for (unsigned i = 0; i < count; i++) {
    error_dif = output[i * (1+extra_output_stride)] - expected_output[i];
    if (axon_abs_u32(error_dif) > margin) {
      nrf_axon_platform_printf(" error @ %d: dif %d not less than %d: got 0x%.8lx, expected 0x%.8lx\r\n", i, error_dif, margin, output[i], expected_output[i]);
      err_cnt++;
    }
  }
  // return error in case of mismatch
  if (err_cnt > 0) {
    nrf_axon_platform_printf(" FAILED!: %d mismatches!\r\n", err_cnt);
    return (-1);
  } else {
    nrf_axon_platform_printf(" PASSED!\r\n");
    return (0);
  }

}

int nrf_axon_verify_vectors(const char *msg, const int32_t *output, const int32_t* expected_output, uint32_t count, uint32_t margin, uint8_t extra_output_stride) {
  return nrf_axon_verify_vectors_output_stride(msg, output, expected_output, count, margin, 0);
}


/*
 * verifies int16_t vectors
 */
int nrf_axon_verify_vectors_16(const char *msg, const int16_t *output, const int16_t* expected_output, uint32_t count, uint32_t margin) {
  int err_cnt = 0;
  int error_dif;

  nrf_axon_platform_printf("Verify %s... ", msg);

  for (unsigned i = 0; i < count; i++) {
    error_dif = output[i] - expected_output[i];
    if (axon_abs_u32(error_dif) > margin) {
      nrf_axon_platform_printf("error @ %d: dif %d not less than %d: got 0x%.8lx, expected 0x%.8lx\r\n", i, error_dif, margin, output[i], expected_output[i]);
      err_cnt++;
    }
  }
  // return error in case of mismatch
  if (err_cnt > 0) {
    nrf_axon_platform_printf(" FAILED!: %d mismatches!\r\n", err_cnt);
    return (-1);
  } else {
    nrf_axon_platform_printf(" PASSED!\r\n");
    return (0);
  }

}

/*
 * verifies int8_t vectors
 */
int nrf_axon_verify_vectors_8(const char *msg, const int8_t *output, const int8_t* expected_output, uint32_t count, uint32_t margin) {
  int err_cnt = 0;
  int error_dif;

  nrf_axon_platform_printf("Verify %s... ", msg);

  for (unsigned i = 0; i < count; i++) {
    error_dif = output[i] - expected_output[i];
    if (axon_abs_u32(error_dif) > margin) {
      nrf_axon_platform_printf("error @ %d: dif %d not less than %d: got 0x%.8lx, expected 0x%.8lx\r\n", i, error_dif, margin, output[i], expected_output[i]);
      err_cnt++;
    }
  }
  // return error in case of mismatch
  if (err_cnt > 0) {
    nrf_axon_platform_printf("FAILED!: %d mismatches!\r\n", err_cnt);
    return (-1);
  } else {
    nrf_axon_platform_printf("PASSED!\r\n");
    return (0);
  }

}

