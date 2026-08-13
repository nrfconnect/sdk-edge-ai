/*
* Copyright (c) 2025-26 Nordic Semiconductor ASA
*
* SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
*/

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "drivers/axon/nrf_axon_driver.h"
#include "axon/nrf_axon_platform.h"
#include "drivers/axon/nrf_axon_dsp_intrinsics.h"
#include "axon/nrf_axon_logging.h"
#include "./nrf_axon_app_test_dsp_intrinsics_vectors.h"

#if AXON_SIMULATION
# include "axon/nrf_axon_platform_simulator.h"
#endif

#define TEST_NAME "AXON_INTRINSICS"
#define ARRAY_LENGTH(the_array) (sizeof(the_array) / sizeof(*the_array))

/**
 * buffer to place test output vectors.
 */
static union {
	int16_t as_i16[2000];
	int32_t as_i32[1024];
} test_results;

/**
 * xty_16_16_32_extra_output_stride tests
 * Perform 5 tests:
 * main_vector_16[0:..] * main_vector[500:...], length=278, round 0, extra_output_stride 0
 * main_vector_16[0:..] * main_vector[500:...], length = 75, round 2, extra_output_stride 0
 * main_vector_16[0:..] * main_vector[500:...], length = 75, round 0, extra_output_stride 1
 * main_vector_16[0:..] * main_vector[500:...], length = 20000, round 0, extra_output_stride 0
 * main_vector_16[0:..] * main_vector[500:...], length = 50, round 0, extra_output_stride 50
 */
#define xty_16_16_32_output_stride_test_cnt 5
static int xty_16_16_32_output_stride_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("xty_16_16_32_output_stride_tests: %d cases\n",
		xty_16_16_32_output_stride_test_cnt);
	/* case 0: no rounding, no extra stride */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xty_16_16_32_output_stride(main_vector_16 + 0, main_vector_16 + 500,
		test_results.as_i32, 278, 0, 0, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			xty_16_16_32_output_stride_round_0_expected_output, 278, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 1: 2bits of rounding, no extra stride */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xty_16_16_32_output_stride(main_vector_16 + 0, main_vector_16 + 500,
			test_results.as_i32, 75, 2, 0, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			xty_16_16_32_output_stride_round_2_expected_output, 75, 1, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 2: round 0, extra stride 1 */
	memset(test_results.as_i32, -1, sizeof(test_results.as_i32));
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xty_16_16_32_output_stride(main_vector_16 + 0, main_vector_16 + 500,
			test_results.as_i32, 80, 0, 1, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			xty_16_16_32_output_stride_round_0_expected_output, 80, 0, 1) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 3: invalid round */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xty_16_16_32_output_stride(main_vector_16 + 0, main_vector_16 + 500,
		test_results.as_i32, 80, 35, 1, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result >= 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! did not catch invalid round value %d\n", 25);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 4: invalid length */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xty_16_16_32_output_stride(main_vector_16 + 0, main_vector_16 + 500,
		test_results.as_i32, 8000, 0, 0, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result >= 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! did not catch invalid length value %d\n", 8000);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;
	return pass_cnt;
}

/**
 * xty_16_16_32 tests
 * Perform 4 tests:
 * main_vector_16[0:..] * main_vector[500:...], length=278, round 0, (2 commands)
 * main_vector_16[0:..] * main_vector[500:...], length = 64, round 2, (1 2D command)
 * main_vector_16[0:..] * main_vector[500:...], length = 15, round 0, (1 1D command)
 * main_vector_16[0:..] * main_vector[500:...], length = 50, round 35, (invalid round)
 */
#define xty_16_16_32_test_cnt 4
int xty_16_16_32_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("xty_16_16_32_tests: %d cases\n", xty_16_16_32_test_cnt);
	/* case 0: no rounding */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xty_16_16_32(main_vector_16 + 0, main_vector_16 + 500, test_results.as_i32,
			278, 0, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			xty_16_16_32_round_0_expected_output, 278, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 1: 2bits of rounding */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xty_16_16_32(main_vector_16 + 0, main_vector_16 + 500, test_results.as_i32,
		64, 2, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			xty_16_16_32_round_2_expected_output, 64, 1, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 2: round 0 */
	memset(test_results.as_i32, -1, sizeof(test_results.as_i32));
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xty_16_16_32(main_vector_16 + 0, main_vector_16 + 500, test_results.as_i32,
		15, 0, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32, xty_16_16_32_round_0_expected_output, 15, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 3: invalid round */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xty_16_16_32(main_vector_16 + 0, main_vector_16 + 500, test_results.as_i32,
		80, 35, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result >= 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! did not catch invalid round value %d\n", 25);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * memset_32_extra_output_stride tests
 * Perform 3 tests:
 * Fill 100 words with 0xabcd, no extra stride
 * Fill 500 words with 0xcdef, extra stride 1
 * Fill 10000 words -> should fail due to excessive length.
 */
#define memset_32_output_stride_test_cnt 3
int memset_32_output_stride_tests(int *test_ndx)
{
	int pass_cnt = 0;
	int16_t set_val;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("memset_32_output_stride tests: %d cases\n",
		memset_32_output_stride_test_cnt);
	/* case 0: no extra stride */
	set_val = 0xabcd;
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_memset_32_output_stride(set_val, test_results.as_i32, 100, 0,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_scalar_output_stride("", test_results.as_i32, set_val,
			100, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 1: with extra stride */
	set_val = 0x1234;
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_memset_32_output_stride(set_val, test_results.as_i32, 500, 1,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_scalar_output_stride("",
			test_results.as_i32, set_val, 500, 0, 1) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 2: invalid length */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_memset_32_output_stride(0xcdef, test_results.as_i32, 10000, 1,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result >= 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! did not catch invalid length value %d\n", 8000);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;
	return pass_cnt;
}


/**
 * saturate_32_24 tests
 * Perform 3 tests:
 * saturate_32_24_test_vector[5:25], length=11, tests single command, 1xwidth
 * saturate_32_24_test_vector[0:32], length=11, tests single command, 2x16
 * saturate_32_24_test_vector[0:length], length=93, tests 2 commands, 5x16, 2x13
 *
 */
#define saturate_32_24_test_cnt 3
int saturate_32_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("saturate_32_24_tests: %d cases\n", saturate_32_24_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_saturate_32_24(saturate_32_24_test_vector + 15, test_results.as_i32, 11,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			saturate_32_24_expected_output + 15, 11, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 1: single command, 2 rows */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_saturate_32_24(saturate_32_24_test_vector, test_results.as_i32, 32,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			saturate_32_24_expected_output, 32, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 2: 2 commands */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_saturate_32_24(saturate_32_24_test_vector, test_results.as_i32,
		ARRAY_LENGTH(saturate_32_24_test_vector),
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			saturate_32_24_expected_output,
			ARRAY_LENGTH(saturate_32_24_test_vector),
			0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}


/**
 * fft_24 tests
 * Perform 1 test:
 * fft_24_test_vector, 512 tap FFT
 *
 */
#define fft_24_test_cnt 1
int fft_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("fft_24_tests: %d cases\n", fft_24_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_fft_24(fft_24_test_vector, test_results.as_i32, 9, false, false,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			fft_24_expected_output, 1 << (9 + 1), 0, 0) != 0) {
		nrf_axon_platform_printf(
		"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * xspys_24_24_24_input_stride2 tests
 * Perform 1 test:
 * Sums the output of the complex fft and rounds by 11.
 *
 */
#define xspys_24_24_24_input_stride2_test_cnt 1
int xspys_24_24_24_input_stride2_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("xspys_24_24_24_input_stride2_tests: %d cases\n",
		xspys_24_24_24_input_stride2_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xspys_24_24_24_input_stride2(fft_24_expected_output,
		fft_24_expected_output + 1,test_results.as_i32, 512, 11,
			NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			xspys_24_24_24_input_stride2_expected_output, 512, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * fft_power_24 tests
 * Combines fft_24 and xspys_24_24_24_input_stride2
 */
#define fft_power_24_test_cnt 1
int fft_power_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("\n\nfft_power_24_tests: %d cases\n", fft_24_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_fft_power_24(fft_24_test_vector, test_results.as_i32, 9, false, 2,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			xspys_24_24_24_input_stride2_expected_output, 512, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}
/**
 * xspys_24_24_24 tests
 * Perform 1 test: main_vector[0] x main_vector[500]
 */
#define xspys_24_24_24_test_cnt 1
int xspys_24_24_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("xspys_24_24_24_tests: %d cases\n", xspys_24_24_24_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xspys_24_24_24(main_vector_32, main_vector_32 + 500,test_results.as_i32,
		500, 0, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			xspys_24_24_24_expected_output, 500, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * abs_24_24_24 tests
 * Perform 1 test: main_vector[0] x main_vector[500]
 */
#define abs_24_24_24_test_cnt 1
int abs_24_24_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("abs_24_24_24_tests: %d cases\n", abs_24_24_24_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_abs_24_24_24(main_vector_32, main_vector_32 + 500,test_results.as_i32,
		500, 12, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			abs_24_24_24_expected_output, 500, 1, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * mar_16_24_32 tests
 * Perform 1 test: mar main_vector[500:699] round 3
 */
#define mar_16_24_32_test_cnt 1
int mar_16_24_32_tests(int *test_ndx)
{
	int pass_cnt = 0;
	static int32_t mar_16_24_32_expected_output[] = {-1186516,};
	nrf_axon_result_e result;

	nrf_axon_platform_printf("mar_16_24_32_tests: %d cases\n", mar_16_24_32_test_cnt);
	/* case 0: mar */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_mar_16_24_32(main_vector_16, main_vector_32 + 500,test_results.as_i32,
		52, 4, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			mar_16_24_32_expected_output, 1, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * mar_16_24_24 tests
 * Perform 2 test: mar main_vector[0:51] x main_vector[500:551] round 4
 * marx main_vector[52:103] x main_vector[500:551] round 8
 * mar main_vector[52:103] x main_vector[500:551] round 4 (should saturate)
 */
#define mar_16_24_24_test_cnt 3
int mar_16_24_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	static int32_t mar_16_24_24_expected_output[] = {-1186516,};
	static int32_t marx_16_24_32_expected_output[] = {825197,};
	static int32_t mar_16_24_24_saturated_expected_output[] = {(1 << 23) - 1,};
	nrf_axon_result_e result;

	nrf_axon_platform_printf("mar_16_24_24_tests: %d cases\n", mar_16_24_24_test_cnt);
	/*
	 * case 0: mar. Need to keep the axon reservation from this operation to leave power
	 * enabled to axon so that subsequent marx does not fail.
	 */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_mar_16_24_24(main_vector_16, main_vector_32 + 500,test_results.as_i32,
		52, 4, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, true);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			mar_16_24_24_expected_output, 1, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 1: marx */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	nrf_axon_platform_printf("\nmarx_16_24\n");
	result = nrf_axon_marx_16_24(main_vector_16 + 52,test_results.as_i32, 52, 8,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			marx_16_24_32_expected_output, 1, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;
	/* case 2: mar should saturate */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_mar_16_24_24(main_vector_16 + 52, main_vector_32 + 500, test_results.as_i32,
		52, 4, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			mar_16_24_24_saturated_expected_output, 1, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;
	return pass_cnt;
}

/**
 * sqrt_24 tests
 * Perform 3 tests:
 * abs_main_vector_32, length = 512
 * abs_main_vector_32, length = 514 (too long)
 * abs_main_vector_32, length = 2 (too short)
 * abs_main_vector_32, length = 511 (not a multiple of 2)
 */
#define sqrt_24_test_cnt 4
int sqrt_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("sqrt_24_tests: %d cases\n", sqrt_24_test_cnt);
	/* case 0: no rounding */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_sqrt_24(abs_main_vector_32, test_results.as_i32, 512,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			sqrt_24_expected_output, 512, 1, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 1: length=514 (too long) */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_sqrt_24(abs_main_vector_32, test_results.as_i32, 514,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result >= 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! did not catch length too long %d\n", 512);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 2: length=2 (too short) */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_sqrt_24(abs_main_vector_32, test_results.as_i32, 1,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result >= 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! did not catch length too short %d\n", 1);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 3: length=511 (not a multiple of 2) */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_sqrt_24(abs_main_vector_32, test_results.as_i32, 511,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result >= 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! did not catch length not a multiple of 2 %d\n", 511);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * logn_11p12 tests
 * Perform 3 tests:
 * abs_main_vector_32, length = 512
 * abs_main_vector_32, length = 514 (too long)
 * abs_main_vector_32, length = 2 (too short)
 * abs_main_vector_32, length = 511 (not a multiple of 2)
 */
#define logn_11p12_test_cnt 4
int logn_11p12_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("logn_11p12_tests: %d cases\n", logn_11p12_test_cnt);
	/* case 0: no rounding */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_logn_11p12(abs_main_vector_32, test_results.as_i32, 512,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			logn_expected_output, 512, 4, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 1: length=514 (too long) */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_logn_11p12(abs_main_vector_32, test_results.as_i32, 514,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result >= 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! did not catch length too long %d\n", 514);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 2: length=2 (too short) */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_logn_11p12(abs_main_vector_32, test_results.as_i32, 1,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result >= 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! did not catch length too short %d\n", 1);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 3: length=511 (not a multiple of 2) */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_logn_11p12(abs_main_vector_32, test_results.as_i32, 511,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result >= 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! did not catch length not a multiple of 2 %d\n", 511);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * exp_11p12 tests
 * Perform 3 tests:
 * abs_main_vector_32, length = 512
 * abs_main_vector_32, length = 514 (too long)
 * abs_main_vector_32, length = 2 (too short)
 * abs_main_vector_32, length = 511 (not a multiple of 2)
 */
#define exp_11p12_test_cnt 4
int exp_11p12_tests(int *test_ndx)
{
	int pass_cnt = 0;
	static const int32_t exp_expected_output[512] = {
		1,79,161,195,240,285,318,314,240,182,142,70,23,118,190,293,329,426,491,513,512,481,414,345,190,14,160,342,448,631,717,740,719,611,453,366,180,64,48,124,196,180,176,193,172,169,187,190,254,353,344,415,438,414,390,234,111,41,195,340,483,588,670,706,645,445,326,192,44,129,330,361,171,7,190,455,900,1496,1378,1334,1491,1026,322,763,1901,3120,3494,4202,4615,4333,3578,3010,1510,184,1428,2675,3603,4115,4369,4396,3863,2595,1934,1437,489,430,1032,1055,862,814,759,441,23,158,141,454,654,1245,1813,2828,3152,3035,2678,1917,1008,345,1581,2614,3657,4191,4358,4024,2751,1515,140,1548,3136,3821,4184,4195,3529,2941,1578,71,1272,2079,2936,3095,2746,1828,1019,127,752,1383,1427,1357,1155,458,375,1296,1899,2657,2884,2647,1818,978,84,1339,2334,2740,3312,3333,2824,2548,1528,345,753,1340,1935,2099,1916,1346,747,82,519,962,1315,1170,777,23,648,1553,1976,2717,2913,2558,1723,1245,468,280,915,1330,1009,711,331,96,725,971,719,260,1378,2307,3159,4595,5501,5517,4913,4123,2901,1412,1037,2765,4158,5152,6070,6431,5821,4630,3563,2851,1162,716,1991,2929,2659,2931,2904,2703,1296,680,197,195,652,658,11,751,1296,1749,2065,2257,1576,708,770,1952,2932,4183,4669,4611,3996,3113,1355,633,2691,3978,4819,5217,5062,4260,2727,1411,327,1531,2595,3661,3992,3558,2711,2297,1460,326,628,1052,1374,1465,1250,590,190,704,1012,1312,1430,1004,318,631,1444,1881,2733,2632,3124,2352,1310,472,643,1492,2491,2751,2635,2073,1409,657,114,679,1139,1320,1024,591,385,1830,3442,4590,4639,3312,945,1711,4170,5974,7491,8588,9218,9164,8014,5532,2277,1363,4312,6354,7680,8396,8528,8065,6940,5098,2638,583,1330,2647,2920,3052,2928,1307,186,864,1544,1875,1435,31,1467,2934,4768,6563,7726,7713,6658,4952,2997,687,4206,7378,9531,10627,10704,9620,7412,4491,1430,2597,5893,8767,9909,9074,7023,4658,2694,108,2252,4135,5576,6561,6516,5166,2739,665,1120,2045,3060,2955,2896,2614,1018,735,2604,4799,6048,6124,5085,3344,1157,1457,4179,5998,7082,7517,7487,6717,4885,2563,395,2915,3968,4741,4737,4205,3045,2533,1352,18,989,1346,678,71,454,915,1346,1995,2641,2737,2100,1458,479,163,518,1033,1118,114,1941,4604,7085,8334,8264,7803,7713,7643,6708,4756,2697,159,2699,5353,6750,6755,6360,6539,6913,6520,5393,4314,3385,1742,722,2830,2703,2832,2928,3056,3442,2749,2581,2232,2486,2301,2340,3061,3326,3455,2710,2038,1038,431,2954,4910,6724,7711,8094,8066,7430,5840,3350,368,2961,4888,6382,6861,6363,5223,4004,2898,1018,648,2590,3612,4432,4785,
	};
	nrf_axon_result_e result;

	nrf_axon_platform_printf("exp_11p12_tests: %d cases\n", exp_11p12_test_cnt);
	/* case 0: no rounding */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_exp_11p12(logn_expected_output, test_results.as_i32, 512,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf("FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			exp_expected_output, 512, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}

	(*test_ndx)++;

	/* case 1: length=514 (too long) */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_exp_11p12(abs_main_vector_32, test_results.as_i32,
		514, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result >= 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! did not catch length too long %d\n", 514);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 2: length=2 (too short) */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_exp_11p12(abs_main_vector_32, test_results.as_i32, 1,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result >= 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! did not catch length too long %d\n", 1);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* case 3: length=511 (not a multiple of 2) */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_exp_11p12(abs_main_vector_32, test_results.as_i32, 511,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result >= 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! did not catch length not a multiple of 2 %d\n", 511);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * axpb_24_24 tests
 * Perform 1 test: main_vector[172], a=3, b=-5, rounding_bits=2
 */
#define axpb_24_24_test_cnt 1
int axpb_24_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	int32_t const_a = 3;
	int32_t const_b = -5;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("axpb_24_24_tests: %d cases\n", axpb_24_24_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_axpb_24_24(main_vector_32, &const_a, &const_b, test_results.as_i32,
		ARRAY_LENGTH(axpb_24_24_expected_output) & ~1, 2,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			axpb_24_24_expected_output, ARRAY_LENGTH(axpb_24_24_expected_output) & ~1,
			1, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}
/**
 * xs_24_24 tests
 * Perform 1 test: main_vector[172], a=3, b=-5, rounding_bits=2
 */
#define xs_24_24_test_cnt 1
int xs_24_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("xs_24_24_tests: %d cases\n", xs_24_24_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xs_24_24(main_vector_32 + 99, test_results.as_i32,
		ARRAY_LENGTH(xs_24_24_expected_output)& ~1, 2,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			xs_24_24_expected_output, ARRAY_LENGTH(xs_24_24_expected_output)&~1,
			0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}
/**
 * axpby_24_24_24 tests
 * Perform 1 test: main_vector[172], a=3, b=-5, rounding_bits=2
 */
#define axpby_24_24_24_test_cnt 1
int axpby_24_24_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	int32_t const_a = 20000;
	int32_t const_b = -15;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("axpby_24_24_24_tests: %d cases\n", axpby_24_24_24_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_axpby_24_24_24(main_vector_32, main_vector_32 + 599, &const_a, &const_b,
		test_results.as_i32, ARRAY_LENGTH(axpby_24_24_24_expected_output) & ~1, 3,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			axpby_24_24_24_expected_output,
			ARRAY_LENGTH(axpby_24_24_24_expected_output) & ~1, 1, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * xsmys_24_24_24 tests
 * Perform 1 test: main_vector[0] x main_vector[500]
 */
#define xsmys_24_24_24_test_cnt 1
int xsmys_24_24_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("xsmys_24_24_24_tests: %d cases\n", xsmys_24_24_24_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xsmys_24_24_24(main_vector_32, main_vector_32 + 500,test_results.as_i32,
		ARRAY_LENGTH(xsmys_24_24_24_expected_output) & ~1, 3,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			xsmys_24_24_24_expected_output,
			ARRAY_LENGTH(xsmys_24_24_24_expected_output) & ~1, 1, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * xty_24_24_24 tests
 * Perform 1 test: main_vector[0] x main_vector[500]
 */
#define xty_24_24_24_test_cnt 1
int xty_24_24_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("xty_24_24_24_tests: %d cases\n", xty_24_24_24_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xty_24_24_24(main_vector_32, main_vector_32 + 500,test_results.as_i32,
		ARRAY_LENGTH(xty_24_24_24_expected_output) & ~1, 5,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			xty_24_24_24_expected_output,
			ARRAY_LENGTH(xty_24_24_24_expected_output) & ~1, 1, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * xpy_24_24_24 tests
 * Perform 1 test: main_vector[0] x main_vector[500]
 */
#define xpy_24_24_24_test_cnt 1
int xpy_24_24_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("xpy_24_24_24_tests: %d cases\n", xpy_24_24_24_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xpy_24_24_24(main_vector_32, main_vector_32 + 500,test_results.as_i32,
		ARRAY_LENGTH(xpy_24_24_24_expected_output) & ~1, 1,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			xpy_24_24_24_expected_output,
			ARRAY_LENGTH(xpy_24_24_24_expected_output) & ~1, 1, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * xmy_24_24_24 tests
 * Perform 1 test: main_vector[0] x main_vector[500]
 */
#define xmy_24_24_24_test_cnt 1
int xmy_24_24_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("xmy_24_24_24_tests: %d cases\n", xmy_24_24_24_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_xmy_24_24_24(main_vector_32, main_vector_32 + 500,test_results.as_i32,
		ARRAY_LENGTH(xmy_24_24_24_expected_output) & ~1, 1,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			xmy_24_24_24_expected_output,
			ARRAY_LENGTH(xmy_24_24_24_expected_output) & ~1, 1, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * acc_16_24 tests
 * Perform 1 test: acc main_vector[500:699] x main_vector[500:551] round 3
 */
#define acc_16_24_test_cnt 1
int acc_16_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	static int32_t acc_16_24_expected_output[] = {4280,};
	nrf_axon_result_e result;

	nrf_axon_platform_printf("acc_16_24_tests: %d cases\n", acc_16_24_test_cnt);
	/* case 0: mar */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_acc_16_24(main_vector_16 + 500,test_results.as_i32, 200, 3,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("", test_results.as_i32,
			acc_16_24_expected_output, 1, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * fir_24_24_24 tests
 * Perform 1 test: main_vector[0] x main_vector[500]
 */
#define fir_24_24_24_test_cnt 1
int fir_24_24_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("fir_24_24_24_tests: %d cases\n", fir_24_24_24_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_fir_24_24_24(main_vector_32, fir_24_24_24_filter,test_results.as_i32,
		ARRAY_LENGTH(fir_24_24_24_filter) * 3, ARRAY_LENGTH(fir_24_24_24_filter),
		0, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
#if defined (CONFIG_SOC_NRF54LM20B) || defined (NOT_A_ZEPHYR_BUILD)
	} else if (nrf_axon_verify_vectors_output_stride("",
			test_results.as_i32 + ARRAY_LENGTH(fir_24_24_24_filter),
			fir_24_24_24_expected_output,
			ARRAY_LENGTH(fir_24_24_24_filter) * 2, 1, 0 != 0)) {
#else
	} else if (nrf_axon_verify_vectors_output_stride("",
		test_results.as_i32,
		fir_24_24_24_expected_output,
		ARRAY_LENGTH(fir_24_24_24_filter) * 2, 1, 0) != 0) {
#endif
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * fir_24_16_24 tests
 * Perform 1 test: main_vector[0] x main_vector[500]
 */
#define fir_24_16_24_test_cnt 1
int fir_24_16_24_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("fir_24_16_24_tests: %d cases\n", fir_24_16_24_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_fir_24_16_24(main_vector_32, fir_24_16_24_filter,test_results.as_i32,
		ARRAY_LENGTH(fir_24_16_24_filter)*3, ARRAY_LENGTH(fir_24_16_24_filter), 0,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
#if defined (CONFIG_SOC_NRF54LM20B) || defined (NOT_A_ZEPHYR_BUILD)
	} else if (nrf_axon_verify_vectors_output_stride("",
			test_results.as_i32 + ARRAY_LENGTH(fir_24_16_24_filter),
			fir_24_16_24_expected_output,
			ARRAY_LENGTH(fir_24_16_24_filter) * 2, 1, 0) != 0) {
#else	/* Newer versions of Axons automatically discard the initial fir outputs */
	} else if (nrf_axon_verify_vectors_output_stride("",
			test_results.as_i32, fir_24_16_24_expected_output,
			ARRAY_LENGTH(fir_24_16_24_filter) * 2, 1, 0) != 0) {
#endif
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * fir_2d_16_16_32_decimate tests
 */
#define fir_2d_16_16_32_decimate_test_cnt 4
int fir_2d_16_16_32_decimate_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("fir_2d_16_16_32_decimate_tests: %d cases\n",
		fir_2d_16_16_32_decimate_test_cnt);
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	/* input: 512x2, filter: 6x2, decimation_width 2, output: 507x1 */

#if AXON_SIMULATION
	nrf_axon_simulator_perfmodel_enable();
	nrf_axon_simulator_perfmodel_init();
#endif
	uint32_t profiling_ticks;
	uint32_t op_ticks;
	op_ticks = nrf_axon_platform_get_ticks();

	result = nrf_axon_fir_2d_16_16_32_decimate(main_vector_16, (const int16_t *)fir_2d_16_16_32_decimate_filter,
		test_results.as_i32, 512, 6, 2, 0, NRF_AXON_SYNC_MODE_BLOCKING_POLLING,
		false);
	profiling_ticks = nrf_axon_platform_get_ticks();
	op_ticks = profiling_ticks - op_ticks;
#if AXON_SIMULATION
	op_ticks = (uint32_t)nrf_axon_simulator_perfmodel_get_cycles();
	nrf_axon_simulator_perfmodel_disable();
#endif
	nrf_axon_platform_printf(
		"2d fir: 1024 input, 12 filter, decimation 2, profiling ticks %u\n", op_ticks);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("",
			fir_2d_16_16_32_decimate_2_6_expected_output,
			test_results.as_i32, 507, 0, 1) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);

#if AXON_SIMULATION
	nrf_axon_simulator_perfmodel_enable();
	nrf_axon_simulator_perfmodel_init();
#endif
	profiling_ticks = nrf_axon_platform_get_ticks();
	result = nrf_axon_fir_16_16_32_1024_256_decimate_4(main_vector_16,
		fir_2d_16_16_24_saturate_filter, test_results.as_i32, 0,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	profiling_ticks = nrf_axon_platform_get_ticks() - profiling_ticks;
#if AXON_SIMULATION
	profiling_ticks = (uint32_t)nrf_axon_simulator_perfmodel_get_cycles();
	nrf_axon_simulator_perfmodel_disable();
#endif
	nrf_axon_platform_printf(
		"2d fir: 1024input, 256 filter, decimation 4,	profiling ticks %u\n", profiling_ticks);

	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_expected_stride("",
			test_results.as_i32,
			fir_2d_16_16_24_saturate_expected_output,
			512 / 4, 0, 3, 32) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/* reserve in advance to skip this in profiling. */
	profiling_ticks = nrf_axon_platform_get_ticks();

	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	/* 1D FIR */
	result = nrf_axon_fir_16_16_32_1024_256_decimate_1(main_vector_16,
		(const int16_t *)fir_2d_16_16_32_decimate_filter, test_results.as_i32, 0,
		NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	profiling_ticks = nrf_axon_platform_get_ticks() - profiling_ticks;
#if AXON_SIMULATION
	profiling_ticks = (uint32_t)nrf_axon_simulator_perfmodel_get_cycles();
	nrf_axon_simulator_perfmodel_disable();
#endif
	nrf_axon_platform_printf(
		"1d fir: 1024input, 256 filter, decimation 1,	profiling ticks %u\n", profiling_ticks);

	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_output_stride("",
			fir_2d_16_16_32_decimate_4_64_expected_output,
			test_results.as_i32, 769, 0, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/** begin 2d_16_16_24 input 768, filter 256, decimate 16 */	
	extern void axon_nn_enable_passlist_candidate_mode();
	axon_nn_enable_passlist_candidate_mode();

	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
#if AXON_SIMULATION
	nrf_axon_simulator_perfmodel_enable();
	nrf_axon_simulator_perfmodel_init();
#endif
	op_ticks = nrf_axon_platform_get_ticks();
	result = nrf_axon_fir_2d_16_16_24_decimate(main_vector_16, fir_2d_16_16_24_saturate_filter,
		test_results.as_i32, 48, 16, 16, 0, NRF_AXON_SYNC_MODE_BLOCKING_POLLING,
		false);
	profiling_ticks = nrf_axon_platform_get_ticks();
	op_ticks = profiling_ticks - op_ticks;
#if AXON_SIMULATION
	op_ticks = (uint32_t)nrf_axon_simulator_perfmodel_get_cycles();
	nrf_axon_simulator_perfmodel_disable();
#endif
	nrf_axon_platform_printf("2d fir 16_16_24: 768 input, 256 filter, decimation 16,profiling ticks %u\n,",
		op_ticks);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_expected_stride("",
			test_results.as_i32,
			fir_2d_16_16_24_saturate_expected_output,
			512 / 16, 0, 15, 24) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;
	/** end 2d_16_16_24 input 768, filter 256, decimate 16 */	
	return pass_cnt;
}
#define fir_cplx_2d_16_16_24_decimate_test_cnt 1
int fir_cplx_2d_16_16_24_decimate_tests(int *test_ndx)
{
	/************* complex filter ************************/
	unsigned op_ticks;
	unsigned profiling_ticks;
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
#if AXON_SIMULATION
	nrf_axon_simulator_perfmodel_enable();
	nrf_axon_simulator_perfmodel_init();
#endif
	op_ticks = nrf_axon_platform_get_ticks();
	result = nrf_axon_fir_cplx_2d_16_16_24_decimate(main_vector_16, (const int16_t *)fir_2d_16_16_32_decimate_filter,
		test_results.as_i32, 48, 16, 16, 0, NRF_AXON_SYNC_MODE_BLOCKING_POLLING,
		false);
	profiling_ticks = nrf_axon_platform_get_ticks();
	op_ticks = profiling_ticks - op_ticks;
#if AXON_SIMULATION
	op_ticks = (uint32_t)nrf_axon_simulator_perfmodel_get_cycles();
	nrf_axon_simulator_perfmodel_disable();
#endif
	nrf_axon_platform_printf("2d complex fir 16_16_24: 768 input, 256 filter, decimation 16,profiling ticks %u\n,",
		op_ticks);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_expected_stride("",
			test_results.as_i32,
			fir_2d_16_16_24_expected_output_real,
			512 / 16 + 1, 0, 15, 24) != 0) {
		nrf_axon_platform_printf( "real output failed\n"
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_expected_stride("",
			test_results.as_i32 + (512 / 16) + 1,
			fir_2d_16_16_24_expected_output_image,
			(512 / 16) + 1, 0, 15, 24) != 0) {
		nrf_axon_platform_printf( "imaginary output failed\n"
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}

	(*test_ndx)++;
	/** end complex filter */
	return pass_cnt;
}

/**
 * dma_2d tests
 */
#define dma_2d_test_cnt 3
int dma_2d_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;

	nrf_axon_platform_printf("dma_2d_tests: %d cases\n", dma_2d_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_dma_2d((int8_t *)main_vector_32, (int8_t *)test_results.as_i32,
		32, 48, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_8("",
			(int8_t *)test_results.as_i32, (int8_t *)main_vector_32,
			16* 32 * 2, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_dma_2d((int8_t *)main_vector_32, (int8_t *)test_results.as_i32,
		32, 47, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result >= 0) {
		nrf_axon_platform_printf(
			"FAILED: error check did not catch width 47 is not a multiple of 4. code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"Failed as expected, width 47 is not a multiple of 4. code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	/**
	 * Test a ping-pong scenario
	 * Copy the 2nd half 1st..
	 */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
 	memset(test_results.as_i16, 0, 48 * 16 * 2);
	result = nrf_axon_dma_2d((int8_t *)(main_vector_16 + (4 * 48)),
		(int8_t *)(test_results.as_i16 + 4 * 48),
		sizeof(int16_t)*(16 - 4), 48, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, true);
	if (result == NRF_AXON_RESULT_SUCCESS) {
		result = nrf_axon_dma_2d((int8_t *)(main_vector_16),
			(int8_t *)test_results.as_i16,
			sizeof(int16_t) * (16 - 4), 48, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	}
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_16("",
			test_results.as_i16, main_vector_16,
			16 * 32, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * axpb_2d_8_16 tests
 */
#define axpb_2d_8_16_test_cnt 1
int axpb_2d_8_16_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;
	const int32_t a_scalar = 40000;
	const int32_t b_scalar = -12332;
	const int8_t rounding_bits = 7;

	nrf_axon_platform_printf("axpb_2d_8_16_tests: %d cases\n", axpb_2d_8_16_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_axpb_2d_8_16(main2_vector8, a_scalar, b_scalar, test_results.as_i16,
		64, 16, true, rounding_bits, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_16("",
			test_results.as_i16, axpb_2d_8_16_expected_output,
			16 * 64, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}
/**
 * axpb_2d_16_16 tests
 */
#define axpb_2d_16_16_test_cnt 1
int axpb_2d_16_16_tests(int *test_ndx)
{
	int pass_cnt = 0;
	nrf_axon_result_e result;
	const int32_t a_scalar = 40000;
	const int32_t b_scalar = -12332;
	const int8_t rounding_bits = 7;

	nrf_axon_platform_printf("axpb_2d_16_16_tests: %d cases\n", axpb_2d_16_16_test_cnt);
	/* case 0: single command, 1 row */
	nrf_axon_platform_printf("\nTEST:\t%s\tSTART CASE NO\t%d\n", TEST_NAME, *test_ndx);
	result = nrf_axon_axpb_2d_16_16(main2_vector16, a_scalar, b_scalar, test_results.as_i16,
		64, 16, true, rounding_bits, NRF_AXON_SYNC_MODE_BLOCKING_POLLING, false);
	if (result < 0) {
		nrf_axon_platform_printf(
			"FUNCTION FAILED! code=%d\n", result);
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else if (nrf_axon_verify_vectors_16("",
			test_results.as_i16, axpb_2d_16_16_expected_output,
			16* 64, 0) != 0) {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tFAIL\n", TEST_NAME, *test_ndx);
	} else {
		nrf_axon_platform_printf(
			"\nTEST:\t%s\tCASE NO\t%d\tRESULT:\tPASS\n", TEST_NAME, *test_ndx);
		pass_cnt++;
	}
	(*test_ndx)++;

	return pass_cnt;
}

/**
 * listing of test cases
 */
struct {
	int case_count;
	int (*test_func)(int *text_ndx);
} test_cases[] = {
	{ .case_count = xty_16_16_32_output_stride_test_cnt,
		.test_func = xty_16_16_32_output_stride_tests,},
	{ .case_count = memset_32_output_stride_test_cnt,
		.test_func = memset_32_output_stride_tests, },
	{ .case_count = saturate_32_24_test_cnt, .test_func = saturate_32_24_tests, },
	{ .case_count = xty_16_16_32_test_cnt, .test_func = xty_16_16_32_tests, },
	{ .case_count = fft_24_test_cnt, .test_func = fft_24_tests, },
	{ .case_count = xspys_24_24_24_input_stride2_test_cnt,
		.test_func = xspys_24_24_24_input_stride2_tests, },
	{ .case_count = xspys_24_24_24_test_cnt, .test_func = xspys_24_24_24_tests,	},
	{ .case_count = xsmys_24_24_24_test_cnt, .test_func = xsmys_24_24_24_tests,	},
	{ .case_count = mar_16_24_32_test_cnt, .test_func = mar_16_24_32_tests, },
	{ .case_count = mar_16_24_24_test_cnt, .test_func = mar_16_24_24_tests, },
	{ .case_count = sqrt_24_test_cnt, .test_func = sqrt_24_tests, },
	{ .case_count = logn_11p12_test_cnt, .test_func = logn_11p12_tests, },
	{ .case_count = exp_11p12_test_cnt, .test_func = exp_11p12_tests, },
	{ .case_count = xs_24_24_test_cnt, .test_func = xs_24_24_tests, },
	{ .case_count = axpby_24_24_24_test_cnt, .test_func = axpby_24_24_24_tests, },
	{ .case_count = xty_24_24_24_test_cnt, .test_func = xty_24_24_24_tests, },
	{ .case_count = xpy_24_24_24_test_cnt, .test_func = xpy_24_24_24_tests, },
	{ .case_count = xmy_24_24_24_test_cnt, .test_func = xmy_24_24_24_tests, },
	{ .case_count = acc_16_24_test_cnt, .test_func = acc_16_24_tests, },
	{ .case_count = fir_24_24_24_test_cnt, .test_func = fir_24_24_24_tests, },
	{ .case_count = fir_24_16_24_test_cnt, .test_func = fir_24_16_24_tests, },
	{ .case_count = axpb_24_24_test_cnt, .test_func = axpb_24_24_tests, },
	{ .case_count = fft_power_24_test_cnt, .test_func = fft_power_24_tests, },
	{ .case_count = fir_2d_16_16_32_decimate_test_cnt,
		.test_func = fir_2d_16_16_32_decimate_tests,},
	{ .case_count = fir_cplx_2d_16_16_24_decimate_test_cnt,
		.test_func  = fir_cplx_2d_16_16_24_decimate_tests,},
	{ .case_count = dma_2d_test_cnt,
		.test_func = dma_2d_tests,},
	{ .case_count = abs_24_24_24_test_cnt,
		.test_func = abs_24_24_24_tests,	},
	{ .case_count = axpb_2d_8_16_test_cnt,
		.test_func = axpb_2d_8_16_tests,},
	{ .case_count = axpb_2d_16_16_test_cnt,
		.test_func = axpb_2d_16_16_tests,},
};
int main_intrinsics_test()
{
	nrf_axon_platform_printf("\n\nStart axon_app_intrinsics!\n\n");
	int test_case_count = 0;
	int test_pass_count = 0;

	nrf_axon_result_e result = nrf_axon_platform_init();

	/* sum up the total number of test cases. */
	for (int ndx = 0; ndx < sizeof(test_cases) / sizeof(*test_cases); ndx++) {
		test_case_count += test_cases[ndx].case_count;
	}

	nrf_axon_platform_printf("\n\nTEST:\t%s\tCASE COUNT\t%d\n",
		TEST_NAME, test_case_count);

	if (result != NRF_AXON_RESULT_SUCCESS) {
		nrf_axon_platform_printf("\n\axon_platform_init failed!\n");
		return result;
	}

	/* run the test case functions. Each one can have multiple tests. */
	int test_ndx = 0;
	for (int ndx = 0; ndx < sizeof(test_cases) / sizeof(*test_cases); ndx++) {
		test_pass_count += test_cases[ndx].test_func(&test_ndx);
	}

	nrf_axon_platform_printf("\n\nTEST:\t%s\tCOMPLETE\tPASS COUNT\t%d\tFAIL COUNT\t%d\n",
		TEST_NAME, test_pass_count, test_case_count - test_pass_count);

	nrf_axon_platform_printf("\r\n axon_app_intrinsics complete!\r\n");
	nrf_axon_platform_close();


	return 0;
}

#if (NOT_A_ZEPHYR_BUILD)
int main() {
	int result = main_intrinsics_test();
	return result;
}
#endif
