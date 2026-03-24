/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MEL_TEST_VECTOR_H
#define MEL_TEST_VECTOR_H

#include <stdint.h>

/** Mel bins per frame (matches okay_nordic external input channel count). */
#define MEL_TEST_VECTOR_MEL_BINS 40

/** Frames captured from ww_kws. */
#define MEL_TEST_VECTOR_NUM_FRAMES 102

/** Raw int32 rows are IEEE-754 float32 bit patterns from DSP */
extern const int32_t mel_test_vector[MEL_TEST_VECTOR_NUM_FRAMES][MEL_TEST_VECTOR_MEL_BINS];

#endif /* MEL_TEST_VECTOR_H */
