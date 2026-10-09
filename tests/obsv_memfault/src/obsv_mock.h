/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef TEST_OBSV_MEMFAULT_OBSV_MOCK_H
#define TEST_OBSV_MEMFAULT_OBSV_MOCK_H

/** Bytes the mock writes per registered context. */
#define OBSV_MOCK_BYTES_PER_CTX 16U

#include <stdint.h>

/** Reset the call counter. Call from before_each. */
void obsv_mock_reset(void);

/** Number of encode calls since the last obsv_mock_reset(). */
uint8_t obsv_mock_call_count(void);

/** Make the next encode call fail (return 0) without writing to the buffer. */
void obsv_mock_fail_next(void);

#endif /* TEST_OBSV_MEMFAULT_OBSV_MOCK_H */
