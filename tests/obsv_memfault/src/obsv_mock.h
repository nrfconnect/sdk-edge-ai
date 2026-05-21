/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef TEST_OBSV_MEMFAULT_OBSV_MOCK_H
#define TEST_OBSV_MEMFAULT_OBSV_MOCK_H

/** Bytes the mock writes per registered context. */
#define OBSV_MOCK_BYTES_PER_CTX 16U

/** Reset the call counter. Call from before_each. */
void obsv_mock_reset(void);

#endif /* TEST_OBSV_MEMFAULT_OBSV_MOCK_H */
