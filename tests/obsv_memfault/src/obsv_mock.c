/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <string.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>

#include "obsv_mock.h"

/*
 * Stateful mock for nrf_edgeai_obsv_encode_list_and_reset().
 *
 * Returns n bytes (one per context), each set to the current call count.
 * The incrementing fill lets tests verify:
 *   - data_size_bytes == n  (correct context count passed through)
 *   - successive collects produce different bytes
 * The call count lets tests verify that a refused collect does not encode.
 * obsv_mock_fail_next() makes one call fail, as an encode error would.
 *
 * Encoding and reset correctness are validated separately in
 * tests/obsv/suite_encoder.c.
 */

static uint8_t s_call_count;
static bool s_fail_next;

void obsv_mock_reset(void)
{
	s_call_count = 0;
	s_fail_next = false;
}

uint8_t obsv_mock_call_count(void)
{
	return s_call_count;
}

void obsv_mock_fail_next(void)
{
	s_fail_next = true;
}

size_t nrf_edgeai_obsv_encode_list_and_reset(nrf_edgeai_obsv_ctx_t *const *ctxs, uint8_t n,
					     uint8_t *buf, size_t buflen)
{
	ARG_UNUSED(ctxs);

	size_t len = (size_t)n * OBSV_MOCK_BYTES_PER_CTX;

	if (s_fail_next) {
		s_fail_next = false;
		return 0;
	}

	if (buflen < len) {
		return 0;
	}

	memset(buf, ++s_call_count, len);
	return len;
}
