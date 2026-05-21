/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <string.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>

#include "obsv_mock.h"

/*
 * Stateful mock for nrf_edgeai_obsv_encode_list().
 *
 * Returns n bytes (one per context), each set to the current call count.
 * The incrementing fill lets tests verify:
 *   - data_size_bytes == n  (correct context count passed through)
 *   - successive collects produce different bytes  (overwrite test)
 *
 * Encoding correctness is validated separately in tests/obsv/suite_encoder.c.
 */

static uint8_t s_call_count;

void obsv_mock_reset(void)
{
	s_call_count = 0;
}

size_t nrf_edgeai_obsv_encode_list(nrf_edgeai_obsv_ctx_t *const *ctxs, uint8_t n,
				   uint8_t *buf, size_t buflen)
{
	ARG_UNUSED(ctxs);

	size_t len = (size_t)n * OBSV_MOCK_BYTES_PER_CTX;

	if (buflen < len) {
		return 0;
	}

	memset(buf, ++s_call_count, len);
	return len;
}
