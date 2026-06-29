/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stdbool.h>
#include <stdint.h>

#include <zephyr/kernel.h>
#include <zcbor_encode.h>

#include "nrf_edgeai_obsv_encode.h"
#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_core.h>

/* CBOR schema is defined authoritatively in lib/nrf_edgeai_obsv/obsv.cddl. */

/* CBOR major type 4 (array), bits [7:5] = 0b100. */
#define CBOR_MAJOR_TYPE_ARRAY 0x80U

/* v2: added "num_features" to the model map, and the "num_features" counter
 * (FEATURES-stream) to the payload. Additive fields within the 2.x line keep
 * format_version at 2; the encoder and the CDDL-generated decoder are
 * regenerated together so in-tree consumers stay in sync.
 */
#define OBSV_FORMAT_VERSION	 2
#define OBSV_TOP_MAP_ELEMENTS	 5
#define OBSV_MODEL_MAP_ELEMENTS	 4
#define OBSV_METRIC_MAP_ELEMENTS 3

/* Five nesting levels: top map → metrics list → metric map → data list → row list. */
#define OBSV_ENCODE_BACKUPS 5

struct encode_ctx {
	zcbor_state_t *zs;
	bool ok;
};

static bool encode_metric(const nrf_edgeai_obsv_metric_snapshot_t *snap, void *user)
{
	struct encode_ctx *ec = user;
	zcbor_state_t *zs = ec->zs;
	bool ok = ec->ok;

	ok = ok && zcbor_map_start_encode(zs, OBSV_METRIC_MAP_ELEMENTS);
	ok = ok && zcbor_tstr_put_lit(zs, "id");
	ok = ok && zcbor_uint32_put(zs, snap->metric_id);
	ok = ok && zcbor_tstr_put_lit(zs, "v");
	ok = ok && zcbor_uint32_put(zs, snap->version);
	ok = ok && zcbor_tstr_put_lit(zs, "d");

	ok = ok && zcbor_list_start_encode(zs, snap->num_rows);
	for (uint16_t r = 0; ok && r < snap->num_rows; r++) {
		ok = zcbor_list_start_encode(zs, snap->num_cols);
		for (uint16_t c = 0; ok && c < snap->num_cols; c++) {
			ok = zcbor_uint32_put(zs, snap->counts[r * snap->num_cols + c]);
		}
		ok = ok && zcbor_list_end_encode(zs, snap->num_cols);
	}
	ok = ok && zcbor_list_end_encode(zs, snap->num_rows);
	ok = ok && zcbor_map_end_encode(zs, OBSV_METRIC_MAP_ELEMENTS);

	ec->ok = ok;
	return ok;
}

size_t nrf_edgeai_obsv_encode_cbor(nrf_edgeai_obsv_core_t *state, uint8_t *buf, size_t max_len)
{
	if ((state == NULL) || (buf == NULL) || (max_len == 0U)) {
		return 0;
	}

	ZCBOR_STATE_E(zs, OBSV_ENCODE_BACKUPS, buf, max_len, 0);

	bool ok = zcbor_map_start_encode(zs, OBSV_TOP_MAP_ELEMENTS);

	ok = ok && zcbor_tstr_put_lit(zs, "format_version");
	ok = ok && zcbor_uint32_put(zs, OBSV_FORMAT_VERSION);

	ok = ok && zcbor_tstr_put_lit(zs, "num_inferences");
	ok = ok && zcbor_uint32_put(zs, state->num_inferences);

	ok = ok && zcbor_tstr_put_lit(zs, "num_features");
	ok = ok && zcbor_uint32_put(zs, state->num_features);

	ok = ok && zcbor_tstr_put_lit(zs, "model");
	ok = ok && zcbor_map_start_encode(zs, OBSV_MODEL_MAP_ELEMENTS);
	ok = ok && zcbor_tstr_put_lit(zs, "id");
	ok = ok && zcbor_uint32_put(zs, state->model.model_id);
	ok = ok && zcbor_tstr_put_lit(zs, "num_classes");
	ok = ok && zcbor_uint32_put(zs, state->model.num_classes);
	ok = ok && zcbor_tstr_put_lit(zs, "num_features");
	ok = ok && zcbor_uint32_put(zs, state->model.num_features);
	ok = ok && zcbor_tstr_put_lit(zs, "version");
	ok = ok && zcbor_uint32_put(zs, state->model.version);
	ok = ok && zcbor_map_end_encode(zs, OBSV_MODEL_MAP_ELEMENTS);

	ok = ok && zcbor_tstr_put_lit(zs, "metrics");
	ok = ok && zcbor_list_start_encode(zs, state->num_metrics);

	if (ok) {
		struct encode_ctx ec = {.zs = zs, .ok = true};

		(void)nrf_edgeai_obsv_core_for_each_metric(state, encode_metric, &ec);
		ok = ec.ok;
	}

	ok = ok && zcbor_list_end_encode(zs, state->num_metrics);
	ok = ok && zcbor_map_end_encode(zs, OBSV_TOP_MAP_ELEMENTS);

	if (!ok) {
		return 0;
	}

	return (size_t)(zs->payload - buf);
}

size_t nrf_edgeai_obsv_encode(nrf_edgeai_obsv_ctx_t *ctx, uint8_t *buf, size_t max_len)
{
	if ((ctx == NULL) || (buf == NULL) || (max_len == 0U)) {
		return 0U;
	}

	k_mutex_lock(&ctx->lock, K_FOREVER);

	size_t len = nrf_edgeai_obsv_encode_cbor(&ctx->state, buf, max_len);

	k_mutex_unlock(&ctx->lock);

	return len;
}

size_t nrf_edgeai_obsv_encode_list(nrf_edgeai_obsv_ctx_t *const *ctxs, uint8_t n,
				   uint8_t *buf, size_t max_len)
{
	if ((ctxs == NULL) || (buf == NULL) || (max_len == 0U) || (n == 0U)) {
		return 0U;
	}

	/* One-byte CBOR array header is only valid for n <= 23 (CBOR tiny integer). */
	if ((n > 23U) || (max_len < 1U)) {
		return 0U;
	}

	uint8_t *p = buf;
	size_t remaining = max_len;

	*p++ = (uint8_t)(CBOR_MAJOR_TYPE_ARRAY | n);
	remaining--;

	for (uint8_t i = 0; i < n; i++) {
		size_t len = nrf_edgeai_obsv_encode(ctxs[i], p, remaining);

		if ((len == 0U) || (len > remaining)) {
			return 0U;
		}

		p += len;
		remaining -= len;
	}

	return (size_t)(p - buf);
}
