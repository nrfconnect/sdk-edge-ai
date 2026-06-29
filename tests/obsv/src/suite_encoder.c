/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stdint.h>
#include <stddef.h>

#include <zephyr/ztest.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#include "cddl_decode.h"
#include "common.h"

ZTEST_SUITE(obsv_encoder, NULL, NULL, NULL, NULL, NULL);

static const nrf_edgeai_obsv_model_info_t test_model = {
	.model_id = TEST_MODEL_ID,
	.num_classes = TEST_NUM_CLASSES,
	.version = TEST_MODEL_VERSION,
};

/*
 * Verify that nrf_edgeai_obsv_encode_multi_cbor() emits bytes that conform to
 * the obsv-cdr CDDL type defined in lib/nrf_edgeai_obsv/obsv.cddl.
 * The decoder is generated from that schema at configure time (CMakeLists.txt),
 * so any structural drift between the encoder and the schema fails here —
 * without involving any transport layer (Memfault, UART, etc.).
 */
ZTEST(obsv_encoder, test_multi_encoder_conforms_to_cddl_schema)
{
	static nrf_edgeai_obsv_ctx_t ctx;

	static uint32_t pd_buf[NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(TEST_NUM_CLASSES) /
				sizeof(uint32_t)];
	static uint32_t tm_buf[NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(TEST_NUM_CLASSES) /
				sizeof(uint32_t)];
	nrf_edgeai_obsv_metric_t pd;
	nrf_edgeai_obsv_metric_t tm;

	nrf_edgeai_obsv_metric_pd_create(&pd, pd_buf, TEST_NUM_CLASSES);
	nrf_edgeai_obsv_metric_tm_create(&tm, tm_buf, TEST_NUM_CLASSES);

	zassert_equal(nrf_edgeai_obsv_init(&ctx, &test_model), 0);
	zassert_equal(nrf_edgeai_obsv_register(&ctx, &pd, NULL), 0);
	zassert_equal(nrf_edgeai_obsv_register(&ctx, &tm, NULL), 0);

	const float probs[TEST_NUM_CLASSES] = {0.7f, 0.1f, 0.1f, 0.1f};

	zassert_equal(nrf_edgeai_obsv_update_probs(&ctx, probs), 0);

	nrf_edgeai_obsv_ctx_t *ctxs[] = {&ctx};
	uint8_t buf[512];
	size_t len = nrf_edgeai_obsv_encode_list(ctxs, ARRAY_SIZE(ctxs), buf, sizeof(buf));

	zassert_true(len > 0, "encoder returned 0 — buffer too small or internal error");

	/* The generated struct may be large; keep it off the ztest stack. */
	static struct obsv_list cddl_decoded;
	size_t consumed = 0;

	int rc = cbor_decode_obsv_list(buf, len, &cddl_decoded, &consumed);

	zassert_equal(rc, 0, "output does not conform to obsv-list schema (rc=%d)", rc);
	zassert_equal(consumed, len,
		      "trailing bytes after schema-valid payload (consumed=%zu, total=%zu)",
		      consumed, len);

	zassert_equal(cddl_decoded.obsv_list_obsv_payload_m_count, 1U);

	const struct obsv_payload *p = &cddl_decoded.obsv_list_obsv_payload_m[0];

	zassert_equal(p->obsv_payload_format_version, 2);
	zassert_equal(p->obsv_payload_num_inferences, 1);
	zassert_equal(p->obsv_payload_num_features, 0,
		      "no feature updates were fed, so the FEATURES counter must be 0");
	zassert_equal(p->obsv_payload_metrics_obsv_metric_m_count, 2);
}
