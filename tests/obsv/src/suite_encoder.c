/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stdint.h>
#include <stddef.h>

#include <zephyr/ztest.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_encode_sizes.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#include "cddl_decode.h"
#include "common.h"

#define TEST_NUM_FEATURES 8

/* Every list the encoder emits must fit the generated decoder. */
#define FITS_DECODER(n) ((n) <= OBSV_CDDL_MAX_QTY)

BUILD_ASSERT(FITS_DECODER(NRF_EDGEAI_OBSV_CPD_ROW_GROUPS * CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES),
	     "class predictions distribution rows exceed OBSV_CDDL_MAX_QTY");
BUILD_ASSERT(FITS_DECODER(NRF_EDGEAI_OBSV_MSD_NUM_ROWS),
	     "mel spectral descriptor rows exceed OBSV_CDDL_MAX_QTY");
BUILD_ASSERT(FITS_DECODER(CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_BIN_NUM) &&
	     FITS_DECODER(CONFIG_NRF_EDGEAI_OBSV_MODEL_CERTAINTY_DESC_BIN_NUM) &&
	     FITS_DECODER(CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_BIN_NUM) &&
	     FITS_DECODER(CONFIG_NRF_EDGEAI_OBSV_MEL_SPECTRAL_DESC_BIN_NUM),
	     "a metric bin count exceeds OBSV_CDDL_MAX_QTY");

/* Negative lower scaling bound, carried in "c" as a CBOR negative int. */
#define TEST_MED_P01_MILLI (-1500)

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

	static uint32_t pd_buf[NRF_EDGEAI_OBSV_CPD_STORAGE_BYTES(TEST_NUM_CLASSES) /
				sizeof(uint32_t)];
	static uint32_t tm_buf[NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(TEST_NUM_CLASSES) /
				sizeof(uint32_t)];
	nrf_edgeai_obsv_metric_t pd;
	nrf_edgeai_obsv_metric_t tm;

	nrf_edgeai_obsv_metric_cpd_create(&pd, pd_buf, TEST_NUM_CLASSES);
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

	zassert_equal(p->obsv_payload_format_version, 3);
	zassert_equal(p->obsv_payload_num_inferences, 1);
	zassert_equal(p->obsv_payload_num_features, 0,
		      "no feature updates were fed, so the FEATURES counter must be 0");
	zassert_equal(p->obsv_payload_metrics_obsv_metric_m_count, 2);

	/* Registration order: class predictions distribution first, then transition
	 * matrix. Only the former reports config ("c"); the latter omits the key.
	 */
	const struct obsv_metric *cpd_m = &p->obsv_payload_metrics_obsv_metric_m[0];
	const struct obsv_metric *tm_m = &p->obsv_payload_metrics_obsv_metric_m[1];

	zassert_equal(cpd_m->obsv_metric_id, NRF_EDGEAI_OBSV_METRIC_ID_CLASS_PRED_DIST);
	zassert_true(cpd_m->obsv_metric_c_present);
	zassert_equal(cpd_m->obsv_metric_c.c_int_l_count, 1, "config is a single row");

	const struct c_int_l *cfg_row = &cpd_m->obsv_metric_c.c_int_l[0];

	zassert_equal(cfg_row->c_int_l_int_count, NRF_EDGEAI_OBSV_CPD_CFG_COUNT);
	zassert_equal(cfg_row->c_int_l_int[NRF_EDGEAI_OBSV_CPD_CFG_STREAK_TOP],
		      CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_STREAK_TOP_BIN);
	zassert_equal(cfg_row->c_int_l_int[NRF_EDGEAI_OBSV_CPD_CFG_STREAK_TOL],
		      CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_STREAK_TOL);

	zassert_equal(tm_m->obsv_metric_id, NRF_EDGEAI_OBSV_METRIC_ID_TRANSITION_MATRIX);
	zassert_false(tm_m->obsv_metric_c_present);
}

/*
 * nrf_edgeai_obsv_encode_list_and_reset() emits the same bytes as
 * nrf_edgeai_obsv_encode_list() and resets every context; when the encode fails
 * (buffer too small) no context is reset, so no interval is lost.
 */
ZTEST(obsv_encoder, test_encode_list_and_reset)
{
	static nrf_edgeai_obsv_ctx_t ctx_a;
	static nrf_edgeai_obsv_ctx_t ctx_b;
	const float probs[TEST_NUM_CLASSES] = {0.7f, 0.1f, 0.1f, 0.1f};

	zassert_equal(nrf_edgeai_obsv_init(&ctx_a, &test_model), 0);
	zassert_equal(nrf_edgeai_obsv_init(&ctx_b, &test_model), 0);
	zassert_equal(nrf_edgeai_obsv_update_probs(&ctx_a, probs), 0);
	zassert_equal(nrf_edgeai_obsv_update_probs(&ctx_b, probs), 0);
	zassert_equal(nrf_edgeai_obsv_update_probs(&ctx_b, probs), 0);

	nrf_edgeai_obsv_ctx_t *ctxs[] = {&ctx_a, &ctx_b};
	uint8_t ref[256];
	uint8_t buf[256];

	/* Room for ctx A only: fails on ctx B, and ctx A (already encoded) is not reset either. */
	size_t a_only_len = nrf_edgeai_obsv_encode_list(ctxs, 1U, ref, sizeof(ref));

	zassert_true(a_only_len > 0U);
	zassert_equal(nrf_edgeai_obsv_encode_list_and_reset(ctxs, ARRAY_SIZE(ctxs), buf,
							    a_only_len),
		      0U);
	zassert_equal(ctx_a.state.num_inferences, 1U, "failed encode reset ctx A");
	zassert_equal(ctx_b.state.num_inferences, 2U, "failed encode reset ctx B");

	size_t ref_len = nrf_edgeai_obsv_encode_list(ctxs, ARRAY_SIZE(ctxs), ref, sizeof(ref));
	size_t len = nrf_edgeai_obsv_encode_list_and_reset(ctxs, ARRAY_SIZE(ctxs), buf,
							   sizeof(buf));

	zassert_true(len > 0U);
	zassert_equal(len, ref_len);
	zassert_mem_equal(buf, ref, len, "encode_list_and_reset bytes differ from encode_list");
	zassert_equal(ctx_a.state.num_inferences, 0U, "ctx A not reset");
	zassert_equal(ctx_b.state.num_inferences, 0U, "ctx B not reset");
}

/* Config reported by a custom metric: negative values must reach the wire as CBOR negative ints. */
#define NEG_CFG_METRIC_ID 100U

static const int32_t neg_cfg[1][2] = {{-1500, INT32_MAX}};
static const uint32_t neg_cfg_counts[1] = {0U};

static void neg_cfg_noop(const void *cfg, void *priv)
{
	ARG_UNUSED(cfg);
	ARG_UNUSED(priv);
}

static void neg_cfg_update(const float *p_probs, uint16_t n, void *priv)
{
	ARG_UNUSED(p_probs);
	ARG_UNUSED(n);
	ARG_UNUSED(priv);
}

static void neg_cfg_clear(void *priv)
{
	ARG_UNUSED(priv);
}

static void neg_cfg_snapshot(nrf_edgeai_obsv_metric_snapshot_t *out, void *priv)
{
	ARG_UNUSED(priv);

	out->metric_id = NEG_CFG_METRIC_ID;
	out->version = 1U;
	out->num_rows = 1U;
	out->num_cols = 1U;
	out->counts = neg_cfg_counts;
	out->config_rows = 1U;
	out->config_cols = 2U;
	out->config = &neg_cfg[0][0];
}

ZTEST(obsv_encoder, test_negative_config_roundtrips_as_signed)
{
	static nrf_edgeai_obsv_ctx_t ctx;
	nrf_edgeai_obsv_metric_t m = {
		.init = neg_cfg_noop,
		.update = neg_cfg_update,
		.clear = neg_cfg_clear,
		.snapshot = neg_cfg_snapshot,
	};

	zassert_equal(nrf_edgeai_obsv_init(&ctx, &test_model), 0);
	zassert_equal(nrf_edgeai_obsv_register(&ctx, &m, NULL), 0);

	nrf_edgeai_obsv_ctx_t *ctxs[] = {&ctx};
	uint8_t buf[256];
	size_t len = nrf_edgeai_obsv_encode_list(ctxs, ARRAY_SIZE(ctxs), buf, sizeof(buf));

	zassert_true(len > 0, "encoder returned 0");

	static struct obsv_list decoded;
	size_t consumed = 0;

	zassert_equal(cbor_decode_obsv_list(buf, len, &decoded, &consumed), 0,
		      "output does not conform to obsv-list schema");
	zassert_equal(consumed, len);
	zassert_equal(decoded.obsv_list_obsv_payload_m_count, 1U);

	const struct obsv_payload *p = &decoded.obsv_list_obsv_payload_m[0];

	zassert_equal(p->obsv_payload_metrics_obsv_metric_m_count, 1U);

	const struct obsv_metric *mm = &p->obsv_payload_metrics_obsv_metric_m[0];

	zassert_true(mm->obsv_metric_c_present);
	zassert_equal(mm->obsv_metric_c.c_int_l_count, 1);
	zassert_equal(mm->obsv_metric_c.c_int_l[0].c_int_l_int_count, 2);
	zassert_equal(mm->obsv_metric_c.c_int_l[0].c_int_l_int[0], -1500);
	zassert_equal(mm->obsv_metric_c.c_int_l[0].c_int_l_int[1], INT32_MAX);
}

/* Encode every built-in metric and verify its wire identity and matrix shape. */
ZTEST(obsv_encoder, test_all_metrics_conform_to_cddl_schema)
{
	static nrf_edgeai_obsv_ctx_t ctx;

	static uint32_t cpd_buf[NRF_EDGEAI_OBSV_CPD_STORAGE_BYTES(TEST_NUM_CLASSES) /
				 sizeof(uint32_t)];
	static uint32_t tm_buf[NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(TEST_NUM_CLASSES) /
				sizeof(uint32_t)];
	static uint32_t mcd_buf[NRF_EDGEAI_OBSV_MCD_STORAGE_BYTES(TEST_NUM_CLASSES) /
				 sizeof(uint32_t)];
	static uint32_t med_buf[NRF_EDGEAI_OBSV_MED_STORAGE_BYTES(TEST_NUM_FEATURES) /
				 sizeof(uint32_t)];
	static uint32_t msd_buf[NRF_EDGEAI_OBSV_MSD_STORAGE_BYTES(TEST_NUM_FEATURES) /
				 sizeof(uint32_t)];
	nrf_edgeai_obsv_metric_t cpd;
	nrf_edgeai_obsv_metric_t tm;
	nrf_edgeai_obsv_metric_t mcd;
	nrf_edgeai_obsv_metric_t med;
	nrf_edgeai_obsv_metric_t msd;

	const nrf_edgeai_obsv_model_info_t model = {
		.model_id = UINT16_MAX,
		.num_classes = TEST_NUM_CLASSES,
		.num_features = TEST_NUM_FEATURES,
		.version = UINT32_MAX,
	};

	nrf_edgeai_obsv_metric_cpd_create(&cpd, cpd_buf, TEST_NUM_CLASSES);
	nrf_edgeai_obsv_metric_tm_create(&tm, tm_buf, TEST_NUM_CLASSES);
	nrf_edgeai_obsv_metric_mcd_create(&mcd, mcd_buf, TEST_NUM_CLASSES);
	nrf_edgeai_obsv_metric_med_create(&med, med_buf, TEST_NUM_FEATURES);
	nrf_edgeai_obsv_metric_msd_create(&msd, msd_buf, TEST_NUM_FEATURES);

	((_nrf_obsv_med_hdr_t *)med_buf)->cfg[NRF_EDGEAI_OBSV_MED_CFG_SCALE_P01_MILLI] =
		TEST_MED_P01_MILLI;

	zassert_equal(nrf_edgeai_obsv_init(&ctx, &model), 0);
	zassert_equal(nrf_edgeai_obsv_register(&ctx, &cpd, NULL), 0);
	zassert_equal(nrf_edgeai_obsv_register(&ctx, &tm, NULL), 0);
	zassert_equal(nrf_edgeai_obsv_register(&ctx, &mcd, NULL), 0);
	zassert_equal(nrf_edgeai_obsv_register(&ctx, &med, NULL), 0);
	zassert_equal(nrf_edgeai_obsv_register(&ctx, &msd, NULL), 0);

	const float probs[TEST_NUM_CLASSES] = {0.7f, 0.1f, 0.1f, 0.1f};
	const float feats[TEST_NUM_FEATURES] = {0.1f, 0.2f, 0.3f, 0.4f,
						0.5f, 0.6f, 0.7f, 0.8f};

	zassert_equal(nrf_edgeai_obsv_update_probs(&ctx, probs), 0);
	zassert_equal(nrf_edgeai_obsv_update_features(&ctx, feats, TEST_NUM_FEATURES), 0);

	static uint8_t buf[NRF_EDGEAI_OBSV_ENCODE_LIST_BUF_SIZE(1)];
	nrf_edgeai_obsv_ctx_t *ctxs[] = {&ctx};
	size_t len = nrf_edgeai_obsv_encode_list(ctxs, ARRAY_SIZE(ctxs), buf, sizeof(buf));

	zassert_true(len > 0,
		     "all-metrics payload exceeds NRF_EDGEAI_OBSV_ENCODE_LIST_BUF_SIZE(1) = %zu",
		     sizeof(buf));

	static struct obsv_list cddl_decoded;
	size_t consumed = 0;

	int rc = cbor_decode_obsv_list(buf, len, &cddl_decoded, &consumed);

	zassert_equal(rc, 0, "output does not conform to obsv-list schema (rc=%d)", rc);
	zassert_equal(consumed, len);
	zassert_equal(cddl_decoded.obsv_list_obsv_payload_m_count, 1U);

	const struct obsv_payload *p = &cddl_decoded.obsv_list_obsv_payload_m[0];

	zassert_equal(p->obsv_payload_num_inferences, 1U);
	zassert_equal(p->obsv_payload_num_features, 1U);
	zassert_equal(p->obsv_payload_metrics_obsv_metric_m_count, 5);

	/* Registration order; only the class predictions distribution and the mel
	 * energy descriptor report config ("c").
	 */
	static const struct {
		uint32_t id;
		uint32_t version;
		uint16_t rows;
		uint16_t cols;
		bool has_config;
	} expected[] = {
		{NRF_EDGEAI_OBSV_METRIC_ID_CLASS_PRED_DIST, 1,
		 NRF_EDGEAI_OBSV_CPD_ROW_GROUPS * TEST_NUM_CLASSES,
		 CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_BIN_NUM, true},
		{NRF_EDGEAI_OBSV_METRIC_ID_TRANSITION_MATRIX, 1, TEST_NUM_CLASSES, TEST_NUM_CLASSES,
		 false},
		{NRF_EDGEAI_OBSV_METRIC_ID_MODEL_CERTAINTY_DESC, 1, NRF_EDGEAI_OBSV_MCD_NUM_ROWS,
		 CONFIG_NRF_EDGEAI_OBSV_MODEL_CERTAINTY_DESC_BIN_NUM, false},
		{NRF_EDGEAI_OBSV_METRIC_ID_MEL_ENERGY_DESC, 2, NRF_EDGEAI_OBSV_MED_NUM_ROWS,
		 CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_BIN_NUM, true},
		{NRF_EDGEAI_OBSV_METRIC_ID_MEL_SPECTRAL_DESC, 1, NRF_EDGEAI_OBSV_MSD_NUM_ROWS,
		 CONFIG_NRF_EDGEAI_OBSV_MEL_SPECTRAL_DESC_BIN_NUM, false},
	};

	for (size_t i = 0; i < ARRAY_SIZE(expected); i++) {
		const struct obsv_metric *m = &p->obsv_payload_metrics_obsv_metric_m[i];

		zassert_equal(m->obsv_metric_id, expected[i].id, "metric %zu", i);
		zassert_equal(m->obsv_metric_v, expected[i].version, "metric %zu has wrong version",
			      i);
		zassert_equal(m->d_uint_l_count, expected[i].rows, "metric %zu has wrong row count",
			      i);
		zassert_equal(m->obsv_metric_c_present, expected[i].has_config, "metric %zu", i);

		for (size_t r = 0; r < m->d_uint_l_count; r++) {
			zassert_equal(m->d_uint_l[r].d_uint_l_uint_count, expected[i].cols,
				      "metric %zu row %zu has wrong column count", i, r);
		}
	}

	const struct c_int_l *med_cfg =
		&p->obsv_payload_metrics_obsv_metric_m[3].obsv_metric_c.c_int_l[0];

	zassert_equal(med_cfg->c_int_l_int[NRF_EDGEAI_OBSV_MED_CFG_SCALE_P01_MILLI],
		      TEST_MED_P01_MILLI, "negative p01 must round-trip as a signed int");
	zassert_equal(med_cfg->c_int_l_int[NRF_EDGEAI_OBSV_MED_CFG_SCALE_P99_MILLI],
		      CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_SCALE_P99_MILLI);
}
