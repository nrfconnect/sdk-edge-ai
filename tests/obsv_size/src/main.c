/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stdint.h>
#include <string.h>

#include <zephyr/ztest.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_encode_sizes.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#define NUM_CLASSES  16U
#define NUM_FEATURES 64U

BUILD_ASSERT(CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES == NUM_CLASSES);
BUILD_ASSERT(CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_BIN_NUM == 16);
BUILD_ASSERT(CONFIG_NRF_EDGEAI_OBSV_MODEL_CERTAINTY_DESC_BIN_NUM == 16);
BUILD_ASSERT(CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_BIN_NUM == 16);
BUILD_ASSERT(CONFIG_NRF_EDGEAI_OBSV_MEL_SPECTRAL_DESC_BIN_NUM == 16);

struct model_fixture {
	nrf_edgeai_obsv_ctx_t ctx;
	nrf_edgeai_obsv_metric_t cpd;
	nrf_edgeai_obsv_metric_t tm;
	nrf_edgeai_obsv_metric_t mcd;
	nrf_edgeai_obsv_metric_t med;
	nrf_edgeai_obsv_metric_t msd;
	uint32_t cpd_buf[NRF_EDGEAI_OBSV_CPD_STORAGE_BYTES(NUM_CLASSES) / sizeof(uint32_t)];
	uint32_t tm_buf[NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(NUM_CLASSES) / sizeof(uint32_t)];
	uint32_t mcd_buf[NRF_EDGEAI_OBSV_MCD_STORAGE_BYTES(NUM_CLASSES) / sizeof(uint32_t)];
	uint32_t med_buf[NRF_EDGEAI_OBSV_MED_STORAGE_BYTES(NUM_FEATURES) / sizeof(uint32_t)];
	uint32_t msd_buf[NRF_EDGEAI_OBSV_MSD_STORAGE_BYTES(NUM_FEATURES) / sizeof(uint32_t)];
};

#define SATURATE_COUNTERS(fixture, member, hdr_t)                                                  \
	memset((uint8_t *)(fixture)->member + sizeof(hdr_t), 0xFF,                                 \
	       sizeof((fixture)->member) - sizeof(hdr_t))

static struct model_fixture first;
static struct model_fixture second;

static void setup_model(struct model_fixture *fixture, uint16_t model_id, uint16_t num_classes)
{
	const nrf_edgeai_obsv_model_info_t model = {
		.model_id = model_id,
		.num_classes = num_classes,
		.num_features = NUM_FEATURES,
		.version = UINT32_MAX,
	};

	memset(fixture, 0, sizeof(*fixture));

	nrf_edgeai_obsv_metric_cpd_create(&fixture->cpd, fixture->cpd_buf, num_classes);
	nrf_edgeai_obsv_metric_tm_create(&fixture->tm, fixture->tm_buf, num_classes);
	nrf_edgeai_obsv_metric_mcd_create(&fixture->mcd, fixture->mcd_buf, num_classes);
	nrf_edgeai_obsv_metric_med_create(&fixture->med, fixture->med_buf, NUM_FEATURES);
	nrf_edgeai_obsv_metric_msd_create(&fixture->msd, fixture->msd_buf, NUM_FEATURES);

	zassert_ok(nrf_edgeai_obsv_init(&fixture->ctx, &model));
	zassert_ok(nrf_edgeai_obsv_register(&fixture->ctx, &fixture->cpd, NULL));
	zassert_ok(nrf_edgeai_obsv_register(&fixture->ctx, &fixture->tm, NULL));
	zassert_ok(nrf_edgeai_obsv_register(&fixture->ctx, &fixture->mcd, NULL));
	zassert_ok(nrf_edgeai_obsv_register(&fixture->ctx, &fixture->med, NULL));
	zassert_ok(nrf_edgeai_obsv_register(&fixture->ctx, &fixture->msd, NULL));

	SATURATE_COUNTERS(fixture, cpd_buf, _nrf_obsv_cpd_hdr_t);
	SATURATE_COUNTERS(fixture, tm_buf, _nrf_obsv_tm_hdr_t);
	SATURATE_COUNTERS(fixture, mcd_buf, _nrf_obsv_mcd_hdr_t);
	SATURATE_COUNTERS(fixture, med_buf, _nrf_obsv_med_hdr_t);
	SATURATE_COUNTERS(fixture, msd_buf, _nrf_obsv_msd_hdr_t);
	fixture->ctx.state.num_inferences = UINT32_MAX;
	fixture->ctx.state.num_features = UINT32_MAX;
}

ZTEST_SUITE(obsv_size, NULL, NULL, NULL, NULL, NULL);

ZTEST(obsv_size, test_saturated_payloads_fit_encode_budget)
{
	setup_model(&first, UINT16_MAX, NUM_CLASSES);
	setup_model(&second, UINT16_MAX - 1U, NUM_CLASSES);

	static uint8_t single_buf[NRF_EDGEAI_OBSV_ENCODE_LIST_BUF_SIZE(1)];
	nrf_edgeai_obsv_ctx_t *single_ctx[] = {&first.ctx};
	size_t len = nrf_edgeai_obsv_encode_list(single_ctx, ARRAY_SIZE(single_ctx), single_buf,
						 sizeof(single_buf));

	zassert_true(len > 0U, "saturated payload exceeds the one-context budget");

	static uint8_t combined_buf[NRF_EDGEAI_OBSV_ENCODE_LIST_BUF_SIZE(2)];
	nrf_edgeai_obsv_ctx_t *ctxs[] = {&first.ctx, &second.ctx};

	len = nrf_edgeai_obsv_encode_list_and_reset(ctxs, ARRAY_SIZE(ctxs), combined_buf,
						    sizeof(combined_buf));

	zassert_true(len > 0U, "saturated payloads exceed the two-context budget");
	zassert_equal(first.ctx.state.num_inferences, 0U);
	zassert_equal(first.ctx.state.num_features, 0U);
	zassert_equal(second.ctx.state.num_inferences, 0U);
	zassert_equal(second.ctx.state.num_features, 0U);
}
