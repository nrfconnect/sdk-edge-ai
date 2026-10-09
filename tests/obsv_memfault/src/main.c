/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <errno.h>
#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/ztest.h>
#include <memfault/core/custom_data_recording.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_memfault.h>
#include "nrf_edgeai_obsv_memfault_priv.h"
#include "obsv_mock.h"

#define TEST_NUM_CLASSES 4

extern nrf_edgeai_obsv_mflt_staging_t nrf_edgeai_obsv_mflt_staging;
extern struct k_mutex obsv_mflt_lock;
extern const sMemfaultCdrSourceImpl obsv_cdr_source;

static void reset_obsv_mflt_staging(nrf_edgeai_obsv_mflt_staging_t *s)
{
	k_mutex_lock(&obsv_mflt_lock, K_FOREVER);
	memset(s->ctxs, 0, sizeof(s->ctxs));
	s->num_ctxs = 0U;
	s->len = 0U;
	s->ready = false;
	s->collecting = false;
	s->collect_pending = false;
	s->staged_duration_ms = 0U;
	s->last_collect_ms = 0U;
	k_mutex_unlock(&obsv_mflt_lock);
}

static uint32_t get_last_collect_ms(void)
{
	k_mutex_lock(&obsv_mflt_lock, K_FOREVER);
	uint32_t last_collect_ms = nrf_edgeai_obsv_mflt_staging.last_collect_ms;

	k_mutex_unlock(&obsv_mflt_lock);

	return last_collect_ms;
}

static nrf_edgeai_obsv_ctx_t obsv;

static const nrf_edgeai_obsv_model_info_t test_model = {
	.model_id = 1,
	.num_classes = TEST_NUM_CLASSES,
	.version = 1,
};

static void setup_ctx(void)
{
	zassert_equal(nrf_edgeai_obsv_init(&obsv, &test_model), 0);
}

static void before_each(void *data)
{
	ARG_UNUSED(data);

	reset_obsv_mflt_staging(&nrf_edgeai_obsv_mflt_staging);
	memset(&obsv, 0, sizeof(obsv));
	obsv_mock_reset();
}

ZTEST_SUITE(obsv_memfault, NULL, NULL, before_each, NULL, NULL);

/* ---------- Unhappy paths ---------- */

ZTEST(obsv_memfault, test_init_null_returns_einval)
{
	zassert_equal(nrf_edgeai_obsv_memfault_init(NULL), -EINVAL);
}

ZTEST(obsv_memfault, test_collect_before_init_returns_einval)
{
	zassert_equal(nrf_edgeai_obsv_memfault_collect(), -EINVAL);
}

ZTEST(obsv_memfault, test_obsv_update_null_returns_einval)
{
	const float probs[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};

	zassert_equal(nrf_edgeai_obsv_update_probs(NULL, probs), -EINVAL);

	setup_ctx();
	zassert_equal(nrf_edgeai_obsv_update_probs(&obsv, NULL), -EINVAL);
}

ZTEST(obsv_memfault, test_has_cdr_before_collect_returns_false)
{
	sMemfaultCdrMetadata md;

	zassert_false(obsv_cdr_source.has_cdr_cb(&md));
}

ZTEST(obsv_memfault, test_read_before_collect_returns_false)
{
	uint8_t buf[4];

	zassert_false(obsv_cdr_source.read_data_cb(0, buf, sizeof(buf)));
}

ZTEST(obsv_memfault, test_read_out_of_bounds_returns_false)
{
	setup_ctx();
	zassert_equal(nrf_edgeai_obsv_memfault_init(&obsv), 0);
	zassert_equal(nrf_edgeai_obsv_memfault_collect(), 0);

	sMemfaultCdrMetadata md;

	zassert_true(obsv_cdr_source.has_cdr_cb(&md));

	uint8_t buf[1];

	zassert_false(obsv_cdr_source.read_data_cb(md.data_size_bytes, buf, 1));
	zassert_false(obsv_cdr_source.read_data_cb(0, buf, md.data_size_bytes + 1));
}

ZTEST(obsv_memfault, test_has_cdr_after_mark_read_returns_false)
{
	setup_ctx();
	zassert_equal(nrf_edgeai_obsv_memfault_init(&obsv), 0);
	zassert_equal(nrf_edgeai_obsv_memfault_collect(), 0);

	sMemfaultCdrMetadata md;

	obsv_cdr_source.mark_cdr_read_cb();
	zassert_false(obsv_cdr_source.has_cdr_cb(&md));
}

/* ---------- Happy path ---------- */

ZTEST(obsv_memfault, test_init_and_collect_exposes_payload)
{
	setup_ctx();

	zassert_equal(nrf_edgeai_obsv_memfault_init(&obsv), 0, "init failed");
	zassert_equal(nrf_edgeai_obsv_memfault_collect(), 0, "collect failed");

	sMemfaultCdrMetadata md = {0};

	zassert_true(obsv_cdr_source.has_cdr_cb(&md));
	zassert_true(md.data_size_bytes > 0);
	zassert_equal(md.num_mimetypes, 1);
	zassert_not_null(md.collection_reason);
	zassert_equal(md.start_time.type, kMemfaultCurrentTimeType_Unknown);

	/* Verify read_data covers the full staged range and partial reads work. */
	uint8_t read_buf[16];

	zassert_true(md.data_size_bytes <= sizeof(read_buf));
	zassert_true(obsv_cdr_source.read_data_cb(0, read_buf, md.data_size_bytes));

	obsv_cdr_source.mark_cdr_read_cb();
	zassert_false(obsv_cdr_source.has_cdr_cb(&md));
}

/* ---------- Two-context (multi-model) test ---------- */

static nrf_edgeai_obsv_ctx_t obsv_b;

static const nrf_edgeai_obsv_model_info_t model_b_info = {
	.model_id = 2,
	.num_classes = TEST_NUM_CLASSES,
	.version = 1,
};

ZTEST(obsv_memfault, test_two_contexts_produce_combined_payload)
{
	/* Set up context A. */
	setup_ctx();
	zassert_equal(nrf_edgeai_obsv_memfault_init(&obsv), 0, "init ctx A");

	/* Set up context B. */
	memset(&obsv_b, 0, sizeof(obsv_b));
	zassert_equal(nrf_edgeai_obsv_init(&obsv_b, &model_b_info), 0);
	zassert_equal(nrf_edgeai_obsv_memfault_init(&obsv_b), 0, "init ctx B");

	zassert_equal(nrf_edgeai_obsv_memfault_collect(), 0, "collect failed");

	sMemfaultCdrMetadata md = {0};

	zassert_true(obsv_cdr_source.has_cdr_cb(&md));

	/* Mock returns OBSV_MOCK_BYTES_PER_CTX bytes per context. */
	zassert_equal(md.data_size_bytes, 2U * OBSV_MOCK_BYTES_PER_CTX);

	uint8_t read_buf[2 * OBSV_MOCK_BYTES_PER_CTX];

	zassert_true(obsv_cdr_source.read_data_cb(0, read_buf, md.data_size_bytes));
}

/* ---------- No-overwrite behavior ---------- */

/*
 * Metrics are incremental: every collect resets the contexts, so a staged CDR
 * cannot be rebuilt once overwritten. collect() must refuse with -EBUSY until
 * the CDR is drained, without encoding or resetting, and the staged bytes must
 * stay stable for the packetizer's chunked reads.
 */
ZTEST(obsv_memfault, test_collect_busy_until_drained)
{
	setup_ctx();
	zassert_equal(nrf_edgeai_obsv_memfault_init(&obsv), 0);

	zassert_equal(nrf_edgeai_obsv_memfault_collect(), 0, "first collect failed");

	sMemfaultCdrMetadata md = {0};

	zassert_true(obsv_cdr_source.has_cdr_cb(&md));
	zassert_true(md.data_size_bytes > 0, "no payload after first collect");

	uint8_t buf_first[16];

	zassert_true(md.data_size_bytes <= sizeof(buf_first));
	zassert_true(obsv_cdr_source.read_data_cb(0, buf_first, md.data_size_bytes));

	const float probs[TEST_NUM_CLASSES] = {0.7f, 0.1f, 0.1f, 0.1f};

	zassert_equal(nrf_edgeai_obsv_update_probs(&obsv, probs), 0);
	zassert_equal(nrf_edgeai_obsv_update_probs(&obsv, probs), 0);

	/* Second collect before drain: refused, nothing encoded (no reset either). */
	zassert_equal(nrf_edgeai_obsv_memfault_collect(), -EBUSY,
		      "collect before drain must be refused");
	zassert_equal(obsv_mock_call_count(), 1U, "refused collect must not encode/reset");
	zassert_equal(obsv.state.num_inferences, 2U, "refused collect touched the live context");

	sMemfaultCdrMetadata md_busy = {0};
	uint8_t buf_busy[16];

	zassert_true(obsv_cdr_source.has_cdr_cb(&md_busy));
	zassert_equal(md_busy.data_size_bytes, md.data_size_bytes);
	zassert_true(obsv_cdr_source.read_data_cb(0, buf_busy, md_busy.data_size_bytes));
	zassert_mem_equal(buf_first, buf_busy, md.data_size_bytes,
			  "refused collect changed the staged CDR");

	/* After drain, collect succeeds and stages a new payload. */
	obsv_cdr_source.mark_cdr_read_cb();
	zassert_equal(nrf_edgeai_obsv_memfault_collect(), 0, "collect after drain failed");

	sMemfaultCdrMetadata md2 = {0};
	uint8_t buf_second[16];

	zassert_true(obsv_cdr_source.has_cdr_cb(&md2));
	zassert_true(md2.data_size_bytes <= sizeof(buf_second));
	zassert_true(obsv_cdr_source.read_data_cb(0, buf_second, md2.data_size_bytes));
	zassert_true(memcmp(buf_first, buf_second, MIN(md.data_size_bytes, md2.data_size_bytes)) !=
			     0,
		     "collect after drain did not stage a new payload");
}

/* A failed encode stages nothing and must not leave collect() stuck at -EBUSY. */
ZTEST(obsv_memfault, test_failed_encode_allows_next_collect)
{
	setup_ctx();
	zassert_equal(nrf_edgeai_obsv_memfault_init(&obsv), 0);

	obsv_mock_fail_next();
	zassert_equal(nrf_edgeai_obsv_memfault_collect(), -ENODATA);

	sMemfaultCdrMetadata md = {0};

	zassert_false(obsv_cdr_source.has_cdr_cb(&md), "failed encode staged a CDR");

	zassert_equal(nrf_edgeai_obsv_memfault_collect(), 0, "collect after failed encode");
	zassert_true(obsv_cdr_source.has_cdr_cb(&md));
}

/* A refused collect must preserve the start of the current collection interval. */
ZTEST(obsv_memfault, test_refused_collect_preserves_interval_start)
{
	setup_ctx();
	zassert_equal(nrf_edgeai_obsv_memfault_init(&obsv), 0);

	k_sleep(K_MSEC(10));
	zassert_equal(nrf_edgeai_obsv_memfault_collect(), 0);

	uint32_t interval_start_ms = get_last_collect_ms();

	zassert_not_equal(interval_start_ms, 0U);

	k_sleep(K_MSEC(10));
	zassert_equal(nrf_edgeai_obsv_memfault_collect(), -EBUSY);

	zassert_equal(get_last_collect_ms(), interval_start_ms,
		      "refused collect changed the interval start");
}

#if defined(CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT)
extern void auto_collect_work_handler(struct k_work *work);

ZTEST(obsv_memfault, test_auto_collect_work_handler_stages_payload)
{
	setup_ctx();
	zassert_equal(nrf_edgeai_obsv_memfault_init(&obsv), 0);

	/* Invoke the work handler directly — no need to wait for the timer. */
	auto_collect_work_handler(NULL);

	sMemfaultCdrMetadata md = {0};

	zassert_true(obsv_cdr_source.has_cdr_cb(&md), "work handler did not stage a payload");
	zassert_true(md.data_size_bytes > 0);
}

/* A drain after a refused collect stages the backlog without waiting an interval. */
ZTEST(obsv_memfault, test_drain_after_refused_collect_schedules_collect)
{
	setup_ctx();
	zassert_equal(nrf_edgeai_obsv_memfault_init(&obsv), 0);

	zassert_equal(nrf_edgeai_obsv_memfault_collect(), 0);
	zassert_equal(nrf_edgeai_obsv_memfault_collect(), -EBUSY);

	obsv_cdr_source.mark_cdr_read_cb();

	/* Let the system workqueue run the deferred collect. */
	k_sleep(K_MSEC(50));

	sMemfaultCdrMetadata md = {0};

	zassert_true(obsv_cdr_source.has_cdr_cb(&md), "drain did not trigger the deferred collect");
	zassert_equal(obsv_mock_call_count(), 2U);
}

/* A normal drain (no refused collect) must not trigger a short-window collect. */
ZTEST(obsv_memfault, test_drain_without_refusal_does_not_collect)
{
	setup_ctx();
	zassert_equal(nrf_edgeai_obsv_memfault_init(&obsv), 0);

	zassert_equal(nrf_edgeai_obsv_memfault_collect(), 0);
	obsv_cdr_source.mark_cdr_read_cb();

	k_sleep(K_MSEC(50));

	sMemfaultCdrMetadata md = {0};

	zassert_false(obsv_cdr_source.has_cdr_cb(&md), "drain staged an unrequested CDR");
	zassert_equal(obsv_mock_call_count(), 1U);
}
#endif
