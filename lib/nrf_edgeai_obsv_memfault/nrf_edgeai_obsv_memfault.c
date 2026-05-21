/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <errno.h>
#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

#include <memfault/core/custom_data_recording.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_memfault.h>
#include "nrf_edgeai_obsv_memfault_priv.h"

LOG_MODULE_REGISTER(nrf_edgeai_obsv_mflt, CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_LOG_LEVEL);

/* External linkage for internal variables when CONFIG_ZTEST. */
#ifdef CONFIG_ZTEST
#define STATIC_EXCEPT_TEST
#else
#define STATIC_EXCEPT_TEST static
#endif

#define OBSV_CDR_REASON "edgeai_observability"

/*
 * CDR wire format: obsv-list = [+ obsv-payload] (see lib/nrf_edgeai_obsv/obsv.cddl).
 * nrf_edgeai_obsv_encode_list() builds the full array; contexts are encoded
 * sequentially under their individual locks. Cross-context consistency is not
 * guaranteed, which is acceptable for observability data.
 */

static bool has_cdr(sMemfaultCdrMetadata *metadata);
static bool read_data(uint32_t offset, void *data, size_t data_len);
static void mark_read(void);

/* obsv_mflt_lock: protects the staging blob and the Memfault SDK read callbacks
 * (has_cdr/read_data/mark_read). collect() uses nrf_edgeai_obsv_encode() which
 * acquires ctx->lock internally, so collect() must not be called while holding
 * obsv_mflt_lock (to avoid lock inversion). collect() holds obsv_mflt_lock only
 * briefly at the start (to snapshot the context array) and at the end (to commit
 * the new payload). Precondition: init() and collect() must not be called concurrently.
 */
K_MUTEX_DEFINE(obsv_mflt_lock);
STATIC_EXCEPT_TEST nrf_edgeai_obsv_mflt_staging_t nrf_edgeai_obsv_mflt_staging;

/* Registered once at the first nrf_edgeai_obsv_memfault_init() call and never
 * cleared.  The Memfault CDR table has a fixed capacity; re-registering the
 * same source on every staging-reset would fill it up.
 */
static bool cdr_source_registered;

STATIC_EXCEPT_TEST const sMemfaultCdrSourceImpl obsv_cdr_source = {
	.has_cdr_cb = has_cdr,
	.read_data_cb = read_data,
	.mark_cdr_read_cb = mark_read,
};

/* Memfault passes this as const char ** (array of pointers to const strings). */
static const char * const mimetypes[] = {MEMFAULT_CDR_BINARY};

static bool has_cdr(sMemfaultCdrMetadata *metadata)
{
	k_mutex_lock(&obsv_mflt_lock, K_FOREVER);

	if (!nrf_edgeai_obsv_mflt_staging.ready) {
		k_mutex_unlock(&obsv_mflt_lock);
		LOG_DBG("CDR: nothing staged");
		return false;
	}

	*metadata = (sMemfaultCdrMetadata){
		.start_time.type = kMemfaultCurrentTimeType_Unknown,
		.start_time.info.unix_timestamp_secs = 0U,
		.mimetypes = (const char **)mimetypes,
		.num_mimetypes = ARRAY_SIZE(mimetypes),
		.data_size_bytes = nrf_edgeai_obsv_mflt_staging.len,
		.duration_ms = nrf_edgeai_obsv_mflt_staging.staged_duration_ms,
		.collection_reason = OBSV_CDR_REASON,
	};

	LOG_DBG("CDR: %u bytes ready", nrf_edgeai_obsv_mflt_staging.len);

	k_mutex_unlock(&obsv_mflt_lock);
	return true;
}

static bool read_data(uint32_t offset, void *data, size_t data_len)
{
	k_mutex_lock(&obsv_mflt_lock, K_FOREVER);

	if (offset + data_len > nrf_edgeai_obsv_mflt_staging.len) {
		k_mutex_unlock(&obsv_mflt_lock);
		LOG_WRN("read_data out of range: off=%u len=%u staged=%u", offset,
			(unsigned int)data_len, nrf_edgeai_obsv_mflt_staging.len);
		return false;
	}

	memcpy(data, nrf_edgeai_obsv_mflt_staging.buf + offset, data_len);

	LOG_DBG("CDR: off=%u len=%u", offset, (unsigned int)data_len);

	k_mutex_unlock(&obsv_mflt_lock);
	return true;
}

static void mark_read(void)
{
	k_mutex_lock(&obsv_mflt_lock, K_FOREVER);
	nrf_edgeai_obsv_mflt_staging.ready = false;
	k_mutex_unlock(&obsv_mflt_lock);
	LOG_DBG("CDR drained");
}

#if defined(CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT)
STATIC_EXCEPT_TEST void auto_collect_work_handler(struct k_work *work);
static K_WORK_DELAYABLE_DEFINE(auto_collect_work, auto_collect_work_handler);

STATIC_EXCEPT_TEST void auto_collect_work_handler(struct k_work *work)
{
	ARG_UNUSED(work);

	(void)nrf_edgeai_obsv_memfault_collect();

	k_timeout_t delay =
		K_SECONDS(CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT_INTERVAL_SEC);

	(void)k_work_reschedule(&auto_collect_work, delay);
}
#endif

int nrf_edgeai_obsv_memfault_init(nrf_edgeai_obsv_ctx_t *ctx)
{
	if (ctx == NULL) {
		return -EINVAL;
	}

	k_mutex_lock(&obsv_mflt_lock, K_FOREVER);

	for (uint8_t i = 0; i < nrf_edgeai_obsv_mflt_staging.num_ctxs; i++) {
		if (nrf_edgeai_obsv_mflt_staging.ctxs[i] == ctx) {
			k_mutex_unlock(&obsv_mflt_lock);
			return -EALREADY;
		}
	}

	if (nrf_edgeai_obsv_mflt_staging.num_ctxs >= CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_MAX_CONTEXTS) {
		k_mutex_unlock(&obsv_mflt_lock);
		LOG_ERR("init: max contexts (%d) reached",
			CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_MAX_CONTEXTS);
		return -ENOMEM;
	}

	bool first = (nrf_edgeai_obsv_mflt_staging.num_ctxs == 0);

	nrf_edgeai_obsv_mflt_staging.ctxs[nrf_edgeai_obsv_mflt_staging.num_ctxs] = ctx;
	nrf_edgeai_obsv_mflt_staging.num_ctxs++;
	nrf_edgeai_obsv_mflt_staging.last_collect_ms = 0U;
	nrf_edgeai_obsv_mflt_staging.staged_duration_ms = 0U;

	k_mutex_unlock(&obsv_mflt_lock);

	if (first && !cdr_source_registered) {
		memfault_cdr_register_source(&obsv_cdr_source);
		cdr_source_registered = true;

#if defined(CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT)
		k_timeout_t delay =
			K_SECONDS(CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT_INTERVAL_SEC);

		(void)k_work_reschedule(&auto_collect_work, delay);
#endif
	}

	return 0;
}

int nrf_edgeai_obsv_memfault_collect(void)
{
	k_mutex_lock(&obsv_mflt_lock, K_FOREVER);

	uint8_t num_ctxs = nrf_edgeai_obsv_mflt_staging.num_ctxs;
	nrf_edgeai_obsv_ctx_t *ctxs[CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_MAX_CONTEXTS];

	if (num_ctxs == 0) {
		k_mutex_unlock(&obsv_mflt_lock);
		LOG_ERR("collect: not initialized");
		return -EINVAL;
	}

	memcpy(ctxs, nrf_edgeai_obsv_mflt_staging.ctxs, num_ctxs * sizeof(ctxs[0]));

	k_mutex_unlock(&obsv_mflt_lock);

	/*
	 * Encode all contexts into a temporary buffer as obsv-list = [+ obsv-payload].
	 * nrf_edgeai_obsv_encode_list() acquires each ctx->lock internally,
	 * so obsv_mflt_lock must not be held here.
	 */
	uint8_t tmp[NRF_EDGEAI_OBSV_ENCODE_LIST_BUFSZ];

	size_t total_len = nrf_edgeai_obsv_encode_list(
		(nrf_edgeai_obsv_ctx_t *const *)ctxs, num_ctxs, tmp, sizeof(tmp));

	if (total_len == 0U) {
		LOG_ERR("collect: CBOR encode failed");
		return -ENODATA;
	}

	if (total_len > UINT16_MAX) {
		LOG_ERR("collect: payload too large (%zu bytes)", total_len);
		return -ENODATA;
	}

	k_mutex_lock(&obsv_mflt_lock, K_FOREVER);

	memcpy(nrf_edgeai_obsv_mflt_staging.buf, tmp, total_len);
	nrf_edgeai_obsv_mflt_staging.len = (uint16_t)total_len;

	const uint32_t now_ms = k_uptime_get_32();

	if (nrf_edgeai_obsv_mflt_staging.last_collect_ms != 0U) {
		nrf_edgeai_obsv_mflt_staging.staged_duration_ms =
			now_ms - nrf_edgeai_obsv_mflt_staging.last_collect_ms;
	} else {
		nrf_edgeai_obsv_mflt_staging.staged_duration_ms = 0U;
	}
	nrf_edgeai_obsv_mflt_staging.last_collect_ms = now_ms;
	nrf_edgeai_obsv_mflt_staging.ready = true;

	k_mutex_unlock(&obsv_mflt_lock);

	LOG_INF("CDR staged: %u bytes (%u context(s))", (unsigned int)total_len, num_ctxs);
	LOG_HEXDUMP_DBG(nrf_edgeai_obsv_mflt_staging.buf, nrf_edgeai_obsv_mflt_staging.len,
			"payload");

	return 0;
}
