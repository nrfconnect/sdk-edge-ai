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

#include <memfault/core/event_storage.h>
#include <memfault/metrics/metrics.h>
#include <memfault/metrics/platform/overrides.h>
#include <memfault/metrics/serializer.h>
#include <memfault_ncs.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_memfault.h>

#include "nrf_edgeai_obsv_memfault_priv.h"

LOG_MODULE_REGISTER(nrf_edgeai_obsv_mflt, CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_LOG_LEVEL);

static struct nrf_edgeai_obsv_mflt_registration
	registrations[CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_MAX_CONTEXTS];
static uint8_t num_registrations;

K_MUTEX_DEFINE(registrations_lock);

struct export_state {
	const struct nrf_edgeai_obsv_memfault_keys *keys;
	int err;
};

static void set_metric(struct export_state *state, MemfaultMetricId key, uint32_t value)
{
	int err = memfault_metrics_heartbeat_set_unsigned(key, value);

	if ((err != 0) && (state->err == 0)) {
		state->err = err;
	}
}

static bool export_snapshot(const nrf_edgeai_obsv_metric_snapshot_t *snapshot, void *user)
{
	struct export_state *state = user;
	const struct nrf_edgeai_obsv_memfault_mcd_keys *keys = &state->keys->mcd;

	if (snapshot->metric_id != NRF_EDGEAI_OBSV_METRIC_ID_MODEL_CERTAINTY_DESC) {
		return true;
	}

	struct nrf_edgeai_obsv_mcd_snapshot mcd;

	state->err = nrf_edgeai_obsv_metric_mcd_snapshot_decode(snapshot, &mcd);
	if (state->err != 0) {
		return false;
	}

	if (mcd.num_bins != ARRAY_SIZE(keys->entropy)) {
		state->err = -EBADMSG;
		return false;
	}

	set_metric(state, keys->version, snapshot->version);

	for (uint16_t i = 0; i < mcd.num_bins; i++) {
		set_metric(state, keys->entropy[i], mcd.entropy[i]);
		set_metric(state, keys->top2_margin[i], mcd.top2_margin[i]);
	}

	set_metric(state, keys->switches, mcd.switches);
	set_metric(state, keys->comparisons, mcd.comparisons);
	set_metric(state, keys->majority_frames, mcd.majority_frames);
	set_metric(state, keys->confident_switches, mcd.confident_switches);

	return true;
}

static void export_model(struct export_state *state, const nrf_edgeai_obsv_core_t *core)
{
	const struct nrf_edgeai_obsv_memfault_model_keys *keys = &state->keys->model;

	set_metric(state, keys->model_id, core->model.model_id);
	set_metric(state, keys->model_version, core->model.version);
	set_metric(state, keys->num_inferences, core->num_inferences);
	set_metric(state, keys->num_feature_updates, core->num_features);
}

static int collect_registration(const struct nrf_edgeai_obsv_mflt_registration *registration)
{
	nrf_edgeai_obsv_ctx_t *ctx = registration->ctx;
	struct export_state state = {
		.keys = registration->keys,
	};

	k_mutex_lock(&ctx->lock, K_FOREVER);

	if (ctx->state.num_inferences == 0U) {
		k_mutex_unlock(&ctx->lock);
		return 0;
	}

	export_model(&state, &ctx->state);

	int err = nrf_edgeai_obsv_core_for_each_metric(&ctx->state, export_snapshot, &state);

	if ((err == 0) && (state.err == 0)) {
		err = nrf_edgeai_obsv_core_reset(&ctx->state);
	}

	k_mutex_unlock(&ctx->lock);

	return (err != 0) ? err : state.err;
}

int nrf_edgeai_obsv_memfault_init(nrf_edgeai_obsv_ctx_t *ctx,
				  const struct nrf_edgeai_obsv_memfault_keys *keys)
{
	if ((ctx == NULL) || (keys == NULL)) {
		return -EINVAL;
	}

	k_mutex_lock(&registrations_lock, K_FOREVER);

	for (uint8_t i = 0; i < num_registrations; i++) {
		if (registrations[i].ctx == ctx) {
			k_mutex_unlock(&registrations_lock);
			return -EALREADY;
		}
	}

	if (num_registrations >= ARRAY_SIZE(registrations)) {
		k_mutex_unlock(&registrations_lock);
		return -ENOMEM;
	}

	registrations[num_registrations++] = (struct nrf_edgeai_obsv_mflt_registration){
		.ctx = ctx,
		.keys = keys,
	};

	k_mutex_unlock(&registrations_lock);
	return 0;
}

int nrf_edgeai_obsv_memfault_collect(void)
{
	struct nrf_edgeai_obsv_mflt_registration local[ARRAY_SIZE(registrations)];

	k_mutex_lock(&registrations_lock, K_FOREVER);

	uint8_t count = num_registrations;

	memcpy(local, registrations, count * sizeof(local[0]));
	k_mutex_unlock(&registrations_lock);

	if (count == 0U) {
		return -EINVAL;
	}

	/* Memfault silently drops a heartbeat that does not fit in event storage,
	 * so leave the counters accumulating for the next heartbeat instead.
	 */
	if (memfault_event_storage_bytes_free() <
	    memfault_metrics_heartbeat_compute_worst_case_storage_size()) {
		return -ENOSPC;
	}

	int first_err = 0;

	for (uint8_t i = 0; i < count; i++) {
		int err = collect_registration(&local[i]);

		if ((err != 0) && (first_err == 0)) {
			first_err = err;
		}
	}

	return first_err;
}

void memfault_metrics_heartbeat_collect_data(void)
{
	memfault_ncs_metrics_collect_data();

	int err = nrf_edgeai_obsv_memfault_collect();

	if (err == -ENOSPC) {
		LOG_INF("Memfault event storage full, holding observability counters");
	} else if ((err != 0) && (err != -EINVAL)) {
		LOG_WRN("Failed to collect observability heartbeat metrics (err %d)", err);
	}
}
