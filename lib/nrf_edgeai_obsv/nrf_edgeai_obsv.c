/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <errno.h>
#include <string.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>

int nrf_edgeai_obsv_init(nrf_edgeai_obsv_ctx_t *ctx, const nrf_edgeai_obsv_model_info_t *model)
{
	if ((ctx == NULL) || (model == NULL)) {
		return -EINVAL;
	}

	k_mutex_init(&ctx->lock);

	return nrf_edgeai_obsv_core_init(&ctx->state, model);
}

int nrf_edgeai_obsv_register(nrf_edgeai_obsv_ctx_t *ctx, nrf_edgeai_obsv_metric_t *metric,
			     const void *cfg)
{
	if ((ctx == NULL) || (metric == NULL)) {
		return -EINVAL;
	}

	k_mutex_lock(&ctx->lock, K_FOREVER);

	int ret = nrf_edgeai_obsv_core_register(&ctx->state, metric, cfg);

	k_mutex_unlock(&ctx->lock);

	return ret;
}

int nrf_edgeai_obsv_deregister(nrf_edgeai_obsv_ctx_t *ctx, nrf_edgeai_obsv_metric_t *metric)
{
	if ((ctx == NULL) || (metric == NULL)) {
		return -EINVAL;
	}

	k_mutex_lock(&ctx->lock, K_FOREVER);

	int ret = nrf_edgeai_obsv_core_deregister(&ctx->state, metric);

	k_mutex_unlock(&ctx->lock);

	return ret;
}

int nrf_edgeai_obsv_reset(nrf_edgeai_obsv_ctx_t *ctx)
{
	if (ctx == NULL) {
		return -EINVAL;
	}

	k_mutex_lock(&ctx->lock, K_FOREVER);

	int ret = nrf_edgeai_obsv_core_reset(&ctx->state);

	k_mutex_unlock(&ctx->lock);

	return ret;
}

int nrf_edgeai_obsv_update_probs(nrf_edgeai_obsv_ctx_t *ctx, const float *probs)
{
	if ((ctx == NULL) || (probs == NULL)) {
		return -EINVAL;
	}

	k_mutex_lock(&ctx->lock, K_FOREVER);

	int ret = nrf_edgeai_obsv_core_update_probs(&ctx->state, probs);

	k_mutex_unlock(&ctx->lock);

	return ret;
}

int nrf_edgeai_obsv_update_features(nrf_edgeai_obsv_ctx_t *ctx, const float *feats, uint16_t n)
{
	if ((ctx == NULL) || (feats == NULL)) {
		return -EINVAL;
	}

	k_mutex_lock(&ctx->lock, K_FOREVER);

	int ret = nrf_edgeai_obsv_core_update_features(&ctx->state, feats, n);

	k_mutex_unlock(&ctx->lock);

	return ret;
}

int nrf_edgeai_obsv_for_each_metric(nrf_edgeai_obsv_ctx_t *ctx, nrf_edgeai_obsv_metric_cb_t cb,
				    void *user)
{
	if ((ctx == NULL) || (cb == NULL)) {
		return -EINVAL;
	}

	k_mutex_lock(&ctx->lock, K_FOREVER);

	int ret = nrf_edgeai_obsv_core_for_each_metric(&ctx->state, cb, user);

	k_mutex_unlock(&ctx->lock);

	return ret;
}
