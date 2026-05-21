/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <errno.h>
#include <string.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_core.h>

int nrf_edgeai_obsv_core_init(nrf_edgeai_obsv_core_t *p_ctx,
			      const nrf_edgeai_obsv_model_info_t *p_model)
{
	if ((p_ctx == NULL) || (p_model == NULL)) {
		return -EINVAL;
	}

	if (p_model->num_classes == 0 ||
	    p_model->num_classes > CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES) {
		return -EINVAL;
	}

	memset(p_ctx, 0, sizeof(*p_ctx));
	p_ctx->model = *p_model;

	return 0;
}

int nrf_edgeai_obsv_core_reset(nrf_edgeai_obsv_core_t *p_ctx)
{
	if (p_ctx == NULL) {
		return -EINVAL;
	}

	p_ctx->num_inferences = 0;

	nrf_edgeai_obsv_metric_t *p_metric = p_ctx->p_metrics_list;

	while (p_metric != NULL) {
		if (p_metric->clear != NULL) {
			p_metric->clear(p_metric->priv);
		}
		p_metric = p_metric->p_next;
	}

	return 0;
}

int nrf_edgeai_obsv_core_register(nrf_edgeai_obsv_core_t *p_ctx, nrf_edgeai_obsv_metric_t *p_metric,
				  const void *p_cfg)
{
	if ((p_ctx == NULL) || (p_metric == NULL)) {
		return -EINVAL;
	}

	if (p_metric->snapshot == NULL) {
		return -EINVAL;
	}

	if (p_ctx->num_metrics == UINT8_MAX) {
		return -ENOMEM;
	}

	nrf_edgeai_obsv_metric_t *p_tail = NULL;
	nrf_edgeai_obsv_metric_t *p = p_ctx->p_metrics_list;

	while (p != NULL) {
		if (p == p_metric) {
			return -EALREADY;
		}
		p_tail = p;
		p = p->p_next;
	}

	if (p_metric->init != NULL) {
		p_metric->init(p_cfg, p_metric->priv);
	}

	p_metric->p_next = NULL;

	if (p_tail == NULL) {
		p_ctx->p_metrics_list = p_metric;
	} else {
		p_tail->p_next = p_metric;
	}

	p_ctx->num_metrics++;

	return 0;
}

int nrf_edgeai_obsv_core_deregister(nrf_edgeai_obsv_core_t *p_ctx,
				    nrf_edgeai_obsv_metric_t *p_metric)
{
	if ((p_ctx == NULL) || (p_metric == NULL)) {
		return -EINVAL;
	}

	bool removed = false;

	if (p_ctx->p_metrics_list == p_metric) {
		p_ctx->p_metrics_list = p_metric->p_next;
		removed = true;
	} else if (p_ctx->p_metrics_list != NULL) {
		nrf_edgeai_obsv_metric_t *p_prev = p_ctx->p_metrics_list;

		while ((p_prev->p_next != NULL) && (p_prev->p_next != p_metric)) {
			p_prev = p_prev->p_next;
		}

		if (p_prev->p_next == p_metric) {
			p_prev->p_next = p_metric->p_next;
			removed = true;
		}
	}

	if (removed) {
		p_metric->p_next = NULL;
		p_ctx->num_metrics--;
	}

	return 0;
}

int nrf_edgeai_obsv_core_update(nrf_edgeai_obsv_core_t *p_ctx, const float *p_probs)
{
	if ((p_ctx == NULL) || (p_probs == NULL)) {
		return -EINVAL;
	}

	nrf_edgeai_obsv_metric_t *p_metric = p_ctx->p_metrics_list;
	uint16_t num_classes = p_ctx->model.num_classes;

	while (p_metric != NULL) {
		if (p_metric->update != NULL) {
			p_metric->update(p_probs, num_classes, p_metric->priv);
		}
		p_metric = p_metric->p_next;
	}

	p_ctx->num_inferences++;

	return 0;
}

int nrf_edgeai_obsv_core_for_each_metric(nrf_edgeai_obsv_core_t *p_ctx,
					 nrf_edgeai_obsv_metric_cb_t cb, void *user)
{
	if ((p_ctx == NULL) || (cb == NULL)) {
		return -EINVAL;
	}

	nrf_edgeai_obsv_metric_t *p_metric = p_ctx->p_metrics_list;

	while (p_metric != NULL) {
		if (p_metric->finalize != NULL) {
			p_metric->finalize(p_metric->priv);
		}

		nrf_edgeai_obsv_metric_snapshot_t snap;

		p_metric->snapshot(&snap, p_metric->priv);

		if (!cb(&snap, user)) {
			break;
		}

		p_metric = p_metric->p_next;
	}

	return 0;
}
