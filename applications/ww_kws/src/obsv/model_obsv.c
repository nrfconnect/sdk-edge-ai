/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/logging/log.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_memfault.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#include "model_obsv.h"

LOG_MODULE_REGISTER(model_obsv);

/* Create @metric with @buf for @n entries, register it with @o->ctx, and return
 * on failure. Used once per enabled metric in model_obsv_init().
 */
#define MODEL_OBSV_REGISTER(o, metric, create_fn, buf, n, name)                                    \
	do {                                                                                       \
		create_fn(&(o)->metric, (o)->buf, (n));                                            \
		int _err = nrf_edgeai_obsv_register(&(o)->ctx, &(o)->metric, NULL);                \
		if (_err) {                                                                        \
			LOG_ERR("%s metric registration failed (err %d)", name, _err);             \
			return _err;                                                               \
		}                                                                                  \
	} while (0)

int model_obsv_init(struct model_obsv *o, const nrf_edgeai_obsv_model_info_t *info,
		    uint16_t num_features)
{
	int err;

	err = nrf_edgeai_obsv_init(&o->ctx, info);
	if (err) {
		LOG_ERR("Observability init failed (err %d)", err);
		return err;
	}

	const uint16_t n_classes = info->num_classes;

	/* PROBS-stream metrics (sized/created for the class count). */
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_DISTRIBUTION)
	MODEL_OBSV_REGISTER(o, pd, nrf_edgeai_obsv_metric_pd_create, pd_buf, n_classes, "PD");
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_TRANSITION_MATRIX)
	MODEL_OBSV_REGISTER(o, tm, nrf_edgeai_obsv_metric_tm_create, tm_buf, n_classes, "TM");
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PREDICTION_SWITCHING_RATE)
	MODEL_OBSV_REGISTER(o, psr, nrf_edgeai_obsv_metric_psr_create, psr_buf, n_classes, "PSR");
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_ENTROPY_DIST)
	MODEL_OBSV_REGISTER(o, ped, nrf_edgeai_obsv_metric_ped_create, ped_buf, n_classes, "PED");
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_TOP2_MARGIN_DIST)
	MODEL_OBSV_REGISTER(o, pmd, nrf_edgeai_obsv_metric_pmd_create, pmd_buf, n_classes, "PMD");
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_STREAK_DIST)
	MODEL_OBSV_REGISTER(o, csd, nrf_edgeai_obsv_metric_csd_create, csd_buf, n_classes, "CSD");
#endif

	/* FEATURES-stream metrics (sized/created for the feature-vector length). */
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_ENERGY_DESC)
	MODEL_OBSV_REGISTER(o, med, nrf_edgeai_obsv_metric_med_create, med_buf, num_features,
		"MED");
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_SPECTRAL_DESC)
	MODEL_OBSV_REGISTER(o, msd, nrf_edgeai_obsv_metric_msd_create, msd_buf, num_features,
		"MSD");
#endif

	err = nrf_edgeai_obsv_memfault_init(&o->ctx);
	if (err) {
		LOG_ERR("Memfault transport init failed (err %d)", err);
		return err;
	}

	return 0;
}

void model_obsv_update_probs(struct model_obsv *o, const float *probs)
{
	int err = nrf_edgeai_obsv_update_probs(&o->ctx, probs);

	if (err) {
		LOG_ERR("Failed to update obsv probs (err %d)", err);
	}
}

void model_obsv_update_features(struct model_obsv *o, const float *feats, uint16_t n)
{
	int err = nrf_edgeai_obsv_update_features(&o->ctx, feats, n);

	if (err) {
		LOG_ERR("Failed to update obsv features (err %d)", err);
	}
}
