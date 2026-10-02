/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_memfault.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#include "../model_utils.h"
#include "kws_obsv.h"

LOG_MODULE_REGISTER(kws_obsv);

/* Model output classes, see nrf_edgeai_generated/nrf_edgeai_user_model_labels.h. */
#define KWS_OBSV_CLASSES 12U

/* Mel feature vector length produced by the model DSP front end. */
#define KWS_OBSV_FEATURES 40U

#define OBSV_WORDS(bytes) DIV_ROUND_UP(bytes, sizeof(uint32_t))

BUILD_ASSERT(CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES >= KWS_OBSV_CLASSES,
	     "CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES is below the model class count");

static nrf_edgeai_obsv_ctx_t ctx;

/* One descriptor and one counter buffer per metric. Only cpd and tm scale with
 * the class count; the other buffers are fixed by their row and bin counts.
 */
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_PRED_DIST)
static nrf_edgeai_obsv_metric_t cpd;
static uint32_t cpd_buf[OBSV_WORDS(NRF_EDGEAI_OBSV_CPD_STORAGE_BYTES(KWS_OBSV_CLASSES))];
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_TRANSITION_MATRIX)
static nrf_edgeai_obsv_metric_t tm;
static uint32_t tm_buf[OBSV_WORDS(NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(KWS_OBSV_CLASSES))];
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MODEL_CERTAINTY_DESC)
static nrf_edgeai_obsv_metric_t mcd;
static uint32_t mcd_buf[OBSV_WORDS(NRF_EDGEAI_OBSV_MCD_STORAGE_BYTES(KWS_OBSV_CLASSES))];
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_ENERGY_DESC)
static nrf_edgeai_obsv_metric_t med;
static uint32_t med_buf[OBSV_WORDS(NRF_EDGEAI_OBSV_MED_STORAGE_BYTES(KWS_OBSV_FEATURES))];
BUILD_ASSERT(KWS_OBSV_FEATURES <= CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_MAX_FEATURES,
	     "CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_MAX_FEATURES is below the feature count");
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_SPECTRAL_DESC)
static nrf_edgeai_obsv_metric_t msd;
static uint32_t msd_buf[OBSV_WORDS(NRF_EDGEAI_OBSV_MSD_STORAGE_BYTES(KWS_OBSV_FEATURES))];
#endif

int kws_obsv_init(nrf_edgeai_t *model)
{
	nrf_edgeai_obsv_model_info_t info;
	int err;

	err = obsv_model_info_from_model(model, KWS_OBSV_CLASSES, &info);
	if (err) {
		return err;
	}

	err = nrf_edgeai_obsv_init(&ctx, &info);
	if (err) {
		LOG_ERR("Observability init failed (err %d)", err);
		return err;
	}

	/* Create each metric over its own buffer, then register it. The dump and CDR
	 * payload follow the registration order.
	 */
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_PRED_DIST)
	nrf_edgeai_obsv_metric_cpd_create(&cpd, cpd_buf, KWS_OBSV_CLASSES);
	err = nrf_edgeai_obsv_register(&ctx, &cpd, NULL);
	if (err) {
		return err;
	}
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_TRANSITION_MATRIX)
	nrf_edgeai_obsv_metric_tm_create(&tm, tm_buf, KWS_OBSV_CLASSES);
	err = nrf_edgeai_obsv_register(&ctx, &tm, NULL);
	if (err) {
		return err;
	}
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MODEL_CERTAINTY_DESC)
	nrf_edgeai_obsv_metric_mcd_create(&mcd, mcd_buf, KWS_OBSV_CLASSES);
	err = nrf_edgeai_obsv_register(&ctx, &mcd, NULL);
	if (err) {
		return err;
	}
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_ENERGY_DESC)
	nrf_edgeai_obsv_metric_med_create(&med, med_buf, KWS_OBSV_FEATURES);
	err = nrf_edgeai_obsv_register(&ctx, &med, NULL);
	if (err) {
		return err;
	}
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_SPECTRAL_DESC)
	nrf_edgeai_obsv_metric_msd_create(&msd, msd_buf, KWS_OBSV_FEATURES);
	err = nrf_edgeai_obsv_register(&ctx, &msd, NULL);
	if (err) {
		return err;
	}
#endif

	err = nrf_edgeai_obsv_memfault_init(&ctx);
	if (err) {
		LOG_ERR("Memfault transport init failed (err %d)", err);
	}

	return err;
}

void kws_obsv_update_features(const float *feats, uint16_t n)
{
	if (n != KWS_OBSV_FEATURES) {
		LOG_ERR("Invalid number of features (expected %d, got %d)", KWS_OBSV_FEATURES, n);
		return;
	}

	int err = nrf_edgeai_obsv_update_features(&ctx, feats, n);

	if (err) {
		LOG_ERR("Failed to update features (err %d)", err);
	}
}

void kws_obsv_update_probs(const float *probs)
{
	int err = nrf_edgeai_obsv_update_probs(&ctx, probs);

	if (err) {
		LOG_ERR("Failed to update probs (err %d)", err);
	}
}
