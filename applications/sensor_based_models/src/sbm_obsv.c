/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <errno.h>
#include <stdlib.h>

#include <zephyr/logging/log.h>
#include <zephyr/sys/__assert.h>
#include <zephyr/sys/util.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_memfault.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#include "inference_postprocessing.h"
#include "sbm_obsv.h"

LOG_MODULE_REGISTER(sbm_obsv);

#define SBM_OBSV_CLASSES CLASS_LABEL_COUNT
#define OBSV_WORDS(bytes) DIV_ROUND_UP(bytes, sizeof(uint32_t))

BUILD_ASSERT(CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES >= SBM_OBSV_CLASSES,
	     "CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES is below the model class count");

static nrf_edgeai_obsv_ctx_t ctx;

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_PRED_DIST)
static nrf_edgeai_obsv_metric_t cpd;
static uint32_t cpd_buf[OBSV_WORDS(NRF_EDGEAI_OBSV_CPD_STORAGE_BYTES(SBM_OBSV_CLASSES))];
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_TRANSITION_MATRIX)
static nrf_edgeai_obsv_metric_t tm;
static uint32_t tm_buf[OBSV_WORDS(NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(SBM_OBSV_CLASSES))];
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MODEL_CERTAINTY_DESC)
static nrf_edgeai_obsv_metric_t mcd;
static uint32_t mcd_buf[OBSV_WORDS(NRF_EDGEAI_OBSV_MCD_STORAGE_BYTES(SBM_OBSV_CLASSES))];
#endif

static int model_info_get(nrf_edgeai_t *model, nrf_edgeai_obsv_model_info_t *info)
{
	__ASSERT_NO_MSG(model != NULL);
	__ASSERT_NO_MSG(info != NULL);
	__ASSERT_NO_MSG(nrf_edgeai_model_outputs_num(model) == SBM_OBSV_CLASSES);

	const unsigned long model_id = strtoul(model->metadata.p_solution_id, NULL, 10);

	if (model_id > UINT16_MAX) {
		return -EINVAL;
	}

	const nrf_edgeai_dsp_feature_extraction_t *features_ctx =
		nrf_edgeai_dsp_features_ctx(model);

	info->model_id = (uint16_t)model_id;
	info->num_classes = SBM_OBSV_CLASSES;
	info->num_features = features_ctx != NULL ? features_ctx->overall_num : 0;
	info->version = model->metadata.version.combined;

	return 0;
}

int sbm_obsv_init(nrf_edgeai_t *model)
{
	nrf_edgeai_obsv_model_info_t info;
	int err;

	err = model_info_get(model, &info);
	if (err) {
		LOG_ERR("Failed to get model info (err %d)", err);
		return err;
	}

	err = nrf_edgeai_obsv_init(&ctx, &info);
	if (err) {
		LOG_ERR("Observability init failed (err %d)", err);
		return err;
	}

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_PRED_DIST)
	nrf_edgeai_obsv_metric_cpd_create(&cpd, cpd_buf, SBM_OBSV_CLASSES);
	err = nrf_edgeai_obsv_register(&ctx, &cpd, NULL);
	if (err) {
		LOG_ERR("Failed to register cpd metric (err %d)", err);
		return err;
	}
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_TRANSITION_MATRIX)
	nrf_edgeai_obsv_metric_tm_create(&tm, tm_buf, SBM_OBSV_CLASSES);
	err = nrf_edgeai_obsv_register(&ctx, &tm, NULL);
	if (err) {
		LOG_ERR("Failed to register tm metric (err %d)", err);
		return err;
	}
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MODEL_CERTAINTY_DESC)
	nrf_edgeai_obsv_metric_mcd_create(&mcd, mcd_buf, SBM_OBSV_CLASSES);
	err = nrf_edgeai_obsv_register(&ctx, &mcd, NULL);
	if (err) {
		LOG_ERR("Failed to register mcd metric (err %d)", err);
		return err;
	}
#endif

	err = nrf_edgeai_obsv_memfault_init(&ctx);
	if (err) {
		LOG_ERR("Memfault transport init failed (err %d)", err);
	}

	return err;
}

void sbm_obsv_update_probs(const float *probs)
{
	int err = nrf_edgeai_obsv_update_probs(&ctx, probs);

	if (err) {
		LOG_ERR("Failed to update probs (err %d)", err);
	}
}
