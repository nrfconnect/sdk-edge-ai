/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "model_obsv.h"

#include <errno.h>
#include <stdlib.h>

#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_memfault.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

LOG_MODULE_REGISTER(model_obsv, LOG_LEVEL_INF);

/* Metric buffers are sized for the configured maximum; the actual class count
 * is read from the model at init.
 */
#define OBSV_MAX_CLASSES CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES

#define OBSV_WORDS(bytes) DIV_ROUND_UP(bytes, sizeof(uint32_t))

static nrf_edgeai_obsv_ctx_t ctx;

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_PRED_DIST)
static nrf_edgeai_obsv_metric_t cpd;
static uint32_t cpd_buf[OBSV_WORDS(NRF_EDGEAI_OBSV_CPD_STORAGE_BYTES(OBSV_MAX_CLASSES))];
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_TRANSITION_MATRIX)
static nrf_edgeai_obsv_metric_t tm;
static uint32_t tm_buf[OBSV_WORDS(NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(OBSV_MAX_CLASSES))];
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MODEL_CERTAINTY_DESC)
static nrf_edgeai_obsv_metric_t mcd;
static uint32_t mcd_buf[OBSV_WORDS(NRF_EDGEAI_OBSV_MCD_STORAGE_BYTES(OBSV_MAX_CLASSES))];
#endif

static int model_info_get(nrf_edgeai_t *model, nrf_edgeai_obsv_model_info_t *info)
{
	errno = 0;
	const unsigned long model_id = strtoul(nrf_edgeai_solution_id_str(model), NULL, 10);
	const uint16_t num_classes = nrf_edgeai_model_outputs_num(model);

	if ((errno == ERANGE) || ((uint64_t)model_id > UINT32_MAX)) {
		LOG_ERR("Solution id %s does not fit model_id", nrf_edgeai_solution_id_str(model));
		return -EINVAL;
	}

	if ((num_classes == 0) || (num_classes > OBSV_MAX_CLASSES)) {
		LOG_ERR("Model has %u classes, CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES is %u",
			num_classes, OBSV_MAX_CLASSES);
		return -EINVAL;
	}

	info->model_id = (uint32_t)model_id;
	info->num_classes = num_classes;
	/* No FEATURES-source metrics: the feature stream is unused */
	info->num_features = 0;
	info->version = model->metadata.version.combined;

	return 0;
}

int model_obsv_init(nrf_edgeai_t *model)
{
	nrf_edgeai_obsv_model_info_t info;
	int err;

	__ASSERT_NO_MSG(model != NULL);

	err = model_info_get(model, &info);
	if (err) {
		return err;
	}

	err = nrf_edgeai_obsv_init(&ctx, &info);
	if (err) {
		LOG_ERR("Observability init failed (err %d)", err);
		return err;
	}

	/* The dump and CDR payload follow the registration order */
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_PRED_DIST)
	nrf_edgeai_obsv_metric_cpd_create(&cpd, cpd_buf, info.num_classes);
	err = nrf_edgeai_obsv_register(&ctx, &cpd, NULL);
	if (err) {
		LOG_ERR("Failed to register cpd metric (err %d)", err);
		return err;
	}
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_TRANSITION_MATRIX)
	nrf_edgeai_obsv_metric_tm_create(&tm, tm_buf, info.num_classes);
	err = nrf_edgeai_obsv_register(&ctx, &tm, NULL);
	if (err) {
		LOG_ERR("Failed to register tm metric (err %d)", err);
		return err;
	}
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MODEL_CERTAINTY_DESC)
	nrf_edgeai_obsv_metric_mcd_create(&mcd, mcd_buf, info.num_classes);
	err = nrf_edgeai_obsv_register(&ctx, &mcd, NULL);
	if (err) {
		LOG_ERR("Failed to register mcd metric (err %d)", err);
		return err;
	}
#endif

	err = nrf_edgeai_obsv_memfault_init(&ctx);
	if (err) {
		LOG_ERR("Memfault transport init failed (err %d)", err);
		return err;
	}

	LOG_INF("Observability initialized: model id %u, %u classes", (unsigned int)info.model_id,
		info.num_classes);

	return 0;
}

int model_obsv_update_probs(const flt32_t *probs)
{
	int err = nrf_edgeai_obsv_update_probs(&ctx, probs);

	if (err) {
		LOG_ERR("Failed to update probs (err %d)", err);
	}
	return err;
}
