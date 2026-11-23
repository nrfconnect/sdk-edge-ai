/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "model_obsv.h"

#include <errno.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_memfault.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

LOG_MODULE_REGISTER(model_obsv, LOG_LEVEL_INF);

#define OBSV_MAX_CLASSES CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES

#define OBSV_WORDS(bytes) DIV_ROUND_UP(bytes, sizeof(uint32_t))

static nrf_edgeai_obsv_ctx_t ctx;

static nrf_edgeai_obsv_metric_t cpd;
static uint32_t cpd_buf[OBSV_WORDS(NRF_EDGEAI_OBSV_CPD_STORAGE_BYTES(OBSV_MAX_CLASSES))];
static nrf_edgeai_obsv_metric_t tm;
static uint32_t tm_buf[OBSV_WORDS(NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(OBSV_MAX_CLASSES))];
static nrf_edgeai_obsv_metric_t mcd;
static uint32_t mcd_buf[OBSV_WORDS(NRF_EDGEAI_OBSV_MCD_STORAGE_BYTES(OBSV_MAX_CLASSES))];

int model_obsv_init(const nrf_edgeai_obsv_model_info_t *info)
{
	int err;

	__ASSERT_NO_MSG(info != NULL);

	if ((info->num_classes == 0) || (info->num_classes > OBSV_MAX_CLASSES)) {
		LOG_ERR("Model has %u classes, CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES is %u",
			info->num_classes, OBSV_MAX_CLASSES);
		return -EINVAL;
	}

	err = nrf_edgeai_obsv_init(&ctx, info);
	if (err) {
		LOG_ERR("Observability init failed (err %d)", err);
		return err;
	}

	/* The dump and CDR payload follow the registration order */
	nrf_edgeai_obsv_metric_cpd_create(&cpd, cpd_buf, info->num_classes);
	err = nrf_edgeai_obsv_register(&ctx, &cpd, NULL);
	if (err) {
		LOG_ERR("Failed to register cpd metric (err %d)", err);
		return err;
	}

	nrf_edgeai_obsv_metric_tm_create(&tm, tm_buf, info->num_classes);
	err = nrf_edgeai_obsv_register(&ctx, &tm, NULL);
	if (err) {
		LOG_ERR("Failed to register tm metric (err %d)", err);
		return err;
	}

	nrf_edgeai_obsv_metric_mcd_create(&mcd, mcd_buf, info->num_classes);
	err = nrf_edgeai_obsv_register(&ctx, &mcd, NULL);
	if (err) {
		LOG_ERR("Failed to register mcd metric (err %d)", err);
		return err;
	}

	err = nrf_edgeai_obsv_memfault_init(&ctx);
	if (err) {
		LOG_ERR("Memfault transport init failed (err %d)", err);
		return err;
	}

	LOG_INF("Observability initialized: model id %u, %u classes", (unsigned int)info->model_id,
		info->num_classes);

	return 0;
}

int model_obsv_update_probs(const float *probs)
{
	int err = nrf_edgeai_obsv_update_probs(&ctx, probs);

	if (err) {
		LOG_ERR("Failed to update probs (err %d)", err);
	}
	return err;
}

uint32_t model_obsv_num_inferences(void)
{
	uint32_t num;

	/* Same lock as update and collect, so the count is not read mid-reset */
	k_mutex_lock(&ctx.lock, K_FOREVER);
	num = ctx.state.num_inferences;
	k_mutex_unlock(&ctx.lock);

	return num;
}
