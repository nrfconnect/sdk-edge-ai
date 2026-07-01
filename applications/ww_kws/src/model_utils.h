/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MODEL_UTILS_H_
#define MODEL_UTILS_H_

/**
 * @{
 * @ingroup ww_kws
 */

#include <errno.h>
#include <stdlib.h>

#include <zephyr/sys/__assert.h>

#include <nrf_edgeai/nrf_edgeai.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_core.h>

/**
 * @brief Fill observability model metadata from an nRF Edge AI model instance.
 *
 * @param model       Initialized model instance.
 * @param num_classes Expected number of output classes.
 * @param info        Output metadata for nrf_edgeai_obsv_init().
 * @retval 0 Success.
 * @retval -EINVAL Solution ID is out of range for uint16_t model_id.
 */
static inline int obsv_model_info_from_model(nrf_edgeai_t *model, uint16_t num_classes,
					     nrf_edgeai_obsv_model_info_t *info)
{
	__ASSERT_NO_MSG(model != NULL);
	__ASSERT_NO_MSG(info != NULL);
	__ASSERT_NO_MSG(nrf_edgeai_model_outputs_num(model) == num_classes);

	const unsigned long model_id = strtoul(model->metadata.p_solution_id, NULL, 10);

	if (model_id > UINT16_MAX) {
		return -EINVAL;
	}

	info->model_id = (uint16_t)model_id;
	info->num_classes = num_classes;
	info->version = model->metadata.version.combined;

	return 0;
}

/**
 * @}
 */

#endif /* MODEL_UTILS_H_ */
