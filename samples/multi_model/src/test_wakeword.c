/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <nrf_edgeai/nrf_edgeai.h>

#include <nrf_edgeai_user_model.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#if defined(CONFIG_MODEL_OTA_AXON)
#include <model_ota/model_ota_axon_edgeai.h>
#include <zephyr/storage/flash_map.h>

MODEL_OTA_AXON_EDGEAI_LOAD_DECL(36711);

BUILD_ASSERT(FIXED_PARTITION_EXISTS(model_wakeword_storage),
	     "board devicetree is missing model_wakeword_storage - see boards/*.overlay");
#else
nrf_edgeai_t *nrf_edgeai_user_model_36711(void);
#endif

LOG_MODULE_REGISTER(multi_wakeword, LOG_LEVEL_INF);

#define WW_WINDOW_SIZE 160U

void run_wakeword_tests(void)
{
	static int16_t dummy_audio[WW_WINDOW_SIZE];

#if defined(CONFIG_MODEL_OTA_AXON)
	/* Model-only OTA: the compiled Axon model is loaded from its flash partition (XIP) at
	 * runtime and wired into the app-compiled nrf_edgeai_t wrapper.
	 */
	nrf_edgeai_t *model = nrf_edgeai_load_user_model_36711(
		PARTITION_ID(model_wakeword_storage),
		(const uint8_t *)PARTITION_ADDRESS(model_wakeword_storage));

	if (model == NULL) {
		LOG_WRN("No valid wakeword model image in model_wakeword_storage - skipping "
			"(flash wakeword_model_partition.hex)");
		return;
	}
#else
	nrf_edgeai_t *model = nrf_edgeai_user_model_36711();

	__ASSERT_NO_MSG(model != NULL);
#endif

	__ASSERT_NO_MSG(nrf_edgeai_input_window_size(model) == WW_WINDOW_SIZE);
	__ASSERT_NO_MSG(nrf_edgeai_uniq_inputs_num(model) == 1U);

	nrf_edgeai_err_t err = nrf_edgeai_init(model);
	__ASSERT(err == NRF_EDGEAI_ERR_SUCCESS, "err %d", err);

	err = nrf_edgeai_feed_inputs(model, dummy_audio, WW_WINDOW_SIZE);
	__ASSERT(err == NRF_EDGEAI_ERR_SUCCESS, "err %d", err);

	err = nrf_edgeai_run_inference(model);
	__ASSERT(err == NRF_EDGEAI_ERR_INPROGRESS, "err %d", err);

	err = nrf_edgeai_feed_inputs(model, dummy_audio, WW_WINDOW_SIZE);
	__ASSERT(err == NRF_EDGEAI_ERR_SUCCESS, "err %d", err);

	err = nrf_edgeai_run_inference(model);
	__ASSERT(err == NRF_EDGEAI_ERR_INPROGRESS, "err %d", err);

	err = nrf_edgeai_feed_inputs(model, dummy_audio, WW_WINDOW_SIZE);
	__ASSERT(err == NRF_EDGEAI_ERR_SUCCESS, "err %d", err);

	err = nrf_edgeai_run_inference(model);
	__ASSERT(err == NRF_EDGEAI_ERR_SUCCESS, "err %d", err);

	const uint16_t predicted_class = model->decoded_output.classif.predicted_class;
	const float probability =
		model->decoded_output.classif.probabilities.p_f32[predicted_class];

	LOG_INF("Wakeword inference on dummy PCM: class %u, probability %f",
		predicted_class, (double)probability);
}
