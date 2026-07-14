/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "model_wiring.h"

#if defined(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA)
#include <model_ota/model_pkg.h>
#endif

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_nn_infer.h>
#include <nrf_edgeai/nrf_edgeai_platform.h>
#include <nrf_edgeai/rt/private/nrf_edgeai_interfaces.h>

#if !defined(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA)
#include <assert.h> /* for the static_assert()s in the generated header below */

/* Needs the axon platform/driver headers above (nrf_axon_interlayer_buffer,
 * NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE, ...) to already be visible.
 */
#include "nrf_edgeai_generated/Axon/nrf_edgeai_user_model_axon.h"
#endif

#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(model_wiring_axon, LOG_LEVEL_INF);

#define EDGEAI_SOLUTION_ID_STR          "36025"
#define EDGEAI_RUNTIME_VERSION_COMBINED 0x00000202

/* Same 9 sensor/environmental features and scaling as the Neuton backend - the input pipeline
 * is shared between backends and is part of the application, not the updatable model package.
 */
#define INPUT_UNIQ_FEATURES_NUM 9

static const flt32_t INPUT_FEATURES_SCALE_MIN[INPUT_UNIQ_FEATURES_NUM] = {
	0.1000000f,  647.0000000f, 387.0000000f, 322.0000000f, 559.0000000f,
	225.0000000f, -1.3000000f, 9.1999998f,   0.1847000f,
};
static const flt32_t INPUT_FEATURES_SCALE_MAX[INPUT_UNIQ_FEATURES_NUM] = {
	11.8999996f,  2040.0000000f, 2214.0000000f, 2683.0000000f, 2775.0000000f,
	2523.0000000f, 44.5999985f,  88.6999969f,   2.1805999f,
};

/* Axon's own output dequantization already yields the final regression value, so no further
 * rescaling is needed here - this identity [0,1] range matches the compiled-in Axon model this
 * backend replaces.
 */
static const flt32_t MODEL_OUTPUT_SCALE_MIN[] = {0.0000000f};
static const flt32_t MODEL_OUTPUT_SCALE_MAX[] = {1.0000000f};

#define MODEL_OUTPUTS_NUM 1

#if defined(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA)
/*
 * The model itself is *not* compiled in: this is only populated (by model_pkg_load_axon()) once
 * a valid package has been read from the model_storage flash partition.
 */
static nrf_axon_nn_compiled_model_s model_instance_;
#define MODEL_INSTANCE (&model_instance_)
#else
/* Compiled directly into the application image - this sample's original pre-model-OTA
 * behavior. model_axon_user_instance_36025 is defined by the generated model header included
 * above.
 */
#define MODEL_INSTANCE (&model_axon_user_instance_36025)
#endif

static flt32_t model_outputs_[MODEL_OUTPUTS_NUM];

static nrf_edgeai_t nrf_edgeai_ = {
	.metadata.p_solution_id = EDGEAI_SOLUTION_ID_STR,
	.metadata.version.combined = EDGEAI_RUNTIME_VERSION_COMBINED,

	.input.p_used_for_lags_mask = NULL,
	.input.p_usage_mask = NULL,
	.input.type = NRF_EDGEAI_INPUT_F32,
	.input.unique_num = INPUT_UNIQ_FEATURES_NUM,
	.input.unique_num_used = INPUT_UNIQ_FEATURES_NUM,
	.input.unique_scales_num = INPUT_UNIQ_FEATURES_NUM,
	.input.window_size = 1,
	.input.window_shift = 0,
	.input.subwindow_num = 0,
	.input.window_memory.p_void = NULL,
	.input.p_window_ctx = NULL,
	.input.scale.f32 =
		{
			.p_min = INPUT_FEATURES_SCALE_MIN,
			.p_max = INPUT_FEATURES_SCALE_MAX,
		},

	.p_dsp = NULL,

	.model.type = NRF_EDGEAI_MODEL_AXON,
	.model.task = NRF_EDGEAI_TASK_REGRESSION,
	.model.instance.p_void = MODEL_INSTANCE,
	.model.output.memory.p_void = model_outputs_,
	.model.output.num = MODEL_OUTPUTS_NUM,
	.model.uses_as_input.all = 1,

	.interfaces.input_init = nrf_edgeai_input_init_no_window,
	.interfaces.feed_inputs = nrf_edgeai_input_feed_no_window,
	.interfaces.process_features = nrf_edgeai_process_features_scale_vector_f32_f32,
	.interfaces.init_inference = nrf_edgeai_init_inference_axon,
	.interfaces.run_inference = nrf_edgeai_run_inference_axon,
	.interfaces.propagate_outputs = nrf_edgeai_output_dequantize_axon_q8_f32,
	.interfaces.decode_outputs = nrf_edgeai_output_decode_regression_f32,

	.decoded_output.regression.meta.p_scale_min = MODEL_OUTPUT_SCALE_MIN,
	.decoded_output.regression.meta.p_scale_max = MODEL_OUTPUT_SCALE_MAX,
};

#if defined(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA)

nrf_edgeai_t *model_ota_load(void)
{
	struct model_pkg_axon_info info;
	int err;

	err = model_pkg_load_axon(&model_instance_, &info);
	if (err != MODEL_PKG_OK) {
		LOG_ERR("No usable model in model_storage (err %d)", err);
		return NULL;
	}

	LOG_INF("Active model: '%s' version 0x%08x (%u cmd words, %u B const)", info.name,
		info.version, info.cmd_buffer_len, info.model_const_size);

	return &nrf_edgeai_;
}

#else

nrf_edgeai_t *model_ota_load(void)
{
	return &nrf_edgeai_;
}

#endif /* CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA */
