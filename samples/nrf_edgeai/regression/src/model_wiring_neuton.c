/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "model_wiring.h"

#if defined(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA)
#include <model_ota/model_pkg.h>
#endif

#include <nrf_edgeai/nrf_edgeai_platform.h>
#include <nrf_edgeai/rt/private/nrf_edgeai_interfaces.h>

#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

LOG_MODULE_REGISTER(model_wiring_neuton, LOG_LEVEL_INF);

/* Air-quality regression inputs: 9 sensor/environmental features. The feature extraction /
 * input-scaling pipeline is part of the application, not the updatable model package: per the
 * model-OTA constraints this PoC is based on, it cannot be changed by a model-only update.
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

#define MODEL_OUTPUTS_NUM 1

#if defined(CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA)

#define EDGEAI_SOLUTION_ID_STR          "36025"
#define EDGEAI_RUNTIME_VERSION_COMBINED 0x00000202

/*
 * The model itself is *not* compiled in: these are only populated (by model_pkg_load_neuton())
 * once a valid package has been read from the model_storage flash partition.
 */
static nrf_edgeai_model_neuton_t model_instance_;
static flt32_t model_neurons_[CONFIG_MODEL_OTA_MAX_NEURONS];
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

	.model.type = NRF_EDGEAI_MODEL_NEUTON,
	.model.task = NRF_EDGEAI_TASK_REGRESSION,
	.model.instance.p_void = &model_instance_,
	.model.output.memory.p_void = model_outputs_,
	.model.output.num = MODEL_OUTPUTS_NUM,
	.model.uses_as_input.all = 1,

	.interfaces.input_init = nrf_edgeai_input_init_no_window,
	.interfaces.feed_inputs = nrf_edgeai_input_feed_no_window,
	.interfaces.process_features = nrf_edgeai_process_features_scale_vector_f32_f32,
	.interfaces.init_inference = nrf_edgeai_init_inference_neuton,
	.interfaces.run_inference = nrf_edgeai_run_inference_neuton_f32,
	.interfaces.propagate_outputs = nrf_edgeai_output_propagate_neuton_f32,
	.interfaces.decode_outputs = nrf_edgeai_output_decode_regression_f32,

	/* Filled in by model_ota_load() once a model package has been validated. */
	.decoded_output.regression.meta.p_scale_min = NULL,
	.decoded_output.regression.meta.p_scale_max = NULL,
};

nrf_edgeai_t *model_ota_load(void)
{
	struct model_pkg_neuton_info info;
	int err;

	err = model_pkg_load_neuton(&model_instance_, model_neurons_, ARRAY_SIZE(model_neurons_),
				     &info);
	if (err != MODEL_PKG_OK) {
		LOG_ERR("No usable model in model_storage (err %d)", err);
		return NULL;
	}

	nrf_edgeai_.decoded_output.regression.meta.p_scale_min = info.output_scale_min;
	nrf_edgeai_.decoded_output.regression.meta.p_scale_max = info.output_scale_max;

	LOG_INF("Active model: '%s' version 0x%08x (%u neurons, %u weights)", info.name,
		info.version, info.neurons_num, info.weights_num);

	return &nrf_edgeai_;
}

#else /* !CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA: compiled-in model (solution 90508), no
       * model_storage partition or runtime loading - this sample's original pre-model-OTA
       * behavior.
       */

#define EDGEAI_SOLUTION_ID_STR          "90508"
#define EDGEAI_RUNTIME_VERSION_COMBINED 0x00000202

#define MODEL_NEURONS_NUM 2
#define MODEL_WEIGHTS_NUM 4

static const flt32_t MODEL_WEIGHTS[MODEL_WEIGHTS_NUM] = {0.5644180f, 0.1098376f, 0.9692999f,
							  -0.4583102f};
static const uint16_t MODEL_NEURONS_LINKS[MODEL_WEIGHTS_NUM] = {2, 9, 0, 9};
static const uint16_t MODEL_NEURON_INTERNAL_LINKS_NUM[MODEL_NEURONS_NUM] = {0, 3};
static const uint16_t MODEL_NEURON_EXTERNAL_LINKS_NUM[MODEL_NEURONS_NUM] = {2, 4};
static const flt32_t MODEL_NEURON_ACTIVATION_WEIGHTS[MODEL_NEURONS_NUM] = {20.0000000f,
									   10.0783634f};
static const uint8_t MODEL_NEURON_ACTIVATION_TYPE_MASK[] = {0x1};
static const uint16_t MODEL_OUTPUT_NEURONS_INDICES[MODEL_OUTPUTS_NUM] = {1};
static const flt32_t MODEL_OUTPUT_SCALE_MIN[MODEL_OUTPUTS_NUM] = {0.2000000f};
static const flt32_t MODEL_OUTPUT_SCALE_MAX[MODEL_OUTPUTS_NUM] = {63.7000008f};

static flt32_t model_neurons_[MODEL_NEURONS_NUM];
static flt32_t model_outputs_[MODEL_OUTPUTS_NUM];

static const nrf_edgeai_model_neuton_t model_instance_ = {
	.meta.p_neuron_internal_links_num = MODEL_NEURON_INTERNAL_LINKS_NUM,
	.meta.p_neuron_external_links_num = MODEL_NEURON_EXTERNAL_LINKS_NUM,
	.meta.p_output_neurons_indices = MODEL_OUTPUT_NEURONS_INDICES,
	.meta.p_neuron_links = MODEL_NEURONS_LINKS,
	.meta.p_neuron_act_type_mask = MODEL_NEURON_ACTIVATION_TYPE_MASK,
	.meta.outputs_num = MODEL_OUTPUTS_NUM,
	.meta.neurons_num = MODEL_NEURONS_NUM,
	.meta.weights_num = MODEL_WEIGHTS_NUM,
	.params.f32 =
		{
			.p_weights = MODEL_WEIGHTS,
			.p_act_weights = MODEL_NEURON_ACTIVATION_WEIGHTS,
			.p_neurons = model_neurons_,
		},
};

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

	.model.type = NRF_EDGEAI_MODEL_NEUTON,
	.model.task = NRF_EDGEAI_TASK_REGRESSION,
	.model.instance.p_void = &model_instance_,
	.model.output.memory.p_void = model_outputs_,
	.model.output.num = MODEL_OUTPUTS_NUM,
	.model.uses_as_input.all = 1,

	.interfaces.input_init = nrf_edgeai_input_init_no_window,
	.interfaces.feed_inputs = nrf_edgeai_input_feed_no_window,
	.interfaces.process_features = nrf_edgeai_process_features_scale_vector_f32_f32,
	.interfaces.init_inference = nrf_edgeai_init_inference_neuton,
	.interfaces.run_inference = nrf_edgeai_run_inference_neuton_f32,
	.interfaces.propagate_outputs = nrf_edgeai_output_propagate_neuton_f32,
	.interfaces.decode_outputs = nrf_edgeai_output_decode_regression_f32,

	.decoded_output.regression.meta.p_scale_min = MODEL_OUTPUT_SCALE_MIN,
	.decoded_output.regression.meta.p_scale_max = MODEL_OUTPUT_SCALE_MAX,
};

nrf_edgeai_t *model_ota_load(void)
{
	return &nrf_edgeai_;
}

#endif /* CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA */
