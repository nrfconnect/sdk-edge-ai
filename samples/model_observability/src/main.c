/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

#include "ble_mds.h"
#include "model_obsv.h"
#include "scenarios.h"
#include "ux_state_manager.h"

LOG_MODULE_REGISTER(main);

/* There is no real model: the identifier and version are application-defined */
#define SAMPLE_MODEL_ID      (777U)
#define SAMPLE_MODEL_VERSION (1U)

/* How often the sample checks whether auto collect has staged the fed scenario */
#define COLLECT_POLL_MS (1000U)

/*
 * Auto collect is rescheduled one interval after every collect, and the next
 * scenario starts within COLLECT_POLL_MS of it. Feeding a whole scenario must
 * end before the next collect, so that a CDR never splits a scenario.
 */
BUILD_ASSERT((SCENARIO_MAX_VECTORS * CONFIG_APP_FEED_INTERVAL_MS + COLLECT_POLL_MS) <
		     (CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT_INTERVAL_SEC * 1000ULL),
	     "A scenario must be fed within one auto collect interval");

static uint16_t predicted_class(const float *probs)
{
	uint16_t top = 0;

	for (uint16_t i = 1; i < ACTIVITY_CLASS_COUNT; i++) {
		if (probs[i] > probs[top]) {
			top = i;
		}
	}
	return top;
}

static void scenario_probs_vectors_feed(uint32_t num, const struct scenario *scenario)
{
	uint32_t class_count[ACTIVITY_CLASS_COUNT] = {0};

	LOG_INF("Scenario %u: feeding %u probability vectors", num,
		(unsigned int)scenario->num_vectors);

	for (size_t i = 0; i < scenario->num_vectors; i++) {
		const float *probs = scenario->vectors[i];

		(void)model_obsv_update_probs(probs);

		class_count[predicted_class(probs)]++;

		k_sleep(K_MSEC(CONFIG_APP_FEED_INTERVAL_MS));
	}

	LOG_INF("Scenario %u fed: %s %u, %s %u, %s %u, %s %u", num,
		activity_class_name[ACTIVITY_CLASS_WALKING], class_count[ACTIVITY_CLASS_WALKING],
		activity_class_name[ACTIVITY_CLASS_SITTING], class_count[ACTIVITY_CLASS_SITTING],
		activity_class_name[ACTIVITY_CLASS_RUNNING], class_count[ACTIVITY_CLASS_RUNNING],
		activity_class_name[ACTIVITY_CLASS_LAYING], class_count[ACTIVITY_CLASS_LAYING]);
}

/*
 * The collect that stages a CDR resets the observability context, so the fed
 * vector count drops to 0. A collect is refused while the previous CDR is not
 * read by the gateway yet, so this also waits until the previous scenario is sent.
 */
static void scenario_wait_collected(uint32_t num)
{
	LOG_INF("Scenario %u: waiting for auto collect", num);

	while (model_obsv_num_inferences() != 0U) {
		k_sleep(K_MSEC(COLLECT_POLL_MS));
	}

	LOG_INF("Scenario %u staged as CDR", num);
}

int main(void)
{
	int err;
	const nrf_edgeai_obsv_model_info_t model_info = {
		.model_id = SAMPLE_MODEL_ID,
		.num_classes = ACTIVITY_CLASS_COUNT,
		/* No FEATURES-source metrics: the feature stream is unused */
		.num_features = 0,
		.version = SAMPLE_MODEL_VERSION,
	};

	LOG_INF("nRF Edge AI Model Observability Sample");

	err = uxsm_init();
	__ASSERT(err == 0, "Failed to initialize UX state manager");

	err = model_obsv_init(&model_info);
	__ASSERT(err == 0, "Failed to initialize model observability (err %d)", err);

	err = ble_mds_init(uxsm_set_ble_connected);
	__ASSERT(err == 0, "Failed to initialize BLE MDS service (err %d)", err);

	/* Repeat the scenarios, each one in its own CDR */
	for (;;) {
		for (uint32_t i = 0; i < SCENARIO_NUM; i++) {
			scenario_probs_vectors_feed(i + 1U, &scenarios[i]);
			scenario_wait_collected(i + 1U);
		}
	}

	return 0;
}
