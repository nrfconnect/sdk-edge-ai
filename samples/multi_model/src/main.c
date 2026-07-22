/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "test_models.h"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(multi_model, LOG_LEVEL_INF);

int main(void)
{
	LOG_INF("Running local multi-model test (Neuton + Edge AI + Axon models)");

	run_anomaly_tests();
	LOG_INF("Anomaly tests passed");

	run_classification_tests();
	LOG_INF("Classification tests passed");

	run_regression_tests();
	LOG_INF("Regression tests passed");

	run_wakeword_tests();
	LOG_INF("Wakeword tests passed");

	run_person_det_tests();
	LOG_INF("Person detection tests passed");

	run_okay_nordic_tests();
	LOG_INF("okay_nordic tests passed");

	while (1) {
		LOG_INF("All multi-model tests completed");
		k_sleep(K_MSEC(5000));
	}

	return 0;
}
