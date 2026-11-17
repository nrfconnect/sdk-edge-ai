/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stdbool.h>
#include <string.h>

#include <zephyr/sys/util.h>

#include <memfault/core/event_storage.h>
#include <memfault/metrics/metrics.h>
#include <memfault/metrics/serializer.h>

#define MOCK_HEARTBEAT_WORST_CASE_SIZE 100U

static uint32_t values[18];
static uint32_t write_count;
static size_t storage_free;

void memfault_ncs_metrics_collect_data(void)
{
}

void memfault_mock_reset(void)
{
	memset(values, 0, sizeof(values));
	write_count = 0U;
	storage_free = MOCK_HEARTBEAT_WORST_CASE_SIZE;
}

void memfault_mock_set_storage_full(bool full)
{
	storage_free = full ? MOCK_HEARTBEAT_WORST_CASE_SIZE - 1U : MOCK_HEARTBEAT_WORST_CASE_SIZE;
}

size_t memfault_event_storage_bytes_free(void)
{
	return storage_free;
}

size_t memfault_metrics_heartbeat_compute_worst_case_storage_size(void)
{
	return MOCK_HEARTBEAT_WORST_CASE_SIZE;
}

uint32_t memfault_mock_value(int key)
{
	return values[key];
}

uint32_t memfault_mock_write_count(void)
{
	return write_count;
}

int memfault_metrics_heartbeat_set_unsigned(MemfaultMetricId key, uint32_t value)
{
	if ((key._impl < 0) || (key._impl >= ARRAY_SIZE(values))) {
		return -1;
	}

	values[key._impl] = value;
	write_count++;
	return 0;
}
