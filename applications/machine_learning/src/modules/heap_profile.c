/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#define MODULE heap_profile

#include "ml_result_event.h"

#include <zephyr/kernel.h>

#include <app_event_manager.h>
#include <caf/events/module_state_event.h>
#include <malloc.h>

extern struct sys_heap _system_heap;

static void print_heap_stats(void)
{
	malloc_stats();

	struct sys_memory_stats stats;
	sys_heap_runtime_stats_get(&_system_heap, &stats);
	printk("zephyr system heap: allocated %zu, free %zu, max allocated %zu, heap size %u\n\n",
	       stats.allocated_bytes, stats.free_bytes, stats.max_allocated_bytes,
	       CONFIG_HEAP_MEM_POOL_SIZE);
}

static bool app_event_handler(const struct app_event_header *aeh)
{
	if (is_ml_result_event(aeh)) {
		print_heap_stats();
		return false;
	}

	__ASSERT_NO_MSG(false);

	return false;
}

APP_EVENT_LISTENER(MODULE, app_event_handler);
APP_EVENT_SUBSCRIBE(MODULE, ml_result_event);
