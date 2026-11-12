/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/*
 * Direct Memfault upload over HTTPS for observability, based on the connection
 * handling of the nRF Connect SDK Memfault sample (samples/debug/memfault).
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/net/conn_mgr_connectivity.h>
#include <zephyr/net/conn_mgr_monitor.h>
#include <zephyr/net/net_mgmt.h>

#include <memfault/ports/zephyr/periodic_upload.h>

#include "memfault_http_obsv.h"

LOG_MODULE_REGISTER(memfault_http_obsv);

#define L4_EVENT_MASK (NET_EVENT_L4_CONNECTED | NET_EVENT_L4_DISCONNECTED)

static struct net_mgmt_event_callback l4_callback;

static void l4_event_handler(struct net_mgmt_event_callback *cb, uint64_t event,
			     struct net_if *iface)
{
	ARG_UNUSED(cb);
	ARG_UNUSED(iface);

	if (event == NET_EVENT_L4_CONNECTED) {
		LOG_INF("Network connected, Memfault upload enabled");
		memfault_zephyr_port_periodic_upload_enable(true);
	} else if (event == NET_EVENT_L4_DISCONNECTED) {
		LOG_INF("Network disconnected, Memfault upload paused");
		memfault_zephyr_port_periodic_upload_enable(false);
	}
}

int init_app_memfault_http(void)
{
	int err;

	net_mgmt_init_event_callback(&l4_callback, l4_event_handler, L4_EVENT_MASK);
	net_mgmt_add_event_callback(&l4_callback);

	err = conn_mgr_all_if_up(true);
	if (err) {
		LOG_ERR("Failed to bring up the network interfaces (err %d)", err);
		return err;
	}

	err = conn_mgr_all_if_connect(true);
	if (err) {
		LOG_ERR("Failed to connect the network interfaces (err %d)", err);
		return err;
	}

	return 0;
}
