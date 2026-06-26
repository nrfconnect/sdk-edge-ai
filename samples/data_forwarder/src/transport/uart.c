/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "transport.h"

#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(transport, CONFIG_LOG_DEFAULT_LEVEL);

static const struct device *uart_dev = DEVICE_DT_GET(DT_CHOSEN(ncs_data_forwarder_uart));
static K_SEM_DEFINE(uart_tx_done, 0, 1);

static void uart_event_handler(const struct device *dev, struct uart_event *evt, void *user_data)
{
	ARG_UNUSED(dev);
	ARG_UNUSED(user_data);

	if ((evt->type == UART_TX_DONE) || (evt->type == UART_TX_ABORTED)) {
		k_sem_give(&uart_tx_done);
	}
}

static int uart_send_cb(const uint8_t *buf, size_t len, void *ctx)
{
	int err;

	ARG_UNUSED(ctx);

	if ((buf == NULL) || (len == 0U)) {
		return -EINVAL;
	}

	err = uart_tx(uart_dev, buf, len, SYS_FOREVER_MS);
	if (err) {
		return err;
	}

	err = k_sem_take(&uart_tx_done, K_FOREVER);
	if (err) {
		return err;
	}

	return 0;
}

int transport_init(struct proto_transport *out_transport)
{
	int err;

	if (out_transport == NULL) {
		return -EINVAL;
	}

	if (!device_is_ready(uart_dev)) {
		return -ENODEV;
	}

	err = uart_callback_set(uart_dev, uart_event_handler, NULL);
	if (err) {
		LOG_ERR("UART callback set failed (err %d)", err);
		return err;
	}

	out_transport->send = uart_send_cb;
	out_transport->ctx = NULL;
	out_transport->has_message_boundaries = false;

	LOG_INF("UART transport ready");
	return 0;
}
