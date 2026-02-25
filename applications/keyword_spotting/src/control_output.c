/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "control_output.h"

#include <stdalign.h>
#include <stddef.h>
#include <stdio.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/atomic.h>

LOG_MODULE_REGISTER(control_output);

struct k_work control_output_work;

K_MSGQ_DEFINE(control_msg_queue, sizeof(struct control_message), 10,
	      alignof(struct control_message));

static const char waiting_ww_msg[] = "Waiting for wakeword\r\n";
static const char detected_ww_msg[] = "Wakeword detected\r\n";

static const struct device *uart_dev = DEVICE_DT_GET(DT_CHOSEN(ncs_control_output_uart));
static atomic_t uart_busy;

static void uart_cb(const struct device *dev, struct uart_event *evt, void *user_data)
{
	int err;

	if (evt->type == UART_TX_DONE) {
		atomic_set(&uart_busy, false);

		if (k_msgq_num_used_get(&control_msg_queue) == 0) {
			return;
		}

		err = k_work_submit(&control_output_work);
		if (err < 0) {
			LOG_ERR("Failed to submit work (err %d)", err);
		}
	}
}

static void control_output_work_handler(struct k_work *work)
{
	int err;
	struct control_message message_item;

	const char *buffer = NULL;
	size_t buffer_len = 0;

	if (atomic_get(&uart_busy)) {
		return;
	}

	err = k_msgq_peek(&control_msg_queue, &message_item);
	if (err) {
		return;
	}

	atomic_set(&uart_busy, true);

	switch (message_item.type) {
	case CONTROL_MESSAGE_WAITING_WW:
		buffer_len = sizeof(waiting_ww_msg) - 1;
		buffer = waiting_ww_msg;
		break;
	case CONTROL_MESSAGE_WW_DETECTED:
		buffer_len = sizeof(detected_ww_msg) - 1;
		buffer = detected_ww_msg;
		break;
	default:
		atomic_set(&uart_busy, false);
		LOG_ERR("Unhandled case");
		return;
	}

	err = uart_tx(uart_dev, buffer, buffer_len, 0);
	if (err) {
		atomic_set(&uart_busy, false);
		LOG_ERR("Failed to transmit data (err %d)", err);
		return;
	}

	k_msgq_get(&control_msg_queue, &message_item, K_NO_WAIT);
}

int control_output_init(void)
{
	int err;

	if (!device_is_ready(uart_dev)) {
		LOG_ERR("Device is not ready");
		return -ENODEV;
	}

	err = uart_callback_set(uart_dev, uart_cb, NULL);
	if (err) {
		LOG_ERR("Failed to set UART callback (err %d)", err);
	}

	k_work_init(&control_output_work, control_output_work_handler);

	return err;
}

void print_control_output(const struct control_message message)
{
	int err;

	err = k_msgq_put(&control_msg_queue, &message, K_NO_WAIT);
	if (err) {
		LOG_ERR("Failed to put message in queue (err %d)", err);
		return;
	}

	err = k_work_submit(&control_output_work);
	if (err < 0) {
		LOG_ERR("Failed to submit work (err %d)", err);
	}
}
