/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stdalign.h>
#include <stddef.h>
#include <stdio.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/atomic.h>

#include "control_output.h"

LOG_MODULE_REGISTER(control_output);

struct k_work control_output_work;

K_MSGQ_DEFINE(control_msg_queue, sizeof(struct control_message), 10,
	      alignof(struct control_message));

static const char *const messages[] = {
	[CONTROL_MESSAGE_WAITING_WW] = "Waiting for wakeword\r\n",
	[CONTROL_MESSAGE_WW_DETECTED] = "Wakeword detected\r\n",

	[CONTROL_MESSAGE_WAITING_KW] = "Waiting for keywords\r\n",
	[CONTROL_MESSAGE_KW_SPOTTED] = "Keyword spotted: %s\r\n",
	[CONTROL_MESSAGE_TIMEOUT_KWS] = "Keyword spotting window timeout\r\n",
};

BUILD_ASSERT(CONTROL_MESSAGE_COUNT == ARRAY_SIZE(messages),
	     "Mismatch between control_message_type and messages size");

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

	static char output_buffer[40];
	const char *buffer = NULL;
	size_t buffer_len = 0;

	if (atomic_get(&uart_busy)) {
		return;
	}

	/* Peek message to not lose it if uart_tx fails. */
	err = k_msgq_peek(&control_msg_queue, &message_item);
	if (err) {
		return;
	}

	atomic_set(&uart_busy, true);

	switch (message_item.type) {
	case CONTROL_MESSAGE_WAITING_WW:
	case CONTROL_MESSAGE_WW_DETECTED:
	case CONTROL_MESSAGE_WAITING_KW:
	case CONTROL_MESSAGE_TIMEOUT_KWS:
		buffer = messages[message_item.type];
		buffer_len = strlen(buffer);
		break;

	case CONTROL_MESSAGE_KW_SPOTTED:
		buffer = output_buffer;
		buffer_len = snprintf(output_buffer, sizeof(output_buffer),
				      messages[message_item.type], message_item.name);
		__ASSERT(buffer_len >= 0, "Error in snprintf call (%d)", buffer_len);
		__ASSERT(buffer_len < sizeof(output_buffer), "Output buffer is too small");

		break;
	default:
		atomic_set(&uart_busy, false);
		k_msgq_get(&control_msg_queue, &message_item, K_NO_WAIT);
		LOG_WRN("Unhandled case");
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
