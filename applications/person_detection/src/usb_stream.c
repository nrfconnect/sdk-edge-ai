/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * USB CDC ACM streaming: send RGB565 frames and detection boxes over USB HS.
 *
 * Uses the Zephyr device_next USB stack with CDC ACM class.
 *
 * The CDC ACM fifo API (uart_fifo_fill, uart_irq_tx_ready, etc.) must only
 * be called from the CDC ACM workqueue context.  Therefore we use an
 * application-level ring buffer: the main thread stages complete messages
 * into it, then calls uart_irq_tx_enable() (safe from any context) to
 * schedule the CDC ACM IRQ callback.  The callback drains the app ring
 * buffer via uart_fifo_fill() from the correct context.
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/usb/usbd.h>
#include <zephyr/sys/atomic.h>
#include <zephyr/sys/crc.h>
#include <zephyr/sys/ring_buffer.h>
#include <zephyr/logging/log.h>

#include <string.h>

#include "usb_stream.h"

LOG_MODULE_REGISTER(usb_stream, LOG_LEVEL_INF);

/* Provided by sample_usbd_init.c (included via common.cmake). */
struct usbd_context *sample_usbd_init_device(usbd_msg_cb_t msg_cb);

static const struct device *cdc_dev = DEVICE_DT_GET_ONE(zephyr_cdc_acm_uart);
static atomic_t dtr_set;
static struct usbd_context *usbd_ctx;

/*
 * Application-level TX ring buffer.  The main thread writes complete
 * messages here; the CDC ACM IRQ callback drains it via uart_fifo_fill().
 */
#define APP_TX_BUF_SIZE 8192
static uint8_t app_tx_buffer[APP_TX_BUF_SIZE];
static struct ring_buf app_tx_rb;

/* Signalled by the IRQ callback when it has drained some data. */
static K_SEM_DEFINE(tx_drain_sem, 0, 1);

/* ------------------------------------------------------------------ */
/* CDC ACM IRQ callback — runs in CDC ACM workqueue context           */
/* ------------------------------------------------------------------ */

static void cdc_acm_irq_handler(const struct device *dev, void *user_data)
{
	ARG_UNUSED(user_data);

	while (uart_irq_update(dev) && uart_irq_is_pending(dev)) {
		if (uart_irq_tx_ready(dev)) {
			uint8_t *data_ptr;
			uint32_t claimed;

			claimed = ring_buf_get_claim(&app_tx_rb, &data_ptr, 512);
			if (claimed == 0) {
				uart_irq_tx_disable(dev);
				break;
			}

			int sent = uart_fifo_fill(dev, data_ptr, claimed);

			if (sent <= 0) {
				ring_buf_get_finish(&app_tx_rb, 0);
				break;
			}
			ring_buf_get_finish(&app_tx_rb, sent);

			/* Signal the main thread that buffer space is available. */
			k_sem_give(&tx_drain_sem);
		}
	}
}

/* ------------------------------------------------------------------ */
/* Message staging helpers — run from the main thread                 */
/* ------------------------------------------------------------------ */

static void usbd_msg_cb(struct usbd_context *const ctx, const struct usbd_msg *msg)
{
	if (usbd_can_detect_vbus(ctx)) {
		if (msg->type == USBD_MSG_VBUS_READY) {
			(void)usbd_enable(ctx);
		}
		if (msg->type == USBD_MSG_VBUS_REMOVED) {
			(void)usbd_disable(ctx);
		}
	}

	if (msg->type == USBD_MSG_CDC_ACM_CONTROL_LINE_STATE) {
		uint32_t dtr = 0U;

		(void)uart_line_ctrl_get(msg->dev, UART_LINE_CTRL_DTR, &dtr);
		atomic_set(&dtr_set, dtr ? 1 : 0);
		if (dtr) {
			LOG_INF("USB host connected (DTR)");
		} else {
			LOG_INF("USB host disconnected (DTR cleared)");
		}
	}
}

/**
 * Stage data into the app TX ring buffer.  Blocks until all data is
 * staged — the CDC ACM IRQ callback drains the buffer in the background.
 * Returns false if DTR was deasserted mid-transfer (host disconnected).
 */
static bool tx_stage(const uint8_t *data, size_t len)
{
	size_t done = 0;

	while (done < len) {
		if (!atomic_get(&dtr_set)) {
			return false;
		}

		uint32_t n = ring_buf_put(&app_tx_rb, data + done, len - done);

		done += n;
		if (done < len) {
			/* Kick CDC ACM TX and block until callback drains some data. */
			uart_irq_tx_enable(cdc_dev);
			k_sem_take(&tx_drain_sem, K_MSEC(10));
		}
	}

	/* Ensure TX path is running after staging new data. */
	uart_irq_tx_enable(cdc_dev);
	return true;
}

static void le16_put(uint8_t *dst, uint16_t v)
{
	dst[0] = (uint8_t)v;
	dst[1] = (uint8_t)(v >> 8);
}

static void le32_put(uint8_t *dst, uint32_t v)
{
	dst[0] = (uint8_t)v;
	dst[1] = (uint8_t)(v >> 8);
	dst[2] = (uint8_t)(v >> 16);
	dst[3] = (uint8_t)(v >> 24);
}

static void float_put(uint8_t *dst, float v)
{
	uint32_t u;

	memcpy(&u, &v, sizeof(u));
	le32_put(dst, u);
}

int usb_stream_init(void)
{
	if (!device_is_ready(cdc_dev)) {
		LOG_ERR("CDC ACM device not ready");
		return -ENODEV;
	}

	ring_buf_init(&app_tx_rb, sizeof(app_tx_buffer), app_tx_buffer);

	uart_irq_callback_set(cdc_dev, cdc_acm_irq_handler);

	usbd_ctx = sample_usbd_init_device(usbd_msg_cb);
	if (usbd_ctx == NULL) {
		LOG_ERR("USB device init failed");
		return -EIO;
	}

	if (!usbd_can_detect_vbus(usbd_ctx)) {
		int err = usbd_enable(usbd_ctx);

		if (err != 0) {
			LOG_ERR("usbd_enable failed: %d", err);
			return err;
		}
	}

	LOG_INF("USB CDC ACM stream ready (waiting for host DTR)");
	return 0;
}

/* ------------------------------------------------------------------ */
/* Streaming frame API                                                */
/* ------------------------------------------------------------------ */

static uint32_t frame_crc;
static uint32_t frame_tx_id;
static bool frame_active;

void usb_stream_frame_begin(uint32_t frame_id, uint16_t w, uint16_t h,
			    uint32_t payload_len)
{
	if (!atomic_get(&dtr_set)) {
		frame_active = false;
		return;
	}

	frame_crc = 0U;
	frame_tx_id = frame_id;

	/* 18-byte header: magic(4) + ver(1) + type(1) + w(2) + h(2) +
	 * payload_len(4) + frame_id(4).  CRC follows payload as trailer.
	 */
	uint8_t hdr[18];

	le32_put(&hdr[0], USB_STREAM_MAGIC);
	hdr[4] = USB_STREAM_VERSION;
	hdr[5] = USB_STREAM_TYPE_FRAME;
	le16_put(&hdr[6], w);
	le16_put(&hdr[8], h);
	le32_put(&hdr[10], payload_len);
	le32_put(&hdr[14], frame_id);

	frame_active = tx_stage(hdr, sizeof(hdr));
}

void usb_stream_frame_chunk(const uint8_t *data, size_t len)
{
	if (!frame_active) {
		return;
	}

	frame_crc = crc32_ieee_update(frame_crc, data, len);

	if (!tx_stage(data, len)) {
		frame_active = false;
	}
}

void usb_stream_frame_end(void)
{
	if (!frame_active) {
		return;
	}

	uint8_t crc_buf[4];

	le32_put(crc_buf, frame_crc);
	tx_stage(crc_buf, sizeof(crc_buf));

	frame_active = false;
}

void usb_stream_send_detections(uint32_t frame_id, uint16_t model_w, uint16_t model_h,
				uint16_t pad_left, uint16_t pad_top,
				const struct person_det_box *boxes, int count)
{
	if (!atomic_get(&dtr_set)) {
		return;
	}

	if (count < 0) {
		count = 0;
	}
	if (count > 255) {
		count = 255;
	}

	/*
	 * Layout: magic(4) + hdr(15) + boxes(count*21) + crc32(4)
	 * CRC covers hdr + boxes (everything after magic, before crc32).
	 */

	/* Build 15-byte header (after magic). */
	uint8_t hdr[15];

	hdr[0] = USB_STREAM_VERSION;
	hdr[1] = USB_STREAM_TYPE_DETECT;
	le32_put(&hdr[2], frame_id);
	le16_put(&hdr[6], model_w);
	le16_put(&hdr[8], model_h);
	le16_put(&hdr[10], pad_left);
	le16_put(&hdr[12], pad_top);
	hdr[14] = (uint8_t)count;

	uint32_t crc = crc32_ieee(hdr, sizeof(hdr));

	/* Serialize boxes into a contiguous buffer for CRC + TX in one pass.
	 * Stack cost: at most 21 * count bytes. The caller typically passes
	 * a small array (person_det uses MAX_BOXES_LOG = 8 → 168 bytes).
	 */
	uint8_t box_data[count * 21];

	for (int i = 0; i < count; i++) {
		uint8_t *bp = &box_data[i * 21];

		float_put(&bp[0], boxes[i].x1);
		float_put(&bp[4], boxes[i].y1);
		float_put(&bp[8], boxes[i].x2);
		float_put(&bp[12], boxes[i].y2);
		float_put(&bp[16], boxes[i].score);
		bp[20] = (uint8_t)boxes[i].head;
	}

	const size_t box_bytes = (size_t)count * 21;

	if (box_bytes > 0) {
		crc = crc32_ieee_update(crc, box_data, box_bytes);
	}

	/* Transmit: magic + hdr + boxes + crc32 */
	uint8_t magic_buf[4];

	le32_put(magic_buf, USB_STREAM_MAGIC);

	if (!tx_stage(magic_buf, sizeof(magic_buf))) {
		return;
	}
	if (!tx_stage(hdr, sizeof(hdr))) {
		return;
	}
	if (box_bytes > 0 && !tx_stage(box_data, box_bytes)) {
		return;
	}

	uint8_t crc_buf[4];

	le32_put(crc_buf, crc);
	tx_stage(crc_buf, sizeof(crc_buf));
}
