/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * USB CDC ACM streaming: RGB565 frames + detection boxes over USB HS.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#include "person_det_postprocess.h"

#define USB_STREAM_MAGIC       0x12345AA5U
#define USB_STREAM_VERSION     1
#define USB_STREAM_TYPE_FRAME  0x01
#define USB_STREAM_TYPE_DETECT 0x02

int usb_stream_init(void);

void usb_stream_send_frame(uint32_t frame_id, uint16_t w, uint16_t h,
			   const uint8_t *rgb565, size_t len);

void usb_stream_send_detections(uint32_t frame_id, uint16_t model_w, uint16_t model_h,
				uint16_t pad_left, uint16_t pad_top,
				const struct person_det_box *boxes, int count);
