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

/**
 * Streaming frame API — overlaps USB TX with camera capture.
 *
 * Frame wire format (CRC is a trailer, not in the header):
 *   [magic(4) ver(1) type(1) w(2) h(2) plen(4) fid(4)]  — 18 B header
 *   [payload: RGB565 BE]                                  — plen bytes
 *   [crc32(payload)]                                      — 4 B trailer
 */
void usb_stream_frame_begin(uint32_t frame_id, uint16_t w, uint16_t h,
			    uint32_t payload_len);
void usb_stream_frame_chunk(const uint8_t *data, size_t len);
void usb_stream_frame_end(void);

void usb_stream_send_detections(uint32_t frame_id, uint16_t model_w, uint16_t model_h,
				uint16_t pad_left, uint16_t pad_top,
				const struct person_det_box *boxes, int count);
