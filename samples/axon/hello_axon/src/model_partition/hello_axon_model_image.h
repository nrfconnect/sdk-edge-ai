/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#include <zephyr/sys/util.h>

#include "hello_axon_model_image_defs.h"

BUILD_ASSERT(sizeof(struct hello_axon_model_image_header) == 32);
BUILD_ASSERT(sizeof(struct hello_axon_model_image_metadata) == 52);
BUILD_ASSERT(sizeof(struct hello_axon_model_const_layout) == HELLO_AXON_MODEL_CONST_SIZE);
