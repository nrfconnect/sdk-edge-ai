/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MODEL_OTA_PARTITION_H_
#define MODEL_OTA_PARTITION_H_

/**
 * @file
 * @brief Devicetree helpers for model OTA mapped partitions and MCUboot image indices.
 *
 * Model storage partitions must be ``zephyr,mapped-partition`` nodes. The MCUboot
 * updateable image index is read from the ``nordic,mcuboot-image`` bootchain node
 * whose primary slot (``partitions[0]``) references the partition.
 */

#include <zephyr/devicetree.h>
#include <zephyr/storage/flash_map.h>
#include <zephyr/sys/util.h>

#if DT_HAS_COMPAT_STATUS_OKAY(nordic_mcuboot)

#define MODEL_OTA_MCUBOOT_IMAGES_NODE DT_CHILD(DT_COMPAT_GET_ANY_STATUS_OKAY(nordic_mcuboot), images)

#define MODEL_OTA_IMAGE_IDX_TERM(node_id, part_node)                                              \
	+ COND_CODE_1(DT_SAME_NODE(DT_PHANDLE_BY_IDX(node_id, partitions, 0), part_node),         \
		      (DT_PROP(node_id, image_index)), (0))

#define MODEL_OTA_IMAGE_IDX_MATCHES(node_id, part_node)                                           \
	+ COND_CODE_1(DT_SAME_NODE(DT_PHANDLE_BY_IDX(node_id, partitions, 0), part_node),         \
		      (1), (0))

/** MCUboot image index for @p label from the bootchain (compile-time). */
#define MODEL_OTA_IMAGE_INDEX(label)                                                              \
	(0 DT_FOREACH_CHILD_VARGS(MODEL_OTA_MCUBOOT_IMAGES_NODE, MODEL_OTA_IMAGE_IDX_TERM,       \
				  DT_NODELABEL(label)))

/** Number of bootchain images whose primary slot is @p label (expect 1). */
#define MODEL_OTA_IMAGE_INDEX_MATCH_COUNT(label)                                                  \
	(0 DT_FOREACH_CHILD_VARGS(MODEL_OTA_MCUBOOT_IMAGES_NODE, MODEL_OTA_IMAGE_IDX_MATCHES,     \
				  DT_NODELABEL(label)))

#else

#error "Model OTA requires a nordic,mcuboot bootchain node in devicetree"

#endif /* DT_HAS_COMPAT_STATUS_OKAY(nordic_mcuboot) */

/** Partition + bootchain checks for OTA-wired models. */
#define MODEL_OTA_PARTITION_ASSERT(label)                                                         \
	BUILD_ASSERT(DT_MAPPED_PARTITION_EXISTS(DT_NODELABEL(label)),                             \
		     #label " must be a zephyr,mapped-partition");                               \
	BUILD_ASSERT(MODEL_OTA_IMAGE_INDEX_MATCH_COUNT(label) == 1,                               \
		     #label " must be the primary slot of exactly one mcuboot-image")

#endif /* MODEL_OTA_PARTITION_H_ */
