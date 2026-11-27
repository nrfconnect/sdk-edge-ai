/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MODEL_OTA_MODEL_OTA_SLOT_H_
#define MODEL_OTA_MODEL_OTA_SLOT_H_

/*
 * Helpers shared by the model slot loaders. Each one is compiled once per model slot, with
 * MODEL_OTA_SLOT_TARGET set to the slot ID, MODEL_OTA_PARTITION_NODELABEL to the devicetree
 * node label of its model partition and MODEL_OTA_IMAGE_OFFSET to the size of the bootloader
 * header preceding the model image.
 */

#include <stdint.h>

#include <zephyr/devicetree.h>
#include <zephyr/sys/util.h>
#include <zephyr/toolchain.h>

#include "model_image.h"

#ifndef MODEL_OTA_SLOT_TARGET
#error "MODEL_OTA_SLOT_TARGET must be defined"
#endif

#ifndef MODEL_OTA_PARTITION_NODELABEL
#error "MODEL_OTA_PARTITION_NODELABEL must be defined"
#endif

#ifndef MODEL_OTA_IMAGE_OFFSET
#error "MODEL_OTA_IMAGE_OFFSET must be defined"
#endif

/* DT_NODELABEL() pastes its argument, so expand the node label macro first. */
#define MODEL_OTA_DT_NODELABEL(label) DT_NODELABEL(label)

#define MODEL_OTA_PARTITION_NODE MODEL_OTA_DT_NODELABEL(MODEL_OTA_PARTITION_NODELABEL)

/** Memory-mapped base address of the model partition. */
#define MODEL_OTA_PARTITION_ADDR                                                                   \
	((const uint8_t *)DT_MAPPED_PARTITION_ADDR(MODEL_OTA_PARTITION_NODE))

/** Size of the model partition, in bytes. */
#define MODEL_OTA_PARTITION_SIZE ((size_t)DT_REG_SIZE(MODEL_OTA_PARTITION_NODE))

/** Memory-mapped address of the model image header. */
#define MODEL_OTA_IMAGE_ADDR (MODEL_OTA_PARTITION_ADDR + MODEL_OTA_IMAGE_OFFSET)

/** Bytes available to the model image, from @ref MODEL_OTA_IMAGE_ADDR to the partition end. */
#define MODEL_OTA_IMAGE_MAX_SIZE (MODEL_OTA_PARTITION_SIZE - MODEL_OTA_IMAGE_OFFSET)

BUILD_ASSERT(DT_MAPPED_PARTITION_EXISTS(MODEL_OTA_PARTITION_NODE),
	     STRINGIFY(MODEL_OTA_PARTITION_NODELABEL) " must be a zephyr,mapped-partition node");
BUILD_ASSERT(MODEL_OTA_PARTITION_SIZE >= MODEL_OTA_IMAGE_OFFSET + sizeof(struct model_image_header),
	     STRINGIFY(MODEL_OTA_PARTITION_NODELABEL) " cannot hold a model image header");

/**
 * Publish the contract hash of the model slot loader as the local object model_ota_contract_hash.
 *
 * The hash folds in sizeof() of runtime structs, so only the compiler can evaluate it. The host
 * tools read it back from the model slot loader object, where it is kept even if the firmware link
 * drops it.
 */
#define MODEL_OTA_CONTRACT_HASH_EXPORT(hash)                                                       \
	__used static const uint32_t model_ota_contract_hash = (hash)

#endif /* MODEL_OTA_MODEL_OTA_SLOT_H_ */
