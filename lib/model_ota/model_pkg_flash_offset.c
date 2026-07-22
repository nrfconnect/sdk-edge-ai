/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <model_ota/model_pkg.h>

#include <zephyr/storage/flash_map.h>

/** MCUboot image header magic (@ref IMAGE_MAGIC in bootutil/image.h). */
#define MCUBOOT_IMAGE_MAGIC 0x96f3b83dU

/** MCUboot fixed image header size (@ref IMAGE_HEADER_SIZE in bootutil/image.h). */
#define MCUBOOT_IMAGE_HEADER_SIZE 32U

size_t model_pkg_partition_content_offset(const struct flash_area *fa)
{
	uint32_t magic;
	int rc;

	rc = flash_area_read(fa, 0, &magic, sizeof(magic));
	if (rc != 0) {
		return 0;
	}

	if (magic == MCUBOOT_IMAGE_MAGIC) {
		return MCUBOOT_IMAGE_HEADER_SIZE;
	}

	return 0;
}
