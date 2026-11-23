/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <string.h>

#include <zephyr/ztest.h>

#include "model_image_common.h"

/* Model image with a payload after the header, so image_size can exceed the header. */
struct common_image {
	struct model_image_header hdr;
	uint8_t payload[16];
};

struct model_image_common_fixture {
	struct common_image image;
	uint8_t raw[sizeof(struct common_image)];
	const uint8_t *image_addr;
	size_t max_image_size;
	const struct model_image_header *hdr;
};

static void *common_setup(void)
{
	static struct model_image_common_fixture fixture;

	return &fixture;
}

static void common_before(void *f)
{
	struct model_image_common_fixture *fixture = f;

	memset(fixture, 0, sizeof(*fixture));
	fixture->image.hdr = (struct model_image_header){
		.magic = MODEL_IMAGE_MAGIC,
		.format_version = MODEL_IMAGE_FORMAT_VERSION,
		.backend = MODEL_IMAGE_BACKEND_AXON,
		.image_size = sizeof(fixture->image),
	};
	fixture->image_addr = (const uint8_t *)&fixture->image;
	fixture->max_image_size = sizeof(fixture->image);
}

static enum model_image_result validate(struct model_image_common_fixture *fixture)
{
	return model_image_validate_header(fixture->image_addr, fixture->max_image_size,
					   &fixture->hdr);
}

ZTEST_SUITE(model_image_common, NULL, common_setup, common_before, NULL, NULL);

/** @brief A well-formed header that exactly fills the space is accepted. */
ZTEST_F(model_image_common, test_valid_header)
{
	enum model_image_result rc;

	rc = validate(fixture);

	zassert_equal(rc, MODEL_IMAGE_OK);
	zassert_equal_ptr(fixture->hdr, &fixture->image.hdr);
}

/** @brief An image smaller than the space after it in the partition is accepted. */
ZTEST_F(model_image_common, test_valid_header_in_larger_partition)
{
	enum model_image_result rc;

	fixture->max_image_size = sizeof(fixture->image) + 4096;

	rc = validate(fixture);

	zassert_equal(rc, MODEL_IMAGE_OK);
	zassert_equal_ptr(fixture->hdr, &fixture->image.hdr);
}

/** @brief A corrupted magic is rejected. */
ZTEST_F(model_image_common, test_bad_magic)
{
	enum model_image_result rc;

	fixture->image.hdr.magic ^= 1u;

	rc = validate(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_BAD_MAGIC);
	zassert_is_null(fixture->hdr);
}

/** @brief An erased flash partition is reported as holding no image. */
ZTEST_F(model_image_common, test_erased_partition)
{
	enum model_image_result rc;

	memset(fixture->raw, 0xff, sizeof(fixture->raw));
	fixture->image_addr = fixture->raw;

	rc = validate(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_BAD_MAGIC);
	zassert_is_null(fixture->hdr);
}

/** @brief A zeroed partition is reported as holding no image. */
ZTEST_F(model_image_common, test_zeroed_partition)
{
	enum model_image_result rc;

	fixture->image_addr = fixture->raw;

	rc = validate(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_BAD_MAGIC);
	zassert_is_null(fixture->hdr);
}

/** @brief An image in an unknown format version is rejected. */
ZTEST_F(model_image_common, test_bad_format_version)
{
	enum model_image_result rc;

	fixture->image.hdr.format_version = MODEL_IMAGE_FORMAT_VERSION + 1;

	rc = validate(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_BAD_FORMAT_VERSION);
	zassert_is_null(fixture->hdr);
}

/** @brief An image size smaller than the header is rejected. */
ZTEST_F(model_image_common, test_image_smaller_than_header)
{
	enum model_image_result rc;

	fixture->image.hdr.image_size = sizeof(fixture->image.hdr) - 1;

	rc = validate(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_BAD_SIZE);
	zassert_is_null(fixture->hdr);
}

/** @brief An image size larger than the space in the partition is rejected. */
ZTEST_F(model_image_common, test_image_larger_than_partition)
{
	enum model_image_result rc;

	fixture->max_image_size = sizeof(fixture->image) - 1;

	rc = validate(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_BAD_SIZE);
	zassert_is_null(fixture->hdr);
}
