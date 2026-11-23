/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <string.h>

#include <zephyr/fff.h>
#include <zephyr/ztest.h>

#include "model_image_axon.h"

#define CONTRACT_HASH	    0x1234abcdu
#define PERSISTENT_VARS_CAP 64u
#define PACKED_OUTPUT_CAP   128u
#define INPUT_VECTOR_CAP    2u
#define HASH_A		    0x11111111u
#define HASH_B		    0x22222222u

DEFINE_FFF_GLOBALS;

FAKE_VALUE_FUNC(nrf_axon_result_e, nrf_axon_nn_model_validate,
		const nrf_axon_nn_compiled_model_s *);

static const nrf_axon_nn_compiled_model_s wrong_model;

static int fw_sym_a;
static int fw_sym_b;

const struct model_image_binding_entry model_ota_axon_binding_table[] = {
	{HASH_A, &fw_sym_a},
	{HASH_B, &fw_sym_b},
};
const uint32_t model_ota_axon_binding_count = ARRAY_SIZE(model_ota_axon_binding_table);

/* Model image with real pointers into itself, as a linked image would have. */
struct axon_image {
	struct model_image_header hdr;
	char name[12];
	struct model_image_binding_entry binding[2];
	nrf_axon_nn_compiled_model_s model;
	const int8_t *input_vectors[INPUT_VECTOR_CAP];
};

struct model_image_axon_fixture {
	struct axon_image image;
	struct model_image_axon_expect expect;
	const nrf_axon_nn_compiled_model_s *model;
};

static void *axon_setup(void)
{
	static struct model_image_axon_fixture fixture;

	return &fixture;
}

static void axon_before(void *f)
{
	struct model_image_axon_fixture *fixture = f;
	struct axon_image *image = &fixture->image;

	memset(fixture, 0, sizeof(*fixture));
	strcpy(image->name, "test_model");
	memcpy(image->binding, model_ota_axon_binding_table, sizeof(image->binding));
	image->model.input_vector_list = image->input_vectors;
	image->model.input_cnt = INPUT_VECTOR_CAP;

	image->hdr = (struct model_image_header){
		.magic = MODEL_IMAGE_MAGIC,
		.format_version = MODEL_IMAGE_FORMAT_VERSION,
		.backend = MODEL_IMAGE_BACKEND_AXON,
		.image_size = sizeof(*image),
		.model_version = 7,
		.contract_hash = CONTRACT_HASH,
		.name = image->name,
		.axon = {
			.model = &image->model,
			.packed_output_bytes = PACKED_OUTPUT_CAP,
			.persistent_vars_required = PERSISTENT_VARS_CAP,
			.binding = image->binding,
			.binding_count = ARRAY_SIZE(image->binding),
		},
	};

	fixture->expect = (struct model_image_axon_expect){
		.contract_hash = CONTRACT_HASH,
		.persistent_vars_cap = PERSISTENT_VARS_CAP,
		.packed_output_cap = PACKED_OUTPUT_CAP,
		.input_vector_cap = INPUT_VECTOR_CAP,
	};

	/* Stale output, so tests see whether a load overwrites it. */
	fixture->model = &wrong_model;

	RESET_FAKE(nrf_axon_nn_model_validate);
	nrf_axon_nn_model_validate_fake.return_val = NRF_AXON_RESULT_SUCCESS;
}

static enum model_image_result load(struct model_image_axon_fixture *fixture)
{
	return model_image_load_axon((const uint8_t *)&fixture->image, sizeof(fixture->image),
				     &fixture->expect, &fixture->model);
}

ZTEST_SUITE(model_image_axon, NULL, axon_setup, axon_before, NULL, NULL);

/** @brief A valid Axon image loads and its model is validated by the Axon driver. */
ZTEST_F(model_image_axon, test_valid_image)
{
	enum model_image_result rc;

	rc = load(fixture);

	zassert_equal(rc, MODEL_IMAGE_OK);
	zassert_equal_ptr(fixture->model, &fixture->image.model);
	zassert_equal(nrf_axon_nn_model_validate_fake.call_count, 1);
	zassert_equal_ptr(nrf_axon_nn_model_validate_fake.arg0_val, &fixture->image.model);
}

/** @brief An image that binds to no firmware symbols loads. */
ZTEST_F(model_image_axon, test_valid_image_without_binding)
{
	enum model_image_result rc;

	fixture->image.hdr.axon.binding = NULL;
	fixture->image.hdr.axon.binding_count = 0;

	rc = load(fixture);

	zassert_equal(rc, MODEL_IMAGE_OK);
	zassert_equal_ptr(fixture->model, &fixture->image.model);
}

/** @brief Header checks run first, so an invalid header never reaches the Axon checks. */
ZTEST_F(model_image_axon, test_invalid_header)
{
	enum model_image_result rc;

	fixture->image.hdr.magic ^= 1u;

	rc = load(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_BAD_MAGIC);
	zassert_equal(nrf_axon_nn_model_validate_fake.call_count, 0);
}

/** @brief An image for another backend is rejected. */
ZTEST_F(model_image_axon, test_backend_none)
{
	enum model_image_result rc;

	fixture->image.hdr.backend = MODEL_IMAGE_BACKEND_NONE;

	rc = load(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_BAD_BACKEND);
}

/** @brief An image built against another firmware contract is rejected. */
ZTEST_F(model_image_axon, test_contract_mismatch)
{
	enum model_image_result rc;

	fixture->image.hdr.contract_hash = CONTRACT_HASH + 1;

	rc = load(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_CONTRACT_MISMATCH);
}

/** @brief A model needing more persistent vars than the slot provides is rejected. */
ZTEST_F(model_image_axon, test_persistent_vars_over_cap)
{
	enum model_image_result rc;

	fixture->image.hdr.axon.persistent_vars_required = PERSISTENT_VARS_CAP + 1;

	rc = load(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_PERSISTENT_VARS_TOO_MANY);
}

/** @brief A model needing a larger packed-output buffer than the slot provides is rejected. */
ZTEST_F(model_image_axon, test_packed_output_over_cap)
{
	enum model_image_result rc;

	fixture->image.hdr.axon.packed_output_bytes = PACKED_OUTPUT_CAP + 1;

	rc = load(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_PACKED_OUTPUT_TOO_LARGE);
}

/** @brief A binding count without a binding table is rejected. */
ZTEST_F(model_image_axon, test_binding_null_with_count)
{
	enum model_image_result rc;

	fixture->image.hdr.axon.binding = NULL;

	rc = load(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_BINDING_MISMATCH);
}

/** @brief A binding to a symbol the firmware does not provide is rejected. */
ZTEST_F(model_image_axon, test_binding_hash_missing)
{
	enum model_image_result rc;

	fixture->image.binding[1].name_hash = 0x33333333u;

	rc = load(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_BINDING_MISMATCH);
}

/** @brief A binding to a firmware symbol at another address is rejected. */
ZTEST_F(model_image_axon, test_binding_address_mismatch)
{
	enum model_image_result rc;

	fixture->image.binding[0].address = &fw_sym_b;

	rc = load(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_BINDING_MISMATCH);
}

/** @brief An image without an Axon model is rejected before the Axon driver sees it. */
ZTEST_F(model_image_axon, test_model_null)
{
	enum model_image_result rc;

	fixture->image.hdr.axon.model = NULL;

	rc = load(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_AXON_VALIDATE);
	zassert_equal(nrf_axon_nn_model_validate_fake.call_count, 0);
}

/** @brief A model the Axon driver rejects is rejected. */
ZTEST_F(model_image_axon, test_axon_validate_failure)
{
	enum model_image_result rc;

	nrf_axon_nn_model_validate_fake.return_val = NRF_AXON_RESULT_FAILURE;

	rc = load(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_AXON_VALIDATE);
	zassert_equal(nrf_axon_nn_model_validate_fake.call_count, 1);
}

/** @brief A model with more inputs than the slot provides is rejected. */
ZTEST_F(model_image_axon, test_input_count_over_cap)
{
	enum model_image_result rc;

	fixture->image.model.input_cnt = INPUT_VECTOR_CAP + 1;

	rc = load(fixture);

	zassert_equal(rc, MODEL_IMAGE_ERR_AXON_VALIDATE);
}
