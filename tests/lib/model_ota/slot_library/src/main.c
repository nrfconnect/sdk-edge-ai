/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stddef.h>

#include <zephyr/devicetree.h>
#include <zephyr/fff.h>
#include <zephyr/ztest.h>

#include <model_ota/model_ota_axon.h>

#include "model_contract.h"
#include "model_image_axon.h"

/* Capacities the first slot loader is built with, to check what it passes on. */
#include "../slots/first/model_ota_axon_model_config.h"

#define FIRST_ADDR  DT_MAPPED_PARTITION_ADDR(DT_NODELABEL(model_first_storage))
#define FIRST_SIZE  DT_REG_SIZE(DT_NODELABEL(model_first_storage))
#define SECOND_ADDR DT_MAPPED_PARTITION_ADDR(DT_NODELABEL(model_second_storage))

DEFINE_FFF_GLOBALS;

/* CONFIG_MODEL_OTA_AXON is disabled, so the slot loaders link against this fake. */
FAKE_VALUE_FUNC(enum model_image_result, model_image_load_axon, const uint8_t *, size_t,
		const struct model_image_axon_expect *, const nrf_axon_nn_compiled_model_s **);

MODEL_OTA_AXON_LOAD_DECL(first);
MODEL_OTA_AXON_LOAD_DECL(second);

static const nrf_axon_nn_compiled_model_s *fake_model = (const void *)0x1234;
static enum model_image_result load_axon_ok(const uint8_t *image_addr, size_t max_image_size,
					    const struct model_image_axon_expect *expect,
					    const nrf_axon_nn_compiled_model_s **out_model)
{
	ARG_UNUSED(image_addr);
	ARG_UNUSED(max_image_size);
	ARG_UNUSED(expect);

	*out_model = fake_model;
	return MODEL_IMAGE_OK;
}

static void reset_fakes(void *f)
{
	ARG_UNUSED(f);

	RESET_FAKE(model_image_load_axon);
	model_image_load_axon_fake.custom_fake = load_axon_ok;
}

ZTEST_SUITE(model_ota_slot_library, NULL, NULL, reset_fakes, NULL, NULL);

/** @brief A slot loader passes the image in its partition and returns the loaded model. */
ZTEST(model_ota_slot_library, test_load_passes_image_in_partition)
{
	const nrf_axon_nn_compiled_model_s *model = NULL;
	enum model_image_result rc;

	rc = model_ota_load_axon_first(&model);

	zassert_equal(rc, MODEL_IMAGE_OK);
	zassert_equal_ptr(model, fake_model);
	zassert_equal(model_image_load_axon_fake.call_count, 1);
	zassert_equal_ptr(model_image_load_axon_fake.arg0_val,
			  (const uint8_t *)FIRST_ADDR + MODEL_OTA_IMAGE_OFFSET);
	zassert_equal(model_image_load_axon_fake.arg1_val, FIRST_SIZE - MODEL_OTA_IMAGE_OFFSET);
	zassert_equal_ptr(model_image_load_axon_fake.arg3_val, &model);
}

/** @brief A slot loader expects its capacities and the contract hash of its partition. */
ZTEST(model_ota_slot_library, test_load_passes_slot_expectations)
{
	const nrf_axon_nn_compiled_model_s *model;

	(void)model_ota_load_axon_first(&model);

	const struct model_image_axon_expect *expect = model_image_load_axon_fake.arg2_val;

	zassert_not_null(expect);
	zassert_equal(expect->contract_hash, MODEL_OTA_CONTRACT_HASH_AXON(FIRST_ADDR));
	zassert_equal(expect->persistent_vars_cap, MODEL_OTA_AXON_PERSISTENT_VARS_CAP);
	zassert_equal(expect->packed_output_cap, MODEL_OTA_AXON_PACKED_OUTPUT_BYTES);
	zassert_equal(expect->input_vector_cap, MODEL_OTA_AXON_INPUT_VECTOR_CAP);
}

/** @brief Each slot loader is built with its own partition. */
ZTEST(model_ota_slot_library, test_slots_are_independent)
{
	const nrf_axon_nn_compiled_model_s *model;

	(void)model_ota_load_axon_first(&model);
	(void)model_ota_load_axon_second(&model);

	zassert_equal(model_image_load_axon_fake.call_count, 2);

	const struct model_image_axon_expect *first = model_image_load_axon_fake.arg2_history[0];
	const struct model_image_axon_expect *second = model_image_load_axon_fake.arg2_history[1];

	zassert_equal_ptr(model_image_load_axon_fake.arg0_history[1],
			  (const uint8_t *)SECOND_ADDR + MODEL_OTA_IMAGE_OFFSET);
	zassert_not_equal(first->contract_hash, second->contract_hash);
}

/** @brief A slot loader returns the image loader's error unchanged. */
ZTEST(model_ota_slot_library, test_load_returns_image_loader_error)
{
	const nrf_axon_nn_compiled_model_s *model = fake_model;
	enum model_image_result rc;

	model_image_load_axon_fake.custom_fake = NULL;
	model_image_load_axon_fake.return_val = MODEL_IMAGE_ERR_CONTRACT_MISMATCH;

	rc = model_ota_load_axon_first(&model);

	zassert_equal(rc, MODEL_IMAGE_ERR_CONTRACT_MISMATCH);
	zassert_is_null(model);
}

/** @brief A slot loader rejects a missing model output without calling the image loader. */
ZTEST(model_ota_slot_library, test_load_null_out_model)
{
	enum model_image_result rc;

	rc = model_ota_load_axon_first(NULL);

	zassert_equal(rc, MODEL_IMAGE_ERR_INVALID_ARG);
	zassert_equal(model_image_load_axon_fake.call_count, 0);
}
