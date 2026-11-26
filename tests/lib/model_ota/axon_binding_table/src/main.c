/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stdint.h>

#include <zephyr/ztest.h>

#include "model_image.h"

/* Generated for the slot, the expected content of the table. */
#include "model_ota_axon_model_config.h"

extern const struct model_image_binding_entry model_ota_axon_binding_table[];
extern const uint32_t model_ota_axon_binding_count;

/*
 * Only names and hashes, never the symbols themselves: the binding table must be what keeps them
 * in the firmware.
 */
struct keep_symbol {
	const char *name;
	uint32_t name_hash;
};

#define KEEP_SYMBOL(symbol) {#symbol, MODEL_OTA_AXON_SYM_HASH(symbol)},
static const struct keep_symbol keep_symbols[] = {MODEL_OTA_AXON_KEEP_REFS(KEEP_SYMBOL)};

static const struct model_image_binding_entry *find_binding(uint32_t name_hash)
{
	for (uint32_t i = 0; i < model_ota_axon_binding_count; i++) {
		if (model_ota_axon_binding_table[i].name_hash == name_hash) {
			return &model_ota_axon_binding_table[i];
		}
	}

	return NULL;
}

ZTEST_SUITE(model_ota_axon_binding_table, NULL, NULL, NULL, NULL, NULL);

/** @brief With one slot, the binding table holds exactly the symbols that slot binds to. */
ZTEST(model_ota_axon_binding_table, test_binding_count_matches_slot)
{
	zassert_equal(model_ota_axon_binding_count, MODEL_OTA_AXON_KEEP_SYMBOL_COUNT);
}

/**
 * @brief A symbol of the slot is in the binding table, with a firmware address.
 *
 * The keep symbols include the slot storage defined by the app stub.
 */
ZTEST_P(model_ota_axon_binding_table, test_slot_symbol_is_bound)
{
	const struct keep_symbol *symbol = ZTEST_GET_PARAM_PTR(struct keep_symbol);
	const struct model_image_binding_entry *entry = find_binding(symbol->name_hash);

	zassert_not_null(entry, "%s missing from the binding table", symbol->name);
	zassert_not_null(entry->address, "%s has no address", symbol->name);
}

ZTEST_DEFINE_PARAM_VALUES_ARRAY(keep_symbol_params, keep_symbols);
ZTEST_INSTANTIATE_TEST_SUITE_P(keep_symbols, model_ota_axon_binding_table,
			       test_slot_symbol_is_bound, keep_symbol_params);
