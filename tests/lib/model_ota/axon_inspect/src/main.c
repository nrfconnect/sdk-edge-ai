/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <zephyr/sys/util.h>
#include <zephyr/toolchain.h>

#include <axon/nrf_axon_platform.h>

#include "model_ota_axon_model_config.h"

#include MODEL_OTA_AXON_HEADER

BUILD_ASSERT(MODEL_OTA_AXON_CONFIG_VERSION == 0);

BUILD_ASSERT(&MODEL_OTA_AXON_MODEL_SYM == &model_okay_nordic,
	     "MODEL_OTA_AXON_MODEL_SYM must refer to model_okay_nordic");

BUILD_ASSERT(&MODEL_OTA_AXON_PERSISTENT_VARS_SYM == &axon_model_okay_nordic_persistent_vars,
	     "MODEL_OTA_AXON_PERSISTENT_VARS_SYM must refer to the persistent-vars array");
BUILD_ASSERT(MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED > 0);
BUILD_ASSERT(MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED * sizeof(int32_t) ==
	     sizeof(axon_model_okay_nordic_persistent_vars));
BUILD_ASSERT(MODEL_OTA_AXON_PERSISTENT_VARS_CAP == MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED);

BUILD_ASSERT(MODEL_OTA_AXON_INPUT_VECTOR_CAP * sizeof(const int8_t *) ==
	     sizeof(MODEL_OTA_AXON_INPUT_VECTOR_LIST_SYM));
BUILD_ASSERT(MODEL_OTA_AXON_INPUT_VECTOR_CAP == NRF_AXON_MODEL_OKAY_NORDIC_INPUT_VECTOR_CNT);

/* The packed-output buffer is not owned by the firmware, but its size is still reported. */
BUILD_ASSERT(MODEL_OTA_AXON_PACKED_OUTPUT_ALLOC == 0);
BUILD_ASSERT(MODEL_OTA_AXON_PACKED_OUTPUT_BYTES == NRF_AXON_MODEL_OKAY_NORDIC_PACKED_OUTPUT_SIZE);
#ifdef MODEL_OTA_AXON_PACKED_OUTPUT_SYM
#error "MODEL_OTA_AXON_PACKED_OUTPUT_SYM must only be set for a firmware-owned buffer"
#endif

BUILD_ASSERT(MODEL_OTA_AXON_KEEP_SYMBOL_COUNT > 0);

/* Every keep symbol must be an identifier the application can take the address of. */
#define KEEP_REF(symbol) (const void *)&(symbol),
static const void *const keep_refs[] = {MODEL_OTA_AXON_KEEP_REFS(KEEP_REF)};
BUILD_ASSERT(ARRAY_SIZE(keep_refs) == MODEL_OTA_AXON_KEEP_SYMBOL_COUNT);

int main(void)
{
	/* Keep the reference table in the link, so every keep symbol must resolve. */
	__asm__ volatile("" : : "r"(keep_refs));

	return 0;
}
