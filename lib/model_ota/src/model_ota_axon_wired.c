/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Raw Axon partition loader. model_ota_axon_model() compiles one instance per TARGET with -D
 * MODEL_OTA_AXON_TARGET, MODEL_OTA_PARTITION_NODELABEL and MODEL_OTA_IMAGE_LINK_BASE, then
 * defines model_ota_load_axon_<target>() (declared via model_ota_axon.h). The contract hash
 * uses MODEL_OTA_IMAGE_LINK_BASE; the mapped partition pointer comes from devicetree.
 */

#include "model_ota_axon_model_config.h"

#include "model_ota_stub_macros.h"

#include <model_ota/model_contract.h>
#include <model_ota/model_image.h>
#include <model_ota/model_ota_axon.h>
#include <zephyr/sys/util.h>

#include <drivers/axon/nrf_axon_nn_infer.h>

#ifndef MODEL_OTA_AXON_TARGET
#error "MODEL_OTA_AXON_TARGET must be defined when compiling model_ota_axon_wired.c"
#endif

#ifndef MODEL_OTA_PARTITION_NODELABEL
#error "MODEL_OTA_PARTITION_NODELABEL must be defined when compiling model_ota_axon_wired.c"
#endif

#ifndef MODEL_OTA_IMAGE_LINK_BASE
#error "MODEL_OTA_IMAGE_LINK_BASE must be defined when compiling model_ota_axon_wired.c"
#endif

MODEL_OTA_BUILD_ASSERT_MAPPED_PARTITION(MODEL_OTA_PARTITION_NODELABEL);
MODEL_OTA_BUILD_ASSERT_IMAGE_LINK_BASE(MODEL_OTA_PARTITION_NODELABEL);

#define MODEL_OTA_AXON_CONTRACT_HASH MODEL_OTA_CONTRACT_HASH_AXON(MODEL_OTA_IMAGE_LINK_BASE)

MODEL_OTA_AXON_LOAD_DECL(MODEL_OTA_AXON_TARGET)
{
	const uint8_t *const partition_addr =
		MODEL_OTA_PARTITION_ADDR(MODEL_OTA_PARTITION_NODELABEL);
	const size_t partition_size =
		MODEL_OTA_PARTITION_SIZE(MODEL_OTA_PARTITION_NODELABEL);
	const struct model_image_axon_expect expect = {
		.contract_hash = MODEL_OTA_AXON_CONTRACT_HASH,
		.persistent_vars_cap = MODEL_OTA_AXON_PERSISTENT_VARS_CAP,
		.packed_output_cap = MODEL_OTA_AXON_PACKED_OUTPUT_BYTES,
	};

	if (out == NULL) {
		return MODEL_IMAGE_ERR_AXON_VALIDATE;
	}

	return model_image_load_axon(partition_addr, partition_size, &expect, out);
}
