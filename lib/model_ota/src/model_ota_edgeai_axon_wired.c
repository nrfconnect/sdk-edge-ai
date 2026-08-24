/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Edge AI Lab / Axon-backend OTA-wired loader. model_ota_edgeai_axon_model() compiles one
 * instance per solution with -D defines (including MODEL_OTA_PARTITION_NODELABEL and
 * MODEL_OTA_IMAGE_LINK_BASE), #includes the generated nrf_edgeai_user_model.c (which skips the
 * generated Axon model header and leaves model.instance.p_void NULL), then defines
 * nrf_edgeai_load_user_model_<SOLUTION_ID>() to fill the context from a loaded model image.
 */

#include "model_ota_axon_model_config.h"

#include "model_ota_stub_macros.h"

#include <errno.h>

#include <model_ota/model_contract.h>
#include <model_ota/model_image.h>
#include <model_ota/model_ota_edgeai.h>
#include <zephyr/sys/util.h>

#include <drivers/axon/nrf_axon_nn_infer.h>

#if IS_ENABLED(CONFIG_MODEL_OTA_SMP)
#include <model_ota/model_ota_partition.h>
#include <model_ota/model_ota_smp.h>
#endif

#ifndef MODEL_OTA_EDGEAI_SOLUTION_ID
#error "MODEL_OTA_EDGEAI_SOLUTION_ID must be defined when compiling model_ota_edgeai_axon_wired.c"
#endif

#ifndef MODEL_OTA_EDGEAI_AXON_MODEL_SRC
#error "MODEL_OTA_EDGEAI_AXON_MODEL_SRC must be defined when compiling model_ota_edgeai_axon_wired.c"
#endif

#ifndef MODEL_OTA_PARTITION_NODELABEL
#error "MODEL_OTA_PARTITION_NODELABEL must be defined when compiling model_ota_edgeai_axon_wired.c"
#endif

#ifndef MODEL_OTA_IMAGE_LINK_BASE
#error "MODEL_OTA_IMAGE_LINK_BASE must be defined when compiling model_ota_edgeai_axon_wired.c"
#endif

#define MODEL_OTA_WIRED 1

#include STRINGIFY(MODEL_OTA_EDGEAI_AXON_MODEL_SRC)

#include "model_ota_scale_select.h"

MODEL_OTA_BUILD_ASSERT_MAPPED_PARTITION(MODEL_OTA_PARTITION_NODELABEL);
MODEL_OTA_BUILD_ASSERT_IMAGE_LINK_BASE(MODEL_OTA_PARTITION_NODELABEL);

#if IS_ENABLED(CONFIG_MODEL_OTA_SMP)
MODEL_OTA_PARTITION_ASSERT(MODEL_OTA_PARTITION_NODELABEL);

#ifndef MODEL_OTA_SMP_SLOT_NAME
#define MODEL_OTA_SMP_SLOT_NAME STRINGIFY(MODEL_OTA_PARTITION_NODELABEL)
#endif

static const struct model_ota_smp_slot model_ota_smp_slot = {
	.image_index = MODEL_OTA_IMAGE_INDEX(MODEL_OTA_PARTITION_NODELABEL),
	.name = MODEL_OTA_SMP_SLOT_NAME,
};
#endif

#define MODEL_OTA_EDGEAI_AXON_CONTRACT_HASH                                                        \
	MODEL_OTA_CONTRACT_HASH_EDGEAI_AXON(MODEL_OTA_IMAGE_LINK_BASE,                             \
					    MODEL_OTA_SOLUTION_CONTRACT_ARGS)

MODEL_OTA_EDGEAI_LOAD_DECL(MODEL_OTA_EDGEAI_SOLUTION_ID)
{
	const uint8_t *const partition_addr =
		MODEL_OTA_PARTITION_ADDR(MODEL_OTA_PARTITION_NODELABEL);
	const size_t partition_size =
		MODEL_OTA_PARTITION_SIZE(MODEL_OTA_PARTITION_NODELABEL);
	const nrf_axon_nn_compiled_model_s *model;
	enum model_image_result rc;
	const struct model_image_axon_expect expect = {
		.contract_hash = MODEL_OTA_EDGEAI_AXON_CONTRACT_HASH,
		.persistent_vars_cap = MODEL_OTA_AXON_PERSISTENT_VARS_CAP,
		.packed_output_cap = MODEL_OTA_AXON_PACKED_OUTPUT_BYTES,
	};

	if (out == NULL) {
		return MODEL_IMAGE_ERR_AXON_VALIDATE;
	}

	*out = NULL;

#if IS_ENABLED(CONFIG_MODEL_OTA_SMP)
	{
		int smp_rc = model_ota_smp_register(&model_ota_smp_slot);

		if (smp_rc != 0 && smp_rc != -EALREADY) {
			return MODEL_IMAGE_ERR_AXON_VALIDATE;
		}
	}
#endif

	rc = model_image_load_axon(partition_addr, partition_size, &expect, &model);
	if (rc != MODEL_IMAGE_OK) {
		return rc;
	}

	rc = model_image_bind_edgeai_params(partition_addr, &nrf_edgeai_);
	if (rc != MODEL_IMAGE_OK) {
		return rc;
	}

	nrf_edgeai_.model.instance.p_void = (void *)model;
	nrf_edgeai_.is_ota_managed = true;
	*out = &nrf_edgeai_;

	return MODEL_IMAGE_OK;
}
