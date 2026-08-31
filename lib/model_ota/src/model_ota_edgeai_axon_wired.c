/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Edge AI Lab / Axon-backend OTA-wired loader. model_ota_edgeai_axon_model() compiles one
 * instance per solution with -D defines, #includes the generated nrf_edgeai_user_model.c (which
 * skips the generated Axon model header and leaves model.instance.p_void NULL), then defines
 * nrf_edgeai_load_user_model_<SOLUTION_ID>() to fill the context from a loaded model image.
 *
 * TODO: MODEL_OTA_AXON_TARGET and MODEL_OTA_AXON_TOKEN exist only because this TU includes the
 * token-suffixed public header. The generated private axon_config.h carries the same values
 * unsuffixed (MODEL_OTA_AXON_PERSISTENT_VARS_CAP, MODEL_OTA_AXON_PACKED_OUTPUT_BYTES);
 * force-including it instead removes both -D symbols.
 *
 * TODO: the hash below is folded over the token-suffixed ..._IMAGE_BASE macro while every other
 * TU folds the same quantity as MODEL_OTA_PARTITION_ADDR. One hashed input should have one name.
 */

#include "model_ota_stub_macros.h"

#include <model_ota/model_contract.h>
#include <model_ota/model_image.h>
#include <model_ota/model_ota_edgeai.h>
#include <zephyr/sys/util.h>

#include <drivers/axon/nrf_axon_nn_infer.h>

#ifndef MODEL_OTA_EDGEAI_SOLUTION_ID
#error "MODEL_OTA_EDGEAI_SOLUTION_ID must be defined when compiling model_ota_edgeai_axon_wired.c"
#endif

#ifndef MODEL_OTA_EDGEAI_AXON_MODEL_SRC
#error "MODEL_OTA_EDGEAI_AXON_MODEL_SRC must be defined when compiling model_ota_edgeai_axon_wired.c"
#endif

#ifndef MODEL_OTA_PARTITION_NODELABEL
#error "MODEL_OTA_PARTITION_NODELABEL must be defined when compiling model_ota_edgeai_axon_wired.c"
#endif

#ifndef MODEL_OTA_AXON_TOKEN
#error "MODEL_OTA_AXON_TOKEN must be defined when compiling model_ota_edgeai_axon_wired.c"
#endif

#ifndef MODEL_OTA_AXON_TARGET
#error "MODEL_OTA_AXON_TARGET must be defined when compiling model_ota_edgeai_axon_wired.c"
#endif

#define MODEL_OTA_AXON_PUBLIC_HDR(target) model_ota/axon/target.h
#define MODEL_OTA_AXON_PUBLIC_HDR_STR(target) STRINGIFY(MODEL_OTA_AXON_PUBLIC_HDR(target))

#include MODEL_OTA_AXON_PUBLIC_HDR_STR(MODEL_OTA_AXON_TARGET)

#define MODEL_OTA_WIRED 1

#include STRINGIFY(MODEL_OTA_EDGEAI_AXON_MODEL_SRC)

#include "model_ota_scale_select.h"

#define MODEL_OTA_AXON_SYM2(token, name) MODEL_OTA_AXON_##token##_##name
#define MODEL_OTA_AXON_SYM1(token, name) MODEL_OTA_AXON_SYM2(token, name)
#define MODEL_OTA_AXON_SYM(name)         MODEL_OTA_AXON_SYM1(MODEL_OTA_AXON_TOKEN, name)

extern const uint32_t MODEL_OTA_AXON_SYM(KEEP_LABEL)[];

MODEL_OTA_BUILD_ASSERT_MAPPED_PARTITION(MODEL_OTA_PARTITION_NODELABEL);

#define MODEL_OTA_EDGEAI_AXON_CONTRACT_HASH                                                        \
	MODEL_OTA_CONTRACT_HASH_EDGEAI_AXON(MODEL_OTA_AXON_SYM(IMAGE_BASE),                        \
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
		.persistent_vars_cap = MODEL_OTA_AXON_SYM(PERSISTENT_VARS_CAP),
		.packed_output_cap = MODEL_OTA_AXON_SYM(PACKED_OUTPUT_BYTES),
		.binding_table = MODEL_OTA_AXON_SYM(KEEP_LABEL),
	};

	if (out == NULL) {
		return MODEL_IMAGE_ERR_AXON_VALIDATE;
	}

	*out = NULL;

	rc = model_image_load_axon(partition_addr, partition_size, &expect, &model);
	if (rc != MODEL_IMAGE_OK) {
		return rc;
	}

	rc = model_image_bind_edgeai_params(partition_addr, &nrf_edgeai_);
	if (rc != MODEL_IMAGE_OK) {
		return rc;
	}

	nrf_edgeai_.model.instance.p_void = (void *)model;
	*out = &nrf_edgeai_;

	return MODEL_IMAGE_OK;
}
