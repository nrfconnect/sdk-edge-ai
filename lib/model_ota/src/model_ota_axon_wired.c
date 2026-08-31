/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Raw Axon partition loader. model_ota_axon_model() compiles one instance per TARGET with -D
 * MODEL_OTA_AXON_TARGET, MODEL_OTA_AXON_TOKEN and MODEL_OTA_PARTITION_NODELABEL, then defines
 * model_ota_load_axon_<target>() (declared via model_ota_axon.h).
 *
 * TODO: MODEL_OTA_AXON_TARGET and MODEL_OTA_AXON_TOKEN exist only because this TU includes the
 * token-suffixed public header. The generated private axon_config.h carries the same values
 * unsuffixed; force-including it instead removes both -D symbols. See model_ota_edgeai_axon_wired.c.
 */

#include "model_ota_stub_macros.h"

#include <model_ota/model_image.h>
#include <model_ota/model_ota_axon.h>
#include <zephyr/sys/util.h>

#include <drivers/axon/nrf_axon_nn_infer.h>

#ifndef MODEL_OTA_AXON_TARGET
#error "MODEL_OTA_AXON_TARGET must be defined when compiling model_ota_axon_wired.c"
#endif

#ifndef MODEL_OTA_AXON_TOKEN
#error "MODEL_OTA_AXON_TOKEN must be defined when compiling model_ota_axon_wired.c"
#endif

#ifndef MODEL_OTA_PARTITION_NODELABEL
#error "MODEL_OTA_PARTITION_NODELABEL must be defined when compiling model_ota_axon_wired.c"
#endif

#define MODEL_OTA_AXON_PUBLIC_HDR(target) model_ota/axon/target.h
#define MODEL_OTA_AXON_PUBLIC_HDR_STR(target) STRINGIFY(MODEL_OTA_AXON_PUBLIC_HDR(target))

#include MODEL_OTA_AXON_PUBLIC_HDR_STR(MODEL_OTA_AXON_TARGET)

#define MODEL_OTA_AXON_SYM2(token, name) MODEL_OTA_AXON_##token##_##name
#define MODEL_OTA_AXON_SYM1(token, name) MODEL_OTA_AXON_SYM2(token, name)
#define MODEL_OTA_AXON_SYM(name)         MODEL_OTA_AXON_SYM1(MODEL_OTA_AXON_TOKEN, name)

extern const uint32_t MODEL_OTA_AXON_SYM(KEEP_LABEL)[];

MODEL_OTA_BUILD_ASSERT_MAPPED_PARTITION(MODEL_OTA_PARTITION_NODELABEL);

MODEL_OTA_AXON_LOAD_DECL(MODEL_OTA_AXON_TARGET)
{
	const uint8_t *const partition_addr =
		MODEL_OTA_PARTITION_ADDR(MODEL_OTA_PARTITION_NODELABEL);
	const size_t partition_size =
		MODEL_OTA_PARTITION_SIZE(MODEL_OTA_PARTITION_NODELABEL);
	const struct model_image_axon_expect expect = {
		.contract_hash = MODEL_OTA_AXON_SYM(CONTRACT_HASH),
		.persistent_vars_cap = MODEL_OTA_AXON_SYM(PERSISTENT_VARS_CAP),
		.packed_output_cap = MODEL_OTA_AXON_SYM(PACKED_OUTPUT_BYTES),
		.binding_table = MODEL_OTA_AXON_SYM(KEEP_LABEL),
	};

	if (out == NULL) {
		return MODEL_IMAGE_ERR_AXON_VALIDATE;
	}

	return model_image_load_axon(partition_addr, partition_size, &expect, out);
}
