/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Edge AI Lab / Neuton-backend OTA-wired loader. model_ota_edgeai_neuton_model() compiles one
 * instance per solution with -D defines (including MODEL_OTA_PARTITION_NODELABEL and
 * MODEL_OTA_IMAGE_LINK_BASE), #includes the generated model, then defines
 * nrf_edgeai_load_user_model_<SOLUTION_ID>().
 */

#include "model_ota_stub_macros.h"

#include <model_ota/model_contract.h>
#include <model_ota/model_image.h>
#include <model_ota/model_ota_edgeai.h>
#include <zephyr/sys/util.h>

#ifndef MODEL_OTA_EDGEAI_SOLUTION_ID
#error "MODEL_OTA_EDGEAI_SOLUTION_ID must be defined when compiling model_ota_edgeai_neuton_wired.c"
#endif

#ifndef MODEL_OTA_EDGEAI_NEUTON_MODEL_SRC
#error "MODEL_OTA_EDGEAI_NEUTON_MODEL_SRC must be defined when compiling model_ota_edgeai_neuton_wired.c"
#endif

#ifndef MODEL_OTA_PARTITION_NODELABEL
#error "MODEL_OTA_PARTITION_NODELABEL must be defined when compiling model_ota_edgeai_neuton_wired.c"
#endif

#ifndef MODEL_OTA_NEUTON_NEURONS_CAP
#error "MODEL_OTA_NEUTON_NEURONS_CAP must be defined when compiling model_ota_edgeai_neuton_wired.c"
#endif

#ifndef MODEL_OTA_IMAGE_LINK_BASE
#error "MODEL_OTA_IMAGE_LINK_BASE must be defined when compiling model_ota_edgeai_neuton_wired.c"
#endif

#define MODEL_OTA_WIRED 1

#include "nrf_edgeai_user_types.h"

/** OTA neuron scratch buffer (capacity = NEURONS_CAP at compile time). */
static nrf_user_neuron_t model_neurons_cap_[MODEL_OTA_NEUTON_NEURONS_CAP];

#include STRINGIFY(MODEL_OTA_EDGEAI_NEUTON_MODEL_SRC)

#include "model_ota_scale_select.h"

MODEL_OTA_BUILD_ASSERT_MAPPED_PARTITION(MODEL_OTA_PARTITION_NODELABEL);
MODEL_OTA_BUILD_ASSERT_IMAGE_LINK_BASE(MODEL_OTA_PARTITION_NODELABEL);

#define MODEL_OTA_EDGEAI_NEUTON_CONTRACT_HASH                                                      \
	MODEL_OTA_CONTRACT_HASH_EDGEAI_NEUTON(MODEL_OTA_IMAGE_LINK_BASE,                           \
					      MODEL_IMAGE_PARAMS_TYPE_OF(MODEL_PARAMS_TYPE),       \
					      MODEL_OTA_SOLUTION_CONTRACT_ARGS)

MODEL_OTA_EDGEAI_LOAD_DECL(MODEL_OTA_EDGEAI_SOLUTION_ID)
{
	const uint8_t *const partition_addr =
		MODEL_OTA_PARTITION_ADDR(MODEL_OTA_PARTITION_NODELABEL);
	const size_t partition_size =
		MODEL_OTA_PARTITION_SIZE(MODEL_OTA_PARTITION_NODELABEL);
	const struct model_image_neuton_expect expect = {
		.params_type = MODEL_IMAGE_PARAMS_TYPE_OF(MODEL_PARAMS_TYPE),
		.neurons_cap = MODEL_OTA_NEUTON_NEURONS_CAP,
		.contract_hash = MODEL_OTA_EDGEAI_NEUTON_CONTRACT_HASH,
	};
	enum model_image_result rc;

	if (out == NULL) {
		return MODEL_IMAGE_ERR_CONTRACT_MISMATCH;
	}

	*out = NULL;

	rc = model_image_load_neuton(partition_addr, partition_size, &nrf_edgeai_, model_neurons_cap_,
				     ARRAY_SIZE(model_neurons_cap_), &expect);
	if (rc != MODEL_IMAGE_OK) {
		return rc;
	}

	rc = model_image_bind_edgeai_params(partition_addr, &nrf_edgeai_);
	if (rc != MODEL_IMAGE_OK) {
		return rc;
	}

	*out = &nrf_edgeai_;

	return MODEL_IMAGE_OK;
}
