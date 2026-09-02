/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Contract-hash probe: one throwaway translation unit per model slot, compiled with the
 * application's own flags and linked into neither the firmware nor the image.
 *
 * MODEL_OTA_CONTRACT_HASH_* folds sizeof() of the runtime structs, so only the compiler can
 * evaluate it - the preprocessor cannot, and a host reimplementation would be a second
 * definition of the hash drifting against this one. The build therefore compiles this file and
 * reads the finished word back out of the object (tools/model_ota/elf_const.py), which is what
 * puts the compiler's own value into model_ota_context.json.
 *
 * The probe must see exactly the firmware's headers and Kconfig. The flavor-specific stub bakes
 * the same macro into the image from a different translation unit, and check_model_compat.py
 * requires the two to agree - a mismatch there means this object was not compiled the way the
 * firmware was.
 *
 * TODO: three translation units must reach the same hash - this probe, the image stub and the
 * wired TU - and each is parameterized differently: raw -D on a hand-built compiler command line
 * here (model_ota_contract_probe() in model_ota_common.cmake), target_compile_definitions() for
 * the stub and the wired TU. Since the whole guarantee is that all three saw identical inputs,
 * they should share one delivery mechanism and one define set, ideally emitted by a single
 * helper. See model_ota_edgeai_neuton_wired.c and model_ota_edgeai_axon_wired.c.
 */

#include "model_ota_stub_macros.h"

#include <stdint.h>

#include <model_ota/model_contract.h>

#ifndef NRF_MODEL_PARTITION_ADDR
#error "NRF_MODEL_PARTITION_ADDR must be defined by model_ota_contract_probe()"
#endif

#ifdef MODEL_OTA_CONTRACT_PROBE_MODEL_SRC
/*
 * An Edge AI Lab solution: its contract covers the generated nrf_edgeai_t pipeline, so the
 * generated source has to be in scope. MODEL_OTA_WIRED keeps the payload initializers out of this
 * object, matching the application's view of the source; MODEL_OTA_SOLUTION_CONTRACT_ARGS does not
 * depend on it, which is what lets the image stub reach the same value without it.
 */
#define MODEL_OTA_WIRED 1

/*
 * Only the source's macros are read here, never its definitions, so every buffer and payload array
 * it emits is unused by construction - unlike in the stubs, where an unused definition would be a
 * real finding. The application compiles the same source with the same flags, so nothing is hidden.
 */

#include STRINGIFY(MODEL_OTA_CONTRACT_PROBE_MODEL_SRC)

#include "model_ota_scale_select.h"
#endif /* MODEL_OTA_CONTRACT_PROBE_MODEL_SRC */

#if defined(MODEL_OTA_CONTRACT_PROBE_EDGEAI_NEUTON)
#define MODEL_OTA_CONTRACT_PROBE_HASH                                                              \
	MODEL_OTA_CONTRACT_HASH_EDGEAI_NEUTON(NRF_MODEL_PARTITION_ADDR,                            \
					      MODEL_IMAGE_PARAMS_TYPE_OF(MODEL_PARAMS_TYPE),       \
					      MODEL_OTA_SOLUTION_CONTRACT_ARGS)
#elif defined(MODEL_OTA_CONTRACT_PROBE_EDGEAI_AXON)
#define MODEL_OTA_CONTRACT_PROBE_HASH                                                              \
	MODEL_OTA_CONTRACT_HASH_EDGEAI_AXON(NRF_MODEL_PARTITION_ADDR,                              \
					    MODEL_OTA_SOLUTION_CONTRACT_ARGS)
#elif defined(MODEL_OTA_CONTRACT_PROBE_AXON)
#define MODEL_OTA_CONTRACT_PROBE_HASH MODEL_OTA_CONTRACT_HASH_AXON(NRF_MODEL_PARTITION_ADDR)
#else
#error "model_ota_contract_probe() must select a flavor"
#endif

/** The word tools/model_ota/elf_const.py reads back; see @ref CONTRACT_HASH_SYMBOL. */
const uint32_t model_ota_contract_hash = MODEL_OTA_CONTRACT_PROBE_HASH;
