/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Minimal compile probe for a generated Axon model header.
 *
 * Compiled without NRF_AXON_MODEL_APP_STORAGE=extern so persistent_vars (and optional
 * packed_output_buf when NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER=1) are defined in
 * the object with sizes discoverable via pyelftools.  Undefined globals list every
 * app/driver symbol the model references.
 */

#include "model_ota_stub_macros.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>

#if defined(MODEL_OTA_AXON_PROBE) && defined(MODEL_OTA_AXON_HEADER)
#include MODEL_OTA_AXON_HEADER
#else
#error "Probe requires MODEL_OTA_AXON_PROBE and MODEL_OTA_AXON_HEADER"
#endif

/*
 * axon_elf.py uses this symbol's ELF size to identify the generated header's
 * nrf_axon_nn_compiled_model_s object without relying on its filename.
 */
const uint8_t model_ota_axon_compiled_model_size[sizeof(nrf_axon_nn_compiled_model_s)] = {0};
