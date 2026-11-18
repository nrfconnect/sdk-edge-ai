/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/*
 * Minimal compile probe for a generated Axon model header.
 *
 * Compiled without NRF_AXON_MODEL_APP_STORAGE=extern so persistent_vars and packed_output_buf are
 * defined in the object with sizes discoverable via pyelftools. The packed-output buffer is always
 * allocated here so its size can be measured. Undefined globals list every app/driver symbol
 * the model references.
 */

#include <assert.h>
#include <stddef.h>

#include <axon/nrf_axon_platform.h>

#define NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER 1
#include MODEL_OTA_AXON_HEADER
