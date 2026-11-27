/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MODEL_OTA_MODEL_CONTRACT_H_
#define MODEL_OTA_MODEL_CONTRACT_H_

/**
 * @file
 * @brief Firmware ABI contract hash of model images.
 *
 * The model image and the firmware compute the same @ref model_image_header.contract_hash from
 * compile-time constants, so every macro here can be used in a static initializer. The hash is
 * FNV-1a over u32 chunks.
 *
 * Only values that must be equal on both sides belong in the hash, such as struct layouts and
 * the partition base. Capacities, where a model image may require less than the firmware
 * provides, are carried in @ref model_image_header and checked by inequality when the model
 * image is loaded, so a model that still fits does not turn into a contract mismatch.
 */

#include <stdint.h>

#include <drivers/axon/nrf_axon_nn_infer.h>

#include "model_image.h"

#ifdef __cplusplus
extern "C" {
#endif

/** FNV-1a 32-bit offset basis. */
#define MODEL_OTA_HASH_SEED 2166136261u

/** FNV-1a 32-bit prime. */
#define MODEL_OTA_HASH_PRIME 16777619u

/** One FNV-1a step over a u32 chunk. */
#define MODEL_OTA_HASH_U32(h, k) (((uint32_t)(h) ^ (uint32_t)(k)) * MODEL_OTA_HASH_PRIME)

/** Fold two chunks into @p h, left to right. */
#define MODEL_OTA_HASH_U32_2(h, a, b) MODEL_OTA_HASH_U32(MODEL_OTA_HASH_U32((h), (a)), (b))

/** Fold three chunks into @p h, left to right. */
#define MODEL_OTA_HASH_U32_3(h, a, b, c)                                                           \
	MODEL_OTA_HASH_U32(MODEL_OTA_HASH_U32_2((h), (a), (b)), (c))

/**
 * Envelope shared by every backend: the image format and the partition base.
 *
 * A model image holds absolute flash addresses, so it is only valid in the partition it was
 * linked for. Hashing the partition base also keeps a model image out of another model slot that
 * agrees on everything else.
 */
#define MODEL_OTA_HASH_ENVELOPE(h, partition_base)                                                 \
	MODEL_OTA_HASH_U32_2((h), MODEL_IMAGE_FORMAT_VERSION, (uint32_t)(uintptr_t)(partition_base))

/**
 * Axon driver ABI. The interlayer and psum buffer sizes are firmware-wide values the model was
 * compiled against, not per-model requirements, so they must be equal.
 */
#define MODEL_OTA_HASH_AXON_MIX(h)                                                                 \
	MODEL_OTA_HASH_U32_3((h), (uint32_t)sizeof(nrf_axon_nn_compiled_model_s),                  \
			     (uint32_t)CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE,                     \
			     (uint32_t)CONFIG_NRF_AXON_PSUM_BUFFER_SIZE)

/**
 * Contract hash of an Axon model image.
 *
 * The persistent-vars and packed-output sizes are left out on purpose. They are per-model
 * requirements, carried in @ref model_image_axon_backend and checked against the model slot
 * capacities when the model image is loaded.
 *
 * @param partition_base Memory-mapped base address of the model partition.
 */
#define MODEL_OTA_CONTRACT_HASH_AXON(partition_base)                                               \
	MODEL_OTA_HASH_AXON_MIX(MODEL_OTA_HASH_ENVELOPE(MODEL_OTA_HASH_SEED, (partition_base)))

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_CONTRACT_H_ */
