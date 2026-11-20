/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MODEL_OTA_MODEL_IMAGE_H_
#define MODEL_OTA_MODEL_IMAGE_H_

/**
 * @file
 * @brief On-flash model image format for model-only OTA.
 *
 * A model image is a fully linked artifact placed in a memory-mapped model partition. It is linked
 * at its final flash address, so every pointer it holds, in this header and in the model data,
 * is an absolute flash address and is used in place without relocation.
 *
 * @ref model_image_header opens the model image, which may itself be preceded in the partition
 * by a bootloader header. It carries a shared envelope (magic, format version, size, contract
 * hash), the model metadata, a backend-specific block and an optional pointer to the runtime
 * parameter block.
 */

#include <stddef.h>
#include <stdint.h>

#include <zephyr/toolchain.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Compiled Axon model, nrf_axon_nn_compiled_model_s in drivers/axon/nrf_axon_nn_infer.h. */
struct nrf_axon_nn_compiled_model_tag_s;

/** Runtime parameters carried by the image, defined by the runtime-specific loader. */
struct model_image_edgeai_params;

/** Image format version, independent of the model's own version. */
#define MODEL_IMAGE_FORMAT_VERSION 0

/** @ref model_image_header.magic value, stored little-endian. */
#define MODEL_IMAGE_MAGIC 0xbf6d98bbu

/** Backend that executes the model, stored in @ref model_image_header.backend. */
enum model_image_backend {
	/** Never valid in an image, so an erased or zeroed header is rejected. */
	MODEL_IMAGE_BACKEND_NONE = 0,
	/** Compiled Axon model; use @ref model_image_header.axon. */
	MODEL_IMAGE_BACKEND_AXON = 1,
};

/** One row of the Axon address binding table (@ref model_image_axon_backend.binding). */
struct model_image_binding_entry {
	/** 32-bit blake2s hash of the firmware symbol name (scripts/model_ota/model_contract.py).
	 */
	uint32_t name_hash;
	/** Address of that symbol the image was linked against. */
	const void *address;
};

/** Axon backend block (@ref model_image_header.backend == @ref MODEL_IMAGE_BACKEND_AXON). */
struct model_image_axon_backend {
	/** Compiled Axon model inside the image. */
	const struct nrf_axon_nn_compiled_model_tag_s *model;
	/** Packed-output buffer size required by the model, in bytes; 0 when unused. */
	uint32_t packed_output_bytes;
	/** Persistent-vars buffer size required by the model, in int32_t elements. */
	uint32_t persistent_vars_required;
	/** Firmware symbols the image was linked against; NULL when @ref binding_count is 0. */
	const struct model_image_binding_entry *binding;
	/** Number of entries in @ref binding. */
	uint32_t binding_count;
};

/**
 * Model image header, at the start of the model image.
 *
 * Offsets are relative to the start of the header, for a target with 32-bit pointers.
 */
struct model_image_header {
	/** off 0: @ref MODEL_IMAGE_MAGIC. */
	uint32_t magic;
	/** off 4: @ref MODEL_IMAGE_FORMAT_VERSION. */
	uint16_t format_version;
	/** off 6: @ref model_image_backend. */
	uint8_t backend;
	/** off 7: Must be 0. */
	uint8_t reserved;
	/** off 8: Bytes from the start of this header to the end of the image. */
	uint32_t image_size;
	/** off 12: Model version, free-form. */
	uint32_t model_version;
	/** off 16: Hash of the firmware ABI contract the image was built against. */
	uint32_t contract_hash;
	/** off 20: NUL-terminated model name inside the image. */
	const char *name;
	/** off 24: Runtime parameter block inside the image; NULL when the image carries none. */
	const struct model_image_edgeai_params *edgeai_params;
	/** off 28: Backend block selected by @ref backend. */
	union {
		struct model_image_axon_backend axon;
	};
};

/*
 * Layout read by the host tools, which unpack these structs as little-endian fields at fixed
 * offsets with 32-bit pointers. Update the host tools together with any change here.
 */
BUILD_ASSERT(sizeof(void *) == 4);

BUILD_ASSERT(offsetof(struct model_image_header, magic) == 0);
BUILD_ASSERT(offsetof(struct model_image_header, format_version) == 4);
BUILD_ASSERT(offsetof(struct model_image_header, backend) == 6);
BUILD_ASSERT(offsetof(struct model_image_header, image_size) == 8);
BUILD_ASSERT(offsetof(struct model_image_header, model_version) == 12);
BUILD_ASSERT(offsetof(struct model_image_header, contract_hash) == 16);
BUILD_ASSERT(offsetof(struct model_image_header, name) == 20);
BUILD_ASSERT(offsetof(struct model_image_header, edgeai_params) == 24);
BUILD_ASSERT(offsetof(struct model_image_header, axon) == 28);
BUILD_ASSERT(sizeof(struct model_image_header) == 48);

BUILD_ASSERT(offsetof(struct model_image_axon_backend, model) == 0);
BUILD_ASSERT(offsetof(struct model_image_axon_backend, packed_output_bytes) == 4);
BUILD_ASSERT(offsetof(struct model_image_axon_backend, persistent_vars_required) == 8);
BUILD_ASSERT(offsetof(struct model_image_axon_backend, binding) == 12);
BUILD_ASSERT(offsetof(struct model_image_axon_backend, binding_count) == 16);
BUILD_ASSERT(sizeof(struct model_image_axon_backend) == 20);

BUILD_ASSERT(offsetof(struct model_image_binding_entry, name_hash) == 0);
BUILD_ASSERT(offsetof(struct model_image_binding_entry, address) == 4);
BUILD_ASSERT(sizeof(struct model_image_binding_entry) == 8);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_IMAGE_H_ */
