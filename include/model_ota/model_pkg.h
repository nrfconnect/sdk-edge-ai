/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef MODEL_OTA_MODEL_PKG_H_
#define MODEL_OTA_MODEL_PKG_H_

/**
 * @file
 * @brief On-flash "model package" format for the model-only OTA update PoC.
 *
 * A model package is what a host-side tool (tools/model_ota/package_model*.py) writes into
 * the model_storage flash partition, independently of the application image. Two independent
 * package layouts are defined, one per model backend:
 *
 * - Neuton (@ref model_pkg_header / @ref model_pkg_load_neuton): a payload made of the model's
 *   own raw arrays, with no embedded absolute addresses - the loader just wires those arrays
 *   into a working nrf_edgeai_model_neuton_t at runtime.
 * - Axon (@ref model_pkg_axon_header / @ref model_pkg_load_axon): a payload made of the model's
 *   *entire* compiler-generated nrf_axon_nn_compiled_model_s struct, verbatim, plus its command
 *   buffer, constants blob, and (optionally) extra_outputs array. The packaging tool
 *   (tools/model_ota/package_model_axon.py) classifies every pointer field in the struct as
 *   either flash-owned model data (rewritten to its final flash address, exactly like the
 *   command buffer's own internal pointers into the constants blob) or app-owned RAM (zeroed;
 *   the on-device loader fills those back in from addresses it actually has, e.g.
 *   nrf_axon_interlayer_buffer). This means the loader does not need to know a model's shape
 *   (input count, output count, ...) at all - see the comment on @ref model_pkg_load_axon.
 */

#include <stdint.h>
#include <stddef.h>

#include <nrf_edgeai/rt/nrf_edgeai_model_types.h>
#include <drivers/axon/nrf_axon_nn_infer.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Container format version (independent of the model's own version). */
#define MODEL_PKG_FORMAT_VERSION 6

/** Length of the @ref model_pkg_header.name field, not necessarily NUL-terminated. */
#define MODEL_PKG_NAME_LEN 16

#define MODEL_PKG_MAGIC0 'N'
#define MODEL_PKG_MAGIC1 'E'
#define MODEL_PKG_MAGIC2 'A'
#define MODEL_PKG_MAGIC3 'I'

/** Model backend a package was built for. Only MODEL_PKG_TYPE_NEUTON is implemented so far. */
enum model_pkg_type {
	MODEL_PKG_TYPE_NEUTON = 0,
	MODEL_PKG_TYPE_AXON = 1,
};

/**
 * Fixed order of the raw data sections concatenated in a Neuton package payload.
 * Byte length of each section is recorded in @ref model_pkg_header.section_len; element
 * counts (weights_num/neurons_num/outputs_num) are derived from those lengths rather than
 * stored separately, since they are redundant with them.
 *
 * Sections are concatenated back-to-back with no padding, so this order is deliberately
 * grouped by element size (float sections, then uint16_t sections, then the one uint8_t
 * section last): every section's byte length here is itself a multiple of that group's
 * element size, so each section boundary naturally falls on a properly aligned address for
 * the next section's element type. MODEL_PKG_NEUTON_SEC_ACT_TYPE_MASK in particular can have
 * an odd byte length (ceil(neurons_num/8)), so it must be last - placing it earlier would
 * misalign every uint16_t/float section that followed it, causing unaligned-access faults
 * when the loaded model's arrays are dereferenced.
 */
enum model_pkg_neuton_section {
	MODEL_PKG_NEUTON_SEC_WEIGHTS = 0,        /**< float[weights_num] */
	MODEL_PKG_NEUTON_SEC_ACT_WEIGHTS,        /**< float[neurons_num] */
	MODEL_PKG_NEUTON_SEC_OUTPUT_SCALE_MIN,   /**< float[outputs_num] */
	MODEL_PKG_NEUTON_SEC_OUTPUT_SCALE_MAX,   /**< float[outputs_num] */
	MODEL_PKG_NEUTON_SEC_NEURON_LINKS,       /**< uint16_t[weights_num] */
	MODEL_PKG_NEUTON_SEC_INTERNAL_LINKS_NUM, /**< uint16_t[neurons_num] */
	MODEL_PKG_NEUTON_SEC_EXTERNAL_LINKS_NUM, /**< uint16_t[neurons_num] */
	MODEL_PKG_NEUTON_SEC_OUTPUT_INDICES,     /**< uint16_t[outputs_num] */
	MODEL_PKG_NEUTON_SEC_ACT_TYPE_MASK,      /**< uint8_t[ceil(neurons_num/8)] - must stay last, see above */
	MODEL_PKG_NEUTON_SECTION_COUNT,
};

/**
 * On-flash model package header.
 *
 * Validated in this order before any payload byte is trusted: magic, format_version,
 * model_type, payload_size vs. buffer capacity, section_len[] sum vs. payload_size, then CRC32
 * over the whole header (with crc32 itself treated as 0) plus the payload.
 */
struct model_pkg_header {
	uint8_t magic[4]; /**< {'N','E','A','I'} */
	uint16_t format_version;
	uint8_t model_type; /**< enum model_pkg_type */
	uint8_t reserved0;
	char name[MODEL_PKG_NAME_LEN];
	uint32_t model_version; /**< free-form, e.g. major.minor.patch packed by the host tool */
	uint32_t payload_size;  /**< bytes following this header; must equal sum(section_len) */
	uint32_t section_len[MODEL_PKG_NEUTON_SECTION_COUNT];
	uint32_t crc32; /**< CRC32 (IEEE) over header (crc32 field zeroed) + payload */
} __packed;

/** Human-readable metadata about a successfully loaded Neuton package. */
struct model_pkg_neuton_info {
	char name[MODEL_PKG_NAME_LEN + 1];
	uint32_t version;
	uint16_t neurons_num;
	uint16_t outputs_num;
	uint32_t weights_num;
	const float *output_scale_min; /**< output_scale_min[outputs_num] */
	const float *output_scale_max; /**< output_scale_max[outputs_num] */
};

/** Return codes for @ref model_pkg_load_neuton. */
enum model_pkg_result {
	MODEL_PKG_OK = 0,
	MODEL_PKG_ERR_NO_PARTITION = -1,
	MODEL_PKG_ERR_FLASH_READ = -2,
	MODEL_PKG_ERR_BAD_MAGIC = -3,
	MODEL_PKG_ERR_BAD_FORMAT_VERSION = -4,
	MODEL_PKG_ERR_WRONG_TYPE = -5,
	MODEL_PKG_ERR_TOO_LARGE = -6,
	MODEL_PKG_ERR_BAD_SECTION_LEN = -7,
	MODEL_PKG_ERR_BAD_CRC = -8,
	MODEL_PKG_ERR_NEURONS_BUF_TOO_SMALL = -9,
	MODEL_PKG_ERR_AXON_VALIDATE_FAILED = -10,
	MODEL_PKG_ERR_BAD_PACKAGE_BASE = -11,
	MODEL_PKG_ERR_BAD_STRUCT_SIZE = -12,
	MODEL_PKG_ERR_UNSUPPORTED_SHAPE = -13,
	MODEL_PKG_ERR_INTERLAYER_MISMATCH = -14,
};

/**
 * @brief Load, validate, and wire up a Neuton (f32) model package from the model_storage
 * flash partition.
 *
 * On success, out_model's meta/params.f32 array pointers point into an internal static RAM
 * copy of the package payload (owned by this library, valid for the life of the program).
 * out_model is left untouched on failure.
 *
 * @param[out] out_model       Model struct to populate.
 * @param[out] neurons_buf     Caller-owned RAM scratch buffer for neuron activations
 *                             (out_model's p_neurons will point here).
 * @param[in]  neurons_buf_cap Capacity of neurons_buf, in elements.
 * @param[out] out_info        Optional; filled with package metadata on success.
 * @retval MODEL_PKG_OK (0) on success, a negative @ref model_pkg_result otherwise.
 */
int model_pkg_load_neuton(nrf_edgeai_model_neuton_t *out_model, float *neurons_buf,
			   size_t neurons_buf_cap, struct model_pkg_neuton_info *out_info);

/**
 * On-flash Axon model package header.
 *
 * Distinct from, and independent of, @ref model_pkg_header: Axon packages carry a payload that
 * is the model stub's own linked memory image, byte-for-byte, rather than a fixed set of
 * separately-tracked sections; and no Neuton-specific fields. Both share the same leading
 * magic/format_version layout and CRC32 validation strategy for consistency, but are not
 * interchangeable.
 *
 * The payload (everything following this header, payload_size bytes) is copied verbatim from
 * the "model stub" ELF's `.model_stub` output section (tools/model_ota/package_model_axon.py's
 * docstring, and doc/libraries/model_ota.rst, describe how that ELF is produced): that stub was
 * linked with its `.model_stub` section placed exactly at package_base, which by construction
 * is where this payload will actually sit once flashed (package_base = the model_storage
 * partition's own address + sizeof(this header)) - so *every* pointer field baked into it by
 * the compiler and linker, including ones that refer to app-owned RAM
 * (nrf_axon_interlayer_buffer, persistent_vars, op-extension function pointers embedded in
 * cmd_buffer), is already the final, correct absolute address for the specific application this
 * package was built against. The on-device loader (@ref model_pkg_load_axon) therefore performs
 * no pointer relocation of any kind - it only checks that package_base matches where the payload
 * actually landed on this device (otherwise the package was built for a different partition
 * layout and cannot be trusted), and that struct_size matches this firmware's own
 * nrf_axon_nn_compiled_model_s, before trusting struct_offset to find the model struct.
 */
struct model_pkg_axon_header {
	uint8_t magic[4]; /**< {'N','E','A','I'} */
	uint16_t format_version;
	uint8_t model_type; /**< enum model_pkg_type, always MODEL_PKG_TYPE_AXON */
	uint8_t reserved0;
	char name[MODEL_PKG_NAME_LEN];
	uint32_t model_version; /**< free-form, e.g. major.minor.patch packed by the host tool */
	uint32_t payload_size;  /**< bytes following this header - the model stub's memory image */
	uint32_t struct_offset; /**< offset within the payload where nrf_axon_nn_compiled_model_s starts */
	uint32_t struct_size;   /**< must equal sizeof(nrf_axon_nn_compiled_model_s) on this device */
	uint32_t package_base;  /**< flash address the payload (right after this header) is expected to occupy */

	uint32_t crc32; /**< CRC32 (IEEE) over header (crc32 field zeroed) + payload */
} __packed;

/** Human-readable metadata about a successfully loaded Axon package. */
struct model_pkg_axon_info {
	char name[MODEL_PKG_NAME_LEN + 1];
	uint32_t version;
	uint32_t cmd_buffer_len;
	uint32_t model_const_size;
};

/**
 * @brief Load, validate, and wire up an Axon model package directly from a memory-mapped flash
 * partition (XIP; no RAM copy of any kind).
 *
 * out_model is populated with a byte-for-byte copy of the compiler-generated
 * nrf_axon_nn_compiled_model_s the packaging tool captured from a "model stub" ELF (see
 * package_model_axon.py and doc/libraries/model_ota.rst for how that stub is produced).
 * *Every* pointer field - including ones that refer to app-owned RAM
 * (nrf_axon_interlayer_buffer, persistent_vars, and op-extension function pointers embedded in
 * cmd_buffer) - already targets its final, correct address for *this specific application*,
 * resolved by that build-time mechanism rather than at load time: this loader performs no
 * pointer relocation or patching of any kind, and does not need to know the model's shape
 * (input count, output count, persistent_vars, op extensions used, ...) at all.
 *
 * The only check performed against an absolute address baked in at packaging time is
 * package_base, which must match where the payload actually landed on this device (otherwise
 * the package was built for a different partition layout, e.g. --address didn't match this
 * board's overlay, or for a different partition than fa_id/partition_addr identify) - rejected
 * rather than wired up with stale pointers.
 *
 * On success, out_model is fully populated and has already passed nrf_axon_nn_model_validate().
 * out_model is left untouched on failure.
 *
 * @param[in]  fa_id          Flash area ID of the partition to load from, e.g.
 *                             FIXED_PARTITION_ID(model_partition).
 * @param[in]  partition_addr Memory-mapped base address of that same partition, e.g.
 *                             PARTITION_ADDRESS(model_partition). Passed separately from fa_id
 *                             (rather than derived from it at runtime) since it is needed
 *                             before the flash_area is opened, and only a "zephyr,mapped-
 *                             partition" node's address is ever CPU-addressable this way.
 * @param[out] out_model      Model struct to populate (not const-qualified, unlike a normal
 *                             compile-time compiled model, since the loader owns it).
 * @param[out] out_info       Optional; filled with package metadata on success.
 * @retval MODEL_PKG_OK (0) on success, a negative @ref model_pkg_result otherwise.
 */
int model_pkg_load_axon(uint8_t fa_id, const uint8_t *partition_addr,
			 nrf_axon_nn_compiled_model_s *out_model,
			 struct model_pkg_axon_info *out_info);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_PKG_H_ */
