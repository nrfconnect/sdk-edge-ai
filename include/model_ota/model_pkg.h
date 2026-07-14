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
#define MODEL_PKG_FORMAT_VERSION 4

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
 * Fixed order of the raw data sections concatenated in an Axon package payload. All sections
 * are element sizes of 4 bytes or less with no internal alignment requirement stricter than 4
 * bytes, and every section's byte length here is itself a multiple of 4, so no padding is
 * needed between them.
 *
 * MODEL_STRUCT is new in format version 3: it holds the *entire*, verbatim
 * nrf_axon_nn_compiled_model_s the Axon NN compiler generated for this model (see the comment
 * on @ref model_pkg_axon_header), which is what lets this package format support any model
 * shape without the loader needing shape-specific code.
 *
 * EXTRA_OUTPUTS is present (section_len != 0) only for models with extra_output_cnt > 0; it is
 * a plain array of nrf_axon_compiled_model_output_s (no pointers of its own), so it can be
 * relocated the same way model_const is. labels and persistent_vars are not yet supported by
 * the packaging tool - see the "Known limitations" comment on @ref model_pkg_load_axon.
 */
enum model_pkg_axon_section {
	MODEL_PKG_AXON_SEC_MODEL_STRUCT = 0, /**< nrf_axon_nn_compiled_model_s, byte-for-byte */
	MODEL_PKG_AXON_SEC_CMD_BUFFER,       /**< uint32_t[cmd_buffer_len], NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE assumed 32-bit */
	MODEL_PKG_AXON_SEC_MODEL_CONST,      /**< uint8_t[model_const_size], opaque raw bytes */
	MODEL_PKG_AXON_SEC_EXTRA_OUTPUTS,    /**< nrf_axon_compiled_model_output_s[extra_output_cnt], optional */
	MODEL_PKG_AXON_SECTION_COUNT,
};

/**
 * On-flash Axon model package header.
 *
 * Distinct from, and independent of, @ref model_pkg_header: Axon packages carry a different set
 * of payload sections (see @ref model_pkg_axon_section) and no Neuton-specific fields. Both
 * share the same leading magic/format_version layout and CRC32 validation strategy for
 * consistency, but are not interchangeable.
 *
 * package_base is the flash address the MODEL_STRUCT section (and therefore the whole payload,
 * since every other section's offset is fixed relative to it) is expected to occupy once
 * flashed into the model_storage partition at the address the host tool
 * (tools/model_ota/package_model_axon.py --address) was told about. Every pointer field
 * inside MODEL_STRUCT that refers to flash-owned data (cmd_buffer_ptr, model_const_ptr,
 * extra_outputs, and cmd_buffer's own internal pointers into model_const) is already rewritten
 * by that same tool to its final flash address, so the on-device loader
 * (@ref model_pkg_load_axon) only needs to check that package_base matches where this package
 * actually landed on this device - no pointer relocation happens at load time. struct_size lets
 * the loader detect, before trusting any field offset, that the packaging tool's copy of
 * nrf_axon_nn_compiled_model_s and this firmware's copy have drifted apart.
 *
 * interlayer_addr is the reference build's address for nrf_axon_interlayer_buffer: cmd_buffer
 * contains literal references to it (for inter-layer data handoff between the NPU's compiled
 * instructions), baked in at compile time by the reference build, and - unlike model_const -
 * this loader does not relocate them (see @ref model_pkg_load_axon). Storing the reference
 * build's address lets the loader at least detect, rather than silently mispredict on, a
 * mismatch against this device's actual nrf_axon_interlayer_buffer address.
 */
struct model_pkg_axon_header {
	uint8_t magic[4]; /**< {'N','E','A','I'} */
	uint16_t format_version;
	uint8_t model_type; /**< enum model_pkg_type, always MODEL_PKG_TYPE_AXON */
	uint8_t reserved0;
	char name[MODEL_PKG_NAME_LEN];
	uint32_t model_version; /**< free-form, e.g. major.minor.patch packed by the host tool */
	uint32_t payload_size;  /**< bytes following this header; must equal sum(section_len) */
	uint32_t section_len[MODEL_PKG_AXON_SECTION_COUNT];

	uint32_t struct_size;  /**< must equal sizeof(nrf_axon_nn_compiled_model_s) on this device */
	uint32_t package_base; /**< flash address the MODEL_STRUCT section is expected to land at */
	uint32_t interlayer_addr; /**< reference build's nrf_axon_interlayer_buffer address */

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
 * @brief Load, validate, and wire up an Axon model package directly from the memory-mapped
 * model_storage flash partition (XIP; no RAM copy of any kind).
 *
 * out_model is populated with a byte-for-byte copy of the compiler-generated
 * nrf_axon_nn_compiled_model_s the packaging tool captured from a reference build's ELF (see
 * package_model_axon.py). Every pointer field that refers to flash-owned model data
 * (cmd_buffer_ptr, model_const_ptr, extra_outputs, and cmd_buffer's own internal pointers into
 * model_const) already targets its final flash address, baked in by the packaging tool - so
 * this loader does not need to know the model's shape (input count, output count, ...) at all.
 * The only pointer relocation actually performed here is patching the handful of fields that
 * refer to *this device's* RAM rather than flash: inputs[i].ptr for external inputs and
 * output_ptr are set to nrf_axon_interlayer_buffer (plus whatever offset into it the model was
 * compiled to expect, computed by the packaging tool - see model_pkg_axon.c for how that offset
 * is smuggled through the pointer field itself while the package sits in flash).
 *
 * Two checks are performed against absolute addresses baked in by the reference build, rather
 * than trusting them silently: package_base must match where the payload actually landed on
 * this device (otherwise the package was built for a different model_storage layout, e.g.
 * --address didn't match this board's overlay); and interlayer_addr must match this device's
 * actual nrf_axon_interlayer_buffer address (otherwise cmd_buffer's own embedded references to
 * that buffer - which, unlike model_const, this loader does not relocate - would silently read
 * and write the wrong RAM). Either mismatch is rejected rather than wired up with stale
 * pointers.
 *
 * Known limitations (see the plan this was implemented from for the reasoning): models with
 * labels or persistent_vars are rejected (MODEL_PKG_ERR_UNSUPPORTED_SHAPE) - the packaging tool
 * already refuses to package them, this is a defense-in-depth check. is_layer_model and
 * multi-input (input_cnt > 1) models are expected to work but, absent any such model in this
 * repo, are only exercised by host-side unit tests, not on real hardware.
 *
 * On success, out_model is fully populated and has already passed nrf_axon_nn_model_validate().
 * out_model is left untouched on failure.
 *
 * @param[out] out_model Model struct to populate (not const-qualified, unlike a normal
 *                       compile-time compiled model, since the loader owns it).
 * @param[out] out_info  Optional; filled with package metadata on success.
 * @retval MODEL_PKG_OK (0) on success, a negative @ref model_pkg_result otherwise.
 */
int model_pkg_load_axon(nrf_axon_nn_compiled_model_s *out_model,
			 struct model_pkg_axon_info *out_info);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_PKG_H_ */
