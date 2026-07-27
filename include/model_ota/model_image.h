/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef MODEL_OTA_MODEL_IMAGE_H_
#define MODEL_OTA_MODEL_IMAGE_H_

/**
 * @file
 * @brief On-flash "model partition image" format for model-only OTA (Neuton and Axon).
 *
 * A model *image* is a self-contained, fully linked artifact - built exactly like the Axon
 * "compiled-into-partition-image" flow:
 *
 *   - The model's compiled nrf_edgeai_model_neuton_t descriptor (model_instance_) AND all of its
 *     data (weights / act_weights / links / indices / act_type_mask / output scales) are emitted
 *     by the *compiler*, gathered into one .model_image output section, and linked AT the model
 *     partition's flash base address. Because the whole image is linked at that base, every
 *     intra-image pointer (the descriptor's p_weights, p_neuron_links, ... and the header's
 *     model pointer) is already a correct *absolute flash address* at link time - no runtime
 *     base+offset arithmetic is ever needed.
 *
 *   - The partition header (@ref model_image_header) therefore stores a DIRECT POINTER to the
 *     baked descriptor (@ref model_image_neuton_backend.model /
 *     @ref model_image_axon_backend.model), not a model_offset. The loader validates the header
 *     and hands that pointer straight back.
 *
 * The one field that cannot be a partition-flash address is
 * nrf_edgeai_model_neuton_t.params.*.p_neurons: it must point at the application's neuron-
 * activation scratch buffer in RAM (the Neuton analogue of Axon's nrf_axon_interlayer_buffer).
 * The image bakes the descriptor with all *flash* pointers absolute and leaves p_neurons to be
 * set by the loader from a caller-owned buffer (see @ref model_image_load_neuton). This "hybrid"
 * choice is what keeps the flow multi-model friendly: the alternative (PROVIDE()-ing
 * model_neurons_ from zephyr.elf, pure Axon style) cannot disambiguate the three identical
 * file-static `model_neurons_` symbols the multi_model sample compiles.
 *
 * Each OTA-wired Neuton model's payload is dropped from its dedicated static library via
 * archive-scoped linker /DISCARD/ rules (model_ota_neuton.cmake). Edge AI Lab / Axon-backend
 * models omit the compiled Axon weights from the app via MODEL_OTA_AXON_RUNTIME_WIRED instead
 * (model_ota_axon_edgeai.cmake). Models compiled directly into the app are unaffected.
 */

#include <stdint.h>
#include <stddef.h>

#include <zephyr/toolchain.h> /* for __packed */

#include <nrf_edgeai/rt/nrf_edgeai_model_types.h>
#include <nrf_edgeai/rt/nrf_edgeai_output_types.h>
#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#include <drivers/axon/nrf_axon_nn_infer.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Image format version (independent of the model's own version). */
#define MODEL_IMAGE_FORMAT_VERSION 5

/* Magic {'N','E','I','\0'} = Neuton Edge-ai Image (version is @ref format_version only). */
#define MODEL_IMAGE_MAGIC0 'N'
#define MODEL_IMAGE_MAGIC1 'E'
#define MODEL_IMAGE_MAGIC2 'I'
#define MODEL_IMAGE_MAGIC3 '\0'

/**
 * Precision of the baked model's weights/act_weights, matching MODEL_PARAMS_TYPE in the
 * generated model source. The Neuton loader selects the matching nrf_edgeai_model_neuton_params_*
 * union member when range-checking baked pointers and patching p_neurons.
 */
enum model_image_params_type {
	MODEL_IMAGE_PARAMS_F32 = 0,
	MODEL_IMAGE_PARAMS_Q16 = 1,
	MODEL_IMAGE_PARAMS_Q8 = 2,
	/** Pure Axon compiled model (nrf_axon_nn_compiled_model_s); use @ref axon. */
	MODEL_IMAGE_PARAMS_AXON = 3,
};

/**
 * Map a generated MODEL_PARAMS_TYPE token (the bare identifier f32/q16/q8 the model source uses
 * for `.params.MODEL_PARAMS_TYPE`) to its @ref model_image_params_type value. Used both by the
 * image header stub (to bake @ref model_image_header.params_type) and by the app-side accessor
 * (to fill @ref model_image_neuton_expect.params_type), so both sides agree by construction.
 */
#define MODEL_IMAGE_PARAMS_TYPE_OF(tok)   MODEL_IMAGE_PARAMS_TYPE_OF_(tok)
#define MODEL_IMAGE_PARAMS_TYPE_OF_(tok)  MODEL_IMAGE_PARAMS_TYPE_OF_##tok
#define MODEL_IMAGE_PARAMS_TYPE_OF_f32    MODEL_IMAGE_PARAMS_F32
#define MODEL_IMAGE_PARAMS_TYPE_OF_q16    MODEL_IMAGE_PARAMS_Q16
#define MODEL_IMAGE_PARAMS_TYPE_OF_q8     MODEL_IMAGE_PARAMS_Q8

/** Byte offset of @ref model_image_header.crc32; used by the host CRC patcher. */
#define MODEL_IMAGE_CRC32_OFFSET 20

/** One row in an Axon address-binding table (@ref model_image_axon_backend.binding). */
struct model_image_binding_entry {
	uint32_t name_hash;
	const void *address;
};

/**
 * Neuton backend fields (@ref params_type != @ref MODEL_IMAGE_PARAMS_AXON).
 */
struct model_image_neuton_backend {
	/** DIRECT absolute-flash pointer to the baked nrf_edgeai_model_neuton_t (NOT an offset). */
	const nrf_edgeai_model_neuton_t *model;
	uint8_t task; /**< nrf_edgeai_model_task_t of the baked model */
	uint8_t _pad[3];
	/** DIRECT pointer to the baked decode-output init (NN_DECODED_OUTPUT_INIT). */
	const nrf_edgeai_decoded_output_t *decoded_output;
};

/**
 * Axon backend fields (@ref params_type == @ref MODEL_IMAGE_PARAMS_AXON).
 */
struct model_image_axon_backend {
	/** DIRECT absolute-flash pointer to the baked compiled Axon model (NOT an offset). */
	const nrf_axon_nn_compiled_model_s *model;
	/** Packed-output bytes required by the baked model (0 when unused). */
	uint32_t axon_packed_output_bytes;
	/** Persistent-vars elements (int32_t) required by the baked model. */
	uint32_t persistent_vars_required;
	/** DIRECT absolute-flash pointer to @ref model_image_binding_entry[count] (NOT an offset). */
	const struct model_image_binding_entry *binding;
	/** Number of binding entries (0 when @ref binding is NULL). */
	uint32_t binding_count;
};

/**
 * On-flash model partition image header, placed at offset 0 of the image (== the partition base
 * address) in section ".model_image.header".
 *
 * Layout: shared envelope and metadata first, then a backend union (20 bytes). @ref name points
 * at a NUL-terminated string stored elsewhere in the image (typically .rodata). All pointer
 * fields are absolute flash addresses baked by the linker (the image is linked at the partition
 * base).
 *
 * Field offsets are fixed (pointers are 32-bit on the target) so the host-side CRC patcher
 * (tools/model_ota/patch_image_crc.py) and layout validator can locate @ref crc32 at a constant
 * offset without parsing the struct. @ref __packed is required to forbid compiler padding inside
 * the backend union so the on-flash layout matches the host tools byte-for-byte. Compile-time
 * layout checks live in model_image_common.c.
 */
struct model_image_header {
	uint8_t magic[4];        /**< off 0:  {'N','E','I','\0'} */
	uint16_t format_version; /**< off 4:  MODEL_IMAGE_FORMAT_VERSION */
	uint8_t params_type;     /**< off 6:  enum model_image_params_type */
	uint8_t _reserved;       /**< off 7:  0 */
	uint32_t image_size;     /**< off 8:  bytes from base to __model_image_end */
	uint32_t model_version;  /**< off 12: free-form major.minor.patch */
	uint32_t contract_hash;  /**< off 16: FNV-1a over the firmware ABI contract */
	uint32_t crc32;          /**< off 20: CRC32/IEEE over the image with this field zeroed */
	/** off 24: DIRECT pointer to a NUL-terminated name stored elsewhere in the image. */
	const char *name;
	union {                  /**< off 28 */
		struct model_image_neuton_backend neuton;   /**< 12 B; tail 8 B unused in slot */
		struct model_image_axon_backend axon; /**< 20 B */
	};
} __packed;

/** Return codes for @ref model_image_load_neuton and @ref model_image_load_axon. */
enum model_image_result {
	MODEL_IMAGE_OK = 0,
	MODEL_IMAGE_ERR_NO_PARTITION = -1,
	MODEL_IMAGE_ERR_FLASH_READ = -2,
	MODEL_IMAGE_ERR_BAD_MAGIC = -3,
	MODEL_IMAGE_ERR_BAD_FORMAT_VERSION = -4,
	MODEL_IMAGE_ERR_TOO_LARGE = -5,
	MODEL_IMAGE_ERR_BAD_CRC = -6,
	MODEL_IMAGE_ERR_MODEL_PTR_OUT_OF_RANGE = -7,
	MODEL_IMAGE_ERR_NEURONS_BUF_TOO_SMALL = -8,
	/** Image's task does not match the app's compiled task (@ref model_image_neuton_expect). */
	MODEL_IMAGE_ERR_TASK_MISMATCH = -9,
	/** Image's weight/neuron precision does not match the app's compiled precision. */
	MODEL_IMAGE_ERR_PARAMS_TYPE_MISMATCH = -10,
	/** Image needs more outputs than the app's output buffers can hold. */
	MODEL_IMAGE_ERR_OUTPUTS_TOO_MANY = -11,
	/** A baked descriptor/scale pointer falls outside the image's flash extent. */
	MODEL_IMAGE_ERR_PTR_OUT_OF_RANGE = -12,
	/** Image is not an Axon model (@ref params_type != @ref MODEL_IMAGE_PARAMS_AXON). */
	MODEL_IMAGE_ERR_NOT_AXON_IMAGE = -13,
	/** Loaded Axon model failed nrf_axon_nn_model_validate(). */
	MODEL_IMAGE_ERR_AXON_VALIDATE = -14,
	/** Image @ref params_type is not a supported Neuton precision (f32/q16/q8). */
	MODEL_IMAGE_ERR_BAD_PARAMS_TYPE = -15,
	/** Image @ref contract_hash does not match the app's compiled contract. */
	MODEL_IMAGE_ERR_CONTRACT_MISMATCH = -16,
	/** Image input count does not match the app's compiled pipeline. */
	MODEL_IMAGE_ERR_INPUTS_MISMATCH = -17,
	/** Axon binding address does not match the running firmware. */
	MODEL_IMAGE_ERR_BINDING_MISMATCH = -18,
	/** Image needs more persistent vars than the app allocated. */
	MODEL_IMAGE_ERR_PERSISTENT_VARS_TOO_MANY = -19,
	/** Image needs more packed-output space than the app allocated. */
	MODEL_IMAGE_ERR_PACKED_OUTPUT_TOO_LARGE = -20,
};

/**
 * App-side expectations validated by @ref model_image_load_neuton. These are the compile-time
 * invariants of the *solution* that a mere model update must not break.
 */
struct model_image_neuton_expect {
	uint8_t task;         /**< expected nrf_edgeai_model_task_t of the baked model */
	uint8_t params_type;  /**< expected enum model_image_params_type */
	uint16_t outputs_cap; /**< capacity of the app's output buffers, in elements */
	uint16_t inputs_num;  /**< INPUT_UNIQ_FEATURES_NUM of the compiled solution */
	uint16_t neurons_cap; /**< MODEL_OTA_NEUTON_NEURONS_CAP scratch buffer capacity */
	uint32_t contract_hash; /**< expected @ref model_image_header.contract_hash */
};

/**
 * App-side expectations validated by @ref model_image_load_axon.
 */
struct model_image_axon_expect {
	uint32_t contract_hash;
	uint32_t persistent_vars_cap;
	uint32_t packed_output_cap; /**< allocated bytes; 0 when not allocated in app */
	/** Live binding table from model_ota_axon_keep_refs.S: [count, hash0, addr0, ...]. */
	const uint32_t *binding_table;
};

/**
 * @brief Validate a linked Neuton model partition image and wire it into a runtime context.
 *
 * The partition is assumed to be memory-mapped (XIP): @p partition_addr is dereferenced
 * directly, no payload is copied to RAM. On success the baked descriptor is written into
 * @p edgeai's model instance (via @c edgeai->model.instance), with only @c p_neurons patched
 * to @p neurons_buf, and the image's baked @c NN_DECODED_OUTPUT_INIT is copied into
 * @p edgeai->decoded_output. @p edgeai is untouched on failure.
 *
 * @param[in]  fa_id           Flash area ID of the partition, e.g. FIXED_PARTITION_ID(x).
 * @param[in]  partition_addr  Memory-mapped base address of that same partition.
 * @param[out] edgeai          Runtime context to wire; @c model.instance must already point at
 *                             the caller-owned writable @ref nrf_edgeai_model_neuton_t.
 * @param[out] neurons_buf     Caller-owned RAM scratch for neuron activations.
 * @param[in]  neurons_buf_cap Capacity of neurons_buf, in elements (not bytes).
 * @param[in]  expect          App-side contract and capacity expectations (required).
 * @retval MODEL_IMAGE_OK (0) on success, a negative @ref model_image_result otherwise.
 */
int model_image_load_neuton(uint8_t fa_id, const uint8_t *partition_addr, nrf_edgeai_t *edgeai,
			    void *neurons_buf, size_t neurons_buf_cap,
			    const struct model_image_neuton_expect *expect);

/**
 * @brief Validate a linked Axon model partition image and return its compiled model pointer.
 *
 * The partition is memory-mapped (XIP). App-owned RAM pointers inside the baked model
 * (interlayer buffer, packed output, op extensions) are resolved at model-image link time
 * from zephyr.elf symbol addresses and verified against the running firmware binding table.
 *
 * @param[in]  fa_id           Flash area ID of the partition.
 * @param[in]  partition_addr  Memory-mapped base address of that partition.
 * @param[in]  expect          App-side contract, caps, and binding expectations (required).
 * @param[out] out_model       On success, pointer to the model inside the partition.
 * @retval MODEL_IMAGE_OK (0) on success, a negative @ref model_image_result otherwise.
 */
int model_image_load_axon(uint8_t fa_id, const uint8_t *partition_addr,
			  const struct model_image_axon_expect *expect,
			  const nrf_axon_nn_compiled_model_s **out_model);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_IMAGE_H_ */
