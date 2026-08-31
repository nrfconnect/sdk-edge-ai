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
 * Each OTA-wired model's payload is dropped from its dedicated static library via archive-scoped
 * linker /DISCARD/ rules (model_ota_edgeai_neuton.cmake, model_ota_edgeai_axon_model()). An Axon-backend
 * Lab solution additionally omits the compiled Axon weights by not including the generated Axon
 * header at all under MODEL_OTA_WIRED. Models compiled directly into the app are unaffected.
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
#define MODEL_IMAGE_FORMAT_VERSION 12

/* Magic {'N','E','I','\0'} = Neuton Edge-ai Image (version is @ref format_version only). */
#define MODEL_IMAGE_MAGIC0 'N'
#define MODEL_IMAGE_MAGIC1 'E'
#define MODEL_IMAGE_MAGIC2 'I'
#define MODEL_IMAGE_MAGIC3 '\0'

/**
 * Precision of the baked model's weights/act_weights, matching MODEL_PARAMS_TYPE in the
 * generated model source. The Neuton loader selects the matching nrf_edgeai_model_neuton_params_*
 * union member when patching p_neurons.
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
 *
 * The rest of the backend union slot is left zeroed for a Neuton image.
 */
struct model_image_neuton_backend {
	/** DIRECT absolute-flash pointer to the baked nrf_edgeai_model_neuton_t (NOT an offset). */
	const nrf_edgeai_model_neuton_t *model;
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
 * The model's share of nrf_edgeai_t, carried by the image so a model update can change it
 * together with the weights.
 *
 * These are the values the runtime keeps outside the backend model descriptor, which is why they
 * live beside the backend union rather than inside it. They are stored by value, in the runtime's
 * own types, so applying them is a plain assignment (see
 * lib/model_ota/model_image_edgeai_params.c). Every pointer inside is an absolute flash address
 * baked by the linker.
 */
struct model_image_edgeai_params {
	/**
	 * Feature scaling factors. A solution scales exactly once on the path into the network,
	 * and which stage does it follows from the solution itself: with a DSP feature pipeline
	 * the extracted features are scaled (nrf_edgeai_process_features_dsp_*), without one the
	 * raw input features are (nrf_edgeai_process_features_scale_vector_*). The live member is
	 * therefore not recorded here - nrf_edgeai_t.p_dsp tells the loader which one to read.
	 */
	union {
		/** INPUT_FEATURES_SCALE_MIN / _MAX -> nrf_edgeai_t.input.scale. */
		nrf_edgeai_input_scale_t input;
		/**
		 * EXTRACTED_FEATURES_SCALE_MIN / _MAX and FEATURES_EXTRACTION_ARGUMENTS ->
		 * nrf_edgeai_t.p_dsp->features.meta.
		 */
		nrf_edgeai_features_meta_t features;
	} scale;

	/** Baked NN_DECODED_OUTPUT_INIT -> nrf_edgeai_t.decoded_output. */
	nrf_edgeai_decoded_output_t decoded_output;

	/**
	 * FEATURES_EXTRACTION_MASK[INPUT_UNIQ_FEATURES_NUM], carried for verification rather than
	 * to be applied: the loader compares it against nrf_edgeai_t.p_dsp->features.p_masks and
	 * rejects a mismatch. NULL when the solution has no DSP pipeline.
	 *
	 * This is the artifact that decides which features are extracted, in which order, per
	 * unique input feature, and therefore how the flat @ref scale arrays are indexed. It is
	 * verified here rather than folded into @ref model_image_header.contract_hash because the
	 * preprocessor cannot hash an array, and because comparing it names the input feature whose
	 * mask diverged instead of reporting an opaque contract mismatch.
	 */
	const nrf_edgeai_features_mask_t *p_extraction_mask;

	/** Elements in each scaling array; 0 when the image carries no parameters at all. */
	uint16_t scale_num;
	/**
	 * Bytes per scaling element. Stored explicitly because the runtime has no field
	 * describing the element size of the DSP feature meta arrays.
	 */
	uint8_t scale_elem_size;
	uint8_t _reserved;
};

/**
 * On-flash model partition image header, placed at offset 0 of the image (== the partition base
 * address) in section ".model_image.header".
 *
 * Layout: shared envelope and metadata first, then a backend union (20 bytes), then the shared
 * nrf_edgeai_t parameter block (36 bytes). @ref name points at a NUL-terminated string stored
 * elsewhere in the image (typically .rodata). All pointer fields are absolute flash addresses
 * baked by the linker (the image is linked at the partition base).
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
		struct model_image_neuton_backend neuton;   /**< 4 B; tail 16 B unused in slot */
		struct model_image_axon_backend axon; /**< 20 B */
	};
	/**
	 * off 48: the model's share of nrf_edgeai_t, shared by both backends. Left zeroed
	 * (@c scale_num == 0) when the image carries none, i.e. the application keeps its
	 * compiled-in values - which is the case for a pure Axon model, having no
	 * nrf_edgeai_t at all.
	 */
	struct model_image_edgeai_params edgeai_params;
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
	/* -7 (model pointer out of range) retired: see -12. */
	MODEL_IMAGE_ERR_NEURONS_BUF_TOO_SMALL = -8,
	/* -9 (task mismatch) retired: the task is part of @ref contract_hash for both backends. */
	/** Image's weight/neuron precision does not match the app's compiled precision. */
	MODEL_IMAGE_ERR_PARAMS_TYPE_MISMATCH = -10,
	/* -11 (too many outputs) retired: the output count is part of @ref contract_hash. */
	/*
	 * -12 (pointer out of range) retired, with -7: the partition base every image pointer was
	 * linked against is part of @ref contract_hash, and containment within the image is gated
	 * at build time by tools/model_ota/validate_model_image_layout.py.
	 */
	/** Image is not an Axon model (@ref params_type != @ref MODEL_IMAGE_PARAMS_AXON). */
	MODEL_IMAGE_ERR_NOT_AXON_IMAGE = -13,
	/** Loaded Axon model failed nrf_axon_nn_model_validate(). */
	MODEL_IMAGE_ERR_AXON_VALIDATE = -14,
	/** Image @ref params_type is not a supported Neuton precision (f32/q16/q8). */
	MODEL_IMAGE_ERR_BAD_PARAMS_TYPE = -15,
	/** Image @ref contract_hash does not match the app's compiled contract. */
	MODEL_IMAGE_ERR_CONTRACT_MISMATCH = -16,
	/* -17 (input count mismatch) retired: the input count is part of @ref contract_hash. */
	/** Axon binding address does not match the running firmware. */
	MODEL_IMAGE_ERR_BINDING_MISMATCH = -18,
	/** Image needs more persistent vars than the app allocated. */
	MODEL_IMAGE_ERR_PERSISTENT_VARS_TOO_MANY = -19,
	/** Image needs more packed-output space than the app allocated. */
	MODEL_IMAGE_ERR_PACKED_OUTPUT_TOO_LARGE = -20,
	/* -21 (scale layout mismatch) retired: the layout is part of @ref contract_hash. */
	/**
	 * Image's FEATURES_EXTRACTION_MASK differs from the application's compiled-in one, so the
	 * app's pipeline would index the image's flat feature arrays with the wrong per-slot
	 * meaning. See @ref model_image_edgeai_params.p_extraction_mask.
	 */
	MODEL_IMAGE_ERR_DSP_MASK_MISMATCH = -22,
};

/**
 * App-side expectations validated by @ref model_image_load_neuton.
 *
 * Only what the contract hash cannot express: the neuron scratch *capacity* (an inequality, so
 * that an oversized model reports "needs new firmware" rather than "incompatible"), and the
 * weight precision. Precision is deliberately redundant with the hash - it selects the
 * nrf_edgeai_model_neuton_params_* union member and therefore the element size of the caller's
 * neuron buffer, so it is worth re-checking directly rather than trusting a 32-bit hash with a
 * memory-safety property.
 */
struct model_image_neuton_expect {
	uint8_t params_type;  /**< expected enum model_image_params_type */
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
 * @p edgeai's model instance (via @c edgeai->model.instance), with only @c p_neurons patched to
 * @p neurons_buf. @p edgeai is untouched on failure.
 *
 * The rest of @p edgeai - the feature scaling factors and the decoded-output init - comes from
 * @ref model_image_header.edgeai_params and is applied separately by
 * @c model_image_bind_edgeai_params(), which the wired translation unit calls next.
 *
 * @param[in]  partition_addr  Memory-mapped base address of a zephyr,mapped-partition node,
 *                             e.g. PARTITION_ADDRESS(model_storage).
 * @param[in]  partition_size  Size of that partition, in bytes, e.g. PARTITION_SIZE(model_storage).
 * @param[out] edgeai          Runtime context to wire; @c model.instance must already point at
 *                             the caller-owned writable @ref nrf_edgeai_model_neuton_t.
 * @param[out] neurons_buf     Caller-owned RAM scratch for neuron activations.
 * @param[in]  neurons_buf_cap Capacity of neurons_buf, in elements (not bytes).
 * @param[in]  expect          App-side contract and capacity expectations (required).
 * @retval MODEL_IMAGE_OK (0) on success, a negative @ref model_image_result otherwise.
 */
enum model_image_result model_image_load_neuton(const uint8_t *partition_addr,
						size_t partition_size, nrf_edgeai_t *edgeai,
						void *neurons_buf, size_t neurons_buf_cap,
						const struct model_image_neuton_expect *expect);

/**
 * @brief Apply the nrf_edgeai_t parameters carried by a model partition image.
 *
 * These are the values that live in nrf_edgeai_t rather than in the backend model descriptor and
 * so have to travel with the model: the feature scaling factors and the decoded-output init (see
 * @ref model_image_edgeai_params). They are stored in the image in the runtime's own types, so
 * applying them is a plain assignment.
 *
 * Which scaling stage the image speaks to is not recorded in the image: it follows from the
 * solution, so it is read off @p edgeai (a context with a DSP pipeline scales its extracted
 * features, one without scales its raw input features).
 *
 * For a solution with a DSP pipeline this also verifies the image's
 * @ref model_image_edgeai_params.p_extraction_mask against the application's compiled-in
 * FEATURES_EXTRACTION_MASK - the one part of the DSP contract that cannot be hashed - and applies
 * nothing if they disagree.
 *
 * The block sits outside the backend union, so this works for either backend: call it after a
 * successful @ref model_image_load_neuton or @ref model_image_load_axon on the same partition.
 * The image is taken to be already validated by that call: the scaling layout the block uses and
 * the partition base its pointers were linked at are both covered by
 * @ref model_image_header.contract_hash.
 *
 * @param[in]  partition_addr Memory-mapped base address of the validated partition.
 * @param[out] edgeai         Runtime context to fill; untouched on failure.
 * @retval MODEL_IMAGE_OK (0) on success, a negative @ref model_image_result otherwise.
 */
enum model_image_result model_image_bind_edgeai_params(const uint8_t *partition_addr,
						       nrf_edgeai_t *edgeai);

/**
 * @brief Validate a linked Axon model partition image and return its compiled model pointer.
 *
 * The partition is memory-mapped (XIP). App-owned RAM pointers inside the baked model
 * (interlayer buffer, packed output, op extensions) are resolved at model-image link time
 * from zephyr.elf symbol addresses and verified against the running firmware binding table.
 *
 * An Axon-backed Lab solution carries its nrf_edgeai_t parameters in
 * @ref model_image_header.edgeai_params, applied by @ref model_image_bind_edgeai_params() just
 * like a Neuton one; a pure Axon model has no nrf_edgeai_t and leaves that block zeroed.
 *
 * @param[in]  partition_addr  Memory-mapped base address of a zephyr,mapped-partition node.
 * @param[in]  partition_size  Size of that partition, in bytes.
 * @param[in]  expect          App-side contract, caps, and binding expectations (required).
 * @param[out] out_model       On success, pointer to the model inside the partition.
 * @retval MODEL_IMAGE_OK (0) on success, a negative @ref model_image_result otherwise.
 */
enum model_image_result model_image_load_axon(const uint8_t *partition_addr, size_t partition_size,
					      const struct model_image_axon_expect *expect,
					      const nrf_axon_nn_compiled_model_s **out_model);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_IMAGE_H_ */
