/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef MODEL_OTA_MODEL_CONTRACT_H_
#define MODEL_OTA_MODEL_CONTRACT_H_

/* TODO: resolve part of issues from security review on confluence */

/**
 * @file
 * @brief Firmware ABI contract hashing for model-only OTA images.
 *
 * Image and application must compute the same @ref model_image_header.contract_hash from
 * compile-time constants. The hash is FNV-1a over typed u32 chunks (little-endian identity on
 * the target), finished with the MurmurHash3 fmix32 avalanche.
 *
 * FNV-1a rather than a MurmurHash3 body mix because the accumulator appears exactly once per
 * step, so a chain of N chunks written as nested macros grows linearly. A MurmurHash3 step
 * expands its accumulator twice (the rotate), which doubles the token count per chunk and is
 * not viable for the ~22-chunk chains below.
 *
 * What goes in here and what does not
 * -----------------------------------
 * Only *identity* invariants belong in the hash: things that must be equal on both sides or the
 * image is meaningless (task, precision, struct layouts, pipeline shape, partition base).
 *
 * *Capability* values must NOT be hashed. Those are the ones with a "the image requires X, the
 * firmware provides Y, X <= Y is fine" relation: the Neuton neuron scratch capacity, the Axon
 * persistent-vars and packed-output caps. They live in @ref model_image_header and are checked
 * by inequality at load time and by tools/model_ota/check_model_compat.py, which is what
 * produces the actionable "model exceeds firmware caps, ship new firmware" verdict. Folding any
 * of them into the hash collapses that verdict into an opaque contract mismatch.
 */

#include <stdint.h>

#include <model_ota/model_image.h>

#include <nrf_edgeai/rt/nrf_edgeai_model_types.h>
#include <nrf_edgeai/rt/nrf_edgeai_output_types.h>
#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#include <drivers/axon/nrf_axon_nn_infer.h>

#ifdef __cplusplus
extern "C" {
#endif

/** FNV-1a 32-bit offset basis. */
#define MODEL_OTA_HASH_SEED 2166136261u

/** FNV-1a 32-bit prime. */
#define MODEL_OTA_HASH_PRIME 16777619u

/** One FNV-1a step over a u32 chunk. Usable in static initializers. */
#define MODEL_OTA_HASH_U32(h, k) (((uint32_t)(h) ^ (uint32_t)(k)) * MODEL_OTA_HASH_PRIME)

/* Fold a fixed number of chunks into @p h, left to right; sugar to keep the chains readable. */
#define MODEL_OTA_HASH_U32_2(h, a, b) MODEL_OTA_HASH_U32(MODEL_OTA_HASH_U32((h), (a)), (b))
#define MODEL_OTA_HASH_U32_3(h, a, b, c)                                                           \
	MODEL_OTA_HASH_U32(MODEL_OTA_HASH_U32_2((h), (a), (b)), (c))
#define MODEL_OTA_HASH_U32_4(h, a, b, c, d)                                                        \
	MODEL_OTA_HASH_U32_2(MODEL_OTA_HASH_U32_2((h), (a), (b)), (c), (d))
#define MODEL_OTA_HASH_U32_5(h, a, b, c, d, e)                                                     \
	MODEL_OTA_HASH_U32(MODEL_OTA_HASH_U32_4((h), (a), (b), (c), (d)), (e))
#define MODEL_OTA_HASH_U32_8(h, a, b, c, d, e, f, g, i)                                            \
	MODEL_OTA_HASH_U32_4(MODEL_OTA_HASH_U32_4((h), (a), (b), (c), (d)), (e), (f), (g), (i))
#define MODEL_OTA_HASH_U32_16(h, a, b, c, d, e, f, g, i, j, k, l, m, n, o, p, q)                   \
	MODEL_OTA_HASH_U32_8(MODEL_OTA_HASH_U32_8((h), (a), (b), (c), (d), (e), (f), (g), (i)),    \
			     (j), (k), (l), (m), (n), (o), (p), (q))

#define MODEL_OTA_FMX32_1(h) ((h) ^ ((h) >> 16u))
#define MODEL_OTA_FMX32_2(h) ((h) * 0x85ebca6bu)
#define MODEL_OTA_FMX32_3(h) ((h) ^ ((h) >> 13u))
#define MODEL_OTA_FMX32_4(h) ((h) * 0xc2b2ae35u)
#define MODEL_OTA_FMX32_5(h) ((h) ^ ((h) >> 16u))

/**
 * MurmurHash3 fmix32 avalanche; apply once after the last @ref MODEL_OTA_HASH_U32 step.
 *
 * FNV-1a only propagates carries upwards, so without this a difference in the last chunk's high
 * bits would barely move the low bits of the result.
 */
#define MODEL_OTA_HASH_FINAL(h)                                                                    \
	MODEL_OTA_FMX32_5(MODEL_OTA_FMX32_4(                                                       \
		MODEL_OTA_FMX32_3(MODEL_OTA_FMX32_2(MODEL_OTA_FMX32_1((uint32_t)(h))))))

/**
 * Envelope common to every flavor: image format and the flash base the image was linked at.
 *
 * @p image_base ties the image to one partition. Every intra-image pointer is an absolute flash
 * address baked at link time, so an image linked for a different partition is not merely a
 * different model - its pointers are wrong. In a multi-model build it also stops an image being
 * accepted by the wrong slot when the two slots happen to agree on everything else.
 */
#define MODEL_OTA_HASH_ENVELOPE(h, image_base)                                                     \
	MODEL_OTA_HASH_U32_2((h), MODEL_IMAGE_FORMAT_VERSION, (uint32_t)(uintptr_t)(image_base))

/**
 * The DSP feature-extraction contract, folded into the solution mix as a single digest.
 *
 * What the image actually carries of the DSP stage is three flat arrays - @c p_min, @c p_max and
 * @c p_arguments - while the extraction code (the timedomain/freqdomain pipelines and the twiddle
 * tables) stays compiled into the application. So the contract does not have to describe the
 * pipeline; it has to pin whatever decides the *layout and per-slot meaning* of those arrays.
 *
 * @p args_bytes / @p args_elem_size  Extent and element width of FEATURES_EXTRACTION_ARGUMENTS.
 *   Both are 0 for a pipeline that takes no arguments. The element width is not implied by
 *   @c scale_elem_size: the arguments are typed @c nrf_user_input_t while a DSP solution's scaling
 *   arrays are @c nrf_user_feature_t, and the loader copies raw bytes.
 * @p spectrum_len / @p rfft_len / @p bitrev_len  FFT geometry, or 0 when the solution has no
 *   frequency-domain pipeline. These select which twiddle and bit-reversal tables the application
 *   compiled in; without them a solution could keep the same window and feature count while
 *   moving to a different spectrum length, and the application would run the new argument set
 *   through the wrong-length FFT. The table *contents* need not be hashed - they are a pure
 *   function of these lengths.
 *
 * Which features are extracted, in which order, per input feature is decided by
 * FEATURES_EXTRACTION_MASK, which the preprocessor cannot fold. It is not hashed: the image
 * carries the array itself in @ref model_image_edgeai_params.p_extraction_mask and the loader
 * compares it word by word against the application's, which reports *which* input feature's mask
 * diverged instead of an opaque contract mismatch.
 */
#define MODEL_OTA_HASH_DSP_MIX(h, args_bytes, args_elem_size, spectrum_len, rfft_len, bitrev_len)  \
	MODEL_OTA_HASH_U32_5((h), (args_bytes), (args_elem_size), (spectrum_len), (rfft_len),       \
			     (bitrev_len))

/**
 * Standalone digest of @ref MODEL_OTA_HASH_DSP_MIX, for solutions that have a DSP pipeline.
 *
 * Solutions without one contribute 0 instead (see MODEL_OTA_DSP_CONTRACT_HASH in
 * lib/model_ota/src/model_ota_scale_select.h), so the solution mix keeps a fixed chunk count.
 */
#define MODEL_OTA_CONTRACT_HASH_DSP(args_bytes, args_elem_size, spectrum_len, rfft_len,            \
				    bitrev_len)                                                    \
	MODEL_OTA_HASH_FINAL(MODEL_OTA_HASH_DSP_MIX(MODEL_OTA_HASH_SEED, (args_bytes),              \
						   (args_elem_size), (spectrum_len), (rfft_len),   \
						   (bitrev_len)))

/**
 * The Edge AI Lab *solution* contract, shared by every nrf_edgeai_t-wrapped flavor (Neuton and
 * Axon backends alike). This is what a mere retrain must not change.
 *
 * Callers pass @ref MODEL_OTA_SOLUTION_CONTRACT_ARGS (lib/model_ota/src/model_ota_scale_select.h)
 * rather than spelling the fourteen arguments out; the two @c sizeof chunks are added here
 * because they are properties of the runtime, not of the generated solution.
 *
 * TODO: two open questions to Neuton on the scope of this mix, to settle before the format is
 * frozen.
 *
 *   1. @p solution_id_hash. Included for now, which makes the hash identify *this* solution
 *      rather than "any solution with this shape". Will the retrained models be based on the same
 *      solution or do we allow different solution when all params match?
 *
 *   2. How much of the DSP pipeline the contract should describe at all. What it pins today is
 *      @p input_feature_type, @p inputs_num, @p window_size, @p window_shift, @p subwindow_num,
 *      @p extracted_features_num, @p uses_as_input_mask, @p scale_num, @p scale_elem_size and
 *      @p dsp_hash (@ref MODEL_OTA_HASH_DSP_MIX: extent and element width of
 *      FEATURES_EXTRACTION_ARGUMENTS plus the spectrum, rfft and bit-reversal lengths). On top of
 *      that, FEATURES_EXTRACTION_MASK - which decides which features are extracted and in what
 *      order, per input feature - is not hashed but travels in the image and is compared word by
 *      word at load time (@ref model_image_edgeai_params.p_extraction_mask), so a divergence
 *      names the offending input feature instead of reporting an opaque mismatch. Neither catches
 *      "same mask, different math", i.e. a Lab release that changes what
 *      @c nrf_edgeai_feature_mad_f32 computes; @p runtime_version is the only defence there.
 *      Do we keep this descriptive split, or does @p solution_id_hash (question 1) already
 *      subsume it, leaving the rest as extra format surface and false mismatches?
 */
#define MODEL_OTA_HASH_SOLUTION_MIX(h, solution_id_hash, runtime_version, task, outputs_num,       \
				    scale_num, scale_elem_size, input_feature_type, inputs_num,    \
				    window_size, window_shift, subwindow_num,                      \
				    extracted_features_num, uses_as_input_mask, dsp_hash)          \
	MODEL_OTA_HASH_U32_16((h), (solution_id_hash), (runtime_version), (task), (outputs_num),   \
			      ((uint32_t)sizeof(nrf_edgeai_decoded_output_t)),                      \
			      ((uint32_t)sizeof(struct model_image_edgeai_params)), (scale_num),   \
			      (scale_elem_size), (input_feature_type), (inputs_num),               \
			      (window_size), (window_shift), (subwindow_num),                      \
			      (extracted_features_num), (uses_as_input_mask), (dsp_hash))

/** Neuton descriptor ABI: what the loader copies out of the image and hands to the engine. */
#define MODEL_OTA_HASH_NEUTON_MIX(h, params_type)                                                  \
	MODEL_OTA_HASH_U32_3((h), (params_type),                                                   \
			     ((uint32_t)sizeof(nrf_edgeai_model_neuton_t)),                        \
			     ((uint32_t)sizeof(nrf_nn_neuton_model_meta_t)))

/**
 * Axon driver ABI. The interlayer and psum sizes are firmware-wide Kconfig values the compiled
 * model was built against, not per-model requirements, so equality is the right relation.
 */
#define MODEL_OTA_HASH_AXON_MIX(h)                                                                 \
	MODEL_OTA_HASH_U32_3((h), ((uint32_t)sizeof(nrf_axon_nn_compiled_model_s)),                \
			     (uint32_t)CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE,                     \
			     (uint32_t)CONFIG_NRF_AXON_PSUM_BUFFER_SIZE)

/**
 * Full Edge AI Lab / Neuton contract hash.
 *
 * @p image_base   Flash base the image is linked at (NRF_MODEL_PARTITION_ADDR).
 * @p params_type  @ref model_image_params_type (0/1/2).
 * @p ...          @ref MODEL_OTA_SOLUTION_CONTRACT_ARGS.
 *
 * The neuron scratch capacity is deliberately absent: it is an application capacity, checked by
 * inequality in model_image_load_neuton().
 */
#define MODEL_OTA_CONTRACT_HASH_EDGEAI_NEUTON(image_base, params_type, ...)                        \
	MODEL_OTA_HASH_FINAL(MODEL_OTA_HASH_SOLUTION_MIX(                                          \
		MODEL_OTA_HASH_NEUTON_MIX(                                                         \
			MODEL_OTA_HASH_ENVELOPE(MODEL_OTA_HASH_SEED, (image_base)),                 \
			(params_type)),                                                            \
		__VA_ARGS__))

/**
 * Full contract hash for an Edge AI Lab solution with an Axon backend.
 *
 * A pure Axon image can never be accepted by a wrapped solution's slot (or the reverse): the two
 * differ by the sixteen solution chunks, which the final avalanche spreads over the whole word.
 * That matters because such an image carries no @ref model_image_edgeai_params, while the wrapped
 * application discarded its own compiled-in copy when it was wired for OTA.
 *
 * @p image_base  Flash base the image is linked at (NRF_MODEL_PARTITION_ADDR).
 * @p ...         @ref MODEL_OTA_SOLUTION_CONTRACT_ARGS.
 */
#define MODEL_OTA_CONTRACT_HASH_EDGEAI_AXON(image_base, ...)                                       \
	MODEL_OTA_HASH_FINAL(MODEL_OTA_HASH_SOLUTION_MIX(                                          \
		MODEL_OTA_HASH_AXON_MIX(                                                           \
			MODEL_OTA_HASH_ENVELOPE(MODEL_OTA_HASH_SEED, (image_base))),                \
		__VA_ARGS__))

/**
 * Full pure-Axon contract hash: a raw compiled model with no nrf_edgeai_t around it.
 *
 * @p image_base  Flash base the image is linked at (NRF_MODEL_PARTITION_ADDR).
 *
 * The model's persistent-vars and packed-output requirements are deliberately absent. They are
 * per-model requirements already carried in @ref model_image_axon_backend and checked by
 * inequality against the application caps; hashing either the requirement or the cap would turn
 * a retrain that still fits into a hard mismatch. See the file comment.
 */
#define MODEL_OTA_CONTRACT_HASH_AXON(image_base)                                                   \
	MODEL_OTA_HASH_FINAL(                                                                      \
		MODEL_OTA_HASH_AXON_MIX(MODEL_OTA_HASH_ENVELOPE(MODEL_OTA_HASH_SEED, (image_base))))

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_CONTRACT_H_ */
