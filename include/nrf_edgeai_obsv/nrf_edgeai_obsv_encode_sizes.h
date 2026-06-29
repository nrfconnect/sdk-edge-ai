/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
/**
 *
 * @defgroup nrf_edgeai_obsv_encode Observability CBOR encoding (buffer sizing)
 * @{
 * @ingroup nrf_edgeai_obsv
 *
 * @brief
 *
 */
#ifndef NRF_EDGEAI_OBSV_ENCODE_SIZES_H
#define NRF_EDGEAI_OBSV_ENCODE_SIZES_H

/**
 * @file nrf_edgeai_obsv_encode_sizes.h
 *
 * Upper bound (in bytes) on the encoded observability payload produced by
 * @c nrf_edgeai_obsv_encode(). The output is CBOR (a compact binary format
 * similar in role to JSON). These numbers are a safe budget for buffer sizing,
 * not an exact wire size. See @c nrf_edgeai_obsv_encode.c for what gets
 * written. Increase the fixed parts if new fields are added to that encoder.
 */

/* --- CBOR encoding primitives --- */

/*
 * These constants are CBOR wire-format budgets, not C struct sizes.
 * CBOR is variable-length; the in-memory structs do not bound encoded size.
 *
 * Helper macros (file-private):
 *
 * _CBOR_TSTR(s)
 *   Encoded size of a text-string literal s whose length is at most 23 bytes.
 *   CBOR represents it as one header byte (0x60 | strlen(s)) followed by the
 *   raw UTF-8 characters. Because sizeof(s) == strlen(s) + 1 (null terminator),
 *   sizeof(s) equals the CBOR encoded size exactly.
 *
 *   WARNING: the string literals used here must exactly match those in
 *   nrf_edgeai_obsv_encode.c. If a key is renamed there, update the
 *   corresponding _CBOR_TSTR() call here too.
 *
 * _CBOR_SMALL_HDR
 *   One-byte map/list header, valid when the element count is <= 23.
 *   In CBOR, the initial byte encodes both the major type (3 high bits)
 *   and the count (5 low bits) when the count fits directly, so no
 *   additional length bytes follow.  Used for fixed-arity maps in the encoder.
 *
 * _CBOR_LARGE_HDR
 *   Three-byte map/list header, valid for element counts up to 65535.
 *   Used wherever the count is a runtime value (metrics list, data rows).
 */
#define _CBOR_TSTR(s)    sizeof(s)
#define _CBOR_SMALL_HDR  1U
#define _CBOR_LARGE_HDR  3U

/** One unsigned 32-bit integer in the stream: 1 CBOR type byte + 4 value bytes. */
#define _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST (1U + sizeof(uint32_t))

/**
 * Fixed outer document overhead, traced from nrf_edgeai_obsv_encode_cbor():
 *
 * @code
 *   map(5) {                              -- _CBOR_SMALL_HDR (count=5 <= 23)
 *     "format_version" : uint32,          -- _CBOR_TSTR + _U32
 *     "num_inferences" : uint32,          -- _CBOR_TSTR + _U32
 *     "num_features"   : uint32,          -- _CBOR_TSTR + _U32
 *     "model" : map(4) {                  -- _CBOR_TSTR + _CBOR_SMALL_HDR
 *       "id"           : uint32,          -- _CBOR_TSTR + _U32
 *       "num_classes"  : uint32,          -- _CBOR_TSTR + _U32
 *       "num_features" : uint32,          -- _CBOR_TSTR + _U32
 *       "version"      : uint32,          -- _CBOR_TSTR + _U32
 *     },
 *     "metrics" : [... metrics ...]       -- _CBOR_TSTR + _CBOR_LARGE_HDR
 *   }
 * @endcode
 */
#define _NRF_EDGEAI_OBSV_ENCODE_OUTER_FIXED                                    \
	(_CBOR_SMALL_HDR                                                       \
	 + _CBOR_TSTR("format_version") + _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST \
	 + _CBOR_TSTR("num_inferences") + _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST \
	 + _CBOR_TSTR("num_features")   + _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST \
	 + _CBOR_TSTR("model") + _CBOR_SMALL_HDR                               \
	   + _CBOR_TSTR("id")           + _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST \
	   + _CBOR_TSTR("num_classes")  + _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST \
	   + _CBOR_TSTR("num_features") + _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST \
	   + _CBOR_TSTR("version")      + _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST \
	 + _CBOR_TSTR("metrics") + _CBOR_LARGE_HDR)

/**
 * Per-metric fixed overhead, traced from encode_metric():
 *
 * @code
 *   map(3) {           -- _CBOR_SMALL_HDR (count=3 <= 23)
 *     "id" : uint32,   -- _CBOR_TSTR + _U32
 *     "v"  : uint32,   -- _CBOR_TSTR + _U32
 *     "d"  : [...]     -- _CBOR_TSTR + _CBOR_LARGE_HDR (rows = runtime value)
 *   }
 * @endcode
 */
#define _NRF_EDGEAI_OBSV_ENCODE_METRIC_BLOCK_FIXED                 \
	(_CBOR_SMALL_HDR                                           \
	 + _CBOR_TSTR("id") + _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST \
	 + _CBOR_TSTR("v")  + _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST \
	 + _CBOR_TSTR("d")  + _CBOR_LARGE_HDR)

/**
 * Per-row overhead: the inner CBOR list header for one row of counters.
 * The column count is a runtime value so the worst-case header is used.
 */
#define _NRF_EDGEAI_OBSV_ENCODE_TABLE_ROW_FIXED _CBOR_LARGE_HDR

/* --- optional metrics (0 when the corresponding Kconfig option is off) --- */

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_DISTRIBUTION)

#define _NRF_EDGEAI_OBSV_ENCODE_PD_PER_CLASS                                         \
	(_NRF_EDGEAI_OBSV_ENCODE_TABLE_ROW_FIXED +                                   \
	 (CONFIG_NRF_EDGEAI_OBSV_PROBS_DISTRIBUTION_BIN_NUM *                        \
	  _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST))

#define _NRF_EDGEAI_OBSV_ENCODE_PD                                                    \
	(_NRF_EDGEAI_OBSV_ENCODE_METRIC_BLOCK_FIXED +                                 \
	 (CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES * _NRF_EDGEAI_OBSV_ENCODE_PD_PER_CLASS))

#else
#define _NRF_EDGEAI_OBSV_ENCODE_PD 0
#endif

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_TRANSITION_MATRIX)

#define _NRF_EDGEAI_OBSV_ENCODE_TM_PER_ROW                                             \
	(_NRF_EDGEAI_OBSV_ENCODE_TABLE_ROW_FIXED +                                     \
	 (CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES * _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST))

#define _NRF_EDGEAI_OBSV_ENCODE_TM                                                     \
	(_NRF_EDGEAI_OBSV_ENCODE_METRIC_BLOCK_FIXED +                                  \
	 (CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES * _NRF_EDGEAI_OBSV_ENCODE_TM_PER_ROW))

#else
#define _NRF_EDGEAI_OBSV_ENCODE_TM 0
#endif

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PREDICTION_SWITCHING_RATE)

/* Prediction switching rate emits a fixed 1 x 2 row: [switches, comparisons]. */
#define _NRF_EDGEAI_OBSV_ENCODE_PSR                                  \
	(_NRF_EDGEAI_OBSV_ENCODE_METRIC_BLOCK_FIXED +                \
	 (_NRF_EDGEAI_OBSV_ENCODE_TABLE_ROW_FIXED +                  \
	  (2U * _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST)))

#else
#define _NRF_EDGEAI_OBSV_ENCODE_PSR 0
#endif

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_ENTROPY_DIST)

/* Entropy distribution emits a single 1 x bin_num histogram row. */
#define _NRF_EDGEAI_OBSV_ENCODE_PED                                            \
	(_NRF_EDGEAI_OBSV_ENCODE_METRIC_BLOCK_FIXED +                          \
	 (_NRF_EDGEAI_OBSV_ENCODE_TABLE_ROW_FIXED +                            \
	  (CONFIG_NRF_EDGEAI_OBSV_PROBS_ENTROPY_DIST_BIN_NUM *                 \
	   _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST)))

#else
#define _NRF_EDGEAI_OBSV_ENCODE_PED 0
#endif

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_TOP2_MARGIN_DIST)

/* Top-2 margin distribution emits a single 1 x bin_num histogram row. */
#define _NRF_EDGEAI_OBSV_ENCODE_PMD                                            \
	(_NRF_EDGEAI_OBSV_ENCODE_METRIC_BLOCK_FIXED +                          \
	 (_NRF_EDGEAI_OBSV_ENCODE_TABLE_ROW_FIXED +                            \
	  (CONFIG_NRF_EDGEAI_OBSV_PROBS_TOP2_MARGIN_DIST_BIN_NUM *             \
	   _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST)))

#else
#define _NRF_EDGEAI_OBSV_ENCODE_PMD 0
#endif

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_ENERGY_DESC)

/* Mel energy descriptor emits 4 rows (mean, max, dynamic range, floor) x bin_num. */
#define _NRF_EDGEAI_OBSV_ENCODE_MED                                            \
	(_NRF_EDGEAI_OBSV_ENCODE_METRIC_BLOCK_FIXED +                          \
	 (4U * (_NRF_EDGEAI_OBSV_ENCODE_TABLE_ROW_FIXED +                      \
		(CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_BIN_NUM *              \
		 _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST))))

#else
#define _NRF_EDGEAI_OBSV_ENCODE_MED 0
#endif

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_SPECTRAL_DESC)

/* Mel spectral descriptor emits 8 rows (band ratios + spectral shape) x bin_num. */
#define _NRF_EDGEAI_OBSV_ENCODE_MSD                                            \
	(_NRF_EDGEAI_OBSV_ENCODE_METRIC_BLOCK_FIXED +                          \
	 (8U * (_NRF_EDGEAI_OBSV_ENCODE_TABLE_ROW_FIXED +                      \
		(CONFIG_NRF_EDGEAI_OBSV_MEL_SPECTRAL_DESC_BIN_NUM *            \
		 _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST))))

#else
#define _NRF_EDGEAI_OBSV_ENCODE_MSD 0
#endif

/**
 * @brief CBOR encoded size budget for one custom metric.
 *
 * Computes the worst-case encoded size (bytes) for a single metric whose
 * @c snapshot() returns @p n_rows rows and @p n_cols columns of @c uint32_t
 * counters. Use this to determine how many extra bytes are needed in the
 * encode buffer for a custom metric, and set
 * @c CONFIG_NRF_EDGEAI_OBSV_EXTRA_ENCODE_BYTES to the sum across all custom
 * metrics. Optionally guard the value with a @c BUILD_ASSERT:
 *
 * @code
 *   BUILD_ASSERT(CONFIG_NRF_EDGEAI_OBSV_EXTRA_ENCODE_BYTES >=
 *                NRF_EDGEAI_OBSV_ENCODE_METRIC_SIZE(MY_ROWS, MY_COLS),
 *                "EXTRA_ENCODE_BYTES too small for custom metric");
 * @endcode
 *
 * @param n_rows Number of rows in the counter matrix (num_rows from snapshot).
 * @param n_cols Number of columns in the counter matrix (num_cols from snapshot).
 */
#define NRF_EDGEAI_OBSV_ENCODE_METRIC_SIZE(n_rows, n_cols)              \
	(_NRF_EDGEAI_OBSV_ENCODE_METRIC_BLOCK_FIXED +                   \
	 (n_rows) * (_NRF_EDGEAI_OBSV_ENCODE_TABLE_ROW_FIXED +          \
		     (n_cols) * _NRF_EDGEAI_OBSV_ENCODE_UINT32_WORST))

/**
 * @brief Largest encoded observability CBOR payload (bytes) for one context.
 *
 * Covers the built-in metrics at their Kconfig dimensions plus the fixed outer
 * document. For custom metrics, set @c CONFIG_NRF_EDGEAI_OBSV_EXTRA_ENCODE_BYTES
 * to the sum of @ref NRF_EDGEAI_OBSV_ENCODE_METRIC_SIZE values for all custom
 * metrics. @ref nrf_edgeai_obsv_encode returns 0 when the buffer is too small.
 */
#define NRF_EDGEAI_OBSV_ENCODE_MAX_SIZE                                          \
	(_NRF_EDGEAI_OBSV_ENCODE_OUTER_FIXED + _NRF_EDGEAI_OBSV_ENCODE_PD +      \
	 _NRF_EDGEAI_OBSV_ENCODE_TM + _NRF_EDGEAI_OBSV_ENCODE_PSR +              \
	 _NRF_EDGEAI_OBSV_ENCODE_PED + _NRF_EDGEAI_OBSV_ENCODE_PMD +             \
	 _NRF_EDGEAI_OBSV_ENCODE_MED + _NRF_EDGEAI_OBSV_ENCODE_MSD +             \
	 CONFIG_NRF_EDGEAI_OBSV_EXTRA_ENCODE_BYTES)

/**
 * @brief CBOR overhead for the outer array header in an obsv-list.
 *
 * @c nrf_edgeai_obsv_encode_list() encodes N contexts as a CBOR array with a
 * single-byte header (@c 0x80|n), valid for n <= 23.
 */
#define _NRF_EDGEAI_OBSV_ENCODE_LIST_ARRAY_OVERHEAD _CBOR_SMALL_HDR

/**
 * @brief Buffer size (bytes) required for an obsv-list encoding of @p n contexts.
 *
 * Use this to size any buffer passed to @ref nrf_edgeai_obsv_encode_list.
 *
 * @param n Number of observability contexts to encode.
 */
#define NRF_EDGEAI_OBSV_ENCODE_LIST_BUF_SIZE(n)                                              \
	(_NRF_EDGEAI_OBSV_ENCODE_LIST_ARRAY_OVERHEAD + (n) * NRF_EDGEAI_OBSV_ENCODE_MAX_SIZE)

/**
 * @}
 */

#endif /* NRF_EDGEAI_OBSV_ENCODE_SIZES_H */
