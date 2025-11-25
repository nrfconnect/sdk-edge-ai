nRF Edge AI DSP
================

The DSP module provides a compact, optimized set of primitives for
preprocessing, filtering, feature extraction, time-domain, frequency-domain signal analysis.

Overview
--------

The DSP module implements the following categories of functionality:

- Fast-math helpers
- Time-domain statistics (mean, RMS, variance, min/max, zero-crossings, etc.)
- Spectral transforms (FFT, RFHT) and derived spectral features (centroid, spread)
- Peak finding and frequency-domain utilities (SNR, THD)
- Windowing, quantization and scaling helpers

The API is intentionally small and self-contained so it can be used directly by
model preprocessing code (for example: feature extraction pipelines that feed
inputs to the Edge AI runtime).

Module structure
----------------

Relevant headers are located under ``include/nrf_edgeai/dsp/``. Major groupings:

- Transform (:file:`include/nrf_edgeai/dsp/nrf_dsp_transform.h`)
  - FFT: :file:`include/nrf_edgeai/dsp/transform/nrf_dsp_fft.h`
  - RFHT: :file:`include/nrf_edgeai/dsp/transform/nrf_dsp_rfht.h`
  - Mel-spectrogram: :file:`include/nrf_edgeai/dsp/transform/nrf_dsp_melspectr.h`

- Spectral (:file:`include/nrf_edgeai/dsp/nrf_dsp_spectral.h`)
  - Peak detection: :file:`include/nrf_edgeai/dsp/spectral/nrf_dsp_findpeaks.h`
  - Frequency SNR/THD: :file:`include/nrf_edgeai/dsp/spectral/nrf_dsp_freq_snr.h`,
    :file:`include/nrf_edgeai/dsp/spectral/nrf_dsp_freq_thd.h`
  - Centroid / Spread: :file:`include/nrf_edgeai/dsp/spectral/nrf_dsp_spectral_centroid.h`,
    :file:`include/nrf_edgeai/dsp/spectral/nrf_dsp_spectral_spread.h`

- Statistic (:file:`include/nrf_edgeai/dsp/nrf_dsp_statistic.h`)
  - Common operations (mean, rms, sum, min/max, moments, autocorrelation):
    many helpers under ``include/nrf_edgeai/dsp/statistic/`` (for example
    :file:`nrf_dsp_mean.h`, :file:`nrf_dsp_rms.h`, :file:`nrf_dsp_autocorr.h`)

- Fast Math (:file:`include/nrf_edgeai/dsp/nrf_dsp_fast_math.h`)

- Support / Utilities
  - Windowing, clipping and scaling helpers under
    :file:`include/nrf_edgeai/dsp/support/` and :file:`include/nrf_edgeai/dsp/utils/`.

Types and contexts
------------------

The DSP API exposes a small number of reusable context types that store
intermediate results to avoid recomputation when deriving multiple metrics from
the same data: for example ``nrf_dsp_stat_ctx_f32_t`` and
``nrf_dsp_spectral_ctx_f32_t``. These contexts include precomputed sums,
total-sum-of-squares, variance and other derived metrics.

Key typedefs (examples):

- ``nrf_dsp_stat_ctx_f32_t`` — floating-point statistics context (sum, tss, var, abssum).
- ``nrf_dsp_spectral_ctx_f32_t`` — floating-point spectral context (magnitude sum, centroid).
- ``nrf_dsp_sigma_factor_t`` — sigma factor enum used by some statistic helpers.

Common API patterns
-------------------

Typical usage is to create a context, reset it and call metric helpers to fill
derived values. Example pattern:

.. code-block:: c

   #include <nrf_edgeai/dsp/nrf_dsp.h>

   void compute_features(const float* samples, size_t n)
   {
       nrf_dsp_stat_ctx_f32_t stat_ctx;
       NRF_DSP_STAT_CTX_RESET(stat_ctx);

       /* Compute mean and RMS (API names follow the nrf_dsp_statistic headers) */
       flt32_t mean = nrf_dsp_mean_f32(samples, n, &stat_ctx);
       flt32_t rms  = nrf_dsp_rms_f32(samples, n, &stat_ctx);

       /* Run FFT and compute spectral centroid */
       /* Use FFT helpers under transform/ and spectral/ headers */
   }

Notes on numerical types
------------------------

The DSP module provides both floating-point and fixed-point (int8/int16/int32)
variants where appropriate. The choice of type depends on available hardware
FPU and model quantization strategy. The types and contexts expose explicit
variants for i8/i16/i32 statistics contexts.

Implementation notes
--------------------

- Functions are designed for embedded usage with predictable memory and CPU
  cost. Where possible, the library avoids dynamic memory allocation and
  exposes context-based APIs to reuse scratch buffers.
- FFT and transform implementations include precomputed tables for common
  sizes (see ``transform/fft/`` headers).

See also
--------

- Header files: :file:`include/nrf_edgeai/dsp/` (all included headers are documented)
- Example usage in samples: :ref:`samples_nrf_edgeai`
nRF Edge AI DSP
===============

**DSP Module** (:file:`include/nrf_edgeai/dsp/`)

Provides signal processing functions organized by category:

- **Fast Math** — Basic mathematical operations.
- **Spectral Analysis** — FFT, frequency-domain processing, spectral features.
- **Statistic** — Statistical measures (mean, variance, RMS, entropy, etc.).
- **Transform** — Signal transformations (FFT, RFHT, Mel-spectrogram, etc.).
- **Support** — Utility functions (windowing, quantization, scaling, clipping).