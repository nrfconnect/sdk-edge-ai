.. _nrf_edgeai_lib_dsp:

nRF Edge AI DSP
###############

.. contents::
   :local:
   :depth: 2

The Digital Signal Processing (DSP) module provides a compact, optimized set of primitives for preprocessing, filtering, feature extraction, time-domain, frequency-domain signal analysis.

Overview
********

The DSP module implements the following categories of functionality:

* Basic mathematical operations
* FFT, frequency-domain processing, spectral features
* Statistical measures (for example, mean, variance, RMS, entropy)
* Signal transformations (for example, FFT, RFHT, Mel-spectrogram)
* Utility functions (windowing, quantization, scaling, clipping)

All functions are designed for embedded usage with predictable memory and CPU usage.
The library avoids dynamic memory allocation and exposes context-based APIs to enable scratch buffer reuse.

FFT and transform functions include precomputed tables for common input sizes.
See the headers in :file:`transform/fft/` for details.

Module structure
****************

Files are located in the :file:`include/nrf_edgeai/dsp/` directory, grouped by functionality:

* :file:`include/nrf_edgeai/dsp/nrf_dsp_transform.h` - Aggregated interface for signal transformations

  * :file:`include/nrf_edgeai/dsp/transform/nrf_dsp_fft.h` - Fast Fourier Transform (FFT) signal transforms
  * :file:`include/nrf_edgeai/dsp/transform/nrf_dsp_rfht.h` - Real Fast Hartley Transform (RFHT) signal transforms
  * :file:`include/nrf_edgeai/dsp/transform/nrf_dsp_melspectr.h` - Mel-spectrogram transforms

* :file:`include/nrf_edgeai/dsp/nrf_dsp_spectral.h` - Aggregated interface for spectral analysis

  * :file:`include/nrf_edgeai/dsp/spectral/nrf_dsp_findpeaks.h` - Peak detection helpers
  * :file:`include/nrf_edgeai/dsp/spectral/nrf_dsp_freq_snr.h` - Frequency SNR computations
  * :file:`include/nrf_edgeai/dsp/spectral/nrf_dsp_freq_thd.h` - Frequency THD computations
  * :file:`include/nrf_edgeai/dsp/spectral/nrf_dsp_spectral_centroid.h` - Spectral centroid calculations
  * :file:`include/nrf_edgeai/dsp/spectral/nrf_dsp_spectral_spread.h` - Spectral spread calculations

* :file:`include/nrf_edgeai/dsp/nrf_dsp_statistic.h` - Aggregated interface for statistical operations

  * :file:`include/nrf_edgeai/dsp/statistic/nrf_dsp_mean.h` - Mean value calculations
  * :file:`include/nrf_edgeai/dsp/statistic/nrf_dsp_rms.h` - RMS calculations
  * :file:`include/nrf_edgeai/dsp/statistic/nrf_dsp_autocorr.h` - Autocorrelation functions

* :file:`include/nrf_edgeai/dsp/nrf_dsp_fast_math.h` - Fast math helper functions
* :file:`include/nrf_edgeai/dsp/support/` - Utility functions for windowing, quantization, clipping, and scaling
* :file:`include/nrf_edgeai/dsp/utils/` - Additional utility functions

Types and contexts
==================

The DSP API provides a small set of reusable context types that store intermediate results and eliminate redundant computation when deriving multiple metrics from the same data.
For example, :c:func:`nrf_dsp_stat_ctx_f32_t` and :c:func:`nrf_dsp_spectral_ctx_f32_t` contexts hold precomputed sums, sum-of-squares, variance, and other derived metrics.

Key typedefs include:

* :c:func:`nrf_dsp_stat_ctx_f32_t` — Floating-point statistics context (sum, tss, var, abssum)
* :c:func:`nrf_dsp_spectral_ctx_f32_t` — Floating-point spectral context (magnitude sum, centroid)
* :c:func:`nrf_dsp_sigma_factor_t` — Sigma factor enum used by statistical helpers

Usage pattern
-------------

A typical usage pattern is to create a context, reset it, and then call metric helper functions to compute derived values.
For example:

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

The DSP module offers both floating-point and fixed-point (int8, int16, int32) variants where appropriate.
The choice depends on hardware FPU availability and model quantization requirements.
Types and contexts expose explicit variants for i8, i16, i32 statistics contexts.
