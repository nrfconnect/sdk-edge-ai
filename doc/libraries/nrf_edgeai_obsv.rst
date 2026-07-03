.. _nrf_edgeai_obsv_lib:

nRF Edge AI Observability Library
#################################

.. contents::
   :local:
   :depth: 2

The Edge AI Observability module tracks how a classification model performs at runtime.
It collects stats from the model's output probabilities and packages them as metric snapshots that can be sent to a monitoring backend.
It works with any inference engine that produces a probability vector, including the :ref:`nrf_edgeai_lib`, :ref:`Axon NPU <lib_axon>`, and `Edge Impulse`_ deployments.

Overview
********

Deployed Edge AI models behave differently in production then in controlled environments.
Class distributions shift, transitions between predicted labels change over time, and confidence scores drift as conditions change.
The observability module gives you a structured way to capture these statistics on-device and send them off-device for analysis.

Metrics are driven by two input streams.
Output metrics consume the model's class-probability vector passed to :c:func:`nrf_edgeai_obsv_update_probs`, while input-feature metrics consume the extracted feature vector fed to the model and passed to :c:func:`nrf_edgeai_obsv_update_features`.
Each metric declares which stream it consumes through its ``source`` field, and the library routes every update only to the metrics that match.

Collected data enables the following:

* Model quality monitoring - Allows tracking whether prediction confidence and class frequencies stay within expected bounds after deployment.
* Dataset collection guidance - Helps identify which classes are under-represented or confused in the field, and target data collection efforts accordingly.
* Retraining triggers - Allows detecting distribution shift early and decide when a model update is needed before accuracy degrades noticeably.
* A/B testing - Allows comparing metric snapshots from devices running different model versions to evaluate improvements in production conditions.

The module is organized as three cooperating layers:

* Core (:file:`lib/nrf_edgeai_obsv/`) - A portable, mutex-free state machine that accumulates metric counters as inference results arrive.
  It has no Zephyr RTOS dependency and you can use it in bare-metal environments, other RTOSes, or host-side test builds.
* Zephyr wrapper (:file:`lib/nrf_edgeai_obsv/`) - Wraps the core in a mutex-protected context so that multiple threads can feed inferences and trigger encoding without data races, and integrates the library into the Zephyr build system (CMake, Kconfig, logging).
* Memfault CDR transport (:file:`lib/nrf_edgeai_obsv_memfault/`) - Encodes the accumulated metric snapshots as a CBOR blob and stages them as a `Memfault Custom Data Recording`_ (CDR) that the Memfault SDK packetizer uploads on the next transport drain cycle.
  For Memfault Kconfig, keys, and transports in |NCS|, see `Memfault in nRF Connect SDK`_.

.. uml::
   :caption: High-level data flow from application inference through observability to nRF Cloud and downstream tools.

   skinparam shadowing false
   skinparam roundcorner 0
   skinparam backgroundColor #FFFFFF
   skinparam defaultTextAlignment center
   skinparam ArrowColor #0077C8
   skinparam ArrowThickness 1
   skinparam linetype ortho
   skinparam componentStyle rectangle

   skinparam component {
     BackgroundColor #13B6FF
     BorderColor #13B6FF
     FontColor #333F48
   }

   skinparam cloud {
     BackgroundColor #C1E8FF
     BorderColor #2149C2
     FontColor #333F48
   }

   together {
     component "Application" as App
     component "nrf_edgeai_obsv\n(Zephyr wrapper + core)" as Obsv
     component "Metrics\n(e.g. probability distribution)" as Metrics
   }

   component "nrf_edgeai_obsv_memfault\n(Memfault CDR transport)" as MfltTransport
   component "Memfault SDK" as MfltSDK
   cloud "nRF Cloud\n(Memfault)" as Cloud
   component "Monitoring tool\n(dashboard / ML pipeline)" as Dashboard

   App -right-> Obsv : inference results\n(class probabilities)
   Obsv -down-> Metrics : accumulate counters
   App -down-> MfltTransport : trigger collect
   MfltTransport -up-> Obsv : encode metrics as CBOR
   MfltTransport -right-> MfltSDK : stage CDR
   MfltSDK -right-> Cloud : upload via BLE or HTTP
   Cloud -right-> Dashboard : fetch CDR\n(REST API)

The following diagram shows the detailed call sequence between the observability layers, the application, and the Memfault SDK.

.. uml::
   :caption: Call sequence for initialization, inference updates, Memfault collect, and CDR drain.

   skinparam shadowing false
   skinparam roundcorner 0
   skinparam backgroundColor #FFFFFF
   skinparam ArrowColor #0077C8
   skinparam sequenceArrowThickness 1

   skinparam sequence {
     DividerBackgroundColor #8DBEFF
     DividerBorderColor #8DBEFF
     LifeLineBackgroundColor #13B6FF
     LifeLineBorderColor #13B6FF
     ParticipantBackgroundColor #13B6FF
     ParticipantBorderColor #13B6FF
     BoxBackgroundColor #C1E8FF
     BoxBorderColor #C1E8FF
     GroupBackgroundColor #8DBEFF
     GroupBorderColor #8DBEFF
   }

   skinparam note {
     BackgroundColor #ABCFFF
     BorderColor #2149C2
     Shadowing false
   }

   skinparam participant {
     Shadowing false
   }

   participant "Application" as App
   participant "nrf_edgeai_obsv\n(Zephyr wrapper)" as Obsv
   participant "nrf_edgeai_obsv_core" as Core
   participant Metric
   participant "nrf_edgeai_obsv_memfault" as Mflt
   participant "Memfault SDK" as SDK

   == Initialization ==

   App -> Obsv : nrf_edgeai_obsv_init(ctx, model)
   App -> Metric : nrf_edgeai_obsv_metric_tm_create(metric, buf, n)
   App -> Metric : nrf_edgeai_obsv_metric_pd_create(metric, buf, n)
   App -> Obsv : nrf_edgeai_obsv_register(ctx, metric, cfg)
   App -> Mflt : nrf_edgeai_obsv_memfault_init(ctx)
   Mflt -> SDK : memfault_cdr_register_source()

   == Inference loop ==

   loop every inference
     App -> Obsv : nrf_edgeai_obsv_update_probs(ctx, probs)
     Obsv -> Obsv : lock ctx->lock
     Obsv -> Core : nrf_edgeai_obsv_core_update_probs()
     Core -> Metric : metric->update(probs, n)
     Obsv -> Obsv : unlock ctx->lock
   end

   == Collect (periodic or on demand) ==

   App -> Mflt : nrf_edgeai_obsv_memfault_collect()
   note right of Mflt : snapshot ctx list under obsv_mflt_lock,\nthen release before encoding
   Mflt -> Obsv : nrf_edgeai_obsv_encode_list(ctxs, n, buf)
   loop per context
     Obsv -> Obsv : lock ctx->lock
     Obsv -> Core : nrf_edgeai_obsv_core_for_each_metric()
     Core -> Metric : metric->finalize()
     Core -> Metric : metric->snapshot()
     Core -> Obsv : zcbor encode → CBOR blob
     Obsv -> Obsv : unlock ctx->lock
   end
   Mflt -> Mflt : copy blob to staging buffer\nunder obsv_mflt_lock

   == Transport drain ==

   SDK -> Mflt : has_cdr_cb()
   Mflt --> SDK : metadata (size, mime type)
   SDK -> Mflt : read_data_cb(offset, len)
   Mflt --> SDK : CBOR payload bytes
   SDK -> Mflt : mark_cdr_read_cb()
   SDK -> SDK : upload via BLE MDS or HTTP

Metrics
*******

The module exposes model behavior through metrics that are updated on every inference and exported as snapshots.
Each metric is a self-contained unit with its own storage and callbacks, which means you can mix built-in metrics with your own custom ones.
The module includes several ready-to-use built-in metrics, and provides an interface for adding your own.

.. _nrf_edgeai_obsv_metrics_built_in:

Built-in metrics
================

The built-in metrics capture different aspects of model behavior over time.
They fall into two groups by the input stream they consume: *output metrics*, which observe the model's class-probability vector, and *input-feature metrics*, which observe the extracted feature vector fed to the model.
See the :ref:`nrf_edgeai_obsv_buffer_config` section for the available options.

.. _nrf_edgeai_obsv_metrics_built_in_transition:

Transition matrix
-----------------

The transition matrix counts how many times the dominant class (argmax of the probability vector) changed from class *i* to class *j* across consecutive calls to :c:func:`nrf_edgeai_obsv_update_probs`.
The result is a square ``num_classes × num_classes`` matrix of ``uint32_t`` counters stored in row-major order, where row *i* is the previous class and column *j* is the current class.

The following table shows rows for the previous dominant class and columns for the current dominant class:

.. list-table:: Example transition matrix counts (four classes, illustrative)
   :widths: auto
   :header-rows: 1
   :stub-columns: 1

   * -
     - idle
     - walk
     - run
     - jump
   * - idle
     - 0
     - 38
     - 3
     - 1
   * - walk
     - 36
     - 0
     - 12
     - 2
   * - run
     - 3
     - 10
     - 0
     - 5
   * - jump
     - 2
     - 3
     - 3
     - 0

The illustrative counts suggest *walk* as the dominant class, frequent transitions between *idle* and *walk*, and little *jump* activity.

.. _nrf_edgeai_obsv_metrics_built_in_probability:

Probability distribution
------------------------

The probability distribution metric builds a per-class histogram over the ``[0, 1]`` probability range.
Each call to :c:func:`nrf_edgeai_obsv_update_probs` function increments one bin per class based on that class's output probability.
The result is a ``num_classes × bin_num`` matrix of ``uint32_t`` bin counts, where row *i* is the class and each column is a histogram bin.

The following table uses four uniform bins with inner edges at 0.25, 0.50, and 0.75.
Each row is a class; each column is a probability bin.

.. list-table:: Example per-class probability histogram counts (four classes, four bins, illustrative)
   :widths: auto
   :header-rows: 1
   :stub-columns: 1

   * -
     - [0, 0.25)
     - [0.25, 0.50)
     - [0.50, 0.75)
     - [0.75, 1.0]
   * - idle
     - 85
     - 22
     - 14
     - 20
   * - walk
     - 18
     - 25
     - 31
     - 67
   * - run
     - 96
     - 30
     - 10
     - 5
   * - jump
     - 130
     - 8
     - 2
     - 1

The illustrative counts suggest *walk* is often predicted with high confidence (67 counts in the top bin), a bimodal spread for *idle*, and mostly low confidence for *run* and *jump*, which may indicate confusion between those classes.

.. _nrf_edgeai_obsv_metrics_built_in_switching:

Prediction switching rate
-------------------------

The prediction switching rate tracks temporal instability: how often the dominant class (argmax of the probability vector) changes between consecutive inferences.
It exports two ``uint32_t`` counters as a ``1 × 2`` row, ``[switches, comparisons]``, from which the rate is derived off-device as ``switches / comparisons``.
A high rate indicates an unstable or noisy input.

.. _nrf_edgeai_obsv_metrics_built_in_entropy:

Probability entropy distribution
--------------------------------

The probability entropy distribution builds a histogram of prediction uncertainty.
For each inference it computes the normalized Shannon entropy ``H(p) / ln(N)`` of the probability vector and bins it over ``[0, 1]``, producing a ``1 × bin_num`` row.
High entropy flags uncertain predictions or out-of-distribution inputs; low entropy flags confident predictions.

.. _nrf_edgeai_obsv_metrics_built_in_margin:

Probability top-2 margin distribution
-------------------------------------

The probability top-2 margin distribution builds a histogram of prediction decisiveness.
For each inference it computes the margin between the two largest class probabilities, ``margin = p_top1 - p_top2``, and bins it over ``[0, 1]`` as a ``1 × bin_num`` row.
A low margin flags ambiguous predictions even when the dominant probability is high.

Input-feature metrics
----------------------

The following metrics consume the input-feature stream fed through :c:func:`nrf_edgeai_obsv_update_features`, rather than the output probabilities.
They target audio mel-spectrogram features (for example, wake-word and keyword-spotting models), but apply to any non-negative feature vector.

.. _nrf_edgeai_obsv_metrics_built_in_mel_energy:

Mel energy descriptor
---------------------

The mel energy descriptor summarizes per-frame energy statistics of the input mel feature vector.
It produces a ``4 × bin_num`` matrix with one ``[0, 1]`` histogram row per statistic: mean energy, max energy, dynamic range (q95 − q05), and the floor-bin ratio.
Feature values are normalized into ``[0, 1]`` against a configured percentile range ``[p01, p99]`` (``CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_SCALE_P01_MILLI`` and ``_SCALE_P99_MILLI``, in thousandths of a feature unit), so the bins are comparable across devices.
Measure the percentiles offline on a representative dataset; the defaults are placeholders.

.. _nrf_edgeai_obsv_metrics_built_in_mel_spectral:

Mel spectral descriptor
-----------------------

The mel spectral descriptor summarizes per-frame spectral shape of the input mel feature vector.
It produces an ``8 × bin_num`` matrix with one ``[0, 1]`` histogram row per statistic: low, mid, and high band energy ratios, spectral centroid, spread, entropy, flatness, and contrast.
Every statistic is scale-invariant (it divides by the total energy or the mean), so no amplitude calibration is needed.

.. _nrf_edgeai_obsv_metrics_custom:

Custom metrics
==============

You can implement additional metrics by filling in an :c:struct:`nrf_edgeai_obsv_metric_t` operation table and registering it with the :c:func:`nrf_edgeai_obsv_register` function.
A metric consists of five callbacks, a ``source`` field, and a ``priv`` pointer to its own storage:

.. list-table:: Observability metric callbacks
   :widths: auto
   :header-rows: 1

   * - Callback
     - Required
     - Purpose
   * - ``init(cfg, priv)``
     - yes
     - Zero counters and apply optional configuration.
   * - ``update(data, n, priv)``
     - yes
     - Consume one vector from the metric's source stream: class probabilities or input features.
   * - ``clear(priv)``
     - no
     - Zero counters without touching configuration (called by :c:func:`nrf_edgeai_obsv_reset`). Set to ``NULL`` if reset is a no-op.
   * - ``finalize(priv)``
     - no
     - Compute derived values before a snapshot is taken. Set to ``NULL`` if not needed.
   * - ``snapshot(out, priv)``
     - yes
     - Populate a read-only :c:struct:`nrf_edgeai_obsv_metric_snapshot_t` view. The ``counts`` pointer must remain valid for the lifetime of the metric instance.

The snapshot exposes counters as a flat row-major ``uint32_t`` matrix of ``num_rows × num_cols`` elements.
Metrics with a single scalar value use ``num_rows = 1, num_cols = 1``.

Set the ``source`` field to select the input stream the metric consumes: ``NRF_EDGEAI_OBSV_SOURCE_PROBS`` (the default, ``0``) for the class-probability vector, or ``NRF_EDGEAI_OBSV_SOURCE_FEATURES`` for the input-feature vector.
The ``update`` callback then receives that stream, and the ``n`` argument is the class count for probabilities or the feature-vector length for features.

Custom metric IDs
-----------------

Choose an ID that does not collide with the built-in values defined in :c:enum:`nrf_edgeai_obsv_metric_id`.
Using values well above the built-in range (for example, 1000 and above) leaves room for future built-in additions.

.. _nrf_edgeai_obsv_buffer_sizing:

Buffer sizing for CBOR encoding
-------------------------------

When the Memfault transport (or :c:func:`nrf_edgeai_obsv_encode_list`) serializes all metrics, the encode buffer must be large enough to hold custom metric data in addition to the built-in ones.
You reserve this space through the ``CONFIG_NRF_EDGEAI_OBSV_EXTRA_ENCODE_BYTES`` Kconfig option (see :ref:`nrf_edgeai_obsv_buffer_config`).

The required value is the sum of ``NRF_EDGEAI_OBSV_ENCODE_METRIC_SIZE(n_rows, n_cols)`` across all custom metrics.
Because ``NRF_EDGEAI_OBSV_ENCODE_METRIC_SIZE`` is a C preprocessor macro, evaluate it at compile time and write the resulting integer directly in :file:`prj.conf`.
To catch mismatches at build time, add a ``BUILD_ASSERT`` in your application code:

.. code-block:: c

   BUILD_ASSERT(CONFIG_NRF_EDGEAI_OBSV_EXTRA_ENCODE_BYTES >=
                NRF_EDGEAI_OBSV_ENCODE_METRIC_SIZE(1, MY_NUM_CLASSES),
                "EXTRA_ENCODE_BYTES too small for custom metric");

Example custom metric: class frequency counter
----------------------------------------------

The following example shows a minimal custom metric that counts how often each class is the argmax across all inferences.
It produces a 1 × ``num_classes`` row of ``uint32_t`` counters, with storage passed through priv to match the built-in metric pattern.

.. code-block:: c

   #include <stdint.h>
   #include <string.h>
   #include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
   #include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

   #define MY_METRIC_ID   1000U
   #define MY_METRIC_VER  1U
   #define MY_NUM_CLASSES 4U


   static void my_init(const void *cfg, void *priv)
   {
       ARG_UNUSED(cfg);
       memset(priv, 0, MY_NUM_CLASSES * sizeof(uint32_t));
   }

   static void my_clear(void *priv)
   {
       memset(priv, 0, MY_NUM_CLASSES * sizeof(uint32_t));
   }

   static void my_update(const float *probs, uint16_t n, void *priv)
   {
       uint32_t *counts = (uint32_t *)priv;
       uint16_t argmax = 0;

       /* When probabilities tie for the maximum, the lowest index wins. */
       for (uint16_t i = 1; i < n; i++) {
           if (probs[i] > probs[argmax]) {
               argmax = i;
           }
       }
       counts[argmax]++;
   }

   static void my_snapshot(nrf_edgeai_obsv_metric_snapshot_t *out, void *priv)
   {
       out->metric_id = MY_METRIC_ID;
       out->version   = MY_METRIC_VER;
       out->num_rows  = 1U;
       out->num_cols  = MY_NUM_CLASSES;
       out->counts    = (uint32_t *)priv;
   }

   /* buf: at least n_classes * sizeof(uint32_t) bytes, uint32_t-aligned. */
   void my_metric_create(nrf_edgeai_obsv_metric_t *metric, void *buf, uint16_t n_classes)
   {
       ARG_UNUSED(n_classes); /* stored implicitly via MY_NUM_CLASSES in callbacks */
       *metric = (nrf_edgeai_obsv_metric_t){
           .init     = my_init,
           .update   = my_update,
           .clear    = my_clear,
           .finalize = NULL,
           .snapshot = my_snapshot,
           .priv     = buf,
       };
   }

Register it alongside the built-in metrics during initialization:

.. code-block:: c

   static uint32_t my_buf[MY_NUM_CLASSES];
   static nrf_edgeai_obsv_metric_t my_metric;

   my_metric_create(&my_metric, my_buf, MY_NUM_CLASSES);
   nrf_edgeai_obsv_init(&obsv_ctx, &model);
   nrf_edgeai_obsv_register(&obsv_ctx, &my_metric, NULL);

.. _nrf_edgeai_obsv_buffer_config:

Configuration
*************

To use the observability library, enable the ``CONFIG_NRF_EDGEAI_OBSV`` Kconfig option in your :file:`prj.conf` file.
Enable at least one metric to start collecting data:

* Built-in metrics:

  * For the :ref:`transition matrix <nrf_edgeai_obsv_metrics_built_in_transition>`, enable
    ``CONFIG_NRF_EDGEAI_OBSV_METRIC_TRANSITION_MATRIX``.
  * For the :ref:`probability distribution <nrf_edgeai_obsv_metrics_built_in_probability>`, enable ``CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_DISTRIBUTION``, and set the number of histogram bins through ``CONFIG_NRF_EDGEAI_OBSV_PROBS_DISTRIBUTION_BIN_NUM``.
  * For the :ref:`prediction switching rate <nrf_edgeai_obsv_metrics_built_in_switching>`, enable ``CONFIG_NRF_EDGEAI_OBSV_METRIC_PREDICTION_SWITCHING_RATE``.
  * For the :ref:`probability entropy distribution <nrf_edgeai_obsv_metrics_built_in_entropy>`, enable ``CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_ENTROPY_DIST``, and set the bin count through ``CONFIG_NRF_EDGEAI_OBSV_PROBS_ENTROPY_DIST_BIN_NUM``.
  * For the :ref:`probability top-2 margin distribution <nrf_edgeai_obsv_metrics_built_in_margin>`, enable ``CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_TOP2_MARGIN_DIST``, and set the bin count through ``CONFIG_NRF_EDGEAI_OBSV_PROBS_TOP2_MARGIN_DIST_BIN_NUM``.
  * For the :ref:`mel energy descriptor <nrf_edgeai_obsv_metrics_built_in_mel_energy>`, enable ``CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_ENERGY_DESC``, and set the bin count, the maximum feature length, and the ``[p01, p99]`` scaling percentiles through the matching ``CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_*`` options.
  * For the :ref:`mel spectral descriptor <nrf_edgeai_obsv_metrics_built_in_mel_spectral>`, enable ``CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_SPECTRAL_DESC``, and set the bin count through ``CONFIG_NRF_EDGEAI_OBSV_MEL_SPECTRAL_DESC_BIN_NUM``.

* Custom metrics:

  * If you wish to implement custom metrics, set ``CONFIG_NRF_EDGEAI_OBSV_EXTRA_ENCODE_BYTES`` to reserve buffer space for CBOR encoding.
    See :ref:`Buffer sizing for CBOR encoding <nrf_edgeai_obsv_buffer_sizing>` for how to calculate the value.

For a full list of available Kconfig options, refer to the following sections:

Core library
============

.. options-from-kconfig:: /lib/nrf_edgeai_obsv/Kconfig
   :show-type:

Memfault CDR transport
======================

The Memfault module registers a CDR source with the Memfault SDK; see `Memfault Custom Data Recording`_ for callback semantics, payload metadata, and upload limits, `Memfault`_ for the vendor platform, and `Memfault in nRF Connect SDK`_ for integration in |NCS|.

.. options-from-kconfig:: /lib/nrf_edgeai_obsv_memfault/Kconfig
   :show-type:

Usage
*****

The observability library works with any inference engine that produces a probability vector per inference.
For a complete example using the nRF Edge AI API, see :ref:`quick_start_nrf_edgeai`.

To integrate the library into your application, complete the following steps:

1. Initialize an observability context with model metadata using the :c:func:`nrf_edgeai_obsv_init` function.
#. Allocate metric storage and initialize each metric descriptor using the :c:func:`nrf_edgeai_obsv_metric_tm_create` and :c:func:`nrf_edgeai_obsv_metric_pd_create` functions.
#. Register the metrics with the context using the :c:func:`nrf_edgeai_obsv_register` function.
#. Bind the Memfault transport once at application startup using the :c:func:`nrf_edgeai_obsv_memfault_init` function.
#. Call the :c:func:`nrf_edgeai_obsv_update_probs` function with the output probability vector after every inference.
#. If you registered input-feature metrics, call the :c:func:`nrf_edgeai_obsv_update_features` function with the extracted feature vector. This routes only to feature-source metrics and does not advance the inference counter.
#. Call the :c:func:`nrf_edgeai_obsv_memfault_collect` function periodically, or enable the ``CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT`` Kconfig option.

The following example shows minimal initialization with both built-in metrics and Memfault upload over Bluetooth using MDS:

.. code-block:: c

   #include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
   #include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>
   #include <nrf_edgeai_obsv/nrf_edgeai_obsv_memfault.h>

   #define NUM_CLASSES 4

   static nrf_edgeai_obsv_ctx_t obsv_ctx;

   /* uint32_t arrays give natural alignment required by the storage macros. */
   static uint32_t tm_buf[NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(NUM_CLASSES) / sizeof(uint32_t)];
   static uint32_t pd_buf[NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(NUM_CLASSES) / sizeof(uint32_t)];
   static nrf_edgeai_obsv_metric_t tm_metric;
   static nrf_edgeai_obsv_metric_t pd_metric;

   void observability_init(void)
   {
       const nrf_edgeai_obsv_model_info_t model = {
           .model_id    = 1,
           .num_classes = NUM_CLASSES,
           .version     = 1,
       };

       nrf_edgeai_obsv_init(&obsv_ctx, &model);

       nrf_edgeai_obsv_metric_tm_create(&tm_metric, tm_buf, NUM_CLASSES);
       nrf_edgeai_obsv_register(&obsv_ctx, &tm_metric, NULL);

       nrf_edgeai_obsv_metric_pd_create(&pd_metric, pd_buf, NUM_CLASSES);
       nrf_edgeai_obsv_register(&obsv_ctx, &pd_metric, NULL);

       /* Bind the Memfault transport. */
       nrf_edgeai_obsv_memfault_init(&obsv_ctx);
   }

   void on_inference_done(const float *probs)
   {
       /* Feed inference result to all registered metrics. */
       nrf_edgeai_obsv_update_probs(&obsv_ctx, probs);
   }

When ``CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT`` is disabled, call :c:func:`nrf_edgeai_obsv_memfault_collect` manually at the interval that matches your transport's drain cadence (for example, once per hour for HTTP, or before each Bluetooth LE connection).

When using a custom transport instead of Memfault, use :c:func:`nrf_edgeai_obsv_encode_list` to encode one or more contexts into a caller-supplied buffer in a single CBOR list:

.. code-block:: c

   uint8_t cbor_buf[NRF_EDGEAI_OBSV_ENCODE_LIST_BUF_SIZE(1)];
   nrf_edgeai_obsv_ctx_t *ctxs[] = {&obsv_ctx};

   size_t len = nrf_edgeai_obsv_encode_list(ctxs, ARRAY_SIZE(ctxs), cbor_buf, sizeof(cbor_buf));
   if (len > 0) {
       my_transport_send(cbor_buf, len);
   }

Thread safety
=============

The following functions acquire ``ctx->lock`` internally and are safe to call from different threads:

* :c:func:`nrf_edgeai_obsv_update_probs`
* :c:func:`nrf_edgeai_obsv_update_features`
* :c:func:`nrf_edgeai_obsv_encode`
* :c:func:`nrf_edgeai_obsv_for_each_metric`

The :c:func:`nrf_edgeai_obsv_memfault_collect` function uses two mutexes:

* ``obsv_mflt_lock`` protects the staging buffer and the registered context list.
* Each ``ctx->lock`` is acquired by :c:func:`nrf_edgeai_obsv_encode_list` during encoding.

To avoid lock inversion, ``obsv_mflt_lock`` is released before encoding begins, so inference is never blocked by an ongoing collect.

.. _nrf_edgeai_obsv_script:

Decoding CDR payloads
*********************

Use the :file:`scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py` script on a host PC to inspect collected observability data as JSON (per-model counters and metric tables).
Run it on payloads that are already in Memfault, or on hex data captured from UART or Bluetooth LE when debugging transport and encoding.

The script accepts Memfault web UI downloads (``--binary --file``), Memfault API fetch (``--from-cloud``), hex-encoded chunks from UART or Bluetooth LE, and multi-chunk reassembly (``--chunks``).

Install the Python dependencies from the sdk-edge-ai tree root:

.. code-block:: shell

   pip install -r scripts/decode_edgeai_obsv_cdr/requirements.txt

The following examples show common usage:

.. code-block:: shell

   # Memfault web UI download
   ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py --binary --file recording.bin

   # Hex from a serial log
   ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py 04a1b2c3d4...

Run ``--help`` on the script for the full option list and Memfault API authentication details.

Dependencies
************

Observability core with Zephyr wrapper
======================================

This module uses the following Zephyr libraries:

* `ZCBOR`_

Memfault CDR transport module
=============================

This module uses the following |EAI| library:

* :file:`lib/nrf_edgeai_obsv`

This module uses the following Zephyr libraries:

* `Logging`_

This module uses the following |NCS| libraries:

* `Memfault in nRF Connect SDK`_

.. _nrf_edgeai_obsv_lib_api:

API Reference
*************

Zephyr context
==============

.. doxygengroup:: nrf_edgeai_obsv

Portable core
=============

.. doxygengroup:: nrf_edgeai_obsv_core

Metrics
=======

.. doxygengroup:: nrf_edgeai_obsv_metrics

Memfault CDR transport
======================

.. doxygengroup:: nrf_edgeai_obsv_memfault
