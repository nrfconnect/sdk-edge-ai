.. _nrf_edgeai_obsv_lib:

nRF Edge AI Observability Library
#################################

.. contents::
   :local:
   :depth: 2

The Edge AI Observability module tracks how a classification model performs at runtime.
It collects stats from the model's output probabilities and, optionally, from the input features fed to the model, and packages them as metric snapshots that can be sent to a monitoring backend.
It works with any inference engine that produces a probability vector, including the :ref:`nrf_edgeai_lib`, :ref:`Axon NPU <lib_axon>`, and `Edge Impulse`_ deployments.

Overview
********

For a conceptual introduction to model observability, what it enables, and how it fits into the wider Edge AI model lifecycle, see :ref:`model_observability`.

Metrics are driven by two input streams.
Output metrics consume the model's class-probability vector passed to :c:func:`nrf_edgeai_obsv_update_probs`, while input-feature metrics consume the extracted feature vector fed to the model and passed to :c:func:`nrf_edgeai_obsv_update_features`.
Each metric declares which stream it consumes through its ``source`` field, and the library routes every update only to the metrics that match.
The two streams are counted independently: every call to :c:func:`nrf_edgeai_obsv_update_probs` advances the ``num_inferences`` counter and every call to :c:func:`nrf_edgeai_obsv_update_features` advances the separate ``num_features`` counter.

The module is organized as three cooperating layers:

* Core (:file:`lib/nrf_edgeai_obsv/nrf_edgeai_obsv_core.c`) - A portable, mutex-free state machine that accumulates metric counters as inference results arrive.
  It has no Zephyr RTOS dependency and you can use it in bare-metal environments, other RTOSes, or host-side test builds.
* Zephyr wrapper (:file:`lib/nrf_edgeai_obsv/nrf_edgeai_obsv.c`) - Wraps the core in a mutex-protected context so that multiple threads can feed inferences and trigger encoding without data races, and integrates the library into the Zephyr build system (CMake, Kconfig, logging).
* Memfault CDR transport (:file:`lib/nrf_edgeai_obsv_memfault/nrf_edgeai_obsv_memfault.c`) - Encodes the accumulated metric snapshots as a CBOR blob and stages them as a `Memfault Custom Data Recording`_ (CDR) that the Memfault SDK packetizer uploads on the next transport drain cycle.
  For Memfault Kconfig, keys, and transports in |NCS|, see :ref:`nrf:mod_memfault`.

.. uml::
   :caption: High-level data flow from application inference through observability to nRF Cloud and downstream tools.

   skinparam shadowing false
   skinparam roundcorner 0
   skinparam backgroundColor #FFFFFF
   skinparam defaultTextAlignment center
   skinparam ArrowColor #0077C8
   skinparam ArrowThickness 1
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
     component "Application" as App #D9E1E2;line:768692;text:333F48
     component "nrf_edgeai_obsv\n(Zephyr wrapper + core)" as Obsv
     component "Metrics\n(e.g. class predictions distribution)" as Metrics
   }

   component "nrf_edgeai_obsv_memfault\n(Memfault CDR transport)" as MfltTransport #0033A0;line:0033A0;text:FFFFFF
   component "Memfault SDK" as MfltSDK #0033A0;line:0033A0;text:FFFFFF
   cloud "nRF Cloud\n(Memfault)" as Cloud #0033A0;line:0033A0;text:FFFFFF
   component "Monitoring tool\n(dashboard / ML pipeline)" as Dashboard #0033A0;line:0033A0;text:FFFFFF

   App -right-> Obsv : input features and\nclass probabilities [1]
   Obsv -down-> Metrics : accumulate counters [2]
   App -down-> MfltTransport : trigger collect [3]
   MfltTransport -up-> Obsv : encode metrics as CBOR [4]
   MfltTransport -right-> MfltSDK : stage CDR [5]
   MfltSDK -right-> Cloud : upload via BLE or HTTP [6]
   Cloud -right-> Dashboard : fetch CDR\n(REST API) [7]

   legend right
     Two independent flows start at "Application":
     1-2 run on every inference.
     3-7 run periodically, or on demand, to drain and upload the accumulated metrics.
   endlegend

The following diagram shows how the application initializes observability and updates metrics during inference.

.. uml::
   :caption: Observability initialization and inference updates.

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

   skinparam participant {
     Shadowing false
   }

   participant "Application" as App
   participant "nRF Edge AI Library" as Rt
   participant "nrf_edgeai_obsv" as Obsv
   participant Metric

   == Initialization ==

   App -> Rt : nrf_edgeai_init(model)
   App -> Obsv : nrf_edgeai_obsv_init(ctx, model_info)
   loop each enabled metric
     App -> Metric : nrf_edgeai_obsv_metric_*_create(metric, buf, n)
     App -> Obsv : nrf_edgeai_obsv_register(ctx, metric, cfg)
   end

   == Inference loop ==

   loop until the input window is full
     App -> Rt : nrf_edgeai_feed_inputs(model, samples, n)
   end

   opt application reports input features
     App -> Rt : nrf_edgeai_process_features(model)
     Rt --> App : extracted feature vector
     App -> Obsv : nrf_edgeai_obsv_update_features(ctx, feats, n)
     Obsv -> Obsv : lock ctx->lock
     Obsv -> Metric : update registered FEATURES-source metrics
     Obsv -> Obsv : unlock ctx->lock
   end

   App -> Rt : nrf_edgeai_run_inference(model)
   Rt -> Rt : process features if needed
   Rt --> App : class probability vector

   App -> Obsv : nrf_edgeai_obsv_update_probs(ctx, probs)
   Obsv -> Obsv : lock ctx->lock
   Obsv -> Metric : update registered PROBS-source metrics
   Obsv -> Obsv : unlock ctx->lock

The following diagram shows how the Memfault integration encodes accumulated metric snapshots and stages them for transport.

.. uml::
   :caption: Observability collection, staging, retry, and transport drain.

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

   participant "Application /\nauto-collect work" as Trigger
   participant "nrf_edgeai_obsv_memfault" as Mflt
   participant "nrf_edgeai_obsv" as Obsv
   participant "Memfault SDK" as SDK
   participant "nRF Cloud" as Cloud

   == Initialization ==

   Trigger -> Mflt : nrf_edgeai_obsv_memfault_init(ctx)
   Mflt -> SDK : memfault_cdr_register_source()

   == Collect (periodic or on demand) ==

   Trigger -> Mflt : nrf_edgeai_obsv_memfault_collect()
   alt previous CDR not drained yet
     Mflt --> Trigger : -EBUSY\n(nothing encoded or reset)
   else staging slot free
     Mflt -> Obsv : nrf_edgeai_obsv_encode_list_and_reset(ctxs, n, buf)
     note right of Obsv
       Hold all context locks while encoding.
       Reset only after the complete list encodes.
     end note
     Obsv --> Mflt : encoded length
     Mflt -> Mflt : copy blob to staging buffer
     Mflt --> Trigger : success
   end

   == Transport drain ==

   SDK -> Mflt : has_cdr_cb()
   Mflt --> SDK : metadata (size, mime type)
   SDK -> Mflt : read_data_cb(offset, len)
   Mflt --> SDK : CBOR payload bytes
   SDK -> Mflt : mark_cdr_read_cb()
   Mflt -> Mflt : release the staging slot
   opt auto-collect enabled and collection pending
     Mflt -> Mflt : reschedule auto-collect immediately
   end
   SDK -> Cloud : upload via BLE MDS or HTTP

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

The transition matrix counts how many times the dominant class (argmax of the probability vector) went from class *i* to class *j* across consecutive calls to :c:func:`nrf_edgeai_obsv_update_probs`.
Every pair of consecutive inferences is counted, so the diagonal (*i* = *j*) holds the inferences that kept the same dominant class, and the off-diagonal cells hold the class changes.
The first inference after initialization or reset has no predecessor and is not counted.
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
     - 120
     - 38
     - 3
     - 1
   * - walk
     - 36
     - 210
     - 12
     - 2
   * - run
     - 3
     - 10
     - 45
     - 5
   * - jump
     - 2
     - 3
     - 3
     - 8

The illustrative counts suggest *walk* as the dominant class (largest diagonal count), frequent transitions between *idle* and *walk*, and little *jump* activity.

.. _nrf_edgeai_obsv_metrics_built_in_class_pred:

Class predictions distribution
------------------------------

The class predictions distribution gives a per-class picture of the model's output in one ``2 × num_classes × bin_num`` matrix of ``uint32_t`` counters, sharing a single bin count between its two row groups.

**Probability distribution rows** (``[0, num_classes)``).

A per-class histogram over the ``[0, 1]`` probability range: each call to the :c:func:`nrf_edgeai_obsv_update_probs` function increments one bin per class based on that class's output probability.
Every inference contributes one sample to every class row, so each of these rows sums to the inference count.

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

**Streak distribution rows** (``[num_classes, 2 × num_classes)``).

A per-class histogram of *streak lengths* - the number of consecutive inferences for which the dominant class (the argmax of the probability vector) stays the same.

A streak is recorded only when it ends.

Up to ``CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_STREAK_TOL`` consecutive mismatching inferences are bridged without ending the streak (flicker tolerance) and are not counted toward its length; a longer run of mismatches ends it.
When it does, the trailing bridged frames of the class that took over are counted toward that class's new streak, so a real class change loses no frames.
Streak lengths are binned uniformly over ``[1, TOP]``, with lengths of ``CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_STREAK_TOP_BIN`` or longer saturating the top bin.
Setting the tolerance to ``0`` reduces the metric to strict "N in a row" runs.

Because each streak contributes a single count only when it completes, these rows do not sum to the inference count; each totals the number of finished streaks per class.
This separates stable, sustained detections (streaks reaching the mid or top bins) from single-frame flicker (streaks pinned in the lowest bin), a distinction the per-inference probability rows cannot make.
The metric reports the configuration the streak rows were gathered with as the single-row matrix ``"c": [[streak_top, streak_tol]]``, so a decoded dump carries the top-bin streak length and the flicker tolerance without relying on the build configuration.

.. _nrf_edgeai_obsv_metrics_built_in_certainty:

Model certainty descriptor
--------------------------

The model certainty descriptor summarizes, per inference, how certain and how temporally stable the model's predictions are, in a single ``3 × bin_num`` matrix.
It merges the former prediction switching rate, probability entropy distribution, and probability top-2 margin distribution metrics, so one metric answers the whole "is the model sure of itself" question.
For each inference it derives, in one pass over the probability vector:

* **Row 0 — normalized entropy histogram.**
  The normalized Shannon entropy ``H(p) / ln(N)`` binned over ``[0, 1]`` (uncertainty).
  High entropy flags uncertain predictions or out-of-distribution inputs; low entropy flags confident predictions.
* **Row 1 — top-2 margin histogram.**
  The margin ``p_top1 - p_top2`` between the two largest class probabilities, binned over ``[0, 1]`` (decisiveness).
  A low margin flags ambiguous predictions even when the dominant probability is high.
* **Row 2 — stability counters.**
  The ``uint32_t`` counters ``[switches, comparisons, majority_frames, confident_switches]``, with the rest of the row zero-padded.
  ``switches / comparisons`` is the off-device switching rate (temporal instability); ``majority_frames`` counts inferences whose winner exceeded ``0.5``; and ``confident_switches`` counts switches into a ``> 0.5`` winner, separating confident class-confusion from low-probability churn.

One bin count, ``CONFIG_NRF_EDGEAI_OBSV_MODEL_CERTAINTY_DESC_BIN_NUM``, is shared by the two histogram rows and sizes the stability row.

Input-feature metrics
---------------------

The following metrics consume the input-feature stream fed through :c:func:`nrf_edgeai_obsv_update_features`, rather than the output probabilities.
They target audio mel-spectrogram features (for example, wake-word and keyword-spotting models), but apply to any non-negative feature vector.

.. _nrf_edgeai_obsv_metrics_built_in_mel_energy:

Mel energy descriptor
---------------------

The mel energy descriptor summarizes per-frame energy statistics of the input mel feature vector.
It produces a ``4 × bin_num`` matrix with one ``[0, 1]`` histogram row per statistic: mean energy, max energy, dynamic range (q95 − q05), and the floor-bin ratio.
Feature values are normalized into ``[0, 1]`` against a configured percentile range ``[p01, p99]`` (``CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_SCALE_P01_MILLI`` and ``_SCALE_P99_MILLI``, in thousandths of a feature unit), so the bins are comparable across devices.
The metric reports these bounds as the single-row matrix ``"c": [[scale_p01_milli, scale_p99_milli]]``, each value a signed integer, so a decoded dump carries the calibration without relying on the build configuration.
The metric reports ``"v": 2``; version 1 had no ``"c"`` key.
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
A metric can also report the configuration its counters were gathered with, as a second flat row-major ``int32_t`` matrix of ``config_rows × config_cols`` elements at ``config``.
Like ``counts``, the pointer must remain valid for the lifetime of the metric instance, and the matrix is encoded under the optional ``"c"`` key of the metric.
The meaning and order of the values is fixed per metric ID and version.
The core zero-initializes the snapshot before calling ``snapshot()``, so a metric that leaves ``config`` as ``NULL`` emits no ``"c"`` key.
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
If your metric reports configuration through ``config``, add ``NRF_EDGEAI_OBSV_ENCODE_METRIC_CONFIG_SIZE(config_rows, config_cols)`` to its size.
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
Then complete the following setup:

* Set ``CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES`` to the largest class count among the observed models (the default is ``4``).
  Metric storage and the CBOR encode buffer scale with this value, and the transition matrix grows as the square of it.
* To encode the snapshots as CBOR, enable ``CONFIG_ZCBOR`` and ``CONFIG_NRF_EDGEAI_OBSV_ENCODE``.
  The Memfault CDR transport and :c:func:`nrf_edgeai_obsv_encode_list` require them.

Enable at least one metric to start collecting data:

* Built-in metrics:

  * For the :ref:`transition matrix <nrf_edgeai_obsv_metrics_built_in_transition>`, enable
    ``CONFIG_NRF_EDGEAI_OBSV_METRIC_TRANSITION_MATRIX``.
  * For the :ref:`class predictions distribution <nrf_edgeai_obsv_metrics_built_in_class_pred>`, enable ``CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_PRED_DIST``, and set the shared bin count, the top-bin streak length, and the flicker tolerance through the matching ``CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_*`` options.
  * For the :ref:`model certainty descriptor <nrf_edgeai_obsv_metrics_built_in_certainty>`, enable ``CONFIG_NRF_EDGEAI_OBSV_METRIC_MODEL_CERTAINTY_DESC``, and set the bin count shared by its entropy and top-2 margin rows through ``CONFIG_NRF_EDGEAI_OBSV_MODEL_CERTAINTY_DESC_BIN_NUM``.
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

The Memfault module registers a CDR source with the Memfault SDK; see `Memfault Custom Data Recording`_ for callback semantics, payload metadata, and upload limits, `Memfault`_ for the vendor platform, and :ref:`nrf:ug_memfault` in |NCS|.

To use it, enable the ``CONFIG_NRF_EDGEAI_OBSV_MEMFAULT`` Kconfig option.
It depends on ``CONFIG_NRF_EDGEAI_OBSV_ENCODE`` and on the Memfault CDR support (``CONFIG_MEMFAULT_CDR_ENABLE``).
Set ``CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_MAX_CONTEXTS`` to the number of observability contexts you register with the transport (one per observed model).
Each additional context enlarges the staging buffer by one maximum-size payload.

The :c:func:`nrf_edgeai_obsv_memfault_collect` function builds its encode buffer on the stack of the calling thread.
When you enable ``CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT``, the caller is the system workqueue, so increase ``CONFIG_SYSTEM_WORKQUEUE_STACK_SIZE`` accordingly.
The build fails if the stack is smaller than the encode buffer plus 1024 bytes.
The default auto-collect interval is one day, which matches the default Memfault limit of one CDR per device per day.

.. options-from-kconfig:: /lib/nrf_edgeai_obsv_memfault/Kconfig
   :show-type:

Usage
*****

The observability library works with any inference engine that produces a probability vector per inference.
For a complete example using the nRF Edge AI API, see :ref:`quick_start_nrf_edgeai`.

To integrate the library into your application, complete the following steps:

1. Initialize an observability context with model metadata using the :c:func:`nrf_edgeai_obsv_init` function.
   Set ``num_features`` in the model metadata to the input-feature vector length if you register input-feature metrics; otherwise leave it at ``0``.
#. Allocate metric storage and initialize each metric descriptor using the matching ``nrf_edgeai_obsv_metric_*_create`` function:

   * :c:func:`nrf_edgeai_obsv_metric_tm_create` for the transition matrix.
   * :c:func:`nrf_edgeai_obsv_metric_cpd_create` for the class predictions distribution.
   * :c:func:`nrf_edgeai_obsv_metric_mcd_create` for the model certainty descriptor.
   * :c:func:`nrf_edgeai_obsv_metric_med_create` for the mel energy descriptor.
   * :c:func:`nrf_edgeai_obsv_metric_msd_create` for the mel spectral descriptor.

   Size each buffer with the matching ``NRF_EDGEAI_OBSV_*_STORAGE_BYTES`` macro.
#. Register the metrics with the context using the :c:func:`nrf_edgeai_obsv_register` function.
#. Bind the Memfault transport once at application startup using the :c:func:`nrf_edgeai_obsv_memfault_init` function.
#. If you registered input-feature metrics, obtain the feature vector from your inference engine and pass it to the :c:func:`nrf_edgeai_obsv_update_features` function before running inference.
   With the nRF Edge AI Lib, call the :c:func:`nrf_edgeai_process_features` function once the input window is full, and read the resulting vector through the :c:func:`nrf_edgeai_dsp_features_ctx` function.
   The :c:func:`nrf_edgeai_obsv_update_features` call routes only to feature-source metrics and advances the feature counter.
#. Call the :c:func:`nrf_edgeai_obsv_update_probs` function with the output probability vector after every inference.
#. Call the :c:func:`nrf_edgeai_obsv_memfault_collect` function periodically, or enable the ``CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT`` Kconfig option.

The following example shows minimal initialization with the transition matrix and class predictions distribution metrics and Memfault upload over Bluetooth using MDS:

.. code-block:: c

   #include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
   #include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>
   #include <nrf_edgeai_obsv/nrf_edgeai_obsv_memfault.h>

   #define NUM_CLASSES 4

   static nrf_edgeai_obsv_ctx_t obsv_ctx;

   /* uint32_t arrays give natural alignment required by the storage macros. */
   static uint32_t tm_buf[NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(NUM_CLASSES) / sizeof(uint32_t)];
   static uint32_t cpd_buf[NRF_EDGEAI_OBSV_CPD_STORAGE_BYTES(NUM_CLASSES) / sizeof(uint32_t)];
   static nrf_edgeai_obsv_metric_t tm_metric;
   static nrf_edgeai_obsv_metric_t cpd_metric;

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

       nrf_edgeai_obsv_metric_cpd_create(&cpd_metric, cpd_buf, NUM_CLASSES);
       nrf_edgeai_obsv_register(&obsv_ctx, &cpd_metric, NULL);

       /* Bind the Memfault transport. */
       nrf_edgeai_obsv_memfault_init(&obsv_ctx);
   }

   void on_inference_done(const float *probs)
   {
       /* Feed inference result to all registered metrics. */
       nrf_edgeai_obsv_update_probs(&obsv_ctx, probs);
   }

If you also registered input-feature metrics, extract the features explicitly before the inference and feed them to the library, as in the following example for the nRF Edge AI Lib:

.. code-block:: c

   int run_model(nrf_edgeai_t *model, void *samples, uint16_t num_values)
   {
       nrf_edgeai_err_t err = nrf_edgeai_feed_inputs(model, samples, num_values);

       if (err != NRF_EDGEAI_ERR_SUCCESS) {
           return err; /* NRF_EDGEAI_ERR_INPROGRESS: window not full yet. */
       }

       err = nrf_edgeai_process_features(model);
       if (err != NRF_EDGEAI_ERR_SUCCESS) {
           return err;
       }

       const nrf_edgeai_dsp_feature_extraction_t *feats = nrf_edgeai_dsp_features_ctx(model);

       if (feats != NULL) {
           nrf_edgeai_obsv_update_features(&obsv_ctx, feats->buffer.p_f32, feats->overall_num);
       }

       err = nrf_edgeai_run_inference(model);
       if (err != NRF_EDGEAI_ERR_SUCCESS) {
           return err;
       }

       nrf_edgeai_obsv_update_probs(&obsv_ctx, model->decoded_output.classif.probabilities.p_f32);

       return 0;
   }

When ``CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT`` is disabled, call :c:func:`nrf_edgeai_obsv_memfault_collect` manually at the interval that matches your transport's drain cadence (for example, aligned with the periodic HTTP upload interval, or before each Bluetooth LE connection).

The :c:func:`nrf_edgeai_obsv_memfault_collect` function resets every registered context while it encodes them, holding the context locks for both steps so that no inference is lost in between.
Each staged payload therefore covers exactly the period since the previous successful collect, and the ``num_inferences`` and ``num_features`` counters restart from zero.

A staged payload is never overwritten, because its data no longer exists anywhere else.
While the Memfault SDK has not drained the previous payload, the :c:func:`nrf_edgeai_obsv_memfault_collect` function returns ``-EBUSY`` and does not encode or reset anything, so the contexts keep accumulating and the next payload covers the longer period.
With ``CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT`` enabled, a refused collect runs again as soon as the payload is drained.
With manual collection, treat ``-EBUSY`` as "try again later", for example at the next interval.

When using a custom transport instead of Memfault, use :c:func:`nrf_edgeai_obsv_encode_list` to encode one or more contexts into a caller-supplied buffer in a single CBOR list:

.. code-block:: c

   uint8_t cbor_buf[NRF_EDGEAI_OBSV_ENCODE_LIST_BUF_SIZE(1)];
   nrf_edgeai_obsv_ctx_t *ctxs[] = {&obsv_ctx};

   size_t len = nrf_edgeai_obsv_encode_list(ctxs, ARRAY_SIZE(ctxs), cbor_buf, sizeof(cbor_buf));
   if (len > 0) {
       my_transport_send(cbor_buf, len);
   }

For per-interval reporting, use :c:func:`nrf_edgeai_obsv_encode_list_and_reset` instead.
It resets the contexts under the same locks as the encode, and only when the whole list encodes successfully.

Thread safety
=============

The following functions acquire ``ctx->lock`` internally and are safe to call from different threads:

* :c:func:`nrf_edgeai_obsv_update_probs`
* :c:func:`nrf_edgeai_obsv_update_features`
* :c:func:`nrf_edgeai_obsv_encode`
* :c:func:`nrf_edgeai_obsv_for_each_metric`

The :c:func:`nrf_edgeai_obsv_encode_list_and_reset` function acquires the locks of all passed contexts, in array order, and holds them until every context is encoded and reset.

The :c:func:`nrf_edgeai_obsv_memfault_collect` function uses two mutexes:

* ``obsv_mflt_lock`` protects the staging buffer and the registered context list.
* Each ``ctx->lock`` is acquired by :c:func:`nrf_edgeai_obsv_encode_list_and_reset` during encoding and reset, thus could lead to stall of inference pipeline when called from low priority threads.

To avoid lock inversion, ``obsv_mflt_lock`` is released before encoding begins.
Inference on a context waits only while the contexts are being encoded and reset, never on the Memfault transport.

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

* :ref:`zephyr:cbor_api`

Memfault CDR transport module
=============================

This module uses the following |EAI| library:

* :file:`lib/nrf_edgeai_obsv`

This module uses the following Zephyr libraries:

* :ref:`zephyr:logging_api`

This module uses the following |NCS| libraries:

* :ref:`nrf:mod_memfault`

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
