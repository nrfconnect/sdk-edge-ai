.. _model_observability:
.. _model_lifecycle:

Model lifecycle and observability
#################################

.. contents::
   :local:
   :depth: 2

Model observability closes the loop between a neural network model running on a device in the field and the tools you used to create it.
This page provides an overview of model observability, the production issues it can help you investigate, and where it fits in the Edge AI model lifecycle — from model creation through deployment, field monitoring, analysis, and remediation.

To integrate observability into your application, see :ref:`nrf_edgeai_obsv_lib`.

Overview
********

Deployed Edge AI models behave differently in production than in controlled environments.
Class distributions shift, transitions between predicted labels change over time, and confidence scores drift as conditions change.
Model observability gives you a structured way to capture these statistics on-device and send them off-device for analysis, so you can tell whether a deployed model is still performing as expected.

The |EAI| implements model observability through the :ref:`nrf_edgeai_obsv_lib`.
The library works with any inference engine that exposes a class-probability vector, an input-feature vector, or both — depending on which built-in or custom metrics you enable — including the :ref:`nrf_edgeai_lib`, :ref:`Axon NPU <lib_axon>`, and `Edge Impulse`_ deployments.

.. note::
   Model observability is designed primarily around |EAILib|, which provides the most seamless integration.
   To get the most out of model observability, consider using |EAILib| for your models.

Use cases
*********

Collected observability data enables the following:

* Model quality monitoring — Tracks if prediction confidence and class frequencies stay within expected bounds after deployment.
* Dataset collection guidance — Identifies which classes are under-represented or confused in the field and targets data collection accordingly.
* Retraining triggers — Detects distribution shift early and decides when a model update is needed before accuracy degrades noticeably.
* A/B testing — Compares metric snapshots from devices running different model versions in production conditions.
* Hardware and environment health signals — Helps investigate whether a shift is model drift, a degrading sensor, or an unexpected deployment environment.
* Software regression detection — Correlates a sudden metric shift with an application update timeline, or rules out post-processing defects when the product misbehaves but metrics stay within bounds.

.. _model_observability_cycle_diagram:

The Edge AI model lifecycle
***************************

A model does not stop evolving once it is deployed.
It moves through a cycle of creation, integration, deployment, observation, and analysis.
Observability is what tells you when to start the cycle again.

The following diagram shows this lifecycle at a glance as a continuous cycle.

.. graphviz::
   :caption: The Edge AI model lifecycle is a continuous cycle.
   :align: center

   digraph cycle {
     layout=circo;
     bgcolor="transparent";
     splines=curved;
     mindist=1.2;
     node [shape=box, style="rounded,filled", color="#0077C8", penwidth=1.5, fontname="Arial", fontsize=13, margin="0.28,0.2"];
     edge [color="#0077C8", penwidth=2, arrowsize=0.9];

     n1 [label="Model training/retraining (1, 6a)", fillcolor="#CFE8FF", fontcolor="#333F48"];
     n2 [label="Application\nintegration (2)", fillcolor="#8CCBFF", fontcolor="#333F48"];
     n3 [label="Deployment (3)", fillcolor="#3AA9F0", fontcolor="#FFFFFF"];
     n4 [label="Observing\nproduction behavior (4)", fillcolor="#0077C8", fontcolor="#FFFFFF"];
     n5 [label="Analyzing\ncollected data (5)", fillcolor="#044570", fontcolor="#FFFFFF"];

     n1 -> n2 -> n3 -> n4 -> n5 -> n1;
   }

The following diagram adds remediation paths (steps 6a–6c) for cases where analysis points outside the model.

.. mermaid::
   :caption: Model observability in the context of the software and product lifecycle.

   %%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#3AA9F0', 'primaryTextColor': '#333F48', 'primaryBorderColor': '#0077C8', 'lineColor': '#0077C8', 'tertiaryColor': '#F5F9FF', 'tertiaryBorderColor': '#8DBEFF', 'edgeLabelBackground': '#F5F9FF'}}}%%
   swimlane-beta TB
       subgraph training[Model training]
           startNode(["Start"]) --> collect["Collect and label data (1)"]
           collect --> train["Train and validate model (1)"]
           analyzeMetrics["Analyze collected metrics (5)"]
           issueDetected{"Issue detected?"}
           rootCause{"Root cause?"}
           analyzeMetrics --> issueDetected
           issueDetected -- yes --> rootCause
           retrainModel["Retrain or fine-tune model (6a)"]
       end

       subgraph integrate[Application integration]
           integrateModel["Integrate model into application (2)"]
           configureObsv["Configure observability metrics (2)"]
           integrateModel --> configureObsv
           updateModel["Update the integrated model (6a)"]
           fixCode["Fix pre-processing or post-processing code (6b)"]
       end

       subgraph production[Production]
           deployProduct["Deploy product to the field (3)"]
           runInference["Run inference and update observability metrics (4)"]
           uploadMetrics["Collect and upload metric snapshots (4)"]
           deployProduct --> runInference --> uploadMetrics
           deployUpdate(["Deploy update to the field (3)"])
           continueMonitoring(["Continue monitoring"])
           reviseHw(["Revise hardware, calibration, or product environment (6c)"])
       end

       train --> integrateModel
       configureObsv --> deployProduct
       uploadMetrics --> analyzeMetrics

       rootCause -- model --> retrainModel --> updateModel --> deployUpdate
       rootCause -- software pipeline --> fixCode --> deployUpdate
       rootCause -- hardware or environment --> reviseHw
       rootCause ~~~ reviseHw
       issueDetected -- no --> continueMonitoring

       classDef decision fill:#8CCBFF,stroke:#0077C8,stroke-width:1.5px,color:#333F48;
       classDef terminal fill:#044570,stroke:#333F48,stroke-width:1.5px,color:#ffffff;
       class issueDetected,rootCause decision
       class startNode,continueMonitoring,deployUpdate,reviseHw terminal

The numbered stages in the diagrams map to the following sections, including analysis and troubleshooting paths when issues are detected in production.

.. rst-class:: numbered-step

Model creation and training
===========================

Create and train a model using one of the following tools, depending on your target hardware and workflow preferences:

* `Nordic Edge AI Lab`_
* :ref:`Axon NPU TFLite compiler <axon_npu_tflite_compiler>`
* `Edge Impulse studio`_

See :ref:`solution_comparison` if you are unsure which tool fits your use case.

.. _model_observability_integration:

.. rst-class:: numbered-step

Application integration
========================

Integrate the trained model into your |NCS| application using one of the following guides:

* :ref:`ug_nrf_edgeai_integration` for models trained with the `Nordic Edge AI Lab`_.
* :ref:`ug_axon_integration` for models compiled directly for the Axon NPU driver.
* :ref:`edge_impulse_integration` for models trained with `Edge Impulse studio`_.

While integrating the model, configure the :ref:`nrf_edgeai_obsv_lib` metrics you want to collect in production.
See :ref:`nrf_edgeai_obsv_lib` for Kconfig options, initialization, and API usage.

.. note::
   |EAILib| does not currently expose the extracted feature vector through its API, so input-feature metrics are not yet available for models integrated through it.
   Support for capturing this data from |EAILib| is planned for a future release.

.. _model_observability_deployment:

.. rst-class:: numbered-step

Deployment
==========

Build and flash your application to your target devices, then ship it as part of your product.
See :ref:`app_ww_kws` for a reference application that integrates model observability through the ``CONFIG_MODELS_OBSERVABILITY`` Kconfig option.

Later deployments — whether they carry a retrained model, an application defect fix, or a hardware-related configuration change — reach devices already in the field through your product's device firmware update (DFU) mechanism.
See :ref:`nrf:app_bootloaders` for the DFU mechanisms available in |NCS|.

If your application uses `Memfault`_ as observability transport, its OTA release management can stage an update to a cohort of devices so you can confirm it performs as expected using the same observability metrics before releasing it to the rest of the fleet.
See `Memfault Over-the-Air Updates`_ for details.

.. note::
   Today, a model is built into the application binary, so updating it means shipping a full application update.
   Support for updating only the model, without re-flashing the rest of the application, is planned.

.. rst-class:: numbered-step

Observing production behavior
=============================

Once the product is in the field, the :ref:`nrf_edgeai_obsv_lib` accumulates metric snapshots from every inference and periodically uploads them off-device for analysis.
The library ships with a Memfault transport that stages snapshots as a `Memfault Custom Data Recording`_ (CDR) uploaded to nRF Cloud, or you can implement a custom transport over any connectivity solution your product already uses.

.. rst-class:: numbered-step

Analyzing collected data
========================

Analyze uploaded metric snapshots to determine whether the deployed model still meets your quality bar, and if not, whether the cause is the model, an application defect, the hardware, or the deployment environment.

.. note::
   The `Nordic Edge AI Lab`_ does not yet provide a way to analyze metric snapshots, but this is planned for future releases.
   For now, download and decode raw payloads with the :ref:`decode_edgeai_obsv_cdr script <nrf_edgeai_obsv_script>` and feed them into your own analysis pipeline.

See :ref:`nrf_edgeai_obsv_metrics_built_in` for the full list of built-in metrics and what each one indicates.

* A gradual trend over weeks or months is typical of drift — whether caused by the environment, changing usage patterns, or the model becoming stale.
* A sudden, sustained shift that starts right after an application update points to a pre-processing regression.
* Visible product misbehavior after an update, with metrics still within bounds, points to a post-processing regression (see :ref:`model_observability_software_pipeline`).

See :ref:`model_observability_investigation_paths` for the detailed patterns behind each of these cases.

.. rst-class:: numbered-step

Addressing the root cause
=========================

Once analysis identifies what is behind an observed issue, address it depending on where the root cause lies.
If the model itself is at fault, retraining closes the loop back into the model lifecycle.
If the root cause is application code or the hardware and deployment environment, the fix happens outside that loop.

.. rst-class:: numbered-step

Retrain or update the model
---------------------------

When analysis indicates that a model update is needed, retrain or fine-tune the model using the same tool you used to create it, guided by the dataset gaps and confusion patterns that observability data revealed.
Then repeat :ref:`application integration <model_observability_integration>` for the updated model and deploy it to devices in the field (see :ref:`model_observability_deployment`).

A model update does not always require full retraining.
Sometimes observability data instead points to a configuration issue — for example, a stale anomaly threshold or a change in feature scaling percentiles — that can be corrected without changing the model itself.

.. _model_observability_software_pipeline:

.. rst-class:: numbered-step

Fix pre-processing or post-processing defects
---------------------------------------------

When a software regression — rather than model or hardware drift — is behind an observed issue, fix the corresponding code path:

* Pre-processing — for example, correct an error in sample windowing that no longer matches what the model was trained on.
* Post-processing — for example, correct a decision threshold applied to the wrong class index.

Deploy the corrected application to devices in the field (see :ref:`model_observability_deployment`).

.. _model_observability_hardware_environment:

.. rst-class:: numbered-step

Revise hardware or the deployment environment
---------------------------------------------

Some issues that observability data reveals cannot be fixed by changing software alone.
A failing sensor, an enclosure that distorts the signal, or a deployment environment consistently different from what the product was designed for are hardware or product design problems.

Depending on the root cause, the fix might be replacing or recalibrating the affected sensor, revising the product's mechanical or electrical design, or updating installation guidance for the deployment environment.
Once you revise your design and introduce improvements, continue observing the product to confirm that the metrics return to expected levels.

.. _model_observability_investigation_paths:

Diagnosing production issues
****************************

Not every quality issue that observability data reveals traces back to the model itself.
The following patterns are paths to explore when metrics move outside expected bounds.
Which one applies depends on your application, sensors, and deployment context.

Model drift
===========

Gradual shifts in class distributions, confidence scores, or streak lengths over weeks or months can suggest that the model no longer matches real-world conditions.
Examples to look for include a rising :ref:`prediction switching rate <nrf_edgeai_obsv_metrics_built_in_switching>`, a :ref:`probability distribution <nrf_edgeai_obsv_metrics_built_in_probability>` that drifts away from the training-time baseline, or a :ref:`class streak distribution <nrf_edgeai_obsv_metrics_built_in_streak>` dominated by short, flickering streaks instead of stable detections.

Hardware and deployment environment
===================================

A sustained drop in confidence or a rise in ambiguous predictions can also come from degrading or poor-quality hardware — for example, a microphone that no longer picks up enough signal — or from a deployment environment the model was not trained for — for example, a location with more background noise than the training data.
If investigation points here rather than at the model, the fix may lie in revising the hardware or the product's environment, not retraining (see :ref:`model_observability_hardware_environment`).

Input-feature metrics, such as the :ref:`mel energy descriptor <nrf_edgeai_obsv_metrics_built_in_mel_energy>` and :ref:`mel spectral descriptor <nrf_edgeai_obsv_metrics_built_in_mel_spectral>`, observe the raw feature vector fed into the model rather than its output.
A sustained drop in mean energy across many devices can suggest a sensor issue rather than the model, while a shift in spectral band ratios can indicate a noisier acoustic environment.
If the shift appears on all devices at once, right after an update, pre-processing code is a more likely place to look than the sensors themselves.

Application pipeline issues
===========================

Because metrics are computed on the probability and feature vectors at inference time, a pre-processing bug can distort them in ways that look like model drift — typically as a sudden change that coincides with a firmware update rather than a slow trend.
This includes defects that basic input validation would not catch, for example an accelerometer's full-scale range shifting from 2g to 16g without the application accounting for it.

A post-processing issue downstream of inference — for example, a decision threshold applied to the wrong class index — does not distort these metrics directly, but can surface indirectly when the product visibly misbehaves in the field while the metrics stay within expected bounds (see :ref:`model_observability_software_pipeline`).

Real-world usage patterns
=========================
Observability data also reflects how people actually use the product, which does not always match the assumptions made while collecting training data.
For example, a wearable held or worn at unexpected angles can show up as class confusion (see :ref:`nrf_edgeai_obsv_metrics_built_in_transition`) or a higher switching rate.
Likewise, a keyword-spotting model can show low decision confidence or repeatedly fail to reach a confident, held detection for a command, consistent with real-world phrasing or accents the model was not trained on.

Depending on the case, one path forward is to retrain with data that better reflects real usage; another is to adjust the product itself (for example, clearer user guidance or ergonomics) rather than the model.

Next steps
***********

* :ref:`nrf_edgeai_obsv_lib` — Library reference, built-in metrics, Kconfig, and integration steps.
* :ref:`app_ww_kws` — Reference application with option to enable model observability.
* :ref:`integrations` — Model integration guides for Nordic Edge AI Lab, Axon, and Edge Impulse.
* :ref:`decode_edgeai_obsv_cdr script <nrf_edgeai_obsv_script>` — Host-side payload decoder.
