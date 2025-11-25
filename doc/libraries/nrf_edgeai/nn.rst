nRF Edge AI NN
===============

Neural network inference engine for different model architectures
This module provides a small neural-network abstraction and a lightweight
implementation for Neuton-style and other networks (both floating-point and quantized
variants).

Overview
--------

The NN module supports running compact feed-forward models. Two levels are provided:

- A thin abstraction layer (``nrf_nn.h``) that exposes available implementations
	to the rest of the Edge AI library.
- A specific implementation for Neuton models (``neuton/nrf_nn_neuton.h``)
	including helpers for q8, q16 and f32 models and low-level raw inference
	entry points.

Module structure
----------------

Files are located under ``include/nrf_edgeai/nn/``. Important headers:

- :file:`include/nrf_edgeai/nn/nrf_nn.h` — Aggregating include for the NN module.
- :file:`include/nrf_edgeai/nn/neuton/nrf_nn_neuton.h` — Neuton model structures
	and inference APIs.

Key types and metadata (Neuton)
-------------------------------

The Neuton implementation exposes compact model descriptors that store model
topology and pointers to weight arrays. Important structures:

- ``nrf_nn_neuton_meta_t`` — Read-only metadata describing neuron link tables,
	indices of output neurons, counts of neurons/weights and activation mask.
- ``nrf_nn_neuton_model_q8_t`` — Descriptor for 8-bit quantized models.
- ``nrf_nn_neuton_model_q16_t`` — Descriptor for 16-bit models.
- ``nrf_nn_neuton_model_f32_t`` — Descriptor for floating-point models.

Inference API
-------------

High-level convenience functions execute inference and store results in the
model's neuron buffer:

- ``nrf_nn_neuton_run_inference_q8`` — Run inference for q8 models.
- ``nrf_nn_neuton_run_inference_q16`` — Run inference for q16 models.
- ``nrf_nn_neuton_run_inference_f32`` — Run inference for f32 models.

For advanced use-cases (custom memory layouts, off-line evaluation) raw
inference entry points are available that accept explicit pointers to the
metadata and arrays instead of a typed model descriptor. Examples:

- ``nrf_nn_neuton_run_inference_raw_f32``
- ``nrf_nn_neuton_run_inference_raw_q8``
- ``nrf_nn_neuton_run_inference_raw_q16``

These raw functions accept the neurons buffer, neuron links, weights and
activation masks directly and write results to the supplied neurons buffer.

Usage pattern
-------------

Typical usage for an F32 Neuton model:

.. code-block:: c

	 #include <nrf_edgeai/nn/nrf_nn.h>
	 #include <nrf_edgeai_generated/nrf_edgeai_user_model.h>

     static nrf_nn_neuton_model_f32_t model = { /* ... */ };

	 void run_model(void)
	 {
			 /* Inputs are prepared according to the model's expected ordering */
			 flt32_t inputs[INPUTS_NUM] = { /* ... */ };

			 nrf_nn_neuton_run_inference_f32(&model, inputs, INPUTS_NUM);

			 /* Results are stored in model.p_neurons or exposed by the wrapper */
	 }

Notes and considerations
------------------------

- The Neuton runner operates in-place on a neurons buffer supplied in the
	model descriptor. Memory for the neurons buffer must be provided by the
	application or by the generated model sources.
- Use the quantized (q8/q16) APIs for memory- and flash-constrained targets
	when models are exported in fixed-point formats.
- The raw inference APIs are useful for unit-testing, model introspection or
	for running models outside the standard descriptor layout.

See also
--------

- Header files: :file:`include/nrf_edgeai/nn/`
- Runtime integration: :ref:`nrf_edgeai_lib`
