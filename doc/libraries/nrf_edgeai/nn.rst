.. _nrf_edgeai_lib_nn:

nRF Edge AI neural network module
#################################

.. contents::
   :local:
   :depth: 2

The Neural Network (NN) inference engine offers a compact abstraction layer and lightweight implementation for running Neuton-style and other neural networks, supporting both floating-point and quantized variants.

Overview
********

The NN module supports running compact feed-forward models.
It consists of two levels:

* A thin abstraction layer (:file:`nrf_nn.h`) that exposes the available implementations to the rest of the Edge AI library.
* A specific implementation for Neuton models (:file:`neuton/nrf_nn_neuton.h`), including helpers for q8, q16, and f32 models, as well as low-level raw inference entry points.

The Neuton runner operates in-place on a neurons buffer defined in the model descriptor.
The application or the generated model sources must allocate and provide memory for this buffer.
Use the quantized (q8/q16) APIs for devices with limited memory and flash, when models are exported in fixed-point formats.
Additionally, you can use raw inference APIs for unit testing, model introspection, or running models outside the standard descriptor layout.

Module structure
****************

Files are located in the :file:`include/nrf_edgeai/nn` directory, and include the following headers:

* :file:`include/nrf_edgeai/nn/nrf_nn.h` - Aggregating include for the NN module
* :file:`include/nrf_edgeai/nn/neuton/nrf_nn_neuton.h` - Neuton model structures and inference APIs

Key types and metadata
======================

The Neuton implementation provides compact model descriptors that store model topology and pointers to weight arrays.
The most important structures are:

* :c:func:`nrf_nn_neuton_meta_t` — Read-only metadata describing neuron link tables, indices of output neurons, counts of neurons and weights, and activation mask
* :c:func:`nrf_nn_neuton_model_q8_t` — Descriptor for 8-bit quantized models
* :c:func:`nrf_nn_neuton_model_q16_t` — Descriptor for 16-bit models
* :c:func:`nrf_nn_neuton_model_f32_t` — Descriptor for floating-point models

Inference API
=============

High-level convenience functions perform inference and store results in the model’s neuron buffer:

* :c:func:`nrf_nn_neuton_run_inference_q8` — Run inference for q8 models
* :c:func:`nrf_nn_neuton_run_inference_q16` — Run inference for q16 models
* :c:func:`nrf_nn_neuton_run_inference_f32` — Run inference for f32 models

For advanced use cases, such as custom memory layouts or offline evaluation, you can use raw inference entry points.
These functions accept explicit pointers to the metadata and arrays, instead of a typed model descriptor.
For example:

* :c:func:`nrf_nn_neuton_run_inference_raw_f32`
* :c:func:`nrf_nn_neuton_run_inference_raw_q8`
* :c:func:`nrf_nn_neuton_run_inference_raw_q16`

These raw functions take the neurons buffer, neuron links, weights, and activation masks directly, and write results to the provided neurons buffer.

Usage pattern
-------------

See a typical usage for an F32 Neuton model:

.. code-block:: c

	#include <nrf_edgeai/nn/nrf_nn.h>
	#include <nrf_edgeai_generated/nrf_edgeai_user_model.h>

	static nrf_nn_neuton_model_f32_t model = { /* ... */ };

	void run_model(void)
	{
		/* Prepare the input array in the order expected by the model */
		flt32_t inputs[INPUTS_NUM] = { /* ... */ };

		nrf_nn_neuton_run_inference_f32(&model, inputs, INPUTS_NUM);

		/* Results are stored in model.p_neurons or accessed through wrapper functions */
	}
