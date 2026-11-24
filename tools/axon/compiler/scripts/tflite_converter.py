"""
Copyright (c) 2026 Nordic Semiconductor

SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""
import os
import logging
import numpy as np
import tensorflow as tf
import utility.util as util


def get_tflite_model_ip_shapes(tflite_path):
    """
    get the model input shape as a shape array
    """
    input_shapes = []
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    # Evaluate the TFLite model using "test" dataset.
    for inputs in interpreter.get_input_details():
        # find the expected shape of the input test vector here
        input_shapes.append(inputs['shape'])
    return input_shapes

# Conversion of keras file to tflite file:


def tflite_conversion(x_train, keras_file):
    # logger = logging.getLogger(__name__)
    """
    Converts the keras file to TFlite quantized model.

    Args:
    x_train : loaded train dataset.
    keras_file : directory of the model. could either be keras file (ending with ".h5") or saved model (a folder including ".pb" file and variables.)
    Tflite_dir : directory to save tflite model, must include desired tfite file name ending with ".tflite".
    Example: Desktop/ImagRec.tflite


    Returns:
    tflite file from keras model.
    """

    x_train = x_train.astype(np.float32)

    def representative_dataset_gen():
        # using a small set of train dataset to get the min and max range of input data.
        for i in range(int(0.1*x_train.shape[0])):
            image = np.expand_dims(x_train[i], axis=0)
            yield [image]

    if keras_file.endswith(".h5"):
        model = tf.keras.models.load_model(keras_file)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(keras_file)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    # Create an end to end integer 8 bit quantized tflite.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # integer 8bit input and output when creating tflite.
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    # check here for the input shapes of the train data and the model input
    if x_train.shape != model.input_shape:
        if (len(model.input_shape) == 4):
            x_train = x_train.reshape(
                len(x_train), model.input_shape[1], model.input_shape[2], model.input_shape[3])

    tflite_quant_model = converter.convert()
    # tf.lite.experimental.Analyzer.analyze(model_content=tflite_quant_model, gpu_compatibility=True)
    return tflite_quant_model


def Test_TFliteModel(Tflite_dir, x_testQ, x_test, y_test, keras_file):
    logger = logging.getLogger(__name__)
    """
    8_bit quantization for float input dataset, using scale and zero points form tflite.
    Tests the accuracy of tflite model created.

    Args:

    Tflite_dir : saved tflite directory, must include desired tfite file name ending with ".tflite".
    x_testQ : directory to save quantized dataset, should be ".npy" and include the name of the file.
    Example: "Desktop/Qdata.npy"
    x_test : loaded test dataset array
    y_test : Array of integer values, representing dataset's classes.
    keras_file : directory of the model. could either be keras file (ending with ".h5") or saved model (a folder including ".pb" file and variables.)


    Returns:
    saves quantized test vectors.
    returns the accuracy of the float and tflite model.

    """

    # first we quantize the input dataset, save it in x_testQ as numpy array.
    interpreter = tf.lite.Interpreter(model_path=Tflite_dir)
    interpreter.allocate_tensors()
    scale = interpreter.get_input_details(
    )[0]['quantization_parameters']["scales"][0]
    zeropoints = interpreter.get_input_details(
    )[0]['quantization_parameters']['zero_points'][0]
    data_q = ((x_test/scale)+zeropoints).astype(np.int8)
    np.save(x_testQ, data_q)

    # Test the accuracy of fixed model:
    # Evaluate the TFLite model using "test" dataset.
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every sample in the "test" dataset.
    prediction_digits = []
    for test_ in data_q:
        test_ = np.expand_dims(test_, axis=0)
        interpreter.set_tensor(input_index, test_)

        # Run inference.
        interpreter.invoke()

        # Save the class predictions for all test samples.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)
    # check accuracy of keras float model.
    model = tf.keras.models.load_model(keras_file)
    acc_float = np.mean(np.load(y_test, allow_pickle=True)
                        == np.argmax(model.predict(x_test), axis=1))
    # print("The accuracy of the floating model is "+str(acc_float))
    logger.info("The accuracy of the floating model is "+str(acc_float))
    # Check accuracy of fixed point model.
    acc_tflite = np.mean(prediction_digits ==
                         np.load(y_test, allow_pickle=True))
    # print ("The accuracy of the tflite model is: "+str(acc_tflite))
    logger.info("The accuracy of the tflite model is: "+str(acc_tflite))

    return acc_float, acc_tflite


def get_tflite_model_test_results(tflite_path, quantized_x_test, y_test=None, classification_model=True, get_results=False, tflite_interpreter=None):
    # logger = logging.getLogger(__name__)
    """
    8_bit quantization for float input dataset, using scale and zero points form tflite.
    Tests the accuracy of tflite model created.

    Args:

    Tflite_dir : saved tflite directory, must include desired tfite file name ending with ".tflite".
    x_testQ : directory to save quantized dataset, should be ".npy" and include the name of the file.
    Example: "Desktop/Qdata.npy"
    x_test : loaded test dataset array
    y_test : Array of integer values, representing dataset's classes.
    keras_file : directory of the model. could either be keras file (ending with ".h5") or saved model (a folder including ".pb" file and variables.)


    Returns:
    saves quantized test vectors.
    returns the accuracy of the float and tflite model.

    """
    # Test the accuracy of fixed model:
    if tflite_interpreter == None:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
    else:
        interpreter = tflite_interpreter
    input_details = interpreter.get_input_details()
    output_index = interpreter.get_output_details()[0]["index"]

    # Accept legacy single ndarray or list per TFLite input order
    if isinstance(quantized_x_test, np.ndarray):
        x_list = [quantized_x_test]
    else:
        x_list = quantized_x_test

    data_q = []
    for ndx, model_inputs in enumerate(input_details):
        arr = np.asarray(x_list[ndx])
        ip_dtype = model_inputs["dtype"]
        if arr.dtype != ip_dtype:
            if ip_dtype in (np.int8, np.uint8):
                scale = model_inputs["quantization_parameters"]["scales"][0]
                zeropoints = model_inputs["quantization_parameters"]["zero_points"][0]
                arr = ((arr / scale) + zeropoints).astype(ip_dtype)
            else:
                arr = arr.astype(ip_dtype)
        data_q.append(arr)

    # Run predictions on every sample in the "test" dataset.
    num_samples = int(data_q[0].shape[0])
    prediction_results = []
    prediction_digits = []
    for n in range(num_samples):
        for ndx, model_inputs in enumerate(input_details):
            test_ = util.check_input_shape_for_inference(
                data_q[ndx][n], model_inputs['shape'])
            interpreter.set_tensor(model_inputs["index"], test_)

        # Run inference.
        interpreter.invoke()

        # Save the class predictions for all test samples.
        output = interpreter.get_tensor(output_index)
        prediction_results.append(output)
    # Check accuracy of fixed point model.
    if y_test is None:
        if (get_results):
            return None, np.array(prediction_results).squeeze()
        return None
    if (classification_model):
        prediction_digits = np.argmax(prediction_results, axis=2)
        if (len(prediction_digits.shape) > 1):
            prediction_digits = prediction_digits.reshape(
                prediction_digits.shape[0])
        acc_tflite = np.mean(prediction_digits.squeeze() == y_test)
    else:
        prediction_digits = prediction_results
        errors = np.sqrt(
            np.square(y_test - np.array(prediction_results).squeeze()))
        acc_tflite = np.mean(errors)
    # print ("The accuracy of the tflite model is: "+str(acc_tflite))

    if (get_results):
        return acc_tflite, prediction_digits
    return acc_tflite


def test_scaled_model_on_simulator(tflite_model_path, x_test, y_test, simulator_exe_path, get_results=False):
    """
    will test the model running on the simulator and give out the accuracy
    """
    # logger = logging.getLogger(__name__)
    test_vector_text = ""
    test_labels_text = ""
    test_results_text = ""
    test_labels_ndx = 0
    x_test_q = quantize_test_dataset(tflite_model_path, x_test)
    for test_vectors in x_test_q:
        #   test_vector_text+="{"
        test_vector_text += np.array2string(test_vectors.squeeze(), separator=',',
                                            max_line_width=1000).replace('[', '').replace(']', '').replace('\n', '')
        test_vector_text += "\n"
        test_labels_text += str(y_test[test_labels_ndx])+"\n"
        test_labels_ndx += 1
    test_labels_text += "\n"
    test_vector_text = test_vector_text.replace(' ', '')
    util.save_to_file(simulator_exe_path,
                      "simulator_test_vectors.txt", test_vector_text)
    util.save_to_file(simulator_exe_path,
                      "simulator_test_label.txt", test_labels_text)
    util.save_to_file(simulator_exe_path,
                      "simulator_test_results.txt", test_results_text)

    # we saved the test vectors in a format that the simulator can take as an input
    _ = util.run_win_app(simulator_exe_path, simulator_exe_path+"\\simulator_test_vectors.txt",
                         simulator_exe_path+"\\simulator_test_results.txt", simulator_exe_name="axonpro_app.exe")

    calculated_results_path = simulator_exe_path+"\\labels.txt"
    calculated_results = np.loadtxt(
        fname=calculated_results_path, dtype=np.int32, delimiter="\n")
    accuracy = np.mean(calculated_results == y_test)
    # print(accuracy)
    # print ("The accuracy of the scaled model on simulator is: "+str(accuracy))
    if (get_results):
        return accuracy, calculated_results
    return accuracy


def is_keras_model_path(value):
    """True if value looks like a filesystem path to load, not an in-memory model."""
    if value is None or value == "":
        return False
    return isinstance(value, (str, os.PathLike))


def is_loaded_keras_model(value):
    """True if value is already a Keras model instance."""
    if value is None:
        return False
    if isinstance(value, (str, bytes, os.PathLike)):
        return False
    return isinstance(value, tf.keras.Model)


def load_keras_model_if_needed(keras_model_or_path):
    """
    Returns a loaded tf.keras.Model.
    - Already loaded -> return as-is
    - str / Path -> load from disk
    """
    if is_loaded_keras_model(keras_model_or_path):
        return keras_model_or_path
    if not is_keras_model_path(keras_model_or_path):
        raise TypeError(
            f"Expected Keras model or path, got {type(keras_model_or_path)!r}"
        )
    path = os.fspath(keras_model_or_path)
    if path.endswith((".h5", ".keras")):
        return tf.keras.models.load_model(path)
    if os.path.isdir(path):
        # SavedModel folder (.pb + variables)
        return tf.keras.models.load_model(path)
    raise ValueError(
        f"Keras path is not .h5, .keras, or an existing directory: {path!r}"
    )


def test_floating_point_model(keras_model_or_path, x_test, y_test=None, classification_model=True, get_results=False):
    """
    tests the floating point model.
    returns the accuracy of the model if y_test is provided and it is a classification model,
    model accuracy
    presiction results
    prediction_indices (if a classification model)
    """
    # check here if the keras_model_path is a variable, if not it must be the loaded model already

    model = load_keras_model_if_needed(keras_model_or_path)

    if isinstance(x_test, np.ndarray):
        x_test = [x_test]
    assert len(x_test) == len(model.inputs), (
        f"floating point input data and test data lengths do not match, expected {len(model.inputs)} input array(s), got {len(x_test)}")
    # Keras: one tensor for single-input, list of tensors for multi-input
    x_for_predict = x_test[0] if len(model.inputs) == 1 else list(x_test)
    prediction_results = model.predict(x_for_predict, verbose=0)
    prediction_indexes = []
    if y_test is None:
        if (get_results):
            return None, np.array(prediction_results).squeeze()
        return None
    if (classification_model):
        prediction_indexes = np.argmax(prediction_results, axis=1)
        acc_float = np.mean(y_test == prediction_indexes)
    else:  # calculates mean error if it is not a classification model.
        prediction_indexes = prediction_results
        errors = np.sqrt(np.square(y_test - prediction_results))
        acc_float = np.mean(errors)
    # print("The accuracy of the floating model is "+str(acc_float))
    if (get_results):
        return acc_float, prediction_indexes
    return acc_float


def quantize_test_dataset(tflite_model_path, x_test, input_index=0, tflite_interpreter=None):
    # first we quantize the input dataset, save it in a text file which can be given as an input to the simulator exe
    # logger = logging.getLogger(__name__)

    if tflite_interpreter == None:
        tflite_interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        tflite_interpreter.allocate_tensors()
    else:
        if not isinstance(tflite_interpreter, tf.lite.Interpreter):
            raise TypeError(
                "Expected tf.lite.Interpreter; "
                f"got {type(tflite_interpreter)!r}")

    input_details = tflite_interpreter.get_input_details()
    if input_details[input_index]['dtype'] == np.int8:
        scale = input_details[input_index]['quantization_parameters']["scales"][0]
        zeropoints = input_details[input_index]['quantization_parameters']['zero_points'][0]
        data_q = ((x_test/scale)+zeropoints).astype(np.int8)
    else:  # TODO the input type of the model is float and this needs extra handling
        return x_test
    return data_q


def generate_representative_dataset(model_type, train_data, input_signatures):
    """
    Returns a calibration generator tailored to the specific model architecture.
    """
    # if model_type in ("Lstm"):
    #     # LSTMs need more samples to properly calibrate the recurring loops
    #     def generator():
    #         for i in range(int(train_data.shape[0])):
    #             image = np.expand_dims(train_data[i], axis=0)
    #             yield [image]
    #         # Generate more data that spans the full operational spectrum
    #         for _ in range(200):
    #             data = np.random.uniform(-1.0, 1.0,
    #                                      size=(1, *train_data.shape[1:]))
    #             yield [data.astype(np.float32)]
    # elif model_type in ("MIMO"):
    #     def generator():
    #         for i in range(int(train_data[0].shape[0])):
    #             input1 = np.expand_dims(train_data[0][i], axis=0)
    #             input2 = np.expand_dims(train_data[1][i], axis=0)
    #             yield [input1, input2]
    # else:
    #     def generator():
    #         for i in range(int(train_data.shape[0])):
    #             image = np.expand_dims(train_data[i], axis=0)
    #             yield [image]

    def generator():
        num_samples = int(train_data[0].shape[0])
        for n in range(num_samples):
            yield {input_signatures[i]: np.expand_dims(train_data[i][n], axis=0).astype(np.float32) for i in range(len(train_data))}

        if model_type in ("Lstm"):
            for _ in range(200):
                yield {input_signatures[i]: np.random.uniform(-1.0, 1.0,
                                                              size=(1, *train_data[i].shape[1:])).astype(np.float32) for i in range(len(train_data))}
    return generator


def convert_tf_to_tflite(model_type, model, train_data, model_path="", disable_fc_chan_q=False, input_signatures=None, act_bits=8):
    """
    supports conversion from keras as well as saved model
    inputs the model and the train data
    outputs the quantized tflite model
    """
    # Convert to TFLite
    if model_path.endswith(".h5") or model_path.endswith(".keras"):
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

    # DYNAMIC CONFIGURATION GUARD
    # Check if the active framework pointer belongs to the legacy engine
    if tf.keras.__name__ == "tf_keras":
        print("[CONVERTER] Legacy Keras 2 detected. Enabling resource variables pass...")
        converter.experimental_enable_resource_variables = True
        print("[CONVERTER] Disabling break down loop-based sequence structures into standard low-level tensor lists...")
        converter._experimental_lower_tensor_list_ops = False

    # Optional: enable optimization to encourage fusion
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Provide a representative dataset for calibration
    converter.representative_dataset = generate_representative_dataset(
        model_type, train_data, input_signatures)

    # Flag to disable per-channel quantization for dense layers (for testing)
    converter._experimental_disable_per_channel_quantization_for_dense_layers = disable_fc_chan_q

    # Set quantization bits
    if act_bits == 16:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        converter.inference_input_type = tf.int16
        converter.inference_output_type = tf.int16
    else:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"[CONVERTER] Error: {e}")
        tflite_model = None

    return tflite_model


def load_keras_model(keras_model_path):
    return tf.keras.models.load_model(keras_model_path)
