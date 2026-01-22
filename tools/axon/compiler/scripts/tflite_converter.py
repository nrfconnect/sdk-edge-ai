""" 
/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */
"""
import logging
import numpy as np
import tensorflow as tf
import utility.util as util


def get_tflite_model_ip_shape(tflite_path):
    """
    get the model input shape as a shape array
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    # Evaluate the TFLite model using "test" dataset.
    input_index = interpreter.get_input_details()[0]["index"]
    # find the expected shape of the input test vector here
    input_shape = interpreter.get_tensor_details()[input_index]['shape']
    return input_shape

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


def test_tflite_model(tflite_path, quantized_x_test, y_test=None, classification_model=True, get_results=False):
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
    # first we quantize the input dataset.
    if quantized_x_test.dtype != np.int8:
        quantized_x_test = quantize_test_dataset(tflite_path, quantized_x_test)
    # Test the accuracy of fixed model:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    # Evaluate the TFLite model using "test" dataset.
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every sample in the "test" dataset.
    prediction_results = []
    prediction_digits = []
    for test_ in quantized_x_test:
        test_ = np.expand_dims(test_, axis=0)
        interpreter.set_tensor(input_index, test_)

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


def test_floating_point_model(keras_model_path, x_test, y_test=None, classification_model=True, get_results=False):
    # logger = logging.getLogger(__name__)
    # check accuracy of keras float model.
    model = tf.keras.models.load_model(keras_model_path)
    prediction_results = model.predict(x_test, verbose=0)
    prediction_indexes = []
    if y_test is None:
        if (get_results):
            return None, np.array(prediction_results).squeeze()
        return None
    if (classification_model):
        prediction_indexes = np.argmax(prediction_results, axis=1)
        acc_float = np.mean(y_test == prediction_indexes)
    else:
        prediction_indexes = prediction_results
        errors = np.sqrt(np.square(y_test - prediction_results))
        acc_float = np.mean(errors)
    # print("The accuracy of the floating model is "+str(acc_float))
    if (get_results):
        return acc_float, prediction_indexes
    return acc_float


def quantize_test_dataset(tflite_model_path, x_test):
    # first we quantize the input dataset, save it in a text file which can be given as an input to the simulator exe
    # logger = logging.getLogger(__name__)
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    if input_details[0]['dtype'] == np.int8:
        scale = input_details[0]['quantization_parameters']["scales"][0]
        zeropoints = input_details[0]['quantization_parameters']['zero_points'][0]
        data_q = ((x_test/scale)+zeropoints).astype(np.int8)
    else:  # TODO the input type of the model is float and this needs extra handling
        return x_test
    return data_q
