""" 
/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */
"""
import os
# Suppress all logs (INFO, WARNING, and ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["MLIR_ENABLE_CRASH_REPRODUCER"] = "0"
from compile_model_helper import generate_compiler_outputs, run_compiler_library
from utility import cpu_operator_options as cpu_operator_options
from utility import operator_options as ops
from utility import util
from pathlib import Path
import compare_models as compare_models
import tflite_converter as tc
import tensorflow as tf
import tflite as tflite
import ctypes as ctypes
import datetime as dt
import numpy as np
import platform
import logging
import time
import copy
import sys
import gc

# from contextlib import contextmanager
# import tempfile

"""
Code to enable debugging when running the executor from within the docker container!
"""
# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for client to attach...")
# debugpy.wait_for_client()

tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
"""create logger object"""
logger = logging.getLogger(__name__)

COMPILER_VERBOSE = 1
GET_PER_CLASS_PRECISION = True
GET_INTERMEDIATE_FILES = False

USER_WORK_DIR = os.getcwd()
TOOL_ROOT_DIR = os.path.dirname(__file__)
COMPILER_TYPES_HDR_FILE = Path(
    TOOL_ROOT_DIR + "/../include/nrf_axon_nn_compiler_types.h").absolute()
AXON_COMPILER_OBJECT = util.get_axon_compiler_object_filepath(TOOL_ROOT_DIR)
# print(AXON_COMPILER_OBJECT)#DEBUG


def axons_compiler(parsed_dict):
    subprocess_return_code = -900
    compiler_return_codetext = "generic error code!"
    USE_MINIMUM_TEST_VECTORS_DEFINE = "AXON_MINIMUM_TEST_VECTORS"
    test_vectors_flag = False

    """Handle None or empty parameters"""
    # for key in parsed_dict:
    #   if(parsed_dict[key] is None):
    #     parsed_dict[key]=""

    """make necessary directories"""
    intermediate_outputs_dir = USER_WORK_DIR + \
        "/intermediate/"  # + parsed_dict['model_name'].lower() + "/"
    # + parsed_dict['model_name'].lower() + "/"
    compiler_outputs_dir = USER_WORK_DIR + "/outputs/"
    # relative_compiler_outputs_dir = os.path.relpath(compiler_outputs_dir)
    # relative_intermediate_outputs_dir = os.path.relpath(intermediate_outputs_dir)

    if not (Path(intermediate_outputs_dir).exists()):
        Path(intermediate_outputs_dir).mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"created intermediate outputs directory {intermediate_outputs_dir}")
    if not Path(compiler_outputs_dir).exists():
        Path(compiler_outputs_dir).mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"created compiler outputs directory {intermediate_outputs_dir}")

    """handle unset params"""
    if 'tflite_model' not in parsed_dict:
        parsed_dict['tflite_model'] = None
    if 'model_name' not in parsed_dict:
        parsed_dict['model_name'] = None
    if 'train_data' not in parsed_dict:
        parsed_dict['train_data'] = None
    if 'float_model' not in parsed_dict:
        parsed_dict['float_model'] = None
    if 'test_data' not in parsed_dict:
        parsed_dict['test_data'] = None
    if 'test_labels' not in parsed_dict:
        parsed_dict['test_labels'] = None
    if 'test_vectors' not in parsed_dict:
        parsed_dict['test_vectors'] = None
    # if 'cpu_op_codes_list' not in parsed_dict:
    #     parsed_dict['cpu_op_codes_list'] = None
    if 'interlayer_buffer_size' not in parsed_dict:
        parsed_dict['interlayer_buffer_size'] = None
    if 'psum_buffer_size' not in parsed_dict:
        parsed_dict['psum_buffer_size'] = None
    if 'op_radix' not in parsed_dict:
        parsed_dict['op_radix'] = None
    if 'get_quantized_data' not in parsed_dict:
        parsed_dict['get_quantized_data'] = None
    if 'header_file_test_vector_cnt' not in parsed_dict:
        parsed_dict['header_file_test_vector_cnt'] = None
    if 'user_handle_accuracy_results' not in parsed_dict:
        parsed_dict['user_handle_accuracy_results'] = None
    if 'user_handle_test_labels' not in parsed_dict:
        parsed_dict['user_handle_test_labels'] = None
    if 'test_labels_format' not in parsed_dict:
        parsed_dict['test_labels_format'] = None
    if 'log_level' not in parsed_dict:
        parsed_dict['log_level'] = "info"
        if (logging.getLogger().getEffectiveLevel() == logging.DEBUG):
            parsed_dict['log_level'] = "debug"
    if 'generate_sim_env_hdrs' not in parsed_dict:
        parsed_dict['generate_sim_env_hdrs'] = None
    if 'precision_threshold' not in parsed_dict:
        parsed_dict['precision_threshold'] = 0.0
    if 'precision_margin' not in parsed_dict:
        parsed_dict['precision_margin'] = 0
    # if 'run_threshold_regression' not in  parsed_dict:
    #   parsed_dict['run_threshold_regression'] = False
    if 'reshape_input' not in parsed_dict:
        parsed_dict['reshape_input'] = False
    # handle all variant settings
    if 'psum_buffer_placement' not in parsed_dict:
        parsed_dict['psum_buffer_placement'] = None
    if 'conv2d_setting' not in parsed_dict:
        parsed_dict['conv2d_setting'] = None
    if 'disable_op_quantization' not in parsed_dict:
        parsed_dict['disable_op_quantization'] = None
    if 'normalize_scaleshift' not in parsed_dict:
        parsed_dict['normalize_scaleshift'] = None
    if 'skip_softmax_op' not in parsed_dict:
        parsed_dict['skip_softmax_op'] = None
    if 'run_all_variants' not in parsed_dict:
        parsed_dict['run_all_variants'] = None
    if 'transpose_kernel' not in parsed_dict:
        parsed_dict['transpose_kernel'] = None

    """set the requested log level"""
    if parsed_dict['log_level'] == "debug":
        log_lvl = logging.DEBUG
    elif parsed_dict['log_level'] == "warn":
        log_lvl = logging.WARN
    elif parsed_dict['log_level'] == "error":
        log_lvl = logging.ERROR
    elif parsed_dict['log_level'] == "critical":
        log_lvl = logging.CRITICAL
    else:
        log_lvl = logging.INFO
    # just do a fresh setup of log levels for all the handlers here once

    tflogger = tf.get_logger()
    tflogger.disabled = True
    tflogger.setLevel(logging.FATAL)
    logger.setLevel(log_lvl)
    logging.getLogger().setLevel(log_lvl)
    for handlers in logger.handlers:
        handlers.setLevel(log_lvl)
    """
  logger.debug("testing debug log messages") 
  logger.info("testing info log messages")   
  logger.warning("testing warning log messages")  
  logger.error("testing error log messages")
  logger.critical("testing critical log messages")
  """

    """set the assertions"""
    try:
        if not (parsed_dict['train_data'] is None or parsed_dict['train_data'] == ""):
            assert parsed_dict['train_data'].endswith(
                ".npy"), "Train dataset is not in a .npy format. "
            parsed_dict['train_data'] = util.append_user_workspace(
                parsed_dict['train_data'], USER_WORK_DIR)

        if not (parsed_dict['float_model'] is None or parsed_dict['float_model'] == ""):
            assert parsed_dict['float_model'].endswith(".h5") or os.path.isdir(
                parsed_dict['float_model']), "Keras/saved model is not valid. "
            parsed_dict['float_model'] = util.append_user_workspace(
                parsed_dict['float_model'], USER_WORK_DIR)

        if not (parsed_dict['tflite_model'] is None or parsed_dict['tflite_model'] == ""):
            assert parsed_dict['tflite_model'].endswith(".tflite") or parsed_dict['tflite_model'].endswith(
                ".lite"), "Tflite file is not ending with .tflite/.lite"
            parsed_dict['tflite_model'] = util.append_user_workspace(
                parsed_dict['tflite_model'], USER_WORK_DIR)

        if not (parsed_dict['test_data'] is None or parsed_dict['test_data'] == ""):
            assert parsed_dict['test_data'].endswith(
                ".npy"), "Test dataset is not in a \".npy\" format."
            parsed_dict['test_data'] = util.append_user_workspace(
                parsed_dict['test_data'], USER_WORK_DIR)

        if not (parsed_dict['test_labels'] is None or parsed_dict['test_labels'] == ""):
            if parsed_dict['test_labels_format'] == "custom":
                assert parsed_dict['user_handle_test_labels'] is not None, "Please provide the user_handle_test_labels function to handle the custom test labels"
            else:
                assert parsed_dict['test_labels'].endswith(
                    ".npy"), "Test label is not in a \".npy\" format. "
                if parsed_dict['test_labels_format'] is not None:
                    assert parsed_dict['test_labels_format'] == "just_labels" or parsed_dict['test_labels_format'] == "edge_impulse_labels" or parsed_dict['test_labels_format'] == "custom" or parsed_dict['test_labels_format'] == "last_layer_vector", \
                        "Please provide the test_labels_format from 'just_labels','edge_impulse_labels','last_layer_vector' or 'custom'."
            parsed_dict['test_labels'] = util.append_user_workspace(
                parsed_dict['test_labels'], USER_WORK_DIR)

        if (parsed_dict['test_vectors'] is not None):
            assert type(parsed_dict['test_vectors']) is list or parsed_dict[
                'test_vectors'] == 'all', "The test vectors should be indices or string of ranges(e.g 2-7) of indices inside the test dataset in a list"
        else:
            parsed_dict['test_vectors'] = []

        # if (parsed_dict['cpu_op_codes_list'] is not None):
        #     assert type(parsed_dict['cpu_op_codes_list']
        #                 ) is dict, "cpu_op_codes_list is not a dictionary "
        #     # pick cpu op codes from the cpu op codes which the user has not entered and add them to the list
        #     for default_cpu_op_code in cpu_operator_options.cpu_operators_list:
        #         if default_cpu_op_code not in parsed_dict['cpu_op_codes_list']:
        #             parsed_dict['cpu_op_codes_list'][default_cpu_op_code] = cpu_operator_options.cpu_operators_list[default_cpu_op_code].GetAxonsOperationEnum(
        #                 default_cpu_op_code)
        # else:
        #     parsed_dict['cpu_op_codes_list'] = {25: 100, 14: 101, 28: 102}

        if (parsed_dict['interlayer_buffer_size'] is not None):
            assert type(parsed_dict['interlayer_buffer_size']
                        ) is int, "interlayer_buffer_size is not an integer "
        else:
            parsed_dict['interlayer_buffer_size'] = 125000

        if (parsed_dict['psum_buffer_size'] is not None):
            assert type(parsed_dict['psum_buffer_size']
                        ) is int, "psum_buffer_size is not an integer "
        else:
            parsed_dict['psum_buffer_size'] = 0

        if (parsed_dict['op_radix'] is not None):
            assert type(parsed_dict['op_radix']
                        ) is int, "op_radix is not an integer "
        else:
            parsed_dict['op_radix'] = 0

        if (parsed_dict['precision_threshold'] is not None):
            assert type(parsed_dict['precision_threshold']
                        ) is float, "precision_threshold is not a floating point number "
        else:
            parsed_dict['precision_threshold'] = 0.0

        if (parsed_dict['precision_margin'] is not None):
            assert type(parsed_dict['precision_margin']
                        ) is int, "precision_margin is not an integer "
        else:
            parsed_dict['precision_margin'] = 0

        if (parsed_dict['get_quantized_data'] is None):
            parsed_dict['get_quantized_data'] = False

        if (parsed_dict['header_file_test_vector_cnt'] is None):
            parsed_dict['header_file_test_vector_cnt'] = 0

        if (parsed_dict['generate_sim_env_hdrs'] is None):
            parsed_dict['generate_sim_env_hdrs'] = False

        if (parsed_dict['reshape_input'] is None):
            parsed_dict['reshape_input'] = False

        """
        Variant Parameters Default Value Setting
        """
        if parsed_dict['run_all_variants'] is None:
            parsed_dict['run_all_variants'] = False  # default

        if parsed_dict['psum_buffer_placement'] is None:
            parsed_dict['psum_buffer_placement'] = [
                'interlayer_buffer']  # default and dormant
        else:
            assert type(parsed_dict['psum_buffer_placement']) is list or type(parsed_dict['psum_buffer_placement']
                                                                              ) is str, "must be one single value or a list of values from 'interlayer_buffer' or 'dedicated_memory'."
            if type(parsed_dict['psum_buffer_placement']) is str:
                parsed_dict['psum_buffer_placement'] = [
                    parsed_dict['psum_buffer_placement']]

        if parsed_dict['conv2d_setting'] is None:
            parsed_dict['conv2d_setting'] = [
                'local_psum']  # default and dormant
        else:
            assert type(parsed_dict['conv2d_setting']) is list or type(parsed_dict['conv2d_setting']
                                                                       ) is str, "must be one single value or a list of values from 'local_psum', 'inner' or 'outer'."
            if type(parsed_dict['conv2d_setting']) is str:
                parsed_dict['conv2d_setting'] = [parsed_dict['conv2d_setting']]

        if (parsed_dict['normalize_scaleshift'] is None):
            parsed_dict['normalize_scaleshift'] = [True]  # default and fixed
        else:
            assert type(parsed_dict['normalize_scaleshift']) is list or type(
                parsed_dict['normalize_scaleshift']) is bool, "must be one single bool value or  [True, False]"
            if type(parsed_dict['normalize_scaleshift']) is bool:
                parsed_dict['normalize_scaleshift'] = [
                    parsed_dict['normalize_scaleshift']]

        if (parsed_dict['disable_op_quantization'] is None):
            parsed_dict['disable_op_quantization'] = [False]  # default
        else:
            assert type(parsed_dict['disable_op_quantization']) is list or type(
                parsed_dict['disable_op_quantization']) is bool, "must be one single bool value or  [True, False]"
            if type(parsed_dict['disable_op_quantization']) is bool:
                parsed_dict['disable_op_quantization'] = [
                    parsed_dict['disable_op_quantization']]

        if (parsed_dict['skip_softmax_op'] is None):
            parsed_dict['skip_softmax_op'] = [False]  # default and dormant
        else:
            assert type(parsed_dict['skip_softmax_op']) is list or type(
                parsed_dict['skip_softmax_op']) is bool, "must be one single bool value or  [True, False]"
            if type(parsed_dict['skip_softmax_op']) is bool:
                parsed_dict['skip_softmax_op'] = [
                    parsed_dict['skip_softmax_op']]

        if (parsed_dict['transpose_kernel'] is None):
            parsed_dict['transpose_kernel'] = [False]  # default
        else:
            assert type(parsed_dict['transpose_kernel']) is list or type(
                parsed_dict['transpose_kernel']) is bool, "must be one single bool value or  [True, False]"
            if type(parsed_dict['transpose_kernel']) is bool:
                parsed_dict['transpose_kernel'] = [
                    parsed_dict['transpose_kernel']]

    except AssertionError as e:
        logger.critical(f"ASSERT! {e}")
        return -916, f" assert error : {e}"

    logger.info("starting compiler...")
    logger.info("model name is : {}".format(parsed_dict['model_name']))
    logger.debug("tflite model filename is : {}".format(
        parsed_dict['tflite_model']))

    try:
        """handle if model name is empty"""
        if (parsed_dict['model_name'] is None):
            model_name = Path(parsed_dict['tflite_model']).name
            parsed_dict['model_name'] = str(model_name.split('.')[-2])

        # check here if the model name is a valid c symbol name
        if not util.valid_c_symbol_name(parsed_dict['model_name']):
            raise Exception(
                f"{parsed_dict['model_name']} is not a valid c symbol name as required!")

        """handling here to check if tflite file is empty and then use the train data and float model to create a tflite file"""
        if (parsed_dict['train_data'] is not None and parsed_dict['float_model'] is not None and parsed_dict['tflite_model'] is None):
            logger.debug(
                f"train dataset is loaded from : {parsed_dict['train_data']}")
            logger.debug(
                f"float model is loaded from : {parsed_dict['float_model']}")
            # load the train dataset only if provided
            x_train = np.load(parsed_dict['train_data'])
            # convert the keras model into tflite model
            tflite_quant_model = tc.tflite_conversion(
                x_train, parsed_dict['float_model'])
            parsed_dict['tflite_model'] = intermediate_outputs_dir + \
                "/" + parsed_dict['model_name'].lower() + ".tflite"
            open(parsed_dict['tflite_model'], 'wb').write(tflite_quant_model)
            # print(f"Tflite model is created and saved in:\n{parsed_dict['tflite_model']}")
            logger.debug(
                f"tflite model is created and saved in:\n{parsed_dict['tflite_model']}")

        test_result_float, test_result_tflite = 'NA', 'NA'

        if (parsed_dict['tflite_model'] is None or parsed_dict['tflite_model'] == ""):
            logger.critical(
                "tflite is empty, please provide tflite file or the keras model along with train data set!")
            return -904, "tflite file is None or empty!"

        if (parsed_dict['test_data'] is not None):
            x_test = np.load(parsed_dict['test_data'])
            logger.debug(
                f"test dataset is loaded from : {parsed_dict['test_data']}")
            # check here if the expected and actual input shapes match and if not throw an error or reshape the input
            actual_input_shape = np.array(
                np.expand_dims(x_test[0], axis=0).shape)
            # get the model input shape
            expected_input_shape = tc.get_tflite_model_ip_shape(
                parsed_dict['tflite_model'])
            if (not util.shapes_are_same(actual_input_shape, expected_input_shape)):
                if (parsed_dict['reshape_input']):
                    logger.debug(
                        f"reshaping test input to {expected_input_shape} to get test vectors and accuracy results!")
                    x_test = util.reshape_input(x_test, expected_input_shape)
                else:
                    raise Exception(
                        f"test input shape {actual_input_shape} does not match with the expected model input shape {expected_input_shape}, suggest setting the 'reshape_input' config option in the yaml if reshaping the input can fix the issue!")
            if (x_test.dtype == np.int8):
                quantized_x_test = x_test
            else:
                quantized_x_test = tc.quantize_test_dataset(
                    parsed_dict['tflite_model'], x_test)
            if (parsed_dict['get_quantized_data']):
                test_quant_data_filename = intermediate_outputs_dir + \
                    parsed_dict['model_name']+"_q_data.npy"
                np.save(test_quant_data_filename, quantized_x_test)
                logger.debug(
                    f"quantized test dataset (Created from scale and zero point in tflite file) is saved in: {test_quant_data_filename}")
            x_test_max_length = len(x_test)
            if (parsed_dict['test_vectors'] == 'all'):
                parsed_dict['test_vectors'] = [f"0-{x_test_max_length}"]
        else:
            x_test = np.zeros(1)
            quantized_x_test = np.zeros(1)
            parsed_dict['test_vectors'] = []
            x_test_max_length = 1
        if (parsed_dict['test_labels'] is not None):
            # if the user provides a custom test labels, they have to provide a user handler function for it
            if (parsed_dict['test_labels_format'] == "custom" and parsed_dict['user_handle_test_labels'] is not None):
                try:
                    user_func = util.load_func(
                        parsed_dict['user_handle_test_labels'])
                    y_test = user_func(parsed_dict['test_labels'])
                except Exception as e:
                    logger.critical(f"Exception {e}")
                    return -915, f"Error when loading a user function {parsed_dict['user_handle_test_labels']} to handle custom test labels!"
            else:
                y_test = np.load(parsed_dict['test_labels']).astype(np.int32)
                logger.debug(
                    f"test labels are loaded from : {parsed_dict['test_labels']} and are in the format {parsed_dict['test_labels_format']}")
                if (parsed_dict['test_labels_format'] == "last_layer_vector" and (y_test.shape[1] == len(parsed_dict['classification_labels']))):
                    # this format has last layer values and we need the labels so getting the maximum value to get the labels
                    y_test = np.argmax(y_test, axis=1)
                elif (parsed_dict['test_labels_format'] == "edge_impulse_labels"):
                    # edge impulse has labels in the first row and has labels from 1-n instead of 0-n-1
                    y_test = y_test[:, 0]  # labels in first column
                    y_test = y_test - 1
                elif (parsed_dict['test_labels_format'] is None) and (parsed_dict['classification_labels'] is not None and parsed_dict['test_data'] is not None):
                    # we have test vectors and labels, classification labels but no test label format, RECIPE FOR DISASTER
                    logger.critical(
                        "invalid test labels format, Please provide a valid test label format from 'just_labels','edge_impulse_labels','last_layer_vector' or 'custom',exiting.....")
                    return -905, "invalid test labels format"
        else:
            y_test = None  # np.zeros(1)
            parsed_dict['classification_labels'] = None
            parsed_dict['test_labels_format'] = None
        # FIXME put this code inside a "get-accuracy results flag"
        if (parsed_dict['test_data'] is not None) and (parsed_dict['test_labels'] is not None):
            if (parsed_dict['user_handle_accuracy_results'] is None):
                if (parsed_dict['float_model'] is not None):  # test the float model
                    test_result_float = tc.test_floating_point_model(
                        parsed_dict['float_model'], x_test, y_test, classification_model=(parsed_dict['classification_labels'] is not None))
                test_result_tflite = tc.test_tflite_model(parsed_dict['tflite_model'], quantized_x_test, y_test, classification_model=(
                    parsed_dict['classification_labels'] is not None))
            else:
                logger.info(
                    "Accuracy results will be handled by the user provided function!")
        elif (Path(parsed_dict['tflite_model']).is_file()):
            logger.info(
                "test data or proper test labels are not provided, we have the tflite file.")
            # x_test = np.zeros(1)
            # y_test = np.zeros(1)
            # quantized_x_test = np.zeros(1)
            # parsed_dict['test_vectors'] = []
            # test_vectors_flag = False
        else:
            logger.critical(
                "need the tflite file or the keras file with train data to run, provide test data and labels to get test vectors and results!")
            return -904,  "tflite file is None or empty!"

        test_io_vector_ndx = np.array(
            util.convert_range_to_list(parsed_dict['test_vectors']))
        # check here if the count of the io vector indices is more than the total number of elements in the data set, if yes provide all the data in the dataset
        if test_io_vector_ndx.size > x_test_max_length:
            test_io_vector_ndx = np.array(
                util.convert_range_to_list([f"0-{x_test_max_length}"]))
        if (test_io_vector_ndx.size >= 1):
            test_vectors_flag = True
            if (parsed_dict['header_file_test_vector_cnt'] > test_io_vector_ndx.size):
                parsed_dict['header_file_test_vector_cnt'] = test_io_vector_ndx.size

        if (parsed_dict['classification_labels'] is not None):
            logger.info(
                f"labels are in order of : {parsed_dict['classification_labels']}")
    except Exception as e:
        logger.critical(f"Exception {e}")
        return -907, "Error when pre-processing input data/models from the yaml file!"
    """echo parameters defined by the user on terminal"""
    # logger.info(log_string)

    # if(parsed_dict['disable_op_quantization']):
    #   if(parsed_dict['op_radix']==0):
    #     parsed_dict['op_radix']=8
    # else:
    #   parsed_dict['op_radix']=0

    """find out if the user has SOFTMAX as a cpu operation, and if yes do not skip softmax as an operation"""
    # if(tflite.BuiltinOperator.SOFTMAX in parsed_dict['cpu_op_codes_list']):
    #   parsed_dict['skip_softmax_op'] = False
    """
  create the iterator for running all the variants if 'run_all_variants' is true,
  else just run once for the input settings accordingly
  """
    variants_object = compare_models.ModelVariantsClass(
        parsed_dict['run_all_variants'], parsed_dict['tflite_model'], parsed_dict)
    variants = variants_object.get_variants()
    # variants_result = variants_object.get_variants_result_dict()

    get_mass_inference_vectors = True
    transposed_before = False
    # #load the library once here
    axons_compiler_lib = None
    try:
        axons_compiler_lib = ctypes.CDLL(str(AXON_COMPILER_OBJECT))
    except Exception as e:
        axons_compiler_lib = None
        logger.critical(
            f"Exception: '{e}' occured when loading compiler object from {str(AXON_COMPILER_OBJECT)}")

    for variant in variants:
        tflite_axon_graph_object = None
        logger.info(f"running for variant {variant}")
        parsed_dict = variants[variant]
        """
    start creating the file names to be generated by the pre-compiler
    the file naming convention is as follows 
    nrf_axon_model_<model_name>_<file_content_type/info>_.extension 
    """
        file_name_prefix = "nrf_axon_model_" + str(variant).lower()
        # const_bin_file_name = file_name_prefix + "_const_.bin"
        # model_desc_bin_file_name = file_name_prefix + "_description_.bin"
        # test_vectors_filename = file_name_prefix + "_test_vectors_.h"
        tflite_test_vectors_filename = file_name_prefix + "_test_vectors_tflite_.h"
        tflite_test_vectors_file_content = ""
        bin_file_name = file_name_prefix + "_bin_.bin"
        csv_test_vectors_file_name = parsed_dict['model_name'].lower(
        ) + "_test_vectors_.csv"  # file_name_prefix + "_test_vectors_.csv"
        compiler_inference_results_filename = file_name_prefix + \
            "_test_inference_labels_.csv"
        compiler_inference_per_layer_results = file_name_prefix + \
            "_test_inference_results_.csv"
        compiler_inference_results_filepath = compiler_outputs_dir + \
            compiler_inference_results_filename
        # compiler_stdout_filename = file_name_prefix + "_compiler_stdout_.txt"
        # compiler_stdout_filepath = intermediate_outputs_dir + compiler_stdout_filename
        """Create a single tflite and axon graph object here, also check if the model is compatible"""
        try:
            tflite_axon_graph_object = ops.TfLiteAxonGraph(
                parsed_dict['tflite_model'])
            # init the axon supported ops object
            tflite_axon_graph_object.init_axon_supported_ops_object(
                parsed_dict['transpose_kernel'], cpu_operator_options.cpu_operators_list)
        except Exception as e:
            ret = -919
            ret_text = f"\n{e}"
            return ret, ret_text

        """check here if the model is supported"""
        try:
            model_support, m_support_text, tr_model_support, tr_text = tflite_axon_graph_object.get_model_operators_axon_support_info()
            if not model_support:
                constraint_reasons = '\n'.join(
                    f"WARN: |Layer {k}: {v}|" for k, v in m_support_text.items() if v)
                tr_support_text = ""
                if tr_model_support:
                    tr_support_text = "INFO: Transposing the model might support it in Axon"
                raise Exception(
                    f"Model is not supported due to the following reasons/constraints :\n{constraint_reasons} \n{tr_support_text}")
        except Exception as e:
            ret = -918
            ret_text = f"{e}"
            return ret, ret_text

        """Generate the compiler outputs, i.e bin file"""
        try:
            log_content, const_bin_content, model_descriptor_bin_content, model_bin_content, file_content, parsed_dict['op_radix'] = generate_compiler_outputs(COMPILER_TYPES_HDR_FILE, parsed_dict['tflite_model'], str(variant).lower(), x_test, y_test, quantized_x_test, parsed_dict['classification_labels'], test_io_vector_ndx, test_vectors_flag, parsed_dict[
                                                                                                                                                               'normalize_scaleshift'], parsed_dict['disable_op_quantization'], parsed_dict['skip_softmax_op'], parsed_dict['op_radix'], parsed_dict['interlayer_buffer_size'], parsed_dict['psum_buffer_size'], parsed_dict['header_file_test_vector_cnt'], parsed_dict['conv2d_setting'], parsed_dict['psum_buffer_placement'], parsed_dict['transpose_kernel'], tflite_axon_graph_object)
        except Exception as e:
            logger.critical(f"Exception: {e}")
            return -908, "exception occured when generating the bin file"
        # util.save_to_file(intermediate_outputs_dir,const_bin_file_name,const_bin_content, file_type="bin")
        # util.save_to_file(intermediate_outputs_dir,model_desc_bin_file_name,model_descriptor_bin_content, file_type="bin")
        if GET_INTERMEDIATE_FILES:
            util.save_to_file(intermediate_outputs_dir,
                            file_name_prefix + "_binfile_content_.txt", file_content)
        util.save_to_file(intermediate_outputs_dir,
                          bin_file_name, model_bin_content, file_type="bin")
        logger.info("compiler intermediate outputs are saved in : " +
                     str(intermediate_outputs_dir))

        # logger.info(log_string)
        # log_string += log_content
        # util.save_to_file(intermediate_outputs_dir,file_name_prefix+"_results_.txt", log_string)

        """Gets the multiple test vectors and layer output for multiple test vectors in seperate header files"""
        # code that will populate the multiple input vectors if test vectors are requested using the test vectors index provided
        if test_vectors_flag:
            logger.debug(
                f"getting the layer outputs for the first {parsed_dict['header_file_test_vector_cnt']} requested test vectors...")
            tflite_test_multiple_input_vectors = "\n\nconst int8_t* " + \
                parsed_dict['model_name'].lower()+"_input_test_vectors[] = {\n"
            tflite_test_multiple_output_vectors = "\n\nconst int8_t* " + \
                parsed_dict['model_name'].lower(
                )+"_expected_output_vectors[] = {\n"
            multiple_test_input_array_names_list = ""
            multiple_test_input_array_values = ""
            multiple_test_output_array_names_list = ""
            multiple_test_output_array_values = ""
            test_layers_name_list = ""
            # print(f"\nSaving the layer outputs for each of the requested test vectors at {compiler_outputs_dir}...")
            output_datawidth = np.int8
            prediction_digits = []
            tflite_interpreter = tflite_axon_graph_object.get_tflite_interpreter()
            tflite_operators_len = tflite_axon_graph_object.get_tflite_operators_len()
            subgraph = tflite_axon_graph_object.get_tflite_subgraph()
            ops_details = tflite_axon_graph_object.get_tflite_operator_details()
            inputs = tflite_axon_graph_object.get_tflite_input_details()
            outputs = tflite_axon_graph_object.get_tflite_output_details()
            tensor_details = tflite_axon_graph_object.get_tflite_tensor_details()
            test_ip_vector = []
            test_op_vector = []
            csv_test_vectors_op = ""
            model_ip_shape = ops.TensorShape(inputs[0]['shape'])
            model_op_shape = ops.TensorShape(outputs[0]['shape'])
            # ip_q={}
            op_q_new = {}
            # create the list of operations to be executed in CPU
            input_transposed = False
            input_reshaped = False
            operators_detail_graph = tflite_axon_graph_object.get_axon_operator_graph_info()
            axon_last_layer, last_layer_ndx = tflite_axon_graph_object.get_axon_layer_num_of_output_operator()
            if (parsed_dict['skip_softmax_op'] and ops_details[last_layer_ndx]['op_name'] == "SOFTMAX"):
                last_layer_ndx = tflite_axon_graph_object.get_index_for_axon_layer_num(
                    axon_last_layer-1)

            reshape_input = False
            transpose_model_input_flag = False
            if (ops_details[0]["op_name"] == "RESHAPE"):
                # check if the test input needs to be transposed
                # we only transpose the test input if the reshape puts some element in the channel
                # reshape_ip_shape = ops.TensorShape(tensor_details[subgraph.Operators(0).InputsAsNumpy()[0]]['shape'])
                reshape_op_shape = ops.TensorShape(
                    tensor_details[subgraph.Operators(0).OutputsAsNumpy()[0]]['shape'])
                reshape_input = True
                if (reshape_op_shape.depth != 1):
                    # we have to transpose the input
                    original_shape = quantized_x_test.shape
                    x_test_q_tr = quantized_x_test.reshape(len(
                        quantized_x_test), reshape_op_shape.height, reshape_op_shape.width, reshape_op_shape.depth)
                    x_test_q_tr = x_test_q_tr.transpose(0, 3, 1, 2)
                    x_test_q_tr = x_test_q_tr.reshape(original_shape)
                    input_transposed = True
                else:
                    x_test_q_reshaped = quantized_x_test.reshape(len(
                        quantized_x_test), reshape_op_shape.height, reshape_op_shape.width, reshape_op_shape.depth)
                    input_reshaped = True

            # figure out here if the input needs to be transposed
            # the input op should not be a FC,
            if parsed_dict['transpose_kernel']:
                # check if the first input is reshape
                if reshape_input:
                    if ops_details[1]["op_name"] != "FULLY_CONNECTED":
                        transpose_model_input_flag = True
                else:
                    if ops_details[0]["op_name"] != "FULLY_CONNECTED":
                        transpose_model_input_flag = True
            """Start of code to get test vectors for mass inference"""
            if (get_mass_inference_vectors or transpose_model_input_flag != transposed_before):
                logger.info("getting test vectors for mass inference...")
                get_mass_inference_start_tick = time.time()
                if (input_transposed):
                    util.save_vectors_for_mass_inference(
                        csv_test_vectors_file_name, test_io_vector_ndx, x_test_q_tr, y_test, intermediate_outputs_dir, transpose_model_input_flag)
                elif (input_reshaped):
                    util.save_vectors_for_mass_inference(
                        csv_test_vectors_file_name, test_io_vector_ndx, x_test_q_reshaped, y_test, intermediate_outputs_dir, transpose_model_input_flag)
                else:
                    util.save_vectors_for_mass_inference(
                        csv_test_vectors_file_name, test_io_vector_ndx, quantized_x_test, y_test, intermediate_outputs_dir, transpose_model_input_flag)
                get_mass_inference_vectors = False
                logger.info(
                    f".... done!, took {time.time()-get_mass_inference_start_tick} seconds! for {len(test_io_vector_ndx)} vectors")
            # #DEBUG TEST CODE COMMENTED OUT BELOW
            # logger.info(f"running mass inference timing calculation, will only generate vectors and break.....")
            # break
            # model_wrapper_ffi = mw.get_model_wrapper_cffi_object(COMPILER_TYPES_HDR_FILE)
            transposed_before = transpose_model_input_flag
            for i, ndx in enumerate(test_io_vector_ndx[0:parsed_dict['header_file_test_vector_cnt']]):
                test_ = np.expand_dims(quantized_x_test[ndx], axis=0)

                tflite_interpreter.set_tensor(inputs[0]['index'], test_)
                # Run inference.
                tflite_interpreter.invoke()
                if (model_ip_shape.depth > 1) and model_ip_shape.shape_size > 3:  # we have a multiple channel input
                    test_ = test_.transpose(0, 3, 1, 2)
                test_ip_vector.append(test_)
                multiple_layer_output_vectors = ""
                if (input_transposed):
                    test_ = x_test_q_tr[ndx]
                multiple_test_input_array_names_list += "    " + \
                    parsed_dict['model_name'].lower(
                    )+"_test_input_"+str(ndx)+",\n"
                if GET_INTERMEDIATE_FILES:
                    multiple_test_input_array_values += util.write_array_to_file(
                        test_.squeeze(), parsed_dict['model_name'].lower()+"_test_input_"+str(ndx))
                if (i == 0):
                    test_layers_name_list += "\n\nconst int8_t* " + parsed_dict['model_name'].lower(
                    )+"_layer_vectors[] = \n{"+f"\n#ifndef {USE_MINIMUM_TEST_VECTORS_DEFINE}"+"\n    "+parsed_dict['model_name'].lower()+"_l0_test_input,\n"
                    multiple_test_input_array_names_list += f"#ifndef {USE_MINIMUM_TEST_VECTORS_DEFINE}\n"
                    multiple_test_input_array_values += f"\n#ifndef {USE_MINIMUM_TEST_VECTORS_DEFINE}"

                op_cnt = 0
                # if we need the output for each of the layers we need to get them in multiple header files
                if GET_INTERMEDIATE_FILES:
                    multiple_layer_output_vectors += util.write_array_to_file(
                        test_.squeeze(), parsed_dict['model_name'].lower()+"_l0_test_input")
                multiple_layer_output_vectors += f"\n#ifndef {USE_MINIMUM_TEST_VECTORS_DEFINE}"
                operator_op_ndx = np.array([-1])
                for ops_graph_index in range(len(operators_detail_graph)):
                    op_num = operators_detail_graph[ops_graph_index]['index']
                    op_dw = np.int8
                    operator = subgraph.Operators(op_num)
                    # ops_options_ndx = operator.BuiltinOptionsType()
                    # operator_code = model.OperatorCodes(
                    #     (operator.OpcodeIndex())).BuiltinCode()
                    operator_code = operators_detail_graph[ops_graph_index]['op_code']
                    op_graph_detail = operators_detail_graph[ops_graph_index]
                    # if(operator_code in supported_ops.supported_operators) or (operator_code in supported_ops.variable_operators):
                    if (op_graph_detail['operator_support'] == ops.OperatorSupportEnum.SUPPORTED or op_graph_detail['operator_support'] == ops.OperatorSupportEnum.PARTIALLY_SUPPORTED or op_graph_detail['operator_support'] == ops.OperatorSupportEnum.VARIABLE):
                        # options = supported_ops.supported_operators[operator_code](operator_code, operator, ops_details[op_num], tensor_details, tflite_interpreter)
                        if i == 0:
                            if op_graph_detail['options_initialized']:
                                options = op_graph_detail['operator_options']
                            else:
                                options = op_graph_detail['operator_options'].CreateOptionsObject(
                                    operator_code, operator, op_graph_detail, tensor_details, tflite_interpreter, operators_detail_graph)
                                op_graph_detail['operator_options'] = options
                                op_graph_detail['options_initialized'] = True
                        else:
                            options = op_graph_detail['operator_options']
                        axon_layer_num = op_graph_detail['axon_layer_num']
                        # if axon_layer_num<0 or operators_detail_graph[op_num]['operator_support']!=ops.OperatorSupportEnum.SUPPORTED:
                        if axon_layer_num < 0:
                            if i == 0:  # just need to check it once for the first test vector index
                                if (operators_detail_graph[op_num]['operator_support'].value) == ops.OperatorSupportEnum.CONVERTED_PASSTHROUGH.value:
                                    tflite_axon_graph_object.update_axon_operator_graph(
                                        operators_detail_graph[op_num])
                            continue
                        # ops_details[op_num]["op_name"]
                        ops_name = options.GetOperationName()
                        # operator_ip_ndx = operator.InputsAsNumpy()
                        # operator_ip_ndx = options.GetInputTensorsNdx()
                        # operator_op_ndx = operator.OutputsAsNumpy()
                        operator_op_ndx = options.GetOutputTensorsNdx()
                        # shape_length = len(tflite_interpreter.tensor(operator_op_ndx[0])().shape)
                        if (ops_name == "PAD") or (("SOFTMAX" in ops_name) and parsed_dict['skip_softmax_op']):
                            continue
                        # ip_q = copy.deepcopy(tensor_details[operator_ip_ndx[0]]['quantization_parameters'])
                        output_index = 0
                        if "SPLIT_V" in operators_detail_graph[ops_graph_index]['op_name']:
                            output_index = operators_detail_graph[ops_graph_index]['splitv_op_ndx']
                        op_q_new = copy.deepcopy(
                            tensor_details[operator_op_ndx[output_index]]['quantization_parameters'])
                        layer_op = util.get_array_from_tensor(
                            tflite_interpreter.tensor(operator_op_ndx[output_index])())
                        if (op_num+1) < tflite_operators_len and (("SOFTMAX" in ops_details[op_num+1]["op_name"])) and (not parsed_dict['skip_softmax_op']):
                            logger.debug(
                                f"adjust the layer outputs to be at Q11.12 for test vector ndx {ndx}")
                            layer_op = (
                                (layer_op - op_q_new['zero_points'][0]) * op_q_new['scales']) * (2**12)
                            op_dw = np.int32
                        if (op_num+1) < tflite_operators_len and (("LOGISTIC" in ops_details[op_num+1]["op_name"]) or ("TANH" in ops_details[op_num+1]["op_name"])):
                            layer_op = np.float32(layer_op)
                            layer_op = (
                                (layer_op - op_q_new['zero_points'][0]) * op_q_new['scales'][0]) * (2**12)
                            op_dw = np.int16
                        if (op_num+1) < tflite_operators_len and ("LEAKY_RELU" in ops_details[op_num+1]["op_name"]):
                            layer_op = util.get_array_from_tensor(tflite_interpreter.get_tensor(
                                operators_detail_graph[op_num+1]['op_tensors'][output_index]))
                            op_dw = np.int8
                        # and (parsed_dict['op_radix'] > 8):#last layer
                        if (ops_graph_index == last_layer_ndx) and (parsed_dict['disable_op_quantization']):
                            # check for the radix here and multiply by that after dequantization
                            # parsed_dict['op_radix'] = util.get_output_radix(parsed_dict['op_radix'],np.array(30), op_q_new['scales'][0], op_q_new['zero_points'][0],op_dw)
                            op_dw = np.int32
                            layer_op = (
                                (layer_op - op_q_new['zero_points'][0]) * op_q_new['scales']) * (2**parsed_dict['op_radix'])
                            layer_op = layer_op.astype(op_dw)

                        # if(shape_length==2):
                        #   # multiple_layer_output_vectors += util.write_array_to_file(tflite_interpreter.tensor(operator_op_ndx[0])().squeeze(),parsed_dict['model_name'].lower()+"_l"+str(op_cnt) + "_"+ops_name+"_tflite_op")
                        #   multiple_layer_output_vectors += util.write_array_to_file(layer_op,parsed_dict['model_name'].lower()+"_l"+str(op_cnt) + "_"+ops_name+"_tflite_op",op_dw)
                        # else:
                        #   # multiple_layer_output_vectors += util.write_array_to_file(tflite_interpreter.tensor(operator_op_ndx[0])().transpose(0,3,1,2).squeeze(),parsed_dict['model_name'].lower()+"_l"+str(op_cnt) + "_"+ops_name+"_tflite_op")
                        if (ops_graph_index == (last_layer_ndx)):
                            multiple_layer_output_vectors += "\n#endif"

                        tflite_output_layer_name = parsed_dict['model_name'].lower(
                        )+"_l"+str(op_cnt) + "_"+ops_name+"_tflite_op"
                        if GET_INTERMEDIATE_FILES:
                            multiple_layer_output_vectors += util.write_array_to_file(
                                layer_op, tflite_output_layer_name, op_dw)
                        if (i == 0):
                            test_layers_name_list += "    " + tflite_output_layer_name.lower() + ",\n"
                        op_cnt += 1
                    # if not parsed_dict['skip_softmax_op']:
                    #   #DEBUG# get the last operation output for testing
                    #   #last_operator = subgraph.Operators(op_num-1)
                    #   #last_operator_op_ndx = last_operator.OutputsAsNumpy()
                    #   #multiple_layer_output_vectors += util.write_array_to_file(tflite_interpreter.tensor(last_operator.OutputsAsNumpy()[0])().squeeze(),parsed_dict['model_name'].lower()+"_OP_BEFORE_SOFTMAX_"+str(op_cnt) + "_op_tflite_"+str(ndx))
                    #   #multiple_layer_output_vectors += util.write_array_to_file(tflite_interpreter.tensor(operator.OutputsAsNumpy()[0])().squeeze(),parsed_dict['model_name'].lower()+"_SOFTMAX_DEPTH"+str(op_cnt) + "_op_tflite_"+str(ndx))
                    #   # operator_op_ndx = operator.OutputsAsNumpy()
                    #   if(shape_length==4):
                    #     multiple_layer_output_vectors += util.write_array_to_file(tflite_interpreter.tensor(operator.OutputsAsNumpy()[0])().transpose(0,3,1,2).squeeze(),parsed_dict['model_name'].lower()+"_SOFTMAX_"+str(op_cnt) + "_op_tflite_"+str(ndx))
                    #   elif(shape_length==2):
                    #     multiple_layer_output_vectors += util.write_array_to_file(tflite_interpreter.tensor(operator.OutputsAsNumpy()[0])().squeeze(),parsed_dict['model_name'].lower()+"_SOFTMAX_"+str(op_cnt) + "_op_tflite_"+str(ndx))
                    final_ops_ndx = operator_op_ndx[0]
                # util.save_to_file(compiler_outputs_dir,file_name_prefix + "_layer_outputs_tflite_"+str(ndx)+"_.h", multiple_layer_output_vectors)
                if (i == 0):
                    tflite_test_vectors_file_content += multiple_layer_output_vectors
                    # tflite_test_layer_vectors +="  " + layer_name.lower()+"_tflite_op,\n  "

                # Save the class predictions for all test samples.
                output = tflite_interpreter.tensor(outputs[0]['index'])
                predict_label = np.argmax(output()[0])
                prediction_digits.append(predict_label)
                op = util.get_array_from_tensor(
                    tflite_interpreter.tensor(final_ops_ndx)())
                # and parsed_dict['op_radix']>8:
                if (parsed_dict['disable_op_quantization']):
                    # deaquantize the output
                    # check for the radix and adjust accordingly
                    output_datawidth = np.int32
                    op = ((op - op_q_new['zero_points'][0]) *
                          op_q_new['scales']) * (2**(parsed_dict['op_radix']))
                    op = op.astype(output_datawidth)

                test_op_vector.append(op)
                if GET_INTERMEDIATE_FILES:
                    csv_test_vectors_op += util.write_array_to_file(
                        op.squeeze(), "", array_bitwidth=op.dtype)
                csv_test_vectors_op = util.get_csv_text(csv_test_vectors_op)
                multiple_test_output_array_names_list += "    " + \
                    parsed_dict['model_name'].lower(
                    )+"_expected_output_"+str(ndx)+",\n"
                if GET_INTERMEDIATE_FILES:
                    multiple_test_output_array_values += util.write_array_to_file(np.array(op).squeeze(
                    ), parsed_dict['model_name'].lower()+"_expected_output_"+str(ndx), output_datawidth)
                if (i == 0):
                    multiple_test_output_array_names_list += f"#ifndef {USE_MINIMUM_TEST_VECTORS_DEFINE}\n"
                    multiple_test_output_array_values += f"\n#ifndef {USE_MINIMUM_TEST_VECTORS_DEFINE}"

            test_layers_name_list += "#else\n    NULL,\n#endif\n};\n"
            tflite_test_vectors_file_content += test_layers_name_list
            tflite_test_vectors_file_content += "\n/*\n  Input vectors and expected outputs to test multiple inputs for the " + \
                parsed_dict['model_name'].upper(
                ) + " Model \n  Test vectors naming format : model_name_[vector_name]_[index_in_test_dataset] \n*/"
            tflite_test_multiple_input_vectors += multiple_test_input_array_names_list
            tflite_test_multiple_output_vectors += multiple_test_output_array_names_list
            tflite_test_vectors_file_content += multiple_test_input_array_values + "\n#endif"
            tflite_test_vectors_file_content += multiple_test_output_array_values + "\n#endif"
            tflite_test_vectors_file_content += tflite_test_multiple_input_vectors + "#endif\n  };"
            tflite_test_vectors_file_content += tflite_test_multiple_output_vectors + "#endif\n  };"
            logger.debug(
                f"saving the layer outputs for each of the requested test vectors at {compiler_outputs_dir}")
            # save the logs in a file in a seperate folder.
            if test_result_float != 'NA' and test_result_tflite != 'NA':
                if (parsed_dict['float_model'] != ""):
                    logger.debug(
                        "the accuracy of the floating model on the complete test data set is "+str(test_result_float))
                logger.debug(
                    "the accuracy of the tflite model on the complete test data set is "+str(test_result_tflite))
            if (test_result_float != 0) and not isinstance(test_result_float, (str)):
                logger.debug(
                    f"total tflite quantization loss is : {((test_result_float-test_result_tflite)/test_result_float)*100}%")
        else:
            """
            const int8_t** tinyml_vww_input_test_vectors = NULL;
            const int8_t** tinyml_vww_layer_vectors = NULL;
            const int8_t** tinyml_vww_expected_output_vectors = NULL;
            """
            tflite_test_vectors_file_content += "\n\nconst int8_t** " + \
                parsed_dict['model_name'].lower(
                )+"_input_test_vectors = NULL;\n"
            tflite_test_vectors_file_content += "\n\nconst int8_t** " + \
                parsed_dict['model_name'].lower()+"_layer_vectors = NULL;\n"
            tflite_test_vectors_file_content += "\n\nconst int8_t** " + \
                parsed_dict['model_name'].lower(
                )+"_expected_output_vectors = NULL;\n"

        if (parsed_dict['header_file_test_vector_cnt'] != 0) and GET_INTERMEDIATE_FILES:
            util.save_to_file(intermediate_outputs_dir,
                              tflite_test_vectors_filename, tflite_test_vectors_file_content)
        """End of Code to get mass inference vectors, test_vectors for inference"""

        """call the function to initiate running the compiler object here"""
        compiler_return_dict, subprocess_return_code, compiler_return_codetext = run_compiler_library(test_vectors_flag,
                                                                                                      False,  # not getting the per layer results
                                                                                                      AXON_COMPILER_OBJECT,
                                                                                                      COMPILER_TYPES_HDR_FILE,
                                                                                                      intermediate_outputs_dir,
                                                                                                      compiler_outputs_dir,
                                                                                                      file_name_prefix,
                                                                                                      bin_file_name,
                                                                                                      csv_test_vectors_file_name,
                                                                                                      compiler_inference_results_filename,
                                                                                                      compiler_inference_per_layer_results,
                                                                                                      axons_compiler_lib)

        """code here to find the model accuracy on the requested test data set"""
        # the compiler object has exited,
        # if the user has requested for test vectors, we should load in the output from the simulator and then do a compare
        model_results_text = "\n"
        quantization_loss_results_text = ""
        accuracy_results_text = ""
        compiler_return_text = ""
        confusion_matrix_text = ""
        precision_score_text = ""
        # classification_report_text=""
        if (test_vectors_flag) and (subprocess_return_code == 0):
            compiler_return_text += "\n\tMemory Usage (in bytes)\n\n"
            for return_value in compiler_return_dict:
                if (return_value == 'profiling_ticks'):
                    compiler_return_text += f"\n\tInference time (estimate, in ticks):\t{compiler_return_dict[return_value]}\n"
                    variants_object.set_profiling_tick_result(
                        variant, compiler_return_dict[return_value])
                else:
                    variants_object.set_memory_footprint_result(
                        variant, return_value, compiler_return_dict[return_value])
                    compiler_return_text += f"\t\t{return_value}:\t{compiler_return_dict[return_value]}\n"
                # print(compiler_return_text)

            # model_results_text=f"Running test on {parsed_dict['model_name'].lower()} for vectors {parsed_dict['test_vectors']}\n"
            results_header = f" Model {parsed_dict['model_name'].upper()} results (variant : {variant})\n"
            title_line = "-"*len(results_header) + "\n"
            model_results_text += title_line + results_header + title_line
            model_results_text += compiler_return_text
            # output_length = model_op_shape.get_length()
            csv_output = util.load_csv_lines_to_np_array(
                compiler_inference_results_filepath)
            if (len(csv_output.shape) == 1):  # doing it for just one vector
                csv_output = csv_output.reshape(1, csv_output.shape[0])
            # accuracy_results_text = f"\n\tAccuracy (test data set size {len(test_io_vector_ndx)})\n"
            if (parsed_dict['test_labels'] is not None and parsed_dict['classification_labels'] is not None):
                accuracy_results_text = f"\n\tAccuracy (test data set size {len(test_io_vector_ndx)})\n"
                true_labels = np.array([y_test[test_vector_ndx]
                                       for test_vector_ndx in test_io_vector_ndx])
                sampled_x_test = np.array(
                    [x_test[test_vector_ndx] for test_vector_ndx in test_io_vector_ndx])
                classification_labels = copy.deepcopy(
                    parsed_dict['classification_labels'])
                total_labels_count = len(parsed_dict['classification_labels'])
                compiler_op_labels = compare_models.get_labels(csv_output)
                if (not parsed_dict['skip_softmax_op']) and (not parsed_dict['disable_op_quantization']):
                    # calculate the probability values and the find the labels accordingly, also add inconclusive to it
                    if parsed_dict['precision_threshold'] or parsed_dict['precision_margin']:
                        # run threshold regression
                        # if parsed_dict['run_threshold_regression']:
                        #   logger.info("running threshold regression...")
                        #   threshold, f1_score = compare_models.run_threshold_regression(true_labels,csv_output,classification_labels,total_labels_count)
                        #   logger.info(f"suggested threshold is {threshold}, score {f1_score}")
                        # converting a percent value into float for conversion
                        parsed_dict['precision_margin'] = parsed_dict['precision_margin'] / 100
                        # if (parsed_dict['disable_op_quantization']):
                        #   parsed_dict['precision_threshold'] = (((parsed_dict['precision_threshold'] -  op_q_new['zero_points'][0])* op_q_new['scales']) * (2**parsed_dict['op_radix']))
                        #   parsed_dict['precision_margin'] = ((parsed_dict['precision_margin'] -  op_q_new['zero_points'][0])* op_q_new['scales']) * (2**parsed_dict['op_radix'])
                        compiler_op_labels = compare_models.get_labels(
                            csv_output, parsed_dict['precision_threshold'], parsed_dict['precision_margin'], True)
                        classification_labels.append('inconclusive')
                # classification_report_text =  compare_models.get_classification_report(true_labels,compiler_op_labels,classification_labels,total_labels_count)
                # logger.debug(classification_report_text)
                precision_score_text, precision_value = compare_models.get_precision_score_text(
                    true_labels, compiler_op_labels, classification_labels, total_labels_count, GET_PER_CLASS_PRECISION)
                variants_object.set_precision_result(variant, precision_value)
                # logger.info(precision_score)
                accuracy_results, labels = compare_models.get_model_accuracy(
                    sampled_x_test, true_labels, compiler_op_labels, parsed_dict['tflite_model'], parsed_dict['float_model'], get_results=True)
                if (parsed_dict['float_model'] is not None):
                    # if(test_result_float!=0):
                    # tflite_quantization_error_overall = ((test_result_float - test_result_tflite)/test_result_float) * 100
                    if (accuracy_results['float'] != 0):
                        tflite_quantization_error_sample = (
                            (accuracy_results['float'] - accuracy_results['tflite'])/accuracy_results['float']) * 100
                        compiler_sim_quantization_error = (
                            (accuracy_results['float'] - accuracy_results['simulator'])/accuracy_results['float']) * 100
                        quantization_loss_results_text = "\n\tQuantization loss (w.r.t float model)%:\n"
                        quantization_loss_results_text += f"\t\tTflite int8 model:\t{tflite_quantization_error_sample:.2f}%\n"
                        quantization_loss_results_text += f"\t\tAxons int8 model:\t{compiler_sim_quantization_error:.2f}%\n"
                        variants_object.set_quantization_loss_results(
                            variant, compiler_sim_quantization_error, tflite_quantization_error_sample)
                    accuracy_results_text += f"\t\tTflite float model:\t{accuracy_results['float']:.4f}\n"
                confusion_matrix_text = compare_models.get_consolidated_confusion_matrix(
                    true_labels, labels['float_label'], labels['tflite_label'], compiler_op_labels, classification_labels)
                accuracy_results_text += f"\t\tTflite int8 model:\t{accuracy_results['tflite']:.4f}\n"
                accuracy_results_text += f"\t\tAxons int8 model:\t{accuracy_results['simulator']:.4f}\n"

                variants_object.set_accuracy_results(variant, accuracy_results)
                # accuracy_results_text = util.print_util(f"[ndx, true_label, simulator_label]\n{compare_models.get_vectors_where_reference_inferred_correctly(true_label=true_labels,reference_label=labels['tflite_label'],test_label=compiler_op_labels)}", accuracy_results_text)
            else:
                # the output inference csv file from the compiler will have the last layer output vectors directly
                # and we need to figure out a way to get labels from that and compare it with true results
                if (parsed_dict['user_handle_accuracy_results'] is not None):
                    logger.info(
                        f"Calling User Handle ({parsed_dict['user_handle_accuracy_results']}) for Model Results (test data set size {len(test_io_vector_ndx)})")
                    try:
                        user_handle_return_text = ""
                        user_func = util.load_func(
                            parsed_dict['user_handle_accuracy_results'])
                        user_handle_return, user_handle_return_text = user_func(
                            parsed_dict, x_test, y_test, csv_output, model_op_shape, variants_object, variant)
                        accuracy_results_text += user_handle_return_text
                    except Exception as e:
                        accuracy_results_text += f"\n\tModel Accuracy/Results NA, Calling User Handle raised exception {e}\n"
                        logger.error(
                            f"Exception occured when calling user handle function [{parsed_dict['user_handle_accuracy_results']}], exception {e}")
                else:
                    accuracy_results_text += "\n\tAccuracy/Results NA, Labels are not provided!\n"

            model_results_text += accuracy_results_text + quantization_loss_results_text + \
                precision_score_text + confusion_matrix_text  # + classification_report_text
            variants_object.set_test_data_set_size(
                variant, test_io_vector_ndx.size)
            if GET_INTERMEDIATE_FILES:
                util.save_to_file(intermediate_outputs_dir, file_name_prefix +
                                "_accuracy_results_.txt", model_results_text)
            logger.info(model_results_text)
        if subprocess_return_code == 0:
            logger.info(
                f"done running variant {variant}, return code {subprocess_return_code}")
        else:
            logger.error(
                f"encountered error running variant {variant}, return code {subprocess_return_code}")

    variants_result_yaml = variants_object.get_variants_result_as_yaml()
    compare_results_yaml = variants_object.get_compare_variant_results(
        as_yaml=True)
    if GET_INTERMEDIATE_FILES:
        util.save_to_file(intermediate_outputs_dir, file_name_prefix +
                        "_variants_result_.yml", variants_result_yaml)        
        util.save_to_file(intermediate_outputs_dir, file_name_prefix +
                        "_compare_variants_result_.yml", compare_results_yaml)
    performance_metric_table = variants_object.get_performance_metrics_as_table()
    logger.info(performance_metric_table)
    # FIXME needs to be addressed as using dlclose with the library causes the docker container to crash
    # not clearing it right now as the system will do the necessary clean up as needed as it is going to exit soon
    # if os.name=="posix":
    #   # axons_compiler_lib.free(axons_compiler_lib._handle)
    #   # Get the handle to the dlclose function
    #   dlclose = ctypes.CDLL(None).dlclose
    #   lib_handle = axons_compiler_lib._handle
    #   del(axons_compiler_lib)
    #   # Unload the shared library
    #   logger.info(f"lib handle ref count {sys.getrefcount(lib_handle) - 1}")
    #   lib_close_return = dlclose(lib_handle)
    #   del(lib_handle)
    #   # lib_handle = 0
    #   # if(lib_close_return!=0):
    #   #   raise Exception(f"did not close library cleanly")
    # elif os.name=="nt":
    #   ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.wintypes.HMODULE]
    #   ctypes.windll.kernel32.FreeLibrary(axons_compiler_lib._handle)
    return subprocess_return_code, compiler_return_codetext


def debug_app(path):  # to be deprecated
    global USER_WORK_DIR
    DEBUG_MULTIPLE_MODELS = True
    log_file_name = r"logs/" + "debug_" + str(Path(__file__).name.split(
        '.')[0])+"_"+str(dt.datetime.now().strftime("%Y%m%d_%H%M%S"))+".log"
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
    USER_WORK_DIR = util.get_abs_dir_from_file(path)
    if (DEBUG_MULTIPLE_MODELS):
        yaml_input = util.load_yaml_file(path)
        yaml_test_list = list(yaml_input.keys())
        for test in yaml_test_list:
            if (test == "default_values"):
                continue
            logger.info(f"debugging for {test}")
            app_return = axons_compiler(yaml_input[test])
            logger.info(f"completed debugging {test} return code {app_return}")
            gc.collect()
    else:
        arguments_dict = util.get_compiler_debug_arguments(path)
        axons_compiler(arguments_dict)


def run_app():
    global USER_WORK_DIR
    # setup logging here
    log_level = logging.INFO
    log_file_name = "run_"
    log_format = "%(levelname)s: %(message)s"

    if (util.debugger_is_active()):
        log_file_name = "debug_"
        log_level = logging.DEBUG
        log_format = "%(funcName)s-%(levelname)s: %(message)s"

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    log_file_name = log_file_name + str(Path(__file__).name.split('.')[0])+"_"+str(
        dt.datetime.now().strftime("%Y%m%d_%H%M%S"))+".log"

    if (len(sys.argv) > 2):
        parser = util.parse_compiler_arguments()
        arguments_dict = vars(parser.parse_args())
        start_time = time.time()
        log_directory = USER_WORK_DIR + "/"
        log_full_path = log_directory + log_file_name

        logging.basicConfig(filename=log_full_path,
                            level=log_level, format=log_format)
        # Log OS details
        logger.debug(
            f"os.name = {os.name}, platform.system() = {platform.system()}, platform.machine() = {platform.machine()}")
        axons_compiler(arguments_dict)
        # print(f"...took {(time.time()-start_time):.2f} seconds to complete")
    else:
        """load yaml file and run for however many test inputs are present inside it"""
        if (len(sys.argv) == 2):
            path = Path(sys.argv[1])

            if (os.name == "posix"):
                if (util.is_windows_path(path)):
                    path = util.get_linux_path(path)
            """user has entered a path to its own custom yaml file, handle that"""
            if Path(path).exists() and (str(path).endswith(".yaml") or str(path).endswith(".yml")):
                USER_WORK_DIR = util.get_abs_dir_from_file(path)
                yaml_input = util.load_yaml_file(path)
                yaml_test_list = list(yaml_input.keys())
                log_directory = USER_WORK_DIR + "/logs/"
                if not (Path(log_directory).exists()):
                    Path(log_directory).mkdir(parents=True, exist_ok=True)
                    logger.info(
                        f"created log outputs directory {log_directory}")
                log_full_path = log_directory + log_file_name
                logging.basicConfig(filename=log_full_path,
                                    level=log_level, format=log_format)
                # Log OS details
                logger.debug(
                    f"os.name = {os.name}, platform.system() = {platform.system()}, platform.machine() = {platform.machine()}")
                for test in yaml_test_list:
                    if (test == "default_values"):
                        continue
                    logger.info(f"running {test}")
                    start_time = time.time()
                    app_return, app_return_text = axons_compiler(
                        yaml_input[test])
                    if app_return != 0:
                        logger.error(
                            f"return code {app_return}, {app_return_text}")
                    else:
                        logger.info(
                            f"completed running {test}, return code {app_return}({app_return_text}), took {(time.time() - start_time):.2f} seconds")
                    gc.collect()
            else:
                raise Exception(f"yaml file doesn't exist at {path}!")
        else:
            raise Exception("provide a yaml file!")


if __name__ == "__main__":
    run_app()
