""" 
/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */
"""
from utility import util
from pathlib import Path
import compare_models as compare_models
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import logging
import copy
import time
import sys
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# from axons_ml_nn_compiler_executor import axons_compiler as executor


"""create logger object"""
logger = logging.getLogger(__name__)


def plot_accuracy_precision(accuracy, precision, threshold, model_name="", margin=0):
    plt.plot(threshold, accuracy, color='b', label="accuracy")
    plt.plot(threshold, precision, color='g', label="precision")
    plt.ylim(0, 1.1)
    plt.xlim(0, 1)
    plt.axhline(y=1, color='m', linestyle='dashed')
    plt.xlabel("Threshold")
    plt.ylabel("Percentage")
    plt.title(f"{model_name} Accuracy vs Precision (margin : {margin})")
    plt.legend()
    plt.draw()
    plt.show()


def run_precision_measurements(parsed_dict, thresholds, margins):
    thresholds = np.array(thresholds)
    margins = np.array(margins)
    logger.info(
        f"running precision measurements for thresholds {thresholds} and margins {margins}")

    compiler_outputs_dir = USER_WORK_DIR + "/outputs/"
    file_name_prefix = "axon_model_" + str(parsed_dict['model_name']).lower()
    compiler_inference_results_filename = file_name_prefix + \
        "_test_inference_labels_.txt"
    compiler_inference_results_filepath = compiler_outputs_dir + \
        compiler_inference_results_filename
    if not (parsed_dict['test_data'] is None or parsed_dict['test_data'] == ""):
        assert parsed_dict['test_data'].endswith(
            ".npy"), "Test dataset is not in a \".npy\" format."
        parsed_dict['test_data'] = util.append_user_workspace(
            parsed_dict['test_data'], USER_WORK_DIR)
        x_test = np.load(parsed_dict['test_data'])
    else:
        raise Exception("No Test Data Provided")

    if (parsed_dict['test_vectors'] == 'all'):
        parsed_dict['test_vectors'] = [f"0-{len(x_test)}"]
    test_io_vector_ndx = np.array(
        util.convert_range_to_list(parsed_dict['test_vectors']))

    if not (parsed_dict['test_labels'] is None or parsed_dict['test_labels'] == ""):
        assert parsed_dict['test_labels'].endswith(
            ".npy"), "Test label is not in a \".npy\" format. "
        if parsed_dict['test_labels_format'] is not None:
            assert parsed_dict['test_labels_format'] == "just_labels" or parsed_dict['test_labels_format'] == "edge_impulse_labels" or parsed_dict['test_labels_format'] == "custom" or parsed_dict['test_labels_format'] == "last_layer_vector", \
                "Please provide the test_labels_format from 'just_labels','edge_impulse_labels','last_layer_vector' or 'custom'."
        parsed_dict['test_labels'] = util.append_user_workspace(
            parsed_dict['test_labels'], USER_WORK_DIR)
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
        # if the user provides a custom test labels, they have to provide a user handler function for it
        elif (parsed_dict['test_labels_format'] == "custom" and parsed_dict['user_handle_test_labels'] is not None):
            user_func = util.load_func(parsed_dict['user_handle_test_labels'])
            y_test = user_func(y_test)
        elif (parsed_dict['test_labels_format'] is None) and (parsed_dict['classification_labels'] is not None and parsed_dict['test_data'] is not None):
            # we have test vectors and labels, classification labels but no test label format, RECIPE FOR DISASTER
            logger.critical(
                "invalid test labels format, Please provide a valid test label format from 'just_labels','edge_impulse_labels','last_layer_vector' or 'custom',exiting.....")
            raise Exception("invalid test labels format")
    else:
        raise Exception("No Test Labels Provided")

    if (parsed_dict['classification_labels'] is not None):
        model_results_text = "\n"
        final_results_text = ""
        true_labels = np.array([y_test[test_vector_ndx]
                               for test_vector_ndx in test_io_vector_ndx])
        sampled_x_test = np.array([x_test[test_vector_ndx]
                                  for test_vector_ndx in test_io_vector_ndx])
        classification_labels = copy.deepcopy(
            parsed_dict['classification_labels'])
        output_length = len(parsed_dict['classification_labels'])
        if (not parsed_dict['skip_softmax_op']) and (not parsed_dict['disable_op_quantization']):
            # load the compiler csv from an earlier run of the compiler
            csv_output = np.loadtxt(compiler_inference_results_filepath,
                                    delimiter=",", dtype=int, usecols=range(0, output_length))
            if (len(csv_output.shape) == 1):  # doing it for just one vector
                csv_output = csv_output.reshape(1, csv_output.shape[0])
            if csv_output.shape[0] != len(sampled_x_test):
                # may be run the compiler executor app here once??
                # try:
                #   executor(parsed_dict)
                #   csv_output = np.loadtxt(compiler_inference_results_filepath, delimiter=",", dtype=int, usecols=range(0,output_length))
                #   if(len(csv_output.shape)==1):#doing it for just one vector
                #     csv_output = csv_output.reshape(1,csv_output.shape[0])
                # except Exception as e:
                #   logger.error(e)
                raise Exception(
                    "Sample data size and compiler output labels size do not match!")
        # compiler_op_labels = compare_models.get_labels(csv_output)
            margins = margins/100
            classification_labels.append('inconclusive')

            for margin in margins:
                final_results_text = f"\n\tPrecision Measurements for model {parsed_dict['model_name']} at margin {margin}"
                final_results_text += "\n\t\tThresholds\tAccuracy\tPrecision"
                accuracy_array_, precision_array_ = [], []
                for threshold in thresholds:
                    compiler_op_labels = compare_models.get_labels(
                        csv_output, threshold, margin, True)
                    precision_score_text, precision_value = compare_models.get_precision_score_text(
                        true_labels, compiler_op_labels, classification_labels, output_length, True)
                    accuracy_results, labels = compare_models.get_model_accuracy(
                        sampled_x_test, true_labels, compiler_op_labels, get_results=True)
                    accuracy_text = f"\n\tAccuracy (test data set size {len(test_io_vector_ndx)}):\t{accuracy_results['simulator']:.4f}\n"
                    # confusion_matrix_text = compare_models.get_consolidated_confusion_matrix(true_labels, labels['float_label'],labels['tflite_label'],compiler_op_labels,classification_labels)
                    final_results_text += f"\n\t\t{threshold}\t{accuracy_results['simulator']}\t{precision_value}"
                    accuracy_array_.append(accuracy_results['simulator'])
                    precision_array_.append(precision_value)
                    # + confusion_matrix_text
                    model_results_text += f"\n\nThreshold:\t{threshold}\t\tMargin:\t{margin}" + \
                        accuracy_text + precision_score_text
            # try plotting the accuracy vs precision for different values of threshold here
                if len(thresholds) > 1:
                    plot_accuracy_precision(
                        accuracy_array_, precision_array_, thresholds, parsed_dict['model_name'], margin)
                final_results_text = model_results_text + final_results_text
                logger.info(final_results_text)
        else:
            final_results_text += "Only models with softmax and quantization enabled output is supported!"

    else:
        raise Exception("classification labels are not provided!")
    return 0


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
    if (len(sys.argv) >= 3):
        thresholds, margins = [], []
        for args in sys.argv[2::]:
            arg = args.split("=")
            if arg[0] == "thresholds":
                thresholds = util.get_list_from_strlist(arg[1])
            elif arg[0] == "margins":
                margins = util.get_list_from_strlist(arg[1])
                # if len(margins)>1:
                #   raise Exception("Only single values for margin is supported right now!")

        if thresholds == [] and margins == []:
            raise Exception(
                "Please pass thresholds and/or margin values as a list. e.g 'thresholds=[0.2,0.5,0.4] margins=[10]")
        yaml_input = sys.argv[1].split("=")
        if len(yaml_input) == 2:
            yaml_path = Path(yaml_input[1])
        else:
            yaml_path = Path(sys.argv[1])

        if (os.name == "posix"):
            if (util.is_windows_path(yaml_path)):
                yaml_path = util.get_linux_path(yaml_path)
        """user has entered a path to its own custom yaml file, handle that"""
        if Path(yaml_path).exists() and str(yaml_path).endswith(".yaml"):
            USER_WORK_DIR = util.get_abs_dir_from_file(yaml_path)
            yaml_input = util.load_yaml_file(yaml_path)
            yaml_test_list = list(yaml_input.keys())
            log_directory = USER_WORK_DIR + "/logs/"
            if not (Path(log_directory).exists()):
                Path(log_directory).mkdir(parents=True, exist_ok=True)
                logger.info(f"created log outputs directory {log_directory}")
            log_full_path = log_directory + log_file_name
            logging.basicConfig(filename=log_full_path,
                                level=log_level, format=log_format)
            for test in yaml_test_list:
                if (test == "default_values"):
                    continue
                logger.info(f"running precision measurement for {test}")
                start_time = time.time()
                try:
                    run_precision_measurements(
                        yaml_input[test], thresholds, margins)
                except Exception as e:
                    #   error_text = str(e) + ", try running the executor with the same input yaml file!"
                    logger.error(e)

                logger.info(
                    f"completed running precision measurement for {test}, took {(time.time() - start_time):.2f} seconds")
                gc.collect()
        else:
            raise Exception("yaml file doesn't exist!")
    else:
        raise Exception("provide a yaml file path, thresholds and/or margins!")
    return 0


if __name__ == '__main__':
    run_app()
