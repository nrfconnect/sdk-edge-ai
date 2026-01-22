""" 
/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */
"""
import copy
import yaml
import argparse
import tflite_converter as tc
import numpy as np
# from pandas_ml import ConfusionMatrix
from utility import util
# from matplotlib import pyplot
# , precision_score, roc_curve, f1_score #commented out imports used for precision calculation
from sklearn.metrics import confusion_matrix, classification_report, f1_score


def get_classification_report(y_true, y_pred, labels, total_labels, get_output_dict=False):
    # get labels indices
    labels_indices = range(len(labels))
    classification_report_text = classification_report(
        y_true, y_pred, target_names=labels, labels=labels_indices, zero_division=0, output_dict=get_output_dict)
    if get_output_dict:
        return classification_report_text
    return "\n" + classification_report_text


def get_precision_score_text(y_true, y_pred, classification_labels, total_labels, get_per_class=False):
    classification_report_dict = get_classification_report(
        y_true, y_pred, classification_labels, total_labels, get_output_dict=True)
    # precisions = precision_score(y_true, y_pred, average=None,zero_division=0)
    precisions_table = ""
    precisions_ = 0
    precisions_table += "\n\t\tLabels\t\tPrecision"
    for ndx, labels in enumerate(classification_labels):
        precisions_table += f"\n\t\t{labels}\t\t{classification_report_dict[labels]['precision']:.2f}"
        precisions_ += classification_report_dict[labels]['precision']
    precisions_avg_ = precisions_/total_labels
    precisions_text = f"\n\tPrecision:\t{precisions_avg_:.2f}"
    if (get_per_class):
        precisions_text += precisions_table
    return precisions_text, precisions_avg_


def get_consolidated_confusion_matrix(y_true, y_float_predict=np.empty, y_tflite_predict=np.empty, y_axons_predict=np.empty, labels=np.empty):
    """
    This function creates a consolidated confusion matrix with data from predicted value of float, tflite and axons model if passed to this function
    """
    cm_text = "\n\n\tConfusion Matrix (tflite float, tflite int8, axons int8)"
    float_cm, tflite_cm, axons_cm = np.array([]), np.array([]), np.array([])
    if (y_float_predict is not np.empty):
        float_cm = get_confusion_matrix(y_true, y_float_predict, labels=labels)
    if (y_tflite_predict is not np.empty):
        tflite_cm = get_confusion_matrix(
            y_true, y_tflite_predict, labels=labels)
    if (y_axons_predict is not np.empty):
        axons_cm = get_confusion_matrix(y_true, y_axons_predict, labels=labels)

    horizontal_label_names = "\n\t\t\t"
    horizontal_label_values = "\n"
    last_row_values = ""
    for act_ndx, actual_labels in enumerate(labels):
        pred_float_values = 0
        pred_tflite_values = 0
        pred_axon_values = 0
        horizontal_label_names += actual_labels + "\t"
        horizontal_label_values += "\t\t" + actual_labels + "\t"
        for prd_ndx, predict_labels in enumerate(labels):
            if (float_cm.size != 0):
                horizontal_label_values += str(float_cm[act_ndx, prd_ndx])
                pred_float_values += float_cm[act_ndx, prd_ndx]
            else:
                horizontal_label_values += "NA"
            horizontal_label_values += ","
            if (tflite_cm.size != 0):
                horizontal_label_values += str(tflite_cm[act_ndx, prd_ndx])
                pred_tflite_values += tflite_cm[act_ndx, prd_ndx]
            else:
                horizontal_label_values += "NA"
            horizontal_label_values += ","
            if (axons_cm.size != 0):
                horizontal_label_values += str(axons_cm[act_ndx, prd_ndx])
                pred_axon_values += axons_cm[act_ndx, prd_ndx]
            else:
                horizontal_label_values += "NA"
            horizontal_label_values += "\t"
        horizontal_label_values += f"{pred_float_values},{pred_tflite_values},{pred_axon_values}\n"

    cm_text += horizontal_label_names + "__actual_total__"
    cm_text += horizontal_label_values
    true_float_sums = np.zeros(len(labels), dtype=np.int32)
    true_tflite_sums = np.zeros(len(labels), dtype=np.int32)
    true_axons_sums = np.zeros(len(labels), dtype=np.int32)
    if (float_cm.size != 0):
        true_float_sums = np.sum(float_cm, axis=0)
    if (tflite_cm.size != 0):
        true_tflite_sums = np.sum(tflite_cm, axis=0)
    if (axons_cm.size != 0):
        true_axons_sums = np.sum(axons_cm, axis=0)

    for i, sum_val in enumerate(range(len(labels))):
        last_row_values += f"{true_float_sums[i]},{true_tflite_sums[i]},{true_axons_sums[i]}\t"

    cm_text += "\t\t__predicted_total__\t" + last_row_values + \
        f"{np.sum(float_cm).astype(np.int32)},{np.sum(tflite_cm).astype(np.int32)},{np.sum(axons_cm).astype(np.int32)}"
    return cm_text


def get_confusion_matrix(y_actual, y_predict, labels):
    # change the indexes to actual labels
    y_actual_labels = np.array([labels[x] for x in y_actual])
    y_predicted_labels = np.array([labels[x] for x in y_predict])

    # y_actu = pd.Series(y_test, name='Actual')
    # y_pred = pd.Series(float_results, name='Predicted')
    # df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

    cm = confusion_matrix(y_actual_labels, y_predicted_labels, labels=labels)
    # y_actu = pd.Series(y_actual_labels, name='Actual')
    # y_pred = pd.Series(y_predicted_labels, name='Predicted')
    # df_confusion = pd.crosstab(y_actu ,y_pred , rownames=['Actual'], colnames= ['Predicted'],  margins=True, margins_name="__all__")
    # print(df_confusion)
    return (cm)


def evaluate_models(test_data, test_labels, tflite_model, keras_model, simulator_exe, labels_order):
    acc_float, acc_tflite, acc_simulator = 0, 0, 0

    # load the test vectors
    x_test = np.load(test_data)
    if (len(x_test.shape) == 3):
        x_test = np.expand_dims(x_test, axis=3)
    y_test = np.load(test_labels)

    _ = tc.quantize_test_dataset(tflite_model, x_test)
    acc_tflite, tflite_results = tc.test_tflite_model(
        tflite_model, x_test, y_test, get_results=True)
    acc_float, float_results = tc.test_floating_point_model(
        keras_model, x_test, y_test, get_results=True)

    print(
        f"The Accuracies of the models are :\n float : {acc_float}\n tflite : {acc_tflite}")
    if (simulator_exe is None):
        return acc_float, acc_tflite
    else:
        acc_simulator, scaled_model_results = tc.test_scaled_model_on_simulator(
            tflite_model, x_test, y_test, simulator_exe, get_results=True)
        print(f" scaled(on simulator) : {acc_simulator} ")

        get_confusion_matrix(y_test, float_results, labels_order)
        get_confusion_matrix(y_test, tflite_results, labels_order)
        get_confusion_matrix(y_test, scaled_model_results, labels_order)

        # TODO write code to get accuracy, sensitivity and other evaluation parameters for the models

        # Get the first five arrays where the tflite and scaled results do not match and tflite matches the expected
        TEST_INDEX_COUNT = 10
        ndx = 0
        mismatch_count = 0
        mismatch_ndx_array = -1*np.ones((TEST_INDEX_COUNT, 3), dtype=np.int32)
        for result in tflite_results:
            if (result != scaled_model_results[ndx]) and (result == float_results[ndx]):
                mismatch_ndx_array[mismatch_count] = (
                    ndx, result, scaled_model_results[ndx])
                mismatch_count += 1
            if (mismatch_count >= TEST_INDEX_COUNT):
                break
            ndx += 1

        print(
            f"the first test input indices where the tflite infered correctly and the scaled model did not are : \n {mismatch_ndx_array}")

    return acc_float, acc_tflite, acc_simulator, mismatch_ndx_array


def get_vectors_where_reference_inferred_correctly(reference_label, test_label, true_label):
    # Get the first five arrays where the tflite and scaled results do not match and tflite matches the expected
    TEST_INDEX_COUNT = len(test_label)
    mismatch_count = 0
    mismatch_ndx_array = []  # -1*np.ones((TEST_INDEX_COUNT,3), dtype=np.int32)
    for ndx, result in enumerate(test_label):
        # Get True Positives
        # Get True Negatives
        # Get False Positives
        # Get False Negatives
        if (result != true_label[ndx]) and (reference_label[ndx] == true_label[ndx]):
            mismatch_ndx_array.append([ndx, reference_label[ndx], result])
            mismatch_count += 1
        if (mismatch_count >= TEST_INDEX_COUNT):
            break
    # print(f"the first test input indices where the tflite infered correctly and the scaled model did not are : \n {mismatch_ndx_array}")
    return mismatch_ndx_array


def get_model_accuracy(x_test, true_labels, simulator_labels, tflite_model=None, float_model=None, get_results=False):
    """
    returns a dictionary with accuracy results for float, tflite and simulator for the input test vector range
    Example : {float : 0.91, tflite : 0.90, simulator:0.90}, {'float_label':float_labels, 'tflite_label':tflite_labels}
    """
    # for simulator results
    float_acc, tflite_acc, simulator_acc = -1, -1, -1
    float_labels, tflite_labels = np.empty, np.empty
    if (simulator_labels is not None):
        simulator_acc = np.mean(simulator_labels == true_labels)
    # for tflite results
    if (tflite_model is not None):
        tflite_acc, tflite_labels = tc.test_tflite_model(
            tflite_model, x_test, true_labels, get_results=get_results)
    # test the float model
    if (float_model is not None):
        float_acc, float_labels = tc.test_floating_point_model(
            float_model, x_test, true_labels, get_results=get_results)
    if get_results:
        return {'float': float_acc, 'tflite': tflite_acc, 'simulator': simulator_acc}, {'float_label': float_labels, 'tflite_label': tflite_labels}
    return {'float': float_acc, 'tflite': tflite_acc, 'simulator': simulator_acc}


def get_labels(inference_csv, threshold=0, margin=0, quantized_softmax_op=False):
    total_count_of_labels = inference_csv.shape[1]
    classification_labels = [-1]*len(inference_csv)
    if quantized_softmax_op:
        inference_csv = (inference_csv + 128) * 0.00390625
        for ndx, op_vectors in enumerate(inference_csv):
            sorted_ndxs = np.argsort(op_vectors)
            highest_vector_ndx = sorted_ndxs[-1]
            second_highest_vector_ndx = sorted_ndxs[-2]
            if (op_vectors[highest_vector_ndx] < threshold) or (op_vectors[highest_vector_ndx] - op_vectors[second_highest_vector_ndx] < margin):
                classification_labels[ndx] = total_count_of_labels
            else:
                classification_labels[ndx] = highest_vector_ndx
    else:
        for ndx, op_vectors in enumerate(inference_csv):
            classification_labels[ndx] = np.argmax(op_vectors)
    return classification_labels


def run_threshold_regression(y_true, y_pred, classification_labels, total_label_count, threshold_start=0, threshold_step=0.01):
    thresholds = np.arange(threshold_start, 1, threshold_step)
    scores = [f1_score(y_true, get_labels(y_pred, t), average='micro')
              for t in thresholds]  # using f1_score
    # scores = [precision_score(y_true,get_labels(y_pred,t), average='macro', labels=range(0,total_label_count)) for t in thresholds] #using precision score
    # scores = [get_classification_report(y_true,get_labels(y_pred,t),classification_labels,total_label_count,True)['macro avg']['precision'] for t in thresholds] #using precision score from classification report
    idx = np.argsort(scores)[-1]
    return thresholds[idx], scores[idx]


class ModelVariantsClass:

    softmax_variant_name_dict = {  # this is dormant as a variant and fixed @ False
        True: '_softmax_skipped',
        False: '_softmax_enabled'}
    norm_ss_variant_name_dict = {  # this is dormant as a variant and fixed @ True
        True: '_normalized_scaleshift',
        False: '_non_normalized_scaleshifts'}
    op_q_variant_name_dict = {
        True: '_op_quantization_disabled',
        False: '_op_quantization_enabled'}
    transpose_kernel_variant_name_dict = {
        True: '_transposed_kernel',
        False: ''}
    conv2d_variant_name_dict = {  # this is dormant as a variant and fixed @ 'local_psum'
        'local_psum': '_local_psum',
        'inner': '_input_inner',
        'outer': '_input_outer'}
    psum_buffer_location_name_dict = {  # this is dormant as a variant and fixed @ 'interlayer_buffer'
        'interlayer_buffer': '_psum_in_interlayer_buf',
        'dedicated_memory': '_psum_in_psum_buf'}

    variant_settings = {'psum_location': None,
                        '2d_conv_setting': None,
                        'normalized_scaleshifts': None,
                        'softmax': None,
                        'quantization': None,
                        'transpose_kernel': None}
    memory_footprint = {'model_const_buffer_size': None,
                        'interlayer_buffer_size': None,
                        'psum_buffer_size': None,
                        'cmd_buffer_size': None}
    accuracy_results = {'float': None,
                        'tflite': None,
                        'axons': None}
    quantization_loss = {'tflite': None,
                         'axons': None}
    performance_metrics = {'inference_time': None,
                           'accuracy': None,
                           'precision': None,
                           'quantization_loss': None,
                           'flash_size': 0,
                           'ram_size': 0,
                           'total_memory': 0}

    """
  model_results_dict = { 'Model_Name' : 
                        {'variant_settings' :
                         { 
                          'psum_location' : '',
                          '2d_conv_setting' : '',
                          'normalized_scaleshifts' : '',
                          'softmax' :'',
                          'quantization' :'',
                          'transpose_kernel':'' },
                        'test_data_size' : 'integer',
                        'memory_footprint' :
                        {
                          'model_const_buffer_size' : 'integer',
                          'interlayer_buffer_size' : 'integer',
                          'psum_buffer_size' : 'integer',
                          'cmd_buffer_size' : 'integer'},
                        'profiling_ticks' : 'integer',
                        'accuracy_results':{
                          'float':'float_value',
                          'tflite':'float_value',
                          'axons' : 'float_value',},
                        'quantization_loss': {
                          'tflite':'float_value', 
                          'axons':'float_value'}
                        }
                        'performance_metrics' : {
                          'accuracy': float,
                          'flash_size' : integer,
                          'ram_size' : integer,
                          'total_memory' : integer,
                          'quantization_loss' : float,
                          'inference_time': integer                        
                        }
                      }
  """
    model_results_dict = {'variant_settings': variant_settings,
                          'test_data_size': None,
                          'memory_footprint': memory_footprint,
                          'profiling_ticks': None,
                          'accuracy_results': accuracy_results,
                          'quantization_loss': quantization_loss,
                          'performance_metrics': performance_metrics}

    variants = None
    variants_results = None
    lowest_memory_variants = None
    highest_accuracy_variants = None
    best_performance_variants = None
    compare_results_dict = None

    def __init__(self, run_all_variants, tflite_model_file, parsed_yaml_dict):
        self.variants = {}
        self.variants_results = {}
        self.lowest_memory_variants = {}
        self.highest_accuracy_variants = {}
        self.best_performance_variants = {}
        self.compare_results_dict = {}
        # find if softmax or 2d convolutions are present in the
        softmax_present, conv2d_present, norm_ss_present, tr_kernel_present = util.find_variant_settings(
            tflite_model_file)
        # handle boolean/str value of variant
        if run_all_variants:
            variant = 'all'
            if type(run_all_variants) is str:
                variant = run_all_variants
            if variant == 'all':
                # parsed_yaml_dict['psum_buffer_placement'] = list(
                #     self.psum_buffer_location_name_dict.keys())
                # parsed_yaml_dict['conv2d_setting'] = list(
                #     self.conv2d_variant_name_dict.keys())
                # parsed_yaml_dict['skip_softmax_op'] = list(
                #     self.softmax_variant_name_dict.keys())
                # parsed_yaml_dict['normalize_scaleshift'] = list(
                #     self.norm_ss_variant_name_dict.keys())
                parsed_yaml_dict['disable_op_quantization'] = list(
                    self.op_q_variant_name_dict.keys())
                parsed_yaml_dict['transpose_kernel'] = list(
                    self.transpose_kernel_variant_name_dict.keys())
            elif (variant == 'conv2d_settings') and conv2d_present:
                parsed_yaml_dict['conv2d_setting'] = list(
                    self.conv2d_variant_name_dict.keys())
            elif (variant == 'softmax') and softmax_present:
                parsed_yaml_dict['skip_softmax_op'] = list(
                    self.softmax_variant_name_dict.keys())
            elif (variant == 'normalized_scaleshifts') and norm_ss_present:
                parsed_yaml_dict['normalize_scaleshift'] = list(
                    self.norm_ss_variant_name_dict.keys())
                parsed_yaml_dict['normalize_scaleshift'] = list(
                    self.norm_ss_variant_name_dict.keys())
            elif (variant == 'op_quantization'):
                parsed_yaml_dict['disable_op_quantization'] = list(
                    self.op_q_variant_name_dict.keys())
            elif (variant == 'transpose_kernel'):
                parsed_yaml_dict['transpose_kernel'] = list(
                    self.transpose_kernel_variant_name_dict.keys())

        for conv_2d_setting in parsed_yaml_dict['conv2d_setting']:
            if not conv2d_present:
                conv2d = ""
            else:
                conv2d = self.conv2d_variant_name_dict[conv_2d_setting]
            for psum_location in parsed_yaml_dict['psum_buffer_placement']:
                if (not conv2d_present) or (conv_2d_setting == 'local_psum'):
                    psum_loc = ""
                else:
                    psum_loc = self.psum_buffer_location_name_dict[psum_location]
                for sftmx in parsed_yaml_dict['skip_softmax_op']:
                    if not softmax_present:
                        sm = ""
                    else:
                        sm = self.softmax_variant_name_dict[sftmx]
                    for nrm_ss in parsed_yaml_dict['normalize_scaleshift']:
                        if not norm_ss_present:
                            norm_ss = ""
                        else:
                            norm_ss = self.norm_ss_variant_name_dict[nrm_ss]
                        for trnspse_krnl in parsed_yaml_dict['transpose_kernel']:
                            if not tr_kernel_present:
                                tr_kernel = ""
                            else:
                                tr_kernel = self.transpose_kernel_variant_name_dict[trnspse_krnl]
                            for dsbl_opq in parsed_yaml_dict['disable_op_quantization']:
                                op_q = self.op_q_variant_name_dict[dsbl_opq]
                                input_dict = parsed_yaml_dict.copy()
                                if conv2d != "":
                                    input_dict['conv2d_setting'] = conv_2d_setting
                                    self.model_results_dict['variant_settings']['2d_conv_setting'] = conv_2d_setting
                                    if psum_loc != "" and conv2d != "local_psum":
                                        input_dict['psum_buffer_placement'] = psum_location
                                        self.model_results_dict['variant_settings']['psum_location'] = psum_location
                                if sm != "":
                                    input_dict['skip_softmax_op'] = sftmx
                                    self.model_results_dict['variant_settings']['softmax'] = sftmx
                                if norm_ss != "":
                                    input_dict['normalize_scaleshift'] = nrm_ss
                                    self.model_results_dict['variant_settings']['normalized_scaleshifts'] = nrm_ss

                                input_dict['transpose_kernel'] = trnspse_krnl
                                self.model_results_dict['variant_settings']['transpose_kernel'] = trnspse_krnl

                                input_dict['disable_op_quantization'] = dsbl_opq
                                self.model_results_dict['variant_settings']['quantization'] = dsbl_opq
                                # dormant settings should not pop up in the variant name
                                if len(parsed_yaml_dict['conv2d_setting']) < 2:
                                    conv2d = ""
                                if len(parsed_yaml_dict['psum_buffer_placement']) < 2:
                                    psum_loc = ""
                                if len(parsed_yaml_dict['skip_softmax_op']) < 2:
                                    sm = ""
                                if len(parsed_yaml_dict['normalize_scaleshift']) < 2:
                                    norm_ss = ""
                                if len(parsed_yaml_dict['transpose_kernel']) < 2:
                                    tr_kernel = ""
                                if len(parsed_yaml_dict['disable_op_quantization']) < 2:
                                    op_q = ""
                                variant_name = parsed_yaml_dict['model_name'] + conv2d + psum_loc + sm + norm_ss + \
                                    tr_kernel + \
                                    op_q
                                self.variants[variant_name] = input_dict
                                self.variants_results[variant_name] = copy.deepcopy(
                                    self.model_results_dict)

        if len(self.variants) == 1:
            # this is being run for just one variant, no need to have a very long model name
            key, self.variants[parsed_yaml_dict['model_name']
                               ] = self.variants.popitem()
            key, self.variants_results[parsed_yaml_dict['model_name']
                                       ] = self.variants_results.popitem()

    def get_variants(self):
        return self.variants

    def get_variants_result_dict(self):
        return self.variants_results

    def set_memory_footprint_result(self, variant, key, value):
        self.variants_results[variant]['memory_footprint'][key] = value
        # update the memory performance metric here
        if (key == 'model_const_buffer_size' or key == 'cmd_buffer_size'):
            self.variants_results[variant]['performance_metrics']['flash_size'] += value
        elif (key == 'psum_buffer_size' or key == 'interlayer_buffer_size'):
            self.variants_results[variant]['performance_metrics']['ram_size'] += value

        self.variants_results[variant]['performance_metrics']['total_memory'] += value

    def set_profiling_tick_result(self, variant, value):
        self.variants_results[variant]['profiling_ticks'] = value
        self.variants_results[variant]['performance_metrics']['inference_time'] = value

    def set_accuracy_results(self, variant, accuracy_dict):
        self.variants_results[variant]['accuracy_results']['float'] = float(
            accuracy_dict['float'])
        self.variants_results[variant]['accuracy_results']['tflite'] = float(
            accuracy_dict['tflite'])
        self.variants_results[variant]['accuracy_results']['axons'] = float(
            accuracy_dict['simulator'])
        self.variants_results[variant]['performance_metrics']['accuracy'] = float(
            accuracy_dict['simulator'])

    def set_axon_accuracy_value(self, variant, float_acc=None, tflite_acc=None, axons_acc=None):
        # self.variants_results[variant]['accuracy_results']['float'] = float(float_acc) if float_acc is not None else "NA"
        # self.variants_results[variant]['accuracy_results']['tflite'] =  float(tflite_acc) if tflite_acc is not None else "NA"
        self.variants_results[variant]['accuracy_results']['axons'] = float(
            axons_acc) if axons_acc is not None else "NA"
        self.variants_results[variant]['performance_metrics']['accuracy'] = float(
            axons_acc) if axons_acc is not None else "NA"

    def set_axon_quantization_loss_value(self, variant, axons_q_loss):
        self.variants_results[variant]['quantization_loss']['axons'] = float(
            axons_q_loss) if axons_q_loss is not None else "NA"
        self.variants_results[variant]['performance_metrics']['quantization_loss'] = float(
            axons_q_loss) if axons_q_loss is not None else "NA"

    def set_axon_precision_value(self, variant, axon_precision):
        self.variants_results[variant]['performance_metrics']['precision'] = float(
            axon_precision) if axon_precision is not None else "NA"

    def set_quantization_loss_results(self, variant, axons_q_loss, tflite_q_loss):
        self.variants_results[variant]['quantization_loss']['axons'] = float(
            axons_q_loss)
        self.variants_results[variant]['quantization_loss']['tflite'] = float(
            tflite_q_loss)
        self.variants_results[variant]['performance_metrics']['quantization_loss'] = float(
            axons_q_loss)

    def set_precision_result(self, variant, precision_value):
        self.variants_results[variant]['performance_metrics']['precision'] = float(
            precision_value)

    def set_test_data_set_size(self, variant, dataset_size):
        self.variants_results[variant]['test_data_size'] = dataset_size

    def get_variants_result_as_yaml(self):
        return yaml.dump(self.variants_results, default_flow_style=False)

    def get_compare_variant_results(self, as_yaml=False):
        lowest_memory_footprint = (2**32-1)
        lowest_memory_variant = ''

        highest_accuracy = -(2**32-1)
        highest_accuracy_variant = ''

        best_performance = (2**32-1)
        best_performance_variant = ''

        q_loss = 2**32-1
        for variant in self.variants_results:
            # find the variant with the lowest memory footprint
            memory_values = list(
                self.variants_results[variant]['memory_footprint'].values())
            if (memory_values[0] is not None):
                memory_footprint = sum(memory_values)
                if (memory_footprint < lowest_memory_footprint):
                    lowest_memory_footprint = memory_footprint
                    lowest_memory_variant = variant
            # find the variant with the highest accuracy
            if (self.variants_results[variant]['accuracy_results']['axons'] is not None):
                if (self.variants_results[variant]['accuracy_results']['axons'] > highest_accuracy):
                    highest_accuracy = self.variants_results[variant]['accuracy_results']['axons']
                    highest_accuracy_variant = variant
            # find the variant with the best performance and lowest quantization loss
            if (self.variants_results[variant]['profiling_ticks'] is not None) and self.variants_results[variant]['quantization_loss']['axons'] is not None:
                if (self.variants_results[variant]['profiling_ticks'] < best_performance) or (self.variants_results[variant]['quantization_loss']['axons'] < q_loss):
                    q_loss = self.variants_results[variant]['quantization_loss']['axons']
                    best_performance = self.variants_results[variant]['profiling_ticks']
                    best_performance_variant = variant
        if (lowest_memory_variant != ""):
            self.lowest_memory_variants[lowest_memory_variant] = self.variants_results[lowest_memory_variant]
        if (highest_accuracy_variant != ""):
            self.highest_accuracy_variants[highest_accuracy_variant] = self.variants_results[highest_accuracy_variant]
        if (best_performance_variant != ""):
            self.best_performance_variants[best_performance_variant] = self.variants_results[best_performance_variant]
        self.compare_results_dict = {"lowest_memory": self.lowest_memory_variants,
                                     "highest_accuracy": self.highest_accuracy_variants, "best_performance": self.best_performance_variants}
        if (as_yaml):
            return yaml.dump(self.compare_results_dict, default_flow_style=False)
        return self.compare_results_dict

    def get_compare_results_as_yaml(self):
        return yaml.dump(self.compare_results_dict, default_flow_style=False)

    def get_performance_metrics_as_table(self):
        performance_metrics_csv = "\n\tPerformance_Metrics\n\t\tModel_Variant\tInference_Time\tAccuracy\tPrecision\tQuantization_Loss\tFlash_Size\tRam_Size\tTotal_Memory\t"
        for variant in self.variants_results:
            performance_metrics_csv += "\n\t\t" + variant
            for metric in self.variants_results[variant]['performance_metrics']:
                performance_metrics_csv += "\t" + \
                    str(self.variants_results[variant]
                        ['performance_metrics'][metric])

        return performance_metrics_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--tflite_model',
        type=str,
        default="training\model\kws_16_49.tflite",
        help="""\
    the path to where the tflite model
    """)
    parser.add_argument(
        '--keras_model',
        type=str,
        default="training\model\kws_dscnn_16_49.h5",
        help="""\
    the path to the keras model
    """)
    parser.add_argument(
        '--simulator_exe',
        type=str,
        default="",
        help="""\
    the path to the simulator exe
    """)
    parser.add_argument(
        '--test_data',
        type=str,
        default="training/data/x_test16_49.npy",
        help="""\
    file name of the test dataset in float (Directory + file name). Should be in ".npy" format of size (#test_samples, x_axis, y_axis). If dataset is in 1D, shape should be (#test_samples, 1 ,x_axis).
    """)
    parser.add_argument(
        '--test_labels',
        type=str,
        default="training\data\y_test16_49.npy",
        help="""\
    Qualified file name of test dataset's labels (Directory + file name). Should be in ".npy" format. It's a numpy array of size (#test_samples), includings integer values showing each class.
    """)
    parser.add_argument(
        '--model_name',
        type=str,
        default="KWS_DSCNN_16_49_TINYML",
        help="""\
    The abbrevation of the model name saved in the precompiler.
    """)
    parser.add_argument(
        '--q_data',
        type=str,
        default="training\data\qdata_16_61_evaluate.npy",
        help="""\
    The path to save the quantized test data
    """)
    parser.add_argument(
        '--test_vectors_dir',
        type=str,
        default="test\data",
        help="""\
    The path to save the temporary files used to run the simulator exe test data
    """)
    parser.add_argument(
        '--labels_order',
        type=str,
        default=["Down", "Go", "Left", "No", "Off", "On", "Right", "Stop",
                 "Up", "Yes", "Silence", "Unknown"],  # ["person", "non_person"],#
        nargs="+",
        help="""\
    Shows what each number in labels represents. This is saved by order.
    """)
    args = parser.parse_args()
    evaluate_models(args.test_data, args.test_labels, args.tflite_model,
                    args.keras_model, args.simulator_exe, args.labels_order)
