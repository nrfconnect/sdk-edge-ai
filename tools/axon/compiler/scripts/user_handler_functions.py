""" 
/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */
"""
import json
import logging
import numpy as np
import tensorflow as tf
import tflite_converter as tc
from utility import util

logger = logging.getLogger(__name__)


class UserHandleHelpers:
    @staticmethod
    def get_tflite_output_quantization(tflite_model_path):
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        op_scale = interpreter.get_output_details(
        )[0]['quantization_parameters']['scales'][0]
        op_zeropoint = interpreter.get_output_details(
        )[0]['quantization_parameters']['zero_points'][0]
        return op_scale, op_zeropoint


def user_handle_anomaly_detection_accuracy_results(input_yaml_dict, x_test, y_test, axon_inference_csv, output_shape, variants_object, variants, *args, **kwargs):
    """
    Calculates the mean error between the float input and axon output, float input and tflite output, tflite output and axon output

    The Anomaly Detection model is an autoencoder model, where the input and output are same shaped vectors
    The model is trained to reconstruct the input vector at the output layer
    * Input features to the model are calculated from 11 second audio recordings of a ToyCar.
    * The mel spectograms are calculated on the 11 second audio using the librosa library with the following paramters:
      sampling rate = 16000
      n_fft = 1024
      hop_length = 512
      n_mels = 128
      power = 2.0
    * The MelSpectogram is then converted into Log-Mel Energy in dB scale
      20.0 / power * numpy.log10(mel_spectrogram); output shape is (128, 344)
    * Central part of the Log-Mel Energy is extracted by selecting 200 values from 50-250 from a size of 0-344
    * The total input feature size is calculated using the frame size
    * The frame size is 5 so we get 196 feature frames; len(log_mel_spectrogram[0, :]) - frames + 1; 200 - 5 + 1 = 196;
    * The multiframe feature vectors is then calculated by concatenating 5 frames together
    * The final input feature vector size is 128 * 5 = 640 with a shape of (196, 640)

    The 11 second audio is thus converted into a feature of shape (196, 640) and fed to the model to get the prediction result
    The output result is determined by calculating a threshold on the mean error between the input and output vector across the full 196 frames for each 11 second audio
    The mean error is calculated as follows:
      mse = numpy.mean(numpy.square(input_features - prediction_results), axis=1) ; giving a shape of (196,)
      and then the error is calculated as : numpy.mean(mse) 
      giving a single value on which the threshold is used to determine if the audio is normal or anomalous

    Thus a full input size of 196 for an 11 second audio corresponds to a single inference output value

    The threshold was observed as follows for the ToyCar dataset when determining the higest acccuracy:
      Toy_Car_Id	Accuracy	Threshold
      1	          80.8	    10.29320244
      2         	85.2	    10.2318731
      3	          65	      10.22951121
      4	          87.3	    10.1355466

    If the user provides the full data set with y_test containing the labels (0 for normal, 1 for anomalous) and x_test containing 196 samples for each label value
      then the accuracy of the model can be calculated using a threshold value, 
      where the threshold value could be the average of the thresholds observed above(10.22) or a value which the user thinks is suitable for their use case
    Else a simple per element RMS error is presented to the user between the input and axon/tflite outputs
    """
    audio_length = input_yaml_dict.get('ad_audio_length', 11)  # 11 #seconds
    frames = input_yaml_dict.get('ad_frames', 5)  # 5
    sampling_rate = input_yaml_dict.get('ad_sampling_rate', 16000)  # 16000
    n_fft = input_yaml_dict.get('ad_n_fft', 1024)  # 1024
    hop_length = input_yaml_dict.get('ad_hop_length', 512)  # 512
    n_mels = input_yaml_dict.get('ad_n_mels', 128)  # 128
    power = input_yaml_dict.get('ad_power', 2.0)  # 2.0
    central_part_start = input_yaml_dict.get('ad_central_part_start', 50)  # 50
    central_part_end = input_yaml_dict.get('ad_central_part_end', 250)  # 250
    input_audio_samples_len = audio_length * sampling_rate  # 176000
    pad = n_fft // 2  # 512
    mel_spectogram_length = 1 + \
        np.floor((input_audio_samples_len + 2 *
                 pad - n_fft) / hop_length)  # 344
    mel_spectrogram_central_part_length = central_part_end - central_part_start  # 200
    feature_size = n_mels * frames  # 640
    per_audio_label_feature_count = mel_spectrogram_central_part_length - frames + 1  # 196
    per_audio_label_feature_shape = (
        per_audio_label_feature_count, feature_size)  # (196, 640)

    return_text = ""
    float_acc = 'NA'
    float_threshold = input_yaml_dict.get(
        'ad_float_threshold', 10.1355466)  # default threshold value
    tflite_threshold = input_yaml_dict.get(
        'ad_tflite_threshold', 10.89711235)  # default threshold value
    # input_yaml_dict.get('anomaly_detection_threshold', 10.89711235) #default threshold value
    axon_threshold = tflite_threshold
    # determine here if the user has provided the full data set with labels corresponding to each 196 input samples or just a few samples
    test_io_vector_ndx = np.array(
        util.convert_range_to_list(input_yaml_dict['test_vectors']))
    per_audio_label_dataset = False
    if len(y_test) == (len(x_test)//per_audio_label_feature_count):
        per_audio_label_dataset = True
        # figure out if the input will be sampled from a larger data set or not
        if len(test_io_vector_ndx) % per_audio_label_feature_count == 0:
            # get the corresponding vector in the labels for this range of input vectors
            y_test_first_index = test_io_vector_ndx[0]//per_audio_label_feature_count
            y_test_last_index = test_io_vector_ndx[-1]//per_audio_label_feature_count
            y_test = y_test[y_test_first_index:y_test_last_index+1]
        else:
            per_audio_label_dataset = False
    # disabling this as only the RMS error is required right now!
    per_audio_label_dataset = False

    sampled_x_test = np.array([x_test[test_vector_ndx]
                              for test_vector_ndx in test_io_vector_ndx])

    tflite_path = input_yaml_dict['tflite_model']
    op_scale, op_zeropoint = UserHandleHelpers.get_tflite_output_quantization(
        tflite_path)
    _, q8_tflite_results = tc.test_tflite_model(
        tflite_path, sampled_x_test, sampled_x_test, False, True)
    f32_tflite_results = (np.array(
        q8_tflite_results, dtype=np.float32).squeeze() - op_zeropoint)*op_scale
    q8_axon_inference_results = axon_inference_csv

    # this is used when the model output is not quantized
    if input_yaml_dict['disable_op_quantization']:
        f32_axon_inference_results = (
            axon_inference_csv/2**input_yaml_dict['op_radix']).astype(np.float32)
        q8_axon_inference_results = f32_axon_inference_results/op_scale + op_zeropoint
        q8_axon_inference_results = np.clip(
            np.round(q8_axon_inference_results), -128, 127).astype(np.int8)
    else:
        # need to dequantize the values to compare with the float input values
        f32_axon_inference_results = (
            np.array(axon_inference_csv, dtype=np.float32) - op_zeropoint)*op_scale

    if input_yaml_dict['float_model'] is not None:
        float_model = input_yaml_dict['float_model']
        _, float_results = tc.test_floating_point_model(
            float_model, sampled_x_test, sampled_x_test, False, True)

    if per_audio_label_dataset:
        return_text += ("\n\tThe input test data contains full audio features(196,) for each label, calculating accuracy based on thresholds")
        axon_pred = np.ones(y_test.shape)*-1
        tflite_pred = np.ones(y_test.shape)*-1
        float_pred = np.ones(y_test.shape)*-1
        # for each label in y_test, we need to calculate the mean error across the 196 input samples and compare it with the threshold
        for label_index, label in enumerate(y_test):
            data = sampled_x_test[label_index*per_audio_label_feature_count:(
                label_index+1)*per_audio_label_feature_count, :]
            # axon
            axon_pred_data = f32_axon_inference_results[label_index*per_audio_label_feature_count:(
                label_index+1)*per_audio_label_feature_count, :]
            axon_errors = np.mean(np.square(data - axon_pred_data), axis=1)
            axon_mean_error = np.mean(axon_errors)
            axon_pred[label_index] = 1 if axon_mean_error > axon_threshold else 0
            # tflite
            tflite_pred_data = f32_tflite_results[label_index*per_audio_label_feature_count:(
                label_index+1)*per_audio_label_feature_count, :]
            tflite_errors = np.mean(np.square(data - tflite_pred_data), axis=1)
            tflite_mean_error = np.mean(tflite_errors)
            tflite_pred[label_index] = 1 if tflite_mean_error > tflite_threshold else 0
            # float
            if input_yaml_dict['float_model'] is not None:
                float_pred_data = float_results[label_index*per_audio_label_feature_count:(
                    label_index+1)*per_audio_label_feature_count, :]
                float_errors = np.mean(
                    np.square(data - float_pred_data), axis=1)
                float_mean_error = np.mean(float_errors)
                float_pred[label_index] = 1 if float_mean_error > float_threshold else 0

        axon_acc = np.mean(y_test == axon_pred)
        tflite_acc = np.mean(y_test == tflite_pred)
        if input_yaml_dict['float_model'] is not None:
            float_acc = np.mean(y_test == float_pred)
        accuracy_results_text = f"\n\tAccuracy (test data set size {len(y_test)})\n"
        accuracy_results_text += f"\n\t\tFloat:\t{float_acc} @threshold {float_threshold:0.3f}"
        accuracy_results_text += f"\n\t\tTflite:\t{tflite_acc} @threshold {tflite_threshold:0.3f}"
        accuracy_results_text += f"\n\t\tAxon:\t{axon_acc} @threshold {axon_threshold:0.3f}"
        return_text += accuracy_results_text + "\n"

        if float_acc != 'NA':
            # get the Quantization Loss between float and tflite
            tflite_quantization_error_sample = (
                (float_acc - tflite_acc)/float_acc) * 100
            axon_quantization_error_sample = (
                (float_acc - axon_acc)/float_acc) * 100
            return_text += "\n\tQuantization Loss (w.r.t float model)%:\n"
            return_text += f"\n\t\tTflite:\t{tflite_quantization_error_sample:.2f} %"
            return_text += f"\n\t\tAxon:\t{axon_quantization_error_sample:.2f} %\n"

        # set the accuracy values in the variants object for final displaying results in the report
        if variants_object is not None:
            variants_object.set_axon_accuracy_value(
                variants, axons_acc=axon_acc)
            variants_object.set_axon_quantization_loss_value(
                variants, axons_q_loss=axon_quantization_error_sample)
    else:
        return_text += f"\n\tMean RMS error in last layer values for a test data set of size {len(test_io_vector_ndx)}\n"
        if input_yaml_dict['float_model'] is not None:
            # errors = np.sqrt(np.mean(np.square(sampled_x_test - float_results)))
            # acc = np.mean(errors)
            # return_text += f"\n\t Between the float input and float output is {acc:.4f}"
            # errors = np.sqrt(np.square(sampled_x_test - f32_tflite_results))
            errors = np.sqrt(
                np.mean(np.square(float_results - f32_tflite_results)))
            acc = np.mean(errors)
            return_text += f"\n\t\tfloat - tflite:\t{acc:.4f}"
            # errors = np.sqrt(np.square(sampled_x_test - f32_axon_inference_results))
            errors = np.sqrt(
                np.mean(np.square(float_results - f32_axon_inference_results)))
            acc = np.mean(errors)
            return_text += f"\n\t\tfloat - axon:\t{acc:.4f}"

        errors = np.sqrt(
            np.mean(np.square(f32_tflite_results - f32_axon_inference_results)))
        acc = np.mean(errors)
        return_text += f"\n\t\ttflite - axon:\t{acc:.4f}\n"

    return 0, return_text


def user_handle_fomo_fd_accuracy_results(input_yaml_dict, x_test, y_test, axon_inference_csv, output_shape, variants_object, variants, *args, **kwargs):
    """
    The FOMO FD model generates a forground and background heatmap of size 12x12 for an input image of size 96x96x3
    The channel 1 value is foreground which can be used to compare with the true labels generated from the bounding boxes

    The accuracy may not make complete sense here as the model is not trained to classify between different classes but to detect the presence of a face in an image
    Thus trying to figure out if the model detected a face in the region of the heatmap gives us a sense of accuracy for the model
    """
    def evaluate_foreground(pred_heatmap, gt_heatmap, threshold=0.5, iou_threshold=0.01):
        """
        Evaluate model performance on foreground heatmap prediction.

        Args:
            pred_heatmap (np.ndarray): Model's predicted foreground heatmap (12x12), values in [0,1].
            gt_heatmap (np.ndarray): Ground truth binary heatmap (12x12), values in {0,1}.
            threshold (float): Threshold to binarize predicted heatmap.

        Returns:
            accuracy, precision, recall, and F1-score.
        """
        # pred_binary = (pred_heatmap >= threshold).astype(np.uint8)

        # # Intersection and union for IoU
        # intersection = np.logical_and(pred_binary, gt_heatmap).sum()
        # union = np.logical_or(pred_binary, gt_heatmap).sum()

        # tp = np.sum((pred_binary == 1) & (gt_heatmap == 1))
        # tn = np.sum((pred_binary == 0) & (gt_heatmap == 0))
        # fp = np.sum((pred_binary == 1) & (gt_heatmap == 0))
        # fn = np.sum((pred_binary == 0) & (gt_heatmap == 1))

        # accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        # precision = tp / (tp + fp + 1e-6)
        # recall = tp / (tp + fn + 1e-6)
        # f1 = 2 * precision * recall / (precision + recall + 1e-6)

        pred_binary = (pred_heatmap >= threshold).astype(np.uint8)
        gt_binary = gt_heatmap.astype(np.uint8)
        _gt_fd = (gt_binary.sum(axis=(1, 2)) > 0).astype(np.int8)

        _accuracy, _precision, _recall, _f1, _fd = [], [], [], [], []

        for ndx, d in enumerate(gt_binary):
            pred_binary = (pred_heatmap[ndx] >= threshold).astype(np.uint8)
            gt_binary = gt_heatmap[ndx].astype(np.uint8)
            # Calculate true/false positives/negatives
            tp = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
            fp = np.logical_and(pred_binary == 1, gt_binary == 0).sum()
            fn = np.logical_and(pred_binary == 0, gt_binary == 1).sum()
            # tn = np.logical_and(pred_binary == 0, gt_binary == 0).sum()

            # total = tp + fp + fn + tn

            # Calculate intersection and union
            intersection = tp
            union = np.logical_or(pred_binary == 1, gt_binary == 1).sum()

            # Ensure IoU = 0 if there is no overlap
            if intersection == 0:
                iou = 0.0
            else:
                iou = intersection / (union + 1e-6)

            # Standard metrics
            # accuracy = ((tp + tn) / (total + 1e-6))
            # using this as accuracy for now
            accuracy = tp / (tp + fp + fn + 1e-6)
            precision = (tp / (tp + fp + 1e-6))
            recall = (tp / (tp + fn + 1e-6))
            f1 = (2 * precision * recall / (precision + recall + 1e-6))
            _accuracy.append(accuracy)
            _precision.append(precision)
            _recall.append(recall)
            _f1.append(f1)
            _fd.append(1 if iou > iou_threshold else 0)

        accuracy = np.mean(_accuracy)
        precision = np.mean(_precision)
        recall = np.mean(_recall)
        f1 = np.mean(_f1)
        acc = np.mean(np.array(_fd) == _gt_fd)
        return acc, precision, recall, f1

    # get the tflite inference done here.
    test_io_vector_ndx = np.array(
        util.convert_range_to_list(input_yaml_dict['test_vectors']))
    sampled_x_test = np.array([x_test[test_vector_ndx]
                              for test_vector_ndx in test_io_vector_ndx])
    if y_test is None:
        return -1, "\n\tNo test labels provided, cannot calculate accuracy results"
    sampled_y_test = np.array([y_test[test_vector_ndx]
                              for test_vector_ndx in test_io_vector_ndx])

    return_text = f"\n\tModel Results ( test data size {len(test_io_vector_ndx)}):"
    return_text += "\n\t\t\tAccuracy"  # \tPrecision\tRecall\tF1-score"
    # float_inference
    float_acc = "NA"
    if input_yaml_dict['float_model'] is not None:
        float_model = input_yaml_dict['float_model']
        _, float_output = tc.test_floating_point_model(
            float_model, sampled_x_test, None, False, True)
        # float_heatmap = np.array(float_output[:,:,:,0]<=float_output[:,:,:,1]).astype(np.uint8)
        float_acc, p, r, f = evaluate_foreground(
            float_output[:, :, :, 1], sampled_y_test)
        # \t{p:.4f}\t{r:.4f}\t{f:.4f}"
        return_text += f"\n\t\tFloat\t{float_acc:.4f}"

    # tflite inference
    tflite_path = input_yaml_dict['tflite_model']
    op_scale, op_zeropoint = UserHandleHelpers.get_tflite_output_quantization(
        tflite_path)
    _, q8_tflite_results = tc.test_tflite_model(
        tflite_path, sampled_x_test, None, False, True)
    tflite_output = (np.array(q8_tflite_results,
                     dtype=np.float32).squeeze() - op_zeropoint)*op_scale
    # tflite_heatmap = np.array(tflite_output[:,:,:,0]<=tflite_output[:,:,:,1]).astype(np.uint8)
    tflite_acc, p, r, f = evaluate_foreground(
        tflite_output[:, :, :, 1], sampled_y_test)
    # \t{p:.4f}\t{r:.4f}\t{f:.4f}"
    return_text += f"\n\t\tTflite\t{tflite_acc:.4f}"

    # axon inference results handling
    axon_output = axon_inference_csv.reshape(
        (-1, output_shape.depth, output_shape.height, output_shape.width))
    # this is used when the model output is not quantized
    if input_yaml_dict['disable_op_quantization']:
        axon_output = (
            axon_output/2**input_yaml_dict['op_radix']).astype(np.float32)
    else:
        axon_output = (np.array(axon_output, dtype=np.float32) -
                       op_zeropoint)*op_scale
    # axon_heatmap = np.array(axon_output[:,0,:,:]<=axon_output[:,1,:,:]).astype(np.uint8)
    axon_acc, axon_precision, r, f = evaluate_foreground(
        axon_output[:, 1, :, :], sampled_y_test)
    # \t{p:.4f}\t{r:.4f}\t{f:.4f}"
    return_text += f"\n\t\tAxon\t{axon_acc:.4f}"

    # axon_quantization_error_sample=None
    # if float_acc!='NA':
    #   #get the Quantization Loss between float and tflite
    #   tflite_quantization_error_sample = ((float_acc - tflite_acc)/float_acc) * 100
    #   axon_quantization_error_sample = ((float_acc - axon_acc)/float_acc) * 100
    #   return_text += f"\n\n\tQuantization Loss (w.r.t float model)%:\n"
    #   return_text += f"\n\t\tTflite:\t{tflite_quantization_error_sample:.2f} %"
    #   return_text += f"\n\t\tAxon:\t{axon_quantization_error_sample:.2f} %\n"

    if variants_object is not None:
        variants_object.set_axon_accuracy_value(variants, axons_acc=axon_acc)
        variants_object.set_axon_precision_value(variants, axon_precision)
        # variants_object.set_axon_quantization_loss_value(variants, axons_q_loss=axon_quantization_error_sample)
    return 0, return_text


def user_handle_fomo_fd_test_labels(y_test=None, tflite_file=None):
    """
    The FOMO FD model test labels are provided in a json file with the following format:
      {
        "version": 1,
        "samples": [
          {
            "sampleId": 102823324,
            "boundingBoxes": [
              {"label": 1, "x": 21, "y": 25, "w": 28, "h": 26}
            ]
          },
          {
            "sampleId": 102823320,
            "boundingBoxes": [
              {"label": 1, "x": 18, "y": 22, "w": 24, "h": 24}
            ]
          }
        ]
      }
    The bounding values have the x,y and the width and height of the bounding box around the detected object
    The input shape is 96x96x3 and the output shape is 12x12x2

    To get the true labels from the json file, we need to convert the bounding box values into the grid format of 12x12
    which is essentially calculating the heatmap out of the bounding box values in the json file

    The 12x12 heatmap output can then be used as true labels to compare with the model output
    """
    INPUT_SIZE = 96
    HEATMAP_SIZE = 12
    SCALE = INPUT_SIZE // HEATMAP_SIZE

    with open(y_test, 'r') as f:
        data = json.load(f)

    true_labels = []
    for samples in data['samples']:
        heatmap_init = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.uint8)
        bounding_boxes = samples['boundingBoxes']
        # print(f"Sample ID: {samples['sampleId']}")
        for box in bounding_boxes:
            # label = box['label']
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            # print(f"  Label: {label}, x: {x}, y: {y}, w: {w}, h: {h}")
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            c_start = int(np.ceil((x_min / SCALE)))
            r_start = int(np.ceil((y_min / SCALE)))
            c_end = int(np.floor(((x_max - 1) / SCALE)))
            r_end = int(np.floor(((y_max - 1) / SCALE)))
            heatmap_init[r_start:r_end+1, c_start:c_end+1] = 1
        true_labels.append(heatmap_init)
    return np.array(true_labels)


def user_handle_ei_audio_mike_test_labels(y_test=None, tflite_file=None):
    """
    takes the test data and the test label as inputs and does the required transformation on it to transform into a format the python executor can handle, 
    can also be used to perform transpose on the test data set  
    """
    if (y_test is not None):
        y_test = y_test[:, 0]  # labels in first column
        y_test = y_test - 1
    return y_test
