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
import sys
import copy
import logging
import numpy as np
# import datetime as dt
import tflite as tflite
import model_wrapper as mw
from ctypes import CDLL, c_char_p, c_int, c_uint32, POINTER, pointer
# from pathlib import Path
from utility import util
from utility import operator_options as ops
from utility import cpu_operator_options as cpu_operator_options

"""create logger object"""
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(
    "%(funcName)s-%(levelname)s: %(message)s"))
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

COMPILER_STDOUT_FILE_NAME = "compiler_lib_stdout.tmp"
REMOVE_COMPILER_LIB_STDOUT_FILE = False
LOG_COMPILER_STDOUT_IN_LINUX = False


def generate_compiler_outputs(compiler_api_filepath: str, tflite_filename: str, model_name: str, x_test, y_test, quantized_data, labels: str, test_vectors_ndx: list, test_flag, normalized_scaleshift=False, disable_op_quantization=True, skip_softmax_op=False, op_radix=8, interlayer_buffer_size=0, psum_buffer_size=0, header_file_test_vector_count=0, conv2d_setting='local_psum', psum_buffer_loc='interlayer_buffer', transpose_kernel=False, tflite_axon_graph_object=None) -> str:
    camel_case_name = util.get_camel_case_model_name(model_name)
    # api_defn = ""
    file_content = ""

    """
    Definitions for the binary file generation
    """
    model_wrapper_ffi = mw.get_model_wrapper_cffi_object(compiler_api_filepath)
    if (model_wrapper_ffi is None):
        logger.critical(
            "model wrapper interface object invalid or incorrect compiler api file path")
        raise Exception("model wrapper object is None!")

    tflite_axon_wrapper = mw.ModelTfliteAxonEnumWrapper(model_wrapper_ffi)
    # get the c struct for the model layer descriptor
    model_desc_info_hdr_struct = mw.get_model_descriptor_info_struct(
        model_wrapper_ffi)
    model_descriptor_layer_struct = mw.get_model_descriptor_layer_struct(
        model_wrapper_ffi)
    model_meta_information_struct = mw.get_model_meta_information_struct(
        model_wrapper_ffi)
    model_compilation_options_struct = mw.get_model_compilations_options_struct(
        model_wrapper_ffi)
    model_info_struct_bin_length = mw.get_length_of_struct(
        model_wrapper_ffi, model_desc_info_hdr_struct)
    model_bin = mw.ModelDescriptionBin(
        model_wrapper_ffi, model_info_struct_bin_length)

    model_const_bin = bytearray()
    model_meta_info_bin = bytearray()
    model_descriptor_bin = bytearray()
    # model_labels_content_bin = bytearray()
    model_layer_desc_struct_bin = bytearray()
    model_compilation_options_bin = bytearray()

    tflite_test_file_content = ""
    tflite_test_layer_vectors = ""

    log_content = ""
    max_interior_layer_len = 0
    max_psum_layer_len = 0
    max_psum_layer_name = ""
    psum_needed = 0
    max_scaling_error = -1
    max_error_layer_name = ""
    max_single_scaling_error = -1
    max_single_error_layer_name = ""
    max_single_error_ch = 0
    # single_scale_shift_error_per_ch_results = ""
    axon_layer_num = 0
    tflite_identifier = 0
    total_layer_count = 0
    if (test_vectors_ndx.size >= 1):
        test_ndx = test_vectors_ndx[0]
    digit = 0
    pad_details = None
    # last_operation_name=""
    last_operator_options = None
    model_input_axon_layer_num = None
    model_output_axon_layer_num = None
    model_input_tf_index = None
    model_output_tf_index = None
    if isinstance(skip_softmax_op, list):
        skip_softmax_op = skip_softmax_op[0]
    """
    definitions for axonpro errata, mostly related to the non-zero padding
    """
    max_interior_padded_layer_len = 0  # TBR
    """
    end of defn
    """
    subgraph = tflite_axon_graph_object.get_tflite_subgraph()
    # tflite_operators_len = tflite_axon_graph_object.get_tflite_operators_len()
    interpreter = tflite_axon_graph_object.get_tflite_interpreter()
    inputs = tflite_axon_graph_object.get_tflite_input_details()
    outputs = tflite_axon_graph_object.get_tflite_output_details()
    tensor_details = tflite_axon_graph_object.get_tflite_tensor_details()
    ops_details = tflite_axon_graph_object.get_tflite_operator_details()

    model_input_operator_name = inputs[0]['name']
    model_output_operator_name = outputs[0]['name']

    operators_detail_graph = tflite_axon_graph_object.get_axon_operator_graph_info()
    axon_last_layer_num,  last_layer_ndx = tflite_axon_graph_object.get_axon_layer_num_of_output_operator()
    scale, model_ip_zeropoint = inputs[0]['quantization']
    input_shape = inputs[0]['shape']
    model_ip_datawidth = inputs[0]['dtype']

    if model_ip_datawidth == np.float32:
        if operators_detail_graph[0]['op_name'] == "QUANTIZE":
            logger.debug(
                "input is float, getting the input quantization from QUANTIZE operator")
            scale, model_ip_zeropoint = tensor_details[operators_detail_graph[0]
                                                       ['op_tensors'][0]]['quantization']
            model_ip_datawidth = tensor_details[operators_detail_graph[0]
                                                ['op_tensors'][0]]['dtype']
        else:
            raise Exception("float input models are not supported!!")
    if model_ip_datawidth == np.uint8:
        if operators_detail_graph[0]['op_name'] == "QUANTIZE":
            logger.debug(
                "input is float, getting the input quantization from QUANTIZE operator")
            scale, model_ip_zeropoint = tensor_details[operators_detail_graph[0]
                                                       ['op_tensors'][0]]['quantization']
            model_ip_datawidth = tensor_details[operators_detail_graph[0]
                                                ['op_tensors'][0]]['dtype']
        else:
            raise Exception(
                f"{model_name} has uint8 inputs, which are not supported!")

    if (skip_softmax_op and operators_detail_graph[last_layer_ndx]['op_name'] == "SOFTMAX"):
        last_layer_ndx = tflite_axon_graph_object.get_index_for_axon_layer_num(
            axon_last_layer_num-1)

    # getting the output scale of the actual final output
    model_op_scale, model_op_zeropoint = copy.deepcopy(
        tensor_details[operators_detail_graph[last_layer_ndx]['op_tensors'][0]]['quantization'])
    model_op_scaleshift = util.optimized_ip_scaling_shift(
        (model_op_scale), 16, 30, 31)[1]
    model_op_multiplier = int(
        np.round((model_op_scale)*(2**model_op_scaleshift)))

    output_shape = outputs[0]['shape']
    model_op_datawidth = outputs[0]['dtype']
    if model_op_datawidth == np.float32 and operators_detail_graph[len(operators_detail_graph)-1]['op_name'] != "DEQUANTIZE":
        # cehck if there is a dequantize if the output layer is float32
        raise Exception("float output models are not supported!!")

    model_ip_scaleshift = util.optimized_ip_scaling_shift(
        1/scale, 0, 20, 31)[1]
    model_ip_multiplier = int(np.round((1/scale)*(2**model_ip_scaleshift)))
    # quantized data calculation
    data_q = quantized_data

    model_ip_shape = ops.TensorShape(input_shape)
    # prepare a text file for adding all the constants into the file and then generate the const.h file
    file_content += "\n#define " + model_name.upper() + "_L0_INPUT_BATCH "+str(model_ip_shape.batch)+"\n#define " + model_name.upper() + "_L0_INPUT_CHANNEL "+str(model_ip_shape.depth)+"\n#define " + model_name.upper() + "_L0_INPUT_HEIGHT "+str(model_ip_shape.height)+"\n" + "#define " + model_name.upper() + "_L0_INPUT_WIDTH "+str(model_ip_shape.width) + \
        "\n" + "#define " + model_name.upper() + "_L0_INPUT_QUANTIZE_INV_SCALING_FACTOR "+str(model_ip_multiplier)+"\n" + "#define " + model_name.upper() + \
        "_L0_INPUT_QUANTIZE_INV_SCALING_FACTOR_SHIFT " + \
        str(int(model_ip_scaleshift))+"\n" + "#define " + model_name.upper() + \
        "_L0_INPUT_QUANTIZE_ZERO_POINT "+str(int(model_ip_zeropoint))
    file_content += "\n#define " + model_name.upper() + "_L0_INPUT_BYTEWIDTH " + \
        tflite_axon_wrapper.GetAxonByteWidthEnum(
            model_ip_datawidth).name + "\n"
    if (test_flag):
        prediction_digits = []
        # adding code to run the tflite inference to compare the output from the implementation on python
        test_ = data_q[test_ndx]
        # test_ = np.expand_dims(test_, axis=0)
        test_ = util.check_input_shape_for_inference(test_, input_shape)
        interpreter.set_tensor(inputs[0]['index'], test_)
        # Run inference.
        interpreter.invoke()
        # Save the class predictions for all test samples.
        output = interpreter.tensor(outputs[0]['index'])
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

        if (model_ip_shape.depth > 1) and model_ip_shape.shape_size > 3:  # we have a multichannel input
            test_ = test_.transpose(0, 3, 1, 2)
        tflite_test_file_content += util.write_array_to_file(
            test_.squeeze(), model_name.lower()+"_l0_test_input")

        # tflite_test_layer_vectors += "\n\nconst int8_t* "+ model_name.lower()+"_
        # input_test_vectors[] = {\n    " +  model_name.lower()+"_test_input,\n  };\n"
        tflite_test_layer_vectors += "\n\nconst int8_t* " + model_name.lower() + \
            "_layer_vectors[] = {\n    " + \
            model_name.lower()+"_l0_test_input,\n  "
    else:
        """
        const int8_t** tinyml_vww_input_test_vectors = NULL;
        const int8_t** tinyml_vww_layer_vectors = NULL;
        const int8_t** tinyml_vww_expected_output_vectors = NULL;
        """
        tflite_test_layer_vectors += "\n\nconst int8_t** " + \
            model_name.lower()+"_input_test_vectors = NULL;\n"
        tflite_test_layer_vectors += "\n\nconst int8_t** " + \
            model_name.lower()+"_layer_vectors = NULL;\n"
        tflite_test_layer_vectors += "\n\nconst int8_t** " + \
            model_name.lower()+"_expected_output_vectors = NULL;\n"

    """
    Prepare the model descriptor starting here
    """
    model_descriptor_defn = "\n{ "  # open(template_dir + "define_model_description.txt.template").read()
    model_descriptor_content = ""
    # paddings = []
    # reshape_ip_shape = None
    reshape_op_shape = None
    SWAP_MAXPOOL_OP = False
    for i, new_op in enumerate(operators_detail_graph):
        if new_op['axon_layer_num'] >= 0:
            axon_layer_num = new_op['axon_layer_num']
        tflite_identifier = new_op['index']
        previous_op_name = ""
        transpose_layer = False
        # LEAKY_RELU_FLAG=False
        # leaky_relu_alpha_scale=1
        filter_tensor = np.array([], dtype=np.int32)
        filter_offset = -1
        b_prime = np.array([], dtype=np.int32)
        b_prime_offset = -1
        scale_shift = np.array([], dtype=np.int8)
        scale_shift_offset = -1
        scale_q = np.array([], dtype=np.int32)
        scale_q_offset = -1
        cpu_op_additional_attrib_list = np.array([], dtype=np.int32)
        # cpu_operation_enum=0
        # layer_max_value=None
        # layer_min_value=None
        model_descriptor = model_descriptor_defn
        operators = subgraph.Operators(new_op['index'])
        operator_code = new_op['op_code']
        # ops_options_ndx = operators.BuiltinOptionsType()
        try:
            if (new_op["op_name"] == "RESHAPE") and i == 0 and test_flag:
                # have to perform special handling if RESHAPE is the first operator
                # check if the test input needs to be transposed
                # we only transpose the test input if the reshape puts some element in the channel
                # reshape_ip_shape = ops.TensorShape(tensor_details[subgraph.Operators(i).InputsAsNumpy()[0]]['shape'])
                reshape_op_shape = ops.TensorShape(
                    tensor_details[subgraph.Operators(i).OutputsAsNumpy()[0]]['shape'])
                if (reshape_op_shape.depth != 1) and test_flag:
                    # we have to transpose the input
                    original_shape = data_q.shape
                    data_q = data_q.reshape(len(
                        data_q), reshape_op_shape.height, reshape_op_shape.width, reshape_op_shape.depth)
                    data_q = data_q.transpose(0, 3, 1, 2)
                    data_q = data_q.reshape(original_shape)
                else:
                    data_q = data_q.reshape(len(
                        data_q), reshape_op_shape.height, reshape_op_shape.width, reshape_op_shape.depth)
            if new_op['options_initialized']:
                options = new_op['operator_options']
            else:
                options = new_op['operator_options'].CreateOptionsObject(
                    operator_code, operators, new_op, tensor_details, interpreter, operators_detail_graph, tflite_axon_wrapper)
                tflite_axon_graph_object.update_operator_options(
                    new_op, options)

            op_name = options.GetOperationName()
            meta_data = options.PrintAttributes()

            # FC layers need not be rotated. and also the layers after them should not have wrong input/output layers
            if i > 0:
                previous_op_name = operators_detail_graph[i-1]['op_name']
            transpose_layer = transpose_kernel and previous_op_name != "FULLY_CONNECTED"

            if ("SOFTMAX" in op_name and skip_softmax_op):
                raise KeyError(-901)
            if ("LEAKY_RELU" == op_name):
                raise KeyError(-910)

            if new_op['operator_support'].value == "COMBINED_VARIABLE":
                # figure out a way to handle the ops accordingly
                if new_op['op_name'] == "READ_VARIABLE" or new_op['op_name'] == "ASSIGN_VARIABLE":
                    raise KeyError(-913)

            if (new_op['operator_support'].value) == "CONVERTED_PASSTHROUGH":
                tflite_axon_graph_object.update_axon_operator_graph(new_op)
                raise KeyError(-914)

            options_error, error_text, error_action = options.GetOptionsError()
            if (options_error):
                if error_action == "CONTINUE":
                    logger.info(error_text)
                    new_op['operator_options'] = options
                    continue
                else:
                    raise Exception(
                        f"The operator {op_name} in the model {model_name} is not supported, {error_text}")

            if -1 in new_op['axon_ip_ops'] and new_op['axon_layer_num'] >= 0:
                model_input_axon_layer_num = new_op['axon_layer_num']
                model_input_tf_index = new_op['index']
                model_input_operator_name = op_name
            if new_op['axon_op_ops'] == [] and new_op['axon_layer_num'] >= 0:
                model_output_axon_layer_num = new_op['axon_layer_num']
                model_output_tf_index = new_op['index']
                model_output_operator_name = op_name
            
            ip_shape, kernel_shape, bias_shape = options.GetInputShapes()
            op_shape = options.GetOutputShape()

            layer_string = f"\n/*\n=== axon layer no {axon_layer_num} (tflite identifier) {tflite_identifier} ================================================================"
            file_content += layer_string
            meta_info_string = f"\n{op_name}\n{meta_data}\nip shape:{ip_shape}\nk/w shape:{kernel_shape}\nbias shape:{bias_shape}\nop shape:{op_shape}\n*/\n"
            file_content += meta_info_string
            log_content += meta_info_string
            logger.debug(
                f"axon layer no {axon_layer_num}[@{tflite_identifier}]: {op_name}:{meta_data}, ip shape:{ip_shape} k/w shape:{kernel_shape} bias shape:{bias_shape} op shape:{op_shape}")

            layer_name = f"{model_name.upper()}_AXON_LAYER_{str(axon_layer_num)}_TF_ID_{str(tflite_identifier)}_{op_name.upper()}"
            line_info = "\n#define " + layer_name.upper()

            # get the input and output bitwidths
            ip_datawidth, op_datawidth = options.GetIpOpBitwidth()
            ip_bytewidth_enum = tflite_axon_wrapper.GetAxonByteWidthEnum(
                ip_datawidth)

            file_content += line_info + "_INPUT_BYTEWIDTH " + ip_bytewidth_enum.name
            last_operator_options = None
            file_content = options.WriteOperatorAttributesToFile(
                file_content, line_info, last_operator_options)

            # ip_ndxs = options.GetInputTensorsNdx()
            op_ndxs = options.GetOutputTensorsNdx()
            ip_q, w_q, bias_q = options.GetIpQuantizationParameters()
            op_q = options.GetOpQuantizationParameters()

            ip_zeropoint = copy.deepcopy(ip_q['zero_points'])
            # ip_scales = copy.deepcopy(ip_q['scales'])
            # op_radix_scales = copy.deepcopy(op_q['scales']) #FIXME - this needs to be handled properly
            # op_radix_zp = copy.deepcopy(op_q['zero_points']) #FIXME - this needs to be handled properly
            op_scales = copy.deepcopy(op_q['scales'])
            op_zeropoint = copy.deepcopy(op_q['zero_points'])

            # last_operation_name=op_name
            last_operator_options = options

            # set flags inside the operator options here to influence the calculation of tensors and other values
            # FIXME need to set this once for the full model and forget
            # may be not we might use this to normalize scaleshifts based on layers
            options.SetNormalizedScaleshiftsFlag(normalized_scaleshift)
            # logger.debug(f"options setting the skip softmax op flag to {skip_softmax_op}")
            options.SetSkipSoftmaxOpFlag(skip_softmax_op)
            # logger.debug(f"options setting the disable output quantization flag to {disable_op_quantization}")
            options.SetOpQuantizationDisableFlag(disable_op_quantization)
            transpose_layer = options.SetTransposeKernelFlag(transpose_layer)

            if (i == last_layer_ndx) and disable_op_quantization:
                op_scales[0] = 1
                options.SetOpQScale(op_scales[0])
                op_zeropoint[0] = 0
                options.SetOpQZeropoint(op_zeropoint[0])
                # model_op_scale = 1
                # model_op_zeropoint = 0

            if (operator_code in cpu_operator_options.cpu_operators_list):
                logger.info(f"{op_name} is a cpu operation!")

                filter_tensor = options.GetFilterTensor()
                if (filter_tensor.size == 0):
                    file_content += line_info.lower()+"_filters NULL"
                scale_q, scale_shift = options.GetMultiplierandScaleshift()
                b_prime = options.GetBPrimeTensor()
                if (b_prime.size == 0):
                    file_content += line_info.lower()+"_bias_prime NULL"

                ip_zeropoint, op_zeropoint = options.GetIpOpZeropoints()
                cpu_op_additional_attrib_list = options.GetCpuAdditionalAttributesTensor()

                if (i == last_layer_ndx) and disable_op_quantization:
                    op_scales[0] = 1
                    op_zeropoint[0] = 0
            else:
                filter_tensor = options.GetFilterTensor()
                if filter_tensor.size == 0:
                    file_content += line_info.lower()+"_filters NULL"
                else:
                    file_content = options.WriteWeightTensorToFile(
                        file_content, layer_name.upper(), filter_tensor)
                scale_q, scale_shift = options.GetMultiplierandScaleshift()
                b_prime = options.GetBPrimeTensor()
                if b_prime.size == 0:
                    file_content += line_info.lower()+"_bias_prime NULL"
                else:
                    file_content = options.WriteBPrimeTensorToFile(
                        file_content, layer_name.upper(), b_prime)
                    file_content += line_info + \
                        "_BIAS_SIZE "+str(bias_shape[0])
                ip_zeropoint, op_zeropoint = options.GetIpOpZeropoints()
                ip_datawidth, op_datawidth = options.GetIpOpBitwidth()

                # check here if we need to swap the MAXPOOL operations height and width
                if (op_name == "AVERAGE_POOL_2D") or (op_name == "MAX_POOL_2D"):
                    if (i != 0 and operators_detail_graph[i-1]['op_name'] == "RESHAPE") and (i != last_layer_ndx and operators_detail_graph[i+1]['op_name'] == "RESHAPE"):
                        # check to see if we need to swap the MAXPOOL
                        maxpool_reshape_ip_ = ops.TensorShape(
                            tensor_details[subgraph.Operators(new_op['index']-1).InputsAsNumpy()[0]]['shape'])
                        maxpool_reshape_op_ = ops.TensorShape(tensor_details[subgraph.Operators(
                            new_op['index']+1).OutputsAsNumpy()[0]]['shape'])
                        # figure if the H and W are getting swapped before the input
                        if ((maxpool_reshape_ip_.get_shape()[1] == ip_shape[2]) and (maxpool_reshape_ip_.get_shape()[2] == ip_shape[1]) and
                            (maxpool_reshape_op_.get_shape()[1] == op_shape[2]) and (maxpool_reshape_op_.get_shape()[2] == op_shape[1])) or\
                                (maxpool_reshape_ip_.get_shape()[1] == ip_shape[2]):  # values were swapped after the input
                            # you have to swap the maxpool operation.
                            SWAP_MAXPOOL_OP = True
                if (op_name == 'STRIDED_SLICE'):
                    kernel_shape = options.GetFilterShape()

            axons_operation_enum = tflite_axon_wrapper.GetAxonOperationEnum(
                options.GetAxonsOperationEnumName())

            if axons_operation_enum.value == -1:
                raise Exception(
                    f"{op_name} is not a supported axon operation!")

            activation_function = options.GetActivationFunctionType()
            activation_function_enum = tflite_axon_wrapper.GetAxonActivationFunctionEnum(
                activation_function)
            file_content += line_info + "_ACTIVATION_FUNCTION "+activation_function_enum.name
            # and op_radix > 8:
            if (disable_op_quantization) and (i == last_layer_ndx):
                if (op_radix <= 0):
                    op_radix = util.get_output_radix(op_radix, np.array(
                        min(scale_shift)), model_op_scale, model_op_zeropoint, op_datawidth)
                else:
                    if (op_radix > scale_shift):
                        op_radix = scale_shift
                op_datawidth = np.int32
                scale_shift -= op_radix
                model_op_multiplier = 1
                model_op_zeropoint = 0
                model_op_scaleshift = op_radix

            file_content += line_info + \
                "_INPUT_ZERO_POINT "+str(ip_zeropoint[0])
            file_content += line_info + \
                "_OUTPUT_ZERO_POINT "+str(op_zeropoint[0])
            file_content = options.WriteScaleShiftsToFile(
                file_content, line_info, scale_q, scale_shift)
            op_bytewidth_enum = tflite_axon_wrapper.GetAxonByteWidthEnum(
                op_datawidth)
            file_content += line_info + "_OUTPUT_BYTEWIDTH " + op_bytewidth_enum.name

            ops_ip_shape = ops.TensorShape(
                np.array(options.GetOperatorInputShape()))
            ops_kernel_shape = ops.TensorShape(np.array(kernel_shape))
            ops_op_shape = ops.TensorShape(np.array(op_shape))

            # trying to figure out if this convolution is mfmc
            if (op_name == "CONV_2D") and (ops_ip_shape.depth > 1):
                psum_needed = 1
                feature_length = ops_op_shape.height*ops_op_shape.width
                if (max_psum_layer_len < (feature_length)):
                    max_psum_layer_len = (feature_length)
                    max_psum_layer_name = layer_name
            if (test_flag) and False:
                tflite_test_file_content = options.WriteTestVectorToFile(
                    tflite_test_file_content, line_info, interpreter.tensor(op_ndxs[0])(), op_datawidth, op_radix)
                tflite_test_layer_vectors += "  " + layer_name.lower()+"_tflite_op,\n  "

            model_descriptor = model_descriptor.replace(
                "[@LAYER_NO]", str(axon_layer_num))
            model_descriptor = model_descriptor.replace(
                "[@UPPER_LAYER_TYPE]", op_name.upper())
            model_descriptor = model_descriptor.replace(
                "[@lower_layer_type]", op_name.lower())
            # cpu_enum_name=""
            # if(cpu_operation_enum>0):
            #   cpu_enum_name = "+" + str(cpu_operation_enum)
            model_descriptor = model_descriptor.replace(
                "[@nrf_axon_nn_op_e]", (axons_operation_enum.name))
            # track_info = nodes['track_info'][i]
            # model_descriptor = model_descriptor.replace("[@CURRENT_TRACK_NO]",str(track_info[0]))
            # model_descriptor = model_descriptor.replace("[@INPUT_TRACK_NO]",str(track_info[1]))
            # model_descriptor = model_descriptor.replace("[@INPUT_MERGE_TRACK_NO]",str(track_info[2]))

            filter_bytewidth_enum = options.GetFilterBytewidthEnum()
            stride_x, stride_y = options.GetOperationStrides()
            ops_padding_details = options.GetOperationPaddings()
            dilation_x, dilation_y = options.GetOperationDilation()

            # fill up the weight/filter tensor if present
            model_const_bin, filter_offset = options.WriteTensorToBinFile(
                model_const_bin, filter_tensor)
            # fill up the bias tensor if present
            model_const_bin, b_prime_offset = options.WriteTensorToBinFile(
                model_const_bin, b_prime)
            # fill up the scale values
            model_const_bin, scale_q_offset = options.WriteTensorToBinFile(
                model_const_bin, scale_q)
            # fill up the scaleshift values
            model_const_bin, scale_shift_offset = options.WriteTensorToBinFile(
                model_const_bin, scale_shift)
            # fill up the custom attrib list
            model_const_bin, cpu_op_additional_attrib_offset = options.WriteTensorToBinFile(
                model_const_bin, cpu_op_additional_attrib_list)

            # FIXME swapping the values for testing
            if (op_name == "MAX_POOL_2D") and SWAP_MAXPOOL_OP:
                stride_x, stride_y = stride_y, stride_x
                ops_ip_shape.height, ops_ip_shape.width = ops_ip_shape.width, ops_ip_shape.height
                ops_op_shape.height, ops_op_shape.width = ops_op_shape.width, ops_op_shape.height
                ops_kernel_shape.height, ops_kernel_shape.width = ops_kernel_shape.width, ops_kernel_shape.height
                ops_padding_details.pad_right, ops_padding_details.pad_bottom = ops_padding_details.pad_bottom, ops_padding_details.pad_right
                ops_padding_details.pad_top, ops_padding_details.pad_left = ops_padding_details.pad_left, ops_padding_details.pad_top

            # fill up the models struct here
            """
            obsolete?
            # model_descriptor_layer_struct[0].track_no = track_info[0]
            # model_descriptor_layer_struct[0].input_track_no = track_info[1]
            # model_descriptor_layer_struct[0].input_merge_track_no = track_info[2]
            """
            model_descriptor_layer_struct[0].input_id_cnt = len(
                new_op['axon_ip_ops'])
            model_descriptor += f"\n{op_name.lower()}_layer_{i}_axon_{axon_layer_num}_tfid_{tflite_identifier}_input_count = {str(model_descriptor_layer_struct[0].input_id_cnt)}, "

            if model_descriptor_layer_struct[0].input_id_cnt > 1:
                for input_idx in range(model_descriptor_layer_struct[0].input_id_cnt):
                    # update the input shapes accordingly
                    if new_op['axon_ip_ops'][input_idx] >= 0:
                        input_operator_index = tflite_axon_graph_object.get_index_for_axon_layer_num(
                            new_op['axon_ip_ops'][input_idx])
                    elif new_op['axon_ip_ops'][input_idx] == -1:  # input to the model
                        input_operator_index = tflite_axon_graph_object.get_index_of_input_operator(i)
                    input_operators = operators_detail_graph[input_operator_index]['op_tensors']
                    input_ops_shape = ops.TensorShape(
                        tensor_details[input_operators[0]]['shape'])
                    input_datatype_enum = tflite_axon_wrapper.GetAxonByteWidthEnum(
                        tensor_details[input_operators[0]]['dtype'])
                    model_descriptor_layer_struct[0].input_ids[input_idx] = new_op['axon_ip_ops'][input_idx]
                    model_descriptor_layer_struct[0].input_dimensions[input_idx].height = input_ops_shape.height
                    model_descriptor_layer_struct[0].input_dimensions[input_idx].width = input_ops_shape.width
                    shape_text = "shapes_(C,H,W,DW)"
                    if transpose_layer:
                        # FIXME Testing here transpose
                        # swap the height and width of the operators
                        shape_text = "shapes_(C,W,H,DW)"
                        model_descriptor_layer_struct[0].input_dimensions[input_idx].height, model_descriptor_layer_struct[0].input_dimensions[
                            input_idx].width = model_descriptor_layer_struct[0].input_dimensions[input_idx].width, model_descriptor_layer_struct[0].input_dimensions[input_idx].height
                    model_descriptor_layer_struct[0].input_dimensions[input_idx].channel_cnt = input_ops_shape.depth
                    model_descriptor_layer_struct[0].input_dimensions[input_idx].byte_width = input_datatype_enum.value
                    model_descriptor += f"\n{op_name.lower()}_layer_{i}_axon_{axon_layer_num}_tfid_{tflite_identifier}_input_id_{input_idx} = {model_descriptor_layer_struct[0].input_ids[input_idx]}, "
                    model_descriptor += f"\n{op_name.lower()}_layer_{i}_axon_{axon_layer_num}_tfid_{tflite_identifier}_input_id_{input_idx}_{shape_text} = {model_descriptor_layer_struct[0].input_dimensions[input_idx].channel_cnt, model_descriptor_layer_struct[0].input_dimensions[input_idx].height,model_descriptor_layer_struct[0].input_dimensions[input_idx].width,model_descriptor_layer_struct[0].input_dimensions[input_idx].byte_width}, "
            else:
                model_descriptor_layer_struct[0].input_ids[0] = new_op['axon_ip_ops'][0]
                model_descriptor_layer_struct[0].input_dimensions[0].height = ops_ip_shape.height
                model_descriptor_layer_struct[0].input_dimensions[0].width = ops_ip_shape.width
                if transpose_layer:
                    # FIXME Testing here transpose
                    # swap the height and width of the operators
                    model_descriptor_layer_struct[0].input_dimensions[0].height, model_descriptor_layer_struct[0].input_dimensions[
                        0].width = model_descriptor_layer_struct[0].input_dimensions[0].width, model_descriptor_layer_struct[0].input_dimensions[0].height
                model_descriptor_layer_struct[0].input_dimensions[0].channel_cnt = ops_ip_shape.depth
                model_descriptor_layer_struct[0].input_dimensions[0].byte_width = ip_bytewidth_enum.value
                model_descriptor += f"\n{op_name.lower()}_layer_{i}_axon_{axon_layer_num}_tfid_{tflite_identifier}_input_id_0 = {model_descriptor_layer_struct[0].input_ids[0]}, "
                model_descriptor += f"\n{op_name.lower()}_layer_{i}_axon_{axon_layer_num}_tfid_{tflite_identifier}_input_id_0_shapes_(C,H,W,DW) = {model_descriptor_layer_struct[0].input_dimensions[0].channel_cnt,model_descriptor_layer_struct[0].input_dimensions[0].height,model_descriptor_layer_struct[0].input_dimensions[0].width,model_descriptor_layer_struct[0].input_dimensions[0].byte_width}, "

            model_descriptor_layer_struct[0].nn_operation = (
                axons_operation_enum.value)
            model_descriptor_layer_struct[0].filter_dimensions.height = ops_kernel_shape.height
            model_descriptor_layer_struct[0].filter_dimensions.width = ops_kernel_shape.width
            model_descriptor_layer_struct[0].filter_dimensions.channel_cnt = ops_kernel_shape.depth
            model_descriptor_layer_struct[0].filter_dimensions.byte_width = filter_bytewidth_enum.value
            model_descriptor_layer_struct[0].output_dimensions.height = ops_op_shape.height
            model_descriptor_layer_struct[0].output_dimensions.width = ops_op_shape.width
            model_descriptor_layer_struct[0].output_dimensions.channel_cnt = ops_op_shape.depth
            model_descriptor_layer_struct[0].output_dimensions.byte_width = op_bytewidth_enum.value
            if op_name == "CONCATENATION":
                model_descriptor_layer_struct[0].concatenate_axis = options.GetConcatenateAxis(
                )
                if transpose_layer:
                    if model_descriptor_layer_struct[0].concatenate_axis == 1:
                        model_descriptor_layer_struct[0].concatenate_axis = 2
                    elif model_descriptor_layer_struct[0].concatenate_axis == 2:
                        model_descriptor_layer_struct[0].concatenate_axis = 1
            model_descriptor_layer_struct[0].stride_x = stride_x
            model_descriptor_layer_struct[0].stride_y = stride_y
            model_descriptor_layer_struct[0].dilation_x = dilation_x
            model_descriptor_layer_struct[0].dilation_y = dilation_y
            model_descriptor_layer_struct[0].pad_left = ops_padding_details.pad_left
            model_descriptor_layer_struct[0].pad_right = ops_padding_details.pad_right
            model_descriptor_layer_struct[0].pad_top = ops_padding_details.pad_top
            model_descriptor_layer_struct[0].pad_bottom = ops_padding_details.pad_bottom
            if transpose_layer:
                # FIXME Testing here transpose
                if not (op_name == "STRIDED_SLICE"):
                    model_descriptor_layer_struct[0].filter_dimensions.height, model_descriptor_layer_struct[
                        0].filter_dimensions.width = model_descriptor_layer_struct[0].filter_dimensions.width, model_descriptor_layer_struct[0].filter_dimensions.height
                model_descriptor_layer_struct[0].stride_x, model_descriptor_layer_struct[
                    0].stride_y = model_descriptor_layer_struct[0].stride_y, model_descriptor_layer_struct[0].stride_x
                model_descriptor_layer_struct[0].dilation_x, model_descriptor_layer_struct[
                    0].dilation_y = model_descriptor_layer_struct[0].dilation_y, model_descriptor_layer_struct[0].dilation_x
                model_descriptor_layer_struct[0].pad_bottom, model_descriptor_layer_struct[
                    0].pad_right = model_descriptor_layer_struct[0].pad_right, model_descriptor_layer_struct[0].pad_bottom
                model_descriptor_layer_struct[0].pad_top, model_descriptor_layer_struct[
                    0].pad_left = model_descriptor_layer_struct[0].pad_left, model_descriptor_layer_struct[0].pad_top
                model_descriptor_layer_struct[0].output_dimensions.height, model_descriptor_layer_struct[
                    0].output_dimensions.width = model_descriptor_layer_struct[0].output_dimensions.width, model_descriptor_layer_struct[0].output_dimensions.height
            # overriding the top and bottom pad values with front and back values for the channel operation
            if (op_name == "CHANNEL_PAD"):
                model_descriptor_layer_struct[0].pad_top = ops_padding_details.pad_front
                model_descriptor_layer_struct[0].pad_bottom = ops_padding_details.pad_back
            model_descriptor_layer_struct[0].input_zero_point = ip_zeropoint[0]
            model_descriptor_layer_struct[0].output_zero_point = op_zeropoint[0]
            model_descriptor_layer_struct[0].scale_shift_cnt = scale_shift.size
            model_descriptor_layer_struct[0].activation_function = activation_function_enum.value
            model_descriptor_layer_struct[0].bias_prime.offset = np.array(
                b_prime_offset).astype(np.uint64)
            model_descriptor_layer_struct[0].output_multipliers.offset = np.array(
                scale_q_offset).astype(np.uint64)
            model_descriptor_layer_struct[0].scale_shifts.offset = np.array(
                scale_shift_offset).astype(np.uint64)
            model_descriptor_layer_struct[0].filter.offset = np.array(
                filter_offset).astype(np.uint64)
            model_descriptor_layer_struct[0].cpu_op_additional_attributes_count = np.array(
                cpu_op_additional_attrib_list.size).astype(np.uint32)
            model_descriptor_layer_struct[0].cpu_op_additional_attributes.offset = np.array(
                cpu_op_additional_attrib_offset).astype(np.uint64)

            # add that into a bytearray for the whole struct
            model_layer_desc_struct_bin.extend(mw.get_bytearray_from_struct(
                model_wrapper_ffi, model_descriptor_layer_struct))
            model_descriptor_layer_struct = mw.clear_model_descriptor_layer_struct(
                model_descriptor_layer_struct)
            # define_layers_content +=api_defn
            model_descriptor += "\n},"
            model_descriptor_content += model_descriptor
            max_interior_layer_len = max(
                util.get_length_from_shape(op_shape), max_interior_layer_len)
            pad_details = options.GetOperationPaddings()
            max_interior_padded_layer_len = max(util.get_length_from_shape(
                ip_shape, pad_details), max_interior_padded_layer_len)
            # update the operator_options in the graph with the actual operator option class object
            new_op['operator_options'] = options
            total_layer_count += 1
        except ValueError as val_err:
            logger.error(val_err.args[0])
            if (len(val_err.args) > 1):
                if val_err.args[1] == "ERR_CHANNEL_PADDING" or val_err.args[1] == "ERR_NO_SUPPORTED_OP_AFTER_PAD" or val_err.args[1] == "ERR_SAME_PADDING_OP":
                    raise Exception(f"{val_err.args[1]}")
        except KeyError as key_err:
            if key_err.args[0] == -901:
                logger.info(op_name + " operator supported but skipped!")
            elif key_err.args[0] == -902:
                # logger.error(ops_details[i]["op_name"]+" operator before softmax has an activation function, try skipping softmax op!")
                logger.error(
                    new_op["op_name"]+" operator before softmax has an activation function, try skipping softmax op!")
            elif key_err.args[0] == -909:
                # logger.error(ops_details[i]["op_name"]+" operator has a fused activation function followed by a LeakyReLU Operator, cannot set leaky_relu as activation function!")
                logger.error(
                    new_op["op_name"]+" operator has a fused activation function followed by a LeakyReLU Operator, cannot set leaky_relu as activation function!")
            elif key_err.args[0] == -910:
                logger.info(
                    f"{op_name} is supported as an Activation Function and not an operator!")
            elif key_err.args[0] == -912:
                logger.error(
                    f"{op_name} cannot set custom actication function to enable running sigmoid and tanh")
            elif key_err.args[0] == -913:
                # ops_combined_string = " ".join(operators_detail_graph[combined_op_ndx]['op_name'] + f"[@{combined_op_ndx}]"  for combined_op_ndx in new_op['combined_ops'])
                ops_combined_string = " ".join(
                    ops_details[combined_op_ndx]['op_name'] + f"[@{combined_op_ndx}]" for combined_op_ndx in new_op['combined_ops'])
                logger.info(
                    f"{new_op['op_name']}[@{tflite_identifier}] is supported and combined with {ops_combined_string} as a PERSISTENT_VARIABLE!")
            elif key_err.args[0] == -914:
                logger.info(
                    f"{new_op['op_name']} was converted to {op_name} so that it can be a PASSTHROUGH Operation!")
            elif key_err.args[0] == -917:
                logger.info(
                    f"{new_op['op_name']} is a PASSTHROUGH Operation!")
            elif key_err.args[0] == -920:
                logger.error(
                    f"{new_op['op_name']} encountered error when handling operator attributes before CPU Extension Operation!")
            elif key_err.args[0] == -921:
                logger.error(
                    f"{new_op['op_name']} encountered error when setting custom activation function before CPU Extension Operation!")
            elif key_err.args[0] == -922:
                logger.error(
                    "CPU Extension Operation is 'None'!")
            elif key_err.args[0] == -923:
                logger.error(
                    "CPU Extension Operation Handle threw an error!")
            else:
                logger.warning(new_op["op_name"]+" operator not supported!")
                raise Exception(key_err)
        except Exception as e:
            logger.critical(
                f"Exception: {e} in {new_op['op_name']} operation @ axon_layer_num {axon_layer_num} ,exiting....")
            raise Exception(f"{e}")
        except AssertionError as a:
            logger.critical(
                f"ASSERT! {a} in {new_op['op_name']} operation @ axon_layer_num {axon_layer_num} ,exiting....")
            raise Exception(f"{a}")
    """
    #end of for each operations
    """
    if (test_flag):
        # tflite_test_layer_vectors += "};\n\nconst int8_t* "+model_name.lower()+"_expected_output_vectors[] = {\n    " + layer_name.lower()+"_tflite_op,\n  };\n"
        tflite_test_layer_vectors += "};\n\n"

    tflite_test_file_content += tflite_test_layer_vectors
    model_descriptor_bin.extend(model_layer_desc_struct_bin)

    # src_file_content = ""#open(template_dir +"axonpro_model_base.c.template").read()
    if (max_psum_layer_name == ""):
        max_psum_layer_name = layer_name

    # src_file_content = src_file_content.replace("[@MODEL_OUTPUT_DATAWIDTH]", util.get_std_ctype_definition(model_op_datawidth))
    # src_file_content = src_file_content.replace("[@MAX_PSUM_LAYER]", max_psum_layer_name)
    # src_file_content = src_file_content.replace("[@UPPER_MODEL_NAME]", model_name.upper())
    # src_file_content = src_file_content.replace("[@lower_model_name]", model_name.lower())
    # src_file_content = src_file_content.replace("[@CamelModelName]", camel_case_name)

    """
    #define [@UPPER_MODEL_NAME]_MAX_PARALLEL_TRACKS // maximum number of concurrent tracks in the model. Typically 1.
    #define [@UPPER_MODEL_NAME]_MIN_IO_BUFFER_LENGTH // length in bytes of the largest inner layer output
    #define [@UPPER_MODEL_NAME]_INPUT_LENGTH // length in bytes of the input vector
    #define [@UPPER_MODEL_NAME]_OUPUT_LENGTH // length in bytes of the output vector
    #define [@UPPER_MODEL_NAME]_LAYER_COUNT // number of descrete operations in the model.

    """
    file_content += "\n#define " + model_name.upper() + "_MIN_IO_BUFFER_LENGTH (" + \
        str(max_interior_layer_len) + \
        ") // length in bytes of the largest inner layer output"
    file_content += "\n#define " + model_name.upper() + "_INPUT_LENGTH (" + \
        str(util.get_size_from_tensor_shape(input_shape)) + \
        ") // length in bytes of the input vector"
    file_content += "\n#define " + model_name.upper() + "_OUTPUT_LENGTH (" + \
        str(util.get_size_from_tensor_shape(output_shape, op_datawidth)) + \
        ") // length in bytes of the output vector"
    file_content += "\n#define " + model_name.upper() + "_LAYER_COUNT (" + \
        str(axon_layer_num)+") // number of descrete operations in the model."
    file_content += "\n#define " + model_name.upper() + "_PSUM_NEEDED (" + \
        str(psum_needed)+") //flag to detect if we need a PSUM."

    # file_content+="\n#define "+ model_name.upper() +"_CMD_BUF_LEN ("+str(np.int32(command_buff_len))+") // the length of the commmand buffer"#REMOVE
    if total_layer_count != (axon_layer_num+1):
        logger.debug(
            f"total Layer Count {total_layer_count} does not match with Total Supported Axon Layers {axon_layer_num+1} due to skipping of layers")
    # model_layer_cnt_bin = bytearray(np.array([total_layer_count]))
    # get the last layer number using the output tensor index

    model_meta_information_struct[0].model_layer_cnt = total_layer_count
    # Fill up the model desc info struct which is acting as the header for the binary file
    model_meta_information_struct[0].model_labels.offset = np.array(
        -1).astype(np.uint32)
    model_meta_information_struct[0].model_name.offset = np.array(
        -1).astype(np.uint32)
    model_meta_information_struct[0].input_quant.mult = model_ip_multiplier
    model_meta_information_struct[0].input_quant.round = model_ip_scaleshift
    model_meta_information_struct[0].input_quant.zero_point = model_ip_zeropoint
    model_meta_information_struct[0].output_dequant.mult = model_op_multiplier
    model_meta_information_struct[0].output_dequant.round = model_op_scaleshift
    model_meta_information_struct[0].output_dequant.zero_point = model_op_zeropoint
    model_desc_info_hdr_struct[0].title.offset = np.array(-1).astype(np.uint32)
    model_desc_info_hdr_struct[0].title.length = 28
    model_desc_info_hdr_struct[0].version.offset = np.array(
        -1).astype(np.uint32)
    model_desc_info_hdr_struct[0].version.length = 4
    model_desc_info_hdr_struct[0].consts.offset = np.array(
        -1).astype(np.uint32)
    # model_desc_info_hdr_struct[0].consts.length = len(model_const_bin)
    model_desc_info_hdr_struct[0].meta.offset = np.array(-1).astype(np.uint32)
    # model_desc_info_hdr_struct[0].layers.length = len(model_descriptor_bin)
    model_desc_info_hdr_struct[0].layers.offset = np.array(
        -1).astype(np.uint32)

    model_desc_info_hdr_struct[0].compilation_option.offset = np.array(
        -1).astype(np.uint32)
    """update compiler options here"""
    model_compilation_options_struct[0].interlayer_buffer_size = interlayer_buffer_size
    model_compilation_options_struct[0].psum_buffer_size = psum_buffer_size
    model_compilation_options_struct[0].header_file_test_vector_cnt = header_file_test_vector_count
    if psum_needed:
        model_compilation_options_struct[0].convolution_2d_setting = tflite_axon_wrapper.GetAxonproConv2DSettingValue(
            conv2d_setting)
        if (conv2d_setting != 'local_psum'):
            model_compilation_options_struct[0].psum_buffer_placement = tflite_axon_wrapper.GetAxonPsumBufferPlacementValue(
                psum_buffer_loc)
    else:
        model_compilation_options_struct[0].convolution_2d_setting = tflite_axon_wrapper.GetAxonproConv2DSettingValue(
            'local_psum')  # default value
        model_compilation_options_struct[0].psum_buffer_placement = tflite_axon_wrapper.GetAxonPsumBufferPlacementValue(
            'interlayer_buffer')
    model_compilation_options_struct[0].log_level = logger.getEffectiveLevel()
    if (op_radix):
        file_content += "\n#define " + model_name.upper() + "_OUTPUT_RADIX (" + \
            str(op_radix)+") // the op radix set when output quantization is disabled"
    labels_np = np.array(labels)
    if (labels_np.size == 1):
        declare_labels = "\n\nstatic const char **" + \
            model_name.lower()+"_labels = NULL;\n"
    else:
        labels_bin_string = ''
        declare_labels = "\n\nstatic const char *" + \
            model_name.lower()+"_labels[] = {\n"
        for ind, val in enumerate(labels):
            declare_labels += ' '*4+"\"" + \
                str(val).upper()+"\", // "+str(ind)+"\n"
            labels_bin_string += str(val).upper()+"\0"
        declare_labels += "};\n"
        labels_bin_string[:-1]
        model_bin.append_string_data(
            labels_bin_string, model_meta_information_struct[0].model_labels, add_null=False)
    file_content += declare_labels

    model_descriptor_declaration = "\n\n\nModelInputDescription axon_model_input_description_[@lower_model_name][[@UPPER_MODEL_NAME]_LAYER_COUNT] = {\n[@MODEL_DESCRIPTION]\n};"
    file_content += model_descriptor_declaration
    file_content = file_content.replace(
        "[@MODEL_DESCRIPTION]", model_descriptor_content)
    file_content = file_content.replace(
        "[@UPPER_MODEL_NAME]", model_name.upper())
    file_content = file_content.replace(
        "[@lower_model_name]", model_name.lower())
    file_content = file_content.replace("[@CamelModelName]", camel_case_name)

    model_bin.append_string_data(
        model_name.lower(), model_meta_information_struct[0].model_name)
    model_meta_info_bin = mw.get_bytearray_from_struct(
        model_wrapper_ffi, model_meta_information_struct)
    model_compilation_options_bin = mw.get_bytearray_from_struct(
        model_wrapper_ffi, model_compilation_options_struct)

    model_bin.append_bin_data(
        model_meta_info_bin, model_desc_info_hdr_struct[0].meta)
    model_bin.append_bin_data(model_descriptor_bin,
                              model_desc_info_hdr_struct[0].layers)
    model_bin.append_bin_data(
        model_const_bin, model_desc_info_hdr_struct[0].consts)
    model_bin.append_bin_data(
        model_compilation_options_bin, model_desc_info_hdr_struct[0].compilation_option)
    model_bin.append_model_title(model_desc_info_hdr_struct[0].title)
    model_bin.append_model_version(model_desc_info_hdr_struct[0].version)

    model_desc_info_hdr_bin = mw.get_bytearray_from_struct(
        model_wrapper_ffi, model_desc_info_hdr_struct)
    final_model_bin = model_bin.get_model_bin(model_desc_info_hdr_bin)

    logger.debug(
        f"maximum scaling error {max_scaling_error:.5f}% from layer : {max_error_layer_name}")
    if (psum_needed):
        logger.debug(f"maximum PSUM layer is at {max_psum_layer_name}")
    if (normalized_scaleshift):
        logger.debug(
            f"maximum scaling error due to a normalized scaleshift is {max_single_scaling_error:.5f}% at layer {max_single_error_layer_name} from channel {max_single_error_ch}")
    if (test_flag) and labels is not None:
        logger.info(f"true Classification : {labels[y_test[test_ndx]]}")
        logger.info(f"tflite Model Prediction : {labels[digit]}")
    logger.debug(f"number of ops in the model {total_layer_count}")
    logger.debug(
        f"{model_input_operator_name} at axon layer number {model_input_axon_layer_num} [@{model_input_tf_index}] is the axon input to the model")
    logger.debug(
        f"{model_output_operator_name} at axon layer number {model_output_axon_layer_num} [@{model_output_tf_index}] is the axon output of the model")

    # logger.debug(f"the total command buffer length is {str(np.int32(command_buff_len))}") #REMOVE
    return log_content, model_const_bin, model_descriptor_bin, final_model_bin, file_content, op_radix


def run_test_app(use_exe=False):
    print("NOT IMPLEMENTED")


def run_c_compiler_lib(compiler_types_hdr_path, axon_compiler_lib, arguments_string_array, compiler_stdout_filepath=COMPILER_STDOUT_FILE_NAME, test_vectors_flag=False):
    # default error code when an exception occurs when calling the compiler library
    compiler_return_code = -903
    compiler_return_dict = {}
    compiler_return_code_text = "Default Error Code, Exception occured when calling the compiler library!"

    test_array_memory = c_char_p*len(arguments_string_array)
    test_array = test_array_memory()
    for i in range(len(arguments_string_array)):
        test_array[i] = arguments_string_array[i].encode("utf-8")

    arguments_string_array_bin = [
        x.encode("utf") for x in arguments_string_array]
    ip_ptr = (c_char_p * len(arguments_string_array_bin)
              )(*arguments_string_array_bin)

    compiler_return_object = mw.CompilerResultsReturnClass(
        compiler_types_hdr_path)
    compiler_return_struct_memory = c_uint32 * \
        compiler_return_object.get_compiler_return_buffer_size()
    compiler_return_struct_ptr = compiler_return_object.get_compiler_return_struct_np_buffer(
    ).ctypes.data_as(POINTER(compiler_return_struct_memory))

    # axon_compiler_lib.CompilerLibMain.argtypes = [c_int, POINTER(
    #     test_array_memory), POINTER(compiler_return_struct_memory)]
    # axon_compiler_lib.CompilerLibMain.restype = c_int

    # scan/compile function lib call setup
    axon_compiler_lib.nrf_axon_compile_model.argtypes = [c_int, POINTER(
        test_array_memory), POINTER(compiler_return_struct_memory)]
    axon_compiler_lib.nrf_axon_compile_model.restype = c_int

    # infer function lib call set up
    axon_compiler_lib.nrf_axon_infer_test_vectors.argtypes = [c_int, POINTER(
        test_array_memory), POINTER(compiler_return_struct_memory)]
    axon_compiler_lib.nrf_axon_infer_test_vectors.restype = c_int

    try:
        if os.name == 'posix' and LOG_COMPILER_STDOUT_IN_LINUX:
            """
            using stdout redirect using contextmanager
            """
            with open(compiler_stdout_filepath, 'w') as f:
                with util.stdout_redirector(f):
                    compiler_return_code = axon_compiler_lib.CompilerLibMain(
                        len(ip_ptr), pointer(test_array), compiler_return_struct_ptr)
            # reading the temp file generated to put the compiler lib stdout into the log file
            with open(compiler_stdout_filepath, 'r') as f:
                for line in f:
                    logger.info(line.strip())
            if REMOVE_COMPILER_LIB_STDOUT_FILE and compiler_stdout_filepath == COMPILER_STDOUT_FILE_NAME:
                os.remove(compiler_stdout_filepath)
        else:
            """
            for windows the stdout redirection is not working and is not populating the temp file descriptor, 
            piping the output to a seperate log file from the command line seems like the best way to get the compiler lib stdouts for windows
            """
            # single call for compiling and running inference
            # compiler_return_code = axon_compiler_lib.CompilerLibMain(
            #     len(ip_ptr), pointer(test_array), compiler_return_struct_ptr)

            compiler_return_code = axon_compiler_lib.nrf_axon_compile_model(
                len(ip_ptr), pointer(test_array), compiler_return_struct_ptr)
            if compiler_return_code == 0:
                logger.info("model compiled successfully ....")
                if test_vectors_flag:
                    logger.info(
                        "proceeding to run inference on test vectors ....")
                    compiler_return_code = axon_compiler_lib.nrf_axon_infer_test_vectors(
                        len(ip_ptr), pointer(test_array), compiler_return_struct_ptr)
                else:
                    logger.info(
                        "not running inference as test vectors are not provided")

            """
            code that should work for windows as well but doesn't work 
            due to some differences in how the visual studio and mingw GCC compiles handles the stdouts

            with open(COMPILER_STDOUT_FILE_NAME, 'w') as f:
                with util.stdout_redirector(f):   
                compiler_return_code = axon_compiler_lib.CompilerLibMain(len(ip_ptr),pointer(test_array), compiler_return_struct_ptr)               
            #reading the temp file generated to put the compiler lib stdout into the log file
            with open(COMPILER_STDOUT_FILE_NAME, 'r') as f: 
                for line in f:
                logger.info(line.strip())
            if REMOVE_COMPILER_LIB_STDOUT_FILE:          
                os.remove(COMPILER_STDOUT_FILE_NAME)
            """
        compiler_return_dict = compiler_return_object.get_compiler_return_dict()
        compiler_return_code_text = compiler_return_object.get_compiler_return_code_as_text(
            compiler_return_code)
        # print(compiler_return_dict)
    except Exception as e:
        logger.critical(
            f"compiler threw {e}, returned {compiler_return_code}({compiler_return_code_text}), try running it again!")
    return compiler_return_code, compiler_return_dict, compiler_return_code_text


def read_pipe(pipe_read):
    with os.fdopen(pipe_read) as pipe:
        for line in pipe:
            logger.info(line.strip())


def run_c_compiler_lib_x64(compiler_types_hdr_path, axon_compiler_lib, arguments_string_array, compiler_stdout_filepath=COMPILER_STDOUT_FILE_NAME, test_vectors_flag=False):
    ret = run_c_compiler_lib(compiler_types_hdr_path, axon_compiler_lib,
                             arguments_string_array, compiler_stdout_filepath, test_vectors_flag)
    return ret


def run_compiler_library(test_vectors_flag,
                         get_per_layer_results,
                         AXON_COMPILER_OBJECT,
                         COMPILER_TYPES_HDR_FILE,
                         intermediate_outputs_dir,
                         compiler_outputs_dir,
                         file_name_prefix,
                         bin_file_name,
                         csv_test_vectors_file_name,
                         csv_last_layer_labels_file_name,
                         csv_per_layer_results_file_name,
                         axons_compiler_lib=None,
                         COMPILER_VERBOSE=True):
    """
    runs the c compiler library after providing the right flags and paths to the library
    returns
    compiler_return_dict - values returning from the compiler like inference times, profiling ticks, memory usage to display the results as needed
    return_code - error code returned by the compiler, 0 is success, <0 is an error
    compiler_return_code_text - short texts explaining the error briefly
    """
    compiler_return_dict = None
    relative_compiler_outputs_dir = os.path.relpath(compiler_outputs_dir)
    relative_intermediate_outputs_dir = os.path.relpath(
        intermediate_outputs_dir)
    test_vectors_filename = file_name_prefix + "_test_vectors_.h"
    """call the function to initiate running the compiler object here"""
    try:
        if (axons_compiler_lib is not None):
            command_string_array = []
            command_string_array.append("-c" + str(AXON_COMPILER_OBJECT))
            command_string_array.append(
                "-b" + relative_intermediate_outputs_dir + "/" + bin_file_name)
            if test_vectors_flag and csv_test_vectors_file_name is not None:
                command_string_array.append(
                    "-v" + relative_intermediate_outputs_dir + "/"+csv_test_vectors_file_name)
                command_string_array.append(
                    "-t" + relative_compiler_outputs_dir + "/"+test_vectors_filename)
                if get_per_layer_results and csv_per_layer_results_file_name is not None:
                    command_string_array.append(
                        "-r" + relative_compiler_outputs_dir + "/"+csv_per_layer_results_file_name)
            command_string_array.append(
                "-f" + relative_compiler_outputs_dir + "/"+file_name_prefix)
            command_string_array.append(
                "-p" + relative_compiler_outputs_dir + "/"+file_name_prefix+"_layers")
            command_string_array.append(
                "-l" + relative_compiler_outputs_dir + "/"+csv_last_layer_labels_file_name)
            print(f"executing compiler object at {AXON_COMPILER_OBJECT}")
            if COMPILER_VERBOSE:
                subprocess_return_code, compiler_return_dict, compiler_return_codetext = run_c_compiler_lib_x64(
                    str(COMPILER_TYPES_HDR_FILE), axons_compiler_lib, command_string_array, test_vectors_flag=test_vectors_flag)
            else:
                # with util.HideOutput() as stdoutput:
                with util.HideOutput():
                    subprocess_return_code, compiler_return_dict, compiler_return_codetext = run_c_compiler_lib_x64(
                        str(COMPILER_TYPES_HDR_FILE), axons_compiler_lib, command_string_array, test_vectors_flag=test_vectors_flag)
                # with util.stdout_redirected(to=compiler_stdout_filepath):
                #   subprocess_return_code, compiler_return_dict, compiler_return_codetext = run_c_compiler_lib_x64(str(COMPILER_TYPES_HDR_FILE),axons_compiler_lib, command_string_array)
            if (subprocess_return_code != 0):
                raise Exception(
                    f"running compiler library returned {subprocess_return_code}")
            else:
                print(
                    f"done executing compiler object at {str(AXON_COMPILER_OBJECT)}")
        else:
            subprocess_return_code, compiler_return_codetext = - \
                906, "compiler library is 'None'"
            raise Exception(
                f"compiler library is 'None', cannot run executor, suggest checking if the shared object(lib) is present within the environment variable path AXON_COMPILER_OBJECT({AXON_COMPILER_OBJECT})")
    except Exception as e:
        print(f"Exception : '{e}', exiting....")
    return compiler_return_dict, subprocess_return_code, compiler_return_codetext


if __name__ == "__main__":
    run_test_app(use_exe=False)
