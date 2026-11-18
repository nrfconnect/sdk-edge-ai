"""
Copyright (c) 2026 Nordic Semiconductor

SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""

import ast
import os
import re
import sys
import math
import yaml
import copy
import logging
import platform
import argparse
import subprocess
import numpy as np
import tensorflow as tf
import tflite as tflite
import matplotlib.pyplot as plt

from importlib import import_module
from pathlib import Path
from utility import operator_options as ops
from contextlib import contextmanager


"""create logger object"""
logger = logging.getLogger(__name__)


def GetAxonproStrideWidth(width, bitwidth=np.int8, pad_left=0, pad_right=0, packed=0):
    bytewidth = 1
    if bitwidth == np.int8:
        bytewidth = 1
        bit_shift = 2
    elif bitwidth == np.int16:
        bytewidth = 2
        bit_shift = 1
    elif bitwidth == np.int32:
        bytewidth = 4
        bit_shift = 0
    if (packed):
        stride = (width+pad_left+pad_right)*bytewidth
    else:
        stride = (np.ceil((width+pad_left+pad_right) / (2**bit_shift))
                  ).astype(np.int8)*(2**bit_shift)
    return stride


def fill_buffer_dump_template(dump_buffer_template, buffer_name, buffer_size=0, buffer_type="int8_t", is_constant=0, is_input=0, is_output=0, is_input_vector=0, is_expected_output=0):
    flags = ""
    buffer_dump_table_entry = dump_buffer_template
    # if(buffer_type=="int32_t"):
    #   buffer_size = 4*buffer_size
    if (buffer_size == 0):
        buffer_dump_table_entry = buffer_dump_table_entry.replace(
            "[@SIZE_OF_BUFFER]", "sizeof([@NAME_OF_BUFFER])")
    else:
        buffer_dump_table_entry = buffer_dump_table_entry.replace(
            "[@SIZE_OF_BUFFER]", str(buffer_size))
    buffer_dump_table_entry = buffer_dump_table_entry.replace(
        "[@NAME_OF_BUFFER]", buffer_name)
    buffer_dump_table_entry = buffer_dump_table_entry.replace(
        "[@SIZE_OF_TYPE]", buffer_type)
    # buffer_dump_table_entry = buffer_dump_table_entry.replace("[@SIZE_OF_BUFFER]",str(buffer_size))
    if (is_constant):
        flags += "\n"
        flags += "      .flags.is_const = 1,"
    if (is_input):
        flags += "\n"
        flags += "      .flags.is_input = 1,"
        if (is_input_vector):
            flags += "\n"
            flags += "      .flags.is_input_vector = 1,"
    if (is_output):
        flags += "\n"
        flags += "      .flags.is_output = 1,"
        if (is_expected_output):
            flags += "\n"
            flags += "      .flags.is_expected_output = 1,"
    # buffer_dump_table_entry = buffer_dump_table_entry.replace("[@IS_CONSTANT]", str(is_constant))
    # buffer_dump_table_entry = buffer_dump_table_entry.replace("[@IS_INPUT]", str(is_input))
    # buffer_dump_table_entry = buffer_dump_table_entry.replace("[@IS_INPUT_VECTOR]", str(is_input_vector))
    # buffer_dump_table_entry = buffer_dump_table_entry.replace("[@IS_OUTPUT]", str(is_output))
    # buffer_dump_table_entry = buffer_dump_table_entry.replace("[@IS_EXPECTED_OUTPUT]", str(is_expected_output))

    buffer_dump_table_entry = buffer_dump_table_entry.replace(
        "[@BUFFER_FLAGS]", flags)
    return buffer_dump_table_entry


def get_camel_case_model_name(model_name: str) -> str:
    # get the CamelCaseModelName in a string here
    model_name_split = model_name.split(sep='_')
    CamelCaseModelName = ""
    for s in range(len(model_name_split)):
        if len(model_name_split[s]) > 0:
            CamelCaseModelName += model_name_split[s][0].upper() + \
                model_name_split[s][1:].lower()
    return CamelCaseModelName


def scale_error(data_, scaling_shift):
    """
    Quantize float dataset using the scale shift, and calculate the error after rounding.
    Args:
    Data_: Dataset before quantizing.
    Scaling_Shift : Number of the left shifts for quantizing float input.
    """
    data_ += 1e-32  # to avoid divide by zero errors
    scaling_shift = np.int32(scaling_shift)
    if bool(np.array(data_).any()):
        scale = np.ldexp(1.0, scaling_shift)  # Exact float64 power of 2 (4294967296.0), never overflows int32
        error = ((np.round(data_ * scale) / scale) - data_) / data_
    else:
        error = 0
    return abs(error*100)


def save_to_file(op_dir, file_name, file_content, file_type="txt"):
    file_dir = os.path.join(op_dir, file_name)
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)
    file_open_mode = "w"
    if (file_type == "bin"):
        # save file to bin
        file_open_mode = "wb"
    with open(file_dir, file_open_mode) as f:
        f.write(file_content)
        f.close()

# choosing the best scale shift giving the best precision:


def optimized_ip_scaling_shift(data_, min_, max_, bit_limit, op_zp=np.zeros(shape=[1]), zp_bit_limit=-1):
    if (zp_bit_limit == -1):
        zp_bit_limit = bit_limit
    """
    Choose the best scaling shift number based on the minimum rounding error.

    Args:
    data_ : dataset before quantizing.
    min_: minimum number of the bits for quantizing. (left shift)
    max_: maximum number of the bits for quantizing. (left shift)

    """
    op_zp_int = int(op_zp)  # Python arbitrary-precision int: never overflows
    data_f64 = np.float64(data_)
    all_ = [(-1, 0)]
    error_max, scaleshift_max = all_[-1]
    error_min, scaleshift_min = all_[-1]
    if (np.isinf(np.ldexp(data_f64, int(min_)))):
        # scaleshift will be used in cfg_round_size where it should be 0 or positive value
        for scaleshift in range(int(min_)-1, -1, -1):
            scaled_data = np.ldexp(data_f64, scaleshift)
            if not np.isinf(scaled_data):
                sc = int(np.round(scaled_data))
                zp = op_zp_int << scaleshift  # Exact bit-shift; works for any scaleshift >= 0
                # Quantized values should not be more than 24 signed bits.
                if (abs(sc) < (1 << bit_limit)) and (abs(zp) < (1 << zp_bit_limit)) and (sc != 0):
                    error = scale_error(data_f64, scaleshift)
                    all_.append((error, scaleshift))
                    # find the biggest scaleshift that will not overflow, stop here
                    scaleshift_min = scaleshift
                    scaleshift_max = scaleshift
                    error_min = error
                    error_max = error
                    break
    else:
        # for scaleshifts normalization and/or insufficient precision case,
        # need to extend the scale left shift max to 63 (limited by 6-bit in axon cfg_round_size)
        min_int = int(min_)
        max_int = int(max_)
        bit_bound = 1 << int(bit_limit)
        zp_bit_bound = 1 << int(zp_bit_limit)

        scaled_temp = np.ldexp(data_f64, max_int - 1)
        temp = 0 if scaled_temp == 0.0 else (1 if np.isinf(scaled_temp) else int(np.round(scaled_temp)))

        for scaleshift in range(min_int, max(max_int, 63)):
            scaled_data = np.ldexp(data_f64, scaleshift)
            if np.isinf(scaled_data):
                break

            sc = int(np.round(scaled_data))
            zp = op_zp_int << scaleshift

            if (abs(sc) < bit_bound) and (abs(zp) < zp_bit_bound):
                error = scale_error(data_f64, scaleshift)
                all_.append((error, scaleshift))
                scaleshift_max = scaleshift
                error_max = error
                if (temp == 0) or (scaleshift < max_int):
                    scaleshift_min = scaleshift
                    error_min = error
            else:
                break

        # reduce the scaleshift if scale_q is even
        while scaleshift_min > min_int:
            sc = int(np.round(np.ldexp(data_f64, scaleshift_min)))
            if (sc & 1) == 0 and sc != 0:
                scaleshift_min -= 1
                error = scale_error(data_f64, scaleshift_min)
                error_min = error
                all_.append((error, scaleshift_min))
            else:
                break

    return error_min, scaleshift_min, error_max, scaleshift_max


def compare_outputs(tflite_vector, calc_vector, layer_name):
    print_string = "\n===" + layer_name + "===\n"
    if (len(tflite_vector.shape) == 4):
        tflite_vector = tflite_vector.transpose(0, 3, 1, 2)
    tflite_vector = tflite_vector.squeeze()
    diff_vector = tflite_vector - calc_vector
    non_zero = np.nonzero(diff_vector)
    number_of_errors = len(non_zero[0])
    max_error_difference = max(np.abs(diff_vector.flatten()))
    if (number_of_errors != 0):
        for e in range(number_of_errors):
            if (len(non_zero) == 3):
                # print(f"non zero at [{non_zero[0][l],non_zero[1][l],non_zero[2][l]}] tflite : {tflite_vector[non_zero[0][l]][non_zero[1][l]][non_zero[2][l]]} calculated : {calc_vector[non_zero[0][l]][non_zero[1][l]][non_zero[2][l]]} difference {diff_vector[non_zero[0][l]][non_zero[1][l]][non_zero[2][l]]}\n")
                print_string += f"non zero at [{non_zero[0][e], non_zero[1][e], non_zero[2][e]}] tflite : {tflite_vector[non_zero[0][e]][non_zero[1][e]][non_zero[2][e]]} calculated : {calc_vector[non_zero[0][e]][non_zero[1][e]][non_zero[2][e]]} difference {diff_vector[non_zero[0][e]][non_zero[1][e]][non_zero[2][e]]}\n"
            elif len(non_zero) == 1:
                print_string += f"non zero at [{non_zero[0][e]}] tflite : {tflite_vector[non_zero[0][e]]} calculated : {calc_vector[non_zero[0][e]]} difference {diff_vector[non_zero[0][e]]}\n"
    else:
        # print("calculation matches!!!!")
        print_string += "calculation matches!!!!\n"
    print_string += f"total mismatch {number_of_errors}, max_diff {max_error_difference}\n"
    # print(print_string)
    return print_string


def write_float_array_to_file(array, array_name):
    header_string = f"\nfloat {array_name}[] = " + "\n{"
    # Add formatted values with 3 decimal precision to the string
    flat_array = array.flatten()
    for value in flat_array:
        header_string += f'{value:.3f},'  # 3 decimal precision
    # Close the array declaration
    header_string += '};'

    """
  # Get the shape of the array
    shape = arr.shape

    # Loop through the dimensions to build the formatted string
    for dim1 in range(shape[0]):  # First dimension (e.g., size 2)
        header_string += '    {\n'  # Start first level of nesting

        for dim2 in range(shape[1]):  # Second dimension (e.g., size 3)
            header_string += '        {\n'  # Start second level of nesting

            for dim3 in range(shape[2]):  # Third dimension (e.g., size 4)
                header_string += '            {'  # Start third level of nesting

                for dim4 in range(shape[3]):  # Fourth dimension (e.g., size 5)
                    value = arr[dim1, dim2, dim3, dim4]
                    # Format the value with 3 decimal precision
                    header_string += f' {value:.3f},' if dim4 < shape[3] - 1 else f' {value:.3f}'  # Avoid trailing comma
                header_string += ' },\n'  # End fourth level

            header_string += '        },\n'  # End third level

        header_string += '    },\n'  # End second level

    header_string += '};\n\n'  # End array definition
  """
    return header_string


def saturate_array_at_bitwidth(array, bitwidth):
    int_info = np.iinfo(bitwidth)
    min_val = int_info.min
    max_val = int_info.max
    array = (np.clip(array, a_min=min_val, a_max=max_val)).astype(bitwidth)
    return array


def write_array_to_file(array, array_name, array_bitwidth=np.int8):
    file_content = ""
    write_values_only = (array_name == "")
    array_ctype_defn = get_std_ctype_definition(array_bitwidth)
    np.printoptions(threshold=np.inf)
    array = np.round(array)
    array = saturate_array_at_bitwidth(array, array_bitwidth)
    if (len(array.shape) == 4):
        if (not write_values_only):
            file_content += "\nconst "+array_ctype_defn + " "+array_name.lower()+"[][" + \
                str(array.shape[1])+"]["+str(array.shape[2]) + \
                "]["+str(array.shape[3])+"] = {\n"
            for n in range(array.shape[0]):
                file_content += "{"
                for b in range(array.shape[1]):
                    file_content += "{"
                    for d in range(array.shape[2]):
                        # file_content+="{"
                        file_content += np.array2string(array[n][b][d], separator=',', max_line_width=1000, threshold=np.inf).replace(
                            '[', '{').replace(']', '}').replace("\n", "")
                        file_content += ","
                    file_content += "},"
                file_content += "},\n"
            file_content += "};"
    elif (len(array.shape) == 3):
        if (not write_values_only):
            file_content += "\nconst "+array_ctype_defn + " " + \
                array_name.lower()+"[]["+str(array.shape[1]) + \
                "]["+str(array.shape[2])+"] = {\n"
        for b in range(array.shape[0]):
            file_content += "{"
            for d in range(array.shape[1]):
                # file_content+="{"
                file_content += np.array2string(array[b][d], separator=',', max_line_width=1000,
                                                threshold=np.inf).replace('[', '{').replace(']', '}').replace("\n", "")
                file_content += ","
            file_content += "},"
        file_content += "};"
    elif (len(array.shape) == 2):
        if (not write_values_only):
            file_content += "\nconst "+array_ctype_defn + " " + \
                array_name.lower()+"[]["+str(array.shape[1])+"] = \n"
        file_content += "{"
        for d in range(array.shape[0]):
            file_content += np.array2string(array[d], separator=',', max_line_width=1000, formatter={
                                            'all': lambda x: str(x)}, threshold=np.inf).replace('[', '{').replace(']', '}').replace("\n", "")
            file_content += ","
        file_content += "};"
    else:
        if (len(array.shape) == 0):
            array = np.expand_dims(array, axis=0)
        if (not write_values_only):
            file_content += "\nconst "+array_ctype_defn + " " + \
                array_name.lower()+"["+str(array.shape[0])+"] = \n"
        file_content += np.array2string(array, separator=',', max_line_width=1000, formatter={
                                        'int_kind': lambda x: str(x)}, threshold=np.inf).replace('[', '{').replace(']', '}').replace("\n", "")
        file_content += ";"

    return file_content


def optimize_scaling_shift_per_channel(ip_scale, op_scale, w_scales, op_zp, bit_limit=31, max_scale=31):
    """
    get the shape of the ip, op and w scale values.
    if the weight scale is not scalar, we will be generating multiple scale_shifts and scale_q values,
    one for each of the different Ws values
    """
    scale_ = np.array([])
    scale_q_ = np.array([])
    scaleshift_ = np.array([])
    scaleshift_range_ = np.empty((0, 2), dtype=np.int8)
    scaleshift_range = []
    error_ = np.array([])
    scaleshift = 8
    scale_q = 0
    # error_arr = np.array([])
    for ch in range(w_scales.shape[0]):
        scale = ((ip_scale[0]*w_scales[ch])/op_scale[0])
        # for scaleshift in range(8, 31):
        #   scale_q = abs(int(np.round(scale*2**scaleshift)))
        #   zp_of =  abs(int(np.round(op_zp[0] * pow(2, scaleshift))))

        #   # error_arr = np.append(error_arr,error)
        #   if ((zp_of >= np.iinfo(np.int32).max) or (scale_q >= (np.iinfo(np.int32).max))):
        #     scaleshift-=1
        #     break
        if bool(scale.all()):
            op = optimized_ip_scaling_shift(
                scale, 8, max_scale, bit_limit, op_zp)
            # error = scale_error(scale,scaleshift)
            scaleshift = op[1]
            error_ = np.append(error_, op[0])
            scaleshift_range.append([op[1], op[3]])
        scale_ = np.append(scale_, scale)
        scale_q = abs(int(np.round(scale*2**scaleshift)))
        scale_q_ = np.append(scale_q_, scale_q)
        scaleshift_ = np.append(scaleshift_, scaleshift)
    scaleshift_range_ = np.array(scaleshift_range, dtype=np.int8)

    return scale_, scaleshift_.astype(np.int8), scale_q_.astype(np.int32), error_, scaleshift_range_


def find_elements_present(input_array, find_in_array):
    result_array = [False]*len(input_array)
    for ndx, element in enumerate(input_array):
        for values in find_in_array:
            result_array[ndx] = (element == values)
    return np.array(result_array)


def get_graph_info_from_ops(operation_details, supported_operators):
    logger = logging.getLogger(__name__)
    # get the input array indexes and output array indexes
    # figure out if the output of an operation is being used in more than one input
    # if yes then that is a branch/track and it connects to another operation at some place
    op_nodes = {}
    merge_nodes = {}
    split_nodes = {}
    max_track_count = -1
    branches = 0
    ip_nodes = {}
    new_graph = []
    pass_through_ops_present = False

    # for ops_support in supported_operators:
    #   if ops_support=="Passthrough":
    #     pass_through_ops_present=True
    #     break

    # if pass_through_ops_present:
    #   new_ops = copy.deepcopy(operation_details)
    #   for ndx, ops in enumerate(new_ops):
    #     if supported_operators[ndx] =="Passthrough":
    #       if (ndx-1)>=0:
    #         new_ops[ndx-1]['outputs'] = ops['outputs']
    #       # else: #the pass through operator is the first op?
    #       if (ndx+1)<len(new_graph):
    #         new_ops[ndx+1]['inputs'] = ops['inputs']
    #       # else: #the pass through operator is the last op?

    # the first loop for graph creation will throw the exception if there is any operation that we do no support or allow to be passed through
    for ndx, operations in enumerate(operation_details):
        if supported_operators[ndx] == "Supported" or supported_operators[ndx] == "Passthrough":
            # FIXME the handling of PAD here:
            if supported_operators[ndx] == "Passthrough" or operations['op_name'] == "PAD":
                pass_through_ops_present = True
            if len(operation_details) > 1:
                for ops_ndx in range(operations['index']+1, len(operation_details)):
                    _is_element_present = False
                    # element_present = find_elements_present(operations['outputs'],operation_details[ops_ndx]['inputs'])
                    if (len(operations['outputs']) <= len(operation_details[ops_ndx]['inputs'])):
                        _is_element_present = (
                            operations['outputs'] in operation_details[ops_ndx]['inputs'])
                    else:
                        _is_element_present = (
                            operation_details[ops_ndx]['inputs'] in operations['outputs'])
                    if (_is_element_present):
                        try:
                            op_nodes[operations['index']].append(ops_ndx)
                        except KeyError:
                            op_nodes[operations['index']] = [ops_ndx]
            else:
                # model with just one operation?
                op_nodes[operations['index']] = [ndx]
        else:
            raise Exception(
                f"{operations['op_name']} operator is not supported!")

    for operations in operation_details:
        for ops_ndx in range(operations['index'], -1, -1):
            _is_element_present = False
            if (len(operations['outputs']) <= len(operation_details[ops_ndx]['inputs'])):
                _is_element_present = operations['inputs'] in operation_details[ops_ndx]['outputs']
            else:
                _is_element_present = operation_details[ops_ndx]['outputs'] in operations['inputs']
            if (_is_element_present):
                try:
                    ip_nodes[operations['index']].append(ops_ndx)
                except KeyError:
                    ip_nodes[operations['index']] = [ops_ndx]
        if ((np.isin(operations['inputs'], operation_details[0]['inputs'])).any()):
            try:
                ip_nodes[operations['index']].append(ops_ndx)
            except KeyError:
                ip_nodes[operations['index']] = [ops_ndx]

    for ops_ndx in op_nodes.keys():
        if (len(op_nodes[ops_ndx]) > 1):
            branches += 1
            for ops in op_nodes[ops_ndx]:
                try:
                    split_nodes[ops_ndx].append(ops)
                except KeyError:
                    split_nodes[ops_ndx] = [ops]
        ops_connects = [(str(operation_details[ops_ndx]['op_name']) +
                         "_"+str(ops_ndx)) for ops_ndx in op_nodes[ops_ndx]]
        logger.debug(
            f"{operation_details[ops_ndx]['op_name']}_{ops_ndx} connects to {ops_connects}")

    for ops_ndx in ip_nodes.keys():
        if (len(ip_nodes[ops_ndx]) > 1):
            merges = [(str(operation_details[ops_ndx]['op_name']) +
                       "_"+str(ops_ndx)) for ops_ndx in ip_nodes[ops_ndx]]
            logger.debug(
                f"{operation_details[ops_ndx]['op_name']}_{ops_ndx} merges {merges}")
            for ops in ip_nodes[ops_ndx]:
                try:
                    merge_nodes[ops_ndx].append(ops)
                except KeyError:
                    merge_nodes[ops_ndx] = [ops]

    # new logic for getting a full node based input and output node graph using the graph and ip_nodes alone
    # making a deepcopy as operators are referenced and we need to modify them as part of generating the new graph
    ops = copy.deepcopy(operation_details)
    # pass_through_ops_count = 0
    axon_layer_num = -1
    for ndx, operations in enumerate(ops):
        new_graph.append(operations)
        new_graph[ndx]['ops_support'] = supported_operators[ndx]
        try:
            # find out if the node has an input from the first tensor or zero tensor
            new_graph[ndx]['ip_tensors'] = new_graph[ndx]['inputs']
            if (operations['index'] == 0 and (0 in operation_details[ndx]['inputs'])):
                new_graph[ndx]['inputs'] = np.array([-1])
            else:
                new_graph[ndx]['inputs'] = ip_nodes[operations['index']]
        except KeyError:
            new_graph[ndx]['inputs'] = []
        try:
            new_graph[ndx]['op_tensors'] = new_graph[ndx]['outputs']
            new_graph[ndx]['outputs'] = op_nodes[operations['index']]
        except KeyError:
            new_graph[ndx]['outputs'] = []
        if new_graph[ndx]['ops_support'] == "Supported":
            axon_layer_num += 1
            new_graph[ndx]['axon_layer_num'] = axon_layer_num
        elif new_graph[ndx]['ops_support'] == "Passthrough":
            # indicates not supported by axon, and is a no op
            new_graph[ndx]['axon_layer_num'] = -1
        new_graph[ndx]['axon_ip_ops'] = new_graph[ndx]['inputs']
        new_graph[ndx]['axon_op_ops'] = new_graph[ndx]['outputs']

    # if pass_through_ops_present:
    #   for ndx, ops in enumerate(new_graph):
    #     if ops['ops_support']=="Passthrough":
    #       if (ndx-1)>=0:
    #         new_graph[ndx-1]['outputs'] = ops['outputs']
    #       # else: #the pass through operator is the first op?
    #       if (ndx+1)<len(new_graph):
    #         new_graph[ndx+1]['inputs'] = ops['inputs']
    #       # else: #the pass through operator is the last op?

    if pass_through_ops_present:
        for ndx, ops in enumerate(new_graph):
            # FIXME the handling of PAD here
            if ops['ops_support'] == "Passthrough" or ops['op_name'] == "PAD":
                for op in new_graph[ndx+1:]:
                    # reduce the index of the op accordingly
                    op['index'] -= 1
                    op['axon_ip_ops'] = [i - 1 if i >= new_graph[ndx]
                                         ['index'] else i for i in op['axon_ip_ops']]
                    op['axon_op_ops'] = [i - 1 if i >= new_graph[ndx]
                                         ['index'] else i for i in op['axon_op_ops']]

    # #code here for detecting if there are more than two tracks in the model, and raise an exception
    """ 031225
  getting rid of this code as we have removed the concept of tracks and buffer management based on that

  # split_node_keys = list(split_nodes.keys())
  # merge_node_keys = list(merge_nodes.keys())
  # for ndx,split_node in enumerate(split_node_keys[0:-1]):
  #   # logger.debug(f"split node {split_node_keys[ndx]}, next split :  {split_node_keys[ndx+1]}")
  #   logger.debug(f"split node:{split_node_keys[ndx]} <= merge node:{merge_node_keys[ndx]} <= next split node:{split_node_keys[ndx+1]}")
  #   #check if there is a merge node in between the two split nodes here
  #   #and if not raise an exception
  #   if not (any(split_node_keys[ndx] < x <= split_node_keys[ndx+1] for x in merge_node_keys )):
  #     raise Exception(f"a split node has been observed between two consecutive split nodes [{split_node_keys[ndx]}:{split_node_keys[ndx+1]}] without a merge node in between!")

  # tracks_count=1
  # track_start=-1
  # track_end=-1
  # #FIXME add check in here for the output being present in multiple tracks
  # for ops_ndx in op_nodes.keys():
  #   if(len(op_nodes[ops_ndx])>1):
  #     tracks_count+=(len(op_nodes[ops_ndx])-1)
  #     track_start = ops_ndx
  #   if(ops_ndx!=0):#not the input node
  #     if(len(ip_nodes[ops_ndx])>1):
  #       track_end = ops_ndx
  #       tracks_count-=(len(ip_nodes[ops_ndx])-1)
    
  #   max_track_count = max(max_track_count,tracks_count)

  # assert max_track_count <=2 , "Cannot handle models with more than two tracks"
  # # print(f"track started at op {track_start}, track ended at op {track_end}, track count {tracks_count}, max_track_count {max_track_count}")
  # # # print(op_nodes)
  # # # print(ip_nodes)

  # possiblePaths = [[]]

  # def depthFirstSearch(graph, currentNode, visited):
  #     visited.append(currentNode)
  #     try:
  #       for node in graph[currentNode]:
  #           if node not in visited:
  #               depthFirstSearch(graph, node, visited.copy())        
  #     except KeyError:
  #       possiblePaths.append(visited)
  #       return
      

  # depthFirstSearch(graph, 0, [])

  # tracks = [ [] for t in range(max_track_count)]
  # for node in op_nodes.keys():
  #   track_zero=0
  #   for path in possiblePaths[1:]:
  #     if node not in path:
  #       track_zero=0
  #       break
  #     track_zero = 1
  #   if(track_zero):
  #     tracks[0].append(node)
  #     #find the nodes present in all the paths, those are the nodes which are track zero because they will be the nodes which either split or merge the graph
  #     # print(path)

  # # print(tracks[0])

  # tracks[0].append(len(op_nodes))

  # if(max_track_count>1):
  #   #now for track one, all the nodes which are present in the first path are track one elements, because they occured first in the ops list
  #   for node in possiblePaths[1]:
  #     if(node not in tracks[0]):
  #       tracks[1].append(node)

  #   # print(tracks[1])

  #   # add the remaining nodes into the track zero, from the last possible path, because that has nodes not present in our selected track1
  #   for node in possiblePaths[-1]:
  #     if node not in tracks[0]:
  #       tracks[0].append(node)
  #   tracks[1].sort()
    
  # track_cnt=0
  # tracks[0].sort()
  # for track in tracks:
  #   # print(f'track_{track_cnt} :{track}')
  #   logger.debug(f'track_{track_cnt} :{track}')
  #   track_cnt+=1

  # track_info = {}

  # track_info[0] = [0,0,0]
  # for node in ip_nodes.keys():
  #   ip_track_no, ops_track_no, merge_track_no = 0,0,0
  #   if(max_track_count>1):
  #     if node in tracks[1]:
  #       ops_track_no = 1  
  #     if node!=0:
  #       if(len(ip_nodes[node])>1):# found a merge node
  #         ip_track_no = 0
  #         merge_track_no = 1     
  #       else:
  #         ip_node_list = list(ip_nodes[node])
  #         if(ip_node_list[0] in tracks[1]):
  #           ip_track_no = 1
  #   track_info[node] = [ops_track_no,ip_track_no,merge_track_no]
  """

    nodes = {'op_nodes': op_nodes, 'ip_nodes': ip_nodes,
             'split_nodes': split_nodes, 'merge_nodes': merge_nodes, 'op_graph': new_graph}
    return nodes, max_track_count


def run_inference_using_vectors(test_vectors, interpreter):
    prediction_output = []
    output_vectors = []
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    # input_shape = inputs[0]['shape']
    # output_shape = outputs[0]['shape']
    # test_output_vectors = -1*np.ones((output_shape))
    for test_ in test_vectors:
        test_ = np.expand_dims(test_, axis=0)
        interpreter.set_tensor(inputs[0]['index'], test_)
        # Run inference.
        interpreter.invoke()
        # Save the class predictions for all test samples.
        output = interpreter.tensor(outputs[0]['index'])
        digit = np.argmax(output()[0])
        prediction_output.append(digit)
        op_vector = output()[0]
        output_vectors.append(op_vector)

    return prediction_output


def get_size_from_tensor_shape(shape, bitwidth=np.int8):
    size = 1
    bytewidth = 1
    if bitwidth == np.int8:
        bytewidth = 1
    elif bitwidth == np.int16:
        bytewidth = 2
    elif bitwidth == np.int32:
        bytewidth = 4

    for dimensions in shape:
        size *= dimensions
    return size*bytewidth


def get_std_ctype_definition(bitwidth=np.int8):
    type_defn = "int8_t"
    if bitwidth == np.int16:
        type_defn = "int16_t"
    elif bitwidth == np.int32:
        type_defn = "int32_t"
    elif bitwidth == np.uint8:
        type_defn = "uint8_t"
    elif bitwidth == np.uint16:
        type_defn = "uint16_t"
    elif bitwidth == np.uint32:
        type_defn = "uint32_t"

    return type_defn


def excel_ceil(x, s):
    return s * math.ceil(float(x)/s)


def excel_floor(x, s):
    return s * math.floor(float(x)/s)


def get_array_from_tensor(input_tensor, transpose_flag=True, op_code=None):
    if op_code:  # checks if the op code of the tensor is in the 'no transpose list' and disable the transpose
        transpose_flag = transpose_flag and (
            op_code not in NO_TRANSPOSE_TFLITE_OPERATORS)
    if transpose_flag:
        if (len(input_tensor.shape) == 4):
            input_tensor = input_tensor.transpose(0, 3, 1, 2)
        elif (len(input_tensor.shape) == 3):
            input_tensor = input_tensor.transpose(2, 0, 1)
    input_tensor = input_tensor.squeeze()
    op = []
    if (len(input_tensor.shape) == 3):
        for d in input_tensor:
            height = []
            for h in d:
                width = [w for w in h]
                height.append(width)
            op.append(height)
    elif (len(input_tensor.shape) == 2):
        op = [x for x in input_tensor.squeeze()]
    else:
        op = input_tensor

    return np.array(op)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def float_to_int32_x86(arr: np.ndarray) -> np.ndarray:
    """Converts a float array to int32, matching x86_64 floating-point overflow behavior

    on ARM64 (Apple Silicon).
    """
    int32_min = np.iinfo(np.int32).min  # -2147483648
    int32_max = np.iinfo(np.int32).max  #  2147483647

    # Identify out-of-bounds floating-point elements (overflow/underflow/NaN)
    out_of_bounds = (arr > int32_max) | (arr < int32_min) | np.isnan(arr)

    # Cast valid values normally, force overflow elements to -2147483648
    result = np.where(out_of_bounds, int32_min, arr).astype(np.int32)
    return result


def getArgDict(arg):
    args = {}
    if isinstance(arg, str):
        arg = arg[1:-1]  # removing the curly braces of the dictionary
        if arg != '':
            cpu_codes = arg.split(',')
            for cpu_code in cpu_codes:
                code_enum = cpu_code.split(':')
                args[int(code_enum[0])] = int(code_enum[1])
    else:
        args = arg
    return args


def parse_precompiler_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_data_filename',
        type=str,
        default="",
        help="""\
    Qualified file name of the train dataset in floating point (Directory + file name). Should be in ".npy" format of size (#train_samples, x_axis, y_axis). If dataset is in 1D, shape should be (#test_samples, 1 ,x_axis).
    """)

    parser.add_argument(
        '--test_data_filename',
        type=str,
        # "data\\ei_fomo_face_detection\\x_test_ei_fomo_fd.npy",#"data\\ei_fomo_face_detection\\x_test_ei_fomo_fd.npy",#"data\\anomaly_detection\\y_test_ad.npy",#"training/data/x_test16_49.npy",#"training/data/x_test_vww.npy",#"training/data/x_test16_49.npy",# 'training/data/x_test'+str(test_type)+'.npy',#"data/kws_g12/x_test61.npy",#
        default="",
        help="""\
      Qualified file name of the test dataset in float (Directory + file name). Should be in ".npy" format of size (#test_samples, x_axis, y_axis). If dataset is in 1D, shape should be (#test_samples, 1 ,x_axis).
    """)

    parser.add_argument(
        '--test_labels',
        type=str,
        # ,"data\\anomaly_detection\\y_test_ad.npy",#"training\data\y_test16_49.npy",#"training\data\y_test_vww.npy",#"training\data\y_test16_49.npy",# 'training/data/y_test'+str(test_type)+'.npy',#"data\kws_g12\y_test.npy",#
        default="",
        help="""\
    Qualified file name of test dataset's labels (Directory + file name). Should be in ".npy" format. It's a numpy array of size (#test_samples), includings integer values showing each class.
    """)

    parser.add_argument(
        '--model',
        type=str,
        default="",
        help="""\
    Qualified file name of the keras model. This can be ".h5" file or saved model (A filder including the "pb" file and variables).
    """)

    parser.add_argument(
        '--labels_order',
        type=str,
        default="",  # ["non_person", "person"],#['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],#["Down", "Go", "Left", "No", "Off", "On", "Right", "Stop", "Up", "Yes", "Silence", "Unknown"],#["non_person", "person"],#["Down", "Go", "Left", "No", "Off", "On", "Right", "Stop", "Up", "Yes", "Silence", "Unknown"],#["silence", "unknown", "yes", "no" ,"up", "down" ,"left" ,"right" ,"on", "off" ,"stop","go"],##["person", "non_person"],#
        nargs="+",
        help="""\
      Shows what each number in labels represents. This is saved by order.
      """)

    parser.add_argument(
        '--model_name',
        type=str,
        # "EI_FOMO_FD",#'TINYML_KWS',#"VWW",#DSCNN_KWS_REF_TEST",#'TINYML_KWS',#"FC4_TEST",# 'TINYML_KWS',# +str(test_type),
        default="",
        help="""\
    The abbrevation of the model name saved in the precompiler.
    """)

    parser.add_argument(
        '--tflite_filename',
        type=str,
        # "models\\edge_impulse\\fomo\\ei_fomo_face_detection_q.lite",#"models\\tinyml\\anomaly_detection\\ad01_int8.tflite",#'training/model/kws_'+str(test_type)+".tflite",#"training/model/vww_.tflite",# 'training/model/kws_'+str(test_type)+".tflite",#"models/fc4/FC4_model_n.tflite", #"models/pretrainedResnet_quant.tflite",#'models\pretrainedResnet_quant.tflite',#
        default="",
        help="""\
    Fully qualified file name to save the created TFlite file.
    """)

    parser.add_argument(
        '--test_quant_data_filename',
        type=str,
        # 'training/data/qdata_'+str(test_type)+".npy", #"training/data/qdata_vww.npy",# 'training/data/qdata_'+str(test_type)+".npy", #"training/data/qdata_fc4_test.npy",#
        default="",
        help="""\
    Qualified file name (Directory+file name) to save the quantized test dataset.
    """)

    parser.add_argument(
        '--output_dir',
        type=str,
        # 'training/precompiles/kws_test/',#'training/precompiles/vww_test/',#"training/precompiles/fc4_test/",#
        default="precompiles/test/",
        help="""\
      Folder name to save the precompiler output files to.
      """)

    parser.add_argument(
        '--model_type',
        type=str,
        default='',
        help="""\
      There are 2 models the user can choose between:
      1.KWS (Keyword Spotting). Choosing this model, user should also define the argument "Audio_feature_type".
      2.ACC (Accelerometer). Choosing this model, user should also define the argument "bit_axis".
      """)

    parser.add_argument(
        '--audio_feature_type',
        type=str,
        default='MfccOrtho',
        help="""\
      Type of the audio feature when Model_Type is KWS.
      There are 4 audio features to select from :
      A hamming window is applied to all audio features before the spectogram calculation:

      Mel32: Applying 32 mel filterbanks to the audio power spectrogram.
      MfccOrtho : Using power spectogram in MFCC calculation.
      EnergyAppend : Replace first cepstral coefficient with log of frame energy (sum of the power spectogram in each frame) in MFCC calculation.
      MfccFftMagOrtho : Using magnitude of the spectogram in MFCC calculation.
      """)

    parser.add_argument(
        '--template_dir',
        type=str,
        default="template_files/",
        help="""\
      Folder name to get the respective template files from.
      """)

    parser.add_argument(
        '--get_test_vectors',
        type=int,
        nargs="*",
        default=[0, 1, 2, 3],
        # action='append',
        help="""\
      used as a flag to get test vectors for the operations for the ndx in the list from the test dataset
      """)

    parser.add_argument(
        '--get_simulator_testfiles',
        type=str2bool,
        default=False,
        help="""\
       Flag to get the text file of input test vectors and labels to run through the simulator
      """)

    parser.add_argument(
        '--normalize_scaleshift',
        type=str2bool,
        default=True,
        help="""\
       Use this flag to get a single scaleshift value for all the channels in an operation
      """)

    parser.add_argument(
        '--skip_softmax_op',
        type=str2bool,
        default=True,
        help="""\
       Use this flag to skip the softmax operation
      """)

    parser.add_argument(
        '--disable_op_quantization',
        type=str2bool,
        default=False,
        help="""\
       Use this flag to disable the output quantization
      """)

    parser.add_argument(
        '--op_radix',
        type=int,
        default=0,
        help="""\
       Use this flag to set the output radix to be used when disabling the output quantization
      """)

    parser.add_argument(
        '--cpu_op_codes_list',
        type=getArgDict,
        # nargs="*",
        # {tflite.BuiltinOperator.SOFTMAX:100,tflite.BuiltinOperator.SQUARE:103,tflite.BuiltinOperator.RESHAPE:7},#[tflite.BuiltinOperator.RESHAPE],#[tflite.BuiltinOperator.RESHAPE,tflite.BuiltinOperator.SQUARE,tflite.BuiltinOperator.SOFTMAX],
        default={},
        help="""\
       Supply the operator codes of the operations to be executed in CPU by the end user with the axon operation enum seperated by a :
       Example: {25:100,92:101}
      """)

    parser.add_argument(
        '--compiler_api_path',
        type=str,
        help="""\
      Pass the file path of the axonpro_compiler_api.h 
      used to get the model layer descriptions and the definitions of different structures to generate the binary file
    """
    )

    return parser


def get_precompiler_debug_arguments():
    # load the first cmd in the batch script with params
    logger = logging.getLogger(__name__)
    path = r"batch_test_precompiler.bat"
    makefile = open(path, 'r').read()
    command_line_arguments = makefile.split('\n')
    for cmd in command_line_arguments:
        if (cmd == ''):
            continue
        # get rid of extra spaces if present in the command line
        cmd = cmd.replace("  ", " ")
        args = cmd.split(' ')
        if args[0] != "python":
            # logger.debug("no command line argument found")
            continue

        pre_compiler_parser = parse_precompiler_arguments()
        parsed_arg = pre_compiler_parser.parse_known_args(args[2:])[0]
        # print(f"running {args[1]} for {parsed_arg.model_name}")
        logger.info(f"running {args[1]} for {parsed_arg.model_name}")
        return parsed_arg


def debugger_is_active() -> bool:
    """Return true if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def write_array_to_bin(array, bin_file, array_bitwidth=np.int8):
    # tensor_header = np.array([124,78 ,69 ,87 ,32 ,86 ,69 ,67 ,84 ,79 ,82 ,124],np.int8)
    # bin_file.extend(bytearray(tensor_header))
    # test_tensor = np.ones((7,5), dtype=np.int8)
    offset = len(bin_file)
    bin_file.extend(array.tobytes())

    # add padding for the arrays to ensure that the next tensor start at 4 byte boundaries
    if (array.nbytes % 4):
        padding_needed = int(np.ceil(array.nbytes/4)*4) - array.nbytes
        bin_file.extend(bytearray(np.zeros(padding_needed, dtype=np.int8)))
    return bin_file, offset


def get_length_from_shape(tensor_shape, pad_details=0, bitwidth=np.int8):
    # check if the width[@ index 2] of the tensor is multiple of 8 as we need to keep the
    bytewidth = 1
    if bitwidth == np.int8:
        bytewidth = 1
    elif bitwidth == np.int16:
        bytewidth = 2
    elif bitwidth == np.int32:
        bytewidth = 4
    # update the tensor dimension based on the bitwidth
    tensor_shape = np.array(tensor_shape)*bytewidth

    width = int(np.ceil(tensor_shape[2]/4)*4)
    if pad_details == 0:
        return tensor_shape[3]*tensor_shape[1]*width
    return tensor_shape[3]*(tensor_shape[1]+pad_details.pad_bottom+pad_details.pad_top)*(width)


def util_log(text, print=True, logger=None):
    if (print):
        print(text)
    if (logger is not None):
        logger(text)


def get_csv_text(text_vectors, add_newline=True):
    text_vectors = text_vectors.replace('{', '')
    text_vectors = text_vectors.replace('}', '')
    text_vectors = text_vectors.replace(',,;', '')
    text_vectors = text_vectors.replace(',;', '')
    text_vectors = text_vectors.replace(';', '')
    text_vectors = text_vectors.replace(',,', ',')
    if (add_newline):
        text_vectors += "\n"
    return text_vectors


def run_win_app(simulator_exe_path, test_input_path, expected_output_path, verbose=False, simulator_exe_name="axonpro_app.exe"):
    # file_dir = os.path.dirname(os.path.realpath(__file__))
    command = simulator_exe_path + simulator_exe_name + \
        " " + test_input_path + " " + expected_output_path
    p = subprocess.call(command, cwd=simulator_exe_path,
                        stderr=subprocess.DEVNULL)
    if (verbose):
        print(f"app returned {p}")
    calculated_output_path = simulator_exe_path
    return calculated_output_path


def parse_compiler_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--tflite_model',
        type=str,
        help="""\
    path + file name to the tflite file of the model. If tflite file is not there provide train data and floating point model.
    """)
    parser.add_argument(
        '--model_name',
        type=str,
        help="""\
    The name of the model.
    """)

    """
  Model Accuracy Params
  """
    parser.add_argument(
        '--float_model',
        type=str,
        default="",
        help="""\
    file name of the keras model.
    """)
    parser.add_argument(
        '--train_data',
        type=str,
        default="",
        help="""\
    file name of the train dataset in floating point (Directory + file name).
    """)
    """
  Test Vectors Params
  """
    parser.add_argument(
        '--test_data',
        type=str,
        default="",
        help="""\
    file name of the test dataset in float (Directory + file name).
    """)
    parser.add_argument(
        '--test_labels',
        type=str,
        default="",
        help="""\
    file name of test dataset's labels (Directory + file name). Should be in ".npy" format. It's a numpy array of size (#test_samples), includings integer values showing each class.
    """)
    parser.add_argument(
        '--test_vectors',
        type=int,
        nargs="*",
        default=[0, 1, 2, 3],
        help="""\
    used as a flag to get test vectors for the operations for the ndx in the list from the test dataset
    """)
    parser.add_argument(
        '--header_file_test_vector_cnt',
        type=int,
        default=0,
        help="""\       
    Number of test vectors in the test vectors header file
    """)
    parser.add_argument(
        '--get_quantized_data',
        type=str2bool,
        default=False,
        help="""\
    flag to save the quantized test dataset, is saved as {model_name}_q_data.npy in the output directory
    """)
    parser.add_argument(
        '--log_level',
        type=str,
        default="info",
        help="""\
    e.g debug, info, warn, error, critical
    """)
    """
  Buffer Sizes Params
  """
    parser.add_argument(
        '--interlayer_buffer_size',
        type=str,
        default=1200000,
        help="""\
    The interlayer buffer size used by the compiler to size the input buffers
    """)
    parser.add_argument(
        '--psum_buffer_size',
        type=str,
        default=180000,
        help="""\
    The psum buffer size used by the compiler
    """)
    parser.add_argument(
        '--precision_threshold',
        type=float,
        default=0.0,
        help="""\       
    A floating point value between 0 and 1, which is the minimum threshold below which a label will be classified as inconclusive, default value is 0
    """)
    parser.add_argument(
        '--precision_margin',
        type=int,
        default=0,
        help="""\       
    The margin between the highest and the 2nd highest which will be used to get a sense of confidence on the classified label. If the difference is less than this value, the label will be classified as inconclusive. 
    """)
    """
  Optimization Level Params
  """
    parser.add_argument(
        '--normalize_scaleshift',
        type=str2bool,
        default=True,
        help="""\
    Use this flag to get a single scaleshift value for all the channels in an operation
    """)
    parser.add_argument(
        '--conv_2d_setting',
        type=str,
        default="local_psum",
        nargs="+",
        help="""\
    User input to set the different modes in which a convolution 2d can be performed. Changing this setting has memory and performance implications.
    E.g local_psum, local_psum, inner, outer
    """)
    parser.add_argument(
        '--psum_buffer_placement',
        type=str,
        default="interlayer_buffer",
        nargs="+",
        help="""\
    User input to select the memory region in which the psum buffer will be placed if they are going to be used by the model.
    E.g interlayer_buffer, dedicated_memory, interlayer_buffer
    """)
    """
  Classification Labels
  """
    parser.add_argument(
        '--classification_labels',
        type=str,
        default="",  # ["non_person", "person"],#['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],#["Down", "Go", "Left", "No", "Off", "On", "Right", "Stop", "Up", "Yes", "Silence", "Unknown"],#["non_person", "person"],#["Down", "Go", "Left", "No", "Off", "On", "Right", "Stop", "Up", "Yes", "Silence", "Unknown"],#["silence", "unknown", "yes", "no" ,"up", "down" ,"left" ,"right" ,"on", "off" ,"stop","go"],##["person", "non_person"],#
        nargs="+",
        help="""\
    Shows what each number in labels represents. This is saved by order.
    """)
    parser.add_argument(
        '--test_labels_format',
        type=str,
        default="",
        nargs="+",
        help="""\
    Read documentation for a detailed info on the labels format
    E.g "edge_impulse_labels","just_labels","last_layer_vector","custom", if custom, also provide the user_handle_test_labels
    """)
    parser.add_argument(
        '--disable_op_quantization',
        type=str2bool,
        default=False,
        help="""\
    Use this flag to disable the output quantization
    """)
    parser.add_argument(
        '--op_radix',
        type=int,
        default=0,
        help="""\
    Use this flag to set the output radix to be used when disabling the output quantization
    """)
    parser.add_argument(
        '--skip_softmax_op',
        type=str2bool,
        default=True,
        help="""\
    Use this flag to skip the softmax operation
    """)
    parser.add_argument(
        '--cpu_op_codes_list',
        type=getArgDict,
        # nargs="*",
        # {tflite.BuiltinOperator.SOFTMAX:100,tflite.BuiltinOperator.SQUARE:103,tflite.BuiltinOperator.RESHAPE:7},#[tflite.BuiltinOperator.RESHAPE],#[tflite.BuiltinOperator.RESHAPE,tflite.BuiltinOperator.SQUARE,tflite.BuiltinOperator.SOFTMAX],
        default={},
        help="""\
    Supply the operator codes of the operations to be executed in CPU by the end user with the axon operation enum seperated by a :
    Example: {25:100,92:101}
    """)
    parser.add_argument(
        '--generate_sim_env_hdrs',
        type=str2bool,
        default=False,
        help="""\
    Flag to generate simulator environment header files
    """)
    parser.add_argument(
        '--user_handle_accuracy_results',
        type=str,
        default="",
        help="""\
    Add user handling function name to get the model accuracy along with module
    Example: compare_models.user_handle_output_labels
    """)
    parser.add_argument(
        '--user_handle_test_labels',
        type=str,
        default="",
        help="""\
    Add user handling function name handle the test labels as needed to match the expected format.
    Example: compare_models.user_handle_output_labels
    """)
    parser.add_argument(
        '--run_all_variants',
        type=str2bool,
        default=False,
        help="""\
    User input flag to run possible variants for an input model to get consolidated results, give the user an understanding of how the model is performing with different settings.
    """)
    parser.add_argument(
        '--reshape_input',
        type=str2bool,
        default=False,
        help="""\
    Flag to allow reshaping the test data input to match the input shape of the model if the only transformation needed on the test data is a simple reshaping of the test input
    """)
    return parser


def get_compiler_debug_arguments(path):
    if not Path(path).exists():
        raise Exception("debug file doesn't exist!")
    # load the first cmd in the batch script with params
    logger = logging.getLogger(__name__)
    if (path.endswith('yaml') or path.endswith('yml')):
        use_yaml = True

    if (use_yaml):
        compiler_input = load_yaml_file(path)
        yaml_test_list = list(compiler_input.keys())
        yaml_test_input = yaml_test_list[0]
        if (yaml_test_input == "default_values"):
            yaml_test_input = yaml_test_list[1]
        parsed_arg_dict = compiler_input[yaml_test_input]

        logger.info(
            f"running {yaml_test_list[0]} for {parsed_arg_dict['model_name']}")
    else:
        makefile = open(path, 'r').read()
        command_line_arguments = makefile.split('\n')
        for cmd in command_line_arguments:
            if (cmd == ''):
                continue
            # get rid of extra spaces if present in the command line
            cmd = cmd.replace("  ", " ")
            args = cmd.split(' ')
            if args[0] != "python":
                # logger.debug("no command line argument found")
                continue

            pre_compiler_parser = parse_precompiler_arguments()
            parsed_arg = pre_compiler_parser.parse_known_args(args[2:])[0]
            parsed_arg_dict = vars(parsed_arg)
            # print(f"running {args[1]} for {parsed_arg.model_name}")
            logger.info(f"running {args[1]} for {parsed_arg.model_name}")
    return parsed_arg_dict


def load_yaml_file(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def convert_range_to_list(range_string: list):
    # range_string = range_string.split(',')
    vectors_list = []
    for values in range_string:
        if (type(values) is str):
            # split at hyphen
            start, end = values.split("-")
            start, end = int(start), int(end)
            values = range(start, end)
            vectors_list.extend(values)
        else:
            vectors_list.append(int(values))
    return list(set(vectors_list))


def normalize_quantized_test_vectors(quantized_test_vectors):
    """Single ndarray -> [ndarray]; list/tuple of ndarrays -> list."""
    if isinstance(quantized_test_vectors, np.ndarray):
        return [quantized_test_vectors]
    if isinstance(quantized_test_vectors, (list, tuple)):
        if len(quantized_test_vectors) == 0:
            raise ValueError("quantized_test_vectors list is empty")
        return list(quantized_test_vectors)
    raise TypeError(
        f"Unsupported quantized_test_vectors type: {type(quantized_test_vectors)}"
    )


def apply_mass_inference_input_layout(q_test_vector, transpose_model_ip_flag=False, op_type=None):
    """Apply NHWC/NCHW layout transforms used by mass inference CSV."""
    q_input_shape = ops.TensorShape(np.array(q_test_vector.shape))
    q_out = q_test_vector
    if transpose_model_ip_flag:
        if q_input_shape.shape_size == 4:
            q_out = q_out.transpose(0, 2, 1, 3)
        elif q_input_shape.shape_size == 3:
            q_out = q_out.transpose(0, 2, 1)
    if op_type != 'Lstm':
        if q_input_shape.depth > 1:
            if q_input_shape.shape_size == 4:
                q_out = q_out.transpose(0, 3, 1, 2)
            elif q_input_shape.shape_size == 3:
                q_out = q_out.transpose(0, 2, 1)
    return q_out


def save_vectors_for_mass_inference(test_vector_file_name, test_vectors_ndx, quantized_test_vectors, test_labels, output_dir, transpose_model_ip_flag=False, op_type=None, save_test_labels=False, act_bits=8):
    logger = logging.getLogger(__name__)
    logger.debug(
        "getting the quantized input test vector and labels for mass inferencing using the simulator..."
    )
    q_test_vectors = normalize_quantized_test_vectors(quantized_test_vectors)
    num_inputs = len(q_test_vectors)
    num_samples = q_test_vectors[0].shape[0]
    for input_idx, q_test_vector in enumerate(q_test_vectors[1:], start=1):
        if q_test_vector.shape[0] != num_samples:
            raise ValueError(
                f"Input {input_idx} has {q_test_vector.shape[0]} samples, "
                f"expected {num_samples}"
            )
    processed_vectors = [
        apply_mass_inference_input_layout(q, transpose_model_ip_flag, op_type)
        for q in q_test_vectors
    ]

    def vector_to_csv_line(vector, act_bits):
        array_bitwidth = np.int8
        if act_bits == 16:
            array_bitwidth = np.int16
        text = write_array_to_file(vector.squeeze(), "", array_bitwidth)
        return get_csv_text(text)

    test_vector_text = ""
    test_labels_text = ""
    for vector_value in test_vectors_ndx:
        # One sample -> num_inputs CSV rows (input order = TFLite input index order)
        for input_idx in range(num_inputs):
            sample = processed_vectors[input_idx][vector_value]
            test_vector_text += vector_to_csv_line(sample, act_bits)
        if save_test_labels:
            test_labels_text += str(test_labels[vector_value]) + "\n"
    test_labels_text += "\n"
    test_vector_text = test_vector_text.replace(' ', '').replace('\r', '')
    save_to_file(output_dir, test_vector_file_name, test_vector_text)
    if save_test_labels:
        label_filename = (
            test_vector_file_name.split(".")[0]
            + "labels_."
            + test_vector_file_name.split(".")[1]
        )
        save_to_file(output_dir, label_filename, test_labels_text)
    logger.debug(
        f"saved the test vector's text files for mass inferencing to {output_dir} "
        f"({num_inputs} input(s), {len(test_vectors_ndx)} sample(s))"
    )
    return 0


def get_abs_dir_from_file(file_path):
    """
    gets the directory of a file from the path and returns the directory if it exists
    if not returns the realpath of the file
    """
    if os.path.exists(file_path):
        directory = os.path.dirname(file_path)
        return os.path.abspath(directory)
    # return os.path.relpath(file_path, start=os.path.abspath(os.sep)) + "/"


def append_user_workspace(path, user_workspace):
    """
    appends user workspace if the path provided is not an absolute path
    """
    if not (os.path.isabs(path)):
        return user_workspace + "/" + path
    return path


def print_util(text, log_string=None):
    """
    prints the text and also appends the text to a file if it is passed"""
    print(text)
    if (log_string is not None):
        text = log_string + "\n" + str(text)
    return text


def load_func(func_string):
    if ("." in func_string):
        module_, func = func_string.rsplit(".", maxsplit=1)
        m = import_module(module_)
    else:
        func = func_string
        m = import_module("axons_compiler_app")
    return getattr(m, func)


def is_windows_path(path: str | Path) -> bool:
    """Determine if the given path is in Windows format."""
    path = str(path)
    # Check for the current directory
    if path == ".":
        return True
    # Check for drive letter and backslashes
    return bool(re.match(r'^[A-Za-z]:\\', path)) or '\\' in path or ':/' in path


def get_linux_path(pure_windows_path: str | Path) -> str:
    pure_windows_path = str(pure_windows_path)
    pure_windows_path = pure_windows_path.replace("\\", '/')
    drive, directory = pure_windows_path.split(":/", 1)
    return f"/mnt/{drive.lower()}/{directory}"


def get_output_radix(op_radix, scale_shift, op_scale, op_zp, op_bw=np.int8, unquant_op_bw=np.int32):
    # determine the best op_radix here based on the output quantization values
    # the output quantization always quantizes the output to 8 bit, but that can be checked and,
    # the necessary max and min values can be calculated
    # using that information the user can determine a radix that might just be apt for the output
    if 0:
        if op_bw == np.int16:
            max_val = 2**15 - 1
            min_val = -2**15
        elif op_bw == np.int32:
            max_val = 2**31 - 1
            min_val = -2**31
        else:
            max_val = 2**7 - 1
            min_val = -2**7
        radix = 0
        layer_max_value = np.ceil(op_scale*(max_val - op_zp))
        layer_min_value = np.ceil(op_scale*(min_val - op_zp))
        # DEBUG #\n print(f"layer max and min values {layer_max_value,layer_min_value}")
        shift_max_range = scale_shift - \
            (len(bin(int(max(abs(layer_max_value), abs(layer_min_value))))) - 2)
        shift_max_range = np.int32(shift_max_range)
        if (op_radix <= 0):  # the user has not set the output radix, we need to figure out one
            op_radix = 8  # lets default that at 8 and then use the scale_shift values to maximize that accordingly
        for radix in range(op_radix, shift_max_range):
            if (layer_max_value*2**radix) > (2**shift_max_range - 1) or (layer_min_value*2**radix) < (-2**shift_max_range):
                break
    else:
        max_val_quant = np.iinfo(op_bw).max
        min_val_quant = np.iinfo(op_bw).min
        max_val_unquant = np.iinfo(unquant_op_bw).max
        min_val_unquant = np.iinfo(unquant_op_bw).min
        radix = 0
        layer_max_value = np.ceil(op_scale*(max_val_quant - op_zp))
        layer_min_value = np.ceil(op_scale*(min_val_quant - op_zp))
        bitwidth_unquant = np.iinfo(unquant_op_bw).bits
        for radix in range(min(bitwidth_unquant,scale_shift), -1, -1):
            if (layer_max_value*2**radix) <= max_val_unquant and (layer_min_value*2**radix) >= min_val_unquant:
                break

    return radix


def find_variant_settings(tflite_interpreter):
    softmax_present = False
    per_channel_scales_present = False
    conv2d_multichannel_present = False
    transpose_kernel_present = False

    # buf = bytearray(open(tflite_model_file, "rb").read())
    # model = tflite.Model.GetRootAsModel(buf, 0)
    # subgraph = model.Subgraphs(0)
    # interpreter = tf.lite.Interpreter(
    #     model_path=tflite_model_file, experimental_preserve_all_tensors=True)
    # interpreter.allocate_tensors()

    tensor_details = tflite_interpreter.get_tensor_details()
    ops_details = tflite_interpreter._get_ops_details()
    for op in ops_details:
        if (len(op['inputs']) > 1):
            per_channel_scales_present = (
                (tensor_details[op['inputs'][1]]['quantization_parameters']['scales'].size > 1) or per_channel_scales_present)
        if (op['op_name'] == "SOFTMAX"):
            softmax_present = True
        elif (op['op_name'] == "CONV_2D"):
            transpose_kernel_present = True
            # find if this is multi-channel input operation
            # print("found a 2d conv")
            ip_shape = ops.TensorShape(
                tensor_details[op['inputs'][0]]['shape'])
            if (ip_shape.depth > 1):
                conv2d_multichannel_present = True

    return softmax_present, conv2d_multichannel_present, per_channel_scales_present, transpose_kernel_present


class HideOutput(object):
    '''
    A context manager that block stdout for its scope, usage:

    with HideOutput():
        os.system('ls -l')
    '''

    def __init__(self, *args, **kw):
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        self._newstdout = os.dup(1)
        os.dup2(self._devnull, 1)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, 1)


def get_list_from_strlist(str_list):
    return ast.literal_eval(str_list)


@contextmanager
def stdout_redirector(stream):
    sys.stdout.flush()  # <--- important when redirecting to files
    # Save the original stdout file descriptor
    original_stdout_fd = sys.stdout.fileno()

    # Create a temporary file and get its file descriptor
    with os.fdopen(os.dup(original_stdout_fd), 'w') as saved_stdout:
        os.dup2(stream.fileno(), original_stdout_fd)
        try:
            sys.stdout.flush()
            yield
        finally:
            sys.stdout.flush()
            # Restore the original stdout file descriptor
            os.dup2(saved_stdout.fileno(), original_stdout_fd)


def shapes_are_same(input_shape, expected_input_shape):
    # check if the length are same, if not return False
    if len(input_shape) != len(expected_input_shape):
        return False
    # now check if the batch, height, width and channels are matching
    ip_shape = ops.TensorShape(input_shape)
    expctd_ip_shape = ops.TensorShape(expected_input_shape)
    if (ip_shape.height == expctd_ip_shape.height) and (ip_shape.width == expctd_ip_shape.width) and (ip_shape.depth == expctd_ip_shape.depth):
        return True
    return None


def reshape_input(input, expected_shape):
    expctd_shape = expected_shape
    if not isinstance(expected_shape, ops.TensorShape):
        expctd_shape = ops.TensorShape(expected_shape)
    input_shape = ops.TensorShape(np.array(input.shape))
    # check if the total length of the input matches or is an integral multiple of the expected shape
    if ((input_shape.get_length() % expctd_shape.get_length()) == 0):
        # reshape the input and return
        if expctd_shape.shape_size == 4:
            input = input.reshape(
                (input.shape[0], expctd_shape.height, expctd_shape.width, expctd_shape.depth))
        elif expctd_shape.shape_size == 2:
            if (expctd_shape.height == 1):
                input = input.reshape((input.shape[0], expctd_shape.width))
            else:
                input = input.reshape(
                    (input.shape[0], expctd_shape.height, expctd_shape.width))
    return input


def parse_get_dataset_script_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_yaml_path',
        type=str,
        help="""\
    path + file name to the configuration yaml file with the different inputs for the model
    """)
    parser.add_argument(
        '--model_name',
        type=str,
        help="""\
    The name of the model.
    """)
    return parser


def valid_c_symbol_name(name):
    if not (name[0].isalpha() or name[0] == '_'):
        return False
    for ch in name[1:]:
        if not (ch.isalnum() or ch == '_'):
            return False
    if name.startswith('__') or (name.startswith('_') and len(name) > 1 and name[1].isupper()):
        return False
    return True


def get_axon_compiler_object_filepath(TOOL_ROOT_DIR):
    AXON_COMPILER_OBJECT_PATH = ""
    # if not 'COMPILER_ROOT_FOLDER' in os.environ:
    #   #assuming that the compiler root location is up one directory
    #   COMPILER_ROOT_FOLDER = r".."
    # else:
    # put a check here for the root folder to be a valid system path
    COMPILER_ROOT_FOLDER = os.environ.get(
        'COMPILER_ROOT_FOLDER', TOOL_ROOT_DIR + r'/..')
    if not (Path(COMPILER_ROOT_FOLDER).exists()):
        logger.critical("COMPILER ROOT FOLDER doesn't exist!")
        exit()
    sys_arch = platform.machine().lower()
    lib_name_arch_suffix = ""
    logger.debug(f"Host Architecture: {sys_arch}")
    if sys_arch == "x86_64" or sys_arch == "amd64" or sys_arch == "x64":
        lib_name_arch_suffix = "amd64"
    elif sys_arch == "aarch64" or sys_arch == "arm64":
        lib_name_arch_suffix = "arm64"
    else:
        logger.error(f"unsupported host architecture: '{sys_arch}'")

    AXON_COMPILER_OBJECT_NAME = "nrf-axon-nn-compiler-lib-" + \
        lib_name_arch_suffix+".dll"  # "libaxons_ml_nn_compiler_lib.dll" #
    if os.name == "posix":
        if platform.system() == "Darwin":  # MacOS
            AXON_COMPILER_OBJECT_NAME = "libnrf-axon-nn-compiler-lib-" + \
                lib_name_arch_suffix+".dylib"
            # AXON_COMPILER_OBJECT_NAME="libnrf-axon-nn-compiler-lib-.dylib"
        else:  # Linux
            AXON_COMPILER_OBJECT_NAME = "libnrf-axon-nn-compiler-lib-"+lib_name_arch_suffix+".so"
    for path in Path(COMPILER_ROOT_FOLDER).rglob(AXON_COMPILER_OBJECT_NAME):
        # print(path)#DEBUG
        AXON_COMPILER_OBJECT_PATH = path
    AXON_COMPILER_OBJECT = Path(str(AXON_COMPILER_OBJECT_PATH)).absolute()

    return AXON_COMPILER_OBJECT


def create_dir_if_not_exists(dir):
    if not (Path(dir).exists()):
        Path(dir).mkdir(parents=True, exist_ok=True)
        print(f"created directory {dir}...")


def load_csv_lines_to_dict(file_path, start_index=0):
    array_dict = {}
    try:
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:  # skip empty lines
                    try:
                        line = line.rstrip(',')
                        values = [int(x) for x in line.split(',')]
                        array_dict[start_index +
                                   idx] = np.array(values, dtype=np.int32)
                    except ValueError as e:
                        print(f" Error parsing line {idx + 1}: {e}")
    except Exception as e:
        print(f"Error when loading csv file : {e}")
        return None
    return array_dict


def load_csv_lines_to_np_array(file_path, start_index=0):
    array_ = []
    try:
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:  # skip empty lines
                    try:
                        line = line.rstrip(',')
                        values = [int(x) for x in line.split(',')]
                        array_.append(np.array(values, dtype=np.int32))
                    except ValueError as e:
                        print(f" Error parsing line {idx + 1}: {e}")
    except Exception as e:
        print(f"Error when loading csv file : {e}")
        return None
    return np.array(array_)


def get_string_from_array_values(array, separator='x'):
    if not isinstance(array, (list, tuple)):
        return str(array)
    if len(array) > 0 and isinstance(array[0], (list, tuple)):
        return '_'.join(
            separator.join(str(v) for v in shape) for shape in array
        )
    return separator.join(str(v) for v in array)


def float_to_str(x): return '0f' + ('%.10f' % x).split('.')[1].rstrip('0')


def get_unit_test_model_name(test_op_info_dict):
    model_name = ""
    OP_TYPE = test_op_info_dict['OP_TYPE']

    if OP_TYPE == "UserModelInput":
        # the user is providing a model, extract the model name from the tflite file name
        if 'MODEL_NAME' in test_op_info_dict:
            model_name = test_op_info_dict['MODEL_NAME']
        else:
            # check if tflite file is given.
            if 'TFLITE_MODEL' in test_op_info_dict:
                model_name = Path(test_op_info_dict['TFLITE_MODEL']).name
                model_name = str(model_name.split('.')[-2])
            else:
                raise Exception(
                    "For OP_TYPE 'MODEL', user must provide the tflite model file!")
        return model_name

    input_shape = test_op_info_dict['IP_SHAPE']
    input_shape_string = get_string_from_array_values(input_shape)
    model_name = f"test_{OP_TYPE}_ip_{input_shape_string}"

    if 'MODEL_VARIANT' in test_op_info_dict:
        if test_op_info_dict['MODEL_VARIANT'] != "default":
            model_name += f"_{test_op_info_dict['MODEL_VARIANT']}"
    else:
        test_op_info_dict['MODEL_VARIANT'] = "default"

    if 'STRIDES' in test_op_info_dict:
        stride_string = get_string_from_array_values(
            test_op_info_dict['STRIDES'])
        model_name = model_name + f"_strides_{stride_string}"
        # model_name = model_name + f"_strd_{stride_string}"

    if 'FILTERS' in test_op_info_dict:
        model_name = model_name + f"_filters_{test_op_info_dict['FILTERS']}"
        # model_name = model_name + f"_fltr_{test_op_info_dict['FILTERS']}"

    if 'KERNEL_SIZE' in test_op_info_dict:
        kernel_string = get_string_from_array_values(
            test_op_info_dict['KERNEL_SIZE'])
        model_name = model_name + f"_kernel_{kernel_string}"
        # model_name = model_name + f"_krnl_{kernel_string}"

    if 'DILATION' in test_op_info_dict:
        dilation_string = get_string_from_array_values(
            test_op_info_dict['DILATION'])
        model_name = model_name + f"_dilation_{dilation_string}"
        # model_name = model_name + f"_dltn_{dilation_string}"

    if 'PADDING_TYPE' in test_op_info_dict:
        model_name = model_name + \
            f"_padding_{test_op_info_dict['PADDING_TYPE']}"
        # model_name = model_name + f"_pd_{test_op_info_dict['PADDING_TYPE']}"

    if 'ACTIVATION' in test_op_info_dict:
        model_name = model_name + \
            f"_activation_{test_op_info_dict['ACTIVATION']}"
        # model_name = model_name + f"_actv_{test_op_info_dict['ACTIVATION']}"

    disable_fc_chan_q = test_op_info_dict.get("DISABLE_FC_CHAN_Q", False)
    if disable_fc_chan_q:
        model_name = model_name + \
            "_fc_per_chan_q_disabled"

    if OP_TYPE == "Pad":
        if 'H_PAD' in test_op_info_dict:
            h_pad_string = get_string_from_array_values(
                test_op_info_dict['H_PAD'])
            model_name = model_name + f"_hpad_{h_pad_string}"
        if 'W_PAD' in test_op_info_dict:
            w_pad_string = get_string_from_array_values(
                test_op_info_dict['W_PAD'])
            model_name = model_name + f"_wpad_{w_pad_string}"
        if 'C_PAD' in test_op_info_dict:
            c_pad_string = get_string_from_array_values(
                test_op_info_dict['C_PAD'])
            model_name = model_name + f"_cpad_{c_pad_string}"
        if 'CONST_VALUE' in test_op_info_dict:
            model_name = model_name + \
                f"_const_{float_to_str(test_op_info_dict['CONST_VALUE'])}"

    if OP_TYPE == "SplitV":
        if 'SPLIT_SIZE' in test_op_info_dict:
            model_name = model_name + \
                f"_split_size_{str(test_op_info_dict['SPLIT_SIZE'])}"
        if 'AXIS' in test_op_info_dict:
            model_name = model_name + f"_axis_{str(test_op_info_dict['AXIS'])}"

    if OP_TYPE == "Mean":
        if 'MEAN_AXIS' in test_op_info_dict:
            mean_axis_string = get_string_from_array_values(
                test_op_info_dict['MEAN_AXIS'])
            model_name = model_name + \
                f"_mean_axis_{mean_axis_string}"

    if OP_TYPE == "LeakyRelu":
        if 'ALPHA' in test_op_info_dict:
            model_name = model_name + \
                f"_alpha_{float_to_str(test_op_info_dict['ALPHA'])}"

    if OP_TYPE in ["Add", "Multiply", "Subtract"]:
        if 'BROADCAST_AXIS' in test_op_info_dict:
            broadcast_axis_string = get_string_from_array_values(
                test_op_info_dict['BROADCAST_AXIS'])
            model_name = model_name + \
                f"_broadcast_axis_{broadcast_axis_string}"

    if OP_TYPE == "Dense":
        if 'OUT_DIM' in test_op_info_dict:
            out_dim_string = get_string_from_array_values(
                test_op_info_dict['OUT_DIM'])
            model_name = model_name + \
                f"_out_dim_{out_dim_string}"

    if OP_TYPE == "SpaceNBatch":
        if 'BLOCK_SIZE' in test_op_info_dict:
            block_size_string = get_string_from_array_values(
                test_op_info_dict['BLOCK_SIZE'])
            model_name = model_name + \
                f"_block_size_{block_size_string}"

    if OP_TYPE in ["Lstm", "LstmV2", "Gru"]:
        if 'UNITS' in test_op_info_dict:
            units_string = get_string_from_array_values(
                test_op_info_dict['UNITS'])
            model_name = model_name + \
                f"_units_{units_string}"
        return_sequence = test_op_info_dict.get("RETURN_SEQUENCES", False)
        if return_sequence:
            model_name = model_name + \
                "_return_seq"

    if 'ACT_BITS' in test_op_info_dict:
        act_bits_string = get_string_from_array_values(
            test_op_info_dict['ACT_BITS'])
        model_name = model_name + f"_act_bits_{act_bits_string}"
    else:
        test_op_info_dict['ACT_BITS'] = 8

    if OP_TYPE == "MIMO":
        if 'MERGE_OP' in test_op_info_dict:
            model_name += f"_merge_op_{test_op_info_dict['MERGE_OP']}"
        if 'OUTPUT_COUNT' in test_op_info_dict:
            model_name += f"_output_cnt_{test_op_info_dict['OUTPUT_COUNT']}"
    return model_name.lower()


def check_input_shape_for_inference(data, expected_shape):
    actual_shape = np.array(data.shape)
    # If shape doesn't match, try to reshape or expand
    if not np.array_equal(expected_shape, actual_shape):
        # print(f"INFO: Adjusting shape for "
        #         f"from {actual_shape.tolist()} to {expected_shape.tolist()}")

        if np.prod(actual_shape) == np.prod(expected_shape):
            data = data.reshape(expected_shape)
        elif actual_shape.size + 1 == len(expected_shape):
            data = np.expand_dims(data, axis=0)
        else:
            raise ValueError(
                f"Cannot automatically reshape input "
                f"from {actual_shape.tolist()} to {expected_shape.tolist()}"
            )
    return data


def plot_histogram_0_255(hist):
    x = np.arange(256)  # 0–255 index space

    plt.figure()
    plt.bar(x, hist)
    plt.title("Value Difference Histogram (0–255 index space)")
    plt.xlabel("Index (diff)")
    plt.ylabel("Count")
    plt.show()


def compute_value_diff_histogram(tflite_tensor, hw_tensor, act_bits=8):

    a = tflite_tensor.flatten().astype(np.int16)
    b = hw_tensor.flatten().astype(np.int16)

    diff_clip = 2**act_bits-1
    diff = abs(a - b)
    diff = np.clip(diff, 0, diff_clip)

    hist = np.bincount(diff, minlength=diff_clip+1)

    return hist, diff


def get_histogram_text_on_console(hist, diff_threshold=2, ratio_threshold=0.01, width=100, n="", act_bits=8):
    total = np.sum(hist)
    diff_max = 2**act_bits
    diff_values = np.arange(0, diff_max)
    nonzero = np.where(hist > 0)[0]
    histogram_text = print_with_borders(
        f"\tBit Comparison Histogram {n} ({total} data points)", border="", get_text=True)
    max_count = np.max(hist)
    # get histogram for zero index
    ratio = hist[0] / total
    bar_len = int((hist[0] / max_count) * width)
    # bar =  "█" * bar_len
    bar = "|" * bar_len
    d = diff_values[0]
    histogram_text += f"\n\t\tDiff\t||graph|| (count, percent of full data points)"
    histogram_text += f"\n\t\t{d:4d}\t|{bar} ({hist[0]}, {ratio:.2%})"

    if len(nonzero) == 1 and diff_values[nonzero[0]] == 0:
        histogram_text += "\n\tNo differences found. All values match perfectly.\n"
        return histogram_text

    for idx in nonzero:
        d = diff_values[idx]
        count = hist[idx]
        ratio = count / total

        if abs(d) <= diff_threshold:
            continue
        if ratio < ratio_threshold:
            continue

        bar_len = int((count / max_count) * width)
        bar = "|" * bar_len
        # bar = "H" * bar_len

        histogram_text += f"\n\t\t{d:4d}\t|{bar} ({count}, {ratio:.2%})"

    return histogram_text


def evaluate_bit_comparison_pass_fail(diff, tolerance=1, pass_ratio=0.99):
    """
    tolerance: allowed absolute difference (e.g., ±1)
    pass_ratio: % of values that must satisfy tolerance
    """

    total = len(diff)
    within_tol = np.sum(np.abs(diff) <= tolerance)

    ratio = within_tol / total

    hdr = print_with_borders("PASS/FAIL METRICS", border="-", get_text=True)
    results_text = f"\nTolerance       : +/- {tolerance}"
    results_text += f"\nWithin tolerance: {within_tol}/{total} ({ratio:.2%})"
    results_text += f"\nRequired ratio  : {pass_ratio:.2%}"

    if ratio >= pass_ratio:
        results_text += "\nRESULT: PASS"
        print(hdr+results_text)
        return True
    else:
        results_text += "\nRESULT: FAIL"
        print(hdr+results_text)
        return False


def save_histogram(hist, filename="value_diff_histogram.txt"):
    with open(filename, "w") as f:
        for i, count in enumerate(hist):
            # if count > 0:
            f.write(f"{i},{count}\n")


def print_with_borders(text, border="=", get_text=False):
    text = f"{border}{border} {text} {border}{border}"
    border_text = border*len(text)
    if get_text:
        ret_text = f"{border_text}\n{text}\n{border_text}"
        return ret_text
    print(f"{border_text}")
    print(text)
    print(f"{border_text}")


def normalize_into_list(vector):
    if isinstance(vector[0], (list, tuple)):
        return list(vector)
    return [vector]


def append_ndx_to_path(filepath, ndx, separator="_"):
    path = Path(filepath)
    return str(path.with_name(f"{path.stem}{separator}{ndx}{path.suffix}"))


def save_test_vectors_as_npy(test_vector, test_vector_file_path):
    if isinstance(test_vector, np.ndarray):
        np.save(test_vector_file_path, test_vector)
        return test_vector_file_path
    if not isinstance(test_vector, list) or len(test_vector) == 0:
        raise TypeError(f"Unsupported type: {type(test_vector)}")
    if len(test_vector) == 1:
        return save_test_vectors_as_npy(test_vector[0], test_vector_file_path)
    arrs = [np.asarray(x) for x in test_vector]
    # Single output stored as many samples, each batch dim 1 -> one stacked file
    first_arr_size = arrs[0].size
    if all(a.ndim >= 1 and a.shape[0] == 1 and a.size == first_arr_size for a in arrs) and len(arrs) > 2:
        np.save(test_vector_file_path, np.stack(arrs, axis=0))
        return test_vector_file_path
    # Multi-input / multi-output -> file_0, file_1, ...
    paths = []
    for ndx, a in enumerate(arrs):
        p = append_ndx_to_path(test_vector_file_path, ndx)
        np.save(p, a)
        paths.append(p)
    return paths


def reshape_test_vector_for_first_reshape_op(x, reshape_ip_shape, reshape_op_shape):
    n = len(x)
    h, w, d = (
        reshape_op_shape.height,
        reshape_op_shape.width,
        reshape_op_shape.depth,
    )
    simple_swap = (
        reshape_ip_shape.depth == reshape_op_shape.width
        and reshape_op_shape.depth == 1
    )
    channel_axis_changes = (
        reshape_op_shape.depth != reshape_ip_shape.depth and not simple_swap
    )
    layout_fix = channel_axis_changes and (
        reshape_ip_shape.shape_size == 4
        or (
            reshape_ip_shape.shape_size == 3
            and reshape_op_shape.shape_size == 2
        )
    )
    if layout_fix:
        original_shape = x.shape
        return x.reshape(n, h, w, d).transpose(0, 3, 1, 2).reshape(original_shape)
    return x.reshape(n, h, w, d)


def format_input_test_vectors_to_run_inference_on_compiler(yaml_dict, tflite_axon_op_graph=None):
    # buf = bytearray(open(yaml_dict['tflite_model'], "rb").read())
    # model = tflite.Model.GetRootAsModel(buf, 0)
    subgraph = tflite_axon_op_graph.get_tflite_subgraph()
    # interpreter = tf.lite.Interpreter(
    #     model_path=yaml_dict['tflite_model'], experimental_preserve_all_tensors=True)
    # interpreter.allocate_tensors()
    interpreter = tflite_axon_op_graph.get_tflite_interpreter()
    ops_details = interpreter._get_ops_details()
    tensor_details = interpreter.get_tensor_details()
    inputs = tflite_axon_op_graph.get_tflite_input_details()
    reshape_input = False
    input_reshaped = False
    input_transposed = False
    transpose_model_input_flag = False
    model_ip_datawidth = inputs[0]['dtype']
    operator_graph_info = tflite_axon_op_graph.get_axon_operator_graph_info()
    x_test_q = copy.deepcopy(yaml_dict['quantized_data'])
    x_test_output_q = x_test_q
    tflite_input_operations = tflite_axon_op_graph.get_tflite_input_operations()

    for ndx, _ip_operation in enumerate(tflite_input_operations):
        # get the operation that has the input tensor in its ip_tensors and check against that
        first_op_index = _ip_operation['index']
        first_op_name = _ip_operation['op_name']

        # if model_ip_datawidth != np.int8 and x_test_q[ndx].dtype != np.int8 and ops_details[first_op_index]['op_name'] == "QUANTIZE":
        if (((model_ip_datawidth != np.int8 and x_test_q[ndx].dtype != np.int8)
            or (model_ip_datawidth != np.int16 and x_test_q[ndx].dtype != np.int16))
                and (ops_details[first_op_index]['op_name'] == "QUANTIZE")):
            scale, model_ip_zeropoint = tensor_details[operator_graph_info[first_op_index]
                                                       ['op_tensors'][0]]['quantization']
            x_test_q[ndx] = np.round(
                # (x_test_q[ndx]/scale)+model_ip_zeropoint).astype(np.int8)
                (x_test_q[ndx]/scale)+model_ip_zeropoint).astype(operator_graph_info[first_op_index]['op_tensors'].dtype)
            x_test_output_q[ndx] = x_test_q[ndx]
            # FIXME update the next op as the first op

        if (first_op_name == "RESHAPE"):
            reshape_input = True
            # check if the test input needs to be transposed
            # we only transpose the test input if the reshape puts some element in the channel
            reshape_ip_shape = ops.TensorShape(
                tensor_details[subgraph.Operators(first_op_index).InputsAsNumpy()[0]]['shape'])
            reshape_op_shape = ops.TensorShape(
                tensor_details[subgraph.Operators(first_op_index).OutputsAsNumpy()[0]]['shape'])
            next_op_index = tflite_axon_op_graph.get_tflite_op_index_from_input_tensor_index(
                ops_details[first_op_index]['outputs'][0])
            if ops_details[next_op_index]["op_name"] == "UNPACK" and reshape_op_shape.shape_size == 3:
                # confirm that this is indeed an LSTM
                fc_op_index = tflite_axon_op_graph.get_tflite_op_index_from_input_tensor_index(
                    ops_details[next_op_index]['outputs'][0])
                if ops_details[fc_op_index]["op_name"] == "FULLY_CONNECTED":
                    # this is an input to an unfused LSTM block
                    x_test_output_q[ndx] = x_test_q[ndx].reshape(len(
                        x_test_q[ndx]), reshape_op_shape.height, reshape_op_shape.depth, reshape_op_shape.width)
            else:
                channel_width_swapped = (
                    reshape_ip_shape.depth == reshape_op_shape.width
                    and reshape_op_shape.depth == 1
                )
                channel_axis_changed = (
                    (reshape_op_shape.depth != reshape_ip_shape.depth)
                    and not channel_width_swapped)

                nhwc_layout_change_needed = channel_axis_changed and (
                    reshape_ip_shape.shape_size == 4
                    or (reshape_ip_shape.shape_size == 3
                        and reshape_op_shape.shape_size == 2)
                )
                if nhwc_layout_change_needed:
                    original_shape = x_test_q[ndx].shape
                    x_test_output_q[ndx] = x_test_q[ndx].reshape(len(
                        x_test_q[ndx]), reshape_op_shape.height, reshape_op_shape.width, reshape_op_shape.depth)
                    x_test_output_q[ndx] = x_test_output_q[ndx].transpose(
                        0, 3, 1, 2)
                    x_test_output_q[ndx] = x_test_output_q[ndx].reshape(
                        original_shape)
                else:
                    x_test_output_q[ndx] = x_test_q[ndx].reshape(len(
                        x_test_q[ndx]), reshape_op_shape.height, reshape_op_shape.width, reshape_op_shape.depth)

        elif (first_op_name == "UNIDIRECTIONAL_SEQUENCE_LSTM"):
            # the input shape needs to be added with a depth so that it works through the save mass vectors which tries to transpose the input based on the input shape
            lstm_ip_shape = ops.TensorShape(x_test_q[ndx].shape)
            x_test_output_q[ndx] = x_test_q[ndx].reshape(
                len(x_test_q[ndx]), lstm_ip_shape.width, lstm_ip_shape.depth, 1)

        if yaml_dict['transpose_kernel']:
            # check if the first input is reshape
            check_index = 1 if reshape_input else 0
            if ops_details[check_index]["op_name"] != "FULLY_CONNECTED":
                # op_type = yaml_dict.get('OP_TYPE', None)
                # x_test_output_q[ndx] = apply_mass_inference_input_layout(x_test_output_q[ndx],True,op_type)
                transpose_model_input_flag = True

    return x_test_output_q, transpose_model_input_flag


def sample_test_data_by_indices(x_test, indices):
    """
    x_test: list of per-input arrays (N, ...), or a single ndarray (legacy).
    indices: iterable of sample indices (test_io_vector_ndx or bit_comparison_range).
    Returns: list of arrays (K, ...) in TFLite input order.
    """
    if isinstance(x_test, np.ndarray):
        x_test = [x_test]
    idx = np.asarray(indices)
    return [np.asarray(x_test[n][idx]) for n in range(len(x_test))]


def tflite_input_index_from_name(name: str, signature: str = "test_input_") -> int:
    m = re.search(fr"{signature}_(\d+)", name or "")
    if not m:
        raise ValueError(
            f"Cannot map TFLite input name to {signature}_i: {name!r}")
    return int(m.group(1))


def align_test_data_to_input_details(test_data, input_details, input_signatures):
    """Returns list ordered like input_details, sourced from test_data[keras_i]."""
    if isinstance(test_data, np.ndarray):
        test_data = [test_data]
    aligned = [None] * len(input_details)
    for tflite_ndx, det in enumerate(input_details):
        keras_i = tflite_input_index_from_name(
            det["name"], signature=input_signatures[tflite_ndx])
        if keras_i >= len(test_data):
            raise IndexError(
                f"test_input_{keras_i} not in test_data (len={len(test_data)})")
        aligned[tflite_ndx] = test_data[keras_i]
    return aligned


def align_data_q_using_input_signatures(user_signatures, data_q, input_details):
    """
    Reorder data_q to match input_details order.
    user_signatures[i] labels data_q[i] (user order).
    input_details[j]['name'] is the TFLite name for slot j.
    Returns a new list: aligned[j] is the array for input_details[j].
    """
    if len(user_signatures) != len(data_q):
        raise ValueError(
            f"len(user_signatures)={len(user_signatures)} != len(data_q)={len(data_q)}"
        )
    if len(data_q) != len(input_details):
        raise ValueError(
            f"len(data_q)={len(data_q)} != len(input_details)={len(input_details)}"
        )
    aligned = [None] * len(input_details)
    used_slots = set()
    for user_idx, user_sig in enumerate(user_signatures):
        for slot, inp in enumerate(input_details):
            if slot in used_slots:
                continue
            if user_sig in inp["name"]:
                aligned[slot] = data_q[user_idx]
                used_slots.add(slot)
                break
        else:
            raise ValueError(
                f"No input_details name contains {user_sig!r}. "
                f"Names: {[inp['name'] for inp in input_details]}"
            )
    if any(x is None for x in aligned):
        missing = [input_details[i]["name"]
                   for i, x in enumerate(aligned) if x is None]
        raise ValueError(f"Unmatched input_details slots: {missing}")
    return aligned


NO_TRANSPOSE_TFLITE_OPERATORS = [
    tflite.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM]
