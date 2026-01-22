""" 
/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */
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
    data_ += 1e-8  # to avoid divide by zero errors
    if bool(np.array(data_).any()):
        error = ((np.round(data_*(2**scaling_shift)) /
                 (2**scaling_shift))-data_)/(data_)
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
    op_zp = (op_zp).astype(np.int64)  # to avoid the zerop point to overflow
    all_ = [(-1, 0)]
    for scaleshift in range(min_, max_):
        sc = int(np.round(data_*(2**scaleshift)))
        zp = int(np.round(int(op_zp*(2**scaleshift))))
        # Quantized values should not be more than 24 signed bits.
        if (abs(sc) < (2**bit_limit)) and (abs(zp) < (2**zp_bit_limit)):
            error = scale_error(data_, scaleshift)
            all_.append((error, scaleshift))
        else:
            break
    return all_[-1]


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
                print_string += f"non zero at [{non_zero[0][e],non_zero[1][e],non_zero[2][e]}] tflite : {tflite_vector[non_zero[0][e]][non_zero[1][e]][non_zero[2][e]]} calculated : {calc_vector[non_zero[0][e]][non_zero[1][e]][non_zero[2][e]]} difference {diff_vector[non_zero[0][e]][non_zero[1][e]][non_zero[2][e]]}\n"
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
    scale_q_ = np.array([])
    scaleshift_ = np.array([])
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
        scale_q = abs(int(np.round(scale*2**scaleshift)))
        scale_q_ = np.append(scale_q_, scale_q)
        scaleshift_ = np.append(scaleshift_, scaleshift)

    return scaleshift_.astype(np.int8), scale_q_.astype(np.int32), error_


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


def get_array_from_tensor(input_tensor, transpose_flag=False):
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


def save_vectors_for_mass_inference(test_vector_file_name, test_vectors_ndx, quantized_test_vectors, test_labels, output_dir, transpose_model_ip_flag=False, save_test_labels=False):
    """create logger object"""
    logger = logging.getLogger(__name__)
    # print("\nGetting the quantized input test vector and labels for the simulator....")
    logger.debug(
        "getting the quantized input test vector and labels for mass inferencing using the simulator...")
    test_vector_text = ""
    test_labels_text = ""
    q_input_shape = ops.TensorShape(np.array(quantized_test_vectors.shape))
    # save_test_labels = test_labels.size>1
    q_test_vector = quantized_test_vectors
    """ also transpose input vector if the rotate model is set"""
    if transpose_model_ip_flag:
        # we have to transpose the input but,
        # check if the operator wants the input to be transposed
        # if the first actual operator is FC, the input may not be needed to be transposed
        # how do we get the actual input, that is the question, a quick fix is to just check if the operator is Fully_connected
        if q_input_shape.shape_size == 4:
            q_test_vector = q_test_vector.transpose(0, 2, 1, 3)
        elif q_input_shape.shape_size == 3:
            q_test_vector = q_test_vector.transpose(0, 2, 1)

    if (q_input_shape.depth > 1):  # we have a multichannel input
        q_test_vector = q_test_vector.transpose(0, 3, 1, 2)

    def get_csv_text_map_func(vector):
        test_vector_text_m = ""
        test_vector_text_m += write_array_to_file(vector.squeeze(), "")
        test_vector_text_m = get_csv_text(test_vector_text_m)
        return test_vector_text_m

    if (len(test_vectors_ndx) == len(q_test_vector)):  # doing it for the whole dataset
        test_vector_iter = list(q_test_vector)
    else:
        test_vector_iter = [q_test_vector[ndx] for ndx in test_vectors_ndx]
    test_vectors_list = list(map(get_csv_text_map_func, test_vector_iter))

    for ndx, vector_value in enumerate(test_vectors_ndx):

        # test_vect = q_test_vector[ndx]
        # test_vector_text += write_array_to_file(test_vect.squeeze(),"")
        # test_vector_text = get_csv_text(test_vector_text)

        test_vector_text += test_vectors_list[ndx]

        if (save_test_labels):
            test_labels_text += str(test_labels[vector_value])+"\n"

    test_labels_text += "\n"
    test_vector_text = test_vector_text.replace(' ', '')
    # replacing the carriage return in windows to ensure that it works when running inference from a linux app!
    test_vector_text = test_vector_text.replace('\r', '')
    save_to_file(output_dir, test_vector_file_name, test_vector_text)
    if save_test_labels:
        label_filename = test_vector_file_name.split(
            ".")[0] + "labels_." + test_vector_file_name.split(".")[1]
        save_to_file(output_dir, label_filename, test_labels_text)
    # print(f"saved the test vector text files to {compiler_outputs_dir}")
    logger.debug(
        f"saved the test vector's text files for mass inferencing to {output_dir}")
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


def get_output_radix(op_radix, scale_shift, op_scale, op_zp, op_bw=np.int8):
    # determine the best op_radix here based on the output quantization values
    # the output quantization always quantizes the output to 8 bit, but that can be checked and,
    # the necessary max and min values can be calculated
    # using that information the user can determine a radix that might just be apt for the output
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
    if (op_radix <= 0):  # the user has not set the output radix, we need to figure out one
        op_radix = 8  # lets default that at 8 and then use the scale_shift values to maximize that accordingly
    for radix in range(op_radix, shift_max_range):
        if (layer_max_value*2**radix) > (2**shift_max_range - 1) or (layer_min_value*2**radix) < (-2**shift_max_range):
            break
    return radix


def find_variant_settings(tflite_model_file):
    softmax_present = False
    per_channel_scales_present = False
    conv2d_multichannel_present = False
    transpose_kernel_present = False

    # buf = bytearray(open(tflite_model_file, "rb").read())
    # model = tflite.Model.GetRootAsModel(buf, 0)
    # subgraph = model.Subgraphs(0)
    interpreter = tf.lite.Interpreter(
        model_path=tflite_model_file, experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()
    ops_details = interpreter._get_ops_details()
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


def get_string_from_array_values(array, seperator='x'):
    if not isinstance(array, list):
        return str(array)
    return seperator.join(str(v) for v in array)


def float_to_str(x): return '0f' + ('%.10f' % x).split('.')[1].rstrip('0')


def get_unit_test_model_name(test_op_info_dict):
    model_name = ""
    OP_TYPE = test_op_info_dict['OP_TYPE']

    if 'TEST_VECTOR_COUNT' not in test_op_info_dict:
        test_op_info_dict['TEST_VECTOR_COUNT'] = 4  # default count of vectors

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

    if 'PADDING_TYPE' in test_op_info_dict:
        model_name = model_name + \
            f"_padding_{test_op_info_dict['PADDING_TYPE']}"
        # model_name = model_name + f"_pd_{test_op_info_dict['PADDING_TYPE']}"

    if 'ACTIVATION' in test_op_info_dict:
        model_name = model_name + \
            f"_activation_{test_op_info_dict['ACTIVATION']}"
        # model_name = model_name + f"_actv_{test_op_info_dict['ACTIVATION']}"

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
            mean_axis_string = get_string_from_array_values(test_op_info_dict['MEAN_AXIS'])
            model_name = model_name + \
                f"_mean_axis_{mean_axis_string}"

    if OP_TYPE == "LeakyRelu":
        if 'ALPHA' in test_op_info_dict:
            model_name = model_name + \
                f"_alpha_{float_to_str(test_op_info_dict['ALPHA'])}"
    
    if OP_TYPE == "Multiply":
        if 'BROADCAST_AXIS' in test_op_info_dict:
            broadcast_axis_string = get_string_from_array_values(test_op_info_dict['BROADCAST_AXIS'])
            model_name = model_name + \
                f"_broadcast_axis_{broadcast_axis_string}"
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
