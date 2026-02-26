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
import logging
import numpy as np
import tensorflow as tf
import tflite as tflite
import model_wrapper as mw
from enum import Enum
from utility import util as util
from utility import cpu_operator_options as cpu_operator_options


class TensorShape:
    batch = 0
    height = 0
    width = 0
    depth = 0
    shape_size = 0
    shape = None

    def __init__(self, shape, shape_rank=None):
        if isinstance(shape, list) or isinstance(shape, tuple):
            shape = np.array(shape)
        if np.any(shape):
            if (shape.size == 2):
                self.height = shape[0]
                self.width = shape[1]
                self.batch = 1
                self.depth = 1
            elif (shape.size == 4):
                self.batch = shape[0]
                self.height = shape[1]
                self.width = shape[2]
                self.depth = shape[3]
            elif (shape.size == 3):
                if shape_rank is None:
                    if shape[0] == 1: #FIXME this could be channel or batches, but if it is one can we keep it as batch and get rid of the channels?
                        self.batch = 1
                        self.height = shape[1] 
                        self.width = shape[2]
                        self.depth = 1
                    else:
                        self.batch = 1
                        self.height = shape[0]
                        self.width = shape[1]
                        self.depth = shape[2]
                else:
                    if shape_rank==4: #preserve channels only when requested shape rank is 4 else, get rid of the channel
                        self.batch = 1
                        self.height = shape[0]
                        self.width = shape[1]
                        self.depth = shape[2]
                    else: 
                        self.batch = shape[0]
                        self.height = shape[1] 
                        self.width = shape[2]
                        self.depth = 1
            elif (shape.size == 1):
                self.batch = shape[0]
                self.height = 1
                self.width = 1
                self.depth = 1
            else:
                self.batch = 0
                self.height = 0
                self.width = 0
                self.depth = 0
            self.shape_size = shape.size
        self.shape = shape

    def get_shape(self):
        return [self.batch, self.height, self.width, self.depth]

    def get_axon_shape(self, with_batch=True):
        if not with_batch:
            return [self.depth, self.height, self.width]
        return [self.batch, self.depth, self.height, self.width]

    def get_length(self):
        return self.batch*self.height*self.width*self.depth

    def get_axon_axis_shape(self, axon_axis_enum_dict):
        axon_axis_shape = [-1]*axon_axis_enum_dict['NRF_AXON_NN_AXIS_COUNT']
        axon_axis_shape[axon_axis_enum_dict['NRF_AXON_NN_AXIS_CHANNEL']] = self.depth
        axon_axis_shape[axon_axis_enum_dict['NRF_AXON_NN_AXIS_HEIGHT']] = self.height
        axon_axis_shape[axon_axis_enum_dict['NRF_AXON_NN_AXIS_WIDTH']] = self.width
        return axon_axis_shape

    def get_transpose_shape(self):
        self.height, self.width = self.width, self.height
        return self.get_shape()

    @classmethod
    def shapes_are_same(cls, shape1, shape2):
        return shape1.batch == shape2.batch and shape1.height == shape2.height and shape1.width == shape2.width and shape1.depth == shape2.depth


class PadDetails:
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    pad_front = 0
    pad_back = 0


class OperatorOptions:
    operator_name = ""
    bytes = 0
    pos = 0
    option = None
    padding = "None"
    activation = "None"
    custom_activation = False
    operation_detail = {}
    pad_info = None
    operator = None
    tflite_interpreter = None
    ip_shape = TensorShape(np.array([]))
    op_shape = TensorShape(np.array([]))
    filter_tensor = np.array([], dtype=np.int32)
    kernel_shape = TensorShape(np.array([]))
    bias_tensor = np.array([])
    b_prime_tensor = np.array([], dtype=np.int32)
    bias_shape = TensorShape(np.array([]))
    stride_shape = TensorShape(np.array([]))
    operand_str = ""
    error = False
    error_text = ""
    error_action = ""
    ip_bitwidth = np.int8
    op_bitwidth = np.int8
    kernel_bitwidth = np.int8
    kernel_bytewidth_enum = ""
    init_q = {'scales': np.array([0]),
              'zero_points': np.array([0])}
    ip_q = init_q
    ip_q_zeropoint = None
    op_q_zeropoint = None
    w_q = init_q  # {}
    bias_q = init_q  # {}
    op_q = init_q  # {}
    stride_q = init_q
    api_defn = ""
    scale_multipliers = np.array([])
    scale_shifts = np.array([])
    command_buf_len = 0
    cmd_buf_len_multiplier = 1
    stride_x = 0
    stride_y = 0
    dilation_x = 0
    dilation_y = 0
    ops_ip_shape = None
    input_count = 0
    custom_cpu_op = False
    cpu_op_attributes = np.array([])
    meta_data_str = "NA"
    ip_ndxs = None
    op_ndxs = None
    normalize_scaleshifts = None
    skip_softmax_op = None
    disable_op_quantization = None
    operator_info_graph = None
    transpose_kernel = None
    LEAKY_RELU_FLAG = None
    max_scaling_error = -1
    max_error_layer_name = ""
    max_single_scaling_error = -1
    max_single_error_layer_name = ""
    layer_output_radix = 0
    scaleshift_max_range = 31
    default_tensor_constraints = {
        "H": 1024,
        "W": 1024,
        "C": 512,
    }
    default_filter_constraints = {
        "H": 16,
        "W": 16
    }
    axons_operation_enum = ""

    def initialize_all_class_variables(self):
        self.operator_name = ""
        self.bytes = 0
        self.pos = 0
        self.option = None
        self.padding = "None"
        self.activation = "None"
        self.custom_activation = False
        self.operation_detail = {}
        self.pad_info = None
        self.operator = None
        self.tflite_interpreter = None
        self.ip_shape = TensorShape(np.array([]))
        self.op_shape = TensorShape(np.array([]))
        self.filter_tensor = np.array([], dtype=np.int32)
        self.kernel_shape = TensorShape(np.array([]))
        self.bias_tensor = np.array([])
        self.b_prime_tensor = np.array([], dtype=np.int32)
        self.bias_shape = TensorShape(np.array([]))
        self.stride_shape = TensorShape(np.array([]))
        self.operand_str = ""
        self.error = False
        self.error_text = ""
        self.error_action = ""
        self.ip_bitwidth = np.int8
        self.op_bitwidth = np.int8
        self.kernel_bitwidth = np.int8
        self.kernel_bytewidth_enum = ""
        self.ip_q = self.init_q
        self.ip_q_zeropoint = None
        self.op_q_zeropoint = None
        self.w_q = self.init_q  # {}
        self.bias_q = self.init_q  # {}
        self.op_q = self.init_q  # {}
        self.stride_q = self.init_q
        self.api_defn = ""
        self.scale_multipliers = np.array([])
        self.scale_shifts = np.array([])
        self.command_buf_len = 0
        self.cmd_buf_len_multiplier = 1
        self.stride_x = 0
        self.stride_y = 0
        self.dilation_x = 0
        self.dilation_y = 0
        self.ops_ip_shape = None
        self.input_count = 0
        self.custom_cpu_op = False
        self.cpu_op_attributes = np.array([])
        self.meta_data_str = "NA"
        self.ip_ndxs = None
        self.op_ndxs = None
        self.normalize_scaleshifts = None
        self.skip_softmax_op = None
        self.disable_op_quantization = None
        self.operator_info_graph = None
        self.transpose_kernel = None
        self.LEAKY_RELU_FLAG = None
        self.max_scaling_error = -1
        self.max_error_layer_name = ""
        self.max_single_scaling_error = -1
        self.max_single_error_layer_name = ""
        self.layer_output_radix = 0
        self.scaleshift_max_range = 31

    @classmethod
    def check_tensor_constraints(cls, tensor_type, tensor, constraints=default_tensor_constraints, transpose_check=False):
        error_text = ""
        if not isinstance(tensor, TensorShape):
            t_shape = TensorShape(tensor.shape)
        else:
            t_shape = tensor
        if transpose_check:
            t_shape.height, t_shape.width = t_shape.width, t_shape.height

        if t_shape.height > constraints['H']:
            error_text += f"{tensor_type} height {t_shape.height} > {constraints['H']} "

        if t_shape.width > constraints['W']:
            error_text += f"{tensor_type} width {t_shape.width} > {constraints['W']} "
        if t_shape.shape_size > 3:
            if constraints['C'] is not None:
                if t_shape.depth > constraints['C']:
                    error_text += f"{tensor_type} channel {t_shape.depth} > {constraints['C']} "
        return error_text

    @classmethod
    def check_filter_constraints(cls, filter_h, filter_w, constraints=default_filter_constraints, transpose_check=False):
        error_text = ""
        if transpose_check:
            filter_h, filter_w = filter_w, filter_h

        if filter_h > constraints['H']:
            error_text += f"filter height {filter_h} > {constraints['H']} "
        if filter_w > constraints['W']:
            error_text += f"filter width {filter_w} > {constraints['W']} "
        return error_text

    @classmethod
    def check_operator_specific_constraints(cls, built_in_options, input_tensors, output_shapes, transpose_check=False):
        error_text = cls.check_default_constraints(
            built_in_options, input_tensors, output_shapes, transpose_check)
        return error_text

    @classmethod
    def check_default_constraints(cls, built_in_options, input_tensors, output_shapes, transpose_check=False):
        # the default check is on the input and the output shape at index 0
        error_text = ""
        if len(input_tensors) > 0:
            error_text += cls.check_tensor_constraints(
                "input", input_tensors[0], transpose_check=transpose_check)
        if len(output_shapes) > 0:
            error_text += cls.check_tensor_constraints(
                "output", TensorShape(output_shapes[0]), transpose_check=transpose_check)

        return error_text

    @classmethod
    def check_for_constraints(cls, op_name, built_in_options, input_tensors, output_shapes):
        error_text = ""
        error_text += cls.check_operator_specific_constraints(
            built_in_options, input_tensors, output_shapes)
        op_supported = error_text == ""
        if not op_supported:
            error_text = f"{op_name} violates constraints: " + error_text
        # check for support if the model is transposed as well
        tr_error_text = ""
        tr_error_text += cls.check_operator_specific_constraints(
            built_in_options, input_tensors, output_shapes, True)
        tr_op_support = tr_error_text == ""
        if not tr_op_support:
            tr_error_text = f" transposed {op_name} violates constraints: " + tr_error_text
        return [(op_supported, error_text), (tr_op_support, tr_error_text)]

    @classmethod
    def get_previous_valid_axon_operator_index(cls, operator_graph, current_operator_index):
        for operator_index in range(current_operator_index-1, -1, -1):
            if operator_graph[operator_index]['axon_layer_num'] >= 0:
                return operator_index
        return None

    @classmethod
    def transpose_tensor_if_needed(cls, tensor, transpose_kernel_flag):
        if transpose_kernel_flag:
            # transpose the tensor here
            tensor_shape = TensorShape(np.array(tensor.shape))
            if tensor_shape.shape_size == 4:
                return tensor.transpose(0, 2, 1, 3)
            elif tensor_shape.shape_size == 2:
                return tensor.transpose(1, 0)
        return tensor

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.initialize_all_class_variables()
        # now throw errors here directly as operation calling this init are passthrough operators, mostly
        raise KeyError(-917)

    @classmethod
    def CreateOptionsObject(cls, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph=None, tflite_axon_enum_wrapper=None):
        return cls(operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)

    def InitOperatorOption(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.initialize_all_class_variables()
        self.error = False
        self.error_action = ""
        self.error_text = ""
        operator_options = operator.BuiltinOptions()
        if (operator_options is None):
            from flatbuffers.table import Table
            operator_options = Table(bytearray(), 0)
        self.bytes = operator_options.Bytes
        self.pos = operator_options.Pos
        self.interpreter = tflite_interpreter
        self.operation_detail = operation_detail
        self.operator = operator
        self.ip_shape = TensorShape(
            tensor_details[operator.InputsAsNumpy()[0]]['shape'])
        self.input_count = self.operator.InputsAsNumpy().size
        self.ip_ndxs = self.operator.InputsAsNumpy()
        self.op_ndxs = self.operator.OutputsAsNumpy()
        if operator_graph is not None:
            self.operator_info_graph = operator_graph
        # FIXME NEED TO MOVE THIS INSIDE EACH OPERATOR
        if (self.input_count == 1):
            self.ip_shape = TensorShape(
                tensor_details[operator.InputsAsNumpy()[0]]['shape'])
            self.ip_q = copy.deepcopy(
                tensor_details[self.ip_ndxs[0]]['quantization_parameters'])
        elif (self.input_count == 2):
            self.ip_shape = TensorShape(
                tensor_details[operator.InputsAsNumpy()[0]]['shape'])
            self.ip_q = copy.deepcopy(
                tensor_details[self.ip_ndxs[0]]['quantization_parameters'])
            self.kernel_shape = TensorShape(
                tensor_details[operator.InputsAsNumpy()[1]]['shape'])  # may not be kernel
            self.w_q = tensor_details[self.ip_ndxs[1]
                                      ]['quantization_parameters']
        elif (self.input_count == 3):
            self.ip_shape = TensorShape(
                tensor_details[operator.InputsAsNumpy()[0]]['shape'])
            self.ip_q = copy.deepcopy(
                tensor_details[self.ip_ndxs[0]]['quantization_parameters'])
            self.kernel_shape = TensorShape(
                tensor_details[operator.InputsAsNumpy()[1]]['shape'])  # may not be kernel
            self.bias_shape = TensorShape(
                tensor_details[operator.InputsAsNumpy()[2]]['shape'])  # may not be bias
            self.w_q = tensor_details[self.ip_ndxs[1]
                                      ]['quantization_parameters']
            self.bias_q = tensor_details[self.ip_ndxs[2]
                                         ]['quantization_parameters']
        elif (self.input_count == 4):  # for strided slcie there are four inputs
            self.ip_shape = TensorShape(
                tensor_details[operator.InputsAsNumpy()[0]]['shape'])  # input
            self.ip_q = copy.deepcopy(
                tensor_details[self.ip_ndxs[0]]['quantization_parameters'])
            self.kernel_shape = TensorShape(
                tensor_details[operator.InputsAsNumpy()[1]]['shape'])  # is begin
            self.bias_shape = TensorShape(
                tensor_details[operator.InputsAsNumpy()[2]]['shape'])  # is end
            self.w_q = tensor_details[self.ip_ndxs[1]
                                      ]['quantization_parameters']
            self.bias_q = tensor_details[self.ip_ndxs[2]
                                         ]['quantization_parameters']
            self.stride_shape = TensorShape(
                tensor_details[operator.InputsAsNumpy()[3]]['shape'])  # is strides
            self.stride_q = tensor_details[operator.InputsAsNumpy(
            )[3]]['quantization_parameters']  # is strides
        else:
            # FIXME : not handling more than four input vectors yet
            self.error = True
            self.error_text += f"|{self.operator_name} has more than 4 inputs, Only 4 inputs are currently supported|"
            self.error_action = "ERROR"
        self.op_shape = TensorShape(
            tensor_details[operator.OutputsAsNumpy()[0]]['shape'])
        self.ip_bitwidth = tensor_details[operator.InputsAsNumpy()[0]]['dtype']
        self.op_bitwidth = tensor_details[operator.OutputsAsNumpy()[
            0]]['dtype']
        self.kernel_bytewidth_enum = tflite_axon_enum_wrapper.GetAxonByteWidthEnum(
            self.kernel_bitwidth)
        self.ip_q_zeropoint = copy.deepcopy(self.ip_q['zero_points'])
        self.op_q = copy.deepcopy(
            tensor_details[self.op_ndxs[0]]['quantization_parameters'])
        self.op_q_zeropoint = copy.deepcopy(self.op_q['zero_points'])
        self.operator_name = self.operation_detail["op_name"]
        self.pad_info = PadDetails()
        self.tflite_axon_enum_wrapper = tflite_axon_enum_wrapper

    def HandleOutputRadix(self):
        if self.layer_output_radix < min(self.scale_shifts):
            self.scale_shifts -= self.layer_output_radix
        else:
            raise Exception(
                f"The output radix {self.layer_output_radix} cannot be acheived with the current scaling values")

    def NormalizeScaleshifts(self):
        # check here if there are multiple scaling values for each channel output
        if self.op_shape.depth <= 1:
            return
        if (self.LEAKY_RELU_FLAG):
            scale_shifts = np.concatenate(
                [self.scale_shifts, self.scale_shifts])
            scale = (self.scale_multipliers /
                     (2**scale_shifts.astype(np.float32))).astype(np.float32)
        else:
            scale = (self.scale_multipliers /
                     (2**self.scale_shifts.astype(np.float32))).astype(np.float32)

        min_error_ndx = np.argmin(self.scale_shifts)
        single_scaleshift = self.scale_shifts[min_error_ndx]
        self.scale_multipliers = abs(
            np.round(scale*2**single_scaleshift)).astype(np.int32)
        errors_ = util.scale_error(scale, single_scaleshift)
        min_error_ndx = np.argmax(errors_)
        self.scale_shifts = np.array([single_scaleshift], dtype=np.int8)

    def CalculateMultiplierandScaleshift(self):  # default one
        op_index = self.operation_detail['index']
        if self.operator_info_graph is not None:
            next_op_graph_index = SupportedOperators.get_next_op_graph_index_for_axon_layer_num(
                self.operator_info_graph, self.operation_detail)
            next_op_names = "|".join(
                [self.operator_info_graph[i]['op_name'] for i in next_op_graph_index])
            next_op_is_not_last_op = (False or sum(
                [n < len(self.operator_info_graph) for n in next_op_graph_index]))

            # check here if any of the next op is in the cpu operations list and call the respective handler for updating the attributes here
            next_op_is_cpu_op = [self.operator_info_graph[i]['op_code']
                                 in cpu_operator_options.cpu_operators_list for i in next_op_graph_index]
            if any(next_op_is_cpu_op):
                # get the op index of the cpu op
                next_op_index = next_op_graph_index[next(
                    (i for i, val in enumerate(next_op_graph_index) if val), None)]
                cpu_op_code = self.operator_info_graph[next_op_index]['op_code']
                ret = cpu_operator_options.HandleOperatorAttributesBeforeCpuOp(
                    self, cpu_op_code)
                # and then calculate the scale shifts and the multipliers
                if ret < 0:
                    if ret == -2:
                        raise KeyError(-921)
                    raise KeyError(-920)
                self.scale_shifts, self.scale_multipliers, error = util.optimize_scaling_shift_per_channel(
                    self.ip_q['scales'], self.op_q['scales'], self.w_q['scales'], self.op_q_zeropoint, max_scale=self.scaleshift_max_range)
            else:
                if next_op_is_not_last_op and ("LEAKY_RELU" in next_op_names):
                    self.LEAKY_RELU_FLAG = True
                    if (not self.SetLeakyReluAsActivationFunction()):
                        raise KeyError(-909)
                    leaky_relu_tflite_operator = self.operator_info_graph[op_index +
                                                                          1]['tflite_operator']
                    leaky_relu_builtin_option = leaky_relu_tflite_operator.BuiltinOptions()
                    leaky_relu_bytes = leaky_relu_builtin_option.Bytes
                    leaky_relu_pos = leaky_relu_builtin_option.Pos
                    leaky_relu_options = tflite.LeakyReluOptions()
                    leaky_relu_options.Init(leaky_relu_bytes, leaky_relu_pos)
                    self.leaky_relu_alpha = leaky_relu_options.Alpha()
                    leaky_relu_op_q = self.interpreter.get_tensor_details(
                    )[leaky_relu_tflite_operator.OutputsAsNumpy()[0]]['quantization_parameters']
                    leaky_relu_op_scale = leaky_relu_op_q['scales']
                    leaky_relu_op_zeropoint = leaky_relu_op_q['zero_points']
                    scale_shift, scale_q, error = util.optimize_scaling_shift_per_channel(
                        self.ip_q['scales'], leaky_relu_op_scale, self.w_q['scales'], leaky_relu_op_zeropoint)
                    # combine both the scale shifts value into one single array
                    scale_q_negative = np.int32(self.leaky_relu_alpha*scale_q)
                    self.scale_multipliers = np.array(
                        list(map(list, zip(scale_q_negative, scale_q)))).reshape(-1)
                    # self.scale_shifts = np.array(
                    #     list(map(list, zip(scale_shift)))).reshape(-1)
                    self.scale_shifts = scale_shift
                    self.op_q['scales'] = leaky_relu_op_scale
                    self.op_q_zeropoint = leaky_relu_op_zeropoint
                # elif next_op_is_not_last_op and ("SOFTMAX" in next_op_names) and not self.skip_softmax_op:
                #     # set the activation function to be custom function
                #     if (not self.SetCustomActivationFunctionType("CustomPrepareSoftmax")):
                #         raise KeyError(-902)
                #     # we have to calulate the maximum possible bitlimit we can shift to, so that we avoid overflow
                #     # the maximum output value we expect is
                #     op_max = np.ceil(self.op_q['scales']
                #                     [0] * (127 - self.op_q_zeropoint[0]))
                #     op_min = np.ceil(
                #         self.op_q['scales'][0] * ((-128) - self.op_q_zeropoint[0]))
                #     # print(f"op_max {op_max}, op_min {op_min}")
                #     self.op_q['scales'] = np.array([1])
                #     self.op_q_zeropoint[0] = np.array([0])
                #     beta = 1  # FIXME get the beta value from the softmax operation somehow, which is the next operation and yet to be encountered
                #     self.ip_q['scales'] *= beta
                #     self.scaleshift_max_range = 31 - \
                #         (len(bin(int(max(abs(op_max), abs(op_min))))) - 2)
                #     self.scale_shifts, self.scale_multipliers, error = util.optimize_scaling_shift_per_channel(
                #         self.ip_q['scales'], self.op_q['scales'], self.w_q['scales'], self.op_q_zeropoint[0], max_scale=self.scaleshift_max_range)
                #     self.op_bitwidth = np.int32
                # elif next_op_is_not_last_op and (("LOGISTIC" in next_op_names) or ("TANH" in next_op_names)):
                #     """
                #     need to use the following paradigm for cpu operators and get rid of this handling in further development
                #     # cpu_operator_options.HandleOperatorAttributesBeforeCpuOp(self,self.operator_info_graph[op_index+1]["op_code"])
                #     """

                #     # set the activation function to be custom function
                #     if (not self.SetCustomActivationFunctionType("None")):
                #         raise KeyError(-912)
                #     self.op_q['scales'] = np.array([1])
                #     self.op_q_zeropoint[0] = np.array([0])
                #     self.scaleshift_max_range = 28
                #     self.scale_shifts, self.scale_multipliers, error = util.optimize_scaling_shift_per_channel(
                #         self.ip_q['scales'], self.op_q['scales'], self.w_q['scales'], self.op_q_zeropoint[0], max_scale=self.scaleshift_max_range, bit_limit=16)
                #     self.op_bitwidth = np.int16
                #     self.layer_output_radix = 12
                else:
                    self.scale_shifts, self.scale_multipliers, error = util.optimize_scaling_shift_per_channel(
                        self.ip_q['scales'], self.op_q['scales'], self.w_q['scales'], self.op_q_zeropoint)

            # if(self.normalize_scaleshifts):#checking for the scaling error when we provide only one scaling value for all the channels
            #   scale_old = self.ip_q['scales']*self.w_q['scales'] / self.op_q['scales']

            #   scale = (self.scale_multipliers / (2**self.scale_shifts.astype(np.float32))).astype(np.float32)
            #   # if(self.LEAKY_RELU_FLAG):
            #   #   # scale_ngtve = ip_scales*w_q['scales'] / leaky_relu_alpha_scale
            #   #   scale_ngtve = scale * self.leaky_relu_alpha
            #   #   scale = np.array(list(map(list, zip(scale_ngtve,scale)))).reshape(-1)
            #   min_error_ndx = np.argmin(self.scale_shifts)
            #   single_scaleshift = self.scale_shifts[min_error_ndx]
            #   self.scale_multipliers = abs(np.round(scale*2**single_scaleshift)).astype(np.int32)
            #   errors_ = util.scale_error(scale,single_scaleshift)
            #   min_error_ndx = np.argmax(errors_)
            #   self.scale_shifts = np.array([single_scaleshift],dtype=np.int8)
            # if self.layer_output_radix: #if a non zero output radix is calculated for the scales, need to subtract that from the scaleshifts, currently only applicable to LOGISTICS
            #   if self.layer_output_radix<min(self.scale_shifts):
            #     self.scale_shifts -= self.layer_output_radix
            #   else:
            #     raise Exception(f"The output radix {self.layer_output_radix} cannot be acheived with the current scaling values")

    def GetOperationName(self):
        return self.operator_name

    def GetStridesInfo(self):
        self.stride_x = self.option.StrideW()
        self.stride_y = self.option.StrideH()
        # self.option.StrideW(), self.option.StrideH()
        return self.stride_x, self.stride_y

    def GetActivationFunctionType(self):
        if self.option is None:
            return "None"
        if (self.custom_activation):  # custom activation function set
            return self.activation
        try:
            if self.option.FusedActivationFunction() == tflite.ActivationFunctionType.NONE:
                self.activation = "None"
            elif self.option.FusedActivationFunction() == tflite.ActivationFunctionType.RELU:
                self.activation = "ReLU"
            elif self.option.FusedActivationFunction() == tflite.ActivationFunctionType.RELU6:
                self.activation = "ReLU6"  # handle it
            elif self.option.FusedActivationFunction() == tflite.ActivationFunctionType.TANH:
                self.activation = "Tanh"
        except Exception:  # FIXME add better handling of errors if acivation function is not present
            return "None"
        return self.activation

    def SetCustomActivationFunctionType(self, activation_name):
        if (self.activation == "None"):
            self.custom_activation = True
            self.activation = activation_name
        return self.custom_activation

    def SetLeakyReluAsActivationFunction(self):
        return self.SetCustomActivationFunctionType("LeakyReLU")

    def GetPaddingType(self):
        if self.option.Padding() == tflite.Padding.SAME:
            self.padding = 'SAME'
        elif self.option.Padding() == tflite.Padding.VALID:
            self.padding = 'VALID'
        return self.padding

    def GetInputTensorsNdx(self):
        return self.operator.InputsAsNumpy()

    def GetOutputTensorsNdx(self):
        return self.operator.OutputsAsNumpy()

    def GetOpQuantizationParameters(self):
        return self.op_q

    def GetIpQuantizationParameters(self):
        return self.ip_q, self.w_q, self.bias_q

    def CalculatePadding(self):
        self.padding = self.GetPaddingType()
        if (self.padding == 'SAME'):
            # out_height = np.ceil( self.ip_shape.height / self.option.StrideH())
            # out_width = np.ceil( self.ip_shape.width / self.option.StrideW())

            if (self.ip_shape.height % self.option.StrideH() == 0):
                pad_along_height = max(
                    self.kernel_shape.height - self.option.StrideH(), 0)
            else:
                pad_along_height = max(
                    self.kernel_shape.height - (self.ip_shape.height % self.option.StrideH()), 0)
            if (self.ip_shape.width % self.option.StrideW() == 0):
                pad_along_width = max(
                    self.kernel_shape.width - self.option.StrideW(), 0)
            else:
                pad_along_width = max(
                    self.kernel_shape.width - (self.ip_shape.width % self.option.StrideW()), 0)

            self.pad_info.pad_top = pad_along_height // 2
            self.pad_info.pad_bottom = pad_along_height - self.pad_info.pad_top
            self.pad_info.pad_left = pad_along_width // 2
            self.pad_info.pad_right = pad_along_width - self.pad_info.pad_left
        elif (self.padding == 'VALID'):
            self.pad_info = PadDetails()
        return self.pad_info

    def GetInputShapes(self):
        return self.ip_shape.get_shape(), self.kernel_shape.get_shape(), self.bias_shape.get_shape()

    def GetOutputShape(self):
        return self.op_shape.get_shape()

    def CalculateBPrime(self):
        """
        We get the weight or the filter tensor here for the conv2d operation,
        we have to reshape the tensor so that we are able to get the right filter size
        and calculate the right value for the B'
        to get the right dimensions, we should get the batch, height, width and depth or features of the filter
        then reshape it accordingly so that we get the right bias prime calculation
        self.ip_q_zeropoint[0], self.filter_tensor, self.bias_tensor
        """
        if self.bias_tensor.size != 0:
            kernel_tensor = self.filter_tensor

            if (len(kernel_tensor.shape) == 4):
                # kernel_tensor = self.filter_tensor.transpose(3,0,1,2) # transposing to get the batch, depth, height, width order
                kernel_tensor = self.filter_tensor.transpose(0, 3, 1, 2)
                # squeeze the tensor, so that we get rid of single dimension vectors
                kernel_tensor = kernel_tensor.squeeze()
                if kernel_tensor.size == 1:  # this was a single value kernel
                    kernel_tensor = np.array([kernel_tensor])
            # now we have squeezed the tensor, we need to get the B' from the filter, while ensuring that we get it through the channels
            x = [-np.sum(kernel_tensor[j].astype(int)*self.ip_q_zeropoint[0])
                 for j in range(kernel_tensor.shape[0])]
            # elif(len(kernel_tensor.shape)==2):#this is one where it has no channel information
            #   x = [-np.sum(kernel_tensor.astype(int)*izp)]
            b_prime = self.bias_tensor.astype(int) + x
            self.b_prime_tensor = b_prime.astype(np.int32)
        else:
            self.b_prime_tensor = np.array([])

    def GetOperandNameString(self) -> str: return self.operand_str

    def GetOptionsError(self):
        return self.error, self.error_text, self.error_action

    def GetIpOpBitwidth(self):
        return self.ip_bitwidth, self.op_bitwidth

    def SetIpBitwidth(self, ip_bw=np.int8):
        self.ip_bitwidth = ip_bw

    def SetOpBitwidth(self, op_bw=np.int8):
        self.op_bitwidth = op_bw

    def GetApiDefinition(self): return self.api_defn

    def WriteWeightTensorToFile(self, file_string, info_string, tensor, get_full_content=False):
        kernel_tensor = tensor
        if (len(tensor.shape) == 4):
            # if(self.operator_name=="POINTWISE_CONV_2D"):
            #   kernel_tensor = tensor.transpose(0,3,1,2)
            # else:
            kernel_tensor = tensor.transpose(0, 3, 1, 2)
        # file_content += "\nconst int8_t "+array_name_str.lower()+"_weights[]["+array_name_str+"_FILTER_CHANNEL]["+array_name_str+"_FILTER_HEIGHT]["+array_name_str+"_FILTER_WIDTH]= \n" #{ { { { } } } };\n"
        file_string += "\nconst int8_t "+info_string.lower()+"_"+self.GetOperandNameString().lower() + \
            "["+info_string+"_FILTER_OUTPUT_CHANNEL_CNT]["+info_string+"_FILTER_INPUT_CHANNEL_CNT][" + \
            info_string+"_FILTER_HEIGHT]["+info_string + \
            "_FILTER_WIDTH]={\n"  # { { { { } } } };\n"
        if get_full_content:
            for output_channel in range(kernel_tensor.shape[0]):
                file_string += "{"
                for input_channel in range(kernel_tensor.shape[1]):
                    file_string += np.array2string(kernel_tensor[output_channel][input_channel], separator=',',
                                                   max_line_width=1000, threshold=np.inf).replace('[', '{').replace(']', '}').replace("\n", "")
                    file_string += ","
                file_string = file_string[:-1] + "},"
            # file_string +="},"
        else:
            file_string += np.array2string(kernel_tensor, separator=',', max_line_width=1000,
                                           threshold=np.inf).replace('[', '{').replace(']', '}').replace("\n", "")
        file_string = file_string[:-1] + "};"
        return file_string

    def WriteBPrimeTensorToFile(self, file_string, info_string, tensor):
        file_string += "\nconst int32_t "+info_string.lower()+"_bias_prime[] = "
        file_string += np.array2string(tensor, separator=',', max_line_width=1000,
                                       threshold=np.inf).replace('[', '{').replace(']', '}')
        file_string += ";"
        return file_string

    def WriteScaleShiftsToFile(self, file_string, info_string, scale_q, scale_shift):
        array_name = info_string[9:]
        self.scale_multipliers = scale_q
        self.scale_shift = scale_shift
        if (self.activation == "CustomPrepareSoftmax"):
            scale_shift -= 12  # PrepareSoftmaxActivation generates a q.12 output
        if (scale_q.size == 0):
            # file_string += "\nconst int32_t *"+array_name.lower()+"_scale_q = NULL"
            file_string += info_string.lower()+"_scale_q NULL"
        else:
            file_string += "\nconst int32_t "+array_name.lower()+"_scale_q[] = "
            file_string += np.array2string(scale_q, separator=',', max_line_width=1000,
                                           threshold=np.inf).replace('[', '{').replace(']', '}')
            file_string += ";"
        file_string += info_string + "_SCALESHIFT_COUNT "+str(scale_shift.size)
        if (scale_shift.size == 0):
            # file_string += "\nconst int8_t *"+array_name.lower()+"_scaleshifts = NULL"
            file_string += info_string.lower()+"_scaleshifts NULL"
        else:
            file_string += "\nconst int8_t "+array_name.lower()+"_scaleshifts[] = "
            file_string += np.array2string(scale_shift, separator=',', max_line_width=1000,
                                           threshold=np.inf).replace('[', '{').replace(']', '}')
            file_string += ";"

        return file_string

    def WriteTestVectorToFile(self, file_string, info_string, raw_test_vector, op_bitwidth=np.int8, op_radix=0):
        array_name = info_string[9:]
        raw_test_vector = (raw_test_vector).astype(op_bitwidth)
        if (op_bitwidth != np.int8):
            # deaquantize the output
            raw_test_vector = (raw_test_vector - self.op_q['zero_points'][0])
            raw_test_vector = (raw_test_vector *
                               self.op_q['scales'][0]).astype(np.float64)
        if (self.activation == "CustomPrepareSoftmax"):
            """
            when the activation function is CustomPrepareSoftmax, the expected output is dequantized and is Q.12
            """
            # shifting it to give an output of Q.12
            raw_test_vector = raw_test_vector*(2**12)
            # DEBUG# calculate the scales to calculate the o_prime value for the test vectors
            # op_scale = (self.ip_q['scales']*self.w_q['scales']) / self.op_q['scales']
            # op_zp = self.op_q['zero_points'][0]
            # raw_test_vector = ((raw_test_vector - op_zp) / op_scale)
            # raw_test_vector = (raw_test_vector * self.scale_multipliers)
            # raw_test_vector = raw_test_vector / (2**self.scale_shift)
        if (op_radix):
            raw_test_vector = raw_test_vector * (2**op_radix)
        raw_test_vector_shape_size = len(raw_test_vector.shape)
        if (raw_test_vector_shape_size == 3):
            raw_test_vector = raw_test_vector.transpose(2, 0, 1).squeeze()
        elif (raw_test_vector_shape_size == 4):
            raw_test_vector = raw_test_vector.transpose(0, 3, 1, 2).squeeze()
        else:
            raw_test_vector = raw_test_vector.squeeze()
        file_string += util.write_array_to_file(
            raw_test_vector, array_name.upper()+"_tflite_op", array_bitwidth=op_bitwidth)
        return file_string

    def WriteTensorToBinFile(self, bin_file, tensor):
        if (tensor.size == 0):
            return bin_file, -1
        if (len(tensor.shape) == 4):
            tensor = tensor.transpose(0, 3, 1, 2)
        elif (len(tensor.shape) == 3):
            tensor = tensor.transpose(2, 0, 1)
        bin_file, offset = util.write_array_to_bin(
            tensor, bin_file, tensor.dtype)
        return bin_file, offset

    def GetFilterBytewidthEnum(self):
        return self.kernel_bytewidth_enum

    def GetOperationStrides(self):
        return self.stride_x, self.stride_y

    def SetOperationStrides(self, stride_x, stride_y):
        self.stride_x, self.stride_y = stride_x, stride_y

    def GetOperationPaddings(self):
        return self.pad_info

    def SetOperationPaddings(self, top, bottom, left, right):
        self.pad_info.pad_top = top
        self.pad_info.pad_bottom = bottom
        self.pad_info.pad_left = left
        self.pad_info.pad_right = right

    def GetOperatorInputShape(self):
        if (self.ops_ip_shape is None):
            return self.ip_shape.get_shape()
        return self.ops_ip_shape.get_shape()

    def PrintAttributes(self):
        return self.meta_data_str

    def GetCommandBufferLen(self, normalized_scaling=True):
        return self.command_buf_len

    def WriteOperatorAttributesToFile(self, file_string, info_string, last_op_name):

        # file_string += info_string +"_INPUT_BATCH "+str(self.ip_shape.batch)
        file_string += info_string + \
            "_INPUT_CHANNEL_CNT "+str(self.ip_shape.depth)
        file_string += info_string + "_INPUT_HEIGHT "+str(self.ip_shape.height)
        file_string += info_string + "_INPUT_WIDTH "+str(self.ip_shape.width)

        file_string += info_string + "_FILTER_OUTPUT_CHANNEL_CNT " + \
            str(self.kernel_shape.depth)+" //NOT_REQUIRED"
        file_string += info_string + "_FILTER_HEIGHT " + \
            str(self.kernel_shape.height)+" //NOT_REQUIRED"
        file_string += info_string + "_FILTER_WIDTH " + \
            str(self.kernel_shape.height)+" //NOT_REQUIRED"
        file_string += info_string + "_FILTER_BYTEWIDTH " + \
            str(self.kernel_bytewidth_enum.name)+" //NOT_REQUIRED"

        file_string += info_string + "_STRIDE_W " + \
            str(self.stride_x)+" //NOT_REQUIRED"
        file_string += info_string + "_STRIDE_H " + \
            str(self.stride_y)+" //NOT_REQUIRED"

        file_string += info_string + "_PADDING_TOP " + \
            str(self.pad_info.pad_top)+" //NOT_REQUIRED"
        file_string += info_string + "_PADDING_BOTTOM " + \
            str(self.pad_info.pad_bottom)+" //NOT_REQUIRED"
        file_string += info_string + "_PADDING_LEFT " + \
            str(self.pad_info.pad_left)+" //NOT_REQUIRED"
        file_string += info_string + "_PADDING_RIGHT " + \
            str(self.pad_info.pad_right)+" //NOT_REQUIRED"

        # file_string += info_string +"_OUTPUT_BATCH "+str(self.op_shape.batch)
        file_string += info_string + \
            "_OUTPUT_CHANNEL_CNT "+str(self.op_shape.depth)
        file_string += info_string + \
            "_OUTPUT_HEIGHT "+str(self.op_shape.height)
        file_string += info_string + "_OUTPUT_WIDTH "+str(self.op_shape.width)
        # file_string += info_string +"_OUTPUT_AXON_STRIDE "+str(util.GetAxonproStrideWidth(self.op_shape.height))

        return file_string

    def GetCpuOpAdditionalAttributes(self):
        return np.array(self.cpu_op_attributes)

    def SetOperatorOptionObject(self, options_object):
        self.option = options_object
        self.option.Init(self.bytes, self.pos)
        return self.option

    def SetOperatorMetaAttributesString(self, meta_data_string):
        self.meta_data_str = meta_data_string

    def GetMultiplierandScaleshift(self):
        if not self.custom_cpu_op:
            self.CalculateMultiplierandScaleshift()
        # checking for the scaling error when we provide only one scaling value for all the channels
        if (self.normalize_scaleshifts):
            self.NormalizeScaleshifts()
        if (self.layer_output_radix):  # if a non zero output radix is calculated for the scales, need to subtract that from the scaleshifts, currently only applicable to LOGISTICS
            self.HandleOutputRadix()
        return self.scale_multipliers, self.scale_shifts

    def GetFilterTensor(self):
        # if self.transpose_kernel:
        #     # transpose the filter here
        #     filter_shape = TensorShape(np.array(self.filter_tensor.shape))
        #     if filter_shape.shape_size == 4:
        #         return self.filter_tensor.transpose(0, 2, 1, 3)
        #     elif filter_shape.shape_size == 2:
        #         return self.filter_tensor.transpose(1, 0)
        return self.transpose_tensor_if_needed(self.filter_tensor, self.transpose_kernel)

    def GetBPrimeTensor(self):
        if not self.custom_cpu_op:
            self.CalculateBPrime()
        return self.b_prime_tensor

    def SetFilterTensor(self, filter_tensor):
        self.filter_tensor = filter_tensor

    def SetBPrimeTensor(self, bprime_tensor):
        self.b_prime_tensor = bprime_tensor

    def SetMultiplierandScaleshift(self, scale_multipliers, scale_shifts):
        self.scale_multipliers = scale_multipliers
        self.scale_shifts = scale_shifts

    def GetTfliteInterpreter(self):
        return self.tflite_interpreter

    def GetInputTensors(self):
        ip_ndxs = self.GetInputTensorsNdx()
        ip_tensor_list = {}
        for n, ndx in enumerate(ip_ndxs):
            tensor = self.interpreter.get_tensor(ndx)
            ip_tensor_list[n] = tensor

        return ip_tensor_list

    def FillAdditionalCpuAttributes(self, additional_attrib_list):
        self.cpu_op_attributes = np.array(
            additional_attrib_list, dtype=np.int32)

    def GetCpuAdditionalAttributesTensor(self):
        return self.cpu_op_attributes

    def SetIpQZeropoint(self, ip_zeropoint):
        self.ip_q_zeropoint[0] = ip_zeropoint

    def SetOpQZeropoint(self, op_zeropoint):
        self.op_q_zeropoint[0] = op_zeropoint

    def GetIpOpZeropoints(self):
        return self.ip_q_zeropoint, self.op_q_zeropoint

    def SetOperatorInputShape(self, height, width, channel, batch):
        self.ip_shape.batch = batch
        self.ip_shape.width = width
        self.ip_shape.height = height
        self.ip_shape.depth = channel

    def SetOperatorOutputShape(self, height, width, channel, batch):
        self.op_shape.batch = batch
        self.op_shape.width = width
        self.op_shape.height = height
        self.op_shape.depth = channel

    def GetOperatorInputShapeSize(self):
        return self.ip_shape.shape_size

    def GetOperationDilation(self):
        return self.dilation_x, self.dilation_y

    def SetOperationDilation(self, dilation_x, dilation_y):
        self.dilation_x, self.dilation_y = dilation_x, dilation_y

    def SetNormalizedScaleshiftsFlag(self, normalize_scaleshift):
        self.normalize_scaleshifts = normalize_scaleshift

    def SetSkipSoftmaxOpFlag(self, skip_softmax_op):
        self.skip_softmax_op = skip_softmax_op

    def GetSkipSoftmaxOpFlag(self):
        return self.skip_softmax_op

    def SetOpQuantizationDisableFlag(self, disable_op_q):
        self.disable_op_quantization = disable_op_q

    def SetOpQScale(self, op_scale):
        self.op_q['scales'][0] = op_scale

    def SetIpQScale(self, ip_scale):
        self.ip_q['scales'][0] = ip_scale

    def SetTransposeKernelFlag(self, transpose_kernel):
        self.transpose_kernel = transpose_kernel
        return self.transpose_kernel

    def GetAxonsOperationEnumName(self):
        return self.axons_operation_enum

    def SetAxonsOperationEnumName(self, axons_op_enum_string):
        self.axons_operation_enum = axons_op_enum_string

    def SetScaleshiftMaxRange(self, scaleshift_max_range):
        self.scaleshift_max_range = scaleshift_max_range

    def SetLayerOutputRadix(self, layer_output_radix):
        self.layer_output_radix = layer_output_radix


class ConvolutionOptions(OperatorOptions):

    conv1d_filter_constraints = {
        "H": 16,
        "W": 32
    }

    @staticmethod
    def is_1d_conv(input_shape, output_shape, filter_shape, padding, strides):
        _, in_h, in_w, _ = input_shape
        _, out_h, out_w, _ = output_shape
        _, k_h, k_w, _ = filter_shape
        s_h, s_w = strides

        if k_h == 1 and k_w == 1:
            return False

        is_h_1d = (in_h == k_h and k_h == s_h and s_h == 1 and ((padding == tflite.Padding.SAME and out_w == in_w) or (
            padding == tflite.Padding.VALID and (in_w-k_w >= (out_w - 1) * s_w))))

        is_w_1d = (in_w == k_w and k_w == s_w and s_w == 1 and ((padding == tflite.Padding.SAME and out_h == in_h) or (
            padding == tflite.Padding.VALID and (in_h-k_h >= (out_h - 1) * s_h))))

        if (is_h_1d and not is_w_1d) or (is_w_1d and not is_h_1d):
            return True
        return False

    def get_conv_operator_tflite_options(cls):
        raise NotImplementedError(
            "get_conv_operator_tflite_options method not implemented!")

    @classmethod
    def check_operator_specific_constraints(cls, built_in_options, input_tensors, output_shapes, transpose_check=False):
        io_error_txt = ""
        io_error_txt += cls.check_default_constraints(
            built_in_options, input_tensors, output_shapes, transpose_check)
        # check for filter dimension constraints
        input_shape = TensorShape(input_tensors[0].shape)
        output_shape = TensorShape(output_shapes[0])
        filter_shape = TensorShape(input_tensors[1].shape)
        options = cls.get_conv_operator_tflite_options()
        options.Init(built_in_options.Bytes, built_in_options.Pos)
        strides = (options.StrideH(), options.StrideW())
        padding = options.Padding()
        filter_constraint = cls.default_filter_constraints

        if cls.is_1d_conv(input_shape.get_shape(), output_shape.get_shape(), filter_shape.get_shape(), padding, strides):
            filter_constraint = cls.conv1d_filter_constraints

        filter_error_text = cls.check_filter_constraints(
            filter_shape.height, filter_shape.width, filter_constraint, transpose_check=transpose_check)
        return io_error_txt + filter_error_text


class Conv2dOperatorOptions(ConvolutionOptions):
    multi_channel_input_convolution = False
    pw_horizontal_stripe = 1
    previous_pad = False

    @classmethod
    def get_conv_operator_tflite_options(cls):
        return tflite.Conv2DOptions()

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.axons_operation_enum = "NRF_AXON_NN_OP_CONV2D"
        self.operand_str = "filters"
        self.operator_name = self.operation_detail["op_name"]
        self.cmd_buf_len_multiplier = 3
        self.option = tflite.Conv2DOptions()
        self.option.Init(self.bytes, self.pos)
        if (self.kernel_shape.height == 1) and (self.kernel_shape.width == 1) and (self.operator_name == "CONV_2D"):
            self.operator_name = "POINTWISE_CONV_2D"
            self.axons_operation_enum = "NRF_AXON_NN_OP_POINTWISE_CONV2D"
            # self.api_defn = open(template_dir + "define_pointwise_layer_c_template.txt").read()
            """
      horizontal_stripe_cnt = CEILING((1+(pw_input_height-1)/pw_y_stride)/( (2048 *3)/(CEILING(pw_input_width,4) * CEILING(pw_input_channel_cnt,6))),1)
      """
            # self.pw_horizontal_stripe = util.excel_ceil((1+(self.ip_shape.height-1)/self.option.StrideH())/util.excel_floor(  (2048 *3)/(util.excel_ceil(self.ip_shape.width,4) * self.ip_shape.depth ),6),1)
            self.pw_horizontal_stripe = util.excel_ceil((1+(self.ip_shape.height-1)/self.option.StrideH())/(
                (2048 * 3)/(util.excel_ceil(self.ip_shape.width, 4) * util.excel_ceil(self.ip_shape.depth, 6))), 1)
        else:  # it is a convolution layer, check if it has multiple input
            if (self.ip_shape.depth > 1):
                self.multi_channel_input_convolution = True
        # FIXME now we have to support Convolution with dilation factor so removing this error check!
        # self.error = not(self.option.DilationHFactor()==1) and (self.option.DilationWFactor()==1)
        self.dilation_y = self.option.DilationHFactor()
        self.dilation_x = self.option.DilationWFactor()
        self.CalculatePadding()
        # figure out if the previous operation is a PAD operation and if yes try to update the padding information accordingly
        if operator_graph is not None:
            current_op_ndx = operation_detail['index']
            operator_graph_info = operator_graph[operation_detail['index']]
            previous_op_index = SupportedOperators.get_index_from_tf_index(
                operator_graph, operator_graph_info['index']-1)
            if previous_op_index > 0:  # not the first operation
                previous_valid_axon_op_ndx = self.get_previous_valid_axon_operator_index(
                    operator_graph, current_op_ndx)
                if previous_valid_axon_op_ndx is not None:
                    if self.ip_shape != operator_graph[previous_valid_axon_op_ndx]['operator_options'].op_shape:
                        # which means we might have to find out why is there a change in shape?
                        # because of a reshape or some other operator
                        for i in range(previous_op_index, previous_valid_axon_op_ndx, -1):
                            # find if PAD is present, just once
                            if operator_graph[i]['op_name'] == "PAD":
                                previous_pad_op = operator_graph[i]
                                # get the padding details
                                self.ops_ip_shape = TensorShape(np.array(
                                    previous_pad_op['operator_options'].GetInputShapes()[0]))  # only the input shape
                                self.pad_info = copy.deepcopy(
                                    previous_pad_op['operator_options'].GetOperationPaddings())
                                self.previous_pad = True
                                break
                            # previous_op = operator_graph[previous_op_index]
                            # if previous_op['op_name']=="PAD": #straight up check if there is a pad in the previous operator
                            #   #get the padding details
                            #   self.ops_ip_shape = TensorShape(np.array(previous_op['operator_options'].GetInputShapes()[0])) #only the input shape
                            #   self.pad_info = copy.deepcopy(previous_op['operator_options'].GetOperationPaddings())
                            #   self.previous_pad=True
                            #   #check if the previous op output shape matches with the current ip shape

        # get the filter and bias tensor
        self.filter_tensor = tflite_interpreter.get_tensor(self.ip_ndxs[1])
        self.bias_tensor = tflite_interpreter.get_tensor(self.ip_ndxs[2])

    def PrintAttributes(self):
        self.meta_data = f"padding:{self.padding}, (stride_w, stride_h):{self.option.StrideW(), self.option.StrideH()}, activation function:{self.GetActivationFunctionType()}, (dilation_h, dilation_w):{self.option.DilationHFactor(), self.option.DilationWFactor()}"
        # print(self.meta_data)
        return self.meta_data

    def WriteOperatorAttributesToFile(self, file_string, info_string, last_ops_options):
        padding = self.GetOperationPaddings()
        if self.previous_pad:
            # get the define for the current line from the info string
            pad_operation_define = info_string[9:]
            pad_operation_define = pad_operation_define[:-(
                len(self.operator_name))]
            pad_operation_define += "PAD"
            # special case handling of the fomo face detection model with PAD layer
            # file_string += info_string +"_INPUT_BATCH "+str(self.ip_shape.batch)
            file_string += info_string + \
                "_INPUT_CHANNEL_CNT ("+pad_operation_define + \
                "_INPUT_CHANNEL_CNT)"
            file_string += info_string + \
                "_INPUT_HEIGHT ("+pad_operation_define+"_INPUT_HEIGHT)"
            file_string += info_string + \
                "_INPUT_WIDTH ("+pad_operation_define+"_INPUT_WIDTH)"

            file_string += info_string + \
                "_PADDING_TOP ("+str(padding.pad_top) + \
                "+("+pad_operation_define+"_PADDING_TOP))"
            file_string += info_string + \
                "_PADDING_BOTTOM ("+str(padding.pad_bottom) + \
                "+("+pad_operation_define+"_PADDING_BOTTOM))"
            file_string += info_string + \
                "_PADDING_LEFT ("+str(padding.pad_left) + \
                "+("+pad_operation_define+"_PADDING_LEFT))"
            file_string += info_string + \
                "_PADDING_RIGHT ("+str(padding.pad_right) + \
                "+("+pad_operation_define+"_PADDING_RIGHT))"
            # self.ops_ip_shape = TensorShape(np.array(last_ops_options.GetInputShapes()[0])) #only the input shape
            # self.pad_info = last_ops_options.GetOperationPaddings()
        else:
            # file_string += info_string +"_INPUT_BATCH "+str(self.ip_shape.batch)
            file_string += info_string + \
                "_INPUT_CHANNEL_CNT "+str(self.ip_shape.depth)
            file_string += info_string + \
                "_INPUT_HEIGHT "+str(self.ip_shape.height)
            file_string += info_string + \
                "_INPUT_WIDTH "+str(self.ip_shape.width)
            file_string += info_string + "_PADDING_TOP "+str(padding.pad_top)
            file_string += info_string + \
                "_PADDING_BOTTOM "+str(padding.pad_bottom)
            file_string += info_string + "_PADDING_LEFT "+str(padding.pad_left)
            file_string += info_string + \
                "_PADDING_RIGHT "+str(padding.pad_right)

        stride_h, stride_w = self.GetStridesInfo()
        file_string += info_string + "_STRIDE_W "+str(stride_w)
        file_string += info_string + "_STRIDE_H "+str(stride_h)
        file_string += info_string + "_FILTER_INPUT_CHANNEL_CNT " + \
            str(self.kernel_shape.depth)
        file_string += info_string + "_FILTER_OUTPUT_CHANNEL_CNT " + \
            str(self.kernel_shape.batch)
        file_string += info_string + "_FILTER_HEIGHT " + \
            str(self.kernel_shape.height)
        file_string += info_string + "_FILTER_WIDTH " + \
            str(self.kernel_shape.width)
        file_string += info_string + "_FILTER_BYTEWIDTH " + self.kernel_bytewidth_enum.name

        # file_string += info_string +"_OUTPUT_BATCH "+str(self.op_shape.batch)
        file_string += info_string + \
            "_OUTPUT_CHANNEL_CNT "+str(self.op_shape.depth)
        file_string += info_string + \
            "_OUTPUT_HEIGHT "+str(self.op_shape.height)
        file_string += info_string + "_OUTPUT_WIDTH "+str(self.op_shape.width)
        # file_string += info_string +"_OUTPUT_AXON_STRIDE "+str(util.GetAxonproStrideWidth(self.op_shape.width))
        return file_string

    def GetAttributes(self):
        d = {'name': self.name,
             'padding': self.padding,
             'stride_w': self.option.StrideW(),
             'stride_h': self.option.StrideH(),
             'activation': self.activation,
             'dilation_w': self.option.DilationWFactor(),
             'dilation_h': self.option.DilationHFactor()}
        return d

    def GetCommandBufferLen(self, normalized_scaling=True):
        """
        multi input-channel norm scaling
        cmd_buffer_len = 39 + // overhead
                          6 * (input_channel_cnt - 2) + // per input_channel, setting up pointers
                          17 +  // last input channeld overhead, set up LUT, rounding, output
                            3 * output_channel_cnt // per output channel bias and multiplier

        """
        if (not normalized_scaling):
            self.cmd_buf_len_multiplier += 3

        if (self.operator_name == "POINTWISE_CONV_2D"):
            self.command_buf_len = 29 + \
                ((self.cmd_buf_len_multiplier * self.op_shape.depth) + 10) * \
                self.pw_horizontal_stripe
        elif (self.multi_channel_input_convolution):
            self.command_buf_len = 39 + 6 * \
                (self.ip_shape.depth - 2) + 17 + \
                self.cmd_buf_len_multiplier*(self.op_shape.depth)
        else:
            self.command_buf_len = (
                39 + self.cmd_buf_len_multiplier*self.op_shape.depth)

        # print(f"command buff len : {self.command_buf_len}")
        return self.command_buf_len


class DepthwiseConv2DOperatorOptions(ConvolutionOptions):
    previous_pad = False

    @classmethod
    def get_conv_operator_tflite_options(cls):
        return tflite.DepthwiseConv2DOptions()

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.operator_name = self.operation_detail["op_name"]
        self.axons_operation_enum = "NRF_AXON_NN_OP_DEPTHWISE_CONV2D"
        self.operand_str = "filters"
        # self.api_defn = open(template_dir + "define_depthwise_layer_c_template.txt").read()
        self.cmd_buf_len_multiplier = 3
        self.interpreter = tflite_interpreter
        self.option = tflite.DepthwiseConv2DOptions()
        self.option.Init(self.bytes, self.pos)
        self.dilation_y = self.option.DilationHFactor()
        self.dilation_x = self.option.DilationWFactor()
        self.error = not ((self.option.DepthMultiplier() == 1))
        if self.error:
            self.error_text += f"Depth Multiplier value of 1 is supported, {self.operator_name} has DepthMultiplier={self.option.DepthMultiplier()}"
            self.error_action = "ERROR"
        self.CalculatePadding()
        # figure out if the previous operation is a PAD operation and if yes try to update the padding information accordingly
        if operator_graph is not None:
            operator_graph_info = operator_graph[operation_detail['index']]
            previous_op_index = SupportedOperators.get_index_from_tf_index(
                operator_graph, operator_graph_info['index']-1)
            if previous_op_index > 0:  # not the first operation
                previous_op = operator_graph[previous_op_index]
                if previous_op['op_name'] == "PAD":
                    # get the padding details
                    self.ops_ip_shape = TensorShape(np.array(
                        previous_op['operator_options'].GetInputShapes()[0]))  # only the input shape
                    self.pad_info = previous_op['operator_options'].GetOperationPaddings(
                    )
                    self.previous_pad = True
        # get the filter and bias tensor
        self.filter_tensor = tflite_interpreter.get_tensor(self.ip_ndxs[1])
        self.bias_tensor = tflite_interpreter.get_tensor(self.ip_ndxs[2])

    def PrintAttributes(self):
        self.meta_data = f"padding:{self.padding}, (stride_w, stride_h):{self.option.StrideW(), self.option.StrideH()}, activation function:{self.GetActivationFunctionType()}, (dilation_h, dilation_w):{self.option.DilationHFactor(), self.option.DilationWFactor()}, depth_multiplier :{self.option.DepthMultiplier()}"
        # print(self.meta_data)
        # self.error = not ( (self.option.DilationHFactor()==1) and (self.option.DilationWFactor()==1) and (self.option.DepthMultiplier()==1))
        return self.meta_data

    def WriteOperatorAttributesToFile(self, file_string, info_string, last_ops_options):
        padding = self.GetOperationPaddings()
        # if last_ops_options is not None and last_ops_options.GetOperationName()=="PAD": #special handling of PADS operation
        if self.previous_pad:
            # check if the current operation has valid padding, if not throw a error
            # if(self.GetPaddingType()!="VALID"):
            #   raise ValueError(f"Padding cannot be applied due to padding type of the operation", "ERR_SAME_PADDING_OP")
            # get the define for the current line from the info string
            pad_operation_define = info_string[9:]
            pad_operation_define = pad_operation_define[:-(
                len(self.operator_name))]
            pad_operation_define += "PAD"
            # special case handling of the fomo face detection model with PAD layer
            # file_string += info_string +"_INPUT_BATCH "+str(self.ip_shape.batch)
            file_string += info_string + \
                "_INPUT_CHANNEL_CNT ("+pad_operation_define + \
                "_INPUT_CHANNEL_CNT)"
            file_string += info_string + \
                "_INPUT_HEIGHT ("+pad_operation_define+"_INPUT_HEIGHT)"
            file_string += info_string + \
                "_INPUT_WIDTH ("+pad_operation_define+"_INPUT_WIDTH)"

            file_string += info_string + \
                "_PADDING_TOP ("+str(padding.pad_top) + \
                "+("+pad_operation_define+"_PADDING_TOP))"
            file_string += info_string + \
                "_PADDING_BOTTOM ("+str(padding.pad_bottom) + \
                "+("+pad_operation_define+"_PADDING_BOTTOM))"
            file_string += info_string + \
                "_PADDING_LEFT ("+str(padding.pad_left) + \
                "+("+pad_operation_define+"_PADDING_LEFT))"
            file_string += info_string + \
                "_PADDING_RIGHT ("+str(padding.pad_right) + \
                "+("+pad_operation_define+"_PADDING_RIGHT))"
            # self.ops_ip_shape = TensorShape(np.array(last_ops_options.GetInputShapes()[0])) #only the input shape
            # self.pad_info = last_ops_options.GetOperationPaddings()
        else:
            # file_string += info_string +"_INPUT_BATCH "+str(self.ip_shape.batch)
            file_string += info_string + \
                "_INPUT_CHANNEL_CNT "+str(self.ip_shape.depth)
            file_string += info_string + \
                "_INPUT_HEIGHT "+str(self.ip_shape.height)
            file_string += info_string + \
                "_INPUT_WIDTH "+str(self.ip_shape.width)

            file_string += info_string + "_PADDING_TOP "+str(padding.pad_top)
            file_string += info_string + \
                "_PADDING_BOTTOM "+str(padding.pad_bottom)
            file_string += info_string + "_PADDING_LEFT "+str(padding.pad_left)
            file_string += info_string + \
                "_PADDING_RIGHT "+str(padding.pad_right)

        stride_h, stride_w = self.GetStridesInfo()
        file_string += info_string + "_STRIDE_W "+str(stride_w)
        file_string += info_string + "_STRIDE_H "+str(stride_h)
        file_string += info_string + "_DEPTH_MULTIPLIER " + \
            str(self.option.DepthMultiplier())
        depth_multiplier_define = info_string[9:] + "_DEPTH_MULTIPLIER"
        # file_string += info_string +"_FILTER_BATCH "+str(self.kernel_shape.batch)
        file_string += info_string + "_FILTER_CHANNEL_CNT " + \
            str(self.kernel_shape.depth)
        filter_channel_define = info_string[9:] + "_FILTER_CHANNEL_CNT"
        file_string += info_string + "_FILTER_HEIGHT " + \
            str(self.kernel_shape.height)
        file_string += info_string + "_FILTER_WIDTH " + \
            str(self.kernel_shape.width)
        file_string += info_string + "_FILTER_BYTEWIDTH "+self.kernel_bytewidth_enum.name

        # file_string += info_string +"_OUTPUT_BATCH "+str(self.op_shape.batch)
        file_string += info_string + \
            "_OUTPUT_CHANNEL_CNT (("+filter_channel_define+")/(" + \
            depth_multiplier_define+")) //"+str(self.op_shape.depth)
        file_string += info_string + \
            "_OUTPUT_HEIGHT "+str(self.op_shape.height)
        file_string += info_string + "_OUTPUT_WIDTH "+str(self.op_shape.width)
        # file_string += info_string +"_OUTPUT_AXON_STRIDE "+str(util.GetAxonproStrideWidth(self.op_shape.width))

        return file_string

    def WriteWeightTensorToFile(self, file_string, info_string, tensor, get_full_content=False):
        if (len(tensor.shape) == 4):
            # if(self.operator_name=="POINTWISE_CONV_2D"):
            #   kernel_tensor = tensor.transpose(0,3,1,2)
            # else:
            kernel_tensor = tensor.transpose(0, 3, 1, 2)
        # file_content += "\nconst int8_t "+array_name_str.lower()+"_weights[]["+array_name_str+"_FILTER_CHANNEL]["+array_name_str+"_FILTER_HEIGHT]["+array_name_str+"_FILTER_WIDTH]= \n" #{ { { { } } } };\n"
        file_string += "\nconst int8_t "+info_string.lower()+"_"+self.GetOperandNameString().lower()+"[]["+info_string + \
            "_FILTER_CHANNEL_CNT]["+info_string+"_FILTER_HEIGHT][" + \
            info_string+"_FILTER_WIDTH]={\n"  # { { { { } } } };\n"
        if (get_full_content):
            for output_channel in range(kernel_tensor.shape[0]):
                file_string += "{"
                for input_channel in range(kernel_tensor.shape[1]):
                    file_string += np.array2string(kernel_tensor[output_channel][input_channel], separator=',',
                                                   max_line_width=1000, threshold=np.inf).replace('[', '{').replace(']', '}').replace("\n", "")
                    file_string += ","
                file_string = file_string[:-1] + "},"
            # file_string +="},"
        else:
            file_string += np.array2string(kernel_tensor, separator=',', max_line_width=1000,
                                           threshold=np.inf).replace('[', '{').replace(']', '}').replace("\n", "")
        file_string = file_string[:-1] + "};"
        return file_string

    def GetAttributes(self):
        d = {'name': self.name,
             'padding': self.padding,
             'stride_w': self.option.StrideW(),
             'stride_h': self.option.StrideH(),
             'activation': self.activation,
             'dilation_w': self.option.DilationWFactor(),
             'dilation_h': self.option.DilationHFactor(),
             'depth_multiplier': self.option.DepthMultiplier()}
        return d

    def GetCommandBufferLen(self, normalized_scaling=True):
        if (not normalized_scaling):
            self.cmd_buf_len_multiplier = +3
        self.command_buf_len = 39 + self.cmd_buf_len_multiplier*self.op_shape.depth
        # print(f"command buff len : {self.command_buf_len}")
        return self.command_buf_len


class FullyConnectedOperatorOptions(OperatorOptions):

    weights_format = ""
    last_layer = False
    reshape_with_channel_before = None
    last_op_og_output_shape = None

    fully_connected_ip_tensor_constraints = {
        "H": 2048,
        "W": 2048,
        "C": None
    }

    @classmethod
    def check_operator_specific_constraints(cls, built_in_options, input_tensors, output_shapes, transpose_check=False):
        io_error_text = ""
        if len(input_tensors) > 0:
            io_error_text += cls.check_tensor_constraints(
                "input", input_tensors[0], cls.fully_connected_ip_tensor_constraints, transpose_check)
        if len(output_shapes) > 0:
            io_error_text += cls.check_tensor_constraints(
                "output", TensorShape(output_shapes[0]), transpose_check=transpose_check)

        return io_error_text

    def last_dense_layer(self, tensor_details, dense_op_ndx) -> bool:
        # next_dense_layer_number = str(int(tensor_details[dense_op_ndx]['name'][tensor_details[dense_op_ndx]['name'].find("dense_")+len('dense_')])+1)
        # #  find if the following tensors have "dense_(next_number_in_the_dense_layer)" in the name, if yes, then it is not the last dense layer
        # next_dense_layer="dense_"+next_dense_layer_number
        for ndx in range(dense_op_ndx+1, len(tensor_details)):
            if ("dense") in tensor_details[ndx]['name']:
                return False
        self.last_layer = True
        return True

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.axons_operation_enum = "NRF_AXON_NN_OP_FULLY_CONNECTED"
        self.operand_str = "filters"
        self.option = tflite.FullyConnectedOptions()
        self.option.Init(self.bytes, self.pos)
        # get the filter and bias tensor
        self.filter_tensor = tflite_interpreter.get_tensor(
            self.ip_ndxs[1])  # weight tensor
        current_op_index = operation_detail['index']
        # get the previous op from the index
        if operator_graph is not None and (current_op_index-1) >= 0:
            previous_op_index = SupportedOperators.get_index_from_tf_index(
                operator_graph, current_op_index-1)
            if (operator_graph[previous_op_index]["op_name"] == "RESHAPE"):
                # print("previous op is reshape")
                # get the input shape of the reshape operation
                self.last_op_og_output_shape = TensorShape(
                    tensor_details[operator_graph[previous_op_index]['ip_tensors'][0]]['shape'])
                test_filter_tensor = np.empty(
                    (self.filter_tensor.shape), dtype=np.int8)
                test_filter_tensor = test_filter_tensor.reshape(len(self.filter_tensor), self.last_op_og_output_shape.batch,
                                                                self.last_op_og_output_shape.depth, self.last_op_og_output_shape.height, self.last_op_og_output_shape.width)
                for channel_ndx, filter in enumerate(self.filter_tensor):
                    test_filter_tensor[channel_ndx] = filter.reshape(
                        self.last_op_og_output_shape.batch, self.last_op_og_output_shape.height, self.last_op_og_output_shape.width, self.last_op_og_output_shape.depth).transpose(0, 3, 1, 2)
                test_filter_tensor = test_filter_tensor.reshape(
                    self.filter_tensor.shape)
                self.filter_tensor = test_filter_tensor
                if self.last_op_og_output_shape.shape_size > 2:
                    if self.last_op_og_output_shape.depth > 1:
                        # need to set this to enable the right transpose action to happen for operators
                        self.reshape_with_channel_before = True

        if self.ip_ndxs[2] != -1:
            self.bias_tensor = tflite_interpreter.get_tensor(self.ip_ndxs[2])
            self.bias_shape = TensorShape(
                tensor_details[operator.InputsAsNumpy()[2]]['shape'])
            self.bias_q = tensor_details[self.ip_ndxs[2]
                                         ]['quantization_parameters']
        else:
            # bias tensor is not present will need to calculate the right shape for it and the quantization
            self.bias_q = self.init_q
            self.bias_shape = self.op_shape
            self.bias_tensor = np.array([0]*self.bias_shape.width)

    def PrintAttributes(self):
        self.meta_data = f"weights :{self.GetWeightsFormat()}, activation function:{self.GetActivationFunctionType()}"
        # print(self.meta_data)
        return self.meta_data

    def WriteOperatorAttributesToFile(self, file_string, info_string, last_op_name):

        file_string += info_string + "_INPUT_HEIGHT "+str(self.ip_shape.height)
        file_string += info_string + "_INPUT_WIDTH "+str(self.ip_shape.width)
        file_string += info_string + \
            "_INPUT_CHANNEL_CNT "+str(self.ip_shape.depth)

        file_string += info_string + "_FILTER_OUTPUT_CHANNEL_CNT " + \
            str(self.kernel_shape.depth) + " //is WEIGHTS"
        file_string += info_string + "_FILTER_HEIGHT " + \
            str(self.kernel_shape.height) + " //is WEIGHTS"
        file_string += info_string + "_FILTER_WIDTH " + \
            str(self.kernel_shape.width) + " //is WEIGHTS"
        file_string += info_string + "_FILTER_BYTEWIDTH " + \
            self.kernel_bytewidth_enum.name + " //is WEIGHTS"

        file_string += info_string + \
            "_OUTPUT_HEIGHT "+str(self.op_shape.height)
        file_string += info_string + "_OUTPUT_WIDTH "+str(self.op_shape.width)
        file_string += info_string + \
            "_OUTPUT_CHANNEL_CNT "+str(self.op_shape.depth)
        # if(self.IsFullyConnectedLastLayer()):
        # file_string += info_string +"_OUTPUT_AXON_STRIDE "+str(util.GetAxonproStrideWidth(self.op_shape.height,np.int32,packed=1))
        # else:
        # file_string += info_string +"_OUTPUT_AXON_STRIDE "+str(util.GetAxonproStrideWidth(self.op_shape.height,np.int8))

        file_string += info_string + "_STRIDE_W "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_STRIDE_H "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_PADDING_TOP "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + \
            "_PADDING_BOTTOM "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_PADDING_LEFT "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_PADDING_RIGHT "+str(0)+" //NOT_REQUIRED"
        return file_string

    def WriteWeightTensorToFile(self, file_string, info_string, tensor, get_full_content=False):
        # file_string += "\nconst int8_t "+info_string.lower()+"_"+self.GetOperandNameString().lower()+"["+info_string+"_WEIGHTS_WIDTH]["+info_string+"_WEIGHTS_HEIGHT]=\n"
        # file_string += np.array2string(tensor.transpose(), separator=',',max_line_width=100).replace('[','{').replace(']','}')
        # file_string += "\nconst int8_t "+info_string.lower()+"_"+self.GetOperandNameString().lower()+"["+info_string+"_WEIGHTS_HEIGHT]["+info_string+"_WEIGHTS_WIDTH]=\n"
        # file_string += np.array2string(tensor, separator=',',max_line_width=10000).replace('[','{').replace(']','}')
        # file_string += ";"
        file_string += "\nconst int8_t "+info_string.lower()+"_"+self.GetOperandNameString().lower() + \
            "["+info_string+"_FILTER_HEIGHT]["+info_string+"_FILTER_WIDTH]={\n"
        for h in range(tensor.shape[0]):
            file_string += np.array2string(tensor[h], separator=',', max_line_width=10000,
                                           threshold=np.inf).replace('[', '{').replace(']', '}')  # .replace("\n","")
            file_string += ",\n"
        file_string = file_string[:-1] + "};"
        return file_string

    def GetWeightsFormat(self):
        if (self.option.WeightsFormat() == tflite.FullyConnectedOptionsWeightsFormat.DEFAULT):
            self.weights_format = "DEFAULT"
        elif (self.option.WeightsFormat() == tflite.FullyConnectedOptionsWeightsFormat.SHUFFLED4x16INT8):
            self.weights_format = "SHUFFLED4x16INT8"
        return self.weights_format

    def IsFullyConnectedLastLayer(self): return self.last_layer

    def SetTransposeKernelFlag(self, transpose_kernel):
        self.transpose_kernel = transpose_kernel
        return False  # we might need to set the transpose kernel flag but the outputs need not be rotated for a FC layer

    def GetFilterTensor(self):
        if self.transpose_kernel and self.reshape_with_channel_before:
            # make that explicit change of vectors for the filter tensor when the operator is a fully connected
            total_length = np.prod(self.last_op_og_output_shape.get_shape())
            original_tensor = np.arange(total_length).reshape(
                self.last_op_og_output_shape.depth, self.last_op_og_output_shape.height, self.last_op_og_output_shape.width)
            permuted_tensor = original_tensor.transpose(0, 2, 1)
            perm_indices = permuted_tensor.flatten()
            new_filter_tensor = self.filter_tensor[:, perm_indices]
            self.filter_tensor = new_filter_tensor
        return self.filter_tensor


class Pool2DOptions(OperatorOptions):
    multiplier = 1
    scaleshift = 0
    scale = None
    area = None
    b_prime_mean = None

    pool_filter_constraints = {
        "H": 32,
        "W": 32
    }

    @classmethod
    def average_pool_is_mean(cls, input_tensor_shape, filter_h, filter_w, output_shapes):
        if not isinstance(input_tensor_shape, TensorShape):
            input_shape = TensorShape(input_tensor_shape.shape)
        else:
            input_shape = input_tensor_shape
        if not isinstance(output_shapes, TensorShape):
            output_shape = TensorShape(output_shapes)
        else:
            output_shape = output_shapes

        # check if the
        if (input_shape.height == filter_h and input_shape.width == filter_w) or (output_shape.height == 1 and output_shape.width == 1):
            return True  # is a global average pool with filter shape matching input shapes
        if input_shape.height == filter_h and filter_w == 1 and (output_shape.height == 1):
            return True  # is a mean along the height axis
        if input_shape.width == filter_w and filter_h == 1 and (output_shape.width == 1):
            return True  # is a mean along the width axis
        return False

    @classmethod
    def check_operator_specific_constraints(cls, built_in_options, input_tensors, output_shapes, transpose_check=False):
        io_error_txt = ""
        io_error_txt += cls.check_default_constraints(
            built_in_options, input_tensors, output_shapes, transpose_check)
        # get the filter dimensions
        option = tflite.Pool2DOptions()
        option.Init(built_in_options.Bytes, built_in_options.Pos)
        filter_h = option.FilterHeight()
        filter_w = option.FilterWidth()
        if not cls.average_pool_is_mean(input_tensors[0], filter_h, filter_w, output_shapes[0]):
            filter_error_text = cls.check_filter_constraints(
                filter_h, filter_w, cls.pool_filter_constraints, transpose_check)
        else:
            filter_error_text = ""
        return io_error_txt+filter_error_text

    def sum_all_padding_values(self):
        # sum all the padding values
        return self.pad_info.pad_left + self.pad_info.pad_right + self.pad_info.pad_top + self.pad_info.pad_bottom

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.option = tflite.Pool2DOptions()
        self.option.Init(self.bytes, self.pos)
        self.command_buf_len = 39
        self.kernel_shape.width = self.option.FilterWidth()
        self.kernel_shape.height = self.option.FilterHeight()
        self.kernel_bytewidth_enum = tflite_axon_enum_wrapper.GetAxonByteWidthEnum(
            self.kernel_bitwidth)
        self.CalculatePadding()
        self.axons_operation_enum = "NRF_AXON_NN_OP_MAX_POOLING"
        if self.operator_name == "AVERAGE_POOL_2D":
            self.axons_operation_enum = "NRF_AXON_NN_OP_AVERAGE_POOLING"
            if self.padding == 'SAME' and self.sum_all_padding_values():
                self.error = True
                self.error_text = f"{self.padding} padding type for Average Pool is not currently supported!"
                self.error_action = "ERROR"
            # check here for the conditions to determine if the operation is mean or avgpool
            if self.average_pool_is_mean(self.ip_shape, self.kernel_shape.height, self.kernel_shape.width, self.op_shape):
                self.axons_operation_enum = "NRF_AXON_NN_OP_MEAN"
                self.operator_name += "_MEAN"

    def PrintAttributes(self):
        self.meta_data = f"Padding :{self.padding}, (stride_w, stride_h):{self.option.StrideW(), self.option.StrideH()}, (filter_w, filter_h):{self.option.FilterWidth(), self.option.FilterHeight()},activation function:{self.GetActivationFunctionType()}"
        # print(self.meta_data)
        return self.meta_data

    def WriteOperatorAttributesToFile(self, file_string, info_string, last_op_name):

        # file_string += info_string +"_INPUT_BATCH "+str(self.ip_shape.batch)
        file_string += info_string + \
            "_INPUT_CHANNEL_CNT "+str(self.ip_shape.depth)
        file_string += info_string + "_INPUT_HEIGHT "+str(self.ip_shape.height)
        file_string += info_string + "_INPUT_WIDTH "+str(self.ip_shape.width)

        file_string += info_string + "_FILTER_OUTPUT_CHANNEL_CNT "+str(0)
        file_string += info_string + "_FILTER_HEIGHT " + \
            str(self.option.FilterHeight())
        file_string += info_string + "_FILTER_WIDTH " + \
            str(self.option.FilterWidth())
        file_string += info_string + "_FILTER_BYTEWIDTH "+self.kernel_bytewidth_enum.name

        stride_h, stride_w = self.GetStridesInfo()
        file_string += info_string + "_STRIDE_W "+str(stride_w)
        file_string += info_string + "_STRIDE_H "+str(stride_h)

        file_string += info_string + "_PADDING_TOP "+str(self.pad_info.pad_top)
        file_string += info_string + "_PADDING_BOTTOM " + \
            str(self.pad_info.pad_bottom)
        file_string += info_string + "_PADDING_LEFT " + \
            str(self.pad_info.pad_left)
        file_string += info_string + "_PADDING_RIGHT " + \
            str(self.pad_info.pad_right)

        # file_string += info_string +"_OUTPUT_BATCH "+str(self.op_shape.batch)
        file_string += info_string + \
            "_OUTPUT_CHANNEL_CNT "+str(self.op_shape.depth)
        file_string += info_string + \
            "_OUTPUT_HEIGHT "+str(self.op_shape.height)
        file_string += info_string + "_OUTPUT_WIDTH "+str(self.op_shape.width)
        # file_string += info_string +"_OUTPUT_AXON_STRIDE "+str(util.GetAxonproStrideWidth(self.op_shape.height))

        # file_string += info_string + "_INPUT_ZERO_POINT "+str(0)+" //NOT_REQUIRED"
        # file_string += info_string + "_OUTPUT_ZERO_POINT "+str(0)+" //NOT_REQUIRED"
        return file_string

    def WriteWeightTensorToFile(self, file_string, info_string, tensor, get_full_content=False):
        return file_string

    def CalculateBPrime(self):
        if self.operator_name == "MAX_POOL_2D":
            self.b_prime_tensor = np.array([])
        else:
            # self.b_prime_tensor = np.array([np.round(self.b_prime_mean[0]*2**self.scale_shifts[0]).astype(np.int32)])
            self.b_prime_tensor = np.array(
                [np.round(self.b_prime_mean[0]).astype(np.int32)])

    def CalculateMultiplierandScaleshift(self):  # for pool operation
        """
        We also have to get a multiplier and a shift for the AVERAGE POOLING so that it works with Axon    
        calculates the scale_q (multiplier/slope value for the operation and the scaleshift before the zeropoint or the scaleshift limiting value saturates)
        """
        if self.operator_name == "MAX_POOL_2D":
            # self.ip_q_zeropoint[0] = 0
            self.op_q_zeropoint[0] = 0
            self.scale_multipliers = np.array([np.int32(1)])
            self.scale_shifts = np.array([np.int32(0)])
            return

        bit_limit = 31
        if (self.ip_q == self.op_q):
            # FIXME this needs to go when the implementation of the pooling operation using b' is done in the compiler/simulator
            bit_limit = 25
        self.area = (self.kernel_shape.height * self.kernel_shape.width)
        b_prime_scale = self.ip_q['scales'] / self.op_q['scales']
        self.b_prime_mean = -1 * self.ip_q_zeropoint * self.area
        self.scale = b_prime_scale / (self.area)
        saturation_limit = self.op_q_zeropoint
        error1, scale_shift_op_1 = util.optimized_ip_scaling_shift(
            self.scale, 8, 31, bit_limit, saturation_limit)
        self.scale_multipliers = np.array(
            [abs(np.round(self.scale[0]*2**scale_shift_op_1)).astype(np.int32)])
        self.scale_shifts = np.array([scale_shift_op_1], dtype=np.int8)


class MeanOperatorOptions(Pool2DOptions):
    axis = None
    keep_dims = None

    @classmethod
    def determine_mean_filter_h_w(cls, axis, ip_shape, kernel_shape=None):
        if isinstance(axis, (int, np.int32)):
            axis = np.array([axis])
        pass_flag = True
        if kernel_shape is None:
            kernel_shape = TensorShape(np.array([]))
        if kernel_shape.shape_size == 0:
            kernel_shape = TensorShape(
                np.ones(ip_shape.shape_size, dtype=np.int32))
        for dim in axis:
            if dim < 0:
                dim += ip_shape.shape_size
            if dim == 0 or dim == 3:  # batch and channels dimensions are not supported either
                pass_flag = False
                break
            elif (dim == 1):  # height
                kernel_shape.height = ip_shape.get_shape()[dim]
            elif (dim == 2):  # width
                kernel_shape.width = ip_shape.get_shape()[dim]
        return pass_flag, kernel_shape

    @classmethod
    def check_operator_specific_constraints(cls, built_in_options, input_tensors, output_shapes, transpose_check=False):
        io_error_txt = ""
        io_error_txt += cls.check_default_constraints(
            built_in_options, input_tensors, output_shapes, transpose_check)
        # No longer need to check filter dimensions on the mean operation
        filter_error_text = ""
        # # get the filter dimensions
        # option = tflite.ReducerOptions()
        # option.Init(built_in_options.Bytes, built_in_options.Pos)
        # # for the mean operation either the input width or the input height is the constraint.
        # # we need to understand the axis along which the mean is being calculated.
        # input_shape = TensorShape(input_tensors[0].shape)
        # pass_flag, filter_shape = cls.determine_mean_filter_h_w(
        #     input_tensors[1], input_shape)
        # if pass_flag:
        #     filter_error_text = cls.check_filter_constraints(
        #         filter_shape.height, filter_shape.width, cls.pool_filter_constraints, transpose_check)
        # else:
        #     filter_error_text = "Only spatial dimensions(HxW) are supported"
        return io_error_txt+filter_error_text

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.option = tflite.ReducerOptions()
        # self.axons_operation_enum = "NRF_AXON_NN_OP_AVERAGE_POOLING"
        self.axons_operation_enum = "NRF_AXON_NN_OP_MEAN"
        self.option.Init(self.bytes, self.pos)
        self.keep_dims = self.option.KeepDims()
        # get the axis and calculate the filter height and width here accordingly
        self.axis = tflite_interpreter.get_tensor(self.ip_ndxs[1])
        pass_flag, self.kernel_shape = self.determine_mean_filter_h_w(
            self.axis, self.ip_shape, self.kernel_shape)
        assert (
            pass_flag), f"Only spatial dimensions(HxW) are supported for the MEAN operator, axis is {self.axis}"

        # figure out the strides using the kernel information
        self.stride_x = self.kernel_shape.width
        self.stride_y = self.kernel_shape.height
        self.padding = 'VALID'
        # determine here how the mean operator is going to affect the output shape and maintain the rank of the output shape.
        # if keep dimensions is false and the input shape rank is 4
        if not self.keep_dims and self.ip_shape.shape_size == 4:
            self.op_shape.shape_size = 4
            if self.axis.size == 2:
                self.op_shape.depth = self.op_shape.width
                self.op_shape.width = 1
                self.op_shape.height = 1
            else:
                if self.axis[0] == 1:  # height is averaged
                    self.op_shape.depth = self.op_shape.width
                    self.op_shape_width = self.op_shape.height
                    self.op_shape.height = 1
                else:  # width is averaged
                    self.op_shape.depth = self.op_shape.width
                    self.op_shape.width = 1

    # def CalculateMultiplierandScaleshift(self):#for mean operator
    #   self.area = self.kernel_shape.height * self.kernel_shape.width
    #   self.scale  = self.ip_q['scales'] / self.op_q['scales']
    #   self.scale = self.scale / self.area
    #   result = util.optimized_ip_scaling_shift((self.scale),8,25 ,25)
    #   error = result[0]
    #   scaleshift = result[1]
    #   self.scale_multipliers=np.array([abs(int(np.round((self.scale)*2**scaleshift)))])
    #   self.scale_shifts=np.array([scaleshift])

    # def CalculateBPrime(self):
    #   self.b_prime_mean = -1 * self.ip_q_zeropoint * self.area
    #   self.b_prime_tensor = np.array([np.round(self.b_prime_mean[0]).astype(np.int32)])

    def PrintAttributes(self):
        self.meta_data = f"axis :{self.axis}, keep_dims :{self.keep_dims}"
        # print(self.meta_data)
        return self.meta_data

    def GetStridesInfo(self):
        return self.stride_x, self.stride_y

    def GetDivisorValue(self):
        return self.divisor_value

    def WriteOperatorAttributesToFile(self, file_string, info_string, last_op_name):

        # file_string += info_string +"_INPUT_BATCH "+str(self.ip_shape.batch)
        file_string += info_string + \
            "_INPUT_CHANNEL_CNT "+str(self.ip_shape.depth)
        file_string += info_string + "_INPUT_HEIGHT "+str(self.ip_shape.height)
        file_string += info_string + "_INPUT_WIDTH "+str(self.ip_shape.width)

        file_string += info_string + "_FILTER_OUTPUT_CHANNEL_CNT " + \
            str(self.kernel_shape.depth)
        file_string += info_string + "_FILTER_HEIGHT " + \
            str(self.kernel_shape.height)
        file_string += info_string + "_FILTER_WIDTH " + \
            str(self.kernel_shape.width)
        file_string += info_string + "_FILTER_BYTEWIDTH "+self.kernel_bytewidth_enum.name

        file_string += info_string + "_STRIDE_W "+str(self.stride_x)
        file_string += info_string + "_STRIDE_H "+str(self.stride_y)

        file_string += info_string + "_PADDING_TOP "+str(self.pad_info.pad_top)
        file_string += info_string + "_PADDING_BOTTOM " + \
            str(self.pad_info.pad_bottom)
        file_string += info_string + "_PADDING_LEFT " + \
            str(self.pad_info.pad_left)
        file_string += info_string + "_PADDING_RIGHT " + \
            str(self.pad_info.pad_right)

        # file_string += info_string +"_OUTPUT_BATCH "+str(self.op_shape.batch)
        file_string += info_string + \
            "_OUTPUT_CHANNEL_CNT "+str(self.op_shape.depth)
        file_string += info_string + \
            "_OUTPUT_HEIGHT "+str(self.op_shape.height)
        file_string += info_string + "_OUTPUT_WIDTH "+str(self.op_shape.width)
        # file_string += info_string +"_OUTPUT_AXON_STRIDE "+str(util.GetAxonproStrideWidth(self.op_shape.height))

        # file_string += info_string + "_INPUT_ZERO_POINT "+str(self.ip_q_zeropoint)
        # file_string += info_string + "_OUTPUT_ZERO_POINT "+str(self.op_q_zeropoint)
        return file_string


class AddOptions(OperatorOptions):
    ip1_q = None
    ip2_q = None
    scale_ip1 = None
    scale_ip2 = None
    op_scales = None
    error1 = None
    error2 = None
    scale_shift_op_1 = None
    scale_shift_op_2 = None
    op_zeropoint = None
    ip_zeropoint = None
    scale_shift = None
    scale_a = None
    scale_b = None
    scale_q = None
    multipliers_calculated = None
    add_with_constant = None

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        # self.kernel_shape = TensorShape(np.array([]))
        self.kernel_bytewidth_enum = tflite_axon_enum_wrapper.GetAxonByteWidthEnum(
            self.kernel_bitwidth)
        self.option = tflite.AddOptions()
        self.axons_operation_enum = "NRF_AXON_NN_OP_ADD2"
        self.option.Init(self.bytes, self.pos)
        self.command_buf_len = 39
        self.multipliers_calculated = False
        self.add_with_constant = False
        if operator_graph is not None:
            operator_graph_info = operator_graph[operation_detail['index']]
            self.ip1_q = tensor_details[operator_graph[operator_graph_info['inputs']
                                                       [0]]['op_tensors'][0]]['quantization_parameters']
            # self.ip2_q = tensor_details[operator_graph[operator_graph_info['inputs']
            #                                            [1]]['op_tensors'][0]]['quantization_parameters']
            if len(operator_graph_info['inputs']) > 1:
                self.ip2_q = tensor_details[operator_graph[operator_graph_info['inputs']
                                                           [1]]['op_tensors'][0]]['quantization_parameters']
            elif len(operator_graph_info['ip_tensors']) > 1:
                # this is because of a constant input to the add operation, need to be handled properly
                self.ip2_q = tensor_details[operator_graph_info['ip_tensors']
                                            [1]]['quantization_parameters']
                # get the filter here as well                
                self.filter_tensor = tflite_interpreter.get_tensor(
                    operator_graph_info['ip_tensors'][1])
                self.kernel_shape = TensorShape(self.filter_tensor.shape)
                self.add_with_constant = True
                #get the input zero point of the constant tensor here.
                self.ip_q = copy.deepcopy(tensor_details[operator_graph_info['ip_tensors'][1]]['quantization_parameters'])
                self.ip_q_zeropoint = copy.deepcopy(self.ip_q['zero_points'])

    def CalculateBPrime(self):
        bias_add_scale_shftd = (-(self.ip1_q['scales']*self.ip1_q['zero_points']+self.ip2_q['scales']
                                * self.ip2_q['zero_points'])/self.op_scales)*(2**self.scale_shifts[0])
        bias_add_array = np.array(bias_add_scale_shftd.astype(np.int32))
        self.b_prime_tensor = bias_add_array

    def PrintAttributes(self):
        self.meta_data = f"activation function:{self.GetActivationFunctionType()}, pot scale int16:{self.option.PotScaleInt16()}"
        # print(self.meta_data)
        return self.meta_data

    def WriteOperatorAttributesToFile(self, file_string, info_string, last_op_name):

        # file_string += info_string +"_INPUT_BATCH "+str(self.ip_shape.batch)
        file_string += info_string + \
            "_INPUT_CHANNEL_CNT "+str(self.ip_shape.depth)
        file_string += info_string + "_INPUT_HEIGHT "+str(self.ip_shape.height)
        file_string += info_string + "_INPUT_WIDTH "+str(self.ip_shape.width)

        file_string += info_string + \
            "_FILTER_OUTPUT_CHANNEL_CNT "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_FILTER_HEIGHT "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_FILTER_WIDTH "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_FILTER_BYTEWIDTH " + \
            self.kernel_bytewidth_enum.name+" //NOT_REQUIRED"

        file_string += info_string + "_STRIDE_W "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_STRIDE_H "+str(0)+" //NOT_REQUIRED"

        file_string += info_string + "_PADDING_TOP "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + \
            "_PADDING_BOTTOM "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_PADDING_LEFT "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_PADDING_RIGHT "+str(0)+" //NOT_REQUIRED"

        # file_string += info_string +"_OUTPUT_BATCH "+str(self.op_shape.batch)
        file_string += info_string + \
            "_OUTPUT_CHANNEL_CNT "+str(self.op_shape.depth)
        file_string += info_string + \
            "_OUTPUT_HEIGHT "+str(self.op_shape.height)
        file_string += info_string + "_OUTPUT_WIDTH "+str(self.op_shape.width)
        # file_string += info_string +"_OUTPUT_AXON_STRIDE "+str(util.GetAxonproStrideWidth(self.op_shape.height))

        return file_string

    def CalculateMultiplierandScaleshift(self):  # for add operation
        if self.multipliers_calculated:
            return 
        self.op_scales = self.op_q['scales']
        self.op_zeropoint = self.op_q['zero_points']
        self.scale_ip1 = self.ip1_q['scales']/self.op_scales
        self.scale_ip2 = self.ip2_q['scales']/self.op_scales
        self.error1, self.scale_shift_op_1 = util.optimized_ip_scaling_shift(
            self.scale_ip1, 8, 31, 15, self.op_zeropoint, zp_bit_limit=31)
        self.error2, self.scale_shift_op_2 = util.optimized_ip_scaling_shift(
            self.scale_ip2, 8, 31, 15, self.op_zeropoint, zp_bit_limit=31)        
        self.scale_shifts = np.array(
            [min(self.scale_shift_op_1, self.scale_shift_op_2)], dtype=np.int8)
        self.scale_a = abs(
            np.round(self.scale_ip1*2**self.scale_shifts[0])).astype(np.int32)
        self.scale_b = abs(
            np.round(self.scale_ip2*2**self.scale_shifts[0])).astype(np.int32)
        self.scale_multipliers = np.array([self.scale_a[0], self.scale_b[0]])
        self.ip_zeropoint = 0
        self.ip_q_zeropoint[0] = self.ip_zeropoint
        self.multipliers_calculated = True

    # def GetFilterTensor(self):
    #     #check here if the second input is a constant tensor with just one value[1x1x1] or one value per channel[1x1xC]
    #     if self.add_with_constant and ( self.kernel_shape.height == 1 and self.kernel_shape.width == 1 ):
    #         #calculate the scale multipliers if not calculated already
    #         if not self.multipliers_calculated :
    #             self.CalculateMultiplierandScaleshift()
    #         self.filter_tensor = (self.filter_tensor * self.scale_b).astype(np.int32)        
    #     return self.transpose_tensor_if_needed(self.filter_tensor, self.transpose_kernel)


class PadOptions(OperatorOptions):
    paddings = None
    operator_graph_info = None
    next_operator_name = ""

    @classmethod
    def is_channel_pad(cls, tflite_interpreter, inputs):
        paddings = tflite_interpreter.get_tensor(inputs[1])
        if len(paddings) == 3:
            return False
        return ((paddings[3][0]+paddings[3][1]) != 0)

    @classmethod
    def check_if_sandwiched_transpose(cls, tflite_interpreter, operator_graph, current_pad_index):
        total_graph_length = len(operator_graph)
        previous_op_is_tr, next_op_tr = False, False
        if (current_pad_index) > 0:
            previous_op_index = current_pad_index-1
            previous_op_is_tr = operator_graph[previous_op_index]['op_name'] == "TRANSPOSE"
        if (current_pad_index <= total_graph_length):
            next_op_index = current_pad_index+1
            next_op_tr = operator_graph[next_op_index]['op_name'] == "TRANSPOSE"
        if previous_op_is_tr and next_op_tr:
            # ensure that the transpose index is the same and return that
            previous_tr = tflite_interpreter.get_tensor(
                operator_graph[previous_op_index]['ip_tensors'][1])
            next_tr = tflite_interpreter.get_tensor(
                operator_graph[next_op_index]['ip_tensors'][1])
            return (previous_tr == next_tr).all()
        return False

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.option = tflite.PadOptions()
        self.option.Init(self.bytes, self.pos)
        self.command_buf_len = 0
        self.paddings = tflite_interpreter.get_tensor(self.ip_ndxs[1])
        operator_graph_index = operation_detail['index']
        assert (self.paddings[0][0]+self.paddings[0][1] ==
                0), "Padding along the batch dimension is not supported!"
        if self.ip_shape.shape_size > 3:
            self.pad_info.pad_top = self.paddings[1][0]
            self.pad_info.pad_bottom = self.paddings[1][1]
            self.pad_info.pad_left = self.paddings[2][0]
            self.pad_info.pad_right = self.paddings[2][1]
            if len(self.paddings) > 3:
                self.pad_info.pad_front = self.paddings[3][0]
                self.pad_info.pad_back = self.paddings[3][1]
        elif self.ip_shape.shape_size == 3:
            if operator_graph is not None:
                # figure out if there is a transpose before and after the current PAD operation
                if (self.check_if_sandwiched_transpose(tflite_interpreter, operator_graph, operator_graph_index)):
                    # get the transpose matrix
                    transpose_by = tflite_interpreter.get_tensor(
                        operator_graph[operator_graph_index+1]['ip_tensors'][1])
                    self.paddings = self.paddings[transpose_by]
                    previous_valid_axon_op_ndx = self.get_previous_valid_axon_operator_index(
                        operator_graph, operator_graph_index)
                    assert previous_valid_axon_op_ndx is not None, "Pad is present with an invalid pattern!"
                    self.ip_shape = operator_graph[previous_valid_axon_op_ndx]['operator_options'].op_shape

                self.pad_info.pad_top = self.paddings[0][0]
                self.pad_info.pad_bottom = self.paddings[0][1]
                self.pad_info.pad_left = self.paddings[1][0]
                self.pad_info.pad_right = self.paddings[1][1]

        # check here if this needs to be an operation or just a carryforward pad operation
        if ((self.pad_info.pad_front+self.pad_info.pad_back) != 0):
            # ensure that there is no top and bottom padding here as that will cause us to lose information about the paddings
            assert (self.pad_info.pad_top+self.pad_info.pad_bottom ==
                    0), "unhandled top and bottom paddings are present along with channel padding!"
            self.operator_name = "CHANNEL_PAD"
            self.axons_operation_enum = "NRF_AXON_NN_OP_CHANNEL_PADDING"

        if operator_graph is not None:
            self.operator_graph_info = operator_graph[operator_graph_index]
            for op_index, operator_info_graph in enumerate(operator_graph[operator_graph_index:]):
                if operator_info_graph['axon_layer_num'] >= 0:
                    self.next_operator_name = operator_info_graph['op_name']
                    break
            if self.operator_name != "CHANNEL_PAD":
                assert (self.next_operator_name == "DEPTHWISE_CONV_2D" or self.next_operator_name ==
                        "CONV_2D"), "No Supported Operation after Pad : ERR_NO_SUPPORTED_OP_AFTER_PAD"

    def PrintAttributes(self):
        self.meta_data = f"padding values (top, bottom) : {self.pad_info.pad_top,self.pad_info.pad_bottom}, (left,right) : {self.pad_info.pad_left,self.pad_info.pad_right}, (front, back) : {self.pad_info.pad_front,self.pad_info.pad_back}"
        # print(self.meta_data)
        return self.meta_data

    def WriteOperatorAttributesToFile(self, file_string, info_string, last_op_name):
        # file_string += info_string +"_INPUT_BATCH "+str(self.ip_shape.batch)
        file_string += info_string + \
            "_INPUT_CHANNEL_CNT "+str(self.ip_shape.depth)
        file_string += info_string + "_INPUT_HEIGHT "+str(self.ip_shape.height)
        file_string += info_string + "_INPUT_WIDTH "+str(self.ip_shape.width)

        file_string += info_string + "_PADDING_FRONT " + \
            str(self.pad_info.pad_front)
        file_string += info_string + "_PADDING_BACK " + \
            str(self.pad_info.pad_back)
        file_string += info_string + "_PADDING_TOP "+str(self.pad_info.pad_top)
        file_string += info_string + "_PADDING_BOTTOM " + \
            str(self.pad_info.pad_bottom)
        file_string += info_string + "_PADDING_LEFT " + \
            str(self.pad_info.pad_left)
        file_string += info_string + "_PADDING_RIGHT " + \
            str(self.pad_info.pad_right)

        # file_string += info_string +"_OUTPUT_BATCH "+str(self.op_shape.batch)
        file_string += info_string + \
            "_OUTPUT_CHANNEL_CNT "+str(self.op_shape.depth)
        file_string += info_string + \
            "_OUTPUT_HEIGHT "+str(self.op_shape.height)
        file_string += info_string + "_OUTPUT_WIDTH "+str(self.op_shape.width)
        # file_string += info_string +"_OUTPUT_AXON_STRIDE "+str(util.GetAxonproStrideWidth(self.op_shape.height))

        return file_string

    def WritePaddings(self, file_string, info_string):
        # the dimension of the input is batch height, width, depth
        # therefore the paddings is the same format and will pad in that dimension based on the values
        file_string += info_string + "_PADDING_FRONT " + \
            str(self.pad_info.pad_front)
        file_string += info_string + "_PADDING_BACK " + \
            str(self.pad_info.pad_back)
        file_string += info_string + "_PADDING_TOP "+str(self.pad_info.pad_top)
        file_string += info_string + "_PADDING_BOTTOM " + \
            str(self.pad_info.pad_bottom)
        file_string += info_string + "_PADDING_LEFT " + \
            str(self.pad_info.pad_left)
        file_string += info_string + "_PADDING_RIGHT " + \
            str(self.pad_info.pad_right)

        return file_string

    def GetOptionsError(self):
        """
        this should return a text value and a function that will be invoked, the function could be a dummy function or simply be a break or continue statement
        """
        if self.operator_graph_info['operator_support'].value == OperatorSupportEnum.PARTIALLY_SUPPORTED.value:
            self.error_text = f"{self.operator_name} operation is partially supported!, continuing the operations!"
            self.error_action = "CONTINUE"
            self.error = True
        return self.error, self.error_text, self.error_action

    def CalculateMultiplierandScaleshift(self):
        scale_q = self.ip_q['scales']/self.op_q['scales']
        result = util.optimized_ip_scaling_shift((scale_q), 8, 25, 25)
        # error = result[0]
        scaleshift = result[1]
        self.scale_multipliers = np.array(
            [abs(int(np.round((scale_q)*2**scaleshift)))])
        self.scale_shifts = np.array([scaleshift])


class SoftmaxOperatorOptions(OperatorOptions):
    beta = -125

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.option = tflite.SoftmaxOptions()
        self.axons_operation_enum = ""
        self.option.Init(self.bytes, self.pos)
        self.command_buf_len = 31
        self.beta = self.option.Beta()
        # if(self.ip_shape.shape_size==2):#FIXME this change breaks the testing on the axon side simulator
        #   self.ip_shape.depth, self.ip_shape.width =  self.ip_shape.width, self.ip_shape.depth
        self.ip_bitwidth = np.int32
        self.kernel_bytewidth_enum = tflite_axon_enum_wrapper.GetAxonByteWidthEnum(
            self.kernel_bitwidth)

        scale = 1/self.op_q['scales']
        error1, scale_shift_op_1 = util.optimized_ip_scaling_shift(
            scale, 8, 31, 31, self.op_q_zeropoint)
        self.scale_multipliers = np.array(
            [abs(np.round(scale[0]*2**scale_shift_op_1)).astype(np.int32)])
        self.scale_shift = np.array([scale_shift_op_1], dtype=np.int8)
        self.ip_q_zeropoint[0] = 0

    def PrintAttributes(self):
        self.meta_data = f"beta value : {self.beta}"
        # print(self.meta_data)
        return self.meta_data

    def WriteOperatorAttributesToFile(self, file_string, info_string, last_op_name):

        file_string += info_string + "_BETA_VALUE "+str(int(self.beta))
        # file_string += info_string +"_INPUT_BATCH "+str(self.ip_shape.batch)
        file_string += info_string + \
            "_INPUT_CHANNEL_CNT "+str(self.ip_shape.depth)
        file_string += info_string + "_INPUT_HEIGHT "+str(self.ip_shape.height)
        file_string += info_string + "_INPUT_WIDTH "+str(self.ip_shape.width)

        file_string += info_string + \
            "_FILTER_OUTPUT_CHANNEL_CNT "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_FILTER_HEIGHT "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_FILTER_WIDTH "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_FILTER_BYTEWIDTH " + \
            self.kernel_bytewidth_enum.name+" //NOT_REQUIRED"

        file_string += info_string + "_STRIDE_W "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_STRIDE_H "+str(0)+" //NOT_REQUIRED"

        file_string += info_string + "_PADDING_TOP "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + \
            "_PADDING_BOTTOM "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_PADDING_LEFT "+str(0)+" //NOT_REQUIRED"
        file_string += info_string + "_PADDING_RIGHT "+str(0)+" //NOT_REQUIRED"

        # file_string += info_string +"_OUTPUT_BATCH "+str(self.op_shape.batch)
        file_string += info_string + \
            "_OUTPUT_CHANNEL_CNT "+str(self.op_shape.depth)
        file_string += info_string + \
            "_OUTPUT_HEIGHT "+str(self.op_shape.height)
        file_string += info_string + "_OUTPUT_WIDTH "+str(self.op_shape.width)
        # file_string += info_string +"_OUTPUT_AXON_STRIDE "+str(util.GetAxonproStrideWidth(self.op_shape.height))

        return file_string

    def GetSoftmaxBeta(self):
        return self.beta


class LeakyReluOptions(OperatorOptions):
    leaky_relu_alpha = 0

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.option = tflite.LeakyReluOptions()
        self.axons_operation_enum = ""
        self.option.Init(self.bytes, self.pos)
        self.leaky_relu_alpha = self.option.Alpha()

    def PrintAttributes(self):
        self.meta_data = f"alpha value : {self.leaky_relu_alpha}"
        # print(self.meta_data)
        return self.meta_data

    def WriteOperatorAttributesToFile(self, file_string, info_string, last_op_name):
        file_string += info_string + "_LEAKY_RELU_ALPHA_VALUE " + \
            f"{self.leaky_relu_alpha:0.1f}"
        return file_string

    def GetLeakyReluAlpha(self):
        return self.leaky_relu_alpha


class ConcatenationOptions(OperatorOptions):
    axis = None

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.option = tflite.ConcatenationOptions()
        self.axons_operation_enum = "NRF_AXON_NN_OP_CONCATENATE"
        self.option.Init(self.bytes, self.pos)
        self.axis = self.option.Axis()
        if self.axis < 0:
            self.axis += self.ip_shape.shape_size
        # for the tflite, the shape format is NHWC with indices 0,1,2,3 for a shape size of 4
        # or if the shape size is 3, then NHW with indices 0,1,2
        # need to translate this to the axon shape info which is CHW with indices 0,1,2 respectively
        # get the CHANNEL NAME from the AXIS
        if self.ip_shape.shape_size == 4:
            axis_channel_name = mw.TFLITE_RANK4_AXON_AXIS_ENUM_MAP[self.axis]
        else:
            axis_channel_name = mw.TFLITE_RANK3_AXON_AXIS_ENUM_MAP[self.axis]
        if self.tflite_axon_enum_wrapper is not None:
            axons_axis_dict = self.tflite_axon_enum_wrapper.GetAxonAxisEnumDict()
            self.axis = axons_axis_dict[axis_channel_name]
        # get the quantization params of the concat inputs and if they are different, throw an error
        assert (
            self.ip_q == self.w_q), "CONCATENATION currently supports inputs with same quantization!"
        # no kernel for concatenation
        self.kernel_shape = TensorShape(np.array([]))

    def PrintAttributes(self):
        self.meta_data = f"concatenation axis : {self.axis}, activation function:{self.GetActivationFunctionType()}"
        # print(self.meta_data)
        return self.meta_data

    def GetConcatenateAxis(self):
        return self.axis


class StridedSliceOptions(OperatorOptions):
    MAX_NUM_AXIS = 4
    begin_mask = None
    ellipsis_mask = None
    end_mask = None
    new_axis_mask = None
    shrink_axis_mask = None
    begin = None
    begin_axon = None
    end = None
    end_axon = None
    stride = None
    stride_axon = None
    stride_slice_filter_tensor = None

    def get_bin_mask(self, mask):
        bin_mask = bin(mask)[2:].zfill(self.MAX_NUM_AXIS)
        bin_mask = bin_mask[::-1]
        return bin_mask

    def convert_mask_to_shape_info(self, mask, slice_array, type=""):
        bin_mask = self.get_bin_mask(mask)
        for i, bit in enumerate(bin_mask):
            bit = bool(int(bit))
            if not bit:
                if slice_array[i] < 0:
                    # adjust for the slice array value using the shape
                    slice_array[i] = self.ip_shape.get_shape()[i] + \
                        slice_array[i]
            else:
                # ignore the bits based on the types
                # if the begin bit is set, ignore the value and use 0
                # if the end bit is set, ignore the value and use the maximum possible value
                if type == "begin":
                    slice_array[i] = 0
                elif type == "end":
                    slice_array[i] = self.ip_shape.get_shape()[i]

    def convert_ellipsis_mask_to_shape_info(self):
        bin_mask = self.get_bin_mask(self.ellipsis_mask)
        for i, bit in enumerate(bin_mask):
            bit = bool(int(bit))
            if bit:
                # taking the end values for which the ellipsis is present
                self.begin[i] = 0
                self.end[i] = self.ip_shape.get_shape()[i]

    def convert_shrink_axis_mask_to_shape_info(self):
        bin_mask = self.get_bin_mask(self.shrink_axis_mask)
        for i, bit in enumerate(bin_mask):
            bit = bool(int(bit))
            if bit:
                # taking the end values for which the ellipsis is present
                self.ip_shape.get_shape()[i] = 0

    @classmethod
    def same_input_output_length(cls, tflite_interpreter, input, output):
        # check if this strided slice can be a reshape op
        # ip_shape = TensorShape(tflite_interpreter.get_tensor(input[0]).shape)
        ip_shape = TensorShape(
            tflite_interpreter.get_tensor_details()[input[0]]['shape'])
        # op_shape = TensorShape(tflite_interpreter.get_tensor(output[0]).shape)
        op_shape = TensorShape(tflite_interpreter.get_tensor_details()[
                               output[0]]['shape'])
        ip_length = ip_shape.get_length()
        op_length = op_shape.get_length()
        return ip_length == op_length

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.option = tflite.StridedSliceOptions()
        self.axons_operation_enum = "NRF_AXON_NN_OP_STRIDED_SLICE"
        self.option.Init(self.bytes, self.pos)
        self.kernel_bitwidth = np.int32
        self.kernel_bytewidth_enum = tflite_axon_enum_wrapper.GetAxonByteWidthEnum(
            self.kernel_bitwidth)
        if operator_graph is not None:
            # operator_graph[operation_detail['index']]
            operator_graph_info = operation_detail
        self.begin_mask = self.option.BeginMask()
        self.ellipsis_mask = self.option.EllipsisMask()
        self.end_mask = self.option.EndMask()
        self.new_axis_mask = self.option.NewAxisMask()
        self.shrink_axis_mask = self.option.ShrinkAxisMask()
        self.begin = tflite_interpreter.get_tensor(self.ip_ndxs[1])
        self.end = tflite_interpreter.get_tensor(self.ip_ndxs[2])
        self.stride = tflite_interpreter.get_tensor(self.ip_ndxs[3])
        self.convert_mask_to_shape_info(self.begin_mask, self.begin, "begin")
        self.convert_mask_to_shape_info(self.end_mask, self.end, "end")
        if self.ellipsis_mask:
            self.convert_ellipsis_mask_to_shape_info()
        if self.shrink_axis_mask:
            self.convert_shrink_axis_mask_to_shape_info()
        # create the nrf_axon_nn_compiler_strided_slice_parameters_s needed by axon
        # FIXME need to add handling for the ellipsis, new_axis and shrink accordingly
        if self.tflite_axon_enum_wrapper is not None:
            # get the tensors in the axon shape format which is NCHW
            axon_axis_enum_dict = self.tflite_axon_enum_wrapper.GetAxonAxisEnumDict()
            self.SetStridedSliceTensorsToAxonAxisShape(axon_axis_enum_dict)
        else:  # continue to use the default shape for axon with NCHW
            self.begin_axon = TensorShape(self.begin).get_axon_shape()
            self.end_axon = TensorShape(self.end).get_axon_shape()
            self.stride_axon = TensorShape(self.stride).get_axon_shape()
            self.SetStridedSliceFilterTensors()
        # check if this strided slice can be a reshape op
        ip_length = self.ip_shape.get_length()
        op_length = self.op_shape.get_length()
        # and operator_graph_info['operator_support']==OperatorSupportEnum.PASSTHROUGH:
        if ip_length == op_length:
            operator_graph_info['operator_support'] = OperatorSupportEnum.PASSTHROUGH
            self.operator_name = "RESHAPE"
            self.error = True
            self.error_action = "CONTINUE"
            self.error_text = "STRIDED_SLICE is converted to RESHAPE"
            # operation_detail['op_name'] = "RESHAPE"

    def SetStridedSliceFilterTensors(self):
        self.stride_slice_filter_tensor = []
        self.stride_slice_filter_tensor.extend(self.begin_axon)
        self.stride_slice_filter_tensor.extend(self.end_axon)
        self.stride_slice_filter_tensor.extend(self.stride_axon)
        self.stride_slice_filter_tensor = np.array(
            self.stride_slice_filter_tensor, dtype=np.int32)
        stride_filter_array = self.stride_slice_filter_tensor.reshape(
            1, len(self.stride_slice_filter_tensor))
        self.kernel_shape = TensorShape(np.array(stride_filter_array.shape))
        self.filter_tensor = np.asarray(
            self.stride_slice_filter_tensor, dtype=self.stride_slice_filter_tensor.dtype)

    def PrintAttributes(self):
        self.meta_data = f"begin mask : {self.begin_mask}, end mask : {self.end_mask}, ellipsis mask : {self.ellipsis_mask}, new axis mask : {self.new_axis_mask}, shrink axis mask : {self.shrink_axis_mask}"
        # print(self.meta_data)
        return self.meta_data

    def GetBeginMask(self):
        return self.begin_mask

    def GetEndMask(self):
        return self.end_mask

    def GetNewAxisMask(self):
        return self.new_axis_mask

    def GetShrinkAxisMask(self):
        return self.shrink_axis_mask

    def GetEllipsisMask(self):
        return self.ellipsis_mask

    def GetBeginTensor(self):
        return self.begin

    def GetEndTensor(self):
        return self.end

    def GetStrideTensor(self):
        return self.stride

    def GetBeginAxonTensor(self):
        return self.begin_axon

    def GetEndAxonTensor(self):
        return self.end_axon

    def GetStrideAxonTensor(self):
        return self.stride_axon

    def GetFilterShape(self):
        return self.kernel_shape.get_shape()

    def SetStridedSliceTensorsToAxonAxisShape(self, axon_axis_dict):
        self.begin_axon = TensorShape(
            self.begin).get_axon_axis_shape(axon_axis_dict)
        self.end_axon = TensorShape(
            self.end).get_axon_axis_shape(axon_axis_dict)
        self.stride_axon = TensorShape(
            self.stride).get_axon_axis_shape(axon_axis_dict)
        self.SetStridedSliceFilterTensors()

    def CalculateMultiplierandScaleshift(self):
        scale_q = self.ip_q['scales']/self.op_q['scales']
        result = util.optimized_ip_scaling_shift((scale_q), 8, 25, 25)
        # error = result[0]
        scaleshift = result[1]
        self.scale_multipliers = np.array(
            [abs(int(np.round((scale_q)*2**scaleshift)))])
        self.scale_shifts = np.array([scaleshift])

    def GetFilterTensor(self):
        # check if the filter tensors need to be transposed
        if self.transpose_kernel:
            begin_tensor = self.GetBeginAxonTensor()
            end_tensor = self.GetEndAxonTensor()
            stride_tensor = self.GetStrideAxonTensor()
            begin_tensor[1], begin_tensor[2] = begin_tensor[2], begin_tensor[1]
            end_tensor[1], end_tensor[2] = end_tensor[2], end_tensor[1]
            stride_tensor[1], stride_tensor[2] = stride_tensor[2], stride_tensor[1]
            rotate_filter_tensor = []
            rotate_filter_tensor.extend(begin_tensor)
            rotate_filter_tensor.extend(end_tensor)
            rotate_filter_tensor.extend(stride_tensor)
            rotate_filter_tensor = np.array(
                rotate_filter_tensor, dtype=np.int32)
            self.filter_tensor = rotate_filter_tensor
        return self.filter_tensor

    def CalculateBPrime(self):
        self.b_prime_tensor = np.array([])


class SplitVOptions(StridedSliceOptions):
    number_of_splits = None
    split_tensor = None
    axis = None

    @classmethod
    def get_number_of_splits(cls, operator):
        pos = operator.BuiltinOptions().Pos
        byte = operator.BuiltinOptions().Bytes
        options = tflite.SplitVOptions()
        options.Init(byte, pos)
        return options.NumSplits()

    @classmethod
    def record_op_ndx_for_node(cls, graph, index):
        graph['splitv_op_ndx'] = index
        return graph

    @classmethod
    def record_begin_end_values(cls, graph, begin, end):
        graph['begin_value'] = begin
        graph['end_value'] = end
        return graph

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.option = tflite.SplitVOptions()
        self.operator_name = "STRIDED_SLICE"
        self.axons_operation_enum = "NRF_AXON_NN_OP_STRIDED_SLICE"
        self.option.Init(self.bytes, self.pos)
        self.number_of_splits = self.option.NumSplits()
        self.kernel_bitwidth = np.int32
        self.kernel_bytewidth_enum = tflite_axon_enum_wrapper.GetAxonByteWidthEnum(
            self.kernel_bitwidth)
        self.split_tensor = tflite_interpreter.get_tensor(self.ip_ndxs[1])
        self.axis = tflite_interpreter.get_tensor(self.ip_ndxs[2])
        self.begin = [0] * self.ip_shape.shape_size
        self.end = self.ip_shape.get_shape()
        self.stride = [1] * self.ip_shape.shape_size
        if self.axis < 0:
            self.axis += self.ip_shape.shape_size
        if operator_graph is not None:
            # operator_graph[operation_detail['index']]
            operator_graph_info = operation_detail
            op_ndx = operator_graph_info['splitv_op_ndx']
            self.op_shape = TensorShape(
                tensor_details[operator.OutputsAsNumpy()[op_ndx]]['shape'])
            self.begin[self.axis] = operator_graph_info['begin_value']
            self.end[self.axis] = operator_graph_info['end_value']

        if self.tflite_axon_enum_wrapper is not None:
            # get the tensors in the axon shape format which is NCHW
            axon_axis_enum_dict = self.tflite_axon_enum_wrapper.GetAxonAxisEnumDict()
            self.SetStridedSliceTensorsToAxonAxisShape(axon_axis_enum_dict)
        else:  # continue to use the default shape for axon with NCHW
            self.begin_axon = TensorShape(self.begin).get_axon_shape()
            self.end_axon = TensorShape(self.end).get_axon_shape()
            self.stride_axon = TensorShape(self.stride).get_axon_shape()
            self.SetStridedSliceFilterTensors()

    def PrintAttributes(self):
        self.meta_data = f"Converted SPLITV with attributes\nsplit axis : {self.axis}, split tensor : {self.split_tensor}, number of splits : {self.number_of_splits}"
        # print(self.meta_data)
        return self.meta_data


class MultiplyOptions(OperatorOptions):
    multiply_with_constant = None
    constant_ip_q = None

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, model_wrapper_ffi):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, model_wrapper_ffi)
        self.option = tflite.MulOptions()
        self.multiply_with_constant = False
        self.option.Init(self.bytes, self.pos)
        # FIXME use the TFLITE_OP_AXON_OP_MAP to get the enum name of the operation
        self.axons_operation_enum = "NRF_AXON_NN_OP_MULTIPLY"
        # determine here if the multiply is happening between two layer outputs or with a constant
        if operator_graph is not None:
            operator_graph_info = operator_graph[operation_detail['index']]
            # the layer inputs and the number of tensor inputs to the multiply operation must be same
            # if they are not, the multiplication is happening with a constant
            if (len(operator_graph_info['inputs']) == 1) and (len(operator_graph_info['ip_tensors']) == 2):
                self.multiply_with_constant = True
                # populate the filter tensor with the constant value
                self.filter_tensor = tflite_interpreter.get_tensor(
                    operator_graph_info['ip_tensors'][1])
                self.kernel_shape = TensorShape(self.filter_tensor.shape)
                #get the input zero point of the constant tensor here.
                self.constant_ip_q = copy.deepcopy(tensor_details[operator_graph_info['ip_tensors'][1]]['quantization_parameters'])

    def CalculateMultiplierandScaleshift(self):
        scale_q = (self.ip_q['scales'] *
                   self.w_q['scales']) / self.op_q['scales']
        # zero_point_max = max(
        #     abs(self.ip_q['zero_points']), abs(self.w_q['zero_points']), abs(self.op_q['zero_points']))
        zero_point_max = abs(self.op_q['zero_points'])
        result = util.optimized_ip_scaling_shift(
            (scale_q), 8, 31, 30, zero_point_max)
        scaleshift = result[1]
        self.scale_multipliers = np.array(
            [abs(int(np.round((scale_q)*2**scaleshift)))])
        self.scale_shifts = np.array([scaleshift])
    
    def GetIpOpZeropoints(self):
        if self.multiply_with_constant:
            return self.constant_ip_q['zero_points'], self.op_q_zeropoint
        return self.ip_q_zeropoint, self.op_q_zeropoint


class PersistentVariableOpOptions(OperatorOptions):

    @classmethod
    def CreateOptionsObject(cls, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph=None, tflite_axon_enum_wrapper=None):
        return cls(operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)

    def InitOperatorOption(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        operator_options = operator.BuiltinOptions()
        if (operator_options is None):
            from flatbuffers.table import Table
            operator_options = Table(bytearray(), 0)
        self.bytes = operator_options.Bytes
        self.pos = operator_options.Pos
        self.interpreter = tflite_interpreter
        self.operation_detail = operation_detail
        self.operator_info_graph = operator_graph
        self.operator = operator
        # self.operation_detail["op_name"]
        self.operator_name = "PERSISTENT_VARIABLE"
        self.kernel_bytewidth_enum = tflite_axon_enum_wrapper.GetAxonByteWidthEnum(
            self.kernel_bitwidth)
        self.pad_info = PadDetails()

    def CalculateMultiplierandScaleshift(self):
        scale_q = self.ip_q['scales'] / \
            self.op_q['scales']  # , self.w_q['scales']
        result = util.optimized_ip_scaling_shift((scale_q), 8, 25, 25)
        # error = result[0]
        scaleshift = result[1]
        self.scale_multipliers = np.array(
            [abs(int(np.round((scale_q)*2**scaleshift)))])
        self.scale_shifts = np.array([scaleshift])

    def GetInputTensorsNdx(self):
        return self.ip_ndxs

    def GetOutputTensorsNdx(self):
        return self.op_ndxs


class VarHandleOptions(PersistentVariableOpOptions):
    container_name = ""
    shared_name = ""

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.option = tflite.VarHandleOptions()
        self.axons_operation_enum = "NRF_AXON_NN_OP_PERSISTENT_VAR"
        self.option.Init(self.bytes, self.pos)
        self.container_name = self.option.Container()
        self.shared_name = self.option.SharedName()
        if operator_graph is not None:
            operator_graph_info = operator_graph[operation_detail['index']]
            self.ip_ndxs = copy.deepcopy(operator_graph_info['ip_tensors'])
            self.op_ndxs = copy.deepcopy(operator_graph_info['op_tensors'])
            self.ip_shape = TensorShape(
                tensor_details[operator_graph_info['ip_tensors'][0]]['shape'])
            self.op_shape = TensorShape(
                tensor_details[operator_graph_info['op_tensors'][0]]['shape'])
            self.ip_q = copy.deepcopy(
                tensor_details[self.ip_ndxs[0]]['quantization_parameters'])
            self.op_q = copy.deepcopy(
                tensor_details[self.op_ndxs[0]]['quantization_parameters'])
            self.ip_bitwidth = tensor_details[self.ip_ndxs[0]]['dtype']
            self.op_bitwidth = tensor_details[self.op_ndxs[0]]['dtype']
            self.ip_q_zeropoint = self.ip_q['zero_points']
            self.op_q_zeropoint = self.op_q['zero_points']

    def PrintAttributes(self):
        self.meta_data = f"container name : {self.container_name}, shared_name : {self.shared_name}"
        # print(self.meta_data)
        return self.meta_data

    def WriteOperatorAttributesToFile(self, file_string, info_string, last_op_name):
        file_string += info_string + "_SHARED_NAME "+f"{self.shared_name}"
        return file_string

    def GetSharedName(self):
        return self.shared_name

    def GetContainerName(self):
        return self.container_name


class ReadVariableOptions(PersistentVariableOpOptions):

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.option = tflite.ReadVariableOptions()
        self.axons_operation_enum = "NRF_AXON_NN_OP_PERSISTENT_VAR"
        self.option.Init(self.bytes, self.pos)


class AssignVariableOptions(PersistentVariableOpOptions):

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.option = tflite.AssignVariableOptions()
        self.axons_operation_enum = "NRF_AXON_NN_OP_PERSISTENT_VAR"
        self.option.Init(self.bytes, self.pos)


class ReshapeOptions(OperatorOptions):

    @classmethod
    def determine_reshape_is_passthrough(cls, ip_shape, op_shape, shape, operator_list, op_index):
        """
        Reshapes are to be determined if they are passthroughs in the executor depending on pre-defined conditions.
        These conditions are as described, if they do not fall under any of the conditions they will be treated as reshapes and handled by the compiler
        1. The first reshape and the last reshape in model.
        2. If the next operation is a FC and the reshape is changing the channel.
        3. If the reshape is performing a transpose on the hxw and preserves the channel information with one dimension being 1 of either h or w.
        """
        if op_index == 0 or op_index == (len(operator_list)-1) or ( 0 in operator_list[op_index]['inputs']): #first or last op is reshape, treated as passthorugh
            return True                

        #shape dimensions remain the same
        if ip_shape.shape_size == op_shape.shape_size :
            #check if the shape sizes are not changing and the dimensions essentially remain the same
            if TensorShape.shapes_are_same(ip_shape, op_shape):
                return True

            if ip_shape.depth == op_shape.depth:
                if ( ip_shape.height == op_shape.width and ip_shape.width == 1 and op_shape.height == 1) or \
                ( op_shape.height == ip_shape.width and op_shape.width == 1 and ip_shape.height == 1) :
                    return True

        #shape dimensions are decreasing
        if ip_shape.shape_size > op_shape.shape_size :

            #check if the shape sizes are reducing but the dimensions essentially remain the same
            if TensorShape.shapes_are_same(ip_shape, op_shape):
                return True

            if ip_shape.depth == op_shape.width and ip_shape.shape_size == 4 and op_shape.shape_size < 4:
                return True
        
            if operator_list[op_index+1]['op_name']=="FULLY_CONNECTED":
                if ip_shape.shape_size > 2:
                    if ip_shape.depth > 1:
                        return True
                                        
        #shape dimensions are increasing
        if ip_shape.shape_size < op_shape.shape_size :
            
            #check if the shape sizes are increasing but the dimensions essentially remain the same
            if TensorShape.shapes_are_same(ip_shape, op_shape):
                return True

            if op_shape.shape_size == 4 and ip_shape.shape_size == 2 :
                if ( op_shape.height == 1 and op_shape.width == 1 and op_shape.depth == ip_shape.get_length() ) :                    
                    return True

        if ip_shape.depth == op_shape.depth and (ip_shape.width % 4 == 0) and (op_shape.width % 4 == 0):
            return True

        return False
    
    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        
        #determine if this is a reshape operator
        if operation_detail is not None:
            if operation_detail['operator_support'] == OperatorSupportEnum.PASSTHROUGH:
                raise KeyError(-917)
            
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.option = tflite.ReshapeOptions()
        self.axons_operation_enum = "NRF_AXON_NN_OP_RESHAPE"
        self.option.Init(self.bytes, self.pos)

    def CalculateMultiplierandScaleshift(self):
        scale_q = self.ip_q['scales'] / \
            self.op_q['scales']  # , self.w_q['scales']
        result = util.optimized_ip_scaling_shift((scale_q), 8, 25, 25)
        # error = result[0]
        scaleshift = result[1]
        self.scale_multipliers = np.array(
            [abs(int(np.round((scale_q)*2**scaleshift)))])
        self.scale_shifts = np.array([scaleshift])


class CpuOperatorOptions(OperatorOptions):
    cpu_extension_object = None

    def __init__(self, operator_code, operator, operation_detail, tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper):
        self.InitOperatorOption(operator_code, operator, operation_detail,
                                tensor_details, tflite_interpreter, operator_graph, tflite_axon_enum_wrapper)
        self.operand_str = "filters"
        self.operator_name = "CPU_" + self.operation_detail["op_name"]
        self.axons_operation_enum = "NRF_AXON_NN_OP_FIRST_EXTENSION"
        self.custom_cpu_op = True
        self.cpu_extension_object = cpu_operator_options.InitializeCpuOperator(
            operator_code)
        if self.cpu_extension_object is None:
            raise KeyError(-922)
        if self.cpu_extension_object.HandleOperatorOptions(self) < 0:
            raise KeyError(-923)
        self.axons_operation_enum = self.cpu_extension_object.GetAxonsCpuExtensionOpEnumName()


class OperatorSupportEnum(Enum):
    VARIABLE = "VARIABLE"
    SUPPORTED = "SUPPORTED"
    # operators which are passthrough but have some options to be used in ops before or after them
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    PASSTHROUGH = "PASSTHROUGH"  # operators which are passthrough
    # Not suported operators and an error will be thrown if found
    NOT_SUPPORTED = "NOT_SUPPORTED"
    # part of the VARIABLE OP TYPE which is now combined into a persistent variable
    COMBINED_VARIABLE = "COMBINED_VARIABLE"
    # Op that was forced to be passthrough after processing and realizing it cannot be an operation
    CONVERTED_PASSTHROUGH = "CONVERTED_PASSTHROUGH"


class SupportedOperators():
    supported_operators = {}
    variable_operators = {}
    nodes_info = {}
    pass_through_operators = []
    operators_detail_graph = []
    model_input_tensor_indices = []
    model_output_tensor_indices = []
    pass_through_ops_present = None
    variable_ops_present = None
    split_ops_present = None
    model_input_info = None
    model_output_info = None
    tflite_interpreter = None
    transposed_model = None    

    @classmethod
    def get_index_from_tf_index(cls, op_graph, tf_index):
        if tf_index >= 0:
            for ndx, n in enumerate(op_graph):
                if tf_index == n['index']:
                    return ndx
        return -1

    @classmethod
    def get_next_op_graph_index_for_axon_layer_num(cls, op_graph, op):
        # get the op connected to this op
        connected_op_layer_num = op['axon_op_ops']
        # Loop through each dictionary in the list
        next_op_ndx = []
        for output in connected_op_layer_num:
            for ndx, d in enumerate(op_graph):
                # Check if the key exists and the value matches
                if "axon_layer_num" in d and d['axon_layer_num'] == output:
                    next_op_ndx.append(ndx)
        connected_op_tensor = op['op_tensors']
        # next_op_connected_tensor_ndx = []
        for op_t in connected_op_tensor:
            for ndx, n in enumerate(op_graph):
                # check if the op_tensor is present in this node
                if op_t in n['ip_tensors']:
                    # check what type of axon support this operator has
                    if n['operator_support'] != OperatorSupportEnum.NOT_SUPPORTED:
                        # store the index
                        next_op_ndx.append(ndx)
        return list(dict.fromkeys(next_op_ndx))

    def __init__(self, tflite_interpreter=None, transpose_model=False):
        # self.supported_operators = {tflite.BuiltinOperator.CONV_2D:Conv2dOperatorOptions.CreateOptionsObject,
        #                             tflite.BuiltinOperator.DEPTHWISE_CONV_2D:DepthwiseConv2DOperatorOptions.CreateOptionsObject,
        #                             #tflite.BuiltinOperator.SOFTMAX:SoftmaxOperatorOptions.CreateOptionsObject, #commenting out to test SOFTMAX as an example for CPU Operator
        #                             tflite.BuiltinOperator.FULLY_CONNECTED:FullyConnectedOperatorOptions.CreateOptionsObject,
        #                             tflite.BuiltinOperator.AVERAGE_POOL_2D:Pool2DOptions.CreateOptionsObject,
        #                             tflite.BuiltinOperator.MAX_POOL_2D:Pool2DOptions.CreateOptionsObject,
        #                             tflite.BuiltinOperator.ADD:AddOptions.CreateOptionsObject,
        #                             tflite.BuiltinOperator.PAD:PadOptions.CreateOptionsObject,
        #                             tflite.BuiltinOperator.LEAKY_RELU:LeakyReluOptions.CreateOptionsObject,
        #                             tflite.BuiltinOperator.MEAN:MeanOperatorOptions.CreateOptionsObject,
        #                             # tflite.BuiltinOperator.QUANTIZE:MeanOperatorOptions.CreateOptionsObject,
        #                             # tflite.BuiltinOperator.DEQUANTIZE:MeanOperatorOptions.CreateOptionsObject,
        #                             tflite.BuiltinOperator.CONCATENATION:ConcatenationOptions.CreateOptionsObject,
        #                             tflite.BuiltinOperator.STRIDED_SLICE:StridedSliceOptions.CreateOptionsObject,
        #                             # tflite.BuiltinOperator.CALL_ONCE:MeanOperatorOptions.CreateOptionsObject,#NoOp Operator
        #                             tflite.BuiltinOperator.SPLIT_V:SplitVOptions.CreateOptionsObject,
        #                             }
        self.supported_operators = {tflite.BuiltinOperator.CONV_2D: Conv2dOperatorOptions,
                                    tflite.BuiltinOperator.DEPTHWISE_CONV_2D: DepthwiseConv2DOperatorOptions,
                                    tflite.BuiltinOperator.FULLY_CONNECTED: FullyConnectedOperatorOptions,
                                    tflite.BuiltinOperator.AVERAGE_POOL_2D: Pool2DOptions,
                                    tflite.BuiltinOperator.MAX_POOL_2D: Pool2DOptions,
                                    tflite.BuiltinOperator.ADD: AddOptions,
                                    tflite.BuiltinOperator.PAD: PadOptions,
                                    tflite.BuiltinOperator.LEAKY_RELU: LeakyReluOptions,
                                    tflite.BuiltinOperator.MEAN: MeanOperatorOptions,
                                    tflite.BuiltinOperator.CONCATENATION: ConcatenationOptions,
                                    tflite.BuiltinOperator.STRIDED_SLICE: StridedSliceOptions,
                                    tflite.BuiltinOperator.SPLIT_V: SplitVOptions,
                                    tflite.BuiltinOperator.MUL: MultiplyOptions,
                                    tflite.BuiltinOperator.RESHAPE:ReshapeOptions,
                                    }
        self.pass_through_operators = [#tflite.BuiltinOperator.RESHAPE,
                                       tflite.BuiltinOperator.QUANTIZE,
                                       tflite.BuiltinOperator.DEQUANTIZE,
                                       tflite.BuiltinOperator.CALL_ONCE,
                                       tflite.BuiltinOperator.TRANSPOSE,
                                       #  tflite.BuiltinOperator.PAD,
                                       # add more operators as needed
                                       ]
        self.variable_operators = {tflite.BuiltinOperator.VAR_HANDLE: VarHandleOptions,
                                   tflite.BuiltinOperator.READ_VARIABLE: ReadVariableOptions,
                                   tflite.BuiltinOperator.ASSIGN_VARIABLE: AssignVariableOptions,
                                   }
        # init the operator support status here
        self.operators_detail_graph = []
        self.pass_through_ops_present = False
        self.nodes_info = {}
        self.variable_ops_present = False
        self.split_ops_present = False
        self.model_input_tensor_indices = []
        self.model_output_tensor_indices = []
        self.model_input_info = None
        self.model_output_info = None
        if tflite_interpreter is not None:
            self.tflite_interpreter = tflite_interpreter
            self.model_input_info = tflite_interpreter.get_input_details()
            for ip_info in self.model_input_info:
                self.model_input_tensor_indices.append(ip_info['index'])

            self.model_output_info = tflite_interpreter.get_output_details()
            for op_info in self.model_output_info:
                self.model_output_tensor_indices.append(op_info['index'])
        self.transposed_model = transpose_model

    def add_operator_codes_to_supported_operators_list(self, code):
        if (code not in self.supported_operators):
            self.supported_operators[code] = CpuOperatorOptions

    def find_last_supported_operation_output(self, operation_list, subgraph, model):
        last_layer_ndx = -1
        for i in range(len(operation_list)):
            operators = subgraph.Operators(i)
            operation_code = model.OperatorCodes(
                (operators.OpcodeIndex())).BuiltinCode()
            if (operation_code in self.supported_operators):
                last_layer_ndx = i
        return last_layer_ndx

    def find_last_supported_operator_ndx(self, op_graph):
        last_layer_ndx = -1
        for ndx, operator in enumerate(op_graph):
            if operator['operator_support'] == OperatorSupportEnum.SUPPORTED and operator['axon_layer_num'] > 0:
                last_layer_ndx = ndx
        return last_layer_ndx

    def get_input_tensors_for_constraint_checking(self, inputs):
        ip_tensor_dict = {}
        for ndx, k in enumerate(inputs):
            if k >= 0:
                try:
                    tensor_details = self.tflite_interpreter.get_tensor_details()[
                        k]
                    if not (isinstance(tensor_details['dtype'], object) and (tensor_details['shape'].size == 0)):
                        tensor = self.tflite_interpreter.get_tensor(k)
                    else:
                        tensor = np.array([])
                except Exception as e:
                    tensor = np.array([])
                ip_tensor_dict[ndx] = tensor
        return ip_tensor_dict

    def generate_supported_operators_list(self, operation_list, subgraph, model):
        self.operators_detail_graph = copy.deepcopy(operation_list)
        model_supported = True
        tr_model_support = False
        not_supported_ops_dict = {}
        tr_model_support_text = {}
        constraint_index = 0
        # check here if the model input data width is supported by axon
        model_ip_datatype = self.model_input_info[0]['dtype']
        if model_ip_datatype != np.int8 and operation_list[0]['op_name'] != "QUANTIZE":
            return False, {0: f"input datatype {np.dtype(model_ip_datatype).name} is not supported "}, False, ""
        if self.transposed_model:
            constraint_index = 1
        for i in range(len(operation_list)):
            operator_supported = True
            layer_error_text = ""
            operators = subgraph.Operators(i)
            operation_code = model.OperatorCodes(
                (operators.OpcodeIndex())).BuiltinCode()
            # supported_ops_list[i] = (operation_code in self.supported_operators) or (operation_code in self.pass_through_operators)
            self.operators_detail_graph[i]["op_code"] = operation_code
            self.operators_detail_graph[i]["tflite_operator"] = operators
            self.operators_detail_graph[i]["custom_activation_output"] = None
            self.operators_detail_graph[i]["options_initialized"] = False
            self.operators_detail_graph[i]["operator_constraints"] = [
                (True, ""), (True, "")]
            if (operation_code in self.supported_operators):
                self.operators_detail_graph[i]["operator_support"] = OperatorSupportEnum.SUPPORTED
                self.operators_detail_graph[i]["operator_options"] = self.supported_operators[operation_code]
                if self.operators_detail_graph[i]['op_name'] == "LEAKY_RELU":
                    self.operators_detail_graph[i]["operator_support"] = OperatorSupportEnum.PARTIALLY_SUPPORTED
                    # a leaky relu is an operator for tflite, but we combine that to be an activation function for the previous operation
                    assert (i-1) >= 0, "LEAKY_RELU cannot be the first operation!"
                    self.operators_detail_graph[i -
                                                1]["custom_activation_output"] = self.operators_detail_graph[i]['outputs']
                elif self.operators_detail_graph[i]['op_name'] == "PAD":
                    # we need to check if the PAD operator has channel padding as channel_padding is fully supported as an operator
                    if not (PadOptions.is_channel_pad(self.tflite_interpreter, self.operators_detail_graph[i]['inputs'])):
                        self.operators_detail_graph[i]["operator_support"] = OperatorSupportEnum.PARTIALLY_SUPPORTED
                elif self.operators_detail_graph[i]['op_name'] == "STRIDED_SLICE":
                    if StridedSliceOptions.same_input_output_length(self.tflite_interpreter, self.operators_detail_graph[i]['inputs'],  self.operators_detail_graph[i]['outputs']):
                        self.operators_detail_graph[i]["operator_support"] = OperatorSupportEnum.PASSTHROUGH
                        self.pass_through_ops_present = True
                    # check here if the operation could be a passthrough operation?
                # "SPLIT"
                elif self.operators_detail_graph[i]['op_name'] == "SPLIT_V":
                    self.split_ops_present = True
                elif self.operators_detail_graph[i]['op_name'] == "RESHAPE":
                    ip_to_reshape = TensorShape(self.tflite_interpreter.get_tensor(self.operators_detail_graph[i]['inputs'][0]).shape)
                    op_of_reshape = TensorShape(self.tflite_interpreter.get_tensor(self.operators_detail_graph[i]['outputs'][0]).shape)
                    shape = self.tflite_interpreter.get_tensor(self.operators_detail_graph[i]['inputs'][1])
                    if ReshapeOptions.determine_reshape_is_passthrough(ip_to_reshape, op_of_reshape, shape,operation_list, i):
                        self.operators_detail_graph[i]["operator_support"] = OperatorSupportEnum.PASSTHROUGH
                        self.pass_through_ops_present = True
                    else:
                        operator_supported = False
                        self.operators_detail_graph[i]["operator_support"] = OperatorSupportEnum.NOT_SUPPORTED
                        layer_error_text = f"{self.operators_detail_graph[i]['op_name']} operator not supported "
            elif (operation_code in self.pass_through_operators):
                self.operators_detail_graph[i]["operator_support"] = OperatorSupportEnum.PASSTHROUGH
                # default class object
                self.operators_detail_graph[i]["operator_options"] = OperatorOptions
                self.pass_through_ops_present = True
            elif (operation_code in self.variable_operators):
                self.operators_detail_graph[i]["operator_support"] = OperatorSupportEnum.VARIABLE
                self.operators_detail_graph[i]["operator_options"] = self.variable_operators[operation_code]
                self.variable_ops_present = True
            else:
                operator_supported = False
                self.operators_detail_graph[i]["operator_support"] = OperatorSupportEnum.NOT_SUPPORTED
                layer_error_text = f"{self.operators_detail_graph[i]['op_name']} operator not supported "

            """
            commenting out the below code as the constraint checking is now part of the white list check implementation
            """
            # check for layer related constraints here and update the operator_supported flag and the layer_error_text
            # if not (self.operators_detail_graph[i]["operator_support"] == OperatorSupportEnum.NOT_SUPPORTED or self.operators_detail_graph[i]["operator_support"] == OperatorSupportEnum.PASSTHROUGH):
            #     # self.operators_detail_graph[i]["input_shapes"] = {ndx : subgraph.Tensors(k).ShapeAsNumpy() for ndx,k in enumerate(self.operators_detail_graph[i]['inputs'])}
            #     # input_tensors = {ndx: self.tflite_interpreter.get_tensor(k) for ndx, k in enumerate(
            #     #     self.operators_detail_graph[i]['inputs']) if k >= 0}
            #     input_tensors = self.get_input_tensors_for_constraint_checking(
            #         self.operators_detail_graph[i]['inputs'])
            #     # self.operators_detail_graph[i]["output_shapes"] = {ndx: subgraph.Tensors(k).ShapeAsNumpy() for ndx, k in enumerate(self.operators_detail_graph[i]['outputs'])}
            #     output_shapes = {ndx: subgraph.Tensors(k).ShapeAsNumpy() for ndx, k in enumerate(
            #         self.operators_detail_graph[i]['outputs']) if k >= 0}
            #     built_in_options = subgraph.Operators(i).BuiltinOptions()
            #     operator_class = self.operators_detail_graph[i]["operator_options"]
            #     self.operators_detail_graph[i]["operator_constraints"] = operator_class.check_for_constraints(
            #         operation_list[i]['op_name'], built_in_options, input_tensors, output_shapes)

            # if not self.transposed_model:  # check to see if the transposed model is supported?
            #     tr_model_support = tr_model_support and operator_supported and self.operators_detail_graph[
            #         i]["operator_constraints"][1][0]
            #     tr_model_support_text[i] = layer_error_text + \
            #         self.operators_detail_graph[i]["operator_constraints"][1][1]
            """
            end of commented code
            """
            model_supported = model_supported and operator_supported and self.operators_detail_graph[
                i]["operator_constraints"][constraint_index][0]
            not_supported_ops_dict[i] = layer_error_text + \
                self.operators_detail_graph[i]["operator_constraints"][constraint_index][1]

        return model_supported, not_supported_ops_dict, tr_model_support, tr_model_support_text

    def return_passthrough_operators_list(self):
        return self.pass_through_operators

    def get_graph_info_from_ops(self, operation_details):
        logger = logging.getLogger(__name__)
        # get the input array indexes and output array indexes
        # figure out if the output of an operation is being used in more than one input
        # if yes then that is a branch/track and it connects to another operation at some place
        op_nodes = {}
        merge_nodes = {}
        split_nodes = {}
        ip_nodes = {}
        new_graph = []

        # the first loop for graph creation will throw the exception if there is any operation that we do no support or allow to be passed through
        for ndx, operations in enumerate(operation_details):
            if self.operators_detail_graph[ndx]['operator_support'] != OperatorSupportEnum.NOT_SUPPORTED:
                if len(operation_details) > 1:
                    for ops_ndx in range(operations['index']+1, len(operation_details)):
                        _is_element_present = False
                        _is_element_present = any(
                            o in operation_details[ops_ndx]['inputs'] for o in operations['outputs'])
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
                _is_element_present = any(
                    i in operations['inputs'] for i in operation_details[ops_ndx]['outputs'])
                if (_is_element_present):
                    try:
                        ip_nodes[operations['index']].append(ops_ndx)
                    except KeyError:
                        ip_nodes[operations['index']] = [ops_ndx]

        """ DO NOT NEED TO FIND OUT THE SPLIT AND MERGE NODES RIGHT NOW THUS COMMENTING OUT    
    # max_track_count=-1
    # branches=0
    # for ops_ndx in op_nodes.keys():
    #   if(len(op_nodes[ops_ndx])>1):
    #     branches+=1
    #     for ops in op_nodes[ops_ndx]:
    #       try:
    #         split_nodes[ops_ndx].append(ops)
    #       except KeyError:
    #         split_nodes[ops_ndx] = [ops] 
    #   ops_connects = [(str(operation_details[ops_ndx]['op_name'])+"_"+str(ops_ndx)) for ops_ndx in op_nodes[ops_ndx]]
    #   logger.debug(f"{operation_details[ops_ndx]['op_name']}_{ops_ndx} connects to {ops_connects}")

    # for ops_ndx in ip_nodes.keys():
    #   if (len(ip_nodes[ops_ndx])>1):
    #     merges = [(str(operation_details[ops_ndx]['op_name'])+"_"+str(ops_ndx)) for ops_ndx in ip_nodes[ops_ndx]]
    #     logger.debug(f"{operation_details[ops_ndx]['op_name']}_{ops_ndx} merges {merges}")
    #     for ops in ip_nodes[ops_ndx]:
    #       try:
    #         merge_nodes[ops_ndx].append(ops)
    #       except KeyError:
    #         merge_nodes[ops_ndx] = [ops]
    """

        """
    new logic for getting a full node based input and output node graph using the graph and ip_nodes alone
    making a deepcopy as operators are referenced and we need to modify them as part of generating the new graph"
    """
        ops_graph = self.operators_detail_graph  # copy.deepcopy(operation_details)
        # new_graph_ndx=0
        for ndx, operations in enumerate(ops_graph):
            new_graph.append(operations)
            try:
                # find out if the node has an input from the first tensor or zero tensor
                new_graph[ndx]['ip_tensors'] = new_graph[ndx]['inputs']
                # zero is not the first input, is it something else? FIXME ONLY HANDLING ONE TENSOR INPUTS
                if (self.model_input_tensor_indices[0] in operation_details[ndx]['inputs']):
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
            if operations['op_name'] == "CONCATENATION":
                new_op_order = []
                for i, input_tensor in enumerate(operations['ip_tensors']):
                    for j, input_ops in enumerate(operations['inputs']):
                        if (new_graph[input_ops]['op_tensors'][0] == input_tensor):
                            new_op_order.append(input_ops)
                            break
                if new_op_order != operations['inputs']:
                    operations['inputs'] = new_op_order

            new_graph[ndx]['axon_ip_ops'] = copy.deepcopy(
                new_graph[ndx]['inputs'])
            new_graph[ndx]['axon_op_ops'] = copy.deepcopy(
                new_graph[ndx]['outputs'])
            """end of for loop for creating new graph"""

        # Add code here to figure out and combine the variable operators together and make them a persistent variable instead
        if self.variable_ops_present:
            for ndx, ops in enumerate(new_graph):
                if (ops['op_name'] == "VAR_HANDLE"):
                    for i in range(len(ops['outputs'])):
                        if new_graph[ops['outputs'][i]]['op_name'] == "READ_VARIABLE":
                            read_variable_op = new_graph[ops['outputs'][i]]
                        elif new_graph[ops['outputs'][i]]['op_name'] == "ASSIGN_VARIABLE":
                            assign_variable_op = new_graph[ops['outputs'][i]]
                        else:
                            raise Exception(
                                f"{ops['op_name']} @index {ndx} connected to an unsupported operator {new_graph[ops['outputs'][i]]['op_name']}")
                    logger.debug(f"{ops['op_name']} @index {ops['index']} connects read_variable_op : {read_variable_op['op_name']} and assign_variable_op : {assign_variable_op['op_name']} @ index {read_variable_op['index'],assign_variable_op['index']}")
                    # print(f"{ops['op_name']} @index {ops['index']} connects read_variable_op : {read_variable_op['op_name']} and assign_variable_op : {assign_variable_op['op_name']} @ index {read_variable_op['index'],assign_variable_op['index']}")

                    # get the write operators in the VAR_HANDLE
                    for ip_ndx in assign_variable_op['inputs']:
                        if new_graph[ip_ndx]['op_name'] != "VAR_HANDLE":
                            ops['axon_ip_ops'].append(ip_ndx)
                            writing_operator = new_graph[ip_ndx]
                            logger.debug(
                                f"{ops['op_name']} @index {ndx} is written by operator {writing_operator['op_name']} @index{writing_operator['index']}")
                            # print(f"{ops['op_name']} @index {ndx} is written by operator {writing_operator['op_name']} @index{writing_operator['index']}")
                            # have to update the writing operators op indices accordingly
                            for w_ndx, ops_ndx in enumerate(writing_operator['axon_op_ops']):
                                if ops_ndx == assign_variable_op['index']:
                                    writing_operator['axon_op_ops'][w_ndx] = ndx

                    new_ip_tensors = []
                    for ip_tensors_ndx in assign_variable_op['ip_tensors']:
                        if ops['op_tensors'][0] != ip_tensors_ndx:
                            # this is not the variable_ output tensor, add it to the input tensors of the var handle
                            new_ip_tensors.append(ip_tensors_ndx)

                    ops['ip_tensors'] = np.array(new_ip_tensors)
                    ops['axon_op_ops'] = read_variable_op['outputs']
                    reading_operator = new_graph[read_variable_op['outputs'][0]]
                    logger.debug(
                        f"{ops['op_name']} @index {ndx} is read by operator {reading_operator['op_name']} @index{reading_operator['index']}")
                    # print(f"{ops['op_name']} @index {ndx} is read by operator {reading_operator['op_name']} @index{reading_operator['index']}")
                    # have to update the reading operators op indices accordingly
                    for r_ndx, ops_ndx in enumerate(reading_operator['axon_ip_ops']):
                        if ops_ndx == read_variable_op['index']:
                            reading_operator['axon_ip_ops'][r_ndx] = ndx

                    ops['op_tensors'] = read_variable_op['op_tensors']
                    # change the read and write variables as pass through ops
                    read_variable_op['operator_support'] = OperatorSupportEnum.COMBINED_VARIABLE
                    assign_variable_op['operator_support'] = OperatorSupportEnum.COMBINED_VARIABLE
                    """
          # logic might be used for something else later on
          combined_ops = {
            'variable_handle' : ops['index'],
            'read_op' : read_variable_op['index'],
            'assign_op' : assign_variable_op['index']
          }
          ops['combined_ops'] = combined_ops
          read_variable_op['combined_ops'] = combined_ops
          assign_variable_op['combined_ops'] = combined_ops
          """
                    ops['combined_ops'] = [
                        read_variable_op['index'], assign_variable_op['index']]
                    read_variable_op['combined_ops'] = [
                        ops['index'], assign_variable_op['index']]
                    assign_variable_op['combined_ops'] = [
                        ops['index'], read_variable_op['index']]

        if self.split_ops_present:  # split v is present
            for ndx, ops in enumerate(new_graph):
                if ops['op_name'] == "SPLIT_V":
                    # get the number of splits
                    # number_of_splits = SplitVOptions.get_number_of_splits(ops['tflite_operator'])
                    split_op = new_graph.pop(ndx)
                    begin = 0
                    end = -1
                    offset = 0
                    # for i in range(number_of_splits):
                    record_index = -1
                    for i, op_o in enumerate(split_op['outputs']):
                        record_index = i
                        new_split_op = copy.deepcopy(split_op)
                        new_split_op['op_name'] = ops['op_name'] + \
                            f"_{i}_connects_{op_o}"
                        new_split_op = SplitVOptions.record_op_ndx_for_node(
                            new_split_op, record_index)
                        size = self.tflite_interpreter.get_tensor(
                            ops['ip_tensors'][1])[i]
                        begin = offset
                        end = offset+size
                        offset += size
                        new_split_op = SplitVOptions.record_begin_end_values(
                            new_split_op, begin, end)
                        new_split_op['axon_op_ops'] = [
                            split_op['axon_op_ops'][record_index]]
                        new_graph.insert(ndx+i, new_split_op)

            new_graph = self.update_graph_connections_after_node_insert(
                new_graph)

        # get the axon layer nums at this point, and update the graph with ops that are not being supported on axon currently
        axon_layer_num = -1
        for ndx, ops in enumerate(new_graph):
            if new_graph[ndx]['operator_support'] == OperatorSupportEnum.SUPPORTED or new_graph[ndx]['operator_support'] == OperatorSupportEnum.VARIABLE:
                axon_layer_num += 1
                new_graph[ndx]['axon_layer_num'] = axon_layer_num
            else:
                # indicates not supported by axon, is a no op or is a passthrough op
                new_graph[ndx]['axon_layer_num'] = -1
                if ops['operator_support'] == OperatorSupportEnum.PARTIALLY_SUPPORTED or ops['operator_support'] == OperatorSupportEnum.PASSTHROUGH:
                    """ OLD LOGIC
                    # if (ndx-1)>=0:
                    #   new_graph[ndx-1]['axon_op_ops'] = ops['axon_op_ops']
                    # # else: #the pass through operator is the first op?          
                    # if (ndx+1)<len(new_graph):
                    #   new_graph[ndx+1]['axon_ip_ops'] = ops['axon_ip_ops']
                    # # else: #the pass through operator is the last op?
                    """
                    for i, output_ndx in enumerate(ops['axon_op_ops']):
                        for j, output_operator_ndx in enumerate(new_graph[output_ndx]['axon_ip_ops']):
                            if output_operator_ndx == ndx:
                                if len(ops['axon_ip_ops']) == 0:
                                    new_graph[output_ndx]['axon_ip_ops'] = ops['axon_ip_ops']
                                else:
                                    new_graph[output_ndx]['axon_ip_ops'][j] = ops['axon_ip_ops'][0]

                    for i, input_ndx in enumerate(ops['axon_ip_ops']):
                        if input_ndx != -1:
                            for j, input_operator_ndx in enumerate(new_graph[input_ndx]['axon_op_ops']):
                                if input_operator_ndx == ndx:
                                    if len(ops['axon_op_ops']) == 0:
                                        new_graph[input_ndx]['axon_op_ops'] = ops['axon_op_ops']
                                    else:
                                        new_graph[input_ndx]['axon_op_ops'][j] = ops['axon_op_ops'][0]

            """ OLD LOGIC
      # if ops['operator_support']==OperatorSupportEnum.PASSTHROUGH or ops['operator_support']==OperatorSupportEnum.PARTIALLY_SUPPORTED:
      #   for op in new_graph[ndx+1:]:
      #     #reduce the index of the op accordingly
      #     op['index']-=1
      #     op['axon_ip_ops'] = [i - 1 if i >= new_graph[ndx]['index'] else i for i in op['axon_ip_ops']]
      #     op['axon_op_ops'] = [i - 1 if i >= new_graph[ndx]['index'] else i for i in op['axon_op_ops']]
      """

        # update the axon ips and ops using the axon layer num instead of the indices to create the node id graph
        for ndx, ops in enumerate(new_graph):
            if ops['operator_support'] == OperatorSupportEnum.SUPPORTED or ops['operator_support'] == OperatorSupportEnum.VARIABLE:
                ops['axon_ip_ops'] = [new_graph[i]['axon_layer_num']
                                      if i >= 0 else i for i in ops['axon_ip_ops']]
                ops['axon_op_ops'] = [new_graph[i]['axon_layer_num']
                                      if i >= 0 else i for i in ops['axon_op_ops']]

        self.nodes_info = {'op_nodes': op_nodes, 'ip_nodes': ip_nodes,
                           'split_nodes': split_nodes, 'merge_nodes': merge_nodes, 'op_graph': new_graph}
        return self.nodes_info

    def get_connected_operation_output_tensor_index(self, op_index, split_op_num, op_graph):
        split_op = op_graph[split_op_num]
        op_o = self.get_index_from_tf_index(
            op_graph, split_op['outputs'][op_index])
        for j, op_t in enumerate(split_op['op_tensors']):
            if op_t in op_graph[op_o]['ip_tensors']:
                return j

    def get_index_for_axon_layer_num(self, axon_layer_num):
        # Loop through each dictionary in the list
        for ndx, d in enumerate(self.nodes_info['op_graph']):
            # Check if the key exists and the value matches
            if "axon_layer_num" in d and d['axon_layer_num'] == axon_layer_num:
                return ndx

    def get_index_for_graph_index(self, graph_index):
        for ndx, node in enumerate(self.nodes_info['op_graph']):
            if node['index']==graph_index:
                return ndx

    def get_index_of_input_operator(self, current_op_index):
        for ndx, node in enumerate(self.nodes_info['op_graph']):
            if node['inputs'] == -1:
                #check if the current index is directly connected to the current op index
                if any( o in self.nodes_info['op_graph'][current_op_index]['ip_tensors'] for o in node['op_tensors']):
                    return ndx
                #they are connected indirectly, let us pass the index of the operator connected to the current op index instead to ge the input shapes
                current_node = self.nodes_info['op_graph'][ndx]
                next_node = self.nodes_info['op_graph'][current_node['outputs'][0]]
                while next_node['outputs'][0]!=current_op_index :
                    next_node = self.nodes_info['op_graph'][next_node['outputs'][0]]
                    continue                
                return self.get_index_for_graph_index(next_node['index'])
                



    def get_axon_layer_num_of_output_operator(self):
        if len(self.nodes_info['op_graph']) == 1:
            return 0, 0
        for ndx, node in enumerate(self.nodes_info['op_graph']):
            if node['axon_op_ops'] == [] and node['axon_layer_num'] >= 0:
                return node['axon_layer_num'], ndx

    def get_axon_layer_num_of_input_operator(self):
        for ndx, node in enumerate(self.nodes_info['op_graph']):
            if -1 in node['axon_ip_ops'] and node['axon_layer_num'] >= 0:
                return node['axon_layer_num'], ndx

    # def check_if_input_operator(self, node):
    #   if -1 in node['axon_ip_ops'] and node['axon_layer_num']>=0:
    #     return node['axon_layer_num'], node['index']

    # def check_if_output_operator(self, node):
    #   if node['axon_op_ops']==[] and node['axon_layer_num']>=0:
    #     return node['axon_layer_num'], node['index']

    def update_axon_operator_graph(self, converted_op):
        new_graph = self.nodes_info['op_graph']
        for node in new_graph:
            node['orig_ip_op'] = node['axon_ip_ops'][:]
            node['orig_op_op'] = node['axon_op_ops'][:]

        for ndx, ops in enumerate(new_graph):
            if ops['operator_support'] == OperatorSupportEnum.CONVERTED_PASSTHROUGH:
                for i, output_ndx in enumerate(ops['orig_op_op']):
                    for j, output_operator_ndx in enumerate(new_graph[output_ndx]['orig_ip_op']):
                        if output_operator_ndx == ndx:
                            if len(ops['axon_ip_ops']) == 0:
                                new_graph[output_ndx]['axon_ip_ops'] = ops['axon_ip_ops']
                            else:
                                new_graph[output_ndx]['axon_ip_ops'][j] = ops['axon_ip_ops'][0]

                for i, input_ndx in enumerate(ops['orig_ip_op']):
                    if input_ndx != -1:
                        for j, input_operator_ndx in enumerate(new_graph[input_ndx]['orig_op_op']):
                            if input_operator_ndx == ndx:
                                if len(ops['axon_op_ops']) == 0:
                                    new_graph[input_ndx]['axon_op_ops'] = ops['axon_op_ops']
                                else:
                                    new_graph[input_ndx]['axon_op_ops'][j] = ops['axon_op_ops'][0]
                ops['axon_layer_num'] = -1
                for i, operator in enumerate(new_graph[ndx+1:]):
                    if operator['operator_support'] == OperatorSupportEnum.SUPPORTED or operator['operator_support'] == OperatorSupportEnum.VARIABLE:
                        old_axon_layer_num = operator['axon_layer_num']
                        operator['axon_layer_num'] = operator['axon_layer_num'] - 1
                        # find all the places where the old_axon layer_num is mentioned and update it with the new axon_layer_num
                        for j, ip_ops_ndx in enumerate(operator['axon_ip_ops']):
                            ip_updating_operator = new_graph[self.get_index_for_axon_layer_num(
                                ip_ops_ndx)]
                            for k, op_ops_ndx in enumerate(ip_updating_operator['axon_op_ops']):
                                if old_axon_layer_num == op_ops_ndx:
                                    ip_updating_operator['axon_op_ops'][k] = operator['axon_layer_num']

                        for op_l, op_ops_ndx in enumerate(operator['axon_op_ops']):
                            op_updating_operator = new_graph[self.get_index_for_axon_layer_num(
                                op_ops_ndx)]
                            for m, ip_ops_ndx in enumerate(op_updating_operator['axon_ip_ops']):
                                if old_axon_layer_num == ip_ops_ndx:
                                    op_updating_operator['axon_ip_ops'][m] = operator['axon_layer_num']

                # Cleanup helper fields
        for node in new_graph:
            node.pop('orig_ip_op', None)
            node.pop('orig_op_op', None)

    def update_graph_connections_after_node_insert(self, graph):
        # n = len(graph)
        for node in graph:
            node['orig_ip_op'] = node['axon_ip_ops'][:]
            node['orig_op_op'] = node['axon_op_ops'][:]

        # Initialize empty connections
        for node in graph:
            node['axon_ip_ops'] = []
            node['axon_op_ops'] = []

        # Build mapping from index to all positions (index may not be unique)
        index_to_positions = {}
        for pos, node in enumerate(graph):
            index_to_positions.setdefault(node['index'], []).append(pos)

        for current_pos, node in enumerate(graph):
            input_indices = node.get('orig_ip_op', [])
            output_indices = node.get('orig_op_op', [])
            if len(input_indices) == 0:
                node['axon_ip_ops'] = input_indices
            else:
                if input_indices[0] == [-1]:
                    node['axon_ip_ops'] = [-1]
                else:
                    for input_idx in input_indices:
                        if input_idx in index_to_positions:
                            for candidate_pos in index_to_positions[input_idx]:
                                candidate_node = graph[candidate_pos]
                                if node['index'] in candidate_node.get('orig_op_op', []):
                                    node['axon_ip_ops'].append(candidate_pos)
                                    graph[candidate_pos]['axon_op_ops'].append(
                                        current_pos)
            if len(output_indices) == 0:
                node['axon_op_ops'] = output_indices
            else:
                for output_idx in output_indices:
                    if output_idx in index_to_positions:
                        for candidate_pos in index_to_positions[output_idx]:
                            candidate_node = graph[candidate_pos]
                            if node['index'] in candidate_node.get('orig_ip_op', []):
                                node['axon_op_ops'].append(candidate_pos)
                                graph[candidate_pos]['axon_ip_ops'].append(
                                    current_pos)

        for node in graph:
            if node['axon_ip_ops'] != [-1]:
                node['axon_ip_ops'] = list(dict.fromkeys(node['axon_ip_ops']))
            node['axon_op_ops'] = list(dict.fromkeys(node['axon_op_ops']))

        # Cleanup helper fields
        for node in graph:
            node.pop('orig_ip_op', None)
            node.pop('orig_op_op', None)
        return graph


class TfLiteAxonGraph(SupportedOperators):
    tflite_graph_dict = None
    axon_supported_operators_object = None
    operator_graph_info = None
    model_support_info_dict = None
    model_input_operator_name = None
    model_output_operator_name = None

    def __init__(self, tflite_filename):
        self.operator_graph_info = None
        self.model_support_info_dict = None
        self.model_input_operator_name = None
        self.model_output_operator_name = None
        self.axon_supported_operators_object = None
        self.tflite_graph_dict = {
            'model': None,
            'subgraph': None,
            'operators_length': None,
            'interpreter': None,
            'tflite_inputs': None,
            'tflite_outputs': None,
            'tflite_operation_details': None,
            'tflite_tensor_details': None,
        }
        with open(tflite_filename, "rb") as f:
            buf = f.read()
        self.tflite_graph_dict['model'] = tflite.Model.GetRootAsModel(buf, 0)
        self.tflite_graph_dict['subgraph'] = self.tflite_graph_dict['model'].Subgraphs(
            0)
        self.tflite_graph_dict['operators_length'] = self.tflite_graph_dict['subgraph'].OperatorsLength(
        )
        self.tflite_graph_dict['interpreter'] = tf.lite.Interpreter(
            model_path=tflite_filename, experimental_preserve_all_tensors=True)
        self.tflite_graph_dict['interpreter'].allocate_tensors()

    def get_tflite_graph_dict(self):
        return self.tflite_graph_dict

    def get_tflite_subgraph(self):
        return self.tflite_graph_dict['subgraph']

    def get_tflite_operators_len(self):
        return self.tflite_graph_dict['operators_length']

    def get_tflite_interpreter(self):
        return self.tflite_graph_dict['interpreter']

    def get_tflite_input_details(self):
        if self.tflite_graph_dict['tflite_inputs'] is None:
            self.tflite_graph_dict['tflite_inputs'] = self.tflite_graph_dict['interpreter'].get_input_details(
            )
        return self.tflite_graph_dict['tflite_inputs']

    def get_tflite_output_details(self):
        if self.tflite_graph_dict['tflite_outputs'] is None:
            self.tflite_graph_dict['tflite_outputs'] = self.tflite_graph_dict['interpreter'].get_output_details(
            )
        return self.tflite_graph_dict['tflite_outputs']

    def get_tflite_operator_details(self):
        if self.tflite_graph_dict['tflite_operation_details'] is None:
            # # This is an internal API and may break in future versions
            self.tflite_graph_dict['tflite_operation_details'] = self.tflite_graph_dict['interpreter']._get_ops_details()[
                0:self.tflite_graph_dict['operators_length']]
        return self.tflite_graph_dict['tflite_operation_details']

    def get_tflite_tensor_details(self):
        if self.tflite_graph_dict['tflite_tensor_details'] is None:
            self.tflite_graph_dict['tflite_tensor_details'] = self.tflite_graph_dict['interpreter'].get_tensor_details(
            )
        return self.tflite_graph_dict['tflite_tensor_details']

    def init_axon_supported_ops_object(self, transpose_model_flag=False, cpu_op_codes_list=None):
        # if self.axon_supported_operators_object is None:
        #     self.axon_supported_operators_object = SupportedOperators(self.tflite_graph_dict['interpreter'], transpose_model_flag)
        self.axon_supported_operators_object = SupportedOperators(
            self.tflite_graph_dict['interpreter'], transpose_model_flag)
        if cpu_op_codes_list is not None:
            self.add_cpu_ops_to_supported_axon_ops(cpu_op_codes_list)
        # also figure if the model operators are supported by axons here
        self.get_model_operators_axon_support_info()
        return self.axon_supported_operators_object

    def get_model_operators_axon_support_info(self):
        if self.model_support_info_dict is None:
            self.model_support_info_dict = {}
            (self.model_support_info_dict['model_supported'],
             self.model_support_info_dict['constraints'],
             self.model_support_info_dict['tr_model_supported'],
             self.model_support_info_dict['tr_constraints'],
             ) = self.axon_supported_operators_object.generate_supported_operators_list(
                self.get_tflite_operator_details(),
                self.tflite_graph_dict['subgraph'],
                self.tflite_graph_dict['model'])
        # #also check if the input and output data width are supported
        # model_ip_datawidth = self.get_tflite_input_details()[0]['dtype']
        # model_input_operator_name = self.get_tflite_operator_details()[0]['op_name']
        # if model_ip_datawidth != np.int8 and model_input_operator_name != "QUANTIZE":
        #     reason = f"model input is {model_ip_datawidth} which is not supported"
        #     self.model_support_info_dict['model_supported'] = self.model_support_info_dict['model_supported'] and False
        #     self.model_support_info_dict['constraints'][0] += reason
        return (self.model_support_info_dict['model_supported'],
                self.model_support_info_dict['constraints'],
                self.model_support_info_dict['tr_model_supported'],
                self.model_support_info_dict['tr_constraints'],
                )

    def add_cpu_ops_to_supported_axon_ops(self, cpu_op_codes):
        for ndx, cpu_op_code in enumerate(cpu_op_codes):
            self.axon_supported_operators_object.add_operator_codes_to_supported_operators_list(
                cpu_op_code)
            assert cpu_op_code in cpu_operator_options.cpu_operators_list, f"Handler function not added for {cpu_op_code} operator code"

    def get_axon_operator_graph_info(self):
        if self.operator_graph_info is None:
            assert self.axon_supported_operators_object is not None, "call 'init_axon_supported_ops_object' before getting graph info!"
            self.operator_graph_info = self.axon_supported_operators_object.get_graph_info_from_ops(
                self.get_tflite_operator_details())['op_graph']
        # assert self.axon_supported_operators_object is not None, "call 'init_axon_supported_ops_object' before getting graph info!"
        # self.operator_graph_info = self.axon_supported_operators_object.get_graph_info_from_ops(self.get_tflite_operator_details())['op_graph']
        return self.operator_graph_info

    def get_axon_layer_num_of_output_operator(self):
        return self.axon_supported_operators_object.get_axon_layer_num_of_output_operator()

    def get_index_for_axon_layer_num(self, axon_layer_num):
        return self.axon_supported_operators_object.get_index_for_axon_layer_num(axon_layer_num)

    def update_axon_operator_graph(self, converted_op):
        return self.axon_supported_operators_object.update_axon_operator_graph(converted_op)

    def get_index_of_input_operator(self, current_op_index):
        return self.axon_supported_operators_object.get_index_of_input_operator(current_op_index)

    def update_operator_options(self, op_graph, operator_options):
        op_graph['operator_options'] = operator_options
        op_graph['options_initialized'] = True
