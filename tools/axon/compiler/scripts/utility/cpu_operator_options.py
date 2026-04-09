"""
Copyright (c) 2026 Nordic Semiconductor

SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
"""

import tflite
import numpy as np
from utility import util as util
from utility import operator_options as ops_options
from abc import ABC, abstractmethod


class CpuExtension(ABC):
    cpu_extension_axon_enum_string = ""

    def __init__(self, enum_string):
        self.cpu_extension_axon_enum_string = enum_string

    @abstractmethod
    def HandleOperatorOptions(cls, operator_object):
        """
        Default handle for operations for which there is no handler written
        will just pass through, after printing a warning to console
        """
        print(
            f"ERROR : No Handle attached for the following {operator_object.GetOperationName()} operation")
        return -1

    @classmethod
    @abstractmethod
    def HandlePreviousOperatorAttributes(cls, operator_object):
        """
        Default handle for operations for which there is no handler written
        will just pass through, after printing a warning to console
        """
        print(
            f"ERROR : No Handle attached for changing the attributes of {operator_object.GetOperationName()} operation")
        return -1

    def GetAxonsCpuExtensionOpEnumName(self):
        return self.cpu_extension_axon_enum_string

    @classmethod
    def DetermineCpuExtensionSupport(cls, operator_object, op_list, op_index):
        """
        Default handle for operations for which there is no handler written
        will just pass through
        """
        return ops_options.OperatorSupportEnum.SUPPORTED


class CpuOpDummy(CpuExtension):
    cpu_extension_axon_enum_string = ""

    @classmethod
    def HandleOperatorOptions(self, operator_object):
        """
        perform all the operation needed for a specific custom operation needed, 
        populate all the fields as needed inside the operator object
        user should write code here to create the custom options object and get the information
        """
        # print("inside handle reshape operator option")
        # print(f"operation name : {operator_options_object.GetOperationName()}")

        # an example for the RESHAPE Operator is present but the user should update that accordingly
        # options = operator_object.SetOperatorOptionObject(tflite.ReshapeOptions())
        new_shape = 610  # dummy value
        operator_object.SetOperatorMetaAttributesString(
            f"Dummy Meta information as follows! {new_shape}")
        operator_object.SetOperationStrides(29, 42)
        operator_object.SetOperationPaddings(3, 7, 17, 19)

        reshape_custom_op_attrib_list = []
        # add more attribtutes as needed and present in the tflite->ReshapeOptions() object
        reshape_custom_op_attrib_list.append(new_shape)
        new_stride = 610*4  # dummy value
        reshape_custom_op_attrib_list.append(new_stride)
        operator_object.FillAdditionalCpuAttributes(
            reshape_custom_op_attrib_list)

        ip_q, w_q, bias_q = operator_object.GetIpQuantizationParameters()
        op_q = operator_object.GetOpQuantizationParameters()

        ip_zeropoint = ip_q['zero_points'].copy()
        ip_scales = ip_q['scales'].copy()
        op_scales = op_q['scales'].copy()
        op_zeropoint = op_q['zero_points'].copy()

        operator_object.SetIpQZeropoint(-12)

        ip_zeropoint, op_zeropoint = operator_object.GetIpOpZeropoints()

        # perform the necessary operations on the scaleshift values
        scale = ip_scales/op_scales
        error1, scale_shift_op_1 = util.optimized_ip_scaling_shift(
            scale, 8, 31, 31, op_q['zero_points'])
        scale_q = np.array(
            [abs(np.round(scale[0]*2**scale_shift_op_1)).astype(np.int32)])
        scale_shift = np.array([scale_shift_op_1], dtype=np.int8)
        operator_object.SetScalingValues(scale_q, scale_shift)

        # get the vector indices and the number of vectors
        # NOTE: We only support bprime, filters and scaleshifts+multipliers as vectors right now
        # find the size of the input tensors

        # input_tensor_ndxs = operator_options_object.GetInputTensorsNdx()
        # output_tensor_ndxs = operator_options_object.GetOutputTensorsNdx()
        # filter_tensor = interpreter.get_tensor(input_tensor_ndxs[1])
        # bias_tensor = interpreter.get_tensor(input_tensor_ndxs[2])

        # if it has three input tensors, they are usually input, filter/weight and biases
        # if only two, it could be just the input and the filter or some other vector information depending on the operator
        # the user should know apriori what those indexes are and how to use them
        # if there are no filter or bias buffers so nothing has to be done,
        # but a user can use the APIs to get and if needed perform the necessary math on them
        # and use the SetAPIs to set the filter tensors, bprime tensors and the scaleshift values

        # interpreter = operator_object.GetTfliteInterpreter()
        # input_tensor = operator_object.GetInputTensors()

        filter_tensor = np.array([])  # input_tensor[1]
        bias_tensor = np.array([])  # input_tensor[2]

        # perform some operations as needed on these vectors
        operator_object.SetFilterTensor(filter_tensor)
        operator_object.SetBPrimeTensor(bias_tensor)
        return -1

    @classmethod
    def HandlePreviousOperatorAttributes(cls, previous_operator_object):
        return -1

    @classmethod
    def DetermineCpuExtensionSupport(cls, operator_object, op_list, op_index):
        return ops_options.OperatorSupportEnum.NOT_SUPPORTED


class CpuSoftmax(CpuExtension):
    cpu_extension_axon_enum_string = "Softmax"

    def HandleOperatorOptions(self, operator_object):
        """
        perform all the operation needed for a specific custom operation needed, 
        populate all the fields as needed inside the operator object
        user should write code here to create the custom options object
        """
        options = operator_object.SetOperatorOptionObject(
            tflite.SoftmaxOptions())
        beta = options.Beta()
        operator_object.SetOperatorMetaAttributesString(f"beta value : {beta}")
        # set the input bitwidth to be 32 as softmax expects a q11.12 input
        operator_object.SetIpBitwidth(np.int32)
        # getting quantization parameters
        ip_q, w_q, bias_q = operator_object.GetIpQuantizationParameters()
        op_q = operator_object.GetOpQuantizationParameters()
        # ip_zeropoint = ip_q['zero_points'].copy()
        # ip_scales = ip_q['scales'].copy()
        op_scales = op_q['scales'].copy()
        op_zeropoint = op_q['zero_points'].copy()

        # performing needed scaling optimizations
        scale = 1/op_scales
        error1, scale_shift_op_1 = util.optimized_ip_scaling_shift(
            scale, 8, 31, 31, op_zeropoint)
        scale_q = np.array(
            [abs(np.round(scale[0]*2**scale_shift_op_1)).astype(np.int32)])
        scale_shift = np.array([scale_shift_op_1], dtype=np.int8)
        filter_tensor = np.array([])  # input_tensor[1]
        bias_tensor = np.array([])  # input_tensor[2]

        # perform some operations as needed on these vectors
        operator_object.SetFilterTensor(filter_tensor)
        # setting scale values
        operator_object.SetMultiplierandScaleshift(scale_q, scale_shift)
        # set the bias_prime tensor
        operator_object.SetBPrimeTensor(bias_tensor)
        # setting the input zero point
        operator_object.SetIpQZeropoint(0)
        return 0

        """
    example code to add extra attributes into the model description as arg list and arg count
    """
        # softmax_cpu_op_attrib_list = []
        # softmax_cpu_op_attrib_list.append(beta)
        # dummy_attribute1 = 1588420333
        # softmax_cpu_op_attrib_list.append(dummy_attribute1)
        # dummy_attribute2 = 195952365
        # softmax_cpu_op_attrib_list.append(dummy_attribute2)
        # operator_object.FillAdditionalCpuAttributes(softmax_cpu_op_attrib_list)
        """
    end of example
    """

    @classmethod
    def HandlePreviousOperatorAttributes(cls, previous_operator_object):
        if previous_operator_object.GetSkipSoftmaxOpFlag():
            return 1
        # set the activation function to be custom function
        if (not previous_operator_object.SetCustomActivationFunctionType("CustomPrepareSoftmax")):
            return -2
        # we have to calculate the maximum possible bitlimit we can shift to, so that we avoid overflow
        op_q = previous_operator_object.GetOpQuantizationParameters()
        op_max = np.ceil(op_q['scales'][0] * (127 - op_q['zero_points'][0]))
        op_min = np.ceil(op_q['scales'][0] * ((-128) - op_q['zero_points'][0]))
        # print(f"op_max {op_max}, op_min {op_min}")
        previous_operator_object.SetOpQScale(np.array([1]))
        previous_operator_object.SetOpQZeropoint(np.array([0]))
        ip_q, _, _ = previous_operator_object.GetIpQuantizationParameters()
        beta = 1  # FIXME get the beta value from the softmax operation somehow, which is the next operation and yet to be encountered
        ip_q_scale = ip_q['scales'] * beta
        previous_operator_object.SetIpQScale(ip_q_scale[0])
        scaleshift_max_range = 31 - \
            (len(bin(int(max(abs(op_max), abs(op_min))))) - 2)
        previous_operator_object.SetScaleshiftMaxRange(scaleshift_max_range)
        previous_operator_object.SetOpBitwidth(np.int32)
        return 0


class CpuActivation(CpuExtension):
    cpu_extension_axon_enum_string = "CpuActivationOpExtensionEnumString"

    def HandleOperatorOptions(self, operator_object):
        # options = None
        op_name = operator_object.GetOperationName()
        operator_object.SetOperatorMetaAttributesString(
            f"no attributes for {op_name}")

        # set the input bitwidth to be 32 as softmax expects a q11.12 input
        operator_object.SetIpBitwidth(np.int16)
        # getting quantization parameters
        ip_q, w_q, bias_q = operator_object.GetIpQuantizationParameters()
        op_q = operator_object.GetOpQuantizationParameters()
        # ip_zeropoint = ip_q['zero_points'].copy()
        # ip_scales = ip_q['scales'].copy()
        op_scales = op_q['scales'].copy()
        op_zeropoint = op_q['zero_points'].copy()

        # performing needed scaling optimizations
        scale = 1/op_scales
        error1, scale_shift_op_1 = util.optimized_ip_scaling_shift(
            scale, 8, 31, 31, op_zeropoint)
        scale_q = np.array(
            [abs(np.round(scale[0]*2**scale_shift_op_1)).astype(np.int32)])
        # scale_shift = np.array([scale_shift_op_1],dtype=np.int8) -12
        scale_shift = np.array([scale_shift_op_1], dtype=np.int8)
        filter_tensor = np.array([])  # input_tensor[1]
        bias_tensor = np.array([])  # input_tensor[2]

        # perform some operations as needed on these vectors
        operator_object.SetFilterTensor(filter_tensor)
        # setting scale values
        operator_object.SetMultiplierandScaleshift(scale_q, scale_shift)
        # set the bias_prime tensor
        operator_object.SetBPrimeTensor(bias_tensor)
        # setting the input zero point
        operator_object.SetIpQZeropoint(0)
        return 0

    @classmethod
    def HandlePreviousOperatorAttributes(cls, previous_operator_object):
        # set the activation function to be custom function
        if (not previous_operator_object.SetCustomActivationFunctionType("None")):
            return -2
        previous_operator_object.SetOpQScale(np.array([1]))
        previous_operator_object.SetOpQZeropoint(np.array([0]))
        previous_operator_object.SetScaleshiftMaxRange(28)
        previous_operator_object.SetOpBitwidth(np.int16)
        previous_operator_object.SetLayerOutputRadix(12)
        return 0


class CpuTanh(CpuActivation):
    cpu_extension_axon_enum_string = "Tanh"


class CpuSigmoid(CpuActivation):
    cpu_extension_axon_enum_string = "Logistic"


class CpuReshape(CpuExtension):
    cpu_extension_axon_enum_string = "Reshape"

    def determine_reshape_is_passthrough(ip_shape, op_shape, shape, operator_list, op_index):
        """
        Reshapes are to be determined if they are passthroughs in the executor depending on pre-defined conditions.
        These conditions are as described, if they do not fall under any of the conditions they will be treated as reshapes and handled by the compiler
        1. The first reshape and the last reshape in model.
        2. If the next operation is a FC and the reshape is changing the channel.
        3. If the reshape is performing a transpose on the hxw and preserves the channel information with one dimension being 1 of either h or w.
        """
        if op_index == 0 or op_index == (len(operator_list)-1) or (0 in operator_list[op_index]['inputs']):  # first or last op is reshape, treated as passthorugh
            return True

        # shape dimensions remain the same
        if ip_shape.shape_size == op_shape.shape_size:
            # check if the shape sizes are not changing and the dimensions essentially remain the same
            if ops_options.TensorShape.shapes_are_same(ip_shape, op_shape):
                return True

            if ip_shape.depth == op_shape.depth:
                if (ip_shape.height == op_shape.width and ip_shape.width == 1 and op_shape.height == 1) or \
                        (op_shape.height == ip_shape.width and op_shape.width == 1 and ip_shape.height == 1):
                    return True

        # shape dimensions are decreasing
        if ip_shape.shape_size > op_shape.shape_size:

            # check if the shape sizes are reducing but the dimensions essentially remain the same
            if ops_options.TensorShape.shapes_are_same(ip_shape, op_shape):
                return True

            if ip_shape.depth == op_shape.width and ip_shape.shape_size == 4 and op_shape.shape_size < 4:
                return True

            if operator_list[op_index+1]['op_name'] == "FULLY_CONNECTED":
                if ip_shape.shape_size > 2:
                    if ip_shape.depth > 1:
                        return True

        # shape dimensions are increasing
        if ip_shape.shape_size < op_shape.shape_size:

            # check if the shape sizes are increasing but the dimensions essentially remain the same
            if ops_options.TensorShape.shapes_are_same(ip_shape, op_shape):
                return True

            if op_shape.shape_size == 4 and ip_shape.shape_size == 2:
                if (op_shape.height == 1 and op_shape.width == 1 and op_shape.depth == ip_shape.get_length()):
                    return True

        if (ip_shape.depth == op_shape.depth) and (ip_shape.depth == 1) and (ip_shape.width % 4 == 0) and (op_shape.width % 4 == 0):
            return True

        return False

    def HandleOperatorOptions(self, operator_object):
        """
        perform all the operation needed for a specific custom operation needed, 
        populate all the fields as needed inside the operator object
        user should write code here to create the custom options object
        """
        options = operator_object.SetOperatorOptionObject(
            tflite.ReshapeOptions())
        # determine if this is a reshape operator
        if operator_object.operation_detail is not None:
            if operator_object.operation_detail['operator_support'] == ops_options.OperatorSupportEnum.PASSTHROUGH:
                operator_object.error = True
                operator_object.error_text = "RESHAPE is a PASSTHROUGH Operator"
                operator_object.error_action = "CONTINUE"
                return 1
        # getting quantization parameters
        ip_q, w_q, bias_q = operator_object.GetIpQuantizationParameters()
        op_q = operator_object.GetOpQuantizationParameters()

        # performing needed scaling optimizations
        scale_q = ip_q['scales'] / op_q['scales']
        result = util.optimized_ip_scaling_shift((scale_q), 8, 28, 28)
        # error = result[0]
        scaleshift = result[1]
        scale_multipliers = np.array(
            [abs(int(np.round((scale_q)*2**scaleshift)))])
        scale_shift = np.array([scaleshift])

        filter_tensor = np.array([])  # input_tensor[1]
        bias_tensor = np.array([])  # input_tensor[2]

        # perform some operations as needed on these vectors
        operator_object.SetFilterTensor(filter_tensor)
        # setting scale values
        operator_object.SetMultiplierandScaleshift(
            scale_multipliers, scale_shift)
        # set the bias_prime tensor
        operator_object.SetBPrimeTensor(bias_tensor)
        return 0

    @classmethod
    def HandlePreviousOperatorAttributes(cls, previous_operator_object):
        """
        nothing to be done with the previous operators with the RESHAPE Operator
        """
        return 0

    @classmethod
    def DetermineCpuExtensionSupport(cls, operator_object, op_list, op_index):
        ip_to_reshape = ops_options.TensorShape(operator_object.tflite_interpreter.get_tensor(
            operator_object.operators_detail_graph[op_index]['inputs'][0]).shape)
        op_of_reshape = ops_options.TensorShape(operator_object.tflite_interpreter.get_tensor(
            operator_object.operators_detail_graph[op_index]['outputs'][0]).shape)
        shape = operator_object.tflite_interpreter.get_tensor(
            operator_object.operators_detail_graph[op_index]['inputs'][1])
        if cls.determine_reshape_is_passthrough(ip_to_reshape, op_of_reshape, shape, op_list, op_index):
            return ops_options.OperatorSupportEnum.PASSTHROUGH
        return ops_options.OperatorSupportEnum.SUPPORTED


class CpuResizeNearestNeighbor(CpuExtension):
    cpu_extension_axon_enum_string = "ResizeNearestNeigbor"


    def HandleOperatorOptions(self, operator_object):
        """
        perform all the operation needed for a specific custom operation needed, 
        populate all the fields as needed inside the operator object
        user should write code here to create the custom options object
        ResizeNearsetNeighbor does not perform quantization/dequantization, nor does it have a bias.
        The primary attributes are output height and width, but these can be retreived from the actual output dimensions.
        Optional attributes are booleans align_corners and half_pixel_centers
        """
        options = operator_object.SetOperatorOptionObject(
            tflite.ResizeNearestNeighborOptions())
        align_corners = options.AlignCorners()
        half_pixel_centers = options.HalfPixelCenters()
        operator_object.SetOperatorMetaAttributesString(f"align_corners value : {align_corners}, half_pixel_centers value : {half_pixel_centers}")
        # @FIXME!! REMOVE THE 175; IT'S JUST THERE TO CONFIRM THAT THE 2 PRECEDING 0s ARE FOR THIS PURPOSE.
        operator_object.FillAdditionalCpuAttributes([align_corners, half_pixel_centers, 175])
        # getting quantization parameters
        ip_q, w_q, bias_q = operator_object.GetIpQuantizationParameters()
        op_q = operator_object.GetOpQuantizationParameters()

        # performing needed scaling optimizations
        scale_q = 1
        scaleshift = 0
        scale_multipliers = np.array(
            [abs(int(np.round((scale_q)*2**scaleshift)))])
        scale_shift = np.array([scaleshift])

        filter_tensor = np.array([])  # input_tensor[1]
        bias_tensor = np.array([])  # input_tensor[2]

        # perform some operations as needed on these vectors
        operator_object.SetFilterTensor(filter_tensor)
        # setting scale values
        operator_object.SetMultiplierandScaleshift(
            scale_multipliers, scale_shift)
        # set the bias_prime tensor
        operator_object.SetBPrimeTensor(bias_tensor)
        return 0

    @classmethod
    def HandlePreviousOperatorAttributes(cls, previous_operator_object):
        """
        nothing to be done with the previous operators with the RESHAPE Operator
        """
        return 0

    @classmethod
    def DetermineCpuExtensionSupport(cls, operator_object, op_list, op_index):
        return ops_options.OperatorSupportEnum.SUPPORTED

"""
add the function to get the custom attributes and other options for a specific cpu operator 
TODO this declaration and it's functions can be moved into a class and its object can be used to automatically add the cpu operators by reading the variable cpu operator list
"""
cpu_operators_list = {  # tflite.BuiltinOperator.RESHAPE: CpuReshapeDummy("DummyOp"),
    tflite.BuiltinOperator.SOFTMAX: CpuSoftmax("NRF_AXON_NN_OP_SOFTMAX"),
    tflite.BuiltinOperator.LOGISTIC: CpuSigmoid("NRF_AXON_NN_OP_SIGMOID"),
    tflite.BuiltinOperator.TANH: CpuTanh("NRF_AXON_NN_OP_TANH"),
    tflite.BuiltinOperator.RESHAPE: CpuReshape("NRF_AXON_NN_OP_RESHAPE"),
    tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR: CpuResizeNearestNeighbor("NRF_AXON_NN_OP_RESIZE_NEAREST_NEIGHBOR"),
}

"""
Global function that links and initializes the cpu_extension_op object to ensure that all the methods have been defined
"""


def InitializeCpuOperator(code):
    if (code in cpu_operators_list):
        # return the object created in the cpu_operators_list
        return cpu_operators_list[code]
    return None


"""
Global function that calls the classmethod to handle certain operations 
before the CPU Op extension to setup the correct quantization values, output bitwidth, etc
"""


def HandleOperatorAttributesBeforeCpuOp(current_operator_object, code):
    if (code in cpu_operators_list):
        return cpu_operators_list[code].HandlePreviousOperatorAttributes(
            current_operator_object)
    return CpuExtension.HandlePreviousOperatorAttributes(current_operator_object)


def DetermineOperatorSupportforCpuExtension(current_operator_object, code, op_list, op_index):
    if (code in cpu_operators_list):
        return cpu_operators_list[code].DetermineCpuExtensionSupport(
            current_operator_object, op_list, op_index)
    return CpuExtension.DetermineCpuExtensionSupport(current_operator_object, op_list, op_index)
