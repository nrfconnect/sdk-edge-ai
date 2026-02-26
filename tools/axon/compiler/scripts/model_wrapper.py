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
import re
import sys
import logging
import numpy as np
import datetime as dt
# import pycparser as pycparser

from cffi import FFI
from enum import Enum
from pathlib import Path
from utility import util as util

TFLITE_RANK4_AXON_AXIS_ENUM_MAP = {
    0: 'BATCH',
    1: 'NRF_AXON_NN_AXIS_HEIGHT',
    2: 'NRF_AXON_NN_AXIS_WIDTH',
    3: 'NRF_AXON_NN_AXIS_CHANNEL'}

TFLITE_RANK3_AXON_AXIS_ENUM_MAP = {
    0: 'NRF_AXON_NN_AXIS_HEIGHT',
    1: 'NRF_AXON_NN_AXIS_WIDTH',
    2: 'NRF_AXON_NN_AXIS_CHANNEL'}

TFLITE_AXON_ACTIVATION_ENUM_MAP = {
    "None":   'NRF_AXON_NN_ACTIVATION_FUNCTION_DISABLED',
    "ReLU":   'NRF_AXON_NN_ACTIVATION_FUNCTION_RELU',
    "ReLU6":   'NRF_AXON_NN_ACTIVATION_FUNCTION_RELU',
    "CustomPrepareSoftmax":   'NRF_AXON_NN_ACTIVATION_FUNCTION_PREPARE_SOFTMAX',
    "LeakyReLU":   'NRF_AXON_NN_ACTIVATION_FUNCTION_LEAKY_RELU'
}

TFLITE_AXON_BYTE_WIDTH_ENUM_MAP = {
    np.int8: "NRF_AXON_NN_BYTEWIDTH_1",
    np.uint8: "NRF_AXON_NN_BYTEWIDTH_1",
    np.int16: "NRF_AXON_NN_BYTEWIDTH_2",
    np.int32: "NRF_AXON_NN_BYTEWIDTH_4",
}


class ModelTfliteAxonEnumWrapper:
    axon_axis_enum = None
    axon_operation_enum = None
    axon_operation_dict = None
    axon_activation_enum = None
    axon_activation_dict = None
    axon_byte_width_enum = None
    axon_byte_width_dict = None
    axon_conv2d_setting_dict = None
    axon_conv2d_setting_enum = None
    axon_psum_placement_buffer_dict = None
    axon_psum_placement_buffer_enum = None
    model_wrapper_ffi_object = None

    def __init__(self, ffi_object):
        self.model_wrapper_ffi_object = ffi_object
        if self.model_wrapper_ffi_object is not None:
            self.axon_axis_dict = self.model_wrapper_ffi_object.typeof(
                'nrf_axon_nn_axis_e').relements
            self.axon_axis_enum = Enum('AxonAxis', self.axon_axis_dict)
            self.axon_operation_dict = self.model_wrapper_ffi_object.typeof(
                'nrf_axon_nn_op_e').relements
            self.axon_operation_enum = Enum(
                'AxonOperation', self.axon_operation_dict)
            self.axon_byte_width_dict = self.model_wrapper_ffi_object.typeof(
                'nrf_axon_nn_byte_width_e').relements
            self.axon_byte_width_enum = Enum(
                'AxonByteWidth', self.axon_byte_width_dict)
            self.axon_activation_dict = self.model_wrapper_ffi_object.typeof(
                'nrf_axon_nn_activation_function_e').relements
            self.axon_activation_enum = Enum(
                'AxonActivation', self.axon_activation_dict)
            self.axon_conv2d_setting_dict = self.model_wrapper_ffi_object.typeof(
                'nrf_axon_nn_conv2d_setting_e').relements
            self.axon_conv2d_setting_enum = Enum(
                'AxonActivation', self.axon_conv2d_setting_dict)
            self.axon_psum_placement_buffer_dict = self.model_wrapper_ffi_object.typeof(
                'nrf_axon_nn_psum_buffer_placement_e').relements
            self.axon_psum_placement_buffer_enum = Enum(
                'AxonActivation', self.axon_psum_placement_buffer_dict)

    def GetModelWrapperFfiObject(self):
        return self.model_wrapper_ffi_object

    def GetAxonByteWidthEnum(self, np_datatype):
        return self.axon_byte_width_enum[TFLITE_AXON_BYTE_WIDTH_ENUM_MAP[np_datatype]]

    def GetAxonOperationEnum(self, axon_op_enum_string):
        if axon_op_enum_string in self.axon_operation_dict:
            return self.axon_operation_enum[axon_op_enum_string]
        return Enum('OpNotSupported', {'OpNotSupported': -1})['OpNotSupported']

    def GetAxonActivationFunctionEnum(self, tflite_activation):
        return self.axon_activation_enum[TFLITE_AXON_ACTIVATION_ENUM_MAP[tflite_activation]]

    def get_bitwidth_from_bytewidth_enum(self, bytewidth_enum):
        # FIXME No support for bitwidth 4
        bitwidth = 0x3
        if (bytewidth_enum == self.axon_byte_width_enum.NRF_AXON_NN_BYTEWIDTH_2.value):
            bitwidth = 0x4
        elif (bytewidth_enum == self.axon_byte_width_enum.NRF_AXON_NN_BYTEWIDTH_4.value):
            bitwidth = 0x5
        return bitwidth

    def GetNpDataTypeFromAxonByteWidth(self, axon_byte_width):
        numpy_datatype = np.int8
        if axon_byte_width == self.axon_byte_width_enum.NRF_AXON_NN_BYTEWIDTH_2.value:
            numpy_datatype = np.int16
        elif axon_byte_width == self.axon_byte_width_enum.NRF_AXON_NN_BYTEWIDTH_4.value:
            numpy_datatype = np.int32
        return numpy_datatype

    def GetAxonByteWidthFromNpDataTypeEnum(self, np_datatype):
        """
        typedef enum {
        NRF_AXON_NN_BYTEWIDTH_1 = 1,
        NRF_AXON_NN_BYTEWIDTH_2 = 2,
        NRF_AXON_NN_BYTEWIDTH_4 = 4,
        } nrf_axon_nn_byte_width_e;
        """
        if np_datatype == np.int8 or np_datatype == np.uint8:
            axonpro_bytewidth = self.axon_byte_width_enum.NRF_AXON_NN_BYTEWIDTH_1
        elif np_datatype == np.int16 or np_datatype == np.uint16:
            axonpro_bytewidth = self.axon_byte_width_enum.NRF_AXON_NN_BYTEWIDTH_2
        elif np_datatype == np.int32 or np_datatype == np.uint32:
            axonpro_bytewidth = self.axon_byte_width_enum.NRF_AXON_NN_BYTEWIDTH_4
        return axonpro_bytewidth

    def GetAxonAxisEnumDict(self):
        return self.axon_axis_dict

    def GetAxonproConv2DSettingValue(self, conv_2d_setting):
        if (conv_2d_setting == 'local_psum'):
            return self.axon_conv2d_setting_enum.NRF_AXON_NN_CONV2D_SETTING_LOCAL_PSUM.value
        if (conv_2d_setting == 'inner'):
            return self.axon_conv2d_setting_enum.NRF_AXON_NN_CONV2D_SETTING_INPUT_INNER_LOOP.value
        if (conv_2d_setting == 'outer'):
            return self.axon_conv2d_setting_enum.NRF_AXON_NN_CONV2D_SETTING_INPUT_OUTER_LOOP.value
        return self.axon_conv2d_setting_enum.NRF_AXON_NN_CONV2D_SETTING_LOCAL_PSUM.value

    def GetAxonPsumBufferPlacementValue(self, psum_buffer_location):
        if (psum_buffer_location == "dedicated_memory"):
            return self.axon_psum_placement_buffer_enum.NRF_AXON_NN_PSUM_BUFFER_PLACEMENT_DEDICATED_MEM.value
        return self.axon_psum_placement_buffer_enum.NRF_AXON_NN_PSUM_BUFFER_PLACEMENT_INTERLAYER_BUFFER.value


def get_model_enum_dict_from_header_file(compiler_types_hdr_file_path, enum_name):
    filepath = Path(compiler_types_hdr_file_path)
    if (filepath.is_file()):
        with open(compiler_types_hdr_file_path, "r") as fo:
            header_text = fo.read()
        # Regex to match the enum block
        pattern = rf"typedef\s+enum\s*\{{([^}}]+)\}}\s*{enum_name}\s*;"
        match = re.search(pattern, header_text, re.MULTILINE)
        if not match:
            raise ValueError(f"Enum {enum_name} not found")

        body = match.group(1)
        items = []
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            # Remove inline comments and trailing commas
            line = re.sub(r'//.*', '', line)  # remove inline comments
            line = line.split('/*')[0]        # remove block comments (if any)
            line = line.strip().rstrip(',')
            if line:
                items.append(line)
        enum_dict = {}
        current_val = 0
        for item in items:
            if '=' in item:
                name, val = map(str.strip, item.split('=', 1))
                current_val = int(eval(val, {}, enum_dict))
            else:
                name = item
            enum_dict[name] = current_val
            current_val += 1
    return enum_dict


class CompilerResultsReturnClass():
    model_wrapper_object = None
    compiler_return_struct = None
    compiler_return_struct_np_buffer = None
    compiler_return_list = None
    compiler_return_fields = None
    compiler_return_dict = {}
    '''
  Memory Usage (in bytes)
  model_const_buffer_size:        1468
                interlayer_buffer_size: 1068
                psum_buffer_size:       0
                cmd_buffer_size:        7320
  Inference time (estimate, in ticks):	0
  '''

    def __init__(self, compiler_types_hdr_path):
        self.model_wrapper_object = get_model_wrapper_cffi_object(
            compiler_types_hdr_path)
        self.compiler_return_struct = get_compiler_return_struct(
            self.model_wrapper_object)
        self.compiler_return_struct_np_buffer = np.frombuffer(
            self.model_wrapper_object.buffer(self.compiler_return_struct), dtype=np.uint32)
        self.compiler_return_fields = self.model_wrapper_object.typeof(
            self.compiler_return_struct).item.fields

    def get_compiler_return_dict(self):
        for ndx, fields in enumerate(self.compiler_return_fields):
            self.compiler_return_dict[fields[0]] = self.compiler_return_struct_np_buffer.tolist()[
                ndx]
        return self.compiler_return_dict

    def get_compiler_return_buffer_size(self):
        return self.compiler_return_struct_np_buffer.size

    def get_compiler_return_struct_np_buffer(self):
        return self.compiler_return_struct_np_buffer

    def get_compiler_return_code_as_text(self, return_code):
        return self.model_wrapper_object.string(self.model_wrapper_object.cast("nrf_axon_compiler_result_e", return_code))


class ModelDescriptionBin:
    MAJOR_VER = 0
    MINOR_VER = 17
    PATCH_VER = 0
    binary_title_string = "AXON_INTERMEDIATE_REPRESENTATION_FILE"
    MODEL_BIN_VER = (MAJOR_VER << 16) + (MINOR_VER << 8) + PATCH_VER
    model_ver_bin = bytearray(np.array([MODEL_BIN_VER], dtype=np.uint32))

    model_data_bin = None
    model_desciption_info_header_offset = 0

    def __init__(self, model_wrapper_ffi_object, model_info_hdr_struct_size) -> None:
        self.model_data_bin = bytearray()
        self.model_desciption_info_header_offset = model_info_hdr_struct_size

    def append_bin_data(self, bin_data, model_info_struct=None, pack=False):
        padding = bytearray()
        length = len(bin_data)
        offset = self.model_desciption_info_header_offset + \
            len(self.model_data_bin)

        if (model_info_struct is not None):
            model_info_struct.offset = offset
            model_info_struct.length = length

        if (not pack and length % 4):
            pad = int(np.ceil(length/4)*4) - length
            padding = bytearray(np.zeros(pad, dtype=np.int8))
        self.model_data_bin += bin_data + padding
        return self.model_data_bin

    def get_model_bin(self, model_description_hdr_struct_bin):
        return model_description_hdr_struct_bin + self.model_data_bin

    def append_model_title(self, model_info_struct):
        self.append_string_data(self.binary_title_string, model_info_struct)

    def append_model_version(self, model_info_struct):
        self.append_bin_data(self.model_ver_bin, model_info_struct)

    def append_string_data(self, string_data, model_info_struct, add_null=True):
        if (add_null):
            string_data += '\0'
        string_bin_data = bytearray(bytes(string_data, encoding="utf-8"))
        self.append_bin_data(string_bin_data, model_info_struct)


def get_model_descriptor_info_struct(ffi_model_wrapper_object):
    return ffi_model_wrapper_object.new("nrf_axon_nn_model_desc_hdr_s model_desc[]", 1)


def prepend_length_to_bytearray(bin_data, pack=False):
    length = len(bin_data)
    # now check if this length is divisible by 4
    length_bin = bytearray()
    if (length % 4) and pack:
        pad = int(np.ceil(length/4)*4) - length
        bin_data.extend(bytearray(np.zeros(pad, dtype=np.int8)))
    length_bin.extend(np.array(len(bin_data), dtype=np.uint32))
    return length_bin + bin_data


def update_length_of_bytearray(bin_data):
    return bytearray(np.array([len(bin_data) - 4], dtype=np.int32)) + bin_data[4:]


def clear_model_descriptor_layer_struct(layer_descriptor):
    layer_descriptor[0].input_id_cnt = 0
    for i in range(4):  # FIXME add the macro definition here somehow
        layer_descriptor[0].input_ids[i] = 0
        layer_descriptor[0].input_dimensions[i].height = 0
        layer_descriptor[0].input_dimensions[i].width = 0
        layer_descriptor[0].input_dimensions[i].channel_cnt = 0
        layer_descriptor[0].input_dimensions[i].byte_width = 0
    # layer_descriptor[0].input_track_no = 0
    # layer_descriptor[0].input_merge_track_no = 0
    # layer_descriptor[0].nn_operation = 0
    # layer_descriptor[0].input_dimensions.height = 0
    # layer_descriptor[0].input_dimensions.width = 0
    # layer_descriptor[0].input_dimensions.channel_cnt = 0
    # layer_descriptor[0].input_dimensions.byte_width = 0
    layer_descriptor[0].filter_dimensions.height = 0
    layer_descriptor[0].filter_dimensions.width = 0
    layer_descriptor[0].filter_dimensions.channel_cnt = 0
    layer_descriptor[0].filter_dimensions.byte_width = 0
    layer_descriptor[0].output_dimensions.height = 0
    layer_descriptor[0].output_dimensions.width = 0
    layer_descriptor[0].output_dimensions.channel_cnt = 0
    layer_descriptor[0].output_dimensions.byte_width = 0
    layer_descriptor[0].stride_x = 0
    layer_descriptor[0].stride_y = 0
    layer_descriptor[0].output_zero_point = 0
    layer_descriptor[0].bias_prime.offset = 0
    layer_descriptor[0].output_multipliers.offset = 0
    layer_descriptor[0].scale_shifts.offset = 0
    layer_descriptor[0].scale_shift_cnt = 0
    layer_descriptor[0].activation_function = 0
    layer_descriptor[0].pad_left = 0
    layer_descriptor[0].pad_right = 0
    layer_descriptor[0].pad_top = 0
    layer_descriptor[0].pad_bottom = 0
    layer_descriptor[0].filter.offset = 0
    layer_descriptor[0].cpu_op_additional_attributes_count = 0
    layer_descriptor[0].cpu_op_additional_attributes.offset = 0

    return layer_descriptor


def release_ffi_object_memory(ffi_model_wrapper_object, object):
    return ffi_model_wrapper_object.release(object)


def get_bytearray_from_struct(ffi_model_wrapper_object, layer_descriptor):
    return bytearray(np.array(ffi_model_wrapper_object.buffer(layer_descriptor)))


def get_model_descriptor_layer_struct(ffi_model_wrapper_object, layer_cnt=1):
    return ffi_model_wrapper_object.new("nrf_axon_nn_model_layer_desc_s layer[]", layer_cnt)


def get_compiler_return_struct(ffi_model_wrapper_object):
    return ffi_model_wrapper_object.new("nrf_axon_nn_compiler_return_s compiler_return[]", 1)


def get_model_wrapper_cffi_object(path, include_dir=r"../include/"):
    logger = logging.getLogger(__name__)
    filepath = Path(path)
    if (filepath.is_file()):
        ffi = FFI()
        with open(path, "r") as fo:
            source_text = fo.read()
        ffi.cdef(source_text)
        return ffi
    logger.critical("compiler_api_filepath is invalid")
    return None
    #   # if(INCLUDE in source_text):
    #   #   find_and_import_includes(source_text, include_dir)
    #   # #find if there is an #include in the file,
    #   # #if there is, find the file and copy the code from the header file into the source_text
    #   # parsed_text = find_and_import_includes(source_text, include_dir)

    # """
    # approach to add the include header directly
    # result : works in a way that we still need to comment out the #define macros which do not have a fixed integer value
    #  and also remove any include directives
    # """
    # # ffi.set_source(module_name="test_parser",source=parsed_text)
    # include_dir = "pre-processed.h" #r"src\c_files\axonpro_api.h"
    # with open(include_dir, "r") as fo:
    #   parsed_hdr = fo.read()
    # # ast = pycparser.parse_file(include_dir, use_cpp=True,cpp_path='gcc')
    # # ffi.set_source("test",header_text )
    # # parsed_hdr = pycparser.preprocess_file(r"src\c_files\axonpro_api.h")
    # ffi.cdef(parsed_hdr+source_text)
    # return ffi


def get_model_meta_information_struct(ffi_model_wrapper_object):
    return ffi_model_wrapper_object.new("nrf_axon_nn_model_meta_info_s meta_info[]", 1)


def get_model_compilations_options_struct(ffi_model_wrapper_object):
    return ffi_model_wrapper_object.new("nrf_axon_nn_model_compilation_options_s compilation_options[]", 1)


def get_integer_value(binary_value, signed=False):
    return int.from_bytes(binary_value, byteorder='little', signed=signed)


def get_length_of_struct(ffi_object, struct):
    return len(get_bytearray_from_struct(ffi_object, struct))


def model_wrapper_test():
    ffi = get_model_wrapper_cffi_object()

    # image = ffi.new("test_pixel_t image[2]",)
    # image[0].r = 4
    # image[0].g = 5
    # image[0].b = 6
    # image[0].value8 = 7
    # image[0].value16_1 = -16658 # 0x3EEE
    # image[0].value16_2 = 0x6EAF
    # image[0].value32 = -559022355 #0xDEADFEED #int.from_bytes(0xDEADFEED, byteorder=sys.byteorder) #(np.int32(0xDEADFEED)) #
    # image[0].bitwidth = 2
    # image[1].r = 8
    # image[1].g = 9
    # image[1].b = 10
    # image[1].bitwidth = 4
    # image[1].value8 = 11
    # image[1].value16_1 = 0x2ADE
    # image[1].value16_2 = 0x3EAD
    # image[1].value32 = -1091576083 #0xBEEFDEED
    print(ffi.string(ffi.cast("nrf_axon_nn_byte_width_e", 7)))
    print(ffi.string(ffi.cast("nrf_axon_nn_byte_width_e", 8)))
    print(ffi.string(ffi.cast("nrf_axon_nn_op_e", 5)))
    # print(ffi.string(ffi.cast(ffi.list_types()[0][2], 6)))
    print(ffi.string(ffi.cast("nrf_axon_nn_op_e", 100)))
    # print(bytearray(np.array(ffi.buffer(image))))

    # ffi.new("ModelLayerDescription []",2)
    layer = get_model_descriptor_layer_struct(ffi, layer_cnt=2)
    layer[0].track_no = 1
    layer[0].input_track_no = 2
    layer[0].input_merge_track_no = 3
    layer[0].nn_operation = 6
    layer[0].input_dimensions.height = 96
    layer[0].input_dimensions.width = 96
    layer[0].input_dimensions.channel_cnt = 96
    layer[0].input_dimensions.byte_width = 4
    layer[0].filter_dimensions.height = 3
    layer[0].filter_dimensions.width = 3
    layer[0].filter_dimensions.channel_cnt = 16
    layer[0].filter_dimensions.byte_width = 2
    layer[0].output_dimensions.height = 48
    layer[0].output_dimensions.width = 48
    layer[0].output_dimensions.channel_cnt = 16
    layer[0].output_dimensions.byte_width = 4
    layer[0].stride_x = 1
    layer[0].stride_y = 2
    layer[0].output_zero_point = -128
    layer[0].bias_prime.offset = 0x10100101
    layer[0].output_multipliers.offset = 0x1DDDBBBB
    layer[0].scale_shifts.offset = 0x2FFFEEEE
    layer[0].scale_shift_cnt = 1
    layer[0].activation_function = 3
    layer[0].pad_left = 0
    layer[0].pad_right = 1
    layer[0].pad_top = 0
    layer[0].pad_bottom = 1
    layer[0].filter.offset = 0x1EADBEEF
    layer[1].track_no = 1
    layer[1].input_track_no = 2
    layer[1].input_merge_track_no = 3
    layer[1].nn_operation = 6
    layer[1].input_dimensions.height = 96
    layer[1].input_dimensions.width = 96
    layer[1].input_dimensions.channel_cnt = 96
    layer[1].input_dimensions.byte_width = 4
    layer[1].filter_dimensions.height = 3
    layer[1].filter_dimensions.width = 3
    layer[1].filter_dimensions.channel_cnt = 16
    layer[1].filter_dimensions.byte_width = 2
    layer[1].output_dimensions.height = 48
    layer[1].output_dimensions.width = 48
    layer[1].output_dimensions.channel_cnt = 16
    layer[1].output_dimensions.byte_width = 4
    layer[1].stride_x = 1
    layer[1].stride_y = 2
    layer[1].output_zero_point = -128
    layer[1].bias_prime.offset = 0x10100101
    layer[1].output_multipliers.offset = 0x1DDDBBBB
    layer[1].scale_shifts.offset = 0x2FFFEEEE
    layer[1].scale_shift_cnt = 1
    layer[1].activation_function = 3
    layer[1].pad_left = 0
    layer[1].pad_right = 1
    layer[1].pad_top = 0
    layer[1].pad_bottom = 1
    layer[1].filter.offset = 0x1EADBEEF
    print(get_bytearray_from_struct(ffi, layer))

    # ffi.new("ModelLayerDescription  layer[]", 1)
    layer2 = get_model_descriptor_layer_struct(ffi)
    layer2[0].track_no = 1
    layer2[0].input_track_no = 2
    layer2[0].input_merge_track_no = 3
    layer2[0].nn_operation = 6
    layer2[0].input_dimensions.height = 96
    layer2[0].input_dimensions.width = 96
    layer2[0].input_dimensions.channel_cnt = 96
    layer2[0].input_dimensions.byte_width = 4
    layer2[0].filter_dimensions.height = 3
    layer2[0].filter_dimensions.width = 3
    layer2[0].filter_dimensions.channel_cnt = 16
    layer2[0].filter_dimensions.byte_width = 2
    layer2[0].output_dimensions.height = 48
    layer2[0].output_dimensions.width = 48
    layer2[0].output_dimensions.channel_cnt = 16
    layer2[0].output_dimensions.byte_width = 4
    layer2[0].stride_x = 1
    layer2[0].stride_y = 2
    layer2[0].output_zero_point = -128
    layer2[0].bias_prime.offset = 0x10100101
    layer2[0].output_multipliers.offset = 0x1DDDBBBB
    layer2[0].scale_shifts.offset = 0x2FFFEEEE
    layer2[0].scale_shift_cnt = 1
    layer2[0].activation_function = 3
    layer2[0].pad_left = 0
    layer2[0].pad_right = 1
    layer2[0].pad_top = 0
    layer2[0].pad_bottom = 1
    layer2[0].filter.offset = 0x1EADBEEF

    print(get_bytearray_from_struct(ffi, layer2))


def TestModelBinFile(binary_file_path, compiler_types_header_path=r"../include/nrf_axon_nn_compiler_types.h"):
    log_file_name = r"logs/" + "debug_test_" + str(os.path.basename(__file__).split(
        '.')[0])+"_"+str(dt.datetime.now().strftime("%Y%m%d_%H%M%S"))+".log"
    log_level = logging.DEBUG
    log_format = "%(levelname)s: %(message)s"
    logging.basicConfig(filename=log_file_name,
                        level=log_level, format=log_format)
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    f = open(binary_file_path, "rb")
    ffi_object = get_model_wrapper_cffi_object(compiler_types_header_path)
    model_desc_struct = get_model_descriptor_info_struct(ffi_object)
    # hdr_struct_len = get_length_of_struct(ffi_object,model_desc_struct)
    bin_file_data = f.read()
    f.seek(0)

    # bin_file_title_len = get_integer_value(bin_file_data[:4])
    # bin_file_name = bin_file_data[4:bin_file_title_len]
    # bin_file_version = bin_file_data[bin_file_title_len:bin_file_title_len+4]
    # model_desc_hdr_offset = get_integer_value(bin_file_data[bin_file_title_len+4:bin_file_title_len+8])
    # model_desc_hdr_length = get_integer_value(bin_file_data[bin_file_title_len+8:bin_file_title_len+12])

    f.readinto(ffi_object.buffer(model_desc_struct))
    bin_file_name = bin_file_data[model_desc_struct[0].title.offset:
                                  model_desc_struct[0].title.offset+model_desc_struct[0].title.length]
    bin_file_version = bin_file_data[model_desc_struct[0].version.offset:
                                     model_desc_struct[0].version.offset+model_desc_struct[0].version.length]
    version_integer = get_integer_value(bin_file_version)
    major_ver = (get_integer_value(bin_file_version) >> 16) & 0xff
    minor_ver = (get_integer_value(bin_file_version) >> 8) & 0xff
    patch_ver = (get_integer_value(bin_file_version)) & 0xff
    logger.debug(f"{bin_file_name}:v{(major_ver)}.{minor_ver}.{patch_ver}")

    # print(f"model description header offset: {model_desc_hdr_offset}, length : {model_desc_hdr_length}")
    # logger.debug(f"model description header offset: {model_desc_hdr_offset}, length : {model_desc_hdr_length}")
    # f.seek(model_desc_hdr_offset)
    # f.readinto(ffi_object.buffer(model_desc_struct))
    f.seek(0)

    # model_meta_info = bin_file_data[model_desc_struct[0].meta.offset:model_desc_struct[0].meta.offset+model_desc_struct[0].meta.length]
    # layer_count = get_integer_value(model_layer_cnt)
    model_meta_info_struct = get_model_meta_information_struct(ffi_object)
    tflite_axon_enum_wrapper = ModelTfliteAxonEnumWrapper(ffi_object)
    f = open(binary_file_path, "rb")
    f.seek(model_desc_struct[0].meta.offset)
    f.readinto(ffi_object.buffer(model_meta_info_struct))
    f.seek(0)
    layer_count = model_meta_info_struct[0].model_layer_cnt
    model_ip_multiplier = model_meta_info_struct[0].input_quant.mult
    model_ip_scaleshift = model_meta_info_struct[0].input_quant.round
    model_ip_zp = model_meta_info_struct[0].input_quant.zero_point
    # model_op_multiplier = model_meta_info_struct[0].output_quant.mult
    # model_op_scaleshift = model_meta_info_struct[0].outut_quant.round
    # model_op_zp = model_meta_info_struct[0].output_quant.zero_point

    if (model_meta_info_struct[0].model_labels.offset != -1):
        labels = bin_file_data[model_meta_info_struct[0].model_labels.offset:
                               model_meta_info_struct[0].model_labels.offset+model_meta_info_struct[0].model_labels.length]
        logger.debug(f"labels {labels}")

    model_name = bin_file_data[model_meta_info_struct[0].model_name.offset:
                               model_meta_info_struct[0].model_name.offset+model_meta_info_struct[0].model_name.length]
    logger.debug(f"model_name {model_name}")

    # print(f"model_layer_cnt {layer_count}, {model_meta_info_struct[0].placeholder0_test_model_info}, {model_meta_info_struct[0].placeholder1_test_model_info}")
    logger.debug(f"model_layer_cnt {layer_count}")
    logger.debug(f"model_input_inv_scaling_multiplier {model_ip_multiplier}")
    logger.debug(f"model_input_rounding {model_ip_scaleshift}")
    logger.debug(f"model_input_zero_point {model_ip_zp}")
    # model_layer_desc = bin_file_data[model_desc_struct[0].layers.offset:model_desc_struct[0].layers.offset+model_desc_struct[0].layers.length]
    model_const = bin_file_data[model_desc_struct[0].consts.offset:
                                model_desc_struct[0].consts.offset+model_desc_struct[0].consts.length]
    # model_compilation_options = bin_file_data[model_desc_struct[0].compilation_option.offset:model_desc_struct[0].compilation_option.offset+model_desc_struct[0].compilation_option.length]

    # print(f"model_layer_desc {model_layer_desc}")#FIXME write code to test the model description independently
    # print(f"model_const {model_const}")#FIXME write code to test the model const independently
    # f =  open(binary_file_path,"rb")
    model_compilation_options_struct = get_model_compilations_options_struct(
        ffi_object)
    model_layer_description_struct = get_model_descriptor_layer_struct(
        ffi_object, layer_cnt=layer_count)
    f.seek(model_desc_struct[0].layers.offset)
    f.readinto(ffi_object.buffer(model_layer_description_struct))

    f.seek(0)
    f.seek(model_desc_struct[0].compilation_option.offset)
    f.readinto(ffi_object.buffer(model_compilation_options_struct))
    f.close()

    logger.debug("Compilation Options : ")
    logger.debug(
        f"\tinterlayer buffer size {model_compilation_options_struct[0].interlayer_buffer_size}")
    logger.debug(
        f"\tpsum buffer size {model_compilation_options_struct[0].psum_buffer_size}")
    logger.debug(
        f"\ttest vector count {model_compilation_options_struct[0].header_file_test_vector_cnt}")
    # log level, psum mode and psum buffer placement added in bin file version 0.12.0
    if (version_integer >= 0x00000C00):
        log_level_name = ffi_object.string(ffi_object.cast(
            "nrf_axon_nn_compiler_log_level_e", model_compilation_options_struct[0].log_level))
        logger.debug(
            f"\tlog level {model_compilation_options_struct[0].log_level} : {log_level_name}")
        conv2d_setting_name = ffi_object.string(ffi_object.cast(
            "nrf_axon_nn_conv2d_setting_e", model_compilation_options_struct[0].convolution_2d_setting))
        logger.debug(
            f"\tpsum mode {model_compilation_options_struct[0].convolution_2d_setting} : {conv2d_setting_name}")
        psum_buffer_loc_name = ffi_object.string(ffi_object.cast(
            "nrf_axon_nn_psum_buffer_placement_e", model_compilation_options_struct[0].psum_buffer_placement))
        logger.debug(
            f"\tpsum buffer placement {model_compilation_options_struct[0].psum_buffer_placement} : {psum_buffer_loc_name}")

    for layers in range(layer_count):
        logger.debug(f"=======LAYER NO : {layers}==========")
        logger.debug(
            f"input count : {model_layer_description_struct[layers].input_id_cnt}")
        for input_ids in range(model_layer_description_struct[layers].input_id_cnt):
            logger.debug(
                f"input id : {model_layer_description_struct[layers].input_ids[input_ids] }")  # 2
            logger.debug(f" id {model_layer_description_struct[layers].input_ids[input_ids]} : input shape (C, H, W) : {model_layer_description_struct[layers].input_dimensions[input_ids].channel_cnt,model_layer_description_struct[layers].input_dimensions[input_ids].height,model_layer_description_struct[layers].input_dimensions[input_ids].width}")  # 3
            input_byte_width_name = ffi_object.string(ffi_object.cast(
                "nrf_axon_nn_byte_width_e", model_layer_description_struct[layers].input_dimensions[input_ids].byte_width))
            input_bitwidth = tflite_axon_enum_wrapper.get_bitwidth_from_bytewidth_enum(
                model_layer_description_struct[layers].input_dimensions[input_ids].byte_width)
            logger.debug(
                f" id {model_layer_description_struct[layers].input_ids[input_ids]} : input byte_width : { model_layer_description_struct[layers].input_dimensions[input_ids].byte_width} , {input_byte_width_name} : bitwidth : {input_bitwidth}")  # 4
        axon_op_name = ffi_object.string(ffi_object.cast(
            "nrf_axon_nn_op_e", model_layer_description_struct[layers].nn_operation))
        logger.debug(
            f"nn_operation : {model_layer_description_struct[layers].nn_operation } : {axon_op_name}")  # 6
        if (axon_op_name == "NRF_AXON_NN_OP_CONCATENATE"):
            logger.debug(
                f"concatenation axis : {model_layer_description_struct[layers].concatenate_axis}")
        # logger.debug(f"input height : {model_layer_description_struct[layers].input_dimensions.height }")#96
        # logger.debug(f"input width : {model_layer_description_struct[layers].input_dimensions.width }")#96
        # logger.debug(f"input channel count : {model_layer_description_struct[layers].input_dimensions.channel_cnt }")#96
        logger.debug(
            f"filter height : {model_layer_description_struct[layers].filter_dimensions.height }")  # 3
        logger.debug(
            f"filter width : {model_layer_description_struct[layers].filter_dimensions.width }")  # 3
        logger.debug(
            f"filter channel count : {model_layer_description_struct[layers].filter_dimensions.channel_cnt }")  # 16
        filter_byte_width_name = ffi_object.string(ffi_object.cast(
            "nrf_axon_nn_byte_width_e", model_layer_description_struct[layers].filter_dimensions.byte_width))
        filter_bitwidth = tflite_axon_enum_wrapper.get_bitwidth_from_bytewidth_enum(
            model_layer_description_struct[layers].filter_dimensions.byte_width)
        logger.debug(
            f"filter byte_width : {model_layer_description_struct[layers].filter_dimensions.byte_width }, {filter_byte_width_name} : bitwidth : {filter_bitwidth}")  # 2
        filter_np_datatype = tflite_axon_enum_wrapper.GetNpDataTypeFromAxonByteWidth(
            model_layer_description_struct[layers].filter_dimensions.byte_width)
        logger.debug(
            f"output height : {model_layer_description_struct[layers].output_dimensions.height }")  # 48
        logger.debug(
            f"output width : {model_layer_description_struct[layers].output_dimensions.width }")  # 48
        logger.debug(
            f"output channel count : {model_layer_description_struct[layers].output_dimensions.channel_cnt }")  # 16
        output_length = model_layer_description_struct[layers].output_dimensions.channel_cnt
        output_bitwidth = tflite_axon_enum_wrapper.get_bitwidth_from_bytewidth_enum(
            model_layer_description_struct[layers].output_dimensions.byte_width)
        if (axon_op_name == "NRF_AXON_NN_OP_FULLY_CONNECTED"):
            # * model_layer_description_struct[layers].output_dimensions.byte_width
            output_length = model_layer_description_struct[layers].output_dimensions.width
        output_byte_width_name = ffi_object.string(ffi_object.cast(
            "nrf_axon_nn_byte_width_e", model_layer_description_struct[layers].output_dimensions.byte_width))
        logger.debug(
            f"output byte_width : {model_layer_description_struct[layers].output_dimensions.byte_width } , {output_byte_width_name} : bitwidth : {output_bitwidth}")  # 3
        logger.debug(
            f"stride x : {model_layer_description_struct[layers].stride_x }")  # 1
        logger.debug(
            f"stride y : {model_layer_description_struct[layers].stride_y }")  # 2
        logger.debug(
            f"output zp : {model_layer_description_struct[layers].output_zero_point }")  # -128
        logger.debug(
            f"scale shift cnt : {model_layer_description_struct[layers].scale_shift_cnt }")  # 1
        af_name = ffi_object.string(ffi_object.cast(
            "nrf_axon_nn_activation_function_e", model_layer_description_struct[layers].activation_function))
        logger.debug(
            f"activation function : {model_layer_description_struct[layers].activation_function } : {af_name}")  # 3
        logger.debug(
            f"pad left : {model_layer_description_struct[layers].pad_left }")  # layers
        logger.debug(
            f"pad right : {model_layer_description_struct[layers].pad_right }")  # 1
        logger.debug(
            f"pad top : {model_layer_description_struct[layers].pad_top }")  # layers
        logger.debug(
            f"pad bottom : {model_layer_description_struct[layers].pad_bottom }")  # 1
        filter_length = model_layer_description_struct[layers].filter_dimensions.height*model_layer_description_struct[layers].filter_dimensions.width*model_layer_description_struct[
            layers].filter_dimensions.channel_cnt * model_layer_description_struct[layers].output_dimensions.channel_cnt*model_layer_description_struct[layers].filter_dimensions.byte_width
        if (axon_op_name == "NRF_AXON_NN_OP_DEPTHWISE_CONV2D") or axon_op_name == "NRF_AXON_NN_OP_STRIDED_SLICE":
            filter_length = int(
                filter_length / model_layer_description_struct[layers].output_dimensions.channel_cnt)
        logger.debug(
            f"filter ptr offset : {model_layer_description_struct[layers].filter.offset }, length : {filter_length}")
        if (np.array(model_layer_description_struct[layers].filter.offset).astype(np.int32) != -1):
            filter_count = int(
                filter_length / model_layer_description_struct[layers].filter_dimensions.byte_width)
            filter_vector = np.frombuffer(
                model_const, offset=model_layer_description_struct[layers].filter.offset, dtype=filter_np_datatype, count=filter_count)
            logger.debug(f"filter_vector : {filter_vector}")
        bias_prime_length = output_length
        logger.debug(
            f"bias_prime offset : {model_layer_description_struct[layers].bias_prime.offset }, length : {bias_prime_length}")
        if (np.array(model_layer_description_struct[layers].bias_prime.offset).astype(np.int32) != -1):
            bias_vector = np.frombuffer(
                model_const, offset=model_layer_description_struct[layers].bias_prime.offset, dtype=np.int32, count=bias_prime_length)
            logger.debug(f"bias_vector : {bias_vector}")
        multiplier_length = model_layer_description_struct[layers].output_dimensions.channel_cnt
        if (np.array(model_layer_description_struct[layers].output_multipliers.offset).astype(np.int32) != -1):
            if (axon_op_name == "NRF_AXON_NN_OP_AVERAGE_POOLING"):
                multiplier_length = 1
            multiplier_vector = np.frombuffer(
                model_const, offset=model_layer_description_struct[layers].output_multipliers.offset, dtype=np.int32, count=multiplier_length)
            logger.debug(f"multiplier_vector : {multiplier_vector}")
        logger.debug(
            f"output multipliers offset : {model_layer_description_struct[layers].output_multipliers.offset } length : {multiplier_length}")
        # 0x2FFFEEEE
        logger.debug(
            f"scale shifts offset : {model_layer_description_struct[layers].scale_shifts.offset }, scaleshift count : {model_layer_description_struct[layers].scale_shift_cnt}")
        if (np.array(model_layer_description_struct[layers].scale_shifts.offset).astype(np.int32) != -1):
            scaleshift_vector = np.frombuffer(
                model_const, offset=model_layer_description_struct[layers].scale_shifts.offset, dtype=np.int8, count=model_layer_description_struct[layers].scale_shift_cnt)
            logger.debug(f"scale_shifts vector : {scaleshift_vector}")
        if (np.array(model_layer_description_struct[layers].cpu_op_additional_attributes.offset).astype(np.int32) != -1):
            cpu_op_additional_attribs = np.frombuffer(
                model_const, offset=model_layer_description_struct[layers].cpu_op_additional_attributes.offset, dtype=np.int32, count=model_layer_description_struct[layers].cpu_op_additional_attributes_count)
            logger.debug(
                f"cpu ops additional attrib list offset : {model_layer_description_struct[layers].cpu_op_additional_attributes.offset }, cpu ops additional attrib count : {model_layer_description_struct[layers].cpu_op_additional_attributes_count}")
            logger.debug(
                f"cpu ops additional attrib list : {cpu_op_additional_attribs}")

    logger.debug(f"model_input_inv_scaling_multiplier {model_ip_multiplier}")
    logger.debug(f"model_input_rounding {model_ip_scaleshift}")
    logger.debug(f"model_input_zero_point {model_ip_zp}")


INCLUDE = "#include"


if __name__ == "__main__":
    # print("testing FC4")
    # TestModelBinFile("../model_desc_bins/axon_model_fc4_kws_bin_.bin")
    """
    to use this unit test follow the steps below
    1 . add the complete or relative path to the variable 'bin_file_path' below
    2 . run or debug the app using python integrated debugger
    """
    bin_file_path = r"../model_desc_bins/axon_model_fc4_kws_bin_.bin"
    print(f"running test for bin file : {os.path.basename(bin_file_path)}")
    TestModelBinFile(bin_file_path)
