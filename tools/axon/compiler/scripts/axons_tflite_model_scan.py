""" 
/*
 * Copyright (c) 2025, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */
"""
from utility import cpu_operator_options as cpu_operator_options
from utility import operator_options as ops
# from utility import util
from pathlib import Path
# import tensorflow as tf
import tflite as tflite
# import numpy as np
import sys
import os
# Suppress all logs (INFO, WARNING, and ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def print_with_borders(text, border="="):
    if border == "":
        print(text)
        return
    border_text = border*len(text)
    print(f"{border_text}")
    print(text)
    print(f"{border_text}")


def scan_tflite_model(model_path):
    model_name = Path(model_path).name
    print_with_borders(f"INFO: Model {model_name} ", border="")
    tflite_axon_graph_object = ops.TfLiteAxonGraph(model_path)
    if tflite_axon_graph_object is None:
        print_with_borders(
            f"ERROR: {model_name} tflite file is None X(", border="")
        return
    # input_details = tflite_axon_graph_object.get_tflite_input_details()
    # output_details = tflite_axon_graph_object.get_tflite_output_details()
    # tensor_details = tflite_axon_graph_object.get_tflite_tensor_details()
    # ops_details = tflite_axon_graph_object.get_tflite_operator_details()

    # print_with_borders("== Model Inputs ==")
    # for t in input_details:
    #     print(f"  - name: {t['name']}, shape: {t['shape']}, dtype: {t['dtype']}")

    # print_with_borders("== Model Outputs ==")
    # for t in output_details:
    #     print(f"  - name: {t['name']}, shape: {t['shape']}, dtype: {t['dtype']}")

    # check if the model input and outputs are supported by axon
    # model_ip_datawidth = input_details[0]['dtype']
    # if model_ip_datawidth == np.float32:
    #     if ops_details[0]['op_name'] != "QUANTIZE":
    #         print_with_borders(
    #             f"FAIL: {model_name} has float inputs, which are not supported X( ", border="")
    #         return
    # if model_ip_datawidth == np.uint8:
    #     if ops_details[0]['op_name'] != "QUANTIZE":
    #         print_with_borders(
    #             f"FAIL: {model_name} has uint8 inputs, which are not supported X( ", border="")
    #         return

    # print_with_borders("== All Tensors ==")
    # for t in tensor_details:
    #     print(f"  - index: {t['index']}, name: {t['name']}, shape: {t['shape']}, dtype: {t['dtype']}")

    # # Print operator details
    # print_with_borders("== Operators ==")
    # for i, op in enumerate(ops_details):
    #     print(f"  - Op #{i}: {op['op_name']}")
    #     # print(f"      inputs: {[tensor_details[idx]['name'] for idx in op['inputs']]}")
    #     # print(f"      outputs: {[tensor_details[idx]['name'] for idx in op['outputs']]}")

    # check if the model is compatible with axon
    tflite_axon_graph_object.init_axon_supported_ops_object(
        cpu_op_codes_list=cpu_operator_options.cpu_operators_list)
    model_supported, not_supported_ops, tr_model_supported, tr_model_ops = tflite_axon_graph_object.get_model_operators_axon_support_info()

    if not model_supported:
        constraint_reasons = '\n'.join(
            f"WARN: |Layer {k}: {v}|" for k, v in not_supported_ops.items() if v)
        print(f"{constraint_reasons}")
        print_with_borders(
            f"FAIL: The model {model_name} is not supported due to above reasons/constraints in the layers X( ", border="")
        # try transposing the model?
        if tr_model_supported:
            print_with_borders(
                "INFO: The model can be supported on Axon if transposed ;)", border="")
        return
    print_with_borders(
        f"PASS: The model {model_name} is supported on Axon :) ", border="")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: python axons_tflite_model_scan.py full_model_path.tflite")
        sys.exit(1)

    model_path = sys.argv[1]
    print_with_borders("\nINFO: Starting scan...", border="")
    scan_tflite_model(model_path)
    print_with_borders("INFO: Tflite model scan complete!\n", border="")
