""" 
/*
 * Copyright (c) 2025, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */
"""
import numpy as np
import tensorflow as tf


def unit_kernel_initializer(shape, dtype=np.float32):
    if len(shape) == 2:
        weights = np.zeros(shape, dtype=np.float32)
        for i in range(min(shape[0], shape[1])):
            weights[i, i] = 1.0
        return tf.convert_to_tensor(weights, dtype=dtype)

    # shape: (1, 1, in_channels, out_channels)
    kernel = np.zeros(shape, dtype=np.float32)
    in_channels = shape[2]
    out_channels = shape[3]
    for i in range(min(in_channels, out_channels)):
        kernel[0, 0, i, i] = 1.0
    return tf.convert_to_tensor(kernel, dtype=dtype)


def strided_slice_fn(input, strides=(1, 1, 1, 1), input_batch=1, begin=[0, 0, 0, 0], begin_mask=0, end_mask=0):
    stride_b, stride_h, stride_w, stride_c = strides
    input_shape_dyn = input.shape
    # Begin, end, and strides for all 4 dims
    end = [
        input_batch,  # batch
        input_shape_dyn[1],  # height
        input_shape_dyn[2],  # width
        input_shape_dyn[3],  # channels
    ]
    stride_b = 1  # forcing the stride_b value to be 1, as striding along batch is not supported
    stride_h = min(input_shape_dyn[1], stride_h)
    stride_w = min(input_shape_dyn[2], stride_w)
    stride_c = min(input_shape_dyn[3], stride_c)
    stride_vals = [stride_b, stride_h, stride_w, stride_c]
    return tf.strided_slice(input, begin=begin, end=end, strides=stride_vals, begin_mask=begin_mask, end_mask=end_mask)


def dense_fn(input_op, out_dim=1, activation=None, use_bias=False):
    in_dim = input_op.shape[-1]
    dense = tf.keras.layers.Dense(
        out_dim, activation=activation, use_bias=True)
    output_op = dense(input_op)
    # Set unit-like weights
    weight_shape = (in_dim, out_dim)
    unit_weights = unit_kernel_initializer(weight_shape)
    bias = np.zeros((out_dim,), dtype=np.float32)
    if use_bias:
        bias = np.ones((out_dim,), dtype=np.float32)
    dense.set_weights([unit_weights.numpy(), bias])
    dense.trainable = False
    return output_op


def check_kernel_size(input_shape, kernel_size):
    kernel_height, kernel_width = kernel_size[0], kernel_size[1]
    ip_height = input_shape[1]
    ip_width = input_shape[2]
    if kernel_size[0] > ip_height:
        kernel_height = ip_height+1
    if kernel_size[1] > ip_width:
        kernel_width = ip_width+1
    return kernel_height, kernel_width


def conv_fn(input_op, filter_size=1, kernel_size=(1, 1), strides=(1, 1), activations=None, padding='same', pw=False):
    channels = input_op.shape[-1]
    if pw:
        kernel_size = (1, 1)
    conv0 = tf.keras.layers.Conv2D(
        filters=filter_size,
        kernel_size=kernel_size,
        padding=padding,
        use_bias=True,
        trainable=False,
        activation=activations,
        strides=strides,
    )
    conv0.build((None, None, None, channels))
    kernel = np.ones(
        (kernel_size[0], kernel_size[1], channels, filter_size), dtype=np.float32)
    bias = np.zeros((filter_size,), dtype=np.float32)
    conv0.set_weights([kernel, bias])
    return conv0(input_op)


def conv1d_fn(input_op, filter_size=1, kernel_size=1, strides=1, activations=None, padding='same', pw=False):
    channels = input_op.shape[-1]
    conv1d0 = tf.keras.layers.Conv1D(
        filters=filter_size,
        kernel_size=kernel_size,
        padding=padding,
        use_bias=True,
        activation=activations,
        strides=strides
    )
    conv1d0.build((None, channels))
    kernel = np.ones((kernel_size, channels, filter_size), dtype=np.float32)
    bias = np.zeros((filter_size,), dtype=np.float32)
    conv1d0.set_weights([kernel, bias])
    return conv1d0(input_op)


def dw_fn(input_op, kernel_size=(1, 1), activation=None, padding='same'):
    channels = input_op.shape[-1]
    depth_multiplier = 1
    # first layer performs a unit depthwise
    depthwise0 = tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        depth_multiplier=depth_multiplier,
        padding=padding,
        use_bias=True,
        trainable=False,
        activation=activation
    )
    depthwise0.build((None, None, None, channels))
    kernel = np.ones(
        (kernel_size[0], kernel_size[1], channels, depth_multiplier), dtype=np.float32)
    bias = np.zeros((channels*depth_multiplier,), dtype=np.float32)
    depthwise0.set_weights([kernel, bias])
    return depthwise0(input_op)


def pool_fn(input_op, pool_type="AvgPool", pool_size=(1, 1), strides=(1, 1), padding='same'):
    if pool_type == "AvgPool":
        return tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)(input_op)
    elif pool_type == "MaxPool":
        return tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding)(input_op)


def create_dense_layers(input_op, operator_info_dict=None):
    dense_model_variant = operator_info_dict['MODEL_VARIANT']
    # input_shape = input_op.shape
    # height = input_shape[1]
    # width = input_shape[2]
    # channels = input_shape[3]
    # reshape_to = height*width*channels
    flat = tf.keras.layers.Flatten(name="flatten")(input_op)
    activation = None
    if 'ACTIVATION' in operator_info_dict:
        activation = operator_info_dict['ACTIVATION']
    if dense_model_variant == "all_io_combinations":
        """ older logic for creating an all io combination model
        #keep defining dense layers based on
        x0 = dense_fn(flat,2048)
        x1 = dense_fn(x0,1048)        
        x2 = dense_fn(x1,768)
        x3 = dense_fn(x2,1020)

        x0_0 = dense_fn(x0, 256)
        x0_1 = dense_fn(x0_0, 512)
        x0_2 = dense_fn(x0_1, 1020)
        # x2_0 = dense_fn(x2, 1024)
        # add1 = tf.keras.layers.Add()([x1, x3])
        add0 = tf.keras.layers.Add()([x0_2, x3]) 
        output_op = dense_fn(add0,10)
        """
        x0 = dense_fn(flat, 2048)
        x1 = dense_fn(x0, 1024)
        x2 = dense_fn(x1, 512)
        x3 = dense_fn(x2, 256)
        x4 = dense_fn(x3, 128)
        x5 = dense_fn(x4, 64)
        x5 = dense_fn(x5, 384)
        x5 = dense_fn(x5, 768)
        x5 = dense_fn(x5, 1280)
        output_op = dense_fn(x5, 16, activation=activation)
    elif dense_model_variant == "single_fc":
        output_op = dense_fn(flat, 20, activation=activation)
    elif dense_model_variant == "fc_w_bias":
        x0 = dense_fn(flat, 20, use_bias=True)
        output_op = dense_fn(x0, 10, use_bias=True, activation=activation)
    elif dense_model_variant == "with_add":
        x0 = dense_fn(flat, 20)
        x1_0 = dense_fn(x0, 25)
        x1_1 = dense_fn(x1_0, 5)
        x1_2 = dense_fn(x1_1, 10)
        x0_0 = dense_fn(x0, 25)
        x0_1 = dense_fn(x0_0, 10)
        add0 = tf.keras.layers.Add()([x0_1, x1_2])
        output_op = dense_fn(add0, 10, activation=activation)
    else:  # get the default model
        if dense_model_variant != "default":
            print(
                f"create_operator_layers:warning:using default model as '{dense_model_variant}' variant is not added!")
        x0 = dense_fn(flat, 25)
        x1 = dense_fn(x0, 20)
        x2 = dense_fn(x1, 15)
        output_op = dense_fn(x2, 10, activation=activation)
    return output_op


def create_conv_layers(input_op, operator_info_dict=None):
    # first layer performs a unit depthwise
    filter_size = 4
    kernel_size = (3, 3)
    kernel_size = check_kernel_size(input_op.shape, kernel_size)
    strides = (1, 1)
    activation = None
    padding = 'same'
    if operator_info_dict:
        if 'STRIDES' in operator_info_dict:
            strides = operator_info_dict['STRIDES']
        if 'ACTIVATION' in operator_info_dict:
            activation = operator_info_dict['ACTIVATION']
        if 'FILTERS' in operator_info_dict:
            filter_size = operator_info_dict['FILTERS']
        if 'KERNEL_SIZE' in operator_info_dict:
            kernel_size = operator_info_dict['KERNEL_SIZE']
        if 'PADDING_TYPE' in operator_info_dict:
            padding = operator_info_dict['PADDING_TYPE']
    output_op = conv_fn(input_op, filter_size, kernel_size,
                        strides=strides, activations=activation, padding=padding)
    return output_op


def create_pointwise_layers(input_op, operator_info_dict=None):
    strides = (1, 1)
    filters = 32
    activation = None
    padding = 'same'
    if operator_info_dict:
        if 'STRIDES' in operator_info_dict:
            strides = operator_info_dict['STRIDES']
        if 'ACTIVATION' in operator_info_dict:
            activation = operator_info_dict['ACTIVATION']
        if 'FILTERS' in operator_info_dict:
            filters = operator_info_dict['FILTERS']
        if 'PADDING_TYPE' in operator_info_dict:
            padding = operator_info_dict['PADDING_TYPE']

    # add layers as needed
    output_op = conv_fn(input_op, filters, pw=True,
                        strides=strides, activations=activation, padding=padding)
    return output_op


def create_depthwise_layers(input_op, operator_info_dict=None):
    # add layers as needed
    padding = 'same'
    activation = None
    kernel_size = (3, 3)
    kernel_size = check_kernel_size(input_op.shape, kernel_size)
    if 'PADDING_TYPE' in operator_info_dict:
        padding = operator_info_dict['PADDING_TYPE']
    if 'ACTIVATION' in operator_info_dict:
        activation = operator_info_dict['ACTIVATION']
    if 'KERNEL_SIZE' in operator_info_dict:
        kernel_size = operator_info_dict['KERNEL_SIZE']

    output_op = dw_fn(input_op, kernel_size,
                      padding=padding, activation=activation)
    return output_op


def create_strided_slice_layers(input_op, batch, operator_info_dict=None):
    strides = (2, 2, 2)
    if operator_info_dict:
        if 'STRIDES' in operator_info_dict:
            strides = operator_info_dict['STRIDES']

    x0 = strided_slice_fn(
        input_op, strides=[1, strides[0], 1, 1], input_batch=batch)  # axis 0
    x1 = strided_slice_fn(
        x0, strides=[1, 1, strides[1], 1], input_batch=batch)  # axis 1
    output_op = strided_slice_fn(
        x1, strides=[1, 1, 1, strides[2]], input_batch=batch)  # axis 2
    return output_op


def create_concat_layers(input_op, operator_info_dict=None):
    avg_pool0 = tf.keras.layers.AveragePooling2D(
        pool_size=(1, 1), strides=(2, 1), padding='same')(input_op)
    # Concatenate along height (axis=1)
    x_h = tf.keras.layers.concatenate(
        [input_op, avg_pool0], axis=1, name="concat_h")
    avg_pool1 = tf.keras.layers.AveragePooling2D(
        pool_size=(1, 1), strides=(1, 2), padding='same')(x_h)
    # Concatenate along width (axis=2)
    x_w = tf.keras.layers.concatenate(
        [x_h, avg_pool1], axis=2, name="concat_w")
    in_channels = x_w.shape[-1]
    out_channels = int(0.5 * in_channels)
    # Create unit-like weights
    kernel2x = unit_kernel_initializer((1, 1, in_channels, out_channels))
    bias_weights = np.zeros((out_channels,), dtype=np.float32)
    conv = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=1,
        padding='same',
        use_bias=True,
        trainable=False,
    )
    conv2d0 = conv(x_w)
    # Set weights manually
    conv.set_weights([kernel2x.numpy(), bias_weights])
    # Concatenate along channels (axis=3)
    output_op = tf.keras.layers.concatenate(
        [conv2d0, x_w], axis=3, name="concat_c")
    return output_op


def create_pad_layers(input_op, operator_info_dict=None):
    h_pad = [1, 1]
    w_pad = [3, 3]
    c_pad = [1, 1]
    constant_value = 0.0
    if operator_info_dict:
        if 'H_PAD' in operator_info_dict:
            h_pad = operator_info_dict['H_PAD']
        if 'W_PAD' in operator_info_dict:
            w_pad = operator_info_dict['W_PAD']
        if 'C_PAD' in operator_info_dict:
            w_pad = operator_info_dict['C_PAD']
        if 'CONST_VALUE' in operator_info_dict:
            constant_value = operator_info_dict['CONST_VALUE']

    xh = tf.pad(input_op, paddings=[[0, 0], h_pad, [0, 0], [
                0, 0]], name="H_Pad", mode='CONSTANT', constant_values=constant_value)  # Height
    channels = xh.shape[-1]
    depthwise0 = tf.keras.layers.DepthwiseConv2D(
        kernel_size=1,
        depth_multiplier=1,
        padding='same',
        use_bias=True,
        trainable=False,
    )
    depthwise0.build((None, None, None, channels))
    kernel = np.ones((1, 1, channels, 1), dtype=np.float32)
    bias = np.zeros((channels,), dtype=np.float32)
    depthwise0.set_weights([kernel, bias])
    dw_conv1 = depthwise0(xh)
    xw = tf.pad(dw_conv1, paddings=[[0, 0], [0, 0], w_pad, [
                0, 0]], name="W_Pad", mode='CONSTANT', constant_values=constant_value)  # Width
    dw_conv2 = depthwise0(xw)
    output_op = tf.pad(dw_conv2, paddings=[[0, 0], [0, 0], [
                       0, 0], c_pad], name="C_Pad", mode='CONSTANT', constant_values=constant_value)  # Channel
    return output_op


def create_add_layers(input_op, operator_info_dict=None):
    channels = input_op.shape[-1]
    kernel = np.ones((1, 1, channels, 1), dtype=np.float32)
    bias = np.zeros((channels,), dtype=np.float32)
    depthwise1 = tf.keras.layers.DepthwiseConv2D(
        kernel_size=1,
        depth_multiplier=1,
        padding='same',
        use_bias=True,
        trainable=False,
    )
    depthwise1.build((None, None, None, channels))
    kernel_2 = kernel * 2
    depthwise1.set_weights([kernel_2, bias])
    dw_conv1 = depthwise1(input_op)
    add1 = tf.keras.layers.Add()([input_op, dw_conv1])
    add1Relu = tf.keras.layers.ReLU()(add1)
    # flat0 = tf.keras.layers.Flatten(name="flatten0")(add1Relu)
    # flat1 = tf.keras.layers.Flatten(name="flatten1")(dw_conv1)
    # dens0 = dense_fn(flat0, 1020)
    # dense1 = dense_fn(flat1, 1020)
    # output_op = tf.keras.layers.Add()([dens0,dense1])
    output_op = tf.keras.layers.Add()([add1Relu, dw_conv1])
    return output_op


def create_pool_layers(input_op, operator_info_dict=None):
    stride = (1, 1)
    kernel_size = (1, 1)
    padding = 'same'
    if operator_info_dict:
        if 'STRIDES' in operator_info_dict:
            stride = operator_info_dict['STRIDES']
        # if 'ACTIVATION' in operator_info_dict:
        #     activation = operator_info_dict['ACTIVATION']
        if 'KERNEL_SIZE' in operator_info_dict:
            kernel_size = operator_info_dict['KERNEL_SIZE']
        if 'PADDING_TYPE' in operator_info_dict:
            padding = operator_info_dict['PADDING_TYPE']
    # add layers as needed
    pool0 = pool_fn(input_op, operator_info_dict['OP_TYPE'],
                    pool_size=kernel_size, strides=stride, padding=padding)
    return pool0


def create_conv1d_layers(input_op, operator_info_dict=None):
    # input_size = input_op.shape[1]*input_op.shape[2]*input_op.shape[3]
    reshape0 = tf.keras.layers.Reshape(
        (input_op.shape[1]*input_op.shape[2], input_op.shape[3]))(input_op)
    # conv1d0 = conv1d_fn(reshape0, 6, 3)
    # conv1d0 = tf.keras.layers.Conv1D(filters=6, kernel_size=3, padding='same', activation='relu')(reshape0)
    conv1d0 = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(input_op.shape[1]*input_op.shape[2], input_op.shape[3])),
        tf.keras.layers.Conv1D(filters=6, kernel_size=3,
                               padding='same', activation='relu'),
        # tf.keras.layers.GlobalAveragePooling1D(),
        # tf.keras.layers.Dense(1, activation='sigmoid')  # For binary classification
    ])
    return conv1d0(reshape0)


def get_split_tensor_from_split_size(axis_dimension, split_size):
    if isinstance(split_size, (int)):
        split_tensor = []
        for split in range(split_size):
            split_tensor.append(axis_dimension//split_size)
        return split_tensor
    return split_size


def splitv_neuton_example_fn(input_op, input_batch, split_size, axis):
    # input_batch=1
    split_tensor = get_split_tensor_from_split_size(
        input_op.shape[axis], split_size)
    split_op = tf.split(input_op, num_or_size_splits=split_tensor, axis=axis)
    h_stride0 = input_op.shape[1]  # 11
    begin0 = [0, -h_stride0, 0, 0]
    strided0 = strided_slice_fn(
        split_op[0], begin=begin0, input_batch=input_batch, end_mask=14)
    h_stride1 = input_op.shape[1]//2  # 7
    begin1 = [0, -h_stride1, 0, 0]
    strided1 = strided_slice_fn(
        split_op[1], begin=begin1, input_batch=input_batch, end_mask=14)
    dw0 = dw_fn(strided0, (h_stride0, 1), padding='valid')
    dw1 = dw_fn(strided1, (h_stride1, 1), padding='valid')
    return tf.keras.layers.concatenate([dw0, dw1], axis=3, name="concat_w")


def splitv_fn(input_op, split_size, axis):
    split_tensor = get_split_tensor_from_split_size(
        input_op.shape[axis], split_size)
    split_op = tf.split(input_op, num_or_size_splits=split_tensor, axis=axis)
    concatenate_inputs = []
    for split_count in range(len(split_op)):
        if (split_count) % 2:
            pad = "valid"
        else:
            pad = "same"
        dw_output = dw_fn(split_op[split_count], (1, 1), padding=pad)
        concatenate_inputs.append(dw_output)
    return tf.keras.layers.concatenate(concatenate_inputs, axis=axis)


def create_splitv_layers(input_op, input_batch=1, operator_info_dict=None):
    # input_shape=input_op.shape
    # channel = input_shape[-1]
    axis = -1
    split_size = 2
    if 'SPLIT_SIZE' in operator_info_dict:
        split_size = operator_info_dict['SPLIT_SIZE']
        if split_size > 2:
            print(
                f"WARNING : SPLIT_SIZE {split_size} not supported, max is 2!")
            split_size = 2
    if 'AXIS' in operator_info_dict:
        axis = operator_info_dict['AXIS']
    if 'MODEL_VARIANT' in operator_info_dict:
        model_variant = operator_info_dict['MODEL_VARIANT']

    if model_variant == "neuton_example":
        output_op = splitv_neuton_example_fn(
            input_op, input_batch, split_size, axis)
    elif model_variant == "double_split":
        layer0 = splitv_fn(input_op, split_size, axis)
        output_op = splitv_fn(layer0, split_size, axis)
    else:
        output_op = splitv_fn(input_op, split_size, axis)
    return output_op


def create_mean_layers(input_op, operator_info_dict=None):
    axis = 1
    if 'MEAN_AXIS' in operator_info_dict:
        axis = operator_info_dict['MEAN_AXIS']
    keepdims = True
    output_op = tf.reduce_mean(input_op, axis=axis, keepdims=keepdims)
    return output_op


def create_leakyrelu_layers(input_op, operator_info_dict=None):
    alpha = 0.5
    if 'ALPHA' in operator_info_dict:
        alpha = operator_info_dict['ALPHA']

    if 'MODEL_VARIANT' in operator_info_dict:
        model_variant = operator_info_dict['MODEL_VARIANT']

    if model_variant == "leaky_relu_test":
        leakyrelu0 = tf.keras.layers.LeakyReLU(
            alpha=alpha, name='leaky_relu')(input_op)
        conv0 = conv_fn(leakyrelu0, 3, (3, 3))
        leakyrelu1 = tf.keras.layers.LeakyReLU(
            alpha=alpha, name='leaky_relu1')(conv0)
        dw0 = dw_fn(leakyrelu1, (1, 1))
        output_op = tf.keras.layers.LeakyReLU(
            alpha=alpha, name='leaky_relu2')(dw0)
    else:
        output_op = tf.keras.layers.LeakyReLU(
            alpha=alpha, name='leaky_relu')(input_op)
    return output_op


def multiply_neuton_example_fn(input_op):
    channels = input_op.shape[-1]
    mean_output_op = tf.reduce_mean(input_op, axis=[1,2], keepdims=False)
    fc0 = dense_fn(mean_output_op, 8, activation='relu')
    fc1 = dense_fn(fc0, channels, activation='sigmoid')
    reshape0 = tf.reshape(fc1,[-1,1,1,channels])
    mul1 = tf.keras.layers.Multiply()([input_op, reshape0])
    return mul1

def mutiply_fn(input_op,activation=None, broadcast_axis=None):
    channels = input_op.shape[-1]
    kernel = np.ones((1, 1, channels, 1), dtype=np.float32)
    bias = np.zeros((channels,), dtype=np.float32)
    depthwise1 = tf.keras.layers.DepthwiseConv2D(
        kernel_size=1,
        depth_multiplier=1,
        padding='same',
        use_bias=True,
        trainable=False,
    )
    depthwise1.build((None, None, None, channels))
    kernel_2 = kernel * 2
    depthwise1.set_weights([kernel_2, bias])
    dw_conv1 = depthwise1(input_op)
    if broadcast_axis is not None:
        mean0 = tf.reduce_mean(dw_conv1, axis=broadcast_axis, keepdims=True)
        mul1 = tf.keras.layers.Multiply()([input_op, mean0])
    else:
        mul1 = tf.keras.layers.Multiply()([input_op, dw_conv1])
    
    output_op = tf.keras.layers.Activation(activation)(mul1)
    return output_op

def create_multiply_layers(input_op, operator_info_dict=None):
    activation=None
    broadcast_axis=None
    if operator_info_dict:
        if 'ACTIVATION' in operator_info_dict:
            activation = operator_info_dict['ACTIVATION']
        if 'MODEL_VARIANT' in operator_info_dict:
            model_variant = operator_info_dict['MODEL_VARIANT']
        if 'BROADCAST_AXIS' in operator_info_dict:
            broadcast_axis = operator_info_dict['BROADCAST_AXIS']

    if model_variant=="neuton_example":
        output_op = multiply_neuton_example_fn(input_op)
    else:
        output_op = mutiply_fn(input_op, activation, broadcast_axis)
    return output_op


SUPPORTED_LAYER_OPS_LIST = {"Add", "Pad", "Concat", "StridedSlice", "Dense", "Conv",
                            "DwConv", "PwConv", "AvgPool", "Conv1D", "MaxPool", "SplitV", "Mean", "LeakyRelu", "Multiply"}


def create_operator_layer(model_input, model_type, model_data_input, test_op_info_dict=None):
    input_shape = model_data_input.shape
    channels = input_shape[-1]
    # width = input_shape[2]
    # height = input_shape[1]
    batch = input_shape[0]

    # first layer performs a unit depthwise
    depthwise0 = tf.keras.layers.DepthwiseConv2D(
        kernel_size=1,
        depth_multiplier=1,
        padding='same',
        use_bias=True,
        trainable=False,
    )
    depthwise0.build((None, None, None, channels))
    kernel = np.ones((1, 1, channels, 1), dtype=np.float32)
    bias = np.zeros((channels,), dtype=np.float32)
    depthwise0.set_weights([kernel, bias])

    dw_conv0 = depthwise0(model_input)

    if model_type == "Add":
        model_output = create_add_layers(dw_conv0, test_op_info_dict)
    elif model_type == "Concat":
        model_output = create_concat_layers(dw_conv0, test_op_info_dict)
    elif model_type == "StridedSlice":
        model_output = create_strided_slice_layers(
            model_input, batch, test_op_info_dict)
    elif model_type == "Pad":
        model_output = create_pad_layers(dw_conv0, test_op_info_dict)
    elif model_type == "Dense":
        model_output = create_dense_layers(dw_conv0, test_op_info_dict)
    elif model_type == "Conv" or model_type == "Conv1D":
        model_output = create_conv_layers(model_input, test_op_info_dict)
    elif model_type == "PwConv":
        model_output = create_pointwise_layers(dw_conv0, test_op_info_dict)
    elif model_type == "DwConv":
        model_output = create_depthwise_layers(model_input, test_op_info_dict)
    elif model_type == "AvgPool" or model_type == "MaxPool":
        model_output = create_pool_layers(model_input, test_op_info_dict)
    elif model_type == "SplitV":
        model_output = create_splitv_layers(dw_conv0, batch, test_op_info_dict)
    elif model_type == "Mean":
        model_output = create_mean_layers(dw_conv0, test_op_info_dict)
    elif model_type == "LeakyRelu":
        model_output = create_leakyrelu_layers(dw_conv0, test_op_info_dict)    
    elif model_type == "Multiply":
        model_output = create_multiply_layers(dw_conv0, test_op_info_dict)
    return model_output
