import os
import glob
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from ctypes import cdll, c_int16, c_int32, c_uint32, POINTER


def is_dir(path):
    """
    checks if the path exists
    """
    return os.path.isdir(path)


def convert_dataset(item, data_key='audio', label_key='label'):
    """
    Puts the dataset in the format Keras expects, (data, labels).
    """
    data = item[data_key]
    label = item[label_key]
    return data, label


def convert_dataset_to_float(data):
    """
    converts the data set into float.
    """
    data['audio'] = tf.cast(data['audio'], tf.float32)
    data['audio'] = data['audio']/tf.constant(2**15, dtype=tf.float32)
    return data


def cast_and_pad_audio(sample_dict, sample_size=16000):
    """
    pads an audio dataset dictionary with zeros to the sample size and casts the audio as int16

    returns the padded sample dataset dictionary
    """
    audio = sample_dict['audio']
    # label = sample_dict['label']
    paddings = [[0, sample_size-tf.shape(audio)[0]]]
    audio = tf.pad(audio, paddings)
    audio16 = tf.cast(audio, tf.int16)
    sample_dict['audio'] = audio16
    return sample_dict


def convert_to_numpy_in_batches(dataset, data_size_fraction=1, batch_size=10000, data_key='audio', label_key='label'):
    """
    converts dataset into numpy arrays in batches
    """
    time_post = time.time()
    features = []
    labels = []
    data_size = np.int64(len(dataset)*data_size_fraction)
    for i in range(0, data_size, batch_size):
        data = dataset.skip(i)
        if (batch_size > (data_size-i)):
            batch_size = (data_size-i)
        batch = list(data.take(batch_size))
        batch_features = [np.array(example[data_key]) for example in batch]
        batch_labels = [example[label_key] for example in batch]
        features.extend(batch_features)
        labels.extend(batch_labels)

    features_np = np.array(features)
    labels_np = np.array(labels)
    print(f"converting data took {time.time() - time_post} seconds")
    return features_np, labels_np


def save_to_csv_in_batches(features, labels, file_name, batch_size=10000, sample_size_in_mb=100):
    """
    saves the raw features and labels into several batches of csv file which can be used for testing axon feature extraction
    """
    time_pre = time.time()
    ONE_MEGABYTE = 10**6
    # observed by converting numpy data set into csv files, the size of the numpy data set doubles when converted to csv
    NPY_CSV_CONVERSION_RATIO = 0.5
    max_sample_size_in_bytes = (
        sample_size_in_mb*ONE_MEGABYTE * NPY_CSV_CONVERSION_RATIO)
    num_samples = len(features)
    # determine the batch size here based on the size of the batches that we create
    if batch_size > num_samples:
        batch_size = num_samples
    if features[0:batch_size].nbytes > max_sample_size_in_bytes:
        size_dataset_bytes = (features.nbytes)
        max_batch_sample_size = np.int64(
            max_sample_size_in_bytes / (size_dataset_bytes / num_samples))
        batch_size = max_batch_sample_size
        print(f"the no of samples in one batch are {max_batch_sample_size}")
    for i in range(0, num_samples, batch_size):
        if (batch_size+i) > num_samples:
            batch_size = num_samples-i
        batch_name = f"_{i}_{i+batch_size-1}_"
        np.savetxt(file_name + "_raw" + batch_name + ".csv",
                   features[i:i+batch_size], delimiter=',', fmt='%d')
        np.savetxt(file_name + "_labels" + batch_name+".csv",
                   labels[i:i+batch_size], delimiter=',', fmt='%d')
    print(f"saving csv test data took {time.time() - time_pre} seconds")


def reshapes_input_data_to_match_model_input(model, data):
    """
    reshapes the input data in the shape the model expects.

    returns the reshaped data
    """
    input_shape = model.input_shape
    try:
        if len(input_shape) == 4:
            data = data.reshape(
                len(data), input_shape[1], input_shape[2], input_shape[3])
        elif len(input_shape) == 3:
            data = data.reshape(len(data), input_shape[1], input_shape[2])
    except Exception:
        Exception(
            f"reshaping input data to model input shape threw an error : model_ip_shape {input_shape}, input_data_shape {data.shape}")

    return data


def load_numpy_from_csv_folder(dir_path):
    """
    loads all the csv files present in a directory and returns a numpy array of all the data from the csv files
    """
    csv_files = glob.glob(dir_path + '/*.csv')
    # Load all CSV files into a list of NumPy arrays
    arrays = []
    for file in csv_files:
        # Read the CSV file in chunks using numpy's loadtxt
        # Adjust max_rows
        for chunk in np.loadtxt(file, delimiter=',', max_rows=10000, dtype=np.int32):
            arrays.append(chunk)
    return np.array(arrays)


def load_and_preprocess_training_data(features_path, labels_path, batch_size=100, shift_value=0, input_shape=None, save_csv_data_as_npy=True, ds_x_key='audio', ds_y_key='label'):
    """
    loads the dataset from a folder os csv files or from a numpy file for training the model
    """
    if is_dir(features_path):
        print(f"loading csv from {features_path}")
        ds_x = load_numpy_from_csv_folder(features_path)
        ds_x = np.float32(ds_x/2**shift_value)
        if save_csv_data_as_npy:
            np.save(features_path + "/" +
                    os.path.basename(os.path.normpath(features_path)) + ".npy", ds_x)
    elif features_path.endswith(".npy") and labels_path.endswith(".npy"):
        ds_x = np.load(features_path)
    else:
        print("Please provide a numpy file or a path to the csv directory")
        exit()
    if input_shape is not None:
        if ds_x.shape != input_shape:
            if (len(input_shape) == 4):
                ds_x = ds_x.reshape(
                    len(ds_x), input_shape[1], input_shape[2], input_shape[3])
    ds_y = np.load(labels_path)
    ds_ = tf.data.Dataset.from_tensor_slices({ds_x_key: ds_x,
                                              ds_y_key: ds_y})
    ds_ = ds_.map(convert_dataset)
    ds_ = ds_.batch(batch_size)
    ds_ = ds_.shuffle(len(ds_))
    return ds_


def unit_test_generate_mfcc_csv(path):
    """
    test code : used to generate a sample of mfccs for testing
    """
    train = np.load("data/kws_g12/training_data/train_mfcc_.npy")
    train = train.reshape(train.shape[0], train.shape[1]*train.shape[2])
    batch_size = 20
    for i in range(0, len(train), batch_size):
        np.savetxt("data/kws_g12/training_data/mfcc_out/" +
                   f"float_{i}_{batch_size-1}.csv", train[i:i+batch_size], delimiter=',', fmt='%d')
        np.savetxt("data/kws_g12/training_data/mfcc_out/" +
                   f"q.12_{i}_{batch_size-1}.csv", train[i:i+batch_size]*(2**12), delimiter=',', fmt='%d')


def get_max_min_ndx(array):
    """
    finds the index of the row of the minimum and maximum values present in the dataset

    returns the row index of the min and max value
    """
    # print(array.shape)
    max_value, max_index = np.max(array), np.unravel_index(
        np.argmax(array), array.shape)
    min_value, min_index = np.min(array), np.unravel_index(
        np.argmin(array), array.shape)
    print(
        f"max @index {max_value, max_index } min @index {min_value, min_index}")
    return max_index[0], min_index[0]


def append_min_max_raw_data(raw_data, array_path, raw_samples):
    """
    returns the raw samples for the max and min indices in the feature dataset
    """
    print(f"getting max and min for {array_path}")
    train_x = np.load(array_path)
    max_index, min_index = get_max_min_ndx(train_x)
    raw_samples.append(raw_data[max_index])
    raw_samples.append(raw_data[min_index])
    return raw_samples


def get_min_max():
    """
    returns the min and max raw samples from the data

    also has older code to calculate bit tolerance for the values coming out in the mfccs
    """
    # load the raw data
    # raw_train = np.load(r"data\kws_g12\training_data\raw_data\train_x.npy")
    raw_test = np.load(r"data\kws_g12\training_data\raw_data\test_x.npy")
    # raw_val = np.load(r"data\kws_g12\training_data\raw_data\val_x.npy")

    raw_samples = []

    # raw_samples = append_min_max_raw_data(raw_train,r"data\kws_g12\training_data\axon_mfccs\train_mfcc.npy", raw_samples)
    raw_samples = append_min_max_raw_data(
        raw_test, r"data\kws_g12\training_data\axon_mfccs\test_mfcc.npy", raw_samples)
    # raw_samples = append_min_max_raw_data(raw_val,r"data\kws_g12\training_data\axon_mfccs\val_mfcc.npy", raw_samples)
    # raw_samples = mdhs.append_min_max_raw_data(raw_train,r"data\kws_g12\training_data\float_mfccs\train_mfcc.npy", raw_samples)
    # raw_samples = mdhs.append_min_max_raw_data(raw_test,r"data\kws_g12\training_data\float_mfccs\test_mfcc.npy", raw_samples)
    # raw_samples = mdhs.append_min_max_raw_data(raw_val,r"data\kws_g12\training_data\float_mfccs\val_mfcc.npy", raw_samples)
    # raw_samples = mdhs.append_min_max_raw_data(raw_train,r"data\kws_g12\training_data\axon_mfccs_q14_hamming\train_csv\train_csv.npy", raw_samples)
    # raw_samples = mdhs.append_min_max_raw_data(raw_test,r"data\kws_g12\training_data\axon_mfccs_q14_hamming\test_csv\test_csv.npy", raw_samples)
    # raw_samples = mdhs.append_min_max_raw_data(raw_val,r"data\kws_g12\training_data\axon_mfccs_q14_hamming\val_csv\val_csv.npy", raw_samples)

    # print(f"test (max, min) {np.max(test_x), np.min(test_x)} @index (max, min) {np.argmax(train_x), np.argmin(train_x)}")
    # print(f"val (max, min) {np.max(val_x), np.min(val_x)} @index (max, min) {np.argmax(train_x), np.argmin(train_x)}")
    # min_bits_needed = (len(bin(int(max(abs(np.max(train_x)),abs(np.min(train_x)))))) - 2)
    # print(f"maximum length needed to represent the data without saturation {min_bits_needed}")

    # train_x = np.load(r"data\kws_g12\training_data\float_mfccs\train_mfcc.npy")
    # # train_y = np.load(r"data\kws_g12\training_data\float_mfccs\train_label_.npy")
    # test_x = np.load(r"data\kws_g12\training_data\float_mfccs\test_mfcc.npy")
    # # test_y = np.load(r"data\kws_g12\training_data\float_mfccs\train_label_.npy")
    # val_x = np.load(r"data\kws_g12\training_data\float_mfccs\val_mfcc.npy")
    # # val_y = np.load(r"data\kws_g12\training_data\float_mfccs\val_label_.npy")

    # print(f"train (max, min) {np.max(train_x), np.min(train_x)} @index (max, min) {np.argmax(train_x), np.argmin(train_x)}")
    # print(f"test (max, min) {np.max(test_x), np.min(test_x)} @index (max, min) {np.argmax(train_x), np.argmin(train_x)}")
    # print(f"val (max, min) {np.max(val_x), np.min(val_x)} @index (max, min) {np.argmax(train_x), np.argmin(train_x)}")

    # shift_len = (len(bin(int(max(abs(np.max(train_x)),abs(np.min(train_x)))))) - 2)
    # print(shift_len)

    # train_x = np.load(train_args['train_feature_data'])
    # test_x = np.load(train_args['test_feature_data'])
    # val_x = np.load(train_args['val_feature_data'])

    # max_value, max_indx_tuple = np.max(train_x), np.unravel_index(np.argmax(train_x))
    # #go into the raw data to get the test audio sample and thus save the raw vectors out of it
    # #also store them as csv's to compare while debugging
    # raw_data = []
    # float_mfccs = []
    # max_value_mfccs = []

    return raw_samples


def get_inference(model, test_data):
    """
    Perform inference for the model and return the results

    """
    return model.predict(test_data)


def compare_models(model_1, model_2, test_data_1, test_data_2, true_labels):
    """
    Get inference results from both models

    """
    preds_model_1 = get_inference(model_1, test_data_1)
    preds_model_2 = get_inference(model_2, test_data_2)
    # selected_vectors = [2052,312,4707,3520,426,1212,1168,2136,3129,609,2667,3771,4368,3754,4099,699,2421,4747,1029,836,2586,4595,2623,4418,3640,4472,2511,1157,4331,2344,1387,634,2470,3713,1913,3587,3178,268,3213,4738]
    # Compare the results
    matches = []
    mismatches = []
    for idx, (pred_1, pred_2) in enumerate(zip(preds_model_1, preds_model_2)):
        # Get the index of the maximum value in the result of model 1
        max_index_1 = np.argmax(pred_1)
        # Get the index of the maximum value in the result of model 2
        max_index_2 = np.argmax(pred_2)
        if max_index_1 == true_labels[idx]:
            if max_index_1 == max_index_2:
                matches.append(idx)
            elif max_index_1 != max_index_2:
                mismatches.append(idx)

    return matches, mismatches, preds_model_1, preds_model_2


def select_test_vectors(matches, mismatches, n=20):
    """
    Select 20 test vectors where the models match and 20 where they do not

    """
    # selected_matches = np.random.choice(matches, size=n, replace=False) if len(matches) >= n else matches
    # selected_mismatches = np.random.choice(mismatches, size=n, replace=False) if len(mismatches) >= n else mismatches
    selected_matches = matches[:n]
    selected_mismatches = mismatches[:n]
    # selected_matches = np.array([2052,312,4707,3520,426,1212,1168,2136,3129,609,2667,3771,4368,3754,4099,699,2421,4747,1029,836])
    # selected_mismatches = np.array([2586,4595,2623,4418,3640,4472,2511,1157,4331,2344,1387,634,2470,3713,1913,3587,3178,268,3213,4738])
    return selected_matches, selected_mismatches


def save_vectors_to_csv(selected_indices, test_data, filename, mfcc_type):
    """
    Select the rows corresponding to the selected indices

    """

    selected_vectors = test_data[selected_indices]
    vectors_shape = selected_vectors.shape
    if len(vectors_shape) == 4:
        selected_vectors = selected_vectors.reshape(
            vectors_shape[0], vectors_shape[1]*vectors_shape[2]*vectors_shape[3])
    elif len(vectors_shape) == 3:
        selected_vectors = selected_vectors.reshape(
            vectors_shape[0], vectors_shape[1]*vectors_shape[2])
    # Convert to DataFrame for easy CSV export
    df = pd.DataFrame(selected_vectors, index=selected_indices)

    # Save to CSV
    df.to_csv(filename, index=True, mode='a+', index_label=mfcc_type)
    print(f"Saved selected vectors to {filename}")


def get_mismatch_matching_vectors(model1_path, model2_path, test_data1_path, test_data2_path, true_labels_path, raw_data_path, save_files_path=r"mfcc_test_files/"):
    """
    provides a way for extensively testing the feature generation by getting matching and mismatching vectors
    and saving them as csv files
    returns the list of mismatch and matching vectors
    """
    test_data_1 = np.load(test_data1_path)
    test_data_2 = np.load(test_data2_path)
    model_1 = tf.keras.models.load_model(model1_path)
    model_2 = tf.keras.models.load_model(model2_path)
    test_data_1 = reshapes_input_data_to_match_model_input(
        model_1, test_data_1)
    test_data_2 = reshapes_input_data_to_match_model_input(
        model_2, test_data_2)
    true_labels = np.load(true_labels_path)
    raw_data = np.load(raw_data_path)
    # Compare inference results
    matches, mismatches, preds_model1, preds_model2 = compare_models(
        model_1, model_2, test_data_1, test_data_2, true_labels)

    # Select 20 matching and 20 mismatching test vectors
    selected_matches, selected_mismatches = select_test_vectors(
        matches, mismatches, n=20)
    full_selected = []
    full_selected.extend(selected_matches)
    full_selected.extend(selected_mismatches)

    print("Selected matching test vectors:", selected_matches)
    print("Selected mismatching test vectors:", selected_mismatches)

    # check if the folder is present, if not create the directory
    if not is_dir(save_files_path):
        # create the directory
        os.mkdir(save_files_path)

    save_vectors_to_csv(full_selected, raw_data,
                        save_files_path+'selected_raw.csv', 'raw_vectors')
    # Save the selected mismatching vectors to a CSV file
    save_vectors_to_csv(full_selected, true_labels,
                        save_files_path+'selected_labels.csv', 'labels')
    save_vectors_to_csv(full_selected, preds_model1,
                        save_files_path+'float_last_layer_.csv', 'float_last_layer')
    save_vectors_to_csv(full_selected, preds_model2,
                        save_files_path+'axon_last_layer_.csv', 'axon_last_layer')

    # Save the selected matching vectors to a CSV file
    save_vectors_to_csv(selected_matches, test_data_1,
                        save_files_path+'selected_matches.csv', 'float_matching')
    # Save the selected mismatching vectors to a CSV file
    save_vectors_to_csv(selected_mismatches, test_data_1,
                        save_files_path+'selected_mismatches.csv', 'float_mismatch')
    # Save the selected matching vectors to a CSV file
    save_vectors_to_csv(selected_matches, test_data_2,
                        save_files_path+'selected_matches.csv', 'axon_matching')
    # Save the selected mismatching vectors to a CSV file
    save_vectors_to_csv(selected_mismatches, test_data_2,
                        save_files_path+'selected_mismatches.csv', 'axon_mismatch')
    return full_selected


def get_sample_raw_vectors_and_labels(raw_data_path, labels_path, vectors_list):
    """
    gets the raw vectors for a given set of vectors list
    saves the vectors along with the labels to a csv file
    """
    raw_data = np.load(raw_data_path)
    labels = np.load(labels_path)
    save_vectors_to_csv(vectors_list, raw_data,
                        'selected_raw.csv', 'raw_vectors')
    # Save the selected mismatching vectors to a CSV file
    save_vectors_to_csv(vectors_list, labels, 'selected_labels.csv', 'labels')


def load_axon_op_library(lib_path=r"..\..\samples\axon_fe_mfcc\simulator\build\Debug\axon_ml_shared_lib_mfcc.dll"):
    axon_ops_lib = cdll.LoadLibrary(lib_path)
    return axon_ops_lib


class AxonMfccLibClass:
    def __init__(self, lib_path):
        # Loading the dll
        self.axon_fe_dll = cdll.LoadLibrary(lib_path)

    def tensor_to_array(tensor1):
        return tensor1.numpy()

    def axon_mfcc_lib_calc_mfcc(self, input_array, output_array, ip_len, op_len):
        self._ip_len = ip_len
        self._op_len = op_len
        self._ip_memory = c_int16*(ip_len)
        self._op_memory = c_int32*(op_len)
        self._op_mfcc_op_width_memory = c_uint32*(1)
        self._mfcc_op_width_array = np.array([0])
        self._ip_ptr = input_array.ctypes.data_as(POINTER(self._ip_memory))
        self._op_ptr = output_array.ctypes.data_as(POINTER(self._op_memory))
        self._mfcc_op_width_array_ptr = self._mfcc_op_width_array.ctypes.data_as(
            POINTER(self._op_mfcc_op_width_memory))
        return self.axon_fe_dll.axon_mfcc_lib_calc_mfcc(self._ip_ptr, self._ip_len, self._op_ptr, self._mfcc_op_width_array_ptr)

    @tf.function
    def tf_axon_mfcc_lib_calc_mfcc(self, input_array, output_array, ip_len, op_len):
        # input_data = np.array(list(input_array.as_numpy_iterator()))
        # input_data = input_array.numpy()
        # input_data = input_array.eval(session=tf.compat.v1.Session())
        # tf.enable_eager_execution()
        # input_data = self.tensor_to_array(input_array)
        # import tensorflow.experimental.numpy as tnp
        # input_data = tnp.asarray(input_array)
        input_data = np.array(list(input_array))
        # input_data= input_array
        return tf.py_function(self.axon_mfcc_lib_calc_mfcc, inp=[input_data, output_array, ip_len, op_len], Tout=tf.int32)

    def axon_mfcc(self, _ip_ptr, _ip_len, _op_ptr, _mfcc_op_width_array_ptr):
        return self.axon_fe_dll.axon_mfcc_lib_calc_mfcc(_ip_ptr, _ip_len, _op_ptr, _mfcc_op_width_array_ptr)

    @tf.function
    def tf_axon_mfcc(self, input_array, output_array, ip_len, op_len):
        with tf.compat.v1.Session() as sess:
            input_data = sess.run(input_array)
        self._ip_len = ip_len
        self._op_len = op_len
        self._ip_memory = c_int16*(ip_len)
        self._op_memory = c_int32*(op_len)
        self._op_mfcc_op_width_memory = c_uint32*(1)
        self._mfcc_op_width_array = np.array([0])
        self._ip_ptr = input_data.ctypes.data_as(POINTER(self._ip_memory))
        self._op_ptr = output_array.ctypes.data_as(POINTER(self._op_memory))
        self._mfcc_op_width_array_ptr = self._mfcc_op_width_array.ctypes.data_as(
            POINTER(self._op_mfcc_op_width_memory))
        return tf.py_function(self.axon_mfcc, inp=[self._ip_ptr, self._ip_len, self._op_ptr, self._mfcc_op_width_array_ptr], Tout=tf.int32)

        # return self.axon_fe_dll.axon_mfcc_lib_calc_mfcc(self._ip_ptr, self._ip_len, self._op_ptr, self._mfcc_op_width_array_ptr)
    @tf.function
    def tf_graph_axon_mfcc(self, _ip_ptr, _ip_len, _op_ptr, _mfcc_op_width_array_ptr):
        return self.axon_fe_dll.axon_mfcc(_ip_ptr, _ip_len, _op_ptr, _mfcc_op_width_array_ptr)


def get_axon_mfcc(axon_ops_library, input_array, output_array, input_size, output_size):
    get_axon_mfcc = axon_ops_library.axon_mfcc_lib_calc_mfcc
    _ip_memory = c_int16*(input_size)
    _op_memory = c_int32*(output_size)
    _op_mfcc_op_width_memory = c_uint32*(1)
    _mfcc_op_width_array = np.array([0])
    _ip_ptr = input_array.ctypes.data_as(POINTER(_ip_memory))
    _op_ptr = output_array.ctypes.data_as(POINTER(_op_memory))
    _mfcc_op_width_array_ptr = _mfcc_op_width_array.ctypes.data_as(
        POINTER(_op_mfcc_op_width_memory))
    get_axon_mfcc(_ip_ptr, input_size, _op_ptr,
                  output_size, _mfcc_op_width_array_ptr)
    return 0


def dummy_scratchpad():
    # """ axon dll test code
    # #load the dll here.
    # # axon_mfcc_fe_lib = mdhs.load_axon_op_library()
    # test_sample = np.load(r"data\kws_g12\training_data\raw_data\val_x.npy")

    # test_sample_size = len(test_sample) #5000
    # ref_mfcc = np.load(train_args['test_feature_data'])[0:test_sample_size]
    # # load the data set, train, test and validation
    # # verify the library
    # # verify the length of the dataset to be given into the
    # # determine the output size of the mfccs
    # test_data = test_sample[0:test_sample_size] #test_sample_size]
    # len_t_data = len(test_data)
    # mfcc_output = np.zeros(shape=[len(test_data),490], dtype=np.int32)
    # start_time = time.time()
    # axon_fe_lib_obj = mdhs.AxonMfccLibClass(r"samples\axon_fe_mfcc\simulator\build\Debug\axon_ml_shared_lib_mfcc.dll")
    # for ndx in range(0,test_sample_size):
    #     ret = axon_fe_lib_obj.axon_mfcc_lib_calc_mfcc(test_data[ndx], mfcc_output[ndx],16000,490)

    # print(f"took {time.time() - start_time} seconds to get {test_sample_size} mfcc")
    # np.savetxt(r"testing_dll_mfcc.csv", mfcc_output, delimiter=',',fmt='%d')
    # mfcc_output = (mfcc_output/2**15).astype(np.float32)
    # return
    # """
    print("scratchpad")


def get_axon_fe_mfcc(axon_fe_lib, input_array, spectogram_length, dct_count, mfcc_shift):
    input_size = len(input_array)
    mfcc_output = np.zeros(
        shape=[input_size, spectogram_length*dct_count], dtype=np.int32)  # FIXME
    start_time = time.time()
    for ndx in range(0, input_size):
        _ = axon_fe_lib.axon_mfcc_lib_calc_mfcc(
            input_array[ndx], mfcc_output[ndx], 16000, 490)
    mfcc_output = (mfcc_output/2**mfcc_shift).astype(np.float32)
    mfcc_output = tf.reshape(
        mfcc_output, [input_size, spectogram_length, dct_count, 1])
    print(f"took {time.time() - start_time} seconds to get {input_size} mfcc")
    return mfcc_output


def convert_to_tensorflow_data_set(ds_x, ds_y, ds_x_key='audio', ds_y_key='label'):
    return tf.data.Dataset.from_tensor_slices({ds_x_key: ds_x,
                                               ds_y_key: ds_y})


if __name__ == "__main__":
    """
    The script can be called directly to run unit tests or to perform certain functions.
    To use the script add paths accordingly to run necessary vectors as needed.    
    """
    # get_min_max()
    # unit_test_generate_mfcc_csv()
    selected_vectors_list = get_mismatch_matching_vectors(r"data\kws_g12\training_data\kws_mfcc_testing_cp\kws_test_kws_data_yaml_tf_mfcc_lr_0.001\_0048_0.96366_9693.h5",
                                                          r"data\kws_g12\training_data\kws_mfcc_testing_cp\kws_axon_mfcc_16kfps_30ms_20ms_hann_fftmag_40fb_20to4khz_10coef_lr_0.001\_0037_0.95010_0.94752_92801.h5",
                                                          r"data\kws_g12\training_data\tf_mfcc_16kfps_30ms_20ms_hann_fftmag_40fb_20to4khz_10coef\test\tf_test_x.npy",
                                                          r"data\kws_g12\training_data\axon_mfcc_16kfps_30ms_20ms_hann_fftmag_40fb_20to4khz_10coef\test\test.npy",
                                                          r"data/kws_g12/training_data/labels/test_label_.npy",
                                                          r"data\kws_g12\training_data\raw_data\test_x.npy")
    # get_sample_raw_vectors_and_labels(r"data\kws_g12\training_data\raw_data\test_x.npy",
    #                                   r"data\kws_g12\training_data\raw_data\test_y.npy",
    #                                   #[2052,312,4707,3520,426,1212,1168,2136,3129,609,2667,3771,4368,3754,4099,699,2421,4747,1029,836,2586,4595,2623,4418,3640,4472,2511,1157,4331,2344,1387,634,2470,3713,1913,3587,3178,268,3213,4738])
    #                                   selected_vectors_list)
    print("Done!")
