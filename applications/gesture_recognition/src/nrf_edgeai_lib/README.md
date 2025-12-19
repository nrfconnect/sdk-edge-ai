# nRF EdgeAI C Library

nRF EdgeAI  is an unique neural network framework, natively creates incredibly compact and accurate TinyML models that can easily be deployed into your firmware project using an automatically generated archive with a C library.

The nRF EdgeAI library is static C library, written in accordance with the C99 standard and has no dependencies other than the standard C language library (libc). Our library does not use dynamic memory allocation, stack usage is kept to a minimum.

# How to integrate library into your firmware project 

## nRF EdgeAI folder structure

There are only two folders that should be added to your project for integrating library:
 * `nrf_edgeai` - main framework folder with all source code and binaries:
    * `nrf_edgeai/lib` -  folder with precompiled libraries for different architectures, e.g. `libnrf_edgeai_cortex-m33.a`.
    * `nrf_edgeai/include` - folder with all header files and interfaces for library usage.
 * `nrf_edgeai_generated` - folder containing all user solution-specific files (`nrf_edgeai_user_model.c/h`).

1. Link nRF EdgeAI C static library for your CPU architecture to your firmware project.
2. Add `./nrf_edgeai/include` folder to your project global INCLUDE paths.
3. **Optional** Add folder where  `./nrf_edgeai_generated` **is located** (not this folder itself) to your project global INCLUDE paths.

## Use nRF EdgeAI in your project

First of all, you need to include the main header file of nRF EdgeAI library:
``` C
#include <nrf_edgeai/nrf_edgeai.h>
#include <nrf_edgeai_generated/nrf_edgeai_user_model.h>
```
For making an inference with nRF EdgeAI, there are only 4 functions that you should use:
* `nrf_edgeai_user_model` - Get user model pointer;
* `nrf_edgeai_init` - Set up the internal components of nRF EdgeAI Lab user model, **should be called first and once**;
* `nrf_edgeai_feed_inputs` - Feed and prepare live input features for model inference;
* `nrf_edgeai_run_inference` - Run live input features into a nRF EdgeAI machine learning algorithm (or “ML/NN model”) to inference an output;

### How to feed input data (raw features / sensor data)

Collect the array of the input data, for example the input data can be sensor data or any type of signal data. The number of samples in the input data array and their order should be **the same** as in the training dataset (for example Accelerometer_X, Accelerometer_Y, Accelerometer_Z).

Pass the array of raw input data to the `nrf_edgeai_feed_inputs` function.
``` C
#include <nrf_edgeai/nrf_edgeai.h>
#include <nrf_edgeai_generated/nrf_edgeai_user_model.h>
#include <assert.h>
// In this example, our raw features are the 3 accelerometer axis values
// The number of raw features and their order should be the same as in the training dataset file
int16_t raw_features[] = 
{
    Accelerometer_X,
    Accelerometer_Y,
    Accelerometer_Z
};
// Pointer to user model
static nrf_edgeai_t* p_edgeai = NULL;

void user_init_edegeai_model(void)
{
    // Get user model pointer
    p_edgeai = nrf_edgeai_user_model();
    // Init EdgeAI library based on user solution, should be called once!
    nrf_edgeai_err_t res = nrf_edgeai_init(p_edgeai);
    // Optional check for success, #include <assert.h> required
    assert(res == NRF_EDGEAI_ERR_SUCCESS);
}
//
// ....
//
void user_feed_data_to_model(void)
{
    uint32_t number_of_features = nrf_edgeai_uniq_inputs_num(p_edgeai); // Will be equal to 3
    // Feed raw input features to prepare them for model inference
    nrf_edgeai_err_t res = nrf_edgeai_feed_inputs(p_edgeai, raw_features, number_of_features);

    // Check if input data is prepared and ready for model inference
    if (res == NRF_EDGEAI_ERR_SUCCESS)
    {
        //make inference
    }
}

```
In case the user solution needs to collect a data window, you can use the following sample-by-sample feed algorithm:
``` C
#include <nrf_edgeai/nrf_edgeai.h>
#include <nrf_edgeai_generated/nrf_edgeai_user_model.h>
#include <assert.h>
// In this example, our raw features is a window of N elements with 3 accelerometer axis values
// The number of raw features and their order should be the same as in the training dataset file
int16_t raw_features[] = 
{
    Accelerometer_X0,
    Accelerometer_Y0,
    Accelerometer_Z0,
    /* ... */
    Accelerometer_Xn,
    Accelerometer_Yn,
    Accelerometer_Zn,
};
// Pointer to user model
static nrf_edgeai_t* p_edgeai = NULL;

void user_init_edegeai_model(void)
{
    // Get user model pointer
    p_edgeai = nrf_edgeai_user_model();
    // Init EdgeAI library based on user solution, should be called once!
    nrf_edgeai_err_t res = nrf_edgeai_init(p_edgeai);
    // Optional check for success, #include <assert.h> required
    assert(res == NRF_EDGEAI_ERR_SUCCESS);
}
//
// ....
//
void user_feed_data_to_model(void)
{
    // Feed raw input features sample-by-sample to prepare them for model inference
    uint32_t number_of_features = nrf_edgeai_uniq_inputs_num(p_edgeai); // Will be equal to 3
    uint32_t window_size_of_features = nrf_edgeai_input_window_size(p_edgeai); // Will be equal to N
    int16_t* p_raw_features_begin = &raw_features[0];

    for (size_t i = 0; i < window_size_of_features; i++)
    {
        nrf_edgeai_err_t res = nrf_edgeai_feed_inputs(p_edgeai, p_raw_features_begin, number_of_features);

        // Check if input data is prepared and ready for model inference
        if (res == NRF_EDGEAI_ERR_SUCCESS)
        {
            //make inference
            break;
        }
        // Shift pointer to raw features buffer by number_of_features
        p_raw_features_begin += number_of_features;
    }
}
```

###	How to Run Model Inference

When input features is ready for running inference, you should call `nrf_edgeai_run_inference` function with three arguments:

``` C
/**
 * @brief Running live input features into a EdgeAI machine learning algorithm (or “ML model”) to inference an output
 * 
 * @details If the operation is succeeded (NRF_EDGEAI_ERR_SUCCESS), 
 *          the inference result you can get from p_edgeai->decoded_output.regression or p_edgeai->decoded_output.classif depending on your model task
 * 
 * @param[in, out] p_edgeai     Pointer to neural network context @ref nrf_edgeai_t
 * 
 * @return Operation status code @ref nrf_edgeai_err_t 
 */
nrf_edgeai_err_t nrf_edgeai_run_inference(nrf_edgeai_t* p_edgeai);
```
Inference results you can get from `p_edgeai->decoded_output` structure:
- `p_edgeai->decoded_output.classif` - for classification models.
- `p_edgeai->decoded_output.regression` - for regression models.
- `p_edgeai->decoded_output.anomaly` - for anomaly detection models.


### Wrapping everything together for Classification

``` C
#include <nrf_edgeai/nrf_edgeai.h>
#include <nrf_edgeai_generated/nrf_edgeai_user_model.h>
#include <assert.h>
#include <stdio.h>
// In this example, our raw features is a window of N elements with 3 accelerometer axis values
// The number of raw features and their order should be the same as in the training dataset file
int16_t raw_features[] = 
{
    Accelerometer_X0,
    Accelerometer_Y0,
    Accelerometer_Z0,
    /* ... */
    Accelerometer_Xn,
    Accelerometer_Yn,
    Accelerometer_Zn,
};
// Pointer to user model
static nrf_edgeai_t* p_edgeai = NULL;

void user_init_edegeai_model(void)
{
    // Get user model pointer
    p_edgeai = nrf_edgeai_user_model();
    // Init EdgeAI library based on user solution, should be called once!
    nrf_edgeai_err_t res = nrf_edgeai_init(p_edgeai);
    // Optional check for success, #include <assert.h> required
    assert(res == NRF_EDGEAI_ERR_SUCCESS);
}
//
// ....
//
void user_feed_data_to_model(void)
{
    // Feed and prepare raw inputs for the model inference
    nrf_edgeai_err_t res = nrf_edgeai_feed_inputs(p_edgeai, raw_features, 
                                            nrf_edgeai_uniq_inputs_num(p_edgeai) * 
                                            nrf_edgeai_input_window_size(p_edgeai));

    // Check if input data is prepared and ready for model inference
    if (res == NRF_EDGEAI_ERR_SUCCESS)
    {
        // Run model inference
        res = nrf_edgeai_run_inference(p_edgeai);
        // Check if model inference is ready and successful
        if (res == NRF_EDGEAI_ERR_SUCCESS)
        {
            uint16_t predicted_class = p_edgeai->decoded_output.classif.predicted_class;
            size_t num_classes = p_edgeai->decoded_output.classif.num_classes;
            // Get probability depending on model quantization: f32, q16, q8. Here is an example for f32 model
            const flt32_t* p_probabilities = p_edgeai->decoded_output.classif.probabilities.p_f32;

            printf("Predicted class %u with probability %f, in %u classes\r\n", predicted_class, 
                                                                                p_probabilities[predicted_class],
                                                                                num_classes);
        }
    }
    
}
```

### Model Inference for Regression

``` C
#include <nrf_edgeai/nrf_edgeai.h>
#include <nrf_edgeai_generated/nrf_edgeai_user_model.h>
#include <assert.h>
#include <stdio.h>
// In this example, our raw features is a window of N elements with 3 accelerometer axis values
// The number of raw features and their order should be the same as in the training dataset file
int16_t raw_features[] = 
{
    Accelerometer_X0,
    Accelerometer_Y0,
    Accelerometer_Z0,
    /* ... */
    Accelerometer_Xn,
    Accelerometer_Yn,
    Accelerometer_Zn,
};
// Pointer to user model
static nrf_edgeai_t* p_edgeai = NULL;

void user_init_edegeai_model(void)
{
    // Get user model pointer
    p_edgeai = nrf_edgeai_user_model();
    // Init EdgeAI library based on user solution, should be called once!
    nrf_edgeai_err_t res = nrf_edgeai_init(p_edgeai);
    // Optional check for success, #include <assert.h> required
    assert(res == NRF_EDGEAI_ERR_SUCCESS);
}
//
// ....
//
void user_feed_data_to_model(void)
{
    // Feed and prepare raw inputs for the model inference
    nrf_edgeai_err_t res = nrf_edgeai_feed_inputs(p_edgeai, raw_features, 
                                            nrf_edgeai_uniq_inputs_num(p_edgeai) * 
                                            nrf_edgeai_input_window_size(p_edgeai));

    // Check if input data is prepared and ready for model inference
    if (res == NRF_EDGEAI_ERR_SUCCESS)
    {
        // Run model inference
        res = nrf_edgeai_run_inference(p_edgeai);
        // Check if model inference is ready and successful
        if (res == NRF_EDGEAI_ERR_SUCCESS)
        {
            const flt32_t* p_predicted_values = p_edgeai->decoded_output.regression.p_outputs;
            size_t values_num = p_edgeai->decoded_output.regression.outputs_num;

            printf("Predicted target values:\r\n");
            for (size_t i = 0; i < values_num; i++)
            {
                printf("%f,", p_predicted_values[i]);
            }
            printf("\r\n");
        }
    }
    
}
```

### Model Inference for Anomaly Detection


In anomaly detection mode, the model's inference yields an Anomaly Score, indicating the similarity of input data to the "normal" data used for training. A higher Anomaly Score signifies greater deviation from the normal data, while a score close to zero indicates normal data. Because the model learns only from normal data, it cannot predict the presence of anomalies, only deviation from the normal data, so user must set a threshold based on the Anomaly Score to identify anomalies.

``` C
#include <nrf_edgeai/nrf_edgeai.h>
#include <nrf_edgeai_generated/nrf_edgeai_user_model.h>
#include <assert.h>
#include <stdio.h>
// User should define Anomaly Score Threshold to identify anomalies by himself,
// specific to user application
#define USER_DEFINED_ANOMALY_SCORE_THRESHOLD 0.6f
// In this example, our raw features is a window of N elements with 3 accelerometer axis values
// The number of raw features and their order should be the same as in the training dataset file
int16_t raw_features[] = 
{
    Accelerometer_X0,
    Accelerometer_Y0,
    Accelerometer_Z0,
    /* ... */
    Accelerometer_Xn,
    Accelerometer_Yn,
    Accelerometer_Zn,
};
// Pointer to user model
static nrf_edgeai_t* p_edgeai = NULL;

void user_init_edegeai_model(void)
{
    // Get user model pointer
    p_edgeai = nrf_edgeai_user_model();
    // Init EdgeAI library based on user solution, should be called once!
    nrf_edgeai_err_t res = nrf_edgeai_init(p_edgeai);
    // Optional check for success, #include <assert.h> required
    assert(res == NRF_EDGEAI_ERR_SUCCESS);
}
//
// ....
//
void user_feed_data_to_model(void)
{
    // Feed and prepare raw inputs for the model inference
    nrf_edgeai_err_t res = nrf_edgeai_feed_inputs(p_edgeai, raw_features, 
                                            nrf_edgeai_uniq_inputs_num(p_edgeai) * 
                                            nrf_edgeai_input_window_size(p_edgeai));

    // Check if input data is prepared and ready for model inference
    if (res == NRF_EDGEAI_ERR_SUCCESS)
    {
        // Run model inference
        res = nrf_edgeai_run_inference(p_edgeai);
        // Check if model inference is ready and successful
        if (res == NRF_EDGEAI_ERR_SUCCESS)
        {
            flt32_t anomaly_score = p_edgeai->decoded_output.anomaly.score;

            printf("Predicted Anomaly score: %f\r\n", anomaly_score);

            if (anomaly_score > USER_DEFINED_ANOMALY_SCORE_THRESHOLD)
            {
                printf("Anomaly detected!\n");
            }
        }
    }
    
}
```

### Additional solution information API

You can use the following API to get solution information:
* `nrf_edgeai_solution_id_str` - Get user solution ID in string format;
* `nrf_edgeai_uniq_inputs_num` - Get number of unique input features on which the model was trained;
* `nrf_edgeai_input_window_size` - Get input features window size in feature samples(vectors);
* `nrf_edgeai_model_neurons_num` - Get number of model neurons;
* `nrf_edgeai_model_weights_num` - Get number of model weights;
* `nrf_edgeai_model_outputs_num` - Get number of model outputs (predicted targets);
* `nrf_edgeai_model_task` - Get model task : nrf_edgeai_TASK_MULT_CLASS, nrf_edgeai_TASK_BIN_CLASS, nrf_edgeai_TASK_REGRESSION
* `nrf_edgeai_user_model_size` - Get user model size in bytes (flash usage)