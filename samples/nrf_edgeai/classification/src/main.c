// ///////////////////////// Package Header Files ////////////////////////////
// ////////////////////// Package Group Header Files /////////////////////////
#include <nrf_edgeai/nrf_edgeai.h>
#include "nrf_edgeai_generated/nrf_edgeai_user_model.h"
// /////////////////// Application Global Header Files ///////////////////////
// /////////////////// 3rd Party Software Header Files ///////////////////////
#include <zephyr/kernel.h>
// ////////////////////// Standard C++ Header Files //////////////////////////
// /////////////////////// Standard C Header Files ///////////////////////////
#include <stdio.h>
#include <assert.h>

/*
 * ============================================================================
 * USE CASE: Parcel State Detection During Delivery
 * ============================================================================
 *
 * OVERVIEW:
 * This application demonstrates a machine learning model for detecting the
 * state and handling conditions of a parcel during the delivery lifecycle.
 * The model uses triaxial accelerometer data to classify different parcel
 * states, which is useful for:
 *   - Detecting rough handling or potential damage to shipments
 *   - Monitoring delivery quality and logistics conditions
 *   - Alerting when parcels experience impacts or free falls
 *   - Tracking parcel state transitions throughout delivery
 *
 * INPUT DATA:
 * - Source: Triaxial accelerometer (X, Y, Z axes)
 * - Processing: Magnitude of acceleration vector
 *   Input = sqrt(x^2 + y^2 + z^2)  (removes directional bias)
 * - Window size: 50 consecutive samples
 * - This rolling window approach captures temporal patterns in acceleration
 *
 * MODEL OUTPUT:
 * The trained neural network classifies parcel state into 7 distinct classes:
 *   0. IDLE          - Parcel at rest (static state, ~1000 m/s^2 gravity)
 *   1. SHAKING       - Parcel vibrating/shaking (high frequency variation)
 *   2. IMPACT        - Sudden impact event (very high acceleration spike)
 *   3. FREE FALL     - Zero or near-zero gravity (dropped/in-air state)
 *   4. CARRYING - Being carried/moved by person
 *   5. IN_CAR        - Inside vehicle during transport
 *   6. PLACED        - Actively placed or set down
 *
 * INFERENCE WORKFLOW:
 * 1. Feed acceleration samples one point at a time (streaming input)
 * 2. Model accumulates 50 samples into a window
 * 3. Once window is full, inference runs automatically
 * 4. Output: predicted class + confidence probability for all 7 classes
 * 5. Result shows which state the parcel is currently in
 *
 * TESTING:
 * This program demonstrates the model on 7 representative pre-recorded
 * acceleration sequences (one per class). Each sequence contains real-world
 * acceleration data captured during that specific parcel state, allowing
 * validation that the model correctly identifies each state.
 * ============================================================================
 */

static const size_t USER_WINDOW_SIZE      = 50; /* Samples per inference window */
static const size_t USER_UNIQ_INPUTS_NUM  = 1;  /* Single input: acceleration magnitude */
static const size_t USER_MODELS_CLASS_NUM = 7;  /* 7 parcel state classes */

/* CLASS 0: IDLE STATE
 * Parcel at rest on a surface (e.g., on a shelf, table, or ground)
 * Characteristics:
 *   - Acceleration values cluster around 1000 mG (gravitational acceleration ~9.81 mG)
 *   - Minimal variation between samples (stable, static state)
 *   - No significant spikes or sudden changes
 * Use case: Parcel waiting at warehouse, on delivery truck stationary, or at customer
 */
static const flt32_t CLASS_0_PARCEL_IDLE_ACCEL_DATA[] = {
    1019.234877, 1018.652192, 1016.69811,  1019.901583, 1017.029378, 1014.381015, 1016.604559,
    1016.201444, 1016.23062,  1018.555051, 1016.319548, 1017.072842, 1018.494815, 1013.635677,
    1017.41575,  1021.132362, 1018.691935, 1019.848198, 1016.620341, 1011.873442, 1020.299764,
    1018.478126, 1015.541777, 1014.450299, 1013.253033, 1015.282387, 1016.849032, 1021.09152,
    1017.657982, 1013.24848,  1018.48856,  1013.41613,  1018.465645, 1017.460544, 1014.099637,
    1016.850408, 1016.396988, 1016.185508, 1014.927525, 1019.275502, 1017.403871, 1016.319314,
    1018.377489, 1019.466485, 1019.221851, 1013.788435, 1015.948113, 1014.715299, 1018.539064,
    1014.715299,
};

/* CLASS 1: SHAKING STATE
 * Parcel being vibrated, shaken, or experiencing continuous motion
 * Characteristics:
 *   - Wide variation in acceleration values (ranging 629–3505 mG)
 *   - High-frequency oscillations (rapid changes between samples)
 *   - Peaks and valleys indicate repetitive shaking motion
 * Use case: Parcel in vehicle with rough road, being shaken by handler, vibrating machine
 */
static const flt32_t CLASS_1_PARCEL_SHAKING_ACCEL_DATA[] = {
    832.512589,  1272.162323, 1263.232939, 1489.776683, 1859.176281, 1772.024529, 1622.484949,
    1421.231322, 988.9233308, 2904.564726, 1004.390724, 961.4903516, 1569.262094, 1926.891127,
    2159.370306, 1897.586384, 1533.182266, 2626.996728, 1044.248622, 866.9409276, 953.1315255,
    970.2099379, 1587.471444, 1810.551686, 1633.204426, 1730.792617, 2137.291854, 629.6878458,
    1069.066947, 1052.676461, 1091.051232, 1590.295378, 1840.453506, 1829.107198, 1790.32045,
    3505.788281, 1398.574137, 1089.641059, 625.6990249, 854.0307434, 1191.972966, 1523.862829,
    1490.407087, 1600.889671, 1487.519159, 1112.465548, 1909.979213, 1078.403761, 971.6052426,
    872.8287226,
};

/* CLASS 2: IMPACT EVENT
 * Parcel experiences a sudden collision or drop impact
 * Characteristics:
 *   - Baseline values ~1000–1400 mG (idle/stable state)
 *   - Extreme spike in middle of window (5943–6924 mG, ~7x gravity)
 *   - Sharp, isolated peak indicates instantaneous collision
 *   - Recovery to normal levels after impact
 * Use case: Parcel dropped, thrown, hits wall/object, collision during handling
 * Risk: May indicate potential damage to contents
 */
static const flt32_t CLASS_2_PARCEL_IMPACT_ACCEL_DATA[] = {
    1015.482154, 1014.753436, 1014.523099, 1012.472024, 1009.822764, 1013.528832, 1010.94256,
    1014.375469, 1010.949774, 1013.925464, 1015.508712, 1012.451178, 968.8407378, 1393.710166,
    5943.221118, 6924.142149, 4296.249304, 3458.151271, 2119.268542, 3263.52814,  1881.500201,
    1479.332158, 1364.573158, 2317.111272, 3159.792859, 2391.741083, 1305.986745, 1169.182085,
    1079.52232,  1048.65378,  1196.683715, 1054.96444,  1147.454278, 971.4221942, 1054.533138,
    1005.604551, 998.3117395, 1026.769424, 991.8001837, 1026.298981, 1007.115343, 1014.293972,
    1007.958389, 1009.272635, 1015.343811, 1013.057529, 1013.781359, 1012.335122, 1011.281924,
    1014.560157,

};

/* CLASS 3: FREE FALL STATE
 * Parcel is in the air, unsupported, experiencing near-zero or reduced gravity
 * Characteristics:
 *   - Very low acceleration values (15–82 mG, ~0.1–0.8g)
 *   - Consistent low baseline (opposed to gravity effect)
 *   - No gravitational component (sensor in free fall frame)
 * Use case: Parcel dropped and falling, in-flight (drone delivery), lifted/suspended
 * Risk: High—indicates drop or loss of support (potential damage)
 */
static const flt32_t CLASS_3_PARCEL_FREE_FALL_ACCEL_DATA[] = {
    36.87388192, 32.60300207, 29.4674831,  32.69417905, 34.11116509, 33.15887284, 33.85626394,
    32.15342346, 52.59657571, 36.6853006,  38.54968347, 35.5160419,  48.3840372,  47.2611057,
    39.59571977, 39.79519569, 47.38188498, 50.11275303, 55.69724245, 64.75752645, 70.27411802,
    68.7569314,  68.42144611, 66.124,      75.38968258, 58.17801248, 81.70503133, 81.17282629,
    74.35437106, 62.17788379, 63.89306679, 46.27838078, 43.04231927, 39.9609133,  39.70982871,
    34.28699252, 43.81753603, 27.73132727, 27.37479921, 46.95272669, 24.82332774, 30.18310574,
    37.00201876, 44.0080229,  39.55886449, 162.4691915, 50.70270683, 15.83370127, 75.95418619,
    82.38026851,

};

/* CLASS 4: TRANSPORTED BY COURIER (person carrying)
 * Parcel being held and carried by a delivery person or handler
 * Characteristics:
 *   - Moderate acceleration variation (793–1350 mG)
 *   - Gradual, smooth transitions (not sharp spikes)
 *   - Pattern reflects natural human motion (walking, arm swinging)
 *   - Organized rhythmic fluctuation (periodic step patterns)
 * Use case: Courier walking with parcel, handler carrying to customer door
 * Quality: Normal handling, no impact events
 */
static const flt32_t CLASS_4_PARCEL_CARRYING_ACCEL_DATA[] = {
    1076.406463, 1109.062753, 1161.315764, 1237.303839, 1304.16174,  1350.922984, 1241.533046,
    1058.547397, 936.9035098, 903.8836329, 909.9999218, 969.2561481, 1018.445741, 978.245237,
    898.4931508, 820.9886822, 793.33496,   823.5297855, 861.7252695, 896.6489743, 934.029546,
    949.6519496, 934.6501455, 959.1392024, 1009.025175, 1096.235981, 1187.678992, 1166.728833,
    1151.409319, 1130.995037, 1206.818376, 1211.209389, 1234.399883, 1316.256909, 1192.097129,
    935.7616639, 849.2381705, 855.3595664, 875.9844676, 940.2683021, 979.1213199, 974.3049546,
    894.6000374, 830.5673266, 806.1867451, 848.9594567, 879.9365677, 904.6198895, 916.9955333,
    963.7698856,

};

/* CLASS 5: IN CAR STATE (vehicle transport)
 * Parcel is inside a delivery vehicle (van, truck, car) during transport
 * Characteristics:
 *   - Moderate, steady acceleration (920–1200 mG)
 *   - Smoother variation compared to courier handling
 *   - Vehicle suspension and road vibration create gentle oscillations
 *   - Less organic (human) motion pattern vs. more mechanical (vehicle) pattern
 * Use case: Parcel in moving delivery van, on truck during road transport
 * Quality: Protected transport, minimal manual handling
 */
static const flt32_t CLASS_5_PARCEL_IN_CAR_ACCEL_DATA[] = {
    1020.14299,  1172.412078, 974.2434799, 1081.052374, 936.2345817, 1076.324048, 952.8650187,
    958.4387398, 919.7921364, 945.1008577, 1010.444803, 1048.425866, 954.5281871, 934.6416098,
    1039.243636, 983.2000815, 1106.467282, 1022.459808, 905.5254547, 1021.745856, 1036.285954,
    1061.250757, 1039.194138, 940.8242922, 1076.395069, 1059.220609, 1131.268418, 973.8779736,
    1029.466912, 1062.131636, 1077.031533, 1055.016753, 972.2285251, 1046.442069, 993.0253833,
    1035.990238, 1008.750742, 1035.737579, 1041.367931, 1042.022619, 960.7748412, 1037.639083,
    977.0186869, 990.7190264, 1054.852976, 979.9177917, 997.7978969, 1034.345125, 959.2851854,
    1041.668807,
};

/* CLASS 6: PLACED STATE (active placement event)
 * Parcel is being placed, set down, or positioned on a surface
 * Characteristics:
 *   - Starts with variable mid-range acceleration (1000–1400 mG)
 *   - Sharp drop to very low value (~76 mG, near free fall) in middle
 *   - Recovery back to stable ~1000 mG (landed and at rest)
 *   - Signature: acceleration change during descent and landing motion
 * Use case: Parcel being gently placed down, set on shelf, lowered to ground
 * Quality: Intentional placement, potentially fragile handling
 */
static const flt32_t CLASS_6_PARCEL_PLACED_ACCEL_DATA[] = {
    1028.084177, 1009.771942, 1023.977186, 1012.436213, 1020.685482, 1018.459741, 1013.2669,
    1402.760074, 1072.973569, 1044.170967, 996.3176143, 1090.362426, 1024.921702, 1037.91844,
    1052.900034, 1062.920713, 1020.651183, 966.0045051, 906.5110973, 871.8735271, 824.1462199,
    799.1876862, 776.2963823, 765.3527818, 802.7631891, 1163.179632, 1016.619697, 76.78637744,
    1418.807213, 1005.875935, 1033.329744, 1119.948199, 946.126825,  1083.038163, 949.8881454,
    1064.367743, 978.8891064, 1023.622866, 1026.007989, 1002.963002, 1026.46044,  1008.698302,
    1032.013223, 1016.890865, 1016.071373, 1022.803209, 1009.949278, 1007.64987,  997.3304425,
    1062.455154,
};

/* Enumeration of parcel state classes */
typedef enum
{
    MODEL_CLASS_IDLE = 0,
    MODEL_CLASS_SHAKING,
    MODEL_CLASS_IMPACT,
    MODEL_CLASS_FREE_FALL,
    MODEL_CLASS_CARRYING,
    MODEL_CLASS_IN_CAR,
    MODEL_CLASS_PLACED,
} user_model_class_t;

/* Human-readable labels for each parcel state class */
static const char* USER_MODEL_LABELS_STR[] = {
    [MODEL_CLASS_IDLE] = "Idle",         [MODEL_CLASS_SHAKING] = "Shaking",
    [MODEL_CLASS_IMPACT] = "Impact",     [MODEL_CLASS_FREE_FALL] = "Free Fall",
    [MODEL_CLASS_CARRYING] = "Carrying", [MODEL_CLASS_IN_CAR] = "in Car",
    [MODEL_CLASS_PLACED] = "Placed",
};

/**
 *  @brief Runs the trained parcel state model on acceleration data.
 * 
 * Parameters:
 *   @param[in] p_user_model: Pointer to initialized Edge AI model instance
 *   @param[in] p_input_data: Array of acceleration magnitude samples (sqrt(x^2+y^2+z^2))
 *   @param[in] data_len:     Total number of samples to process
 *
 * Process:
 *   1. Feed samples one-by-one (simulates real streaming sensor input)
 *   2. Model accumulates 50 samples into internal window
 *   3. When window is full, inference automatically triggers
 *   4. Extract predicted class and confidence probabilities
 *   5. Print results and return predicted class
 *
 * Return:
 *   Predicted class (0–6) on success, -1 if no prediction made
 */
static int32_t model_predict(nrf_edgeai_t*  p_user_model,
                             const flt32_t* p_input_data,
                             size_t         data_len)
{
    nrf_edgeai_err_t res;

    /* Feed samples point-by-point to match real-world streaming scenario */
    for (size_t i = 0; i < data_len; i++)
    {
        /* Accumulate single sample into the model's input window */
        flt32_t input_sample = p_input_data[i];
        res = nrf_edgeai_feed_inputs(p_user_model, &input_sample, 1 * USER_UNIQ_INPUTS_NUM);

        if (res == NRF_EDGEAI_ERR_SUCCESS)
        {
            /* Window full—run inference on collected 50-sample window */
            res = nrf_edgeai_run_inference(p_user_model);

            /* Check if inference completed and was successful */
            if (res == NRF_EDGEAI_ERR_SUCCESS)
            {
                /* Extract results from model output */
                uint16_t predicted_class = p_user_model->decoded_output.classif.predicted_class;
                uint16_t num_classes     = p_user_model->decoded_output.classif.num_classes;
                /* Confidence scores (probabilities) for all classes (f32, q16, q8 depending on quantization) */
                const flt32_t* p_probabilities =
                    p_user_model->decoded_output.classif.probabilities.p_f32;

                printk("In %u classes, predicted %u with probability %f\r\n",
                       num_classes,
                       predicted_class,
                       p_probabilities[predicted_class]);

                return predicted_class;
            }
        }
    }
    return -1;
}

/* ========================================================================
 * PARCEL STATE DETECTION - MODEL VALIDATION & DEMONSTRATION
 * ========================================================================
 *
 * This main function validates and demonstrates the trained parcel state
 * classification model on 7 representative test cases covering all classes.
 *
 * SETUP:
 *   1. Initialize the pre-trained neural network model
 *   2. Validate that model parameters match expected constants
 *   3. Run inference on each class test case
 *   4. Print predicted state and confidence for each
 *
 * EXPECTED BEHAVIOR:
 *   - Model should correctly classify each test sequence to its class
 *   - Confidence (probability) should be high for correct predictions
 *   - Output shows both numeric class ID and human-readable state name
 * ======================================================================== */
int main(void)
{
    /* Get user generated model pointer */
    nrf_edgeai_t* p_user_model = nrf_edgeai_user_model();

    /** Validate model parameters against expected configuration */
    assert(nrf_edgeai_input_window_size(p_user_model) == USER_WINDOW_SIZE);
    assert(nrf_edgeai_uniq_inputs_num(p_user_model) == USER_UNIQ_INPUTS_NUM);
    assert(nrf_edgeai_model_outputs_num(p_user_model) == USER_MODELS_CLASS_NUM);

    /** Initialize Edge AI runtime for inference execution */
    nrf_edgeai_err_t res = nrf_edgeai_init(p_user_model);
    assert(res == NRF_EDGEAI_ERR_SUCCESS);

    int32_t predicted_class;
    const size_t DATA_LEN = USER_WINDOW_SIZE * USER_UNIQ_INPUTS_NUM;

    /** TEST 1: Predict class 0 - Parcel in the IDLE state */
    printk("\n--- Testing IDLE state (parcel at rest) ---\r\n");
    predicted_class = model_predict(p_user_model, CLASS_0_PARCEL_IDLE_ACCEL_DATA, DATA_LEN);

    assert(predicted_class == MODEL_CLASS_IDLE);
    printk("Expected class IDLE - predicted %s\r\n", USER_MODEL_LABELS_STR[predicted_class]);

    /** TEST 2: Predict class 1 - Parcel is SHAKING */
    printk("\n--- Testing SHAKING state (parcel vibrating) ---\r\n");
    predicted_class = model_predict(p_user_model, CLASS_1_PARCEL_SHAKING_ACCEL_DATA, DATA_LEN);

    assert(predicted_class == MODEL_CLASS_SHAKING);
    printk("Expected class SHAKING - predicted %s\r\n", USER_MODEL_LABELS_STR[predicted_class]);

    /** TEST 3: Predict class 2 - Parcel IMPACT event */
    printk("\n--- Testing IMPACT event (collision detected) ---\r\n");
    predicted_class = model_predict(p_user_model, CLASS_2_PARCEL_IMPACT_ACCEL_DATA, DATA_LEN);

    assert(predicted_class == MODEL_CLASS_IMPACT);
    printk("Expected class IMPACT - predicted %s\r\n", USER_MODEL_LABELS_STR[predicted_class]);

    /** TEST 4: Predict class 3 - Parcel FREE FALL event */
    printk("\n--- Testing FREE FALL event (parcel in air/unsupported) ---\r\n");
    predicted_class = model_predict(p_user_model, CLASS_3_PARCEL_FREE_FALL_ACCEL_DATA, DATA_LEN);

    assert(predicted_class == MODEL_CLASS_FREE_FALL);
    printk("Expected class FREE FALL - predicted %s\r\n", USER_MODEL_LABELS_STR[predicted_class]);

    /** TEST 5: Predict class 4 - Parcel TRANSPORTED BY COURIER */
    printk("\n--- Testing CARRYING (person carrying) ---\r\n");
    predicted_class = model_predict(p_user_model, CLASS_4_PARCEL_CARRYING_ACCEL_DATA, DATA_LEN);

    assert(predicted_class == MODEL_CLASS_CARRYING);
    printk("Expected class CARRYING - predicted %s\r\n", USER_MODEL_LABELS_STR[predicted_class]);

    /** TEST 6: Predict class 5 - Parcel IN CAR */
    printk("\n--- Testing IN CAR state (vehicle transport) ---\r\n");
    predicted_class = model_predict(p_user_model, CLASS_5_PARCEL_IN_CAR_ACCEL_DATA, DATA_LEN);

    assert(predicted_class == MODEL_CLASS_IN_CAR);
    printk("Expected class IN CAR - predicted %s\r\n", USER_MODEL_LABELS_STR[predicted_class]);

    /** TEST 7: Predict class 6 - Parcel PLACED */
    printk("\n--- Testing PLACED state (active placement event) ---\r\n");
    predicted_class = model_predict(p_user_model, CLASS_6_PARCEL_PLACED_ACCEL_DATA, DATA_LEN);

    assert(predicted_class == MODEL_CLASS_PLACED);
    printk("Expected class PLACED - predicted %s\r\n", USER_MODEL_LABELS_STR[predicted_class]);

    while (1)
    {
        printk("\n========== All test cases completed ==========\r\n");
        k_sleep(K_MSEC(1000));
    }

    return 0;
}