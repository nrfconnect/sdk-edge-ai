/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef TEST_REGRESSION_DATA_H_
#define TEST_REGRESSION_DATA_H_

#include <nrf_edgeai/nrf_edgeai.h>

static const size_t USER_WINDOW_SIZE = 1;	 /* Samples per inference window */
static const size_t USER_UNIQ_INPUTS_NUM = 9;	 /* Gas sensor and environmental input features */
static const size_t USER_MODELS_OUTPUTS_NUM = 1; /* Single air quality prediction output */
static const flt32_t INVALID_PREDICTION_VALUE = -9999.0f; /* Invalid prediction indicator */
static const flt32_t EXPECTED_MODEL_MAE = 2.0f; /* Expected Mean Absolute Error for validation */

/**
 * @brief Test Dataset Structure and Values
 *
 * This structure defines the schema for test samples:
 * - COGT: CO sensor (main carbon monoxide detector)
 * - PT08S1-5: Five metal oxide semiconductor (MOS) sensors:
 *   * PT08S1: CO sensor output
 *   * PT08S2: NMHC (non-methane hydrocarbons) sensor
 *   * PT08S3: NOx (nitrogen oxides) sensor
 *   * PT08S4: NO2 (nitrogen dioxide) sensor
 *   * PT08S5: O3 (ozone) sensor
 * - T: Temperature in Celsius (affects pollutant concentration and sensor response)
 * - RH: Relative Humidity in percentage (influences sensor readings and pollutant behavior)
 * - AH: Absolute Humidity in kg/m³ (moisture content in air)
 * - target: Ground truth air quality value (used for validation)
 *
 * @note All sensor values are normalized or raw readings from the measurement device.
 * The 29 test samples cover various environmental conditions to validate model performance
 * across different temperature, humidity, and pollution levels.
 */
struct {
	flt32_t COGT;	/**< CO sensor reading */
	flt32_t PT08S1; /**< PT08 MOS sensor 1 (CO) */
	flt32_t PT08S2; /**< PT08 MOS sensor 2 (NMHC) */
	flt32_t PT08S3; /**< PT08 MOS sensor 3 (NOx) */
	flt32_t PT08S4; /**< PT08 MOS sensor 4 (NO2) */
	flt32_t PT08S5; /**< PT08 MOS sensor 5 (O3) */
	flt32_t T;	/**< Temperature in Celsius */
	flt32_t RH;	/**< Relative Humidity in percentage */
	flt32_t AH;	/**< Absolute Humidity in kg/m³ */
	flt32_t target; /**< Ground truth air quality value */
} static const USER_INPUT_DATA[] = {
	[0] = {2.7, 1146, 1125, 846, 1511, 1016, 33.4, 20.3, 1.027, 14.3},
	[1] = {3.3, 1272, 1328, 567, 2085, 1463, 19.6, 54.3, 1.2278, 21.1},
	[2] = {0.6, 919, 571, 1017, 1082, 521, 13.5, 63.8, 0.9801, 1.8},
	[3] = {1.7, 1162, 1019, 622, 1904, 1178, 27.9, 56.5, 2.0895, 11.1},
	[4] = {2.7, 1381, 1227, 595, 1903, 1845, 29.8, 33.1, 1.3693, 17.6},
	[5] = {1.1, 1010, 679, 854, 1046, 889, 5, 82.9, 0.73, 3.4},
	[6] = {1.9, 1055, 1037, 635, 1632, 1161, 25.3, 41.7, 1.3247, 11.6},
	[7] = {3.4, 1417, 1303, 635, 1964, 1752, 20.7, 40, 0.9639, 20.2},
	[8] = {2, 1273, 988, 563, 1387, 1257, 24.1, 33.3, 0.9881, 10.3},
	[9] = {4.8, 1435, 1429, 499, 2072, 1449, 21.7, 59.1, 1.5115, 25},
	[10] = {1, 1003, 635, 829, 1235, 711, 13.7, 80.6, 1.2528, 2.7},
	[11] = {1.3, 1259, 1152, 610, 1165, 1569, 8.2, 36.8, 0.4016, 15.1},
	[12] = {2.7, 1229, 1114, 598, 1723, 1247, 19.7, 0.73, 0.6708, 13.9},
	[13] = {2.8, 1261, 1258, 629, 1813, 1315, 43.4, 14.8, 1.2882, 18.6},
	[14] = {1.3, 997, 752, 837, 952, 724, 11.8, 34.4, 0.4758, 4.8},
	[15] = {1.3, 942, 846, 980, 1615, 905, 21.7, 50.8, 1.3019, 6.7},
	[16] = {0.6, 883, 518, 1135, 962, 606, 3.3, 84.5, 0.6612, 1.2},
	[17] = {3.2, 1174, 1264, 670, 1598, 1287, 28.9, 21.3, 0.8382, 18.8},
	[18] = {0.5, 748, 595, 1208, 1089, 677, 12.8, 0.598, 0.8801, 2.2},
	[19] = {1.4, 920, 783, 1046, 1550, 588, 24.5, 0.384, 1.1669, 5.4},
	[20] = {1.3, 906, 790, 893, 837, 642, 12.3, 19.3, 0.274, 5.5},
	[21] = {0.6, 882, 563, 978, 936, 660, 8.9, 0.527, 0.6034, 1.7},
	[22] = {3.4, 1403, 1443, 508, 2234, 1811, 25.2, 0.386, 1.2215, 25.6},
	[23] = {3.9, 1297, 1102, 507, 1375, 1583, 18.2, 0.363, 0.7487, 13.6},
	[24] = {1.3, 987, 800, 989, 1462, 658, 15.5, 0.571, 0.996, 5.7},
	[25] = {1.3, 869, 866, 1107, 1212, 596, 0.238, 0.145, 0.4222, 7.2},
	[26] = {3.2, 1336, 1340, 540, 2049, 1400, 21.3, 0.635, 1.5941, 21.6},
	[27] = {1, 955, 723, 1129, 1393, 559, 27.7, 0.258, 0.9467, 4.2},
	[28] = {4.3, 1373, 1364, 597, 2005, 1745, 33.7, 0.226, 1.1658, 22.5}};


#endif /* TEST_REGRESSION_DATA_H_ */
