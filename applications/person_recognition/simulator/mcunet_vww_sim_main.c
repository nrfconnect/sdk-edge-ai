/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Host Axon simulator: run mcunet_vww_320kb once (same path as on-device).
 * Input: raw int8 CHW packed buffer (see scripts/export_mcunet_input_bin.py).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "axon/nrf_axon_platform.h"
#include "drivers/axon/nrf_axon_driver.h"
#include "drivers/axon/nrf_axon_nn_infer.h"

#define NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER 1
#include "nrf_axon_model_mcunet_vww_320kb_.h"

static size_t external_input_bytes(const nrf_axon_nn_compiled_model_s *m)
{
	const nrf_axon_nn_compiled_model_input_s *in = &m->inputs[m->external_input_ndx];

	return (size_t)in->stride * in->dimensions.height * in->dimensions.channel_cnt;
}

int main(int argc, char *argv[])
{
	const nrf_axon_nn_compiled_model_s *model = &model_mcunet_vww_320kb;
	const size_t in_bytes = external_input_bytes(model);
	static int8_t input_buf[70000];
	int8_t output_buf[NRF_AXON_MODEL_MCUNET_VWW_320KB_PACKED_OUTPUT_SIZE];

	if (in_bytes > sizeof(input_buf)) {
		fprintf(stderr, "input %zu exceeds static buffer\n", in_bytes);
		return 1;
	}

	if (argc >= 2) {
		FILE *fp = fopen(argv[1], "rb");
		if (!fp) {
			perror(argv[1]);
			return 1;
		}
		size_t n = fread(input_buf, 1, in_bytes, fp);
		fclose(fp);
		if (n != in_bytes) {
			fprintf(stderr, "expected %zu input bytes, got %zu\n", in_bytes, n);
			return 1;
		}
		printf("Loaded %zu bytes from %s\n", n, argv[1]);
	} else {
		memset(input_buf, 0, in_bytes);
		printf("No input file: using zeros (%zu bytes)\n", in_bytes);
	}

	printf("Axon sim: model %s input %zu bytes\n", model->model_name, in_bytes);

	if (nrf_axon_platform_init() != NRF_AXON_RESULT_SUCCESS) {
		fprintf(stderr, "platform_init failed\n");
		return 1;
	}
	if (nrf_axon_nn_model_validate(model) != NRF_AXON_RESULT_SUCCESS) {
		fprintf(stderr, "model_validate failed\n");
		return 1;
	}
	if (nrf_axon_nn_model_init_vars(model) != 0) {
		fprintf(stderr, "init_vars failed\n");
		return 1;
	}

	nrf_axon_result_e r = nrf_axon_nn_model_infer_sync(model, input_buf, output_buf);
	if (r != NRF_AXON_RESULT_SUCCESS) {
		fprintf(stderr, "infer_sync failed: %d\n", r);
		return 1;
	}

	int32_t score = 0;
	int16_t cls = nrf_axon_nn_get_classification(model, output_buf, NULL, &score);

	printf("classification idx=%d score=%d\n", (int)cls, (int)score);
	printf("raw output (%d bytes):", NRF_AXON_MODEL_MCUNET_VWW_320KB_PACKED_OUTPUT_SIZE);
	for (int i = 0; i < NRF_AXON_MODEL_MCUNET_VWW_320KB_PACKED_OUTPUT_SIZE; i++) {
		printf(" %d", output_buf[i]);
	}
	printf("\n");

	nrf_axon_platform_close();
	return 0;
}
