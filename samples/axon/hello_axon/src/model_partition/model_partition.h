/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#include <drivers/axon/nrf_axon_nn_infer.h>

/**
 * @brief Load hello_axon model metadata and point it at the partition-backed weights.
 *
 * @param[out] model Populated compiled model descriptor.
 *
 * @retval 0 on success.
 * @retval -EINVAL if the partition image is invalid.
 */
int hello_axon_model_partition_load(nrf_axon_nn_compiled_model_s *model);
