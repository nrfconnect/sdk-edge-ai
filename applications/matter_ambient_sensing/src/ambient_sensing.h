/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#include <nrf_edgeai/nrf_edgeai.h>

namespace Nrf
{
namespace AmbientSensing
{
    bool process(nrf_edgeai_t *p_model);
    const char* getModelName();

} // namespace AmbientSensing
} // namespace Nrf
