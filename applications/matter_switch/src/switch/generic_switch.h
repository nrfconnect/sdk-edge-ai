/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#include "light_switch.h"


/** @class GenericSwitch
 *  @brief Class for controlling a CHIP generic switch over a Thread network
 *
 *  Features:
 *  - implementing the Momentary Switch (`MS`) and Momentary Switch Release (`MSR`) features.
 */
class GenericSwitch : public Switch {

public:

    GenericSwitch() = default;

    void Init() override;
    void InitiateActionSwitch(Action action) override;
    void DimmerChangeBrightness() override;

private:

    void InitialPressHandler();
    void ShortReleaseHandler();
    void LongReleaseHandler();
    void LongPressHandler();
};
