/*
 * Copyright (c) 2022 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#include <lib/core/CHIPError.h>
#include <lib/shell/Engine.h>
#include <lib/shell/commands/Help.h>
#include "switch.h"

namespace SwitchCommands
{
void RegisterSwitchCommands(Switch &switchInstance);

} /* namespace SwitchCommands */
