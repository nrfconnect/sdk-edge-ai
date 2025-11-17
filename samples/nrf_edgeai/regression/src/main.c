/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <nrf_edgeai/nrf_edgeai.h>
#include "nrf_edgeai_generated/nrf_edgeai_user_model.h"
#include <stdio.h>

int main(void)
{
	printk("Nordic EdgeAI Lab Regression sample project: \r\n");

    nrf_edgeai_t* p_edgeai = nrf_edgeai_user_model();
    nrf_edgeai_rt_version_t ver = nrf_edgeai_runtime_version();

    printk("\t EdgeAI Lib Runtime Version: %d.%d.%d\r\n", ver.field.major, ver.field.minor, ver.field.patch);
    printk("\t EdgeAI User Model Runtime Version: %d.%d.%d\r\n",
           p_edgeai->metadata.version.field.major,
           p_edgeai->metadata.version.field.minor,
           p_edgeai->metadata.version.field.patch);
    printk("\t EdgeAI User Solution id: %s\r\n", p_edgeai->metadata.p_solution_id);

	return 0;
}