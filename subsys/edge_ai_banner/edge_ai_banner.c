/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/init.h>
#include <zephyr/sys/printk.h>
#include <edge_ai_version.h>
#include <edge_ai_commit.h>

static int edge_ai_boot_banner(void)
{
	printk("*** Using Edge AI Add-on v" EDGE_AI_VERSION_STRING
	       "-" EDGE_AI_COMMIT_STRING " ***\n");
	return 0;
}
SYS_INIT(edge_ai_boot_banner, APPLICATION, 1);
