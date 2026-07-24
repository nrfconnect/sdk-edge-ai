/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Keep app symbols referenced by an Axon partition image. The per-model
 * private configuration header is force-included by CMake.
 */

#if !defined(MODEL_OTA_AXON_CONFIG_VERSION) || (MODEL_OTA_AXON_CONFIG_VERSION != 1)
#error "Unsupported or missing Axon OTA configuration"
#endif

#ifndef MODEL_OTA_AXON_KEEP_LABEL
#error "MODEL_OTA_AXON_KEEP_LABEL is missing"
#endif

.syntax unified
.section .rodata.model_ota_axon_keep,"a",%progbits
.align 2
.global MODEL_OTA_AXON_KEEP_LABEL
MODEL_OTA_AXON_KEEP_LABEL:

#define MODEL_OTA_AXON_KEEP_REF(symbol) .word symbol ;
MODEL_OTA_AXON_KEEP_REFS(MODEL_OTA_AXON_KEEP_REF)

.size MODEL_OTA_AXON_KEEP_LABEL, .-MODEL_OTA_AXON_KEEP_LABEL
