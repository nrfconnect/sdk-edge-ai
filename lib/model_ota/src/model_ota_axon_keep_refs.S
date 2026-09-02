/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Device-wide Axon binding table. The merged binding-table header is force-included
 * by CMake (see model_ota_axon_binding_finalize()).
 *
 * Layout: [count, hash0, addr0, hash1, addr1, ...]
 */

#ifndef MODEL_OTA_AXON_BINDING_SYMBOL_COUNT
#error "MODEL_OTA_AXON_BINDING_SYMBOL_COUNT is missing"
#endif

.syntax unified
.section .rodata.model_ota_axon_binding,"a",%progbits
.align 2
.global model_ota_axon_binding_table
model_ota_axon_binding_table:
	.word MODEL_OTA_AXON_BINDING_SYMBOL_COUNT

#define MODEL_OTA_AXON_BINDING_REF(symbol) \
	.word MODEL_OTA_AXON_SYM_HASH(symbol) ; \
	.word symbol ;
MODEL_OTA_AXON_BINDING_REFS(MODEL_OTA_AXON_BINDING_REF)

.size model_ota_axon_binding_table, .-model_ota_axon_binding_table
