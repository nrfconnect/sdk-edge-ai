#!/usr/bin/env python3
# Copyright (c) 2026 Nordic Semiconductor ASA
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

"""FNV-1a contract hashing mirroring include/model_ota/model_contract.h."""

from __future__ import annotations

import re
from pathlib import Path

MODEL_IMAGE_FORMAT_VERSION = 5
MODEL_IMAGE_OFFSET_MCUBOOT = 32
MODEL_OTA_CONTRACT_BACKEND_NEUTON = 0
MODEL_OTA_CONTRACT_BACKEND_AXON = 1

FNV1A_INIT = 2166136261
FNV1A_MUL = 16777619

# Struct sizes on target (arm-zephyr-eabi, 32-bit pointers) — must match sizeof() in
# include/model_ota/model_contract.h on the firmware build.
SIZEOF_NEUTON_MODEL = 40  # nrf_edgeai_model_neuton_t (meta + params union)
SIZEOF_NEUTON_META = 28     # nrf_nn_neuton_model_meta_t
SIZEOF_DECODED_OUTPUT = 16  # nrf_edgeai_decoded_output_t (union of task outputs)
NEURON_ELEM = {0: 4, 1: 2, 2: 1}

INPUT_FEATURE_DATA_TYPE_MAP = {
    "NRF_EDGEAI_INPUT_I8": 1,
    "NRF_EDGEAI_INPUT_I16": 2,
    "NRF_EDGEAI_INPUT_F32": 4,
}



def fnv1a_u32(state: int, value: int) -> int:
    return ((state ^ (value & 0xFFFFFFFF)) * FNV1A_MUL) & 0xFFFFFFFF


def fnv1a_str(state: int, text: str) -> int:
    for byte in text.encode("ascii"):
        state = fnv1a_u32(state, byte)
    return state


def neuton_pipeline_hash(
    input_feature_type: int,
    window_size: int,
    window_shift: int,
    uniq_features: int,
    uses_input: int,
    uses_dsp: int,
) -> int:
    state = FNV1A_INIT
    state = fnv1a_u32(state, input_feature_type)
    state = fnv1a_u32(state, window_size)
    state = fnv1a_u32(state, window_shift)
    state = fnv1a_u32(state, uniq_features)
    state = fnv1a_u32(state, uses_input)
    return fnv1a_u32(state, uses_dsp)


def contract_hash_neuton(
    *,
    task: int,
    params_type: int,
    outputs_cap: int,
    inputs_num: int,
    neurons_cap: int,
    solution_id: str,
    pipeline_hash: int,
) -> int:
    state = FNV1A_INIT
    state = fnv1a_u32(state, MODEL_IMAGE_FORMAT_VERSION)
    state = fnv1a_u32(state, MODEL_OTA_CONTRACT_BACKEND_NEUTON)
    state = fnv1a_u32(state, task)
    state = fnv1a_u32(state, params_type)
    state = fnv1a_u32(state, SIZEOF_NEUTON_MODEL)
    state = fnv1a_u32(state, SIZEOF_NEUTON_META)
    state = fnv1a_u32(state, SIZEOF_DECODED_OUTPUT)
    state = fnv1a_u32(state, NEURON_ELEM[params_type])
    state = fnv1a_u32(state, outputs_cap)
    state = fnv1a_u32(state, inputs_num)
    state = fnv1a_u32(state, neurons_cap)
    state = fnv1a_u32(state, pipeline_hash)
    return fnv1a_str(state, solution_id)


def contract_hash_axon(
    *,
    compiled_model_size: int,
    interlayer_size: int,
    psum_size: int,
    persistent_required: int,
    packed_output_bytes: int,
) -> int:
    state = FNV1A_INIT
    state = fnv1a_u32(state, MODEL_IMAGE_FORMAT_VERSION)
    state = fnv1a_u32(state, MODEL_OTA_CONTRACT_BACKEND_AXON)
    state = fnv1a_u32(state, compiled_model_size)
    state = fnv1a_u32(state, interlayer_size)
    state = fnv1a_u32(state, psum_size)
    state = fnv1a_u32(state, persistent_required)
    return fnv1a_u32(state, packed_output_bytes)


def parse_define_int(
    source: str,
    name: str,
    default: int | None = None,
    *,
    token_map: dict[str, int] | None = None,
) -> int:
    match = re.search(rf"^\s*#define\s+{re.escape(name)}\s+(\d+)\s*$", source, re.MULTILINE)
    if match is not None:
        return int(match.group(1))
    match = re.search(rf"^\s*#define\s+{re.escape(name)}\s+(\S+)\s*$", source, re.MULTILINE)
    if match is not None and token_map is not None:
        token = match.group(1)
        if token in token_map:
            return token_map[token]
    if default is not None:
        return default
    raise ValueError(f"{name} not found in model source")


def parse_define_token(source: str, name: str) -> str:
    match = re.search(rf'^\s*#define\s+{re.escape(name)}\s+"([^"]+)"\s*$', source, re.MULTILINE)
    if match is None:
        match = re.search(rf"^\s*#define\s+{re.escape(name)}\s+(\S+)\s*$", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"{name} not found in model source")
    return match.group(1).strip('"')


def params_type_from_model(source: str) -> int:
    token = parse_define_token(source, "MODEL_PARAMS_TYPE")
    mapping = {"f32": 0, "q16": 1, "q8": 2}
    if token not in mapping:
        raise ValueError(f"unsupported MODEL_PARAMS_TYPE {token!r}")
    return mapping[token]


def neuton_contract_from_model_c(path: Path, neurons_cap: int) -> int:
    source = path.read_text(encoding="utf-8")
    pipeline = neuton_pipeline_hash(
        parse_define_int(source, "INPUT_FEATURE_DATA_TYPE", token_map=INPUT_FEATURE_DATA_TYPE_MAP),
        parse_define_int(source, "INPUT_WINDOW_SIZE"),
        parse_define_int(source, "INPUT_WINDOW_SHIFT"),
        parse_define_int(source, "INPUT_UNIQ_FEATURES_NUM"),
        parse_define_int(source, "MODEL_USES_AS_INPUT_INPUT_FEATURES"),
        parse_define_int(source, "MODEL_USES_AS_INPUT_DSP_FEATURES"),
    )
    return contract_hash_neuton(
        task=parse_define_int(source, "MODEL_TASK"),
        params_type=params_type_from_model(source),
        outputs_cap=parse_define_int(source, "MODEL_OUTPUTS_NUM"),
        inputs_num=parse_define_int(source, "INPUT_UNIQ_FEATURES_NUM"),
        neurons_cap=neurons_cap,
        solution_id=parse_define_token(source, "EDGEAI_LAB_SOLUTION_ID_STR"),
        pipeline_hash=pipeline,
    )


def config_define(path: Path, name: str) -> int | None:
    match = re.search(
        rf"^\s*#define\s+{re.escape(name)}\s+(\d+)[uUlL]*\s*$",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return int(match.group(1)) if match else None


def axon_contract_from_config(
    config_header: Path,
    *,
    compiled_model_size: int,
    interlayer_size: int,
    psum_size: int,
) -> int:
    persistent = config_define(config_header, "MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED") or 0
    packed = config_define(config_header, "MODEL_OTA_AXON_PACKED_OUTPUT_BYTES") or 0
    return contract_hash_axon(
        compiled_model_size=compiled_model_size,
        interlayer_size=interlayer_size,
        psum_size=psum_size,
        persistent_required=persistent,
        packed_output_bytes=packed,
    )


def symbol_name_hash(name: str) -> int:
    return fnv1a_str(FNV1A_INIT, name)
