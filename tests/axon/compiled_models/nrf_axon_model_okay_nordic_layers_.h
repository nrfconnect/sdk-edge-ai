/*********************************************************************************
 * Auto-generated nrf Axon compiled neural network model header file.
 * Model Name: okay_nordic
 * Axon Neural Network Compiler Version: 2.1.0
 *********************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
#include <stdalign.h>
#include "drivers/axon/nrf_axon_nn_infer.h"
#if (AXON_LAYER_TEST_START_LAYER<=6) && (AXON_LAYER_TEST_STOP_LAYER>=6)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_6[39] = {
// segment 0,length 37,Axon NN
0x1fff0025,0x00000028,
0x06000080,0x00010002,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x780),0x00330004,0x00000004,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00030028,
0x080000a0,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x78),0x00030008,0x00000008,0x00000200,0x00000000,
0x040000c8,0x00000000,0x00000000,0x00000000,0x00000003,0x00000000,
0x000000f0,0x00000100,
0x03000080,0x00010003,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00330003,0x00000003,
0x000000b0,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x7c),
0x000000bc,0x00000001,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_6[2];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_6[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x780),
    .dimensions = {
      .height = 1,
      .width = 2,
      .channel_cnt = 40,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 3,
    .stride = 4,
  }, /* 0 */
  {/* 1 */
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 3,
      .channel_cnt = 40,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = -1,
    .stride = 3,
  }, /* 1 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_6[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x7a),
    .dimensions = {
      .height = 1,
      .width = 5,
      .channel_cnt = 40,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 855309,
    .node_id = 6,
    .dequant_round = 23,
    .dequant_zp = -128,
    .stride = 8,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_6 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_6,
    .input_vector_list = okay_nordic_input_vector_list_6,
    .input_cnt = 2,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_6,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 39,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_6,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 6,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=7) && (AXON_LAYER_TEST_STOP_LAYER>=7)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_7[27] = {
// segment 0,length 25,Axon NN
0x1fff0019,0x0000001c,
0x0f000080,0x00010002,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x7d),0x00330008,0x00000008,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330028,0x00000000,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x00010002,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x780),0x00030004,0x00000004,0x00000003,
0x040000c8,0x00000000,0x01010000,0x00000000,0x00000003,0x00000001,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_7[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_7[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x7a),
    .dimensions = {
      .height = 1,
      .width = 5,
      .channel_cnt = 40,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 6,
    .stride = 8,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_7[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x780),
    .dimensions = {
      .height = 1,
      .width = 2,
      .channel_cnt = 40,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 855309,
    .node_id = 7,
    .dequant_round = 23,
    .dequant_zp = -128,
    .stride = 4,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_7 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_7,
    .input_vector_list = okay_nordic_input_vector_list_7,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_7,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 27,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_7,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 7,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=8) && (AXON_LAYER_TEST_STOP_LAYER>=8)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_8[133] = {
// segment 0,length 131,Axon NN
0x1fff0083,0x00000086,
0x10000080,0x00280005,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x7a),0x30030008,0x00040000,0x00280005,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_okay_nordic.l08_filters,0x02a30020,0x000000c8,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00150000,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x1b8),0x00050004,0x00000004,0x00000000,0x0000ff80,
0x030000c8,0x00010000,0x01030000,0x03010000,0x00011703,
0x01000180,0x00000003,0x00000000,
0x000001a4,0x00000000,
0x010001c8,0xc0000000,0xc0000000,
0x01000164,0x00017600,0x000007e0,
0x000000d8,0x01000001,
0x000000f0,0x00000100,
0x01000164,0xfffc1a80,0x00000900,
0x01000164,0xfffe7c00,0x00000881,
0x01000164,0x0002c780,0x00000999,
0x01000164,0x0002a200,0x00000797,
0x01000164,0x0001d080,0x00000974,
0x01000164,0x0002be00,0x000009ea,
0x01000164,0xffff5500,0x000008a9,
0x01000164,0x00045b00,0x000008b4,
0x01000164,0xfffcd600,0x00000a41,
0x01000164,0x00040800,0x000008e7,
0x01000164,0xfffe6f80,0x00000820,
0x01000164,0xfffe3200,0x0000086d,
0x01000164,0x000a6d80,0x00000737,
0x01000164,0xfffde580,0x00000977,
0x01000164,0xfffa8500,0x00000798,
0x01000164,0x00014800,0x000007d2,
0x01000164,0xfffea800,0x000009e7,
0x01000164,0x0001c900,0x00000828,
0x01000164,0x0000d300,0x00000941,
0x01000164,0xfffd0a00,0x00000696,
0x01000164,0x0004ec00,0x0000078e,
0x01000164,0xffff4f00,0x000008aa,
0x01000164,0xfffaee80,0x000009d3,
0x01000164,0x00011200,0x000009b6,
0x01000164,0x0001fa00,0x000007e9,
0x01000164,0xfffd0a00,0x000009ed,
0x01000164,0xfffd4b00,0x0000090d,
0x01000164,0xffff0e00,0x00000858,
0x01000164,0xfffeaf00,0x00000961,
0x01000164,0x00006a80,0x000008f2,
0x01000164,0xfffca580,0x00000baa,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_8[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_8[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x7a),
    .dimensions = {
      .height = 1,
      .width = 5,
      .channel_cnt = 40,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 6,
    .stride = 8,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_8[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x1b8),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 4798977,
    .node_id = 8,
    .dequant_round = 22,
    .dequant_zp = -128,
    .stride = 4,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_8 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_8,
    .input_vector_list = okay_nordic_input_vector_list_8,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_8,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 133,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_8,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 8,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=9) && (AXON_LAYER_TEST_STOP_LAYER>=9)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_9[36] = {
// segment 0,length 34,Axon NN
0x1fff0022,0x00000024,
0x06000080,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x700),0x00330004,0x00000004,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00030020,
0x070000a0,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00030008,0x00000008,0x00000000,
0x040000c8,0x00000000,0x00000000,0x00000000,0x00000003,0x00000000,
0x000000f0,0x00000100,
0x01000080,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x1b8),
0x000000b0,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x4),
0x000000bc,0x00000003,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_9[2];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_9[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x700),
    .dimensions = {
      .height = 1,
      .width = 4,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 2,
    .stride = 4,
  }, /* 0 */
  {/* 1 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x1b8),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 8,
    .stride = 4,
  }, /* 1 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_9[] = {
  {
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 5,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 4798977,
    .node_id = 9,
    .dequant_round = 22,
    .dequant_zp = -128,
    .stride = 8,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_9 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_9,
    .input_vector_list = okay_nordic_input_vector_list_9,
    .input_cnt = 2,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_9,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 36,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_9,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 9,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=10) && (AXON_LAYER_TEST_STOP_LAYER>=10)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_10[132] = {
// segment 0,length 130,Axon NN
0x1fff0082,0x00000084,
0x10000080,0x00010005,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x18030008,0x00000008,0x00010005,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_okay_nordic.l10_filters,0x00330020,0x00000005,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00150000,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x00050004,0x00000004,0x00000000,0x0000ff80,
0x030000c8,0x00010000,0x01010000,0x03010000,0x00011b03,
0x01000180,0x00000002,0x80000000,
0x000001a4,0x00000000,
0x000001cc,0x40000000,
0x01000164,0xffffd18b,0x00093d57,
0x000000d8,0x01000001,
0x000000f0,0x00000100,
0x01000164,0x000094f9,0x0006255d,
0x01000164,0x0000187c,0x0009078e,
0x01000164,0x00001781,0x000a9f7c,
0x01000164,0x00004075,0x0006a3c6,
0x01000164,0x000020fc,0x00061285,
0x01000164,0x000005ff,0x000ab948,
0x01000164,0xffff7cfb,0x00052cbc,
0x01000164,0xfffff4fe,0x0007c7c1,
0x01000164,0x000003ff,0x000b35ec,
0x01000164,0xfffff27f,0x00071f30,
0x01000164,0xffffb808,0x000991e1,
0x01000164,0x0000bd8e,0x00093175,
0x01000164,0x00000087,0x000784ad,
0x01000164,0x00008883,0x00095205,
0x01000164,0xffffdf07,0x00064684,
0x01000164,0xffffb8f9,0x0006f336,
0x01000164,0xfffff8fa,0x000a1d97,
0x01000164,0x000037f7,0x000a9bfa,
0x01000164,0x00003006,0x00079bd5,
0x01000164,0x0000a384,0x00060ce0,
0x01000164,0xffffbdf9,0x00082bc8,
0x01000164,0x0000758e,0x0007e0b3,
0x01000164,0x0000b505,0x0007e326,
0x01000164,0xffffc805,0x0007c852,
0x01000164,0x00007175,0x00067816,
0x01000164,0xffff7106,0x0007e88d,
0x01000164,0xffff6b7f,0x00086364,
0x01000164,0x0000647e,0x000a4ad7,
0x01000164,0xfffff403,0x00092867,
0x01000164,0xffff8473,0x0006585c,
0x01000164,0xffff5082,0x000ae651,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_10[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_10[] = {
  {/* 0 */
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 5,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 9,
    .stride = 8,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_10[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 2418451,
    .node_id = 10,
    .dequant_round = 22,
    .dequant_zp = 8,
    .stride = 4,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_10 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_10,
    .input_vector_list = okay_nordic_input_vector_list_10,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_10,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 132,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_10,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 10,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=11) && (AXON_LAYER_TEST_STOP_LAYER>=11)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_11[253] = {
// segment 0,length 251,Axon NN
0x1fff00fb,0x000000fe,
0x10000080,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x03030400,0x00000000,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330001,0x00000004,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00150000,0x00010020,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x180),0x00030020,0x00000020,0x00000000,0x00000000,
0x030000c8,0x00000000,0x01040000,0x00000000,0x00000003,
0x000000d8,0x00000001,
0x000000f0,0x00000100,
0x07000080,0x00010020,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x180),0x00330020,0x00040000,0x00010020,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_okay_nordic.l11_filters,0x00330040,0x00000020,
0x040000a8,0x00050000,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x1a0),0x00030004,0x00000004,
0x030000c8,0x00010000,0x01010000,0x03010000,0x00011703,
0x01000180,0x00000003,0x00000000,
0x000001a4,0x00000000,
0x010001c8,0xc0000000,0xc0000000,
0x01000164,0x000001ce,0x00012249,
0x000000d8,0x01000005,
0x000000f0,0x00000100,
0x01000164,0x000018af,0x00011980,
0x01000164,0x00000d2b,0x0000ebec,
0x01000164,0xfffff326,0x0000ddf9,
0x01000164,0xffffef70,0x00014aa9,
0x01000164,0x000010f5,0x000117e8,
0x01000164,0xffffefe1,0x0000fa71,
0x01000164,0x00000547,0x000106fb,
0x01000164,0x0000015c,0x000116d0,
0x01000164,0xfffffdc2,0x00010d0e,
0x01000164,0x0000074c,0x0000cc86,
0x01000164,0xfffffbc5,0x00010ce5,
0x01000164,0xffffffa5,0x000133e4,
0x01000164,0xfffffc52,0x000220ac,
0x01000164,0x00001999,0x0000c23c,
0x01000164,0xfffff996,0x0000feb6,
0x01000164,0xfffff687,0x00018752,
0x01000164,0x00000a86,0x000136bc,
0x01000164,0x00002486,0x0000893a,
0x01000164,0xfffffd16,0x0000be60,
0x01000164,0x00000288,0x0000e9b5,
0x01000164,0xffffed0b,0x0000a51d,
0x01000164,0xffffff45,0x00017407,
0x01000164,0xffffffa4,0x0000b521,
0x01000164,0x000001bd,0x00010a8e,
0x01000164,0x000006c8,0x0000de82,
0x01000164,0xfffffb5e,0x00011cf7,
0x01000164,0xfffff769,0x00012e73,
0x01000164,0xffffff3b,0x000139ec,
0x01000164,0x00000c49,0x00014823,
0x01000164,0x00000adc,0x00010139,
0x01000164,0x00000515,0x0000f1bd,
0x01000164,0xffffea10,0x0000d325,
0x01000164,0xfffff870,0x0000e94f,
0x01000164,0xfffff10d,0x0000fb2a,
0x01000164,0x00000891,0x0000e563,
0x01000164,0xffffefe6,0x0000c605,
0x01000164,0xfffff58d,0x0001bcde,
0x01000164,0x00000e73,0x0000e11e,
0x01000164,0x0000015d,0x000100ee,
0x01000164,0x000002a2,0x0000c4f8,
0x01000164,0xfffffa70,0x0001324f,
0x01000164,0xfffff677,0x0000bfcf,
0x01000164,0x00001cec,0x00008771,
0x01000164,0x000006a5,0x00012bfa,
0x01000164,0x00000185,0x0000bb34,
0x01000164,0x00000787,0x0000f409,
0x01000164,0x000007e4,0x0000fdce,
0x01000164,0xfffff9b9,0x00012300,
0x01000164,0x00000d86,0x00011468,
0x01000164,0x00000fba,0x0000a437,
0x01000164,0x00000edc,0x0000edd8,
0x01000164,0x000007cb,0x0001bd8b,
0x01000164,0x00000713,0x00008f45,
0x01000164,0x00000156,0x0000db13,
0x01000164,0x00000230,0x0001288b,
0x01000164,0x00000372,0x00008b28,
0x01000164,0xffffef44,0x0000fed1,
0x01000164,0x00001936,0x0000fa86,
0x01000164,0xfffff52a,0x0000c35b,
0x01000164,0x000003f7,0x0000c3bf,
0x01000164,0xfffff960,0x0000a204,
0x01000164,0x000000e2,0x0000c8d5,
0x01000164,0x00001b8f,0x0000ec76,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_11[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_11[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 10,
    .stride = 4,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_11[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x1a0),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 342231,
    .node_id = 11,
    .dequant_round = 23,
    .dequant_zp = -128,
    .stride = 4,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_11 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_11,
    .input_vector_list = okay_nordic_input_vector_list_11,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_11,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 253,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_11,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 11,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=12) && (AXON_LAYER_TEST_STOP_LAYER>=12)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_12[27] = {
// segment 0,length 25,Axon NN
0x1fff0019,0x0000001c,
0x0f000080,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x1),0x00330008,0x00000008,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330020,0x00000000,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x700),0x00030004,0x00000004,0x00000003,
0x040000c8,0x00000000,0x01010000,0x00000000,0x00000003,0x00000001,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_12[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_12[] = {
  {/* 0 */
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 5,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 9,
    .stride = 8,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_12[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x700),
    .dimensions = {
      .height = 1,
      .width = 4,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 4798977,
    .node_id = 12,
    .dequant_round = 22,
    .dequant_zp = -128,
    .stride = 4,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_12 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_12,
    .input_vector_list = okay_nordic_input_vector_list_12,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_12,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 27,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_12,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 12,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=13) && (AXON_LAYER_TEST_STOP_LAYER>=13)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_13[40] = {
// segment 0,length 38,Axon NN
0x1fff0026,0x00000028,
0x06000080,0x0001000a,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x400),0x0033000c,0x0000000c,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00030040,
0x080000a0,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x0001000c,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x2a0),0x00030010,0x00000010,0x00000200,0x00000000,
0x040000c8,0x00000000,0x00000000,0x00000000,0x00000003,0x00000000,
0x000000f0,0x00000100,
0x03000080,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x1a0),0x00330004,0x00000004,
0x010000ac,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x2ac),
0x000000bc,0x00000003,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_13[2];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_13[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x400),
    .dimensions = {
      .height = 1,
      .width = 10,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 1,
    .stride = 12,
  }, /* 0 */
  {/* 1 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x1a0),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 11,
    .stride = 4,
  }, /* 1 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_13[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x2a2),
    .dimensions = {
      .height = 1,
      .width = 11,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 342231,
    .node_id = 13,
    .dequant_round = 23,
    .dequant_zp = -128,
    .stride = 16,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_13 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_13,
    .input_vector_list = okay_nordic_input_vector_list_13,
    .input_cnt = 2,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_13,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 40,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_13,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 13,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=14) && (AXON_LAYER_TEST_STOP_LAYER>=14)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_14[27] = {
// segment 0,length 25,Axon NN
0x1fff0019,0x0000001c,
0x0f000080,0x0001000b,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x2a2),0x00330010,0x00000010,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330020,0x00000000,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x0001000b,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x0003000c,0x0000000c,0x00000003,
0x040000c8,0x00000000,0x01010000,0x00000000,0x00000003,0x00000001,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_14[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_14[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x2a2),
    .dimensions = {
      .height = 1,
      .width = 11,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 13,
    .stride = 16,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_14[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
    .dimensions = {
      .height = 1,
      .width = 11,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 342231,
    .node_id = 14,
    .dequant_round = 23,
    .dequant_zp = -128,
    .stride = 12,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_14 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_14,
    .input_vector_list = okay_nordic_input_vector_list_14,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_14,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 27,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_14,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 14,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=15) && (AXON_LAYER_TEST_STOP_LAYER>=15)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_15[27] = {
// segment 0,length 25,Axon NN
0x1fff0019,0x0000001c,
0x0f000080,0x0001000b,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x4a2),0x00330010,0x00000010,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330020,0x00000000,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x0001000b,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6a0),0x0003000c,0x0000000c,0x00000003,
0x040000c8,0x00000000,0x01010000,0x00000000,0x00000003,0x00000001,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_15[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_15[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x2a2),
    .dimensions = {
      .height = 1,
      .width = 11,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 13,
    .stride = 16,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_15[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6a0),
    .dimensions = {
      .height = 1,
      .width = 11,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 342231,
    .node_id = 15,
    .dequant_round = 23,
    .dequant_zp = -128,
    .stride = 12,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_15 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_15,
    .input_vector_list = okay_nordic_input_vector_list_15,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_15,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 27,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_15,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 15,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=16) && (AXON_LAYER_TEST_STOP_LAYER>=16)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_16[27] = {
// segment 0,length 25,Axon NN
0x1fff0019,0x0000001c,
0x0f000080,0x0001000a,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x2a3),0x00330010,0x00000010,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330040,0x00000000,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x0001000a,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x400),0x0003000c,0x0000000c,0x00000003,
0x040000c8,0x00000000,0x01010000,0x00000000,0x00000003,0x00000001,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_16[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_16[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x2a2),
    .dimensions = {
      .height = 1,
      .width = 11,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 13,
    .stride = 16,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_16[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x400),
    .dimensions = {
      .height = 1,
      .width = 10,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 342231,
    .node_id = 16,
    .dequant_round = 23,
    .dequant_zp = -128,
    .stride = 12,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_16 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_16,
    .input_vector_list = okay_nordic_input_vector_list_16,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_16,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 27,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_16,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 16,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=17) && (AXON_LAYER_TEST_STOP_LAYER>=17)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_17[27] = {
// segment 0,length 25,Axon NN
0x1fff0019,0x0000001c,
0x0f000080,0x00010007,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x104),0x0033000c,0x0000000c,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330020,0x00000000,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x00010007,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x820),0x00030008,0x00000008,0x00000003,
0x040000c8,0x00000000,0x01010000,0x00000000,0x00000003,0x00000001,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_17[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_17[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
    .dimensions = {
      .height = 1,
      .width = 11,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 14,
    .stride = 12,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_17[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x820),
    .dimensions = {
      .height = 1,
      .width = 7,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 342231,
    .node_id = 17,
    .dequant_round = 23,
    .dequant_zp = -128,
    .stride = 8,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_17 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_17,
    .input_vector_list = okay_nordic_input_vector_list_17,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_17,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 27,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_17,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 17,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=18) && (AXON_LAYER_TEST_STOP_LAYER>=18)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_18[132] = {
// segment 0,length 130,Axon NN
0x1fff0082,0x00000084,
0x10000080,0x00010007,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x820),0x18030008,0x00000008,0x00010007,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_okay_nordic.l18_filters,0x00330020,0x00000007,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00150000,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x00050004,0x00000004,0x00000000,0x0000ff80,
0x030000c8,0x00010000,0x01010000,0x03010000,0x00011903,
0x01000180,0x00000002,0x80000000,
0x000001a4,0x00000000,
0x000001cc,0xb8000000,
0x01000164,0x00002405,0x000102cd,
0x000000d8,0x01000001,
0x000000f0,0x00000100,
0x01000164,0x0000ad57,0x0000e886,
0x01000164,0xffff25c7,0x0000de19,
0x01000164,0x00008615,0x0000fea3,
0x01000164,0x0000e1cc,0x0000d126,
0x01000164,0x00006248,0x0000eae6,
0x01000164,0x00006e25,0x00012565,
0x01000164,0x0000077d,0x0000fd09,
0x01000164,0xffffcc2e,0x0000e2c9,
0x01000164,0x0000929c,0x0000d1ac,
0x01000164,0x00004085,0x00011819,
0x01000164,0xfffffd3e,0x0000b204,
0x01000164,0x000022f8,0x0000d4c3,
0x01000164,0x00003718,0x00011d85,
0x01000164,0x0000b404,0x0000bdba,
0x01000164,0xffffb344,0x0000ce3e,
0x01000164,0x000058c6,0x0000dad3,
0x01000164,0xffff589c,0x0001140a,
0x01000164,0x00003715,0x0000ea87,
0x01000164,0xffffd7dd,0x000109ac,
0x01000164,0x0000348a,0x0000e420,
0x01000164,0x000055eb,0x000106ff,
0x01000164,0x0000b723,0x0000f1ab,
0x01000164,0xffffac0f,0x000125fe,
0x01000164,0xffffb069,0x00010c53,
0x01000164,0x0000cf39,0x0000c545,
0x01000164,0x000055ac,0x0000cacc,
0x01000164,0x0000153d,0x0000a049,
0x01000164,0xfffffff9,0x0001269f,
0x01000164,0xffff4f84,0x0000f185,
0x01000164,0xffff501e,0x000124e2,
0x01000164,0x00011322,0x0000b747,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_18[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_18[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x820),
    .dimensions = {
      .height = 1,
      .width = 7,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 17,
    .stride = 8,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_18[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 775467,
    .node_id = 18,
    .dequant_round = 24,
    .dequant_zp = -36,
    .stride = 4,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_18 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_18,
    .input_vector_list = okay_nordic_input_vector_list_18,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_18,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 132,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_18,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 18,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=19) && (AXON_LAYER_TEST_STOP_LAYER>=19)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_19[132] = {
// segment 0,length 130,Axon NN
0x1fff0082,0x00000084,
0x10000080,0x0001000b,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6a0),0x1803000c,0x0000000c,0x0001000b,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_okay_nordic.l19_filters,0x00330020,0x0000000b,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00150000,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x180),0x00050004,0x00000004,0x00000000,0x0000ff80,
0x030000c8,0x00010000,0x01010000,0x03010000,0x00011903,
0x01000180,0x00000002,0x80000000,
0x000001a4,0x00000000,
0x000001cc,0xb8000000,
0x01000164,0x0000a9d6,0x00009820,
0x000000d8,0x01000001,
0x000000f0,0x00000100,
0x01000164,0xffff54e6,0x0000ba79,
0x01000164,0xffffa610,0x0000c172,
0x01000164,0xffff60e6,0x0000d83a,
0x01000164,0xffff629a,0x0000a2c0,
0x01000164,0xfffff18b,0x0001257a,
0x01000164,0xfffff089,0x0000f75c,
0x01000164,0x00005497,0x0000b6e6,
0x01000164,0x000001e2,0x0001192c,
0x01000164,0xffff8b28,0x0000f0a9,
0x01000164,0x0000213c,0x0000ae9e,
0x01000164,0x00004072,0x0000b1a9,
0x01000164,0x0000045b,0x0000cbaa,
0x01000164,0x00010c09,0x0000d441,
0x01000164,0xffff72a9,0x0000e3ff,
0x01000164,0xffff48a2,0x0000e231,
0x01000164,0x00005feb,0x0000c5ac,
0x01000164,0x000024c4,0x0000cc36,
0x01000164,0x00000fa5,0x0000d01c,
0x01000164,0x0000b0c9,0x0000fdd5,
0x01000164,0x0000acca,0x000125e8,
0x01000164,0xffff5817,0x0000c22c,
0x01000164,0xffff6bde,0x0000bd46,
0x01000164,0x0000f1e6,0x0000cb1b,
0x01000164,0xffff9a30,0x000085c9,
0x01000164,0x000000dd,0x00010e1d,
0x01000164,0x0000138d,0x0000d646,
0x01000164,0xffff17ef,0x00012383,
0x01000164,0xffffbb82,0x0000c995,
0x01000164,0xffff1781,0x00009da0,
0x01000164,0x0000671e,0x0000c0fe,
0x01000164,0xffffaab7,0x0001394c,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_19[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_19[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6a0),
    .dimensions = {
      .height = 1,
      .width = 11,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 15,
    .stride = 12,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_19[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x180),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 775467,
    .node_id = 19,
    .dequant_round = 24,
    .dequant_zp = -36,
    .stride = 4,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_19 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_19,
    .input_vector_list = okay_nordic_input_vector_list_19,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_19,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 132,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_19,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 19,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=20) && (AXON_LAYER_TEST_STOP_LAYER>=20)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_20[34] = {
// segment 0,length 32,Axon NN
0x1fff0020,0x00000022,
0x06000080,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x00330004,0x00000004,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00030020,
0x080000a0,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6a0),0x00030004,0x00000004,0x00000003,0x00000000,
0x040000c8,0x00000000,0x00000000,0x00000000,0x00000003,0x00000000,
0x000000f0,0x00000100,
0x00000084,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x180),
0x000000b0,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x720),
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_20[2];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_20[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 18,
    .stride = 4,
  }, /* 0 */
  {/* 1 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x180),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 19,
    .stride = 4,
  }, /* 1 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_20[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6a0),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 775467,
    .node_id = 20,
    .dequant_round = 24,
    .dequant_zp = -36,
    .stride = 4,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_20 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_20,
    .input_vector_list = okay_nordic_input_vector_list_20,
    .input_cnt = 2,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_20,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 34,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_20,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 20,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=21) && (AXON_LAYER_TEST_STOP_LAYER>=21)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_21[253] = {
// segment 0,length 251,Axon NN
0x1fff00fb,0x000000fe,
0x10000080,0x00010100,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6a0),0x01830400,0x00000000,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330001,0x00000004,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00150000,0x00010040,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x00030040,0x00000040,0x00000000,0x00000000,
0x030000c8,0x00000000,0x01040000,0x00000000,0x00000003,
0x000000d8,0x00000001,
0x000000f0,0x00000100,
0x07000080,0x00010040,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x00330040,0x00040000,0x00010040,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_okay_nordic.l21_filters,0x00330040,0x00000040,
0x040000a8,0x00050000,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x140),0x00030004,0x00000004,
0x030000c8,0x00010000,0x01010000,0x03010000,0x00011703,
0x01000180,0x00000003,0x00000000,
0x000001a4,0x00000000,
0x010001c8,0xc0000000,0xc0000000,
0x01000164,0x000042e7,0x000153b1,
0x000000d8,0x01000005,
0x000000f0,0x00000100,
0x01000164,0x00002bbb,0x0001437f,
0x01000164,0xffffd169,0x00011f9a,
0x01000164,0x000000b4,0x0001266b,
0x01000164,0xfffffd36,0x00010547,
0x01000164,0xfffffae3,0x00010726,
0x01000164,0x000042f2,0x0000dfd5,
0x01000164,0x0000083b,0x00011aa8,
0x01000164,0x00000862,0x0001195e,
0x01000164,0xfffff4e0,0x00010775,
0x01000164,0x000035f8,0x00015741,
0x01000164,0xfffffb35,0x00013a97,
0x01000164,0xffffaf53,0x0000f194,
0x01000164,0xffffc135,0x0000eeab,
0x01000164,0x00005a7b,0x0000dc42,
0x01000164,0xffffa9a2,0x0000c302,
0x01000164,0x00002865,0x00012dd1,
0x01000164,0xffff9f14,0x0000cf2e,
0x01000164,0xffffdda9,0x000123b1,
0x01000164,0x00004993,0x0001399e,
0x01000164,0xffffd2db,0x00011a2b,
0x01000164,0x0000195e,0x0001005c,
0x01000164,0xffffeb0f,0x0001264c,
0x01000164,0x000047f0,0x0001394c,
0x01000164,0xffffe4fa,0x000113ab,
0x01000164,0x000001b2,0x000118c0,
0x01000164,0xfffff9a7,0x0001097f,
0x01000164,0x00002881,0x0000d216,
0x01000164,0x00001f14,0x00011235,
0x01000164,0xffffcef1,0x0001048e,
0x01000164,0x00001b0a,0x0000e926,
0x01000164,0xfffff933,0x0000eadc,
0x01000164,0x000000ec,0x00015d9d,
0x01000164,0x00002a68,0x0000c107,
0x01000164,0x00000050,0x0000c0f2,
0x01000164,0xffffcf81,0x0000e196,
0x01000164,0xffffdc8e,0x000140a7,
0x01000164,0xffff8b2a,0x0000fb48,
0x01000164,0xffffd946,0x00010a8c,
0x01000164,0xffffe903,0x0000dcba,
0x01000164,0xfffff5d7,0x0000ede5,
0x01000164,0xffff993c,0x0000c948,
0x01000164,0xffffc97a,0x0000dad8,
0x01000164,0xffffe017,0x00011e05,
0x01000164,0xffffced2,0x00010384,
0x01000164,0xfffff7fb,0x00014b64,
0x01000164,0xffffe0ac,0x0001596a,
0x01000164,0x0000058d,0x0000ff35,
0x01000164,0x00003b70,0x0000d9d2,
0x01000164,0x0000246c,0x0000e1c5,
0x01000164,0x0000008c,0x0000e0c2,
0x01000164,0xffffe39c,0x0000fc0c,
0x01000164,0xffffd85c,0x0000f9a3,
0x01000164,0xffffcc99,0x0000e9fd,
0x01000164,0x00004f5b,0x0000fe05,
0x01000164,0x00004bb3,0x0000e424,
0x01000164,0xfffff852,0x000140c3,
0x01000164,0xffffdd69,0x0000e37e,
0x01000164,0x00004e35,0x0000ec9f,
0x01000164,0x00000403,0x0000d3d2,
0x01000164,0xffff84c2,0x0000c3c2,
0x01000164,0xfffff162,0x00014f1a,
0x01000164,0x00001a5d,0x00017f9b,
0x01000164,0x00005161,0x0000e775,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_21[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_21[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6a0),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 20,
    .stride = 4,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_21[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x140),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 159885,
    .node_id = 21,
    .dequant_round = 22,
    .dequant_zp = -128,
    .stride = 4,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_21 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_21,
    .input_vector_list = okay_nordic_input_vector_list_21,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_21,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 253,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_21,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 21,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=22) && (AXON_LAYER_TEST_STOP_LAYER>=22)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_22[40] = {
// segment 0,length 38,Axon NN
0x1fff0026,0x00000028,
0x06000080,0x0001000e,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_okay_nordic_persistent_vars,0x00330010,0x00000010,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00030040,
0x080000a0,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x00010010,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6a0),0x00030014,0x00000014,0x00000200,0x00000000,
0x040000c8,0x00000000,0x00000000,0x00000000,0x00000003,0x00000000,
0x000000f0,0x00000100,
0x03000080,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x140),0x00330004,0x00000004,
0x010000ac,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6b0),
0x000000bc,0x00000003,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_22[2];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_22[] = {
  {/* 0 */
    .ptr = (int8_t*)axon_model_okay_nordic_persistent_vars,
    .dimensions = {
      .height = 1,
      .width = 14,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 0,
    .stride = 16,
  }, /* 0 */
  {/* 1 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x140),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 21,
    .stride = 4,
  }, /* 1 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_22[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6a2),
    .dimensions = {
      .height = 1,
      .width = 15,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 159885,
    .node_id = 22,
    .dequant_round = 22,
    .dequant_zp = -128,
    .stride = 20,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_22 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_22,
    .input_vector_list = okay_nordic_input_vector_list_22,
    .input_cnt = 2,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_22,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 40,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_22,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 22,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=23) && (AXON_LAYER_TEST_STOP_LAYER>=23)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_23[27] = {
// segment 0,length 25,Axon NN
0x1fff0019,0x0000001c,
0x0f000080,0x0001000f,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6a2),0x00330014,0x00000014,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330020,0x00000000,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x0001000f,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0xba0),0x00030010,0x00000010,0x00000003,
0x040000c8,0x00000000,0x01010000,0x00000000,0x00000003,0x00000001,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_23[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_23[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6a2),
    .dimensions = {
      .height = 1,
      .width = 15,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 22,
    .stride = 20,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_23[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0xba0),
    .dimensions = {
      .height = 1,
      .width = 15,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 159885,
    .node_id = 23,
    .dequant_round = 22,
    .dequant_zp = -128,
    .stride = 16,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_23 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_23,
    .input_vector_list = okay_nordic_input_vector_list_23,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_23,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 27,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_23,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 23,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=24) && (AXON_LAYER_TEST_STOP_LAYER>=24)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_24[27] = {
// segment 0,length 25,Axon NN
0x1fff0019,0x0000001c,
0x0f000080,0x0001000f,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x922),0x00330014,0x00000014,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330020,0x00000000,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x0001000f,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0xda0),0x00030010,0x00000010,0x00000003,
0x040000c8,0x00000000,0x01010000,0x00000000,0x00000003,0x00000001,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_24[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_24[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6a2),
    .dimensions = {
      .height = 1,
      .width = 15,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 22,
    .stride = 20,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_24[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0xda0),
    .dimensions = {
      .height = 1,
      .width = 15,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 159885,
    .node_id = 24,
    .dequant_round = 22,
    .dequant_zp = -128,
    .stride = 16,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_24 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_24,
    .input_vector_list = okay_nordic_input_vector_list_24,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_24,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 27,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_24,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 24,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=25) && (AXON_LAYER_TEST_STOP_LAYER>=25)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_25[27] = {
// segment 0,length 25,Axon NN
0x1fff0019,0x0000001c,
0x0f000080,0x0001000e,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6a3),0x00330014,0x00000014,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330040,0x00000000,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x0001000e,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_okay_nordic_persistent_vars,0x00030010,0x00000010,0x00000003,
0x040000c8,0x00000000,0x01010000,0x00000000,0x00000003,0x00000001,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_25[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_25[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x6a2),
    .dimensions = {
      .height = 1,
      .width = 15,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 22,
    .stride = 20,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_25[] = {
  {
    .ptr = (int8_t*)axon_model_okay_nordic_persistent_vars,
    .dimensions = {
      .height = 1,
      .width = 14,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 159885,
    .node_id = 25,
    .dequant_round = 22,
    .dequant_zp = -128,
    .stride = 16,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_25 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_25,
    .input_vector_list = okay_nordic_input_vector_list_25,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_25,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 27,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_25,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 25,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=26) && (AXON_LAYER_TEST_STOP_LAYER>=26)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_26[27] = {
// segment 0,length 25,Axon NN
0x1fff0019,0x0000001c,
0x0f000080,0x00010009,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0xba6),0x00330010,0x00000010,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330020,0x00000000,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x00010009,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x0003000c,0x0000000c,0x00000003,
0x040000c8,0x00000000,0x01010000,0x00000000,0x00000003,0x00000001,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_26[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_26[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0xba0),
    .dimensions = {
      .height = 1,
      .width = 15,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 23,
    .stride = 16,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_26[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
    .dimensions = {
      .height = 1,
      .width = 9,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 159885,
    .node_id = 26,
    .dequant_round = 22,
    .dequant_zp = -128,
    .stride = 12,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_26 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_26,
    .input_vector_list = okay_nordic_input_vector_list_26,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_26,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 27,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_26,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 26,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=27) && (AXON_LAYER_TEST_STOP_LAYER>=27)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_27[132] = {
// segment 0,length 130,Axon NN
0x1fff0082,0x00000084,
0x10000080,0x00010009,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x1803000c,0x0000000c,0x00010009,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_okay_nordic.l27_filters,0x00330020,0x00000009,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00150000,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x00050004,0x00000004,0x00000000,0x0000ff80,
0x030000c8,0x00010000,0x01010000,0x03010000,0x00011a03,
0x01000180,0x00000002,0x80000000,
0x000001a4,0x00000000,
0x000001cc,0xac000000,
0x01000164,0x00002108,0x00025093,
0x000000d8,0x01000001,
0x000000f0,0x00000100,
0x01000164,0xffffd9d8,0x0001dcf8,
0x01000164,0x0000894b,0x00017f85,
0x01000164,0x00001a1c,0x00020e75,
0x01000164,0x0001469c,0x00016533,
0x01000164,0x00007b1d,0x00016991,
0x01000164,0xfffeef24,0x000185fb,
0x01000164,0x0000f916,0x00020f5a,
0x01000164,0xffffa296,0x000269a6,
0x01000164,0x0000de19,0x0001ea47,
0x01000164,0xffff91f4,0x0002e591,
0x01000164,0x00015727,0x00020d99,
0x01000164,0x00014267,0x00018131,
0x01000164,0xfffffec1,0x00018bdd,
0x01000164,0x0000417e,0x00024696,
0x01000164,0x00007d10,0x0001e4e3,
0x01000164,0xffffc70d,0x00016aaf,
0x01000164,0x0000d9d3,0x0001e489,
0x01000164,0xffffa0d3,0x0001fd58,
0x01000164,0xffff93a8,0x000262d0,
0x01000164,0x000040f3,0x000198ce,
0x01000164,0xffffb68e,0x0001c388,
0x01000164,0x0000c497,0x00021e3a,
0x01000164,0xffff9581,0x00015d8e,
0x01000164,0x000001cf,0x00021904,
0x01000164,0xffffeee6,0x0001ae63,
0x01000164,0xffff75f1,0x00027883,
0x01000164,0x00007b4e,0x000122d7,
0x01000164,0xfffeea0f,0x00017333,
0x01000164,0xffffadac,0x0001cde5,
0x01000164,0x00010374,0x000266cc,
0x01000164,0x00002c89,0x00017611,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_27[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_27[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
    .dimensions = {
      .height = 1,
      .width = 9,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 26,
    .stride = 12,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_27[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 1481057,
    .node_id = 27,
    .dequant_round = 25,
    .dequant_zp = -21,
    .stride = 4,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_27 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_27,
    .input_vector_list = okay_nordic_input_vector_list_27,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_27,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 132,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_27,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 27,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=28) && (AXON_LAYER_TEST_STOP_LAYER>=28)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_28[132] = {
// segment 0,length 130,Axon NN
0x1fff0082,0x00000084,
0x10000080,0x0001000f,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0xda0),0x18030010,0x00000010,0x0001000f,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_okay_nordic.l28_filters,0x00330020,0x0000000f,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00150000,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x180),0x00050004,0x00000004,0x00000000,0x0000ff80,
0x030000c8,0x00010000,0x01010000,0x03010000,0x00011a03,
0x01000180,0x00000002,0x80000000,
0x000001a4,0x00000000,
0x000001cc,0xac000000,
0x01000164,0x00009d3f,0x0002fcb0,
0x000000d8,0x01000001,
0x000000f0,0x00000100,
0x01000164,0xffff9859,0x00013d25,
0x01000164,0xffffeb21,0x00016658,
0x01000164,0x0000b725,0x000152d8,
0x01000164,0xffffafeb,0x0002182a,
0x01000164,0xffffec88,0x00017daf,
0x01000164,0xfffe4b44,0x00016b2c,
0x01000164,0x00011336,0x0000ec79,
0x01000164,0x000006af,0x0001f82e,
0x01000164,0xffffb9bf,0x00017df5,
0x01000164,0x000012a7,0x0001726f,
0x01000164,0x00002bd7,0x00019a3c,
0x01000164,0xfffed7dc,0x0001a8e3,
0x01000164,0xffff4670,0x0001f670,
0x01000164,0xffff7b05,0x0001de2d,
0x01000164,0xfffebcb9,0x0001c60c,
0x01000164,0xffffbd95,0x0001de53,
0x01000164,0x0000770e,0x0001959f,
0x01000164,0x0000067a,0x00016f9c,
0x01000164,0x00003e91,0x00019c3a,
0x01000164,0xffff9f45,0x0001e1e7,
0x01000164,0xffff3ac8,0x0001e5fc,
0x01000164,0x00010954,0x0001f43a,
0x01000164,0xffff543a,0x00010dbf,
0x01000164,0x00001c8a,0x0002574b,
0x01000164,0xfffead16,0x00016dea,
0x01000164,0xffff60cb,0x0001863b,
0x01000164,0x00014a67,0x00016e1d,
0x01000164,0x00008b1f,0x0001b044,
0x01000164,0xffffc9cb,0x000207fe,
0x01000164,0x00003c38,0x00021d88,
0x01000164,0xffff991b,0x00017bee,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_28[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_28[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0xda0),
    .dimensions = {
      .height = 1,
      .width = 15,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 24,
    .stride = 16,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_28[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x180),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 1481057,
    .node_id = 28,
    .dequant_round = 25,
    .dequant_zp = -21,
    .stride = 4,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_28 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_28,
    .input_vector_list = okay_nordic_input_vector_list_28,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_28,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 132,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_28,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 28,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=29) && (AXON_LAYER_TEST_STOP_LAYER>=29)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_29[34] = {
// segment 0,length 32,Axon NN
0x1fff0020,0x00000022,
0x06000080,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x00330004,0x00000004,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00030020,
0x080000a0,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0xba0),0x00030004,0x00000004,0x00000003,0x00000000,
0x040000c8,0x00000000,0x00000000,0x00000000,0x00000003,0x00000000,
0x000000f0,0x00000100,
0x00000084,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x180),
0x000000b0,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0xc20),
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_29[2];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_29[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 27,
    .stride = 4,
  }, /* 0 */
  {/* 1 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x180),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 32,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 28,
    .stride = 4,
  }, /* 1 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_29[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0xba0),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 1481057,
    .node_id = 29,
    .dequant_round = 25,
    .dequant_zp = -21,
    .stride = 4,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_29 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_29,
    .input_vector_list = okay_nordic_input_vector_list_29,
    .input_cnt = 2,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_29,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 34,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_29,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 29,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=30) && (AXON_LAYER_TEST_STOP_LAYER>=30)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_30[253] = {
// segment 0,length 251,Axon NN
0x1fff00fb,0x000000fe,
0x10000080,0x00010100,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0xba0),0x01830400,0x00000000,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330001,0x00000004,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00150000,0x00010040,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x00030040,0x00000040,0x00000000,0x00000000,
0x030000c8,0x00000000,0x01040000,0x00000000,0x00000003,
0x000000d8,0x00000001,
0x000000f0,0x00000100,
0x07000080,0x00010040,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x00330040,0x00040000,0x00010040,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_okay_nordic.l30_filters,0x00330040,0x00000040,
0x040000a8,0x00050000,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x140),0x00030004,0x00000004,
0x030000c8,0x00010000,0x01010000,0x03010000,0x00011703,
0x01000180,0x00000003,0x00000000,
0x000001a4,0x00000000,
0x010001c8,0xc0000000,0xc0000000,
0x01000164,0xffffd8b9,0x0000d929,
0x000000d8,0x01000005,
0x000000f0,0x00000100,
0x01000164,0xfffff3bc,0x0000f13b,
0x01000164,0x00000e8e,0x0000f37d,
0x01000164,0x000024b1,0x0000bc84,
0x01000164,0xfffface1,0x0000c62e,
0x01000164,0xffffc897,0x000115d2,
0x01000164,0x000012d8,0x0001017a,
0x01000164,0xffffe77f,0x0001186e,
0x01000164,0x00002aef,0x0000b546,
0x01000164,0x00003112,0x0000ef30,
0x01000164,0x0000216b,0x0000d818,
0x01000164,0xffffee6a,0x0000a880,
0x01000164,0xffffb23a,0x0000cf0f,
0x01000164,0xffffe703,0x0000fca3,
0x01000164,0x0000225f,0x0000be3c,
0x01000164,0x000004e6,0x0000d23c,
0x01000164,0x000014eb,0x00012a77,
0x01000164,0x00000774,0x0000d15e,
0x01000164,0x00001439,0x0000c987,
0x01000164,0xffffdfc1,0x00010e3d,
0x01000164,0x0000155c,0x0000eeec,
0x01000164,0x00000da1,0x0000da25,
0x01000164,0x000029de,0x00010699,
0x01000164,0xffffeb96,0x0000b7e7,
0x01000164,0xffffea4a,0x0000b538,
0x01000164,0x000017f3,0x00010a6c,
0x01000164,0x00000c0a,0x0000ef9a,
0x01000164,0xfffff31a,0x000187c2,
0x01000164,0x00000a9b,0x0000d211,
0x01000164,0xfffffe7a,0x0000de73,
0x01000164,0x0000145b,0x0000c493,
0x01000164,0x000030f2,0x0000da62,
0x01000164,0xffffe2c3,0x0000dee7,
0x01000164,0xfffff2e6,0x00015430,
0x01000164,0x000035e1,0x000101d2,
0x01000164,0xffffe8cf,0x0000e8ae,
0x01000164,0x00005b66,0x0000d88e,
0x01000164,0xffffe560,0x00015fc6,
0x01000164,0xffffcfc1,0x00011689,
0x01000164,0x000027c1,0x0000e6ee,
0x01000164,0x0000157f,0x00010370,
0x01000164,0x0000035b,0x000115b2,
0x01000164,0x0000306c,0x0000c3b6,
0x01000164,0xfffff872,0x00010dfe,
0x01000164,0xffffcc45,0x0000d222,
0x01000164,0x000033f3,0x0000e3a1,
0x01000164,0xffff95aa,0x0000e1d1,
0x01000164,0x00000441,0x0000f374,
0x01000164,0xffffdeb5,0x0000c6f1,
0x01000164,0x00000e51,0x0000e815,
0x01000164,0xfffff81a,0x00010d2a,
0x01000164,0x00002626,0x0000b095,
0x01000164,0x00000f5a,0x0000e53e,
0x01000164,0xffffda53,0x0000e607,
0x01000164,0xffffd3c8,0x00012f3f,
0x01000164,0x00002c22,0x0000f7b1,
0x01000164,0x00002b1a,0x0000d91e,
0x01000164,0x000001c0,0x0000b2ce,
0x01000164,0x00003a56,0x0000f746,
0x01000164,0xffffcbc2,0x0000e943,
0x01000164,0xffffc42d,0x0000e174,
0x01000164,0xffffa2bc,0x0000d2ba,
0x01000164,0xffffeb72,0x0000b50f,
0x01000164,0x00000875,0x00010757,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_30[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_30[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0xba0),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 29,
    .stride = 4,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_30[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x140),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 283731,
    .node_id = 30,
    .dequant_round = 23,
    .dequant_zp = -128,
    .stride = 4,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_30 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_30,
    .input_vector_list = okay_nordic_input_vector_list_30,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_30,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 253,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_30,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 30,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=31) && (AXON_LAYER_TEST_STOP_LAYER>=31)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_31[40] = {
// segment 0,length 38,Axon NN
0x1fff0026,0x00000028,
0x06000080,0x00010016,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0xc20),0x00330018,0x00000018,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00030040,
0x080000a0,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x00010018,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0xba0),0x0003001c,0x0000001c,0x00000200,0x00000000,
0x040000c8,0x00000000,0x00000000,0x00000000,0x00000003,0x00000000,
0x000000f0,0x00000100,
0x03000080,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x140),0x00330004,0x00000004,
0x010000ac,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0xbb8),
0x000000bc,0x00000003,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_31[2];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_31[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0xc20),
    .dimensions = {
      .height = 1,
      .width = 22,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 5,
    .stride = 24,
  }, /* 0 */
  {/* 1 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x140),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 30,
    .stride = 4,
  }, /* 1 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_31[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0xba2),
    .dimensions = {
      .height = 1,
      .width = 23,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 283731,
    .node_id = 31,
    .dequant_round = 23,
    .dequant_zp = -128,
    .stride = 28,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_31 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_31,
    .input_vector_list = okay_nordic_input_vector_list_31,
    .input_cnt = 2,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_31,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 40,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_31,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 31,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=32) && (AXON_LAYER_TEST_STOP_LAYER>=32)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_32[228] = {
// segment 0,length 226,Axon NN
0x1fff00e2,0x000000e4,
0x10000080,0x00010017,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0xba2),0x0ff3001c,0x0000001c,0x00010017,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_okay_nordic.l32_filters,0x00330040,0x00000017,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00150000,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x00050004,0x00000004,0x00000000,0x0000ff80,
0x030000c8,0x00010000,0x01010000,0x03010000,0x00011a03,
0x01000180,0x00000002,0x80000000,
0x000001a4,0x00000000,
0x000001cc,0x54000000,
0x01000164,0x0000ac6f,0x0000840e,
0x000000d8,0x01000001,
0x000000f0,0x00000100,
0x01000164,0xfffed947,0x0000f1f6,
0x01000164,0x00026da5,0x0001071c,
0x01000164,0xfffea88a,0x0001a473,
0x01000164,0xfffe4f86,0x000109c9,
0x01000164,0x0001b875,0x00010a27,
0x01000164,0x0001a5df,0x0000b610,
0x01000164,0xffff7e08,0x0000fce7,
0x01000164,0x000123a2,0x0000eab2,
0x01000164,0x0000d3fa,0x0000f763,
0x01000164,0xfffe86da,0x00010c7a,
0x01000164,0xfffe3f28,0x0000f1ca,
0x01000164,0xfffdb9fd,0x0000df9d,
0x01000164,0xffff138f,0x0000f312,
0x01000164,0x0000c534,0x00011848,
0x01000164,0xfffe13d4,0x00009e10,
0x01000164,0x00019ca0,0x0000dd80,
0x01000164,0xfffda319,0x0000e6f1,
0x01000164,0x00011a7b,0x0000abce,
0x01000164,0x0000c10d,0x000092cb,
0x01000164,0xfffe21d5,0x0000c800,
0x01000164,0x00009f78,0x00010cd4,
0x01000164,0x00007996,0x00012c52,
0x01000164,0xffff4131,0x00007443,
0x01000164,0xffff9fa3,0x0000909e,
0x01000164,0xffff473d,0x0000d52e,
0x01000164,0x00006208,0x00005fcb,
0x01000164,0x00017848,0x00011f02,
0x01000164,0xfffeec4c,0x0000db90,
0x01000164,0x00012674,0x000095c7,
0x01000164,0xffff0308,0x0001ad8c,
0x01000164,0x00023c2a,0x0000f621,
0x01000164,0x000191c3,0x0000bfd7,
0x01000164,0xffff82b0,0x0000e9d0,
0x01000164,0xffff610d,0x0000dba3,
0x01000164,0xfffe1788,0x0000b294,
0x01000164,0x0000a95c,0x0000aa9e,
0x01000164,0x000103ec,0x00009323,
0x01000164,0xffff8257,0x00016660,
0x01000164,0xfffea5fe,0x000089d4,
0x01000164,0x0000885f,0x00009416,
0x01000164,0x0000d297,0x0000c6f6,
0x01000164,0xffffd5b6,0x00008d4a,
0x01000164,0xfffe3c40,0x000106fb,
0x01000164,0xffff7bb9,0x0000d1d8,
0x01000164,0x00008a02,0x0001252f,
0x01000164,0x000099a4,0x00009e14,
0x01000164,0x00008ef1,0x0000823a,
0x01000164,0x00013da6,0x0000cdf0,
0x01000164,0xfffe3d4e,0x0000891a,
0x01000164,0x00001dbb,0x000113fa,
0x01000164,0xffffa8c1,0x0000bd5f,
0x01000164,0x00029341,0x0000889f,
0x01000164,0xffff8065,0x0000a527,
0x01000164,0xffffbcc6,0x00009def,
0x01000164,0xffffcd77,0x00008c38,
0x01000164,0x00003037,0x00008f1c,
0x01000164,0x0000f12c,0x0001279d,
0x01000164,0xffff55d4,0x00013168,
0x01000164,0xffff7613,0x0000f524,
0x01000164,0x0000750f,0x0000bed7,
0x01000164,0xffff5d4c,0x0000e44e,
0x01000164,0xfffe95c0,0x000138d1,
0x01000164,0x00011a72,0x0000c76e,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_32[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_32[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0xba2),
    .dimensions = {
      .height = 1,
      .width = 23,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 31,
    .stride = 28,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_32[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 2703531,
    .node_id = 32,
    .dequant_round = 25,
    .dequant_zp = 21,
    .stride = 4,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_32 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_32,
    .input_vector_list = okay_nordic_input_vector_list_32,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_32,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 228,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_32,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 32,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=33) && (AXON_LAYER_TEST_STOP_LAYER>=33)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_33[253] = {
// segment 0,length 251,Axon NN
0x1fff00fb,0x000000fe,
0x10000080,0x00010100,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x01830400,0x00000000,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330001,0x00000004,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00150000,0x00010040,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x200),0x00030040,0x00000040,0x00000000,0x00000000,
0x030000c8,0x00000000,0x01040000,0x00000000,0x00000003,
0x000000d8,0x00000001,
0x000000f0,0x00000100,
0x07000080,0x00010040,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x200),0x00330040,0x00040000,0x00010040,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_okay_nordic.l33_filters,0x00330040,0x00000040,
0x040000a8,0x00050000,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x00030004,0x00000004,
0x030000c8,0x00010000,0x01010000,0x03010000,0x00011703,
0x01000180,0x00000003,0x00000000,
0x000001a4,0x00000000,
0x010001c8,0xc0000000,0xc0000000,
0x01000164,0x000011ac,0x0000dd8f,
0x000000d8,0x01000005,
0x000000f0,0x00000100,
0x01000164,0xffffd660,0x00015a55,
0x01000164,0xfffff000,0x0000f02f,
0x01000164,0xffff801f,0x0000b1c7,
0x01000164,0x000005ae,0x000110d7,
0x01000164,0xffffb437,0x0000eb0a,
0x01000164,0xffffc7c7,0x0000e3c9,
0x01000164,0xfffff4f3,0x0000b1cc,
0x01000164,0xffffe71f,0x0000e6cb,
0x01000164,0xffffe1aa,0x0000ddf0,
0x01000164,0xffffcf0e,0x0000e4a1,
0x01000164,0x00000a06,0x00011d18,
0x01000164,0xfffffd8c,0x0000dbd8,
0x01000164,0x000031d2,0x0000b97d,
0x01000164,0x0000134d,0x00016604,
0x01000164,0xffff9564,0x0000df0f,
0x01000164,0x00000d61,0x0000eae1,
0x01000164,0x000029ab,0x00011651,
0x01000164,0xffffd6fe,0x0000f6ac,
0x01000164,0x00004817,0x0000d734,
0x01000164,0x000036ac,0x0000c21b,
0x01000164,0xfffff1ea,0x0000eb08,
0x01000164,0xfffff1fd,0x0000e443,
0x01000164,0x000024bf,0x0000cc19,
0x01000164,0xffffbac7,0x0000fc6f,
0x01000164,0xffffeda8,0x00011704,
0x01000164,0x00000af4,0x000090f8,
0x01000164,0x000014b4,0x0000df0f,
0x01000164,0x000014fa,0x0000c6c4,
0x01000164,0x000003a9,0x00011c57,
0x01000164,0xffffc27b,0x0000e33e,
0x01000164,0x00000ba6,0x0000c8bc,
0x01000164,0xffffd546,0x00009d2e,
0x01000164,0x00004000,0x0000cf2d,
0x01000164,0xffffd826,0x0000f396,
0x01000164,0x00000347,0x0000fc67,
0x01000164,0xfffffd68,0x0001325b,
0x01000164,0xffffe5d9,0x00013316,
0x01000164,0xffffb95b,0x0000ea59,
0x01000164,0xffffd4db,0x00012b8a,
0x01000164,0xffffe3b9,0x0000ea42,
0x01000164,0x0000145b,0x0000bd03,
0x01000164,0x000000b3,0x0000f7e2,
0x01000164,0xfffff3c0,0x0000c980,
0x01000164,0xffffb2fa,0x0000c11b,
0x01000164,0x00000434,0x0000c8bb,
0x01000164,0x000009e3,0x00013328,
0x01000164,0xffffcaa7,0x0000d0aa,
0x01000164,0x00000894,0x0000df22,
0x01000164,0xffffe846,0x00010eb4,
0x01000164,0xffffc158,0x0000cb2a,
0x01000164,0xffffae66,0x0000ea36,
0x01000164,0xffffe138,0x0000ba5f,
0x01000164,0xfffff289,0x0000e959,
0x01000164,0x0000041d,0x0000c35d,
0x01000164,0xffff9d29,0x0000d954,
0x01000164,0x00002277,0x00015e09,
0x01000164,0x0000035b,0x00010813,
0x01000164,0xffffcee1,0x0000d352,
0x01000164,0xffffef1c,0x0000e17a,
0x01000164,0xfffffbdb,0x0000edc7,
0x01000164,0xffffc46b,0x0000d4df,
0x01000164,0xffffee73,0x0000e5ef,
0x01000164,0x00002f08,0x0000eb64,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_33[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_33[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 32,
    .stride = 4,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_33[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 33641,
    .node_id = 33,
    .dequant_round = 20,
    .dequant_zp = -128,
    .stride = 4,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_33 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_33,
    .input_vector_list = okay_nordic_input_vector_list_33,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_33,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 253,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_33,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 33,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=34) && (AXON_LAYER_TEST_STOP_LAYER>=34)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_34[39] = {
// segment 0,length 37,Axon NN
0x1fff0025,0x00000028,
0x06000080,0x00010010,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x820),0x00330010,0x00000010,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00030040,
0x070000a0,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x00010010,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x12a0),0x00030014,0x00000014,0x00000000,
0x040000c8,0x00000000,0x00000000,0x00000000,0x00000003,0x00000000,
0x000000f0,0x00000100,
0x03000080,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x00330004,0x00000004,
0x010000ac,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x12b0),
0x000000bc,0x00000003,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_34[2];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_34[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x820),
    .dimensions = {
      .height = 1,
      .width = 16,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 4,
    .stride = 16,
  }, /* 0 */
  {/* 1 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 33,
    .stride = 4,
  }, /* 1 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_34[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x12a0),
    .dimensions = {
      .height = 1,
      .width = 17,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 33641,
    .node_id = 34,
    .dequant_round = 20,
    .dequant_zp = -128,
    .stride = 20,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_34 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_34,
    .input_vector_list = okay_nordic_input_vector_list_34,
    .input_cnt = 2,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_34,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 39,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_34,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 34,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=35) && (AXON_LAYER_TEST_STOP_LAYER>=35)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_35[51] = {
// segment 0,length 49,Axon NN
0x1fff0031,0x00000034,
0x06000080,0x000103ff,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x12a0),0x003303ff,0x00000000,0x000103ff,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_okay_nordic.l35_weights,0x00330001,
0x050000a0,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_okay_nordic.l35_biasp,0x00050004,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x00050002,
0x000000bc,0x00000000,
0x040000c8,0x00000000,0x00000000,0x00000000,0x00000005,0x00000205,
0x01000180,0x00000002,0x80000000,
0x010001a4,0x00000000,0x00017ee3,
0x000001cc,0x00000000,
0x000000f0,0x00000100,
0x01000080,0x000100fe,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x169f),
0x01000090,0x000100fe,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)(axon_model_const_okay_nordic.l35_weights+0x3ff),
0x000000a4,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
0x010000b0,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x104),0x00040002,
0x010000d0,0x03010000,0x00011204,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_35[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_35[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x12a0),
    .dimensions = {
      .height = 1,
      .width = 17,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 34,
    .stride = 20,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_35[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x104),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 2,
    },
    .dequant_mult = 4096,
    .node_id = 35,
    .dequant_round = 12,
    .dequant_zp = 0,
    .stride = 2,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_35 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_35,
    .input_vector_list = okay_nordic_input_vector_list_35,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_35,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 51,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_35,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 35,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=36) && (AXON_LAYER_TEST_STOP_LAYER>=36)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_36[11] = {
// segment 0,length 9,CPU(software)
0x2fff0009,0x0000000c,
0x00000001,
(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_nn_op_extension_sigmoid_v2,0x00000006,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x104),(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),0x00000100,0x00010001,0x80000001,0x00000001,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_36[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_36[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x104),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 2,
    },
    .node_id = 35,
    .stride = 2,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_36[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x100),
    .dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 1,
    .node_id = 36,
    .dequant_round = 8,
    .dequant_zp = -128,
    .stride = 1,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_36 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_36,
    .input_vector_list = okay_nordic_input_vector_list_36,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_36,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 11,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_36,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 36,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=37) && (AXON_LAYER_TEST_STOP_LAYER>=37)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_37[27] = {
// segment 0,length 25,Axon NN
0x1fff0019,0x0000001c,
0x0f000080,0x00010010,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x12a1),0x00330014,0x00000014,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330040,0x00000000,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x00010010,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x820),0x00030010,0x00000010,0x00000003,
0x040000c8,0x00000000,0x01010000,0x00000000,0x00000003,0x00000001,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_37[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_37[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x12a0),
    .dimensions = {
      .height = 1,
      .width = 17,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 34,
    .stride = 20,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_37[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0x820),
    .dimensions = {
      .height = 1,
      .width = 16,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 33641,
    .node_id = 37,
    .dequant_round = 20,
    .dequant_zp = -128,
    .stride = 16,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_37 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_37,
    .input_vector_list = okay_nordic_input_vector_list_37,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_37,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 27,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_37,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 37,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=38) && (AXON_LAYER_TEST_STOP_LAYER>=38)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_okay_nordic_38[27] = {
// segment 0,length 25,Axon NN
0x1fff0019,0x0000001c,
0x0f000080,0x00010016,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0xba3),0x0033001c,0x0000001c,0x00010004,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axonpro_int8_packing_filter,0x00330040,0x00000000,0x00000000,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)NULL,0x00050000,0x00010016,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0xc20),0x00030018,0x00000018,0x00000003,
0x040000c8,0x00000000,0x01010000,0x00000000,0x00000003,0x00000001,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *okay_nordic_input_vector_list_38[1];
const nrf_axon_nn_compiled_model_input_s okay_nordic_inputs_38[] = {
  {/* 0 */
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0xba2),
    .dimensions = {
      .height = 1,
      .width = 23,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 31,
    .stride = 28,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s okay_nordic_outputs_38[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(axon_model_okay_nordic_persistent_vars)+0xc20),
    .dimensions = {
      .height = 1,
      .width = 22,
      .channel_cnt = 64,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 283731,
    .node_id = 38,
    .dequant_round = 23,
    .dequant_zp = -128,
    .stride = 24,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_okay_nordic_38 = {
  .base = {
    .compiler_version = 0x00020100,
    .model_name = "okay_nordic",
    .labels = NULL,
    .inputs = okay_nordic_inputs_38,
    .input_vector_list = okay_nordic_input_vector_list_38,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_OKAY_NORDIC_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_okay_nordic_38,

    .model_const_ptr = &axon_model_const_okay_nordic,
    .model_const_size = sizeof(axon_model_const_okay_nordic),
    .cmd_buffer_len = 27,
    .persistent_vars = {
      .buf_ptr = (int8_t *)axon_model_okay_nordic_persistent_vars,
      .buf_size = sizeof(axon_model_okay_nordic_persistent_vars),
      .vars = axon_model_okay_nordic_persistent_var_list,
      .count = 6,
    },
    .output_cnt = 1,
    .outputs = okay_nordic_outputs_38,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)okay_nordic_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 38,
};
#endif
#define MODEL_okay_nordic_FIRST_COMPUTE_LAYER (6)
nrf_axon_nn_compiled_model_layer_s const *model_okay_nordic_layer_list[] = {
	  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
#if (AXON_LAYER_TEST_START_LAYER<=6) && (AXON_LAYER_TEST_STOP_LAYER>=6)
  &model_okay_nordic_6,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=7) && (AXON_LAYER_TEST_STOP_LAYER>=7)
  &model_okay_nordic_7,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=8) && (AXON_LAYER_TEST_STOP_LAYER>=8)
  &model_okay_nordic_8,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=9) && (AXON_LAYER_TEST_STOP_LAYER>=9)
  &model_okay_nordic_9,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=10) && (AXON_LAYER_TEST_STOP_LAYER>=10)
  &model_okay_nordic_10,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=11) && (AXON_LAYER_TEST_STOP_LAYER>=11)
  &model_okay_nordic_11,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=12) && (AXON_LAYER_TEST_STOP_LAYER>=12)
  &model_okay_nordic_12,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=13) && (AXON_LAYER_TEST_STOP_LAYER>=13)
  &model_okay_nordic_13,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=14) && (AXON_LAYER_TEST_STOP_LAYER>=14)
  &model_okay_nordic_14,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=15) && (AXON_LAYER_TEST_STOP_LAYER>=15)
  &model_okay_nordic_15,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=16) && (AXON_LAYER_TEST_STOP_LAYER>=16)
  &model_okay_nordic_16,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=17) && (AXON_LAYER_TEST_STOP_LAYER>=17)
  &model_okay_nordic_17,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=18) && (AXON_LAYER_TEST_STOP_LAYER>=18)
  &model_okay_nordic_18,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=19) && (AXON_LAYER_TEST_STOP_LAYER>=19)
  &model_okay_nordic_19,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=20) && (AXON_LAYER_TEST_STOP_LAYER>=20)
  &model_okay_nordic_20,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=21) && (AXON_LAYER_TEST_STOP_LAYER>=21)
  &model_okay_nordic_21,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=22) && (AXON_LAYER_TEST_STOP_LAYER>=22)
  &model_okay_nordic_22,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=23) && (AXON_LAYER_TEST_STOP_LAYER>=23)
  &model_okay_nordic_23,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=24) && (AXON_LAYER_TEST_STOP_LAYER>=24)
  &model_okay_nordic_24,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=25) && (AXON_LAYER_TEST_STOP_LAYER>=25)
  &model_okay_nordic_25,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=26) && (AXON_LAYER_TEST_STOP_LAYER>=26)
  &model_okay_nordic_26,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=27) && (AXON_LAYER_TEST_STOP_LAYER>=27)
  &model_okay_nordic_27,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=28) && (AXON_LAYER_TEST_STOP_LAYER>=28)
  &model_okay_nordic_28,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=29) && (AXON_LAYER_TEST_STOP_LAYER>=29)
  &model_okay_nordic_29,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=30) && (AXON_LAYER_TEST_STOP_LAYER>=30)
  &model_okay_nordic_30,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=31) && (AXON_LAYER_TEST_STOP_LAYER>=31)
  &model_okay_nordic_31,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=32) && (AXON_LAYER_TEST_STOP_LAYER>=32)
  &model_okay_nordic_32,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=33) && (AXON_LAYER_TEST_STOP_LAYER>=33)
  &model_okay_nordic_33,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=34) && (AXON_LAYER_TEST_STOP_LAYER>=34)
  &model_okay_nordic_34,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=35) && (AXON_LAYER_TEST_STOP_LAYER>=35)
  &model_okay_nordic_35,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=36) && (AXON_LAYER_TEST_STOP_LAYER>=36)
  &model_okay_nordic_36,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=37) && (AXON_LAYER_TEST_STOP_LAYER>=37)
  &model_okay_nordic_37,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=38) && (AXON_LAYER_TEST_STOP_LAYER>=38)
  &model_okay_nordic_38,
#else
  NULL,
#endif

};

#ifdef __cplusplus
}
#endif
