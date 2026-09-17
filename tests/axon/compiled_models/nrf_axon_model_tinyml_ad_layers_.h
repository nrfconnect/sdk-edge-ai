/*********************************************************************************
 * Auto-generated nrf Axon compiled neural network model header file.
 * Model Name: tinyml_ad
 * Axon Neural Network Compiler Version: 2.0.0
 *********************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
#include <stdalign.h>
#include "drivers/axon/nrf_axon_nn_infer.h"
#if (AXON_LAYER_TEST_START_LAYER<=0) && (AXON_LAYER_TEST_STOP_LAYER>=0)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_0[36] = {
// segment 0,length 34,Axon NN
0x1fff0022,0x00000024,
0x02000080,0x00010280,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00330280,
0x02000090,0x00800280,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l00_weights,0x00330001,
0x050000a0,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l00_biasp,0x00050200,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00030080,
0x000000bc,0x00000000,
0x040000c8,0x00000000,0x00000000,0x03010000,0x00011703,0x00000205,
0x01000180,0x00000003,0x00000000,
0x010001a4,0x00000000,0x000061a2,
0x010001c8,0xc0000000,0xc0000000,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *tinyml_ad_input_vector_list_0[1];
const nrf_axon_nn_compiled_model_input_s tinyml_ad_inputs_0[] = {
  {/* 0 */
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 640,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .quant_mult = 1340838,
    .node_id = -1,
    .stride = 640,
    .quant_round = 19,
    .quant_zp = 89,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s tinyml_ad_outputs_0[] = {
  {
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 0,
    .node_id = 0,
    .dequant_round = 0,
    .dequant_zp = 0,
    .stride = 128,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_0 = {
  .base = {
    .compiler_version = 0x00020000,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = tinyml_ad_inputs_0,
    .input_vector_list = tinyml_ad_input_vector_list_0,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_0,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 36,
    .persistent_vars = {
      .count = 0,
    },
    .output_cnt = 1,
    .outputs = tinyml_ad_outputs_0,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)tinyml_ad_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 0,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=1) && (AXON_LAYER_TEST_STOP_LAYER>=1)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_1[36] = {
// segment 0,length 34,Axon NN
0x1fff0022,0x00000024,
0x02000080,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00330080,
0x02000090,0x00800080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l01_weights,0x00330001,
0x050000a0,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l01_biasp,0x00050200,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00030080,
0x000000bc,0x00000000,
0x040000c8,0x00000000,0x00000000,0x03010000,0x00011703,0x00000205,
0x01000180,0x00000003,0x00000000,
0x010001a4,0x00000000,0x0002afea,
0x010001c8,0xc0000000,0xc0000000,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *tinyml_ad_input_vector_list_1[1];
const nrf_axon_nn_compiled_model_input_s tinyml_ad_inputs_1[] = {
  {/* 0 */
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 0,
    .stride = 128,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s tinyml_ad_outputs_1[] = {
  {
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 0,
    .node_id = 0,
    .dequant_round = 0,
    .dequant_zp = 0,
    .stride = 128,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_1 = {
  .base = {
    .compiler_version = 0x00020000,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = tinyml_ad_inputs_1,
    .input_vector_list = tinyml_ad_input_vector_list_1,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_1,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 36,
    .persistent_vars = {
      .count = 0,
    },
    .output_cnt = 1,
    .outputs = tinyml_ad_outputs_1,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)tinyml_ad_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 1,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=2) && (AXON_LAYER_TEST_STOP_LAYER>=2)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_2[36] = {
// segment 0,length 34,Axon NN
0x1fff0022,0x00000024,
0x02000080,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00330080,
0x02000090,0x00800080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l02_weights,0x00330001,
0x050000a0,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l02_biasp,0x00050200,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00030080,
0x000000bc,0x00000000,
0x040000c8,0x00000000,0x00000000,0x03010000,0x00011703,0x00000205,
0x01000180,0x00000003,0x00000000,
0x010001a4,0x00000000,0x0011a87e,
0x010001c8,0xc0000000,0xc0000000,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *tinyml_ad_input_vector_list_2[1];
const nrf_axon_nn_compiled_model_input_s tinyml_ad_inputs_2[] = {
  {/* 0 */
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 1,
    .stride = 128,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s tinyml_ad_outputs_2[] = {
  {
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 0,
    .node_id = 0,
    .dequant_round = 0,
    .dequant_zp = 0,
    .stride = 128,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_2 = {
  .base = {
    .compiler_version = 0x00020000,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = tinyml_ad_inputs_2,
    .input_vector_list = tinyml_ad_input_vector_list_2,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_2,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 36,
    .persistent_vars = {
      .count = 0,
    },
    .output_cnt = 1,
    .outputs = tinyml_ad_outputs_2,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)tinyml_ad_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 2,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=3) && (AXON_LAYER_TEST_STOP_LAYER>=3)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_3[36] = {
// segment 0,length 34,Axon NN
0x1fff0022,0x00000024,
0x02000080,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00330080,
0x02000090,0x00800080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l03_weights,0x00330001,
0x050000a0,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l03_biasp,0x00050200,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00030080,
0x000000bc,0x00000000,
0x040000c8,0x00000000,0x00000000,0x03010000,0x00011703,0x00000205,
0x01000180,0x00000003,0x00000000,
0x010001a4,0x00000000,0x00055d1f,
0x010001c8,0xc0000000,0xc0000000,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *tinyml_ad_input_vector_list_3[1];
const nrf_axon_nn_compiled_model_input_s tinyml_ad_inputs_3[] = {
  {/* 0 */
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 2,
    .stride = 128,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s tinyml_ad_outputs_3[] = {
  {
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 0,
    .node_id = 0,
    .dequant_round = 0,
    .dequant_zp = 0,
    .stride = 128,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_3 = {
  .base = {
    .compiler_version = 0x00020000,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = tinyml_ad_inputs_3,
    .input_vector_list = tinyml_ad_input_vector_list_3,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_3,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 36,
    .persistent_vars = {
      .count = 0,
    },
    .output_cnt = 1,
    .outputs = tinyml_ad_outputs_3,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)tinyml_ad_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 3,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=4) && (AXON_LAYER_TEST_STOP_LAYER>=4)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_4[36] = {
// segment 0,length 34,Axon NN
0x1fff0022,0x00000024,
0x02000080,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00330080,
0x02000090,0x00080080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l04_weights,0x00330001,
0x050000a0,0x00010008,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l04_biasp,0x00050020,0x00010008,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00030008,
0x000000bc,0x00000000,
0x040000c8,0x00000000,0x00000000,0x03010000,0x00011703,0x00000205,
0x01000180,0x00000003,0x00000000,
0x010001a4,0x00000000,0x000102e5,
0x010001c8,0xc0000000,0xc0000000,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *tinyml_ad_input_vector_list_4[1];
const nrf_axon_nn_compiled_model_input_s tinyml_ad_inputs_4[] = {
  {/* 0 */
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 3,
    .stride = 128,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s tinyml_ad_outputs_4[] = {
  {
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 8,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 0,
    .node_id = 0,
    .dequant_round = 0,
    .dequant_zp = 0,
    .stride = 8,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_4 = {
  .base = {
    .compiler_version = 0x00020000,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = tinyml_ad_inputs_4,
    .input_vector_list = tinyml_ad_input_vector_list_4,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_4,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 36,
    .persistent_vars = {
      .count = 0,
    },
    .output_cnt = 1,
    .outputs = tinyml_ad_outputs_4,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)tinyml_ad_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 4,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=5) && (AXON_LAYER_TEST_STOP_LAYER>=5)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_5[36] = {
// segment 0,length 34,Axon NN
0x1fff0022,0x00000024,
0x02000080,0x00010008,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00330008,
0x02000090,0x00800008,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l05_weights,0x00330001,
0x050000a0,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l05_biasp,0x00050200,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00030080,
0x000000bc,0x00000000,
0x040000c8,0x00000000,0x00000000,0x03010000,0x00011703,0x00000205,
0x01000180,0x00000003,0x00000000,
0x010001a4,0x00000000,0x0002afb6,
0x010001c8,0xc0000000,0xc0000000,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *tinyml_ad_input_vector_list_5[1];
const nrf_axon_nn_compiled_model_input_s tinyml_ad_inputs_5[] = {
  {/* 0 */
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 8,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 4,
    .stride = 8,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s tinyml_ad_outputs_5[] = {
  {
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 0,
    .node_id = 0,
    .dequant_round = 0,
    .dequant_zp = 0,
    .stride = 128,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_5 = {
  .base = {
    .compiler_version = 0x00020000,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = tinyml_ad_inputs_5,
    .input_vector_list = tinyml_ad_input_vector_list_5,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_5,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 36,
    .persistent_vars = {
      .count = 0,
    },
    .output_cnt = 1,
    .outputs = tinyml_ad_outputs_5,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)tinyml_ad_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 5,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=6) && (AXON_LAYER_TEST_STOP_LAYER>=6)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_6[36] = {
// segment 0,length 34,Axon NN
0x1fff0022,0x00000024,
0x02000080,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00330080,
0x02000090,0x00800080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l06_weights,0x00330001,
0x050000a0,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l06_biasp,0x00050200,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00030080,
0x000000bc,0x00000000,
0x040000c8,0x00000000,0x00000000,0x03010000,0x00011703,0x00000205,
0x01000180,0x00000003,0x00000000,
0x010001a4,0x00000000,0x0002735c,
0x010001c8,0xc0000000,0xc0000000,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *tinyml_ad_input_vector_list_6[1];
const nrf_axon_nn_compiled_model_input_s tinyml_ad_inputs_6[] = {
  {/* 0 */
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 5,
    .stride = 128,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s tinyml_ad_outputs_6[] = {
  {
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 0,
    .node_id = 0,
    .dequant_round = 0,
    .dequant_zp = 0,
    .stride = 128,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_6 = {
  .base = {
    .compiler_version = 0x00020000,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = tinyml_ad_inputs_6,
    .input_vector_list = tinyml_ad_input_vector_list_6,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_6,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 36,
    .persistent_vars = {
      .count = 0,
    },
    .output_cnt = 1,
    .outputs = tinyml_ad_outputs_6,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)tinyml_ad_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 6,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=7) && (AXON_LAYER_TEST_STOP_LAYER>=7)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_7[36] = {
// segment 0,length 34,Axon NN
0x1fff0022,0x00000024,
0x02000080,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00330080,
0x02000090,0x00800080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l07_weights,0x00330001,
0x050000a0,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l07_biasp,0x00050200,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00030080,
0x000000bc,0x00000000,
0x040000c8,0x00000000,0x00000000,0x03010000,0x00011703,0x00000205,
0x01000180,0x00000003,0x00000000,
0x010001a4,0x00000000,0x0001db7e,
0x010001c8,0xc0000000,0xc0000000,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *tinyml_ad_input_vector_list_7[1];
const nrf_axon_nn_compiled_model_input_s tinyml_ad_inputs_7[] = {
  {/* 0 */
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 6,
    .stride = 128,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s tinyml_ad_outputs_7[] = {
  {
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 0,
    .node_id = 0,
    .dequant_round = 0,
    .dequant_zp = 0,
    .stride = 128,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_7 = {
  .base = {
    .compiler_version = 0x00020000,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = tinyml_ad_inputs_7,
    .input_vector_list = tinyml_ad_input_vector_list_7,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_7,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 36,
    .persistent_vars = {
      .count = 0,
    },
    .output_cnt = 1,
    .outputs = tinyml_ad_outputs_7,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)tinyml_ad_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 7,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=8) && (AXON_LAYER_TEST_STOP_LAYER>=8)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_8[36] = {
// segment 0,length 34,Axon NN
0x1fff0022,0x00000024,
0x02000080,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00330080,
0x02000090,0x00800080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l08_weights,0x00330001,
0x050000a0,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l08_biasp,0x00050200,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00030080,
0x000000bc,0x00000000,
0x040000c8,0x00000000,0x00000000,0x03010000,0x00011703,0x00000205,
0x01000180,0x00000003,0x00000000,
0x010001a4,0x00000000,0x000107ac,
0x010001c8,0xc0000000,0xc0000000,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *tinyml_ad_input_vector_list_8[1];
const nrf_axon_nn_compiled_model_input_s tinyml_ad_inputs_8[] = {
  {/* 0 */
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 7,
    .stride = 128,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s tinyml_ad_outputs_8[] = {
  {
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .dequant_mult = 0,
    .node_id = 0,
    .dequant_round = 0,
    .dequant_zp = 0,
    .stride = 128,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_8 = {
  .base = {
    .compiler_version = 0x00020000,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = tinyml_ad_inputs_8,
    .input_vector_list = tinyml_ad_input_vector_list_8,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_8,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 36,
    .persistent_vars = {
      .count = 0,
    },
    .output_cnt = 1,
    .outputs = tinyml_ad_outputs_8,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)tinyml_ad_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 8,
};
#endif
#if (AXON_LAYER_TEST_START_LAYER<=9) && (AXON_LAYER_TEST_STOP_LAYER>=9)

NRF_AXON_CMD_BUFFER_ALIGN
const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_9[47] = {
// segment 0,length 45,Axon NN
0x1fff002d,0x00000030,
0x02000080,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00330080,
0x02000090,0x02000080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l09_weights,0x00330001,
0x050000a0,0x00010200,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_tinyml_ad.l09_biasp,0x00050800,0x00010200,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x80),0x00050800,
0x000000bc,0x00000000,
0x040000c8,0x00000000,0x00000000,0x03010000,0x00010d05,0x00000205,
0x01000180,0x00000002,0x80000000,
0x010001a4,0x00000000,0x0007f184,
0x000001cc,0x00000000,
0x000000f0,0x00000100,
0x01000090,0x00800080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)(axon_model_const_tinyml_ad.l09_weights+0x10000),
0x050000a0,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)(axon_model_const_tinyml_ad.l09_biasp+0x200),0x00050200,0x00010080,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)((uint8_t*)(nrf_axon_interlayer_buffer)+0x880),0x00050200,
0x000000f0,0x00000100,
};
NRF_AXON_MODEL_APP_STORAGE const int8_t *tinyml_ad_input_vector_list_9[1];
const nrf_axon_nn_compiled_model_input_s tinyml_ad_inputs_9[] = {
  {/* 0 */
    .ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 1,
    },
    .node_id = 8,
    .stride = 128,
  }, /* 0 */
}; /* inputs */

const nrf_axon_compiled_model_output_s tinyml_ad_outputs_9[] = {
  {
    .ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x80),
    .dimensions = {
      .height = 1,
      .width = 640,
      .channel_cnt = 1,
      .batch_cnt = 1,
      .byte_width = 4,
    },
    .dequant_mult = 0,
    .node_id = 0,
    .dequant_round = 0,
    .dequant_zp = 0,
    .stride = 2560,
  },
};
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_9 = {
  .base = {
    .compiler_version = 0x00020000,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = tinyml_ad_inputs_9,
    .input_vector_list = tinyml_ad_input_vector_list_9,
    .input_cnt = 1,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_9,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 47,
    .persistent_vars = {
      .count = 0,
    },
    .output_cnt = 1,
    .outputs = tinyml_ad_outputs_9,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t *)tinyml_ad_packed_output_buf,
#endif
    .min_driver_version_required = 0x00010501,
    .is_layer_model = true,
  },
  .layer_ndx = 9,
};
#endif
#define MODEL_tinyml_ad_FIRST_COMPUTE_LAYER (0)
nrf_axon_nn_compiled_model_layer_s const *model_tinyml_ad_layer_list[] = {
	#if (AXON_LAYER_TEST_START_LAYER<=0) && (AXON_LAYER_TEST_STOP_LAYER>=0)
  &model_tinyml_ad_0,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=1) && (AXON_LAYER_TEST_STOP_LAYER>=1)
  &model_tinyml_ad_1,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=2) && (AXON_LAYER_TEST_STOP_LAYER>=2)
  &model_tinyml_ad_2,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=3) && (AXON_LAYER_TEST_STOP_LAYER>=3)
  &model_tinyml_ad_3,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=4) && (AXON_LAYER_TEST_STOP_LAYER>=4)
  &model_tinyml_ad_4,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=5) && (AXON_LAYER_TEST_STOP_LAYER>=5)
  &model_tinyml_ad_5,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=6) && (AXON_LAYER_TEST_STOP_LAYER>=6)
  &model_tinyml_ad_6,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=7) && (AXON_LAYER_TEST_STOP_LAYER>=7)
  &model_tinyml_ad_7,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=8) && (AXON_LAYER_TEST_STOP_LAYER>=8)
  &model_tinyml_ad_8,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=9) && (AXON_LAYER_TEST_STOP_LAYER>=9)
  &model_tinyml_ad_9,
#else
  NULL,
#endif

};

#ifdef __cplusplus
}
#endif
