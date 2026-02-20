/*********************************************************************************
 * Auto-generated nrf Axon compiled neural network model header file.
 * Model Name: tinyml_ad
 * Axon Neural Network Compiler Version: 0.2.0
 *********************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
#if (AXON_LAYER_TEST_START_LAYER<=0) && (AXON_LAYER_TEST_STOP_LAYER>=0)

const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_0_0[35] = {
// segment 0,length 34,Axon NN
0x1fff0022,
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
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_0_0 = {
  .base = {
    .compiler_version = 0x00000200,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = {
      {// 0
        .ptr = (int8_t*)nrf_axon_interlayer_buffer,
        .dimensions = {
          .height = 1,
          .width = 640,
          .channel_cnt = 1,
          .byte_width = 1,
        },
        .quant_mult = 1340838,
        .stride = 640,
        .quant_round = 19,
        .quant_zp = 89,
        .is_external = true,
      }, // 0
    }, // inputs
    .input_cnt = 1,
    .external_input_ndx = 0,
    .output_ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .packed_output_buf = NULL,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_0_0,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 35,
    .persistent_vars = {
      .count = 0,
    },

    .output_dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .byte_width = 1,
    },
    .output_dequant_mult = 1,
    .output_dequant_round = 17,
    .output_dequant_zp = 0,
    .output_stride = 128,
    .is_layer_model = true,
  },// .base
  .layer_ndx = 0,
  .input0_layer_ndx = -1,
  .input1_layer_ndx = -1,
};
#endif
#ifdef __cplusplus
}
#endif
/*********************************************************************************
 * Auto-generated nrf Axon compiled neural network model header file.
 * Model Name: tinyml_ad
 * Axon Neural Network Compiler Version: 0.2.0
 *********************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
#if (AXON_LAYER_TEST_START_LAYER<=1) && (AXON_LAYER_TEST_STOP_LAYER>=1)

const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_1_1[35] = {
// segment 0,length 34,Axon NN
0x1fff0022,
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
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_1_1 = {
  .base = {
    .compiler_version = 0x00000200,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = {
      {// 0
        .ptr = (int8_t*)nrf_axon_interlayer_buffer,
        .dimensions = {
          .height = 1,
          .width = 128,
          .channel_cnt = 1,
          .byte_width = 1,
        },
        .stride = 128,
        .is_external = false,
      }, // 0
    }, // inputs
    .input_cnt = 1,
    .external_input_ndx = -1,
    .output_ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .packed_output_buf = NULL,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_1_1,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 35,
    .persistent_vars = {
      .count = 0,
    },

    .output_dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .byte_width = 1,
    },
    .output_dequant_mult = 1,
    .output_dequant_round = 17,
    .output_dequant_zp = 0,
    .output_stride = 128,
    .is_layer_model = true,
  },// .base
  .layer_ndx = 1,
  .input0_layer_ndx = 0,
  .input1_layer_ndx = -1,
};
#endif
#ifdef __cplusplus
}
#endif
/*********************************************************************************
 * Auto-generated nrf Axon compiled neural network model header file.
 * Model Name: tinyml_ad
 * Axon Neural Network Compiler Version: 0.2.0
 *********************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
#if (AXON_LAYER_TEST_START_LAYER<=2) && (AXON_LAYER_TEST_STOP_LAYER>=2)

const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_2_2[35] = {
// segment 0,length 34,Axon NN
0x1fff0022,
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
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_2_2 = {
  .base = {
    .compiler_version = 0x00000200,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = {
      {// 0
        .ptr = (int8_t*)nrf_axon_interlayer_buffer,
        .dimensions = {
          .height = 1,
          .width = 128,
          .channel_cnt = 1,
          .byte_width = 1,
        },
        .stride = 128,
        .is_external = false,
      }, // 0
    }, // inputs
    .input_cnt = 1,
    .external_input_ndx = -1,
    .output_ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .packed_output_buf = NULL,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_2_2,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 35,
    .persistent_vars = {
      .count = 0,
    },

    .output_dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .byte_width = 1,
    },
    .output_dequant_mult = 1,
    .output_dequant_round = 17,
    .output_dequant_zp = 0,
    .output_stride = 128,
    .is_layer_model = true,
  },// .base
  .layer_ndx = 2,
  .input0_layer_ndx = 1,
  .input1_layer_ndx = -1,
};
#endif
#ifdef __cplusplus
}
#endif
/*********************************************************************************
 * Auto-generated nrf Axon compiled neural network model header file.
 * Model Name: tinyml_ad
 * Axon Neural Network Compiler Version: 0.2.0
 *********************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
#if (AXON_LAYER_TEST_START_LAYER<=3) && (AXON_LAYER_TEST_STOP_LAYER>=3)

const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_3_3[35] = {
// segment 0,length 34,Axon NN
0x1fff0022,
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
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_3_3 = {
  .base = {
    .compiler_version = 0x00000200,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = {
      {// 0
        .ptr = (int8_t*)nrf_axon_interlayer_buffer,
        .dimensions = {
          .height = 1,
          .width = 128,
          .channel_cnt = 1,
          .byte_width = 1,
        },
        .stride = 128,
        .is_external = false,
      }, // 0
    }, // inputs
    .input_cnt = 1,
    .external_input_ndx = -1,
    .output_ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .packed_output_buf = NULL,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_3_3,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 35,
    .persistent_vars = {
      .count = 0,
    },

    .output_dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .byte_width = 1,
    },
    .output_dequant_mult = 1,
    .output_dequant_round = 17,
    .output_dequant_zp = 0,
    .output_stride = 128,
    .is_layer_model = true,
  },// .base
  .layer_ndx = 3,
  .input0_layer_ndx = 2,
  .input1_layer_ndx = -1,
};
#endif
#ifdef __cplusplus
}
#endif
/*********************************************************************************
 * Auto-generated nrf Axon compiled neural network model header file.
 * Model Name: tinyml_ad
 * Axon Neural Network Compiler Version: 0.2.0
 *********************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
#if (AXON_LAYER_TEST_START_LAYER<=4) && (AXON_LAYER_TEST_STOP_LAYER>=4)

const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_4_4[35] = {
// segment 0,length 34,Axon NN
0x1fff0022,
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
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_4_4 = {
  .base = {
    .compiler_version = 0x00000200,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = {
      {// 0
        .ptr = (int8_t*)nrf_axon_interlayer_buffer,
        .dimensions = {
          .height = 1,
          .width = 128,
          .channel_cnt = 1,
          .byte_width = 1,
        },
        .stride = 128,
        .is_external = false,
      }, // 0
    }, // inputs
    .input_cnt = 1,
    .external_input_ndx = -1,
    .output_ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .packed_output_buf = NULL,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_4_4,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 35,
    .persistent_vars = {
      .count = 0,
    },

    .output_dimensions = {
      .height = 1,
      .width = 8,
      .channel_cnt = 1,
      .byte_width = 1,
    },
    .output_dequant_mult = 1,
    .output_dequant_round = 17,
    .output_dequant_zp = 0,
    .output_stride = 8,
    .is_layer_model = true,
  },// .base
  .layer_ndx = 4,
  .input0_layer_ndx = 3,
  .input1_layer_ndx = -1,
};
#endif
#ifdef __cplusplus
}
#endif
/*********************************************************************************
 * Auto-generated nrf Axon compiled neural network model header file.
 * Model Name: tinyml_ad
 * Axon Neural Network Compiler Version: 0.2.0
 *********************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
#if (AXON_LAYER_TEST_START_LAYER<=5) && (AXON_LAYER_TEST_STOP_LAYER>=5)

const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_5_5[35] = {
// segment 0,length 34,Axon NN
0x1fff0022,
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
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_5_5 = {
  .base = {
    .compiler_version = 0x00000200,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = {
      {// 0
        .ptr = (int8_t*)nrf_axon_interlayer_buffer,
        .dimensions = {
          .height = 1,
          .width = 8,
          .channel_cnt = 1,
          .byte_width = 1,
        },
        .stride = 8,
        .is_external = false,
      }, // 0
    }, // inputs
    .input_cnt = 1,
    .external_input_ndx = -1,
    .output_ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .packed_output_buf = NULL,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_5_5,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 35,
    .persistent_vars = {
      .count = 0,
    },

    .output_dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .byte_width = 1,
    },
    .output_dequant_mult = 1,
    .output_dequant_round = 17,
    .output_dequant_zp = 0,
    .output_stride = 128,
    .is_layer_model = true,
  },// .base
  .layer_ndx = 5,
  .input0_layer_ndx = 4,
  .input1_layer_ndx = -1,
};
#endif
#ifdef __cplusplus
}
#endif
/*********************************************************************************
 * Auto-generated nrf Axon compiled neural network model header file.
 * Model Name: tinyml_ad
 * Axon Neural Network Compiler Version: 0.2.0
 *********************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
#if (AXON_LAYER_TEST_START_LAYER<=6) && (AXON_LAYER_TEST_STOP_LAYER>=6)

const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_6_6[35] = {
// segment 0,length 34,Axon NN
0x1fff0022,
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
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_6_6 = {
  .base = {
    .compiler_version = 0x00000200,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = {
      {// 0
        .ptr = (int8_t*)nrf_axon_interlayer_buffer,
        .dimensions = {
          .height = 1,
          .width = 128,
          .channel_cnt = 1,
          .byte_width = 1,
        },
        .stride = 128,
        .is_external = false,
      }, // 0
    }, // inputs
    .input_cnt = 1,
    .external_input_ndx = -1,
    .output_ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .packed_output_buf = NULL,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_6_6,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 35,
    .persistent_vars = {
      .count = 0,
    },

    .output_dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .byte_width = 1,
    },
    .output_dequant_mult = 1,
    .output_dequant_round = 17,
    .output_dequant_zp = 0,
    .output_stride = 128,
    .is_layer_model = true,
  },// .base
  .layer_ndx = 6,
  .input0_layer_ndx = 5,
  .input1_layer_ndx = -1,
};
#endif
#ifdef __cplusplus
}
#endif
/*********************************************************************************
 * Auto-generated nrf Axon compiled neural network model header file.
 * Model Name: tinyml_ad
 * Axon Neural Network Compiler Version: 0.2.0
 *********************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
#if (AXON_LAYER_TEST_START_LAYER<=7) && (AXON_LAYER_TEST_STOP_LAYER>=7)

const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_7_7[35] = {
// segment 0,length 34,Axon NN
0x1fff0022,
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
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_7_7 = {
  .base = {
    .compiler_version = 0x00000200,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = {
      {// 0
        .ptr = (int8_t*)nrf_axon_interlayer_buffer,
        .dimensions = {
          .height = 1,
          .width = 128,
          .channel_cnt = 1,
          .byte_width = 1,
        },
        .stride = 128,
        .is_external = false,
      }, // 0
    }, // inputs
    .input_cnt = 1,
    .external_input_ndx = -1,
    .output_ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .packed_output_buf = NULL,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_7_7,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 35,
    .persistent_vars = {
      .count = 0,
    },

    .output_dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .byte_width = 1,
    },
    .output_dequant_mult = 1,
    .output_dequant_round = 17,
    .output_dequant_zp = 0,
    .output_stride = 128,
    .is_layer_model = true,
  },// .base
  .layer_ndx = 7,
  .input0_layer_ndx = 6,
  .input1_layer_ndx = -1,
};
#endif
#ifdef __cplusplus
}
#endif
/*********************************************************************************
 * Auto-generated nrf Axon compiled neural network model header file.
 * Model Name: tinyml_ad
 * Axon Neural Network Compiler Version: 0.2.0
 *********************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
#if (AXON_LAYER_TEST_START_LAYER<=8) && (AXON_LAYER_TEST_STOP_LAYER>=8)

const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_8_8[35] = {
// segment 0,length 34,Axon NN
0x1fff0022,
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
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_8_8 = {
  .base = {
    .compiler_version = 0x00000200,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = {
      {// 0
        .ptr = (int8_t*)nrf_axon_interlayer_buffer,
        .dimensions = {
          .height = 1,
          .width = 128,
          .channel_cnt = 1,
          .byte_width = 1,
        },
        .stride = 128,
        .is_external = false,
      }, // 0
    }, // inputs
    .input_cnt = 1,
    .external_input_ndx = -1,
    .output_ptr = (int8_t*)nrf_axon_interlayer_buffer,
    .packed_output_buf = NULL,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_8_8,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 35,
    .persistent_vars = {
      .count = 0,
    },

    .output_dimensions = {
      .height = 1,
      .width = 128,
      .channel_cnt = 1,
      .byte_width = 1,
    },
    .output_dequant_mult = 1,
    .output_dequant_round = 17,
    .output_dequant_zp = 0,
    .output_stride = 128,
    .is_layer_model = true,
  },// .base
  .layer_ndx = 8,
  .input0_layer_ndx = 7,
  .input1_layer_ndx = -1,
};
#endif
#ifdef __cplusplus
}
#endif
/*********************************************************************************
 * Auto-generated nrf Axon compiled neural network model header file.
 * Model Name: tinyml_ad
 * Axon Neural Network Compiler Version: 0.2.0
 *********************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif
#if (AXON_LAYER_TEST_START_LAYER<=9) && (AXON_LAYER_TEST_STOP_LAYER>=9)

const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_tinyml_ad_9_9[46] = {
// segment 0,length 45,Axon NN
0x1fff002d,
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
const nrf_axon_nn_compiled_model_layer_s model_tinyml_ad_9_9 = {
  .base = {
    .compiler_version = 0x00000200,
    .model_name = "tinyml_ad",
    .labels = NULL,
    .inputs = {
      {// 0
        .ptr = (int8_t*)nrf_axon_interlayer_buffer,
        .dimensions = {
          .height = 1,
          .width = 128,
          .channel_cnt = 1,
          .byte_width = 1,
        },
        .stride = 128,
        .is_external = false,
      }, // 0
    }, // inputs
    .input_cnt = 1,
    .external_input_ndx = -1,
    .output_ptr = (int8_t*)((uint8_t*)(nrf_axon_interlayer_buffer)+0x80),
    .packed_output_buf = NULL,

    .interlayer_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_TINYML_AD_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_tinyml_ad_9_9,

    .model_const_ptr = &axon_model_const_tinyml_ad,
    .model_const_size = sizeof(axon_model_const_tinyml_ad),
    .cmd_buffer_len = 46,
    .persistent_vars = {
      .count = 0,
    },

    .output_dimensions = {
      .height = 1,
      .width = 640,
      .channel_cnt = 1,
      .byte_width = 4,
    },
    .output_dequant_mult = 1,
    .output_dequant_round = 17,
    .output_dequant_zp = 0,
    .output_stride = 2560,
    .is_layer_model = true,
  },// .base
  .layer_ndx = 9,
  .input0_layer_ndx = 8,
  .input1_layer_ndx = -1,
};
#endif
#ifdef __cplusplus
}
#endif
#define MODEL_tinyml_ad_FIRST_COMPUTE_LAYER (0)
nrf_axon_nn_compiled_model_layer_s const *model_tinyml_ad_layer_list[] = {
	#if (AXON_LAYER_TEST_START_LAYER<=0) && (AXON_LAYER_TEST_STOP_LAYER>=0)
  &model_tinyml_ad_0_0,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=1) && (AXON_LAYER_TEST_STOP_LAYER>=1)
  &model_tinyml_ad_1_1,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=2) && (AXON_LAYER_TEST_STOP_LAYER>=2)
  &model_tinyml_ad_2_2,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=3) && (AXON_LAYER_TEST_STOP_LAYER>=3)
  &model_tinyml_ad_3_3,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=4) && (AXON_LAYER_TEST_STOP_LAYER>=4)
  &model_tinyml_ad_4_4,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=5) && (AXON_LAYER_TEST_STOP_LAYER>=5)
  &model_tinyml_ad_5_5,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=6) && (AXON_LAYER_TEST_STOP_LAYER>=6)
  &model_tinyml_ad_6_6,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=7) && (AXON_LAYER_TEST_STOP_LAYER>=7)
  &model_tinyml_ad_7_7,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=8) && (AXON_LAYER_TEST_STOP_LAYER>=8)
  &model_tinyml_ad_8_8,
#else
  NULL,
#endif
#if (AXON_LAYER_TEST_START_LAYER<=9) && (AXON_LAYER_TEST_STOP_LAYER>=9)
  &model_tinyml_ad_9_9,
#else
  NULL,
#endif

};

