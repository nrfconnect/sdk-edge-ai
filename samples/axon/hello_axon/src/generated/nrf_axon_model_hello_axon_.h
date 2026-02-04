/*********************************************************************************
 * Auto-generated nrf Axon compiled neural network model header file.
 * Model Name: hello_axon
 * Axon Neural Network Compiler Version: 0.1.0
 *********************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif

#define NRF_AXON_MODEL_HELLO_AXON_MAX_IL_BUFFER_USED 16
#define NRF_AXON_MODEL_HELLO_AXON_MAX_PSUM_BUFFER_USED 0

#if AXON_COMPILE_TIME_BUFFER_CHECK
static_assert(NRF_AXON_MODEL_HELLO_AXON_MAX_IL_BUFFER_USED < sizeof(nrf_axon_interlayer_buffer), "nrf_axon_interlayer_buffer TOO SMALL!!!!\n");

#endif
// size of axon_model_const_hello_axon: 420
const static struct {
	int8_t l00_weights[16];
	int32_t l00_biasp[16];
	int8_t l01_weights[256];
	int32_t l01_biasp[16];
	int8_t l02_weights[16];
	int32_t l02_biasp[1];

} axon_model_const_hello_axon = {
	.l00_weights = {117,28,17,-31,12,-127,-91,66,-2,-43,-44,-78,97,120,25,-33,},
	.l00_biasp = {14982,6519,-301,-3968,4727,-16256,-11648,10195,-256,-5504,2930,-9984,14255,12647,-844,-4224,},
	.l01_weights = {-18,-4,0,-20,5,23,-17,-20,-26,-8,3,1,0,-6,-8,-11,-36,-21,39,20,-15,-34,-30,-37,-16,-34,49,6,2,-26,-18,-7,0,22,7,-32,-2,-1,-23,6,-25,-17,-127,27,24,-22,-55,1,15,0,-38,-9,14,-20,19,31,4,19,-76,-26,-3,6,-71,-32,13,-20,-16,-34,-21,-9,5,38,26,-28,111,26,-22,30,53,-33,26,-13,-15,25,15,3,27,-31,-34,19,-10,25,-1,-10,27,24,-16,28,-38,27,27,32,-27,26,-11,-1,-106,11,0,1,-51,-34,13,-10,22,-29,-19,-4,14,-23,-6,-21,92,-4,29,2,91,-30,-31,-11,21,-20,-12,0,19,5,-20,12,29,20,14,-25,11,-12,25,0,-41,5,39,2,21,-22,-22,2,-101,0,12,-6,-24,-22,-3,0,20,-3,11,2,-17,-18,6,-18,1,13,6,-26,-9,17,-9,9,-8,-15,33,-1,14,-13,-20,18,38,29,-14,-23,40,24,-32,-5,-13,-12,5,29,29,-5,-3,30,-4,17,-24,7,9,3,18,-14,54,-5,-36,28,-7,-17,-13,-25,111,12,29,0,69,-3,14,-16,11,25,26,-6,-32,25,31,19,54,28,18,-21,59,12,-76,-53,-26,19,-6,-21,-15,6,28,-6,24,-27,-21,-53,12,-12,},
	.l01_biasp = {-13568,-19019,-25096,-19795,13297,9856,-14183,12926,1295,-14380,-2745,14262,3968,23596,29899,-27702,},
	.l02_weights = {33,-91,-117,-54,94,29,-50,66,-99,-50,31,-80,-33,84,47,-127,},
	.l02_biasp = {-44788,},
};


const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_hello_axon[69] = {
// segment 0,length 68,Axon NN
0x1fff0044,
0x02000080,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00330001,
0x02000090,0x00100001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_hello_axon.l00_weights,0x00330001,
0x050000a0,0x00010010,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_hello_axon.l00_biasp,0x00050040,0x00010010,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)nrf_axon_interlayer_buffer,0x00030010,
0x000000bc,0x00000000,
0x040000c8,0x00000000,0x00000000,0x03010000,0x00011703,0x00000205,
0x01000180,0x00000003,0x00000000,
0x010001a4,0x00000000,0x000116d5,
0x010001c8,0xc0000000,0xc0000000,
0x000000f0,0x00000100,
0x00000080,0x00010010,
0x00000088,0x00330010,
0x01000090,0x00100010,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_hello_axon.l01_weights,
0x000000a4,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_hello_axon.l01_biasp,
0x000001a8,0x00035a4e,
0x000000f0,0x00000100,
0x01000090,0x00010010,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_hello_axon.l02_weights,
0x030000a0,0x00010001,(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)axon_model_const_hello_axon.l02_biasp,0x00050004,0x00010001,
0x000000b4,0x00030001,
0x000000d4,0x00011c03,
0x01000180,0x00000002,0x80000000,
0x000001a8,0x00183066,
0x000001cc,0x40000000,
0x000000f0,0x00000100,
};
#define NRF_AXON_MODEL_HELLO_AXON_PACKED_OUTPUT_SIZE 4

#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
  uint32_t axon_model_hello_axon_packed_output_buf[NRF_AXON_MODEL_HELLO_AXON_PACKED_OUTPUT_SIZE/sizeof(uint32_t)];
#endif
const nrf_axon_nn_compiled_model_s model_hello_axon = {
    .compiler_version = 0x00000100,
    .model_name = "hello_axon",
    .labels = NULL,
    .inputs = {
      {// 0
        .ptr = (int8_t*)nrf_axon_interlayer_buffer,
        .dimensions = {
          .height = 1,
          .width = 1,
          .channel_cnt = 1,
          .byte_width = 1,
        },
        .quant_mult = 21335090,
        .stride = 1,
        .quant_round = 19,
        .quant_zp = -128,
        .is_external = true,
      }, // 0
    }, // inputs
    .input_cnt = 1,
    .external_input_ndx = 0,
    .output_ptr = (int8_t*)nrf_axon_interlayer_buffer,
#if NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER
    .packed_output_buf = (int8_t*)axon_model_hello_axon_packed_output_buf,
#else
    .packed_output_buf = NULL,
#endif

    .interlayer_buffer_needed = NRF_AXON_MODEL_HELLO_AXON_MAX_IL_BUFFER_USED,
    .psum_buffer_needed = NRF_AXON_MODEL_HELLO_AXON_MAX_PSUM_BUFFER_USED,
    .cmd_buffer_ptr = cmd_buffer_hello_axon,

    .model_const_ptr = &axon_model_const_hello_axon,
    .model_const_size = sizeof(axon_model_const_hello_axon),
    .cmd_buffer_len = 69,
    .persistent_vars = {
      .count = 0,
    },

    .output_dimensions = {
      .height = 1,
      .width = 1,
      .channel_cnt = 1,
      .byte_width = 1,
    },
    .output_dequant_mult = 4548375,
    .output_dequant_round = 29,
    .output_dequant_zp = 4,
    .output_stride = 4,
    .is_layer_model = false,
};
#ifdef __cplusplus
}
#endif
