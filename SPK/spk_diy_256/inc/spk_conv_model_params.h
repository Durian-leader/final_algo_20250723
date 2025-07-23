#pragma once
#include <stdint.h>
#include "riscv_nnfunctions.h"

#define SPK_CONV_IN_H      31
#define SPK_CONV_IN_W      12
#define SPK_CONV_IN_CH     1
#define SPK_CONV_OUT_CH    256
#define SPK_CONV_KH        7
#define SPK_CONV_KW        12
#define SPK_CONV_OUT_H     25
#define SPK_CONV_OUT_W     1

extern const int8_t  spk_conv_weights[];
extern const int32_t spk_conv_bias[];
extern const int32_t spk_conv_mult[];
extern const int32_t spk_conv_shift[];

static const nmsis_nn_dims spk_conv_input_dims  = {1, SPK_CONV_IN_H,  SPK_CONV_IN_W,  SPK_CONV_IN_CH};
static const nmsis_nn_dims spk_conv_filter_dims = {SPK_CONV_OUT_CH,  SPK_CONV_KH,    SPK_CONV_KW,    SPK_CONV_IN_CH};
static const nmsis_nn_dims spk_conv_bias_dims   = {SPK_CONV_OUT_CH,  1,          1,          1};
static const nmsis_nn_dims spk_conv_out_dims    = {1, SPK_CONV_OUT_H, SPK_CONV_OUT_W, SPK_CONV_OUT_CH};

static const nmsis_nn_conv_params spk_conv_params = {
    .input_offset  = 0,
    .output_offset = -128,
    .stride        = {1,1},
    .padding       = {0,0},
    .dilation      = {1,1},
    .activation    = {128,127}
};

static const nmsis_nn_per_channel_quant_params spk_conv_pc_quant = {
    .multiplier = (int32_t*)spk_conv_mult,
    .shift      = (int32_t*)spk_conv_shift
};