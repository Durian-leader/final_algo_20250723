#pragma once
#include <stdint.h>
#include "riscv_nnfunctions.h"

#define VAD_CONV_IN_H      31
#define VAD_CONV_IN_W      12
#define VAD_CONV_IN_CH     1
#define VAD_CONV_OUT_CH    4
#define VAD_CONV_KH        3
#define VAD_CONV_KW        3
#define VAD_CONV_OUT_H     31
#define VAD_CONV_OUT_W     12

extern const int8_t  vad_conv_weights[];
extern const int32_t vad_conv_bias[];
extern const int32_t vad_conv_mult[];
extern const int32_t vad_conv_shift[];

static const nmsis_nn_dims vad_conv_input_dims  = {1, VAD_CONV_IN_H,  VAD_CONV_IN_W,  VAD_CONV_IN_CH};
static const nmsis_nn_dims vad_conv_filter_dims = {VAD_CONV_OUT_CH,  VAD_CONV_KH,    VAD_CONV_KW,    VAD_CONV_IN_CH};
static const nmsis_nn_dims vad_conv_bias_dims   = {VAD_CONV_OUT_CH,  1,          1,          1};
static const nmsis_nn_dims vad_conv_out_dims    = {1, VAD_CONV_OUT_H, VAD_CONV_OUT_W, VAD_CONV_OUT_CH};

static const nmsis_nn_conv_params vad_conv_params = {
    .input_offset  = 34,
    .output_offset = -128,
    .stride        = {1,1},
    .padding       = {1,1},
    .dilation      = {1,1},
    .activation    = {128,127}
};

static const nmsis_nn_per_channel_quant_params vad_conv_pc_quant = {
    .multiplier = (int32_t*)vad_conv_mult,
    .shift      = (int32_t*)vad_conv_shift
};