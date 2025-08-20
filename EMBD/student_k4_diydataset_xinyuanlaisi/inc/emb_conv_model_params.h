#pragma once
#include <stdint.h>
#include "riscv_nnfunctions.h"

#define EMB_CONV_IN_H      31
#define EMB_CONV_IN_W      12
#define EMB_CONV_IN_CH     1
#define EMB_CONV_OUT_CH    256
#define EMB_CONV_KH        7
#define EMB_CONV_KW        12
#define EMB_CONV_OUT_H     25
#define EMB_CONV_OUT_W     1

extern const int8_t  emb_conv_weights[];
extern const int32_t emb_conv_bias[];
extern const int32_t emb_conv_mult[];
extern const int32_t emb_conv_shift[];

static const nmsis_nn_dims emb_conv_input_dims  = {1, EMB_CONV_IN_H,  EMB_CONV_IN_W,  EMB_CONV_IN_CH};
static const nmsis_nn_dims emb_conv_filter_dims = {EMB_CONV_OUT_CH,  EMB_CONV_KH,    EMB_CONV_KW,    EMB_CONV_IN_CH};
static const nmsis_nn_dims emb_conv_bias_dims   = {EMB_CONV_OUT_CH,  1,          1,          1};
static const nmsis_nn_dims emb_conv_out_dims    = {1, EMB_CONV_OUT_H, EMB_CONV_OUT_W, EMB_CONV_OUT_CH};

static const nmsis_nn_conv_params emb_conv_params = {
    .input_offset  = -8,
    .output_offset = -128,
    .stride        = {1,1},
    .padding       = {0,0},
    .dilation      = {1,1},
    .activation    = {128,127}
};

static const nmsis_nn_per_channel_quant_params emb_conv_pc_quant = {
    .multiplier = (int32_t*)emb_conv_mult,
    .shift      = (int32_t*)emb_conv_shift
};