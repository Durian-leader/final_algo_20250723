#pragma once
#include <stdint.h>
#include "riscv_nnfunctions.h"

#define VAD_FC_IN_DIM   (4)
#define VAD_FC_OUT_DIM  (2)

extern const int8_t  vad_fc_weights[];
extern const int32_t vad_fc_bias[];
extern const int32_t vad_fc_mult[];
extern const int32_t vad_fc_shift[];

static const nmsis_nn_dims vad_fc_input_dims  = {1,1,1,VAD_FC_IN_DIM};
static const nmsis_nn_dims vad_fc_filter_dims = {VAD_FC_OUT_DIM, 1, 1, VAD_FC_IN_DIM};
static const nmsis_nn_dims vad_fc_bias_dims   = {VAD_FC_OUT_DIM, 1, 1, 1};
static const nmsis_nn_dims vad_fc_out_dims    = {1, 1, 1, VAD_FC_OUT_DIM};

static const nmsis_nn_fc_params vad_fc_params = {
    .input_offset  = 128,
    .filter_offset = 0,
    .output_offset = 127,
    .activation = {-128, 127}
};

static const nmsis_nn_per_channel_quant_params vad_fc_pc_quant = {
    .multiplier = (int32_t *)vad_fc_mult,
    .shift      = (int32_t *)vad_fc_shift
};