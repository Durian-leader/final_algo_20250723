#pragma once
#include <stdint.h>
#include "riscv_nnfunctions.h"

#define SPK_FC_IN_DIM   (1024)
#define SPK_FC_OUT_DIM  (5)

extern const int8_t  spk_fc_weights[];
extern const int32_t spk_fc_bias[];
extern const int32_t spk_fc_mult[];
extern const int32_t spk_fc_shift[];

static const nmsis_nn_dims spk_fc_input_dims  = {1,1,1,SPK_FC_IN_DIM};
static const nmsis_nn_dims spk_fc_filter_dims = {SPK_FC_OUT_DIM, 1, 1, SPK_FC_IN_DIM};
static const nmsis_nn_dims spk_fc_bias_dims   = {SPK_FC_OUT_DIM, 1, 1, 1};
static const nmsis_nn_dims spk_fc_out_dims    = {1, 1, 1, SPK_FC_OUT_DIM};

static const nmsis_nn_fc_params spk_fc_params = {
    .input_offset  = 128,
    .filter_offset = 0,
    .output_offset = 52,
    .activation = {-128, 127}
};

static const nmsis_nn_per_channel_quant_params spk_fc_pc_quant = {
    .multiplier = (int32_t *)spk_fc_mult,
    .shift      = (int32_t *)spk_fc_shift
};