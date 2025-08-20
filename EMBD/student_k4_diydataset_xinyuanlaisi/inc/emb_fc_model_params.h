#pragma once
#include <stdint.h>
#include "riscv_nnfunctions.h"

#define EMB_FC_IN_DIM   (256)
#define EMB_FC_OUT_DIM  (192)

extern const int8_t  emb_fc_weights[];
extern const int32_t emb_fc_bias[];
extern const int32_t emb_fc_mult[];
extern const int32_t emb_fc_shift[];

static const nmsis_nn_dims emb_fc_input_dims  = {1,1,1,EMB_FC_IN_DIM};
static const nmsis_nn_dims emb_fc_filter_dims = {EMB_FC_OUT_DIM, 1, 1, EMB_FC_IN_DIM};
static const nmsis_nn_dims emb_fc_bias_dims   = {EMB_FC_OUT_DIM, 1, 1, 1};
static const nmsis_nn_dims emb_fc_out_dims    = {1, 1, 1, EMB_FC_OUT_DIM};

static const nmsis_nn_fc_params emb_fc_params = {
    .input_offset  = 128,
    .filter_offset = 0,
    .output_offset = 10,
    .activation = {-128, 127}
};

static const nmsis_nn_per_channel_quant_params emb_fc_pc_quant = {
    .multiplier = (int32_t *)emb_fc_mult,
    .shift      = (int32_t *)emb_fc_shift
};