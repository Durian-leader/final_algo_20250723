# -*- coding: utf-8 -*-
from pathlib import Path
import math, textwrap, numpy as np, flatbuffers

from tflite import (
    Model, BuiltinOperator,
    Conv2DOptions, Padding, ActivationFunctionType
)

# ────────────────────────────────────────────────
# helper：将浮点 scale 转 mantissa(Q31)+shift，满足
#   result = (acc * mantissa) >> (31 - shift)
# shift >0 ⇒ 左移；若你用旧版 NMSIS-NN，改宏即可
# ────────────────────────────────────────────────
def quantize_scale(double_scale: float):
    if double_scale == 0:
        return 0, 0
    mant, exp = math.frexp(double_scale)         # double_scale = mant * 2**exp, 0.5<=mant<1
    mant_q31 = int(round(mant * (1 << 31)))
    if mant_q31 == (1 << 31):                    # 处理边界
        mant_q31 //= 2
        exp += 1
    return mant_q31, -exp                        # shift = -exp  (2**exp = 1<<(-shift))

# ────────────────────────────────────────────────
# 核心导出函数
# ────────────────────────────────────────────────
def export_conv_params(
    tflite_path: str | Path,
    output_dir: str | Path,
    subgraph_index: int,
    conv_op_index: int,
    prefix: str = "conv",
    SHIFT_LEFT_POSITIVE: bool = True,            # NMSIS-NN ≥ 1.11 建议为 True
):
    tflite_path = Path(tflite_path)
    output_dir  = Path(output_dir)
    out_inc = output_dir / "inc" / f"{prefix}_model_params.h"
    out_src = output_dir / "src" / f"{prefix}_model_data.c"
    out_inc.parent.mkdir(parents=True, exist_ok=True)
    out_src.parent.mkdir(parents=True, exist_ok=True)

    # ── 解析 flatbuffer ──────────────────────────
    buf   = tflite_path.read_bytes()
    model = Model.GetRootAsModel(buf, 0)
    sub   = model.Subgraphs(subgraph_index)
    op    = sub.Operators(conv_op_index)

    # 类型检查
    if model.OperatorCodes(op.OpcodeIndex()).BuiltinCode() != BuiltinOperator.CONV_2D:
        raise ValueError("指定的 operator 不是 Conv2D")

    t_in  = sub.Tensors(op.Inputs(0))
    t_w   = sub.Tensors(op.Inputs(1))
    has_bias = op.InputsLength() >= 3
    t_b   = sub.Tensors(op.Inputs(2)) if has_bias else None
    t_out = sub.Tensors(op.Outputs(0))

    # ── 权重 / bias 数据 ─────────────────────────
    w_raw = model.Buffers(t_w.Buffer()).DataAsNumpy().view(np.int8)
    COUT, KH, KW, CIN = (t_w.Shape(i) for i in range(t_w.ShapeLength()))
    weight = w_raw.reshape(COUT, KH, KW, CIN)          # OHWI

    if has_bias:
        b_raw = model.Buffers(t_b.Buffer()).DataAsNumpy().view(np.int32)
    else:                                              # 无 bias 时生成 0 数组
        b_raw = np.zeros(COUT, dtype=np.int32)

    # ── 量化参数：per-tensor / per-channel ────────
    in_scale  = t_in.Quantization().Scale(0)
    out_scale = t_out.Quantization().Scale(0)

    w_q = t_w.Quantization()
    if w_q.ScaleLength() == 1:     # per-tensor
        w_scales = [w_q.Scale(0)] * COUT
    else:                          # per-channel
        w_scales = [w_q.Scale(i) for i in range(COUT)]

    mult, shift = [], []
    for s in w_scales:
        m, sh = quantize_scale(in_scale * s / out_scale)
        mult.append(m)
        shift.append(sh if SHIFT_LEFT_POSITIVE else -sh)
    mult  = np.array(mult,  dtype=np.int32)
    shift = np.array(shift, dtype=np.int32)

    # ── Conv2DOptions：stride / dilation / padding ─
    opts = Conv2DOptions()
    opts.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)

    stride_h, stride_w     = opts.StrideH(), opts.StrideW()
    dilation_h             = opts.DilationHFactor() or 1
    dilation_w             = opts.DilationWFactor() or 1
    is_same_pad            = (opts.Padding() == Padding.SAME)

    in_h, in_w = t_in.Shape(1), t_in.Shape(2)

    if is_same_pad:
        out_h = math.ceil(in_h / stride_h)
        out_w = math.ceil(in_w / stride_w)
        pad_h_total = max((out_h - 1) * stride_h +
                          (KH - 1) * dilation_h + 1 - in_h, 0)
        pad_w_total = max((out_w - 1) * stride_w +
                          (KW - 1) * dilation_w + 1 - in_w, 0)
        pad_top  = pad_h_total // 2
        pad_left = pad_w_total // 2
    else:  # VALID
        pad_top = pad_left = 0
        out_h = (in_h - (KH - 1) * dilation_h - 1) // stride_h + 1
        out_w = (in_w - (KW - 1) * dilation_w - 1) // stride_w + 1

    # ── fused activation → output clamp ───────────
    act_type = opts.FusedActivationFunction()
    act_min, act_max = -128, 127
    zp_out = t_out.Quantization().ZeroPoint(0)

    if act_type == ActivationFunctionType.RELU:
        act_min = max(0, -zp_out)
    elif act_type == ActivationFunctionType.RELU6:
        act_min = max(0, -zp_out)
        act_max = int(round(6.0 / out_scale)) - zp_out

    # ── C 数组辅助生成器 ─────────────────────────
    def c_array(name, data, ctype):
        flat = data.flatten()
        vals = ",".join(str(int(x)) for x in flat)
        return f"const {ctype} {name}[{len(flat)}] = {{{vals}}};"

    # ── 写 .h ────────────────────────────────────
    up = prefix.upper()
    out_inc.write_text(textwrap.dedent(f"""
        #pragma once
        #include <stdint.h>
        #include "riscv_nnfunctions.h"

        #define {up}_IN_H      {in_h}
        #define {up}_IN_W      {in_w}
        #define {up}_IN_CH     {CIN}
        #define {up}_OUT_CH    {COUT}
        #define {up}_KH        {KH}
        #define {up}_KW        {KW}
        #define {up}_OUT_H     {out_h}
        #define {up}_OUT_W     {out_w}

        extern const int8_t  {prefix}_weights[];
        extern const int32_t {prefix}_bias[];
        extern const int32_t {prefix}_mult[];
        extern const int32_t {prefix}_shift[];

        static const nmsis_nn_dims {prefix}_input_dims  = {{1, {up}_IN_H,  {up}_IN_W,  {up}_IN_CH}};
        static const nmsis_nn_dims {prefix}_filter_dims = {{{up}_OUT_CH,  {up}_KH,    {up}_KW,    {up}_IN_CH}};
        static const nmsis_nn_dims {prefix}_bias_dims   = {{{up}_OUT_CH,  1,          1,          1}};
        static const nmsis_nn_dims {prefix}_out_dims    = {{1, {up}_OUT_H, {up}_OUT_W, {up}_OUT_CH}};

        static const nmsis_nn_conv_params {prefix}_params = {{
            .input_offset  = {-int(t_in.Quantization().ZeroPoint(0))},
            .output_offset = { int(zp_out)},
            .stride        = {{{stride_w},{stride_h}}},
            .padding       = {{{pad_left},{pad_top}}},
            .dilation      = {{{dilation_w},{dilation_h}}},
            .activation    = {{{act_min},{act_max}}}
        }};

        static const nmsis_nn_per_channel_quant_params {prefix}_pc_quant = {{
            .multiplier = (int32_t*){prefix}_mult,
            .shift      = (int32_t*){prefix}_shift
        }};
    """).strip())

    # ── 写 .c ────────────────────────────────────
    out_src.write_text(textwrap.dedent(f"""
        #include "{prefix}_model_params.h"

        {c_array(f"{prefix}_weights", weight, "int8_t")}
        {c_array(f"{prefix}_bias",    b_raw,   "int32_t")}
        {c_array(f"{prefix}_mult",    mult,    "int32_t")}
        {c_array(f"{prefix}_shift",   shift,   "int32_t")}
    """).strip())

    try:
        rel_inc = out_inc.relative_to(Path.cwd())
        rel_src = out_src.relative_to(Path.cwd())
        print(f"✅ 生成成功 → {rel_inc} | {rel_src}")
    except ValueError:
        print(f"✅ 生成成功 → {out_inc} | {out_src}")


# from pathlib import Path
# import flatbuffers, textwrap, numpy as np, math
# from tflite import Model, BuiltinOperator

# def quantize_scale(double_scale: float):
#     """
#     将浮点 scale 转换为定点表示（mantissa + shift）
#     """
#     if double_scale == 0:
#         return 0, 0
#     mant, exp = math.frexp(double_scale)
#     mant_q31 = int(round(mant * (1 << 31)))
#     if mant_q31 == (1 << 31):
#         mant_q31 //= 2
#         exp += 1
#     return mant_q31, -exp

# def export_conv_params(
#     tflite_path: str | Path,
#     output_dir: str | Path,
#     subgraph_index: int,
#     conv_op_index: int,
#     prefix: str = "conv"
# ):
#     """
#     导出 TFLite 中指定 Conv2D 层的权重/bias/量化参数等，生成 .h / .c 文件
    
#     Args:
#         tflite_path: TFLite 文件路径
#         output_dir: 输出目录（会自动在其中生成 inc/ 和 src/）
#         subgraph_index: subgraph 索引
#         conv_op_index: Conv2D op 在 subgraph 中的索引
#         prefix: 变量名和宏定义的前缀，默认为 "conv"
#     """
#     tflite_path = Path(tflite_path)
#     output_dir = Path(output_dir)
#     out_inc = output_dir / "inc" / f"{prefix}_model_params.h"
#     out_src = output_dir / "src" / f"{prefix}_model_data.c"
    
#     buf = tflite_path.read_bytes()
#     model = Model.GetRootAsModel(buf, 0)
#     subgraph = model.Subgraphs(subgraph_index)
#     op = subgraph.Operators(conv_op_index)
    
#     # 确认 op 是 Conv2D
#     opcode_idx = op.OpcodeIndex()
#     builtin = model.OperatorCodes(opcode_idx).BuiltinCode()
#     assert builtin == BuiltinOperator.CONV_2D, "该 operator 不是 Conv2D！"
    
#     # Tensor 索引 & 对象
#     t_in  = subgraph.Tensors(op.Inputs(0))
#     t_w   = subgraph.Tensors(op.Inputs(1))
#     t_b   = subgraph.Tensors(op.Inputs(2))
#     t_out = subgraph.Tensors(op.Outputs(0))
    
#     # 读取权重和 bias
#     w_raw = model.Buffers(t_w.Buffer()).DataAsNumpy().view(np.int8)
#     b_raw = model.Buffers(t_b.Buffer()).DataAsNumpy().view(np.int32)
    
#     COUT, KH, KW, CIN = (t_w.Shape(i) for i in range(t_w.ShapeLength()))
#     weight = w_raw.reshape(COUT, KH, KW, CIN)  # OHWI 格式
    
#     # 量化参数
#     in_scale  = t_in.Quantization().Scale(0)
#     out_scale = t_out.Quantization().Scale(0)
#     w_scales  = [t_w.Quantization().Scale(i) for i in range(COUT)]
    
#     mult, shift = [], []
#     for s in w_scales:
#         m, sh = quantize_scale(in_scale * s / out_scale)
#         mult.append(m)
#         shift.append(sh)
#     mult = np.array(mult, dtype=np.int32)
#     shift = np.array(shift, dtype=np.int32)
    
#     # 生成 C 数组
#     def c_array(name, data, ctype):
#         flat = data.flatten()
#         vals = ','.join(str(x) for x in flat)
#         return f"const {ctype} {name}[{len(flat)}] = {{{vals}}};"
    
#     # 创建目录
#     out_inc.parent.mkdir(parents=True, exist_ok=True)
#     out_src.parent.mkdir(parents=True, exist_ok=True)
    
#     # 转换为大写的前缀用于宏定义
#     prefix_upper = prefix.upper()
    
#     # 写入 .h 文件
#     out_inc.write_text(textwrap.dedent(f"""
#         #pragma once
#         #include <stdint.h>
#         #include "riscv_nnfunctions.h"

#         #define {prefix_upper}_IN_H     {t_in.Shape(1)}
#         #define {prefix_upper}_IN_W     {t_in.Shape(2)}
#         #define {prefix_upper}_IN_CH    {CIN}
#         #define {prefix_upper}_OUT_CH   {COUT}
#         #define {prefix_upper}_KH       {KH}
#         #define {prefix_upper}_KW       {KW}

#         extern const int8_t  {prefix}_weights[];
#         extern const int32_t {prefix}_bias[];
#         extern const int32_t {prefix}_mult[];
#         extern const int32_t {prefix}_shift[];

#         static const nmsis_nn_dims {prefix}_input_dims  = {{1, {prefix_upper}_IN_H, {prefix_upper}_IN_W, {prefix_upper}_IN_CH}};
#         static const nmsis_nn_dims {prefix}_filter_dims = {{{prefix_upper}_OUT_CH, {prefix_upper}_KH, {prefix_upper}_KW, {prefix_upper}_IN_CH}};
#         static const nmsis_nn_dims {prefix}_bias_dims   = {{{prefix_upper}_OUT_CH, 1, 1, 1}};
#         static const nmsis_nn_dims {prefix}_out_dims    = {{1, {prefix_upper}_IN_H, {prefix_upper}_IN_W, {prefix_upper}_OUT_CH}}; // stride=1,padding=same

#         static const nmsis_nn_conv_params {prefix}_params = {{
#             .input_offset  = {-int(t_in.Quantization().ZeroPoint(0))},
#             .output_offset = { int(t_out.Quantization().ZeroPoint(0))},
#             .stride   = {{1,1}},
#             .padding  = {{{KH//2},{KW//2}}},
#             .dilation = {{1,1}},
#             .activation = {{-128,127}}
#         }};

#         static const nmsis_nn_per_channel_quant_params {prefix}_pc_quant = {{
#             .multiplier = (int32_t *){prefix}_mult,
#             .shift      = (int32_t *){prefix}_shift
#         }};
#     """).strip())
    
#     # 写入 .c 文件
#     out_src.write_text(textwrap.dedent(f"""
#         #include "{prefix}_model_params.h"

#         {c_array(f"{prefix}_weights", weight, "int8_t")}
#         {c_array(f"{prefix}_bias",    b_raw,   "int32_t")}
#         {c_array(f"{prefix}_mult",    mult,    "int32_t")}
#         {c_array(f"{prefix}_shift",   -shift,   "int32_t")}
#     """).strip())
    
#     print(f"✅ 已生成: {out_inc} 和 {out_src}")

from tflite import BuiltinOperator

def find_all_conv2d_ops(model):
    """
    查找模型中所有 Conv2D 层的位置。
    
    Args:
        model: flatbuffers 加载的 TFLite 模型对象
    
    Returns:
        List of (subgraph_index, operator_index) 索引对
    """
    result = []
    for subgraph_idx in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(subgraph_idx)
        for op_idx in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(op_idx)
            opcode_idx = op.OpcodeIndex()
            builtin = model.OperatorCodes(opcode_idx).BuiltinCode()
            if builtin == BuiltinOperator.CONV_2D:
                result.append((subgraph_idx, op_idx))
    return result


# 示例调用
if __name__ == "__main__":
    # 示例：为不同模型使用不同前缀
    export_conv_params(
        tflite_path="build/conv_model_int8.tflite",
        output_dir="./me",
        subgraph_index=0,
        conv_op_index=0,
        prefix="model1_conv"  # 自定义前缀
    )
    
    # 另一个模型的示例
    export_conv_params(
        tflite_path="build/another_model_int8.tflite",
        output_dir="./me",
        subgraph_index=0,
        conv_op_index=0,
        prefix="model2_conv"  # 不同的前缀避免冲突
    )


# from pathlib import Path
# import flatbuffers, textwrap, numpy as np, math
# from tflite import Model, BuiltinOperator

# def quantize_scale(double_scale: float):
#     """
#     将浮点 scale 转换为定点表示（mantissa + shift）
#     """
#     if double_scale == 0:
#         return 0, 0
#     mant, exp = math.frexp(double_scale)
#     mant_q31 = int(round(mant * (1 << 31)))
#     if mant_q31 == (1 << 31):
#         mant_q31 //= 2
#         exp += 1
#     return mant_q31, -exp

# def export_conv_params(
#     tflite_path: str | Path,
#     output_dir: str | Path,
#     subgraph_index: int,
#     conv_op_index: int
# ):
#     """
#     导出 TFLite 中指定 Conv2D 层的权重/bias/量化参数等，生成 .h / .c 文件
    
#     Args:
#         tflite_path: TFLite 文件路径
#         output_dir: 输出目录（会自动在其中生成 inc/ 和 src/）
#         subgraph_index: subgraph 索引
#         conv_op_index: Conv2D op 在 subgraph 中的索引
#     """
#     tflite_path = Path(tflite_path)
#     output_dir = Path(output_dir)
#     out_inc = output_dir / "inc" / "conv_model_params.h"
#     out_src = output_dir / "src" / "conv_model_data.c"
    
#     buf = tflite_path.read_bytes()
#     model = Model.GetRootAsModel(buf, 0)
#     subgraph = model.Subgraphs(subgraph_index)
#     op = subgraph.Operators(conv_op_index)
    
#     # 确认 op 是 Conv2D
#     opcode_idx = op.OpcodeIndex()
#     builtin = model.OperatorCodes(opcode_idx).BuiltinCode()
#     assert builtin == BuiltinOperator.CONV_2D, "该 operator 不是 Conv2D！"
    
#     # Tensor 索引 & 对象
#     t_in  = subgraph.Tensors(op.Inputs(0))
#     t_w   = subgraph.Tensors(op.Inputs(1))
#     t_b   = subgraph.Tensors(op.Inputs(2))
#     t_out = subgraph.Tensors(op.Outputs(0))
    
#     # 读取权重和 bias
#     w_raw = model.Buffers(t_w.Buffer()).DataAsNumpy().view(np.int8)
#     b_raw = model.Buffers(t_b.Buffer()).DataAsNumpy().view(np.int32)
    
#     COUT, KH, KW, CIN = (t_w.Shape(i) for i in range(t_w.ShapeLength()))
#     weight = w_raw.reshape(COUT, KH, KW, CIN)  # OHWI 格式
    
#     # 量化参数
#     in_scale  = t_in.Quantization().Scale(0)
#     out_scale = t_out.Quantization().Scale(0)
#     w_scales  = [t_w.Quantization().Scale(i) for i in range(COUT)]
    
#     mult, shift = [], []
#     for s in w_scales:
#         m, sh = quantize_scale(in_scale * s / out_scale)
#         mult.append(m)
#         shift.append(sh)
#     mult = np.array(mult, dtype=np.int32)
#     shift = np.array(shift, dtype=np.int32)
    
#     # 生成 C 数组
#     def c_array(name, data, ctype):
#         flat = data.flatten()
#         vals = ','.join(str(x) for x in flat)
#         return f"const {ctype} {name}[{len(flat)}] = {{{vals}}};"
    
#     # 创建目录
#     out_inc.parent.mkdir(parents=True, exist_ok=True)
#     out_src.parent.mkdir(parents=True, exist_ok=True)
    
#     # 写入 .h 文件
#     out_inc.write_text(textwrap.dedent(f"""
#         #pragma once
#         #include <stdint.h>
#         #include "riscv_nnfunctions.h"

#         #define CONV_IN_H     {t_in.Shape(1)}
#         #define CONV_IN_W     {t_in.Shape(2)}
#         #define CONV_IN_CH    {CIN}
#         #define CONV_OUT_CH   {COUT}
#         #define CONV_KH       {KH}
#         #define CONV_KW       {KW}

#         extern const int8_t  conv_weights[];
#         extern const int32_t conv_bias[];
#         extern const int32_t conv_mult[];
#         extern const int32_t conv_shift[];

#         static const nmsis_nn_dims conv_input_dims  = {{1, CONV_IN_H, CONV_IN_W, CONV_IN_CH}};
#         static const nmsis_nn_dims conv_filter_dims = {{CONV_OUT_CH, CONV_KH, CONV_KW, CONV_IN_CH}};
#         static const nmsis_nn_dims conv_bias_dims   = {{CONV_OUT_CH, 1, 1, 1}};
#         static const nmsis_nn_dims conv_out_dims    = {{1, CONV_IN_H, CONV_IN_W, CONV_OUT_CH}}; // stride=1,padding=same

#         static const nmsis_nn_conv_params conv_params = {{
#             .input_offset  = {-int(t_in.Quantization().ZeroPoint(0))},
#             .output_offset = { int(t_out.Quantization().ZeroPoint(0))},
#             .stride   = {{1,1}},
#             .padding  = {{{KH//2},{KW//2}}},
#             .dilation = {{1,1}},
#             .activation = {{-128,127}}
#         }};

#         static const nmsis_nn_per_channel_quant_params conv_pc_quant = {{
#             .multiplier = (int32_t *)conv_mult,
#             .shift      = (int32_t *)conv_shift
#         }};
#     """).strip())
    
#     # 写入 .c 文件
#     out_src.write_text(textwrap.dedent(f"""
#         #include "conv_model_params.h"

#         {c_array("conv_weights", weight, "int8_t")}
#         {c_array("conv_bias",    b_raw,   "int32_t")}
#         {c_array("conv_mult",    mult,    "int32_t")}
#         {c_array("conv_shift",   -shift,   "int32_t")}
#     """).strip())
    
#     print(f"✅ 已生成: {out_inc} 和 {out_src}")

# from tflite import BuiltinOperator

# def find_all_conv2d_ops(model):
#     """
#     查找模型中所有 Conv2D 层的位置。
    
#     Args:
#         model: flatbuffers 加载的 TFLite 模型对象
    
#     Returns:
#         List of (subgraph_index, operator_index) 索引对
#     """
#     result = []
#     for subgraph_idx in range(model.SubgraphsLength()):
#         subgraph = model.Subgraphs(subgraph_idx)
#         for op_idx in range(subgraph.OperatorsLength()):
#             op = subgraph.Operators(op_idx)
#             opcode_idx = op.OpcodeIndex()
#             builtin = model.OperatorCodes(opcode_idx).BuiltinCode()
#             if builtin == BuiltinOperator.CONV_2D:
#                 result.append((subgraph_idx, op_idx))
#     return result


# # 示例调用
# if __name__ == "__main__":
#     export_conv_params(
#         tflite_path="build/conv_model_int8.tflite",
#         output_dir="./me",
#         subgraph_index=0,
#         conv_op_index=0
#     )
