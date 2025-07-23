from pathlib import Path
import flatbuffers
from tflite import Model
import numpy as np
import textwrap, math

def quantize_scale(double_scale: float):
    """
    将浮点 scale 转换为定点表示（mantissa + shift）。
    """
    if double_scale == 0:
        return 0, 0
    mant, exp = math.frexp(double_scale)
    mant_q31 = int(round(mant * (1 << 31)))
    if mant_q31 == (1 << 31):
        mant_q31 //= 2
        exp += 1
    return mant_q31, -exp

def export_fc_params(
    tflite_path: str | Path,
    output_dir: str | Path,
    subgraph_index: int,
    operator_index: int,
    prefix: str = "fc"
):
    """
    从 TFLite 模型中提取 FC 层参数，生成 .h 和 .c 文件（NMSIS NN 格式）。
    
    Args:
        tflite_path: TFLite 文件路径
        output_dir: 输出目录（会在其中生成 inc/ 和 src/ 子目录）
        subgraph_index: FC 层所在的 subgraph 索引
        operator_index: FC 层在 subgraph 中的 operator 索引
        prefix: 变量名和宏定义的前缀，默认为 "fc"
    """
    tflite_path = Path(tflite_path)
    output_dir = Path(output_dir)
    out_inc = output_dir / "inc" / f"{prefix}_model_params.h"
    out_src = output_dir / "src" / f"{prefix}_model_data.c"

    # 读取 TFLite 模型
    buf = tflite_path.read_bytes()
    model = Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(subgraph_index)
    op = subgraph.Operators(operator_index)
    
    # 权重、bias、输出 tensor
    w_tensor = subgraph.Tensors(op.Inputs(1))
    b_tensor = subgraph.Tensors(op.Inputs(2))
    o_tensor = subgraph.Tensors(op.Outputs(0))

    print("=== Tensor 信息 ===")
    print(f"权重 tensor 形状: {[w_tensor.Shape(i) for i in range(w_tensor.ShapeLength())]}")
    print(f"偏置 tensor 形状: {[b_tensor.Shape(i) for i in range(b_tensor.ShapeLength())]}")
    print(f"输出 tensor 形状: {[o_tensor.Shape(i) for i in range(o_tensor.ShapeLength())]}")

    # 解析 buffer 数据
    w_buf = model.Buffers(w_tensor.Buffer()).DataAsNumpy()
    b_buf = model.Buffers(b_tensor.Buffer()).DataAsNumpy().view(np.int32)
    weight = w_buf.view(np.int8)
    bias = b_buf
    out_ch, in_ch = w_tensor.Shape(0), w_tensor.Shape(1)

    print(f"\n=== Buffer 数据 ===")
    print(f"权重 buffer 大小: {len(w_buf)}")
    print(f"权重 buffer 原始数据 (前20个): {w_buf[:20]}")
    print(f"权重 buffer 完整数据: {w_buf}")
    print(f"权重 int8 数据 (前20个): {weight[:20]}")
    print(f"权重 int8 完整数据: {weight}")

    print(f"\n偏置 buffer 大小: {len(b_buf)}")
    print(f"偏置 buffer 原始数据: {b_buf}")
    print(f"偏置 int32 数据: {bias}")

    print(f"\n输出通道数: {out_ch}")
    print(f"输入通道数: {in_ch}")

    # 量化参数
    in_scale = subgraph.Tensors(op.Inputs(0)).Quantization().Scale(0)
    out_scale = o_tensor.Quantization().Scale(0)
    w_scales = [w_tensor.Quantization().Scale(i) for i in range(out_ch)]

    print(f"\n=== 量化参数 ===")
    print(f"输入缩放因子: {in_scale}")
    print(f"输出缩放因子: {out_scale}")
    print(f"权重缩放因子数量: {len(w_scales)}")
    print(f"权重缩放因子完整列表: {w_scales}")

    multipliers, shifts = [], []
    for i, s in enumerate(w_scales):
        scale_product = in_scale * s / out_scale
        m, sh = quantize_scale(scale_product)
        multipliers.append(m)
        shifts.append(sh)
        print(f"通道 {i}: scale={s}, scale_product={scale_product}, multiplier={m}, shift={sh}")

    print(f"\n=== 最终结果 ===")
    print(f"乘数列表完整数据: {multipliers}")
    print(f"移位列表完整数据: {shifts}")
    print(f"乘数数量: {len(multipliers)}")
    print(f"移位数量: {len(shifts)}")

    # 额外的调试信息
    print(f"\n=== 额外调试信息 ===")
    print(f"权重数据类型: {type(weight)}, 数据形状: {weight.shape}")
    print(f"偏置数据类型: {type(bias)}, 数据形状: {bias.shape}")
    print(f"权重数据范围: min={weight.min()}, max={weight.max()}")
    print(f"偏置数据范围: min={bias.min()}, max={bias.max()}")
    print(f"乘数数据范围: min={min(multipliers)}, max={max(multipliers)}")
    print(f"移位数据范围: min={min(shifts)}, max={max(shifts)}")
    
    # 生成 .c/.h 文件的 C 数组格式
    def to_c_array(name, data, dtype):
        arr = ','.join(str(x) for x in data.flatten())
        return f"const {dtype} {name}[{len(data.flatten())}] = {{{arr}}};"
    
    # 输出目录创建
    out_inc.parent.mkdir(exist_ok=True, parents=True)
    out_src.parent.mkdir(exist_ok=True, parents=True)
    
    # 转换为大写的前缀用于宏定义
    prefix_upper = prefix.upper()
    
    # 写入 .h 文件
    out_inc.write_text(textwrap.dedent(f"""
        #pragma once
        #include <stdint.h>
        #include "riscv_nnfunctions.h"

        #define {prefix_upper}_IN_DIM   ({in_ch})
        #define {prefix_upper}_OUT_DIM  ({out_ch})

        extern const int8_t  {prefix}_weights[];
        extern const int32_t {prefix}_bias[];
        extern const int32_t {prefix}_mult[];
        extern const int32_t {prefix}_shift[];

        static const nmsis_nn_dims {prefix}_input_dims  = {{1,1,1,{prefix_upper}_IN_DIM}};
        static const nmsis_nn_dims {prefix}_filter_dims = {{{prefix_upper}_OUT_DIM, 1, 1, {prefix_upper}_IN_DIM}};
        static const nmsis_nn_dims {prefix}_bias_dims   = {{{prefix_upper}_OUT_DIM, 1, 1, 1}};
        static const nmsis_nn_dims {prefix}_out_dims    = {{1, 1, 1, {prefix_upper}_OUT_DIM}};

        static const nmsis_nn_fc_params {prefix}_params = {{
            .input_offset  = {-int(subgraph.Tensors(op.Inputs(0)).Quantization().ZeroPoint(0))},
            .filter_offset = 0,
            .output_offset = { int(o_tensor.Quantization().ZeroPoint(0))},
            .activation = {{-128, 127}}
        }};

        static const nmsis_nn_per_channel_quant_params {prefix}_pc_quant = {{
            .multiplier = (int32_t *){prefix}_mult,
            .shift      = (int32_t *){prefix}_shift
        }};
    """).strip())
    
    # 写入 .c 文件
    out_src.write_text(textwrap.dedent(f"""
        #include "{prefix}_model_params.h"

        {to_c_array(f"{prefix}_weights", weight, "int8_t")}
        {to_c_array(f"{prefix}_bias", bias, "int32_t")}
        {to_c_array(f"{prefix}_mult", np.array(multipliers, dtype=np.int32), "int32_t")}
        {to_c_array(f"{prefix}_shift", -np.array(shifts, dtype=np.int32), "int32_t")}
    """).strip())
    
    print(f"✅ 已生成: {out_inc} 和 {out_src}")

from tflite import BuiltinOperator

def find_all_fc_ops(model):
    """
    查找模型中所有 FullyConnected 层的位置。
    
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
            if builtin == BuiltinOperator.FULLY_CONNECTED:
                result.append((subgraph_idx, op_idx))
    return result


# 示例调用
if __name__ == "__main__":
    # 示例：为不同模型使用不同前缀
    export_fc_params(
        tflite_path="/path/to/model1.tflite",
        output_dir="./output",
        subgraph_index=0,
        operator_index=0,
        prefix="model1_fc"  # 自定义前缀
    )
    
    # 另一个模型的示例
    export_fc_params(
        tflite_path="/path/to/model2.tflite",
        output_dir="./output",
        subgraph_index=0,
        operator_index=0,
        prefix="model2_fc"  # 不同的前缀避免冲突
    )

# from pathlib import Path
# import flatbuffers
# from tflite import Model
# import numpy as np
# import textwrap, math

# def quantize_scale(double_scale: float):
#     """
#     将浮点 scale 转换为定点表示（mantissa + shift）。
#     """
#     if double_scale == 0:
#         return 0, 0
#     mant, exp = math.frexp(double_scale)
#     mant_q31 = int(round(mant * (1 << 31)))
#     if mant_q31 == (1 << 31):
#         mant_q31 //= 2
#         exp += 1
#     return mant_q31, -exp

# def export_fc_params(
#     tflite_path: str | Path,
#     output_dir: str | Path,
#     subgraph_index: int,
#     operator_index: int,
# ):
#     """
#     从 TFLite 模型中提取 FC 层参数，生成 .h 和 .c 文件（NMSIS NN 格式）。
    
#     Args:
#         tflite_path: TFLite 文件路径
#         output_dir: 输出目录（会在其中生成 inc/ 和 src/ 子目录）
#         subgraph_index: FC 层所在的 subgraph 索引
#         operator_index: FC 层在 subgraph 中的 operator 索引
#     """
#     tflite_path = Path(tflite_path)
#     output_dir = Path(output_dir)
#     out_inc = output_dir / "inc" / "fc_model_params.h"
#     out_src = output_dir / "src" / "fc_model_data.c"

#     # 读取 TFLite 模型
#     buf = tflite_path.read_bytes()
#     model = Model.GetRootAsModel(buf, 0)
#     subgraph = model.Subgraphs(subgraph_index)
#     op = subgraph.Operators(operator_index)
    
#     # # 权重、bias、输出 tensor
#     # w_tensor = subgraph.Tensors(op.Inputs(1))
#     # b_tensor = subgraph.Tensors(op.Inputs(2))
#     # o_tensor = subgraph.Tensors(op.Outputs(0))
    
#     # # 解析 buffer 数据
#     # w_buf = model.Buffers(w_tensor.Buffer()).DataAsNumpy()
#     # b_buf = model.Buffers(b_tensor.Buffer()).DataAsNumpy().view(np.int32)
#     # weight = w_buf.view(np.int8)
#     # bias = b_buf
#     # out_ch, in_ch = w_tensor.Shape(0), w_tensor.Shape(1)
    
#     # # 量化参数
#     # in_scale = subgraph.Tensors(op.Inputs(0)).Quantization().Scale(0)
#     # out_scale = o_tensor.Quantization().Scale(0)
#     # w_scales = [w_tensor.Quantization().Scale(i) for i in range(out_ch)]
    
#     # multipliers, shifts = [], []
#     # for s in w_scales:
#     #     m, sh = quantize_scale(in_scale * s / out_scale)
#     #     multipliers.append(m)
#     #     shifts.append(sh)
#     # 权重、bias、输出 tensor
#     w_tensor = subgraph.Tensors(op.Inputs(1))
#     b_tensor = subgraph.Tensors(op.Inputs(2))
#     o_tensor = subgraph.Tensors(op.Outputs(0))

#     print("=== Tensor 信息 ===")
#     print(f"权重 tensor 形状: {[w_tensor.Shape(i) for i in range(w_tensor.ShapeLength())]}")
#     print(f"偏置 tensor 形状: {[b_tensor.Shape(i) for i in range(b_tensor.ShapeLength())]}")
#     print(f"输出 tensor 形状: {[o_tensor.Shape(i) for i in range(o_tensor.ShapeLength())]}")

#     # 解析 buffer 数据
#     w_buf = model.Buffers(w_tensor.Buffer()).DataAsNumpy()
#     b_buf = model.Buffers(b_tensor.Buffer()).DataAsNumpy().view(np.int32)
#     weight = w_buf.view(np.int8)
#     bias = b_buf
#     out_ch, in_ch = w_tensor.Shape(0), w_tensor.Shape(1)

#     print(f"\n=== Buffer 数据 ===")
#     print(f"权重 buffer 大小: {len(w_buf)}")
#     print(f"权重 buffer 原始数据 (前20个): {w_buf[:20]}")
#     print(f"权重 buffer 完整数据: {w_buf}")
#     print(f"权重 int8 数据 (前20个): {weight[:20]}")
#     print(f"权重 int8 完整数据: {weight}")

#     print(f"\n偏置 buffer 大小: {len(b_buf)}")
#     print(f"偏置 buffer 原始数据: {b_buf}")
#     print(f"偏置 int32 数据: {bias}")

#     print(f"\n输出通道数: {out_ch}")
#     print(f"输入通道数: {in_ch}")

#     # 量化参数
#     in_scale = subgraph.Tensors(op.Inputs(0)).Quantization().Scale(0)
#     out_scale = o_tensor.Quantization().Scale(0)
#     w_scales = [w_tensor.Quantization().Scale(i) for i in range(out_ch)]

#     print(f"\n=== 量化参数 ===")
#     print(f"输入缩放因子: {in_scale}")
#     print(f"输出缩放因子: {out_scale}")
#     print(f"权重缩放因子数量: {len(w_scales)}")
#     print(f"权重缩放因子完整列表: {w_scales}")

#     multipliers, shifts = [], []
#     for i, s in enumerate(w_scales):
#         scale_product = in_scale * s / out_scale
#         m, sh = quantize_scale(scale_product)
#         multipliers.append(m)
#         shifts.append(sh)
#         print(f"通道 {i}: scale={s}, scale_product={scale_product}, multiplier={m}, shift={sh}")

#     print(f"\n=== 最终结果 ===")
#     print(f"乘数列表完整数据: {multipliers}")
#     print(f"移位列表完整数据: {shifts}")
#     print(f"乘数数量: {len(multipliers)}")
#     print(f"移位数量: {len(shifts)}")

#     # 额外的调试信息
#     print(f"\n=== 额外调试信息 ===")
#     print(f"权重数据类型: {type(weight)}, 数据形状: {weight.shape}")
#     print(f"偏置数据类型: {type(bias)}, 数据形状: {bias.shape}")
#     print(f"权重数据范围: min={weight.min()}, max={weight.max()}")
#     print(f"偏置数据范围: min={bias.min()}, max={bias.max()}")
#     print(f"乘数数据范围: min={min(multipliers)}, max={max(multipliers)}")
#     print(f"移位数据范围: min={min(shifts)}, max={max(shifts)}")
    
#     # 生成 .c/.h 文件的 C 数组格式
#     def to_c_array(name, data, dtype):
#         arr = ','.join(str(x) for x in data.flatten())
#         return f"const {dtype} {name}[{len(data.flatten())}] = {{{arr}}};"
    
#     # 输出目录创建
#     out_inc.parent.mkdir(exist_ok=True, parents=True)
#     out_src.parent.mkdir(exist_ok=True, parents=True)
    
#     # 写入 .h 文件
#     out_inc.write_text(textwrap.dedent(f"""
#         #pragma once
#         #include <stdint.h>
#         #include "riscv_nnfunctions.h"

#         #define FC_IN_DIM   ({in_ch})
#         #define FC_OUT_DIM  ({out_ch})

#         extern const int8_t  fc_weights[];
#         extern const int32_t fc_bias[];
#         extern const int32_t fc_mult[];
#         extern const int32_t fc_shift[];

#         static const nmsis_nn_dims fc_input_dims  = {{1,1,1,FC_IN_DIM}};
#         static const nmsis_nn_dims fc_filter_dims = {{FC_OUT_DIM, 1, 1, FC_IN_DIM}};
#         static const nmsis_nn_dims fc_bias_dims   = {{FC_OUT_DIM, 1, 1, 1}};
#         static const nmsis_nn_dims fc_out_dims    = {{1, 1, 1, FC_OUT_DIM}};

#         static const nmsis_nn_fc_params fc_params = {{
#             .input_offset  = {-int(subgraph.Tensors(op.Inputs(0)).Quantization().ZeroPoint(0))},
#             .filter_offset = 0,
#             .output_offset = { int(o_tensor.Quantization().ZeroPoint(0))},
#             .activation = {{-128, 127}}
#         }};

#         static const nmsis_nn_per_channel_quant_params fc_pc_quant = {{
#             .multiplier = (int32_t *)fc_mult,
#             .shift      = (int32_t *)fc_shift
#         }};
#     """).strip())
    
#     # 写入 .c 文件
#     out_src.write_text(textwrap.dedent(f"""
#         #include "fc_model_params.h"

#         {to_c_array("fc_weights", weight, "int8_t")}
#         {to_c_array("fc_bias", bias, "int32_t")}
#         {to_c_array("fc_mult", np.array(multipliers, dtype=np.int32), "int32_t")}
#         {to_c_array("fc_shift", -np.array(shifts, dtype=np.int32), "int32_t")}
#     """).strip())
    
#     print(f"✅ 已生成: {out_inc} 和 {out_src}")

# from tflite import BuiltinOperator

# def find_all_fc_ops(model):
#     """
#     查找模型中所有 FullyConnected 层的位置。
    
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
#             if builtin == BuiltinOperator.FULLY_CONNECTED:
#                 result.append((subgraph_idx, op_idx))
#     return result


# # 示例调用
# if __name__ == "__main__":
#     export_fc_params(
#         tflite_path="/home/lidonghaowsl/develop/VeriSilicon_Cup_Competition_preliminary_round/export_params_lib/vad_model_quant_mini.tflite",
#         output_dir="/home/lidonghaowsl/develop/VeriSilicon_Cup_Competition_preliminary_round/export_params_lib/exported",
#         subgraph_index=0,
#         operator_index=0
#     )
