from mfccdata import (
    prepareWindowConfig,
    prepareMelconfig,
    prepareDctconfig,
    checkF16,
    genMfccHeader,
    genMfccInit,
)
import json

# ========= 0. 选择数值格式 =========
# 嵌入式上如果你想要 Q31 / Q15，只需把 "type" 改成 "q31" 或 "q15"
dtype = "f32"

# ========= 1. 配置字典 =========
configs = {
    "window": {
        "hann256": {
            "type": dtype,
            "fftlength": 256,
            "win": "hanning",   # TorchAudio 默认 hann_window(periodic) ⇔ script 的 hann(sym=False)
        }
    },
    "melfilter": {
        "mel40": {
            "type": dtype,
            "fftlength": 256,
            "melFilters": 40,
            "samplingRate": 8000,
            "fmin": 0.0,              # 与 TorchAudio 一致
            "fmax": 8000 / 2,         # Nyquist
        }
    },
    "dct": {
        "dct13": {
            "type": dtype,
            "dctOutputs": 13,
            "melFilters": 40,
        }
    },
}

# ========= 2. 生成各类系数 =========
prepareWindowConfig(configs["window"])
prepareMelconfig(configs["melfilter"])
prepareDctconfig(configs["dct"])
checkF16(configs)        # 如果同时生成了 f16，会自动加宏

# ========= 3. 输出 .h / .c 文件 =========
with open("mfcc_data.h", "w", encoding="utf-8") as hf:
    genMfccHeader(hf, configs, "mfcc_data")

with open("mfcc_data.c", "w", encoding="utf-8") as cf:
    genMfccInit(cf, configs, "mfcc_data")

print("✓ 生成完成：mfcc_data.h / mfcc_data.c")
