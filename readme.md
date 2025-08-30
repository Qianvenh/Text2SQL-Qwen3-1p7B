# Text2SQL-Qwen3-1p7B

SFT and Context Engineering the qwen3 1.7B for Text2SQL task

## Overview

训练整体基于[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)框架；整合[ROLL](https://github.com/alibaba/ROLL)框架的子模块[MCoreAdapter](https://github.com/alibaba/ROLL/tree/main/mcore_adapter)，使得LLaMA-Factory兼容Megatron。

设备：NVIDIA GeForce RTX 4090 (24G) * 4

LoRA SFT 采用 DeepSpeed Zero-3 方案

Full Parameters SFT 采用 DeepSpeed Zero-2 + Megatron TP 方案

Text2SQL评估框架：[test-suite-sql-eval](https://github.com/Qianvenh/Text2SQL-Qwen3-1p7B/tree/main/test-suite-sql-eval)

## 模型SQL生成表现

采用执行准确率（EXECUTION ACCURACY）作为Benchmark

### EXECUTION ACCURACY 比较

| 模型 | Easy | Medium | Hard | Extra | All | Joint_All |
|------|------|--------|------|-------|-----|-----------|
| baseline | 0.859 | 0.587 | 0.431 | 0.151 | 0.556 | 0.556 |
| qwen3_1p7B_lora_32r | 0.798 | 0.473 | 0.374 | 0.193 | 0.489 | 0.489 |
| qwen3_1p7B_lora_128r | 0.734 | 0.426 | 0.322 | 0.151 | 0.438 | 0.438 |
| qwen3_1p7B_full  | 0.839 | 0.574 | 0.500 | 0.241 | 0.572 | 0.572 |

### 模型性能排名

#### 总体准确率 (All)
1. **qwen3_1p7B_full**: 0.572
2. **baseline**: 0.556  
3. **qwen3_1p7B_lora_32r**: 0.489
4. **qwen3_1p7B_lora_128r**: 0.438

#### 各难度级别最佳模型
- **Easy**: baseline (0.859)
- **Medium**: qwen3_1p7B_full (0.574)
- **Hard**: qwen3_1p7B_full (0.500)
- **Extra**: qwen3_1p7B_full (0.241)

### 关键观察
- `qwen3_1p7B_full` 在总体性能上表现最佳，特别是在 Medium、Hard 和 Extra 难度上
- `baseline` 在 Easy 难度上表现最好，但在复杂查询上性能下降明显  
- 随着 LoRA rank 增加（32r → 128r），模型性能反而下降
- 所有模型在 Extra 难度上的表现都相对较差，显示了极难查询的挑战性

### 性能对比图表

#### 各难度级别性能对比
```
Easy    ████████▌  baseline (0.859)
        ████████▍ qwen3_1p7B_full (0.839)  
        ███████▊  qwen3_1p7B_lora_32r (0.798)
        ███████▎  qwen3_1p7B_lora_128r (0.734)

Medium  █████▊    qwen3_1p7B_full (0.574)
        █████▉    baseline (0.587)
        ████▋     qwen3_1p7B_lora_32r (0.473)  
        ████▎     qwen3_1p7B_lora_128r (0.426)

Hard    █████      qwen3_1p7B_full (0.500)
        ████▎     baseline (0.431)
        ███▋      qwen3_1p7B_lora_32r (0.374)
        ███▎      qwen3_1p7B_lora_128r (0.322)

Extra   ██▍       qwen3_1p7B_full (0.241)
        ██▍       qwen3_1p7B_lora_32r (0.193)  
        █▌         baseline (0.151)
        █▌         qwen3_1p7B_lora_128r (0.151)
```
