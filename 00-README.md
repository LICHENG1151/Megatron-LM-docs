# Megatron-LM 代码仓深度分析文档

> 分析对象：`/Users/a1/work/work_space/Megatron-LM`
> 版本：Megatron Core `0.19.0`（分支 `main`，commit `59bb1c16a`，2026-06）
> 规模：约 1062 个 Python 文件，`megatron/` 目录下约 18.8 万行代码

本文档集对 NVIDIA Megatron-LM 进行自顶向下的系统化拆解：先建立**全局框架透视图**，理清各部分的组成与依赖关系；再按子系统逐个深入；最后给出**整体代码逻辑综述**，把所有子系统串成一条完整的训练/推理执行链路。

---

## 阅读顺序

| 序号 | 文档 | 内容概述 |
|------|------|----------|
| 00 | 本 README | 文档导航与全局速览 |
| 01 | [框架透视图解](./01-框架透视图解.md) | 双层架构、目录结构、分层依赖关系总图、执行链路鸟瞰 |
| 02 | [并行化子系统](./02-并行化子系统.md) | 导览：五维并行总览 + 子文档索引（下列 02.0–02.8） |
| 02.0 | [Transformer 与 MoE 结构基础](./02.0-Transformer与MoE结构基础.md) | 〔铺垫〕单卡逻辑视角的 MoE 层、MHA/MLA 分轨、三段式与下刀点地图 |
| 02.1 | [显存、激活值与重计算](./02.1-显存、激活值与重计算.md) | 〔铺垫〕为什么单卡装不下：16P 显存账、激活值、重计算、FlashAttention 类比 |
| 02.1.1 | [FlashAttention](./02.1.1-FlashAttention.md) | 〔铺垫〕专题：online-softmax 分块 + 反向重算，激活 O(s²)→O(s)；TE 后端与 `--attention-backend` |
| 02.2 | [优化器数学原理与对比](./02.2-优化器数学原理与对比.md) | 〔铺垫〕SGD / Adam / Muon 公式与对比，状态多少决定显存 |
| 02.3 | [通信原语速查](./02.3-通信原语速查.md) | 〔铺垫〕Broadcast…All-to-All 全套原语、通信量、All-Reduce=RS+AG |
| 02.3.1 | [FSDP](./02.3.1-FSDP.md) | 〔铺垫〕专题：ZeRO-1/2/3 显存账+通信账，把 16P 切成 16P/N；Torch FSDP2 / Megatron-FSDP |
| 02.4 | [并行组构建与通信详解](./02.4-并行组构建与通信详解.md) | 〔实现〕`parallel_state` 分组算法、建组三步、DDP 分桶 + 分布式优化器 ZeRO |
| 02.5 | [张量并行实现详解](./02.5-张量并行实现详解.md) | 〔实现〕列/行并行、`f`/`g` 对偶算子、序列并行 |
| 02.6 | [流水线并行与 1F1B 调度](./02.6-流水线并行与1F1B调度.md) | 〔实现〕气泡、microbatch、1F1B 三阶段、交错式流水线 |
| 02.7 | [上下文并行与专家并行](./02.7-上下文并行与专家并行.md) | 〔实现〕CP（Ring/Ulysses、Zigzag 负载均衡）、EP（两次 All-to-All）、通信特征总表 |
| 02.8 | [进阶专题](./02.8-进阶专题.md) | 〔进阶〕Muon 分布式化、长序列工具箱（Cut-CE/2D CP）、FSDP vs TP+PP 选型 |
| 03 | [Transformer 与模型子系统](./03-Transformer与模型子系统.md) | Spec 机制、TransformerBlock/Layer、GPT/Mamba/Hybrid/MoE/多模态 |
| 04 | [分布式训练与优化器](./04-分布式训练与优化器.md) | DDP / FSDP、梯度桶、DistributedOptimizer、梯度裁剪与缩放 |
| 05 | [数据集与分词器](./05-数据集与分词器.md) | IndexedDataset、BlendedDataset、GPTDataset、tokenizers |
| 06 | [训练框架 Harness](./06-训练框架Harness.md) | 入口脚本、`pretrain()`、训练主循环、参数系统、检查点、**性能剖析（torch.profiler / Nsys / 显存历史）** |
| 07 | [推理子系统](./07-推理子系统.md) | 静态/动态引擎、调度器、采样、文本生成服务 |
| 08 | [检查点与重切分](./08-检查点与重切分.md) | 分布式检查点格式、ShardedTensor、resharding |
| 09 | [后训练与强化学习](./09-后训练与RL.md) | ModelOpt 量化/蒸馏/剪枝、RL/GRPO 训练 |
| 10 | [工具、示例与测试 CI](./10-工具示例与测试CI.md) | tools/、examples/、tests/、CI 工作流 |
| 11 | [整体逻辑综述](./11-整体逻辑综述.md) | 端到端执行链路、跨子系统协作、设计哲学总结 |

---

## 一句话总览

Megatron-LM 由两层构成：

- **Megatron Core**（`megatron/core/`）：可组合的 GPU 优化底层库，提供 Transformer 构件、五维并行（TP/PP/DP/CP/EP）、混合精度、分布式优化器、分布式检查点与推理引擎。面向**框架开发者**。
- **Megatron-LM 训练参考实现**（`megatron/training/` + 根目录 `pretrain_*.py`）：在 Core 之上提供开箱即用的训练脚本、命令行参数、数据管线与日志。面向**研究者与快速实验**。

核心依赖方向（自上而下，上层依赖下层）：

```
入口脚本 → training 框架 → models → transformer → tensor/pipeline_parallel → parallel_state / distributed → PyTorch
```
