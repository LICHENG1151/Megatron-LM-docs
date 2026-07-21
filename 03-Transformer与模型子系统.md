# 03 · Transformer 与模型子系统

本篇拆解 Megatron 如何用「Spec 规格 + 可组合构件」搭建各类模型：从 `TransformerConfig` 配置，到 `TransformerBlock`/`TransformerLayer` 构件，再到 GPT/Mamba/Hybrid/MoE/多模态等具体模型。

相关路径：
- `megatron/core/transformer/`（构件层 L3）
- `megatron/core/models/`（模型层 L2）

---

## 1. 核心设计：Spec 机制

Megatron 不把模型结构写死，而是用 **`ModuleSpec`（规格）** 描述「某个位置该放哪个模块、用什么参数」，再由 `build_module()` 动态实例化。这是整个模型层最重要的设计模式。

```12:13:megatron/core/transformer/spec_utils.py
@dataclass
class ModuleSpec:
```

```mermaid
flowchart LR
    SPEC["ModuleSpec<br/>(module=类, submodules=子规格, params=参数)"]
    BUILD["build_module(spec, *args)"]
    SPEC --> BUILD --> INST["实例化的 nn.Module"]
```

### 带来的好处

- **可插拔后端**：同一个注意力位置，既可填本地实现，也可填 Transformer Engine 的融合实现（FP8 加速），只换 spec 不改模型代码。
- **结构变体统一**：Dense / MoE / MLA（多头潜在注意力）/ Mamba 层只是不同的 layer spec 组合。
- **配方化**：`gpt_layer_specs.py`、`mamba_layer_specs.py`、`moe_module_specs.py` 等文件就是「预制 spec 工厂」。

例如 GPT 提供 `get_gpt_layer_with_transformer_engine_spec()` 与本地版本，按是否安装 TE 选择。

---

## 2. Transformer 构件层（transformer/）

自底向上的构件层级：

```mermaid
flowchart TD
    BLOCK["TransformerBlock<br/>(transformer_block.py)<br/>= 多个 TransformerLayer 堆叠"]
    LAYER["TransformerLayer<br/>(transformer_layer.py)<br/>= 自注意力子层 + MLP 子层 + 残差/归一化"]
    ATTN["Attention<br/>(attention.py)<br/>self/cross-attn"]
    MLP["MLP / MoE<br/>(mlp.py, moe/)"]
    DPA["DotProductAttention<br/>(dot_product_attention.py)"]
    MLA["MultiLatentAttention<br/>(multi_latent_attention.py)"]

    BLOCK --> LAYER
    LAYER --> ATTN
    LAYER --> MLP
    ATTN --> DPA
    ATTN --> MLA
```

### 关键文件

| 文件 | 职责 |
|------|------|
| `transformer_config.py` | `TransformerConfig`（继承 `ModelParallelConfig`），统一结构超参 |
| `transformer_block.py` | `TransformerBlock`：层堆叠 + 最终归一化，处理 PP 切层 |
| `transformer_layer.py` | `TransformerLayer`：标准「注意力 + MLP」双子层结构 |
| `attention.py` | 自/交叉注意力，QKV 投影（列并行）、输出投影（行并行） |
| `dot_product_attention.py` | 核心注意力计算（可走 flash/fused 后端） |
| `multi_latent_attention.py` | MLA（DeepSeek 系列的低秩 KV 压缩注意力） |
| `mlp.py` | 前馈网络（列并行升维 + 行并行降维） |
| `multi_token_prediction.py` | MTP 多 token 预测（DeepSeek-V3 特性） |
| `module.py` | `MegatronModule` 基类（提供 sharded_state_dict 等） |
| `cuda_graphs.py` | CUDA Graph 捕获以降低 kernel launch 开销 |
| `moe/` | 混合专家子系统（见下） |

`TransformerConfig` 是模型构建的统一配置入口：

```53:53:megatron/core/transformer/transformer_config.py
class TransformerConfig(ModelParallelConfig):
```

---

## 3. MoE 混合专家子系统（transformer/moe/）

MoE 把单个 MLP 替换为「路由器 + 多个专家」，每个 token 只激活少数专家，从而在不显著增加计算的前提下放大参数量。

```mermaid
flowchart LR
    X["tokens"] --> R["Router 路由器<br/>(router.py) 计算 top-k 专家"]
    R --> D["TokenDispatcher<br/>(token_dispatcher.py)<br/>all-to-all 把 token 发到专家所在 GPU"]
    D --> E["Experts<br/>(experts.py) 分组 GEMM"]
    E --> C["combine 收回结果"]
    R -.负载均衡 loss.-> AUX["moe_utils.py"]
    X --> S["SharedExperts 共享专家<br/>(shared_experts.py)"]
```

| 文件 | 职责 |
|------|------|
| `router.py` | 路由：top-k 专家选择、负载均衡 |
| `token_dispatcher.py` | token 在 EP 组间的 all-to-all 分发与收回 |
| `fused_a2a.py` | 融合 all-to-all 通信优化 |
| `experts.py` | 专家网络（分组 GEMM 实现） |
| `shared_experts.py` | 共享专家（所有 token 都过） |
| `moe_utils.py` | 辅助损失、容量因子、辅助工具 |
| `token_dispatcher_inference.py` | 推理期专用 dispatcher |
| `upcycling_utils.py` | 从 dense 模型「升级」为 MoE |

EP 进程组由 `parallel_state` 提供，见 [02 并行化子系统](./02-并行化子系统.md)。

---

## 4. 模型层（models/）

各模型本质上是「一组 layer spec + 嵌入 + 输出头」的组装。

### 4.1 GPT（models/gpt/）

最核心、被复用最多的模型。`GPTModel` 继承 `LanguageModule`，依赖几乎覆盖全栈：

```9:39:megatron/core/models/gpt/gpt_model.py
from megatron.core import tensor_parallel
...
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from megatron.core.models.common.language_module.language_module import LanguageModule
...
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
```

| 文件 | 职责 |
|------|------|
| `gpt_model.py` | `GPTModel`：嵌入 → TransformerBlock → 输出投影 |
| `gpt_layer_specs.py` | 预制 layer spec（TE 版 / 本地版 / MoE 版 / MLA 版） |
| `moe_module_specs.py` | MoE 层规格 |
| `fine_grained_callables.py` | 细粒度调度节点（前/后处理、层节点），支持计算-通信重叠 |
| `heterogeneous/` | 异构层（不同层不同结构） |

### 4.2 公共构件（models/common/）

被所有语言模型复用：

- `embeddings/`：`LanguageModelEmbedding`（词嵌入 + 位置嵌入）、`RotaryEmbedding`（RoPE）、`YarnRotaryEmbedding`（YaRN 长上下文扩展）、`MultimodalRotaryEmbedding`。
- `language_module/language_module.py`：`LanguageModule` 基类，提供 logits 计算、词嵌入权重共享（PP 首尾 tying）、loss 计算。
- `model_chunk_schedule_plan.py`：模型分块的调度计划（配合 VP/细粒度调度）。

### 4.3 Mamba 与 Hybrid（models/mamba、models/hybrid）

- **Mamba**：状态空间模型（SSM），用 `ssm/` 下的算子，`mamba_layer_specs.py` 定义 Mamba 层规格。
- **Hybrid**：Transformer + Mamba 混合架构（如 Falcon-H1）。`hybrid_layer_allocation.py` 决定哪些层用注意力、哪些用 Mamba；`hybrid_block.py` 组装混合块。

```mermaid
flowchart LR
    HYBRID["HybridModel"] --> ALLOC["hybrid_layer_allocation<br/>按比例分配层类型"]
    ALLOC --> A["注意力层"]
    ALLOC --> M["Mamba 层"]
```

### 4.4 其他模型

| 目录 | 模型 |
|------|------|
| `models/T5/` | T5 编码器-解码器 |
| `models/bert/` | BERT 双向编码器 |
| `models/multimodal/` | 视觉-语言多模态（VLM） |
| `models/vision/` | 视觉骨干 |
| `models/mimo/` | 多输入多输出 |
| `models/huggingface/` | HuggingFace 模型桥接 |

---

## 5. 模型结构如何与并行逻辑对接

> 第二章已经把并行原语的**理论与实现**讲透（[02.4 组构建](./02.4-并行组构建与通信详解.md)/[02.5 TP](./02.5-张量并行实现详解.md)/[02.6 PP](./02.6-流水线并行与1F1B调度.md)/[02.7 CP](./02.7-上下文并行.md)/[02.8 EP](./02.8-专家并行.md)）。本节只回答一件事：**这里定义的模型结构（L2/L3），到底在哪几个点、用什么机制"接上"那些已经造好的并行能力**——包括你自己写一个新模型时要遵守什么。

核心结论：**模型代码里几乎看不到并行**。它不 all-reduce、不查 rank、不切 tensor，而是靠下面三条"接缝"把并行能力**注入**进来，一切从 `parallel_state`（第二章建好的并行组"真相源"）自动流出。

### 5.1 接缝一：配置驱动——`TransformerConfig ← ModelParallelConfig`

一切并行的"开关与尺寸"都从**配置**进来，模型不自己决定：

```
parallel_state.initialize_model_parallel(tp, pp, cp, ep, …)   ← 第二章：先织好所有进程组
        │  把 world 切成 TP×CP×EP×DP×PP 的多维网格
        ▼
ModelParallelConfig   (tensor_model_parallel_size / pipeline_… / context_… / sequence_parallel …)
        ▼  被继承
TransformerConfig     ← 模型结构超参 + 上面这些并行尺寸，合成一个对象
        ▼  一路透传
GPTModel / TransformerBlock / Attention / MLP …   ← 构件按 config 里的并行尺寸自我裁剪
```

- `TransformerConfig(ModelParallelConfig)`（03 §2 已列）是**唯一入口**：结构超参和并行尺寸捏在同一个对象里，逐层透传。
- 于是"开多大 TP/PP/CP/EP"是 **L1 配置层**的事；L2/L3 只是**读** config、按尺寸把自己切开，从不硬编码 `world_size/rank`。

### 5.2 接缝二：四个挂接点——每个都对应第二章一节

模型结构在**四个确定的位置**接入并行，每处都只是"换算子/换组装方式"，通信细节留在第二章讲的原语里：

| 挂接点 | 模型侧做的事（L3 代码） | 接入的并行原语（第二章） |
|---|---|---|
| **Attention / MLP 的 Linear** | QKV/升维用 `ColumnParallelLinear`、输出/降维用 `RowParallelLinear`，一列一行配对 | **TP/SP** → [02.5](./02.5-张量并行实现详解.md) |
| **`TransformerBlock` 建层** | 按 PP rank 只 `build` 本 stage 的层；首/尾 stage 才建 embedding/输出头 | **PP** → [02.6](./02.6-流水线并行与1F1B调度.md) |
| **`DotProductAttention` 内部** | 注意力按 CP 组做 ring/all-gather + online-softmax 累加；数据侧 zigzag 重排 | **CP** → [02.7](./02.7-上下文并行.md) |
| **MoE 的 `TokenDispatcher`** | 路由后经 all-to-all 把 token 发到专家所在卡，算完 combine 收回 | **EP** → [02.8](./02.8-专家并行.md) |

```mermaid
flowchart TD
    GPT["GPTModel.forward"] --> EMB["embedding (词表并行 VocabParallelEmbedding)"]
    EMB --> BLK["TransformerBlock<br/>(只含本 PP stage 的层)"]
    BLK --> LAYER["TransformerLayer × N"]
    LAYER --> ATTN["Attention<br/>QKV列并行 → 核心attn(CP累加) → 输出行并行"]
    LAYER --> FFN["MLP(列升维/行降维) 或 MoE(EP all-to-all)"]
    BLK --> HEAD["输出投影 + VocabParallelCrossEntropy"]
```

> 注意"配对"不是随意的：列并行的输出正好是行并行的输入，中间那次通信被**抵消**成子层末尾一次 all-reduce（TP）——这是 02.5 讲的，模型侧只需**成对使用**就自动享受，不用管通信在哪发。

### 5.3 接缝三：Spec 机制——让"接哪种并行实现"可插拔

`ModuleSpec` + `build_module()`（§1）决定每个位置**实例化哪个类**：注意力后端选 flash/fused、MLP 选 dense 还是 MoE、Linear 选 TE 版还是本地并行版。**换 spec 就换并行实现，模型主干代码一行不改**——`gpt_layer_specs.py` 里的 "TE 版 / 本地版 / MoE 版 / MLA 版"就是同一骨架配不同 spec。

### 5.4 自定义模型要遵守的"契约"

正因为并行是**注入**而非**内嵌**的，你写一个新模型（Mamba/Hybrid 就是这么加进来的）时，**不需要写任何并行代码**，只要遵守下面几条契约，就能白嫖第二章所有并行：

1. **继承对的基类**：`MegatronModule`（拿到 `sharded_state_dict` → 分布式 ckpt 自动可切/可重切，见 [08 检查点](./08-检查点与重切分.md)）；语言模型再继承 `LanguageModule`（拿到 logits、PP 首尾词嵌入 tying、loss）。
2. **接受并透传 `TransformerConfig`**：所有子模块从同一 config 读并行尺寸，别自己 `torch.distributed` 建组。
3. **线性层一律用并行版**：`Column/RowParallelLinear`，并遵守"列-行配对"；词表相关用 `VocabParallel*`。
4. **用 spec 组装、`build_module()` 实例化**：不要在 `__init__` 里 `new` 死具体类，好处是后端/并行实现可换。
5. **按 PP 切层**：层堆叠交给 `TransformerBlock`（或仿它按 PP rank 只建本段），别在模型里一次性建满所有层。
6. **需要新通信就用组接口**：从 `parallel_state.get_*_group()` 取进程组（[02.4](./02.4-并行组构建与通信详解.md)），别硬编码 rank/world_size。

**你不需要做的**：手写 all-reduce/all-gather、判断 rank、切 tensor、写 DP 梯度同步（DP + 分布式优化器在**模型外面**包一层，见 [04](./04-分布式训练与优化器.md)）。做到以上契约，一个新结构就能和 TP/SP/CP/PP/EP/DP **正交组合**、开箱即用。

```mermaid
flowchart LR
    CFG["TransformerConfig<br/>(含并行尺寸)"] -->|透传| M["你的新模型<br/>继承 MegatronModule/LanguageModule"]
    SPEC["ModuleSpec"] -->|build_module| M
    PS["parallel_state<br/>(第二章建好的组)"] -.就近查表.-> OP["Column/RowParallelLinear<br/>TokenDispatcher / CP 通信"]
    M --> OP
    OP -->|通信细节在第二章原语里| DONE["自动获得 TP/SP/CP/PP/EP"]
```

---

## 6. 依赖关系小结

- 模型层（L2）**组合** Transformer 构件（L3），构件层**调用** 并行原语（L4）。
- Spec 机制是解耦「结构定义」与「具体实现/后端」的关键。
- 所有模型共享 `models/common/` 的嵌入与 `LanguageModule` 基类。
- 并行能力对模型代码「透明」——靠并行版算子与 `parallel_state` 自动生效。

下一篇：[分布式训练与优化器](./04-分布式训练与优化器.md)。
