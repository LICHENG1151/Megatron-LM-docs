<!--
name: 分布式Profiling系统Perfetto方案
creator: Li Cheng
created: 2026-07-02
modified: 2026-07-02
related_issues:
-->

# Megatron-LM 分布式 Profiling 系统 → Perfetto

**状态**：未启动

## 目标与背景

需要一套面向 **多机多卡分布式训练** 的细粒度 profiling 系统：从抓取 trace → 转成固定数据格式 → 在外部 UI 展示，且**优先复用已有平台能力**。目标是对**每个进程、每张卡、每个 step** 的每个操作算子（计算 / 读写 / 通信）都能明确看到耗时。

调研结论：Megatron-LM **已经具备大部分抓取能力**，缺的是「把多机多卡的原生 trace 打上并行坐标元数据并合并成一份可视化 trace」这一环。因此本方案**不重写抓取器**，而是补齐三处缺口：
1. 每个 rank 的**并行坐标元数据 + 跨机时钟锚点**（sidecar）。
2. 少量 **NVTX 覆盖补齐**（compute / 通信 / IO），让算子级窗口在 Perfetto 里成为有名字的 slice。
3. 一个**离线合并工具**，把所有 rank 的 chrome trace 重映射 pid、注入元数据、按机器/rank/并行坐标分组，输出**一份可直接在 perfetto.dev 打开**的 trace。

### 已锁定的用户决策
- **外部 UI = Perfetto**（perfetto.dev，Chrome/Perfetto trace 格式）。
- **抓取粒度 = 两者结合**：全程保留低开销的 Timers/NVTX；在指定 step 窗口用 `torch.profiler` 抓算子/kernel 级（复用现有 `--profile-step-start/end`）。
- **固定数据格式 = 原生 trace + 元数据增强**（不新建聚合 schema）：给每个 rank 的 trace 注入 node/hostname、local_rank、global_rank、TP/PP/DP/CP/EP 坐标，再合并成一份 trace。

## 影响范围

- 代码仓：`/Users/a1/work/work_space/Megatron-LM`
- 新增：`megatron/training/profiler_utils.py`、`tools/profiler/merge_perfetto_traces.py`、`tools/profiler/__init__.py`、（可选）`docs/developer/profiling_perfetto.md`
- 修改：`megatron/training/config/common_config.py`、`megatron/training/training.py`、`megatron/core/pipeline_parallel/schedules.py`、`pretrain_gpt.py`、`megatron/elastification/pretrain_hybrid_flex.py`、`megatron/core/distributed/distributed_data_parallel.py`
- 默认全部开关关闭，不影响现有训练路径。

## 已有基础设施（已读源码核实）

- **torch.profiler 已接好**：`megatron/training/training.py:3380-3411`。`trace_handler`（:3394-3397）导出 `{tensorboard_dir}/../torch_profile/rank-{rank}.json.gz`；schedule 来自 `profile_step_start/end`；`prof.start()` :3410。逐迭代 `prof.step()` 在 :3450-3451；`configure_nvtx_profiling(True)` 在 `iteration==profile_step_start and args.nvtx_ranges` 时触发（:3448-3449）。
- **ProfilingConfig**：`megatron/training/config/common_config.py:24-67`。**新增字段会由 `ArgumentGroupFactory` 自动生成 CLI flag**（`arguments.py:~2531` 的 `ArgumentGroupFactory(ProfilingConfig).build_group(parser,"profiling")`；bool→`store_true`，snake_case→`--kebab-case`）。
- **NVTX helpers**：`megatron/core/utils.py`（`configure_nvtx_profiling`、`nvtx_range_push/pop`、`nvtx_decorator`），受 call-time `_nvtx_enabled` 控制，带 pinned-string pool（CUDA graph 安全）。P2P 收发已带 `@nvtx_decorator`；schedule 只有阶段级 range；**forward/backward-compute、DDP grad sync、train_step all-reduce、batch-generator 尚无 NVTX**。
- **Timers**：`megatron/core/timers.py`（`Timers.__call__`、`Timer.active_time()`）。每 step 命名计时集见 `training.py:2470-2493`。
- **并行坐标 getter**：`megatron/core/parallel_state.py`（TP/PP/VPP/DP/CP/EP 的 rank/world_size）。global rank 用 `torch.distributed.get_rank()`；local rank 用 `args.local_rank`/`LOCAL_RANK`；hostname 用 `socket.gethostname()`。
- `tools/` 下暂无 `profiler/` 目录。

**Chrome trace 格式要点**（合并工具据此实现）：顶层 `{"traceEvents":[...], "displayTimeUnit":"ms", ...}`；事件 `ph:"X"`（完整 slice，带 `pid/tid/ts/dur`，`ts/dur` 为**微秒 float**）与 `ph:"M"`（元数据：`process_name`/`process_labels`/`process_sort_index`/`thread_name`）。文件为 gzip JSON。**exporter 的 pid 是本地小计数器，跨 rank 会冲突 → 合并时必须按 rank 重映射 pid。**

## 实现方案

### 1. 每 rank 元数据 sidecar —— 新建 `megatron/training/profiler_utils.py`
- `gather_rank_metadata(args)`：收集 global/local rank、hostname、`node_id`（`SLURM_NODEID`/`GROUP_RANK` 回退 hostname）、device_index、world_size、TP/PP/VPP/DP/CP/EP 的 rank+size。**每个 parallel_state 调用用 try/except 包裹**，未启用维度返回 `-1`。
- `capture_clock_anchor()`：`torch.distributed.barrier()` + `torch.cuda.synchronize()` 后，同一瞬间记录 `time.time()`（wall）与 `time.perf_counter()`（与 chrome ts 同族），用于跨机对齐。
- `write_rank_metadata(args, profile_dir=None)`：写 `rank-{global_rank}.meta.json` 到 `profiler_output_dir` 或默认 `{tensorboard_dir}/../torch_profile`。
- **插入点**：`training.py:3448-3449` 紧邻 NVTX enable 处（所有 rank 都会到 `profile_step_start`，barrier 安全，且此时 dist/parallel_state 已初始化）：
  ```python
  if iteration == args.profile_step_start and args.profiler_emit_metadata:
      from megatron.training.profiler_utils import write_rank_metadata
      write_rank_metadata(args)
  ```
- **风险**：若 `profile_ranks` 非空，只有子集 rank 会到这行 —— `capture_clock_anchor` 的 barrier **必须对全体 rank 集合**（不要放在子集 guard 后，否则 hang）；或规定 `--align-clocks` 时要求全量抓取。

### 2. 补齐 NVTX / record_function 覆盖（沿用 `nvtx_ranges` 开关）
复用 `nvtx_range_push/pop` 与 `@nvtx_decorator`（均受 `_nvtx_enabled` 控制），仅在缺口处补：
1. **compute** —— `schedules.py`：在 `forward_step_func(...)` 调用（:~472-478）外包 `"forward-compute"`；backward 主体（:~510）外包 `"backward-compute"`（与同名 timer 对齐）。
2. **数据加载** —— `pretrain_gpt.py:210-250` 在 `timers('batch-generator')` 区间内包 `"batch-generator"`（参考 `megatron/elastification/pretrain_hybrid_flex.py:404-416`）。
3. **DDP grad sync** —— `megatron/core/distributed/distributed_data_parallel.py:544` 的 `finish_grad_sync` 加 `@nvtx_decorator(message="all-grads-sync")`。
4. **train_step all-reduce** —— `training.py:2387` loss all-reduce 外包 `"loss-all-reduce"`（可选 :2352/:2355 的 `"mp-stat-allreduce"`）。

命名前缀稳定，供合并工具的 `--category-filter` 归类 compute/comm/io。**CUDA graph 注意**：捕获区内用字符串字面量（走 pool），push/pop 在所有分支平衡。

### 3.（组合方案的补充，flag 控制，in-scope）全程每 step Timers JSONL
在 `profiler_utils.py` 加 `dump_step_timers(args, iteration, timers)`：读 `training.py:2470-2493` 的命名计时集，用 **`active_time()`（累计、非破坏性，不 reset、不 barrier）** 避免扰动 logging 路径，追加写 `rank-{rank}.timers.jsonl`。从 `training_log`（`training.py:2408`）调用，受 `args.profiler_per_step_timer_dump` 控制。**本期只落 flag + writer**（合并工具渲染 JSONL 作为后续增量）。

### 4. 离线合并工具 —— 新建 `tools/profiler/merge_perfetto_traces.py`（+ `tools/profiler/__init__.py`）
```
python tools/profiler/merge_perfetto_traces.py --input-dir <dir> --output merged.json.gz \
  [--ranks 0,1,4-7] [--time-range-us START END] [--category-filter compute,comm,io] \
  [--sort-order node_rank|global_rank|parallel_coords] [--align-clocks] [--max-events-per-rank N]
```
算法：
1. 递归发现 `rank-*.json.gz` + 同名 `rank-*.meta.json`，按 `global_rank` 关联；缺 meta 时合成最小元数据。
2. `--ranks` 子集过滤。
3. **时钟对齐**（`--align-clocks`）：以最小 global_rank 为基准，`offset_us=(wall_r-wall_ref)*1e6-(perf_r-perf_ref)*1e6`，逐事件 `ts+=offset_us`；`|offset|>1s` 告警（NTP 偏差）。
4. **pid 重映射（强制）**：`new_pid=global_rank`，改写所有事件 pid，保留 tid。
5. **注入 `ph:"M"` 事件**：`process_name=host={hostname} R{gr} L{lr}`；`process_labels=TP{..}/{..} PP{..}/{..} DP{..}/{..} CP{..} EP{..} node={..} cuda:{dev}`；`process_sort_index` 按 `--sort-order`（`node_rank`:`node_id*1000+local_rank`；`parallel_coords`:`((pp*dp_size+dp)*tp_size)+tp`；`global_rank`:`global_rank`）。保留原 `thread_name`（改 pid）。
6. **category 过滤**：comm=nccl/`*-recv`/`*-send`/`*-all-reduce`/`all-grads-sync`；io=`batch-generator`/dataloader；compute=`forward/backward-compute`/`aten::*`/gemm。
7. **time-range** 在对齐后过滤。
8. **流式合并**：逐 rank 加载→变换→经 `gzip.open(out,"wt")` 增量写（`{"traceEvents":[ ...chunks... ]}`），常驻 ≤1 个 rank；`--max-events-per-rank` 兜底。
9. 产物直接在 https://ui.perfetto.dev 打开。

格式纪律：仅 `ph:"X"`/`ph:"M"` 带 pid；`ts/dur` float µs；不重排 tid；保留 `args`；`displayTimeUnit`/`deviceProperties` 只取基准 rank 一份。

### 5. 多机采集流程
- **共享 FS（首选）**：`--tensorboard-dir` 指向共享路径 → 各 rank 自动把 `rank-{r}.json.gz`+`.meta.json`(+`.timers.jsonl`) 写到同一目录，原地 `--align-clocks` 合并，无需 gather。
- **节点本地盘**：作业结束后 `rsync/scp` 各节点 `torch_profile/` 汇总到一个目录（rank id 全局唯一，不冲突），再合并。
- **打开**：把 `merged.json.gz` 拖进 perfetto.dev；track 按 host+rank 分组、按并行坐标标注；搜 `forward-compute`/`all-grads-sync`/nccl。
- 参考 skill `mcore-run-on-slurm`（`SLURM_NODEID`/`LOCAL_RANK`/`MASTER_ADDR`）；可选写 `docs/developer/profiling_perfetto.md`。

### 6. 新增 CLI 参数（加到 `common_config.py:67` 之后，自动生成 flag）
```python
    profiler_emit_metadata: bool = False
    """写 per-rank rank-{global_rank}.meta.json（host/rank/并行坐标 + 时钟锚点）。"""
    profiler_per_step_timer_dump: bool = False
    """全程追加 per-step Timers 到 rank-{rank}.timers.jsonl。"""
    profiler_output_dir: str | None = None
    """覆盖 profiler 输出目录，默认 {tensorboard_dir}/../torch_profile。"""
```
`profiler_output_dir` 需接入 `trace_handler`（`training.py:3395`）、`write_rank_metadata`、`dump_step_timers`。改了 import 的文件跑 `uv run isort`。

## 约束与风险
- **跨机时钟偏差**：wall 对齐受 NTP 限制（ms～数十 ms）；用 wall+perf_counter 双锚点，`|offset|>1s` 告警；机内精确、跨机尽力。
- **trace 体积 / 合并内存**：`gzip.open` 流式增量写；`--ranks`/`--time-range-us`/`--category-filter`/`--max-events-per-rank` 收敛。
- **gzip 识别**：按 magic bytes（`1f 8b`）区分 gzip 与纯 JSON；输出恒 gzip。
- **pid 冲突**：合并的头号正确性点 —— 重映射为 `global_rank`。
- **开销**：NVTX 廉价且仅窗口内生效；JSONL 用 `active_time()`（无 reset/barrier）；两者默认关闭。
- **子集 `profile_ranks` + barrier**：锚点 barrier 对全体 rank，或 `--align-clocks` 要求全量抓取。
- **进程组规范（Megatron CLAUDE.md）**：元数据在 `megatron/training` 读 rank getter（非 `get_*_group()`）；`megatron/core` 内 NVTX 改动不新增 group 读取 —— 合规。

## 验证与结果
**最小复现**：单机 2 卡 tiny GPT：`--nproc-per-node 2 --use-pytorch-profiler --profile-step-start 2 --profile-step-end 4 --nvtx-ranges --profiler-emit-metadata --profiler-per-step-timer-dump --train-iters 6 --tensorboard-dir /tmp/tb`（用 `run`/`mcore-testing` 的 tiny 启动，跑在 CI 容器里，见 `mcore-build-and-dependency`）。

**预期产物**（`/tmp/torch_profile/`）：`rank-{0,1}.json.gz`、`rank-{0,1}.meta.json`（断言 global/local rank、hostname、并行坐标、`clock.anchor_*`）、`rank-{0,1}.timers.jsonl`（每 step ≥1 行，含 `forward-compute`/`backward-compute`）。

**合并 + 确认**：`python tools/profiler/merge_perfetto_traces.py --input-dir /tmp/torch_profile --output /tmp/merged.json.gz --align-clocks` → 断言合法 JSON、2 个不同 pid、每 rank 有 `process_name`、无 pid 冲突；在 ui.perfetto.dev 打开：两条带标注的 process track，可见 `forward-compute`/`backward-compute`/`batch-generator`/`all-grads-sync`/nccl slice，时间轴对齐。

**自动化**：对 2 文件 fixture 写一个 pytest，断言 pid 唯一、`process_name` 存在、对齐后 ts 单调。

## 进展记录
- 2026-07-02：完成代码仓调研与方案设计，落档（未启动实现）。

## 关联问题
- 暂无
