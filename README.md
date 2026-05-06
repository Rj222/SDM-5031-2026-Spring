# Phased Fine-Tuning of POMO with Three Inductive-Bias Modules

> 本项目对 SDM-5031 课程提供的 POMO baseline 进行 fine-tune，引入三个独立模块和一套三阶段训练框架，在 validation 集上将 `avg_aug_gap` 从 baseline 的 **2.330%** 降低到 **1.117%**（相对降幅 **52.0%**）。

## 1. 改动总览

### 1.1 新增模块（三个独立创新点）

| 模块 | 作用层 | 类型 | 位置 |
|---|---|---|---|
| **Distance/kNN Logit Bias** | Forward path（decoder logit）| 归纳偏置（无可学参数）| `TSP/POMO/model_ext/distance_bias.py` |
| **Mixed Structured Curriculu (MSC)** | 数据生成 | 训练分布混合（uniform / clustered / gaussian）| `TSP/TSProblemDef.py`（generators 部分）|
| **Leader-Focused Reward** | Loss 函数 | 优化目标（强化 best-of-pomo 信号）| `TSP/POMO/train_ext/leader_reward.py` |

### 1.2 新增训练框架

**Phased Fine-Tune（三阶段）**：[TSP/POMO/finetune_phased.py](TSP/POMO/finetune_phased.py)

| Phase | epochs (B=400) | 主任务 | learning rate |
|---|---|---|---|
| Phase 1 (`1_bias`) | 60 (15%) | bias 模块线性 warm-up | 1e-5 |
| Phase 2 (`2_msc`) | 200 (50%) | MSC 分布主适应 | 1e-5 |
| Phase 3 (`3_leader`) | 140 (35%) | leader-focused reward + lr decay | 5e-5 → 5e-6 |

配置文件：[TSP/POMO/configs/finetune_phased.json](TSP/POMO/configs/finetune_phased.json)

### 1.3 关键工程改动

| 改动 | 原因 | 涉及文件 |
|---|---|---|
| **Checkpoint 自带 `bias_cfg`** | 课程组用标准 `test.py` 命令评测时不传额外 flag，bias 模块需自动从 ckpt 激活 | `TSPTrainer.py`, `TSPTester_LIB.py` |
| **`--phase_epochs` CLI 覆盖** | 支持只跑 phase 3（从 phase 2 ckpt resume），用于快速 hyperparameter sweep | `finetune_phased.py` |
| **`control` ablation preset** | 全模块关闭的 fine-tune，作为 ablation 表的 zero-row | `finetune_phased.py` |


### 1.4 新增脚本

| 脚本 | 用途 |
|---|---|
| [TSP/POMO/scripts/run_sweep.sh](TSP/POMO/scripts/run_sweep.sh) | 多任务 GPU 共享 sweep 启动器 |
| [TSP/POMO/scripts/validate_phased.sh](TSP/POMO/scripts/validate_phased.sh) | 在 val 集上自动测试一个 result 目录的所有 phase ckpt |
| [TSP/POMO/scripts/show_sweep_results.py](TSP/POMO/scripts/show_sweep_results.py) | 把多个 run 的结果汇总成 CSV |

---

## 2. 训练命令

### 2.1 默认配置（B=400, all_three）

复现本项目 winner 模型（`winner_k20`）：

```bash
cd TSP/POMO
python finetune_phased.py \
    --config ./configs/finetune_phased.json \
    --total_finetune_epochs 400 \
    --resume_checkpoint ./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
    --ablation all_three --knn_k 20 \
    --desc winner_k20
```

输出目录：`result/<timestamp>_phased_winner_k20_B400_all_three/`，包含：
- `checkpoint-phase_3_leader_best.pt` — 推荐用于评测的 winner ckpt
- `checkpoint-{epoch}.pt` — periodic checkpoints
- `log.txt`, `finetune_phased_config.json` — 完整训练日志和参数快照

### 2.2 Ablation 配置

```bash
# control: 全模块关闭（用于排除"多训 400 epoch"混淆变量）
python finetune_phased.py --config ./configs/finetune_phased.json \
    --total_finetune_epochs 400 \
    --resume_checkpoint ./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
    --ablation control --desc abl_control

# 单模块 ablation
python finetune_phased.py --config ./configs/finetune_phased.json \
    --total_finetune_epochs 400 \
    --resume_checkpoint ./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
    --ablation bias_only --knn_k 20 --desc abl_bias_only
# 同理可用 --ablation msc_only / leader_only
```

可选 `--ablation` 取值：`all_three / bias_only / msc_only / leader_only / bias_msc / msc_leader / bias_leader / control`。

### 2.3 超参 sweep（仅 phase 3，从 phase 2 ckpt resume）

跳过 phase 1+2，只跑 phase 3，用于快速 leader_alpha / phase 3 length 探索：

```bash
# 例: leader_alpha=10→20 + 280 epoch（phase 3 加倍）
CKPT_P2=./result/<winner_k20 dir>/checkpoint-3260.pt
python finetune_phased.py --config ./configs/finetune_phased.json \
    --phase_epochs 0,0,280 \
    --resume_checkpoint "$CKPT_P2" \
    --ablation all_three --knn_k 20 \
    --leader_alpha 10 --leader_alpha_final 20 \
    --desc tune_alpha10_20_280
```

### 2.4 多任务并行（在单 GPU 上）

在 24GB 4090 上可同时跑两个任务：

```bash
# Task 1 后台启动
CUDA_VISIBLE_DEVICES=0 nohup python finetune_phased.py \
    --config ./configs/finetune_phased.json --total_finetune_epochs 400 \
    --resume_checkpoint .../checkpoint-3000.pt \
    --ablation all_three --knn_k 20 --desc run1 > logs/run1.log 2>&1 &

sleep 15  # 错开启动，避免初始化撞 OOM

# Task 2 后台启动
CUDA_VISIBLE_DEVICES=0 nohup python finetune_phased.py \
    --config ./configs/finetune_phased.json --total_finetune_epochs 400 \
    --resume_checkpoint .../checkpoint-3000.pt \
    --ablation bias_only --knn_k 20 --desc run2 > logs/run2.log 2>&1 &
```

每个任务约占 5-6 GB 显存，dual 共享 GPU-Util ~97% 时每个 epoch 约 6.8 分钟（vs solo ~5 min）。

---

## 3. 测试命令

### 3.1 ⭐ 课程组官方评测命令（提交时使用）

直接使用项目原始 README 中规定的标准接口。**Bias 配置已嵌入 ckpt，无需额外 flag**：

```bash
cd TSP/POMO
python test.py \
    --data_path /path/to/hidden_test_set \
    --checkpoint_path /path/to/checkpoint.pt \
    --use_cuda true \
    --cuda_device_num 0 \
    --augmentation_enable true \
    --aug_factor 8 \
    --detailed_log false \
    --output_json /path/to/eval_result.json
```

`--checkpoint_path` 推荐使用 `checkpoint-phase_3_leader_best.pt`。`bias_cfg` 会从 ckpt 自动加载并 attach 到模型，与训练时配置完全一致。

### 3.2 在 validation 集上评测

```bash
cd TSP/POMO
python test.py \
    --data_path ../data/val \
    --checkpoint_path ./result/<run_dir>/checkpoint-phase_3_leader_best.pt \
    --use_cuda true --cuda_device_num 0 \
    --augmentation_enable true --aug_factor 8 \
    --detailed_log false \
    --output_json /tmp/eval.json
```

或用脚本一次性测一个 run 目录里所有 phase 的 best ckpt：

```bash
bash scripts/validate_phased.sh result/<run_dir>
```

## 4. 主要实验结果

### 4.1 Validation 集（10 个 TSPLIB 实例，N=100-300）

| 配置 | avg_no_aug_gap | **avg_aug_gap** | Δ vs baseline |
|---|---|---|---|
| baseline (3000ep) | 3.754% | 2.330% | — |
| control (3000+400ep, all off) | 3.199% | 1.962% | -0.368% |
| winner_k20 (k=20, α=20→40) | 1.693% | **1.122%** | -1.208% |
| alpha=10→20 (140 ep) | 1.779% | 1.181% | -1.149% |
| alpha=40→60 (140 ep) | 1.641% | 1.184% | -1.146% |
| **p3double (α=20→40, 280 ep)** | 1.815% | **1.117%** ⭐ | **-1.213%** |
| **TODO:p3double (α=10→20, 280 ep)** | - |- | -|

→ **In-distribution (N=100-200)**：`p3double` 最佳
→ **OOD (N>200)**：`alpha=10→20` 最佳（**size-generalization trade-off**）

---

## 5. 项目结构

```
SDM-5031-2026-Spring/
├── README.md                         # 课程原始 README（保留）
├── README_new.md                     # 本文件
├── slides/
│   └── sdm5031_project_pre.tex     # 答辩 slides
├── TSP/
│   ├── TSProblemDef.py               # +MSC 三个 generator
│   └── POMO/
│       ├── finetune_phased.py        # 三阶段 fine-tune 主入口（新增）
│       ├── test.py                   # 标准评测入口（保留原接口）
│       ├── TSPTrainer.py             # +bias_cfg 嵌入 ckpt
│       ├── TSPTester_LIB.py          # +bias_cfg 自动 attach
│       ├── TSPModel.py               # +distance_bias_module hook
│       ├── model_ext/
│       │   └── distance_bias.py      
│       ├── train_ext/
│       │   └── leader_reward.py      
│       ├── configs/
│       │   └── finetune_phased.json  # 默认配置
│       ├── scripts/
│       │   ├── run_sweep.sh
│       │   ├── validate_phased.sh
│       │   ├── test_on_external.sh
│       │   └── show_sweep_results.py
│       └── result/
│           ├── saved_tsp100_model2_longTrain/  # baseline ckpt（保留）
│           └── <timestamp>_phased_*_B400_*/    # 各次 fine-tune 输出（gitignore）
└── utils/
```

---

## 6. 复现 valid data上winner（暂时） 的最快路径

```bash
# 1. 训练（约 33h dual / 20h solo on RTX 4090）
cd TSP/POMO
python finetune_phased.py --config ./configs/finetune_phased.json \
    --total_finetune_epochs 400 \
    --resume_checkpoint ./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt \
    --ablation all_three --knn_k 20 --desc winner_k20

# 2. 评测（用项目原始 README 规定的标准命令）
python test.py \
    --data_path ../data/val \
    --checkpoint_path ./result/best_ckpt_1/checkpoint-phase_3_leader_best.pt \
    --use_cuda true --cuda_device_num 0 \
    --augmentation_enable true --aug_factor 8 \
    --detailed_log false \
    --output_json /tmp/winner_eval.json

```

