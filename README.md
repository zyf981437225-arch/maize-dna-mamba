# Maize DNA-Mamba / OneMaize

本仓库提供玉米基因组 DNA-Mamba 的两阶段预训练流程。当前版本将数据构建、完整性校验、性能基准、分布式 smoke test、正式训练、断点续训和评估拆成可独立执行的步骤，适用于单节点 8×H200，也可先在单卡 A100/H200 上完成小规模验证。

## 1. 当前训练方案

| 阶段 | 数据 | 序列长度 | 采样方式 | 主要目标 |
| --- | --- | ---: | --- | --- |
| Phase I | B73，染色体 1–10 | 8,192 bp | 从坐标 0 开始连续、非重叠切片；末尾不足 8,192 bp 的片段保留并 PAD | 建立全基因组基础表示 |
| Phase II | B73 + 25 个 NAM 材料 | 16,384 bp | gene / TE / intergenic 按 50% / 30% / 20% 采样，并在候选区域内动态裁剪 | 扩展跨材料和功能区域表示 |

两阶段的定义不可混用：Phase I 不使用 region-aware sampler；Phase II 不使用固定的全基因组非重叠窗口。

当前实现状态：

- Phase I 的 B73 manifest 构建、严格校验、dataset、训练配置、benchmark、smoke test 和审计脚本已经完成。
- Phase II 的 16K region-aware sampler 保持原设计；B73 候选区域已经通过长度审计。
- 26 个材料的正式 Phase II metadata 仍需在全部 FASTA/GFF3 到位后构建并验收。
- 8×H200 的最终 batch size、吞吐率和训练预算应以目标计算节点上的 benchmark 结果为准。

## 2. 模型与训练定义

- backbone：双向 Mamba DNA language model
- 层数：24
- hidden size：864
- 参数量：约 121,191,553
- 词表：`A/C/G/T/N`，`PAD=4`
- 训练目标：masked language modeling
- mask 比例：15%
- 被选中位置：80% 替换为 MASK/N，10% 替换为随机碱基，10% 保留原碱基
- 训练集 reverse-complement 概率：0.5
- 验证集和测试集：确定性，不做随机 reverse-complement
- 数值精度：BF16
- Phase II 正式材料划分：23 train / 1 validation / 2 test，且 B73 必须在 train split

## 3. 关键入口

| 用途 | 文件 |
| --- | --- |
| Phase I manifest 构建 | `scripts/build_b73_phase1_8k_manifest.py` |
| Phase I manifest 校验 | `scripts/validate_b73_phase1_8k_manifest.py` |
| Phase I dataset | `src/dataloaders/datasets/b73_full_genome.py` |
| Phase I 训练配置 | `configs/experiment/onemaize_b73_phase1_8k_full_genome.yaml` |
| Phase I benchmark | `scripts/benchmark_b73_phase1_8k.py` |
| Phase I 启动器 | `scripts/run_onemaize_phase1_h200.sh` |
| Phase I Slurm 模板 | `slurm_scripts/run_onemaize_phase1_h200.slurm` |
| Phase I 切片统计 | `scripts/stat_b73_8192_full_genome_slicing.py` |
| Phase II region metadata 构建 | `scripts/build_onemaize_regions.py` |
| Phase II 数据校验 | `scripts/validate_onemaize_data.py` |
| Phase II 候选长度审计 | `scripts/audit_onemaize_phase2_candidate_lengths.py` |
| Phase II 训练配置 | `configs/experiment/onemaize_b73_phase2_16k_region_aware.yaml` |
| Phase II 启动器 | `scripts/run_onemaize_h200.sh` |
| Phase II Slurm 模板 | `slurm_scripts/run_onemaize_h200.slurm` |

## 4. 完整执行顺序

正式运行按以下顺序进行；前一项未通过时，不进入下一项。

1. 克隆仓库并建立 Python/CUDA 环境。
2. 准备带 `.fai` 和 `.gzi` 索引的 B73 BGZF FASTA。
3. 构建 Phase I 的 8,192-bp 全基因组 manifest。
4. 严格校验 Phase I manifest，并运行单元测试。
5. 在目标 GPU 节点运行 Phase I benchmark。
6. 运行 8 卡 Phase I smoke test，检查 loss、显存、吞吐和 checkpoint。
7. 根据 benchmark 冻结 batch size 和训练步数，提交 Phase I 正式训练。
8. 监控训练，并在中断时从 `last.ckpt` 恢复。
9. 使用 Phase I 最优 checkpoint 完成技术评估并归档结果。
10. 全部 26 个材料到位后构建 Phase II metadata，冻结 23/1/2 split。
11. 对 Phase II 做严格校验、16K benchmark 和 8 卡 smoke test。
12. 从 Phase I 最优 checkpoint 初始化 Phase II，提交正式训练并评估 held-out genotypes。

## 5. 环境准备

### 5.1 获取代码

```bash
git clone https://github.com/zyf981437225-arch/maize-dna-mamba.git
cd maize-dna-mamba
export PROJECT_DIR="$PWD"
```

### 5.2 建立环境

仓库提供 Linux 环境安装脚本：

```bash
bash setup_linux_env.sh
source .venv/bin/activate
```

如果超算已经提供 Conda 环境，也可以直接激活已有环境：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caduceus_env
```

确认 PyTorch、CUDA、GPU 数量和 BF16：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("bf16 supported:", torch.cuda.is_bf16_supported())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

正式训练前还应检查磁盘。建议为 metadata、日志、多个 checkpoint 和临时文件预留至少 50 GB：

```bash
df -h "$PROJECT_DIR"
```

## 6. Phase I：B73 8K 全基因组预训练

### 6.1 准备路径

FASTA 必须是可随机访问的 BGZF 文件，并同时存在同名前缀的 `.fai` 和 `.gzi`：

```bash
export ONEMAIZE_B73_FASTA=/shared/onemaize/raw/B73/Zm-B73-REFERENCE-NAM-5.0.fa.gz
export ONEMAIZE_PHASE1_DIR=/shared/onemaize/metadata/phase1_b73_8k
export ONEMAIZE_PHASE1_MANIFEST=$ONEMAIZE_PHASE1_DIR/B73_phase1_8192_full_genome.parquet
export RUN_ROOT=/shared/onemaize/runs/phase1_b73_8k
export NUM_DEVICES=8
export BATCH_SIZE=1
export BATCH_SIZE_EVAL=1
export NUM_WORKERS=8

test -f "$ONEMAIZE_B73_FASTA"
test -f "$ONEMAIZE_B73_FASTA.fai"
test -f "$ONEMAIZE_B73_FASTA.gzi"
mkdir -p "$ONEMAIZE_PHASE1_DIR" "$RUN_ROOT"
```

### 6.2 构建固定窗口 manifest

```bash
python scripts/build_b73_phase1_8k_manifest.py \
  --fasta "$ONEMAIZE_B73_FASTA" \
  --output "$ONEMAIZE_PHASE1_MANIFEST" \
  --window-size 8192 \
  --stride 8192
```

构建规则：

- 只使用 B73 染色体 1–10。
- 每条染色体从坐标 0 开始，以 `window_size=8192`、`stride=8192` 连续切片。
- 完整窗口的 `valid_length=8192`。
- 每条染色体最后不足 8,192 bp 的尾部保留一条记录，并在 dataset 中 PAD 到 8,192 token。
- PAD token 不参与 MLM loss。
- 每个有效基因组 bp 恰好被覆盖一次，不重叠、不遗漏。

当前 B73 参考基因组的预期统计如下：

| 指标 | 数值 |
| --- | ---: |
| chr1–10 总有效长度 | 2,131,846,805 bp |
| 完整 8,192-bp 窗口 | 260,229 |
| 尾部窗口 | 10 |
| manifest 总序列数 | 260,239 |
| 尾部有效 bp | 50,837 |
| 需要 PAD 的 token | 31,083 |
| 有效 bp 覆盖比例 | 100% |

若结果与该表不同，应先检查参考版本、染色体命名和索引，不要直接开始训练。

### 6.3 严格校验 manifest

```bash
python scripts/validate_b73_phase1_8k_manifest.py \
  --manifest "$ONEMAIZE_PHASE1_MANIFEST" \
  --fasta "$ONEMAIZE_B73_FASTA" \
  | tee "$ONEMAIZE_PHASE1_DIR/validation.json"
```

该步骤检查染色体集合、窗口顺序、边界、重叠/缺口、尾部策略、总覆盖量以及 FASTA 一致性。只有命令返回 0 且报告为通过时才能继续。

### 6.4 运行测试

```bash
python -m pytest -q \
  tests/test_onemaize_phase1.py \
  tests/test_onemaize_pipeline.py \
  tests/test_maize_dna_pipeline.py
```

当前版本预期为 `18 passed`。

### 6.5 性能 benchmark

Slurm 节点上提交：

```bash
export CONDA_ENV_NAME=caduceus_env
sbatch --export=ALL,STAGE=benchmark \
  slurm_scripts/run_onemaize_phase1_h200.slurm
```

已经处于交互式 GPU 节点时，可直接运行：

```bash
BENCHMARK_WARMUP_STEPS=5 \
BENCHMARK_STEPS=100 \
bash scripts/run_onemaize_phase1_h200.sh benchmark
```

输出写入 `$RUN_ROOT/benchmark.json`。至少记录：单步耗时、sequences/s、tokens/s、峰值显存、I/O 吞吐和 GPU 数量。正式训练预算必须由目标 H200 节点的实测结果推导。

### 6.6 8 卡 smoke test

```bash
export SMOKE_STEPS=20
export SMOKE_WARMUP_STEPS=2
export CHECKPOINT_INTERVAL=20

sbatch --export=ALL,STAGE=smoke \
  slurm_scripts/run_onemaize_phase1_h200.slurm
```

验收条件：

- 8 个 rank 均正常启动，且没有 NCCL hang。
- 无 `NaN`、`Inf`、CUDA OOM 或 traceback。
- train loss 能在很短的 smoke 过程中明显下降。
- GPU 利用率和显存占用合理。
- smoke test 不保存完整的周期性 optimizer checkpoint，避免无意义占满共享磁盘。

### 6.7 计算一个 epoch 的步数

Phase I manifest 固定为 `N=260,239` 条序列。DistributedSampler 会把每个 rank 补齐到相同长度：

```text
world_size = 8
samples_per_rank = ceil(260239 / 8) = 32530
global_draws_per_epoch = 32530 × 8 = 260240
sampler padding duplicates = 1
```

optimizer step 数公式为：

```text
steps_per_epoch = ceil(samples_per_rank / (batch_size_per_gpu × accumulate_grad_batches))
```

默认 `8 GPU × batch 1 × accumulate 1` 时，一个数据 epoch 为 32,530 optimizer steps。若 batch size 或梯度累积发生变化，必须重新计算，不能沿用 32,530。

### 6.8 提交 Phase I 正式训练

以下示例训练一个完整数据 epoch，warmup 取总步数约 5%。`MAX_STEPS` 表示本次实验的总目标步数：

```bash
export RUN_ROOT=/shared/onemaize/runs/phase1_b73_8k_production
export NUM_DEVICES=8
export BATCH_SIZE=1
export BATCH_SIZE_EVAL=1
export NUM_WORKERS=8
export MAX_STEPS=32530
export WARMUP_STEPS=1626
export CHECKPOINT_INTERVAL=2000

sbatch --export=ALL,STAGE=train \
  slurm_scripts/run_onemaize_phase1_h200.slurm
```

主要输出：

```text
$RUN_ROOT/train/console.log
$RUN_ROOT/train/checkpoints_best/val_loss.ckpt
$RUN_ROOT/train/checkpoints_resume/last.ckpt
```

Phase I dataset 还会记录窗口覆盖审计指标，包括实际样本数、唯一窗口数、重复窗口数和覆盖比例。一个完整 epoch 结束后，应确认唯一窗口覆盖接近 260,239；DistributedSampler 允许出现预期的 1 条补齐重复。

若计划训练多个数据 epoch，可把 `MAX_STEPS` 设为 `32530 × epoch 数`。建议先完成一个 epoch 并检查 validation loss，再决定是否延长。

### 6.9 实时监控

```bash
# 查看作业状态
squeue -u "$USER"

# 实时查看训练日志
tail -f "$RUN_ROOT/train/console.log"

# 提取 loss、epoch 和 step
grep -E "train/loss|val/loss|epoch|global_step" \
  "$RUN_ROOT/train/console.log" | tail -n 50

# 查看 GPU
nvidia-smi

# 查看磁盘和 checkpoint 大小
df -h "$RUN_ROOT"
du -sh "$RUN_ROOT/train"/* 2>/dev/null
ls -lh "$RUN_ROOT/train/checkpoints_best" \
       "$RUN_ROOT/train/checkpoints_resume"
```

### 6.10 断点续训

作业因时间限制或节点故障退出后，从同一次实验的 `last.ckpt` 恢复：

```bash
export RESUME_CKPT="$RUN_ROOT/train/checkpoints_resume/last.ckpt"
export MAX_STEPS=32530
export WARMUP_STEPS=1626

sbatch --export=ALL,STAGE=train \
  slurm_scripts/run_onemaize_phase1_h200.slurm
```

`MAX_STEPS` 是恢复后的总目标 global step，不是额外增加的步数。例如 checkpoint 已在 step 20,000，`MAX_STEPS=32,530` 时会继续到 32,530。

### 6.11 Phase I 技术评估

```bash
export EVAL_CKPT="$RUN_ROOT/train/checkpoints_best/val_loss.ckpt"

sbatch --export=ALL,STAGE=test \
  slurm_scripts/run_onemaize_phase1_h200.slurm
```

Phase I 的 validation/test 是 B73 内部的确定性评估，用于检查训练质量和选择 checkpoint，不代表跨材料泛化性能。跨材料结论应来自 Phase II 的 held-out genotype test split。

## 7. Phase II：NAM26 16K region-aware 预训练

### 7.1 每个材料需要的文件

每个材料至少需要：

- BGZF FASTA：`*.fa.gz`
- FASTA 索引：`*.fa.gz.fai`
- BGZF 索引：`*.fa.gz.gzi`
- gene annotation：GFF3
- TE annotation：GFF3

推荐目录结构：

```text
/shared/onemaize/raw/
├── B73/
│   ├── genome.fa.gz
│   ├── genome.fa.gz.fai
│   ├── genome.fa.gz.gzi
│   ├── genes.gff3.gz
│   └── transposable_elements.gff3.gz
├── B97/
│   └── ...
└── ...
```

正式材料集合必须恰好包含：

```text
B73 B97 CML103 CML228 CML247 CML277 CML322 CML333 CML52 CML69
Hp301 Il14H Ki11 Ki3 Ky21 M162W M37W Mo18W Ms71 NC350 NC358
Oh43 Oh7B P39 Tx303 Tzi8
```

### 7.2 冻结输入表和数据划分

先建立 `onemaize_26.tsv`，每行一个材料：

```text
genotype<TAB>fasta<TAB>gene_gff<TAB>te_gff<TAB>split
```

示例：

```text
B73	/shared/onemaize/raw/B73/genome.fa.gz	/shared/onemaize/raw/B73/genes.gff3.gz	/shared/onemaize/raw/B73/transposable_elements.gff3.gz	train
B97	/shared/onemaize/raw/B97/genome.fa.gz	/shared/onemaize/raw/B97/genes.gff3.gz	/shared/onemaize/raw/B97/transposable_elements.gff3.gz	train
```

要求：

- 23 个 train、1 个 validation、2 个 test。
- B73 必须属于 train。
- split 一旦用于正式实验就应冻结，并与模型结果一起归档。
- validation/test genotype 不得进入训练 sampler。

### 7.3 构建 Phase II metadata

```bash
export ONEMAIZE_DATA_DIR=/shared/onemaize/metadata/phase2_nam26
mkdir -p "$ONEMAIZE_DATA_DIR"

python scripts/build_onemaize_regions.py \
  --input-manifest /shared/onemaize/manifests/onemaize_26.tsv \
  --output-dir "$ONEMAIZE_DATA_DIR" \
  --max-n-fraction 0.10 \
  --formal \
  | tee "$ONEMAIZE_DATA_DIR/build.log"
```

核心输出：

```text
$ONEMAIZE_DATA_DIR/manifest.json
$ONEMAIZE_DATA_DIR/genomes.parquet
$ONEMAIZE_DATA_DIR/regions.parquet
```

`--formal` 是正式数据门禁：材料数、材料名称、split、索引、染色体映射或 annotation 有问题时会直接失败。不要通过删除该参数绕过数据问题。

### 7.4 校验 Phase II 数据

```bash
export RUN_ROOT=/shared/onemaize/runs/phase2_nam26_preflight
export REQUIRE_FORMAL=1

bash scripts/run_onemaize_h200.sh validate

python scripts/audit_onemaize_phase2_candidate_lengths.py \
  --data-dir "$ONEMAIZE_DATA_DIR" \
  --context-length 16384 \
  --output-md "$ONEMAIZE_DATA_DIR/phase2_16k_candidate_audit.md"
```

校验内容包括：schema、26 材料集合、23/1/2 split、B73 归属、FASTA/索引可读性、GFF3 seqid 映射、region 边界、N 比例以及 16K 动态裁剪的候选长度。

### 7.5 单张 H200 的 16K benchmark

先在单卡测可行 batch size 和显存：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_onemaize.py \
  --data-dir "$ONEMAIZE_DATA_DIR" \
  --context-length 16384 \
  --d-model 864 \
  --n-layer 24 \
  --batch-size 1 \
  --precision bf16 \
  --warmup-steps 2 \
  --steps 10 \
  --io-steps 32 \
  --num-workers 8 \
  --output-json "$RUN_ROOT/benchmark_16k_h200.json"
```

如果 OOM，先保持 `batch-size=1`，再检查实现和显存碎片；不要通过减少层数或 hidden size 改变正式模型定义。

### 7.6 8 卡 Phase II smoke test

Phase II 从 Phase I 的最佳 checkpoint 初始化。`INIT_CKPT` 表示阶段迁移，只加载模型权重并开始新的训练状态：

```bash
export INIT_CKPT=/shared/onemaize/runs/phase1_b73_8k_production/train/checkpoints_best/val_loss.ckpt
export RUN_ROOT=/shared/onemaize/runs/phase2_nam26_smoke
export NUM_DEVICES=8
export BATCH_SIZE=1
export BATCH_SIZE_EVAL=1
export NUM_WORKERS=8
export MAX_STEPS=20
export WARMUP_STEPS=2
export CHECKPOINT_INTERVAL=20
export REQUIRE_FORMAL=1

sbatch --export=ALL,STAGE=16k \
  slurm_scripts/run_onemaize_h200.slurm
```

必须确认加载 checkpoint 时不存在关键参数缺失、shape mismatch 或意外重初始化，并检查 16K 下的 loss、吞吐、显存和 NCCL 状态。

### 7.7 冻结 Phase II 正式预算

Phase II 使用动态 region-aware 采样，没有与 Phase I 相同的固定“全基因组 epoch”。预算用总 optimizer steps 表示：

```text
global_batch = GPU 数 × 每卡 batch × accumulate_grad_batches
total_sequences = MAX_STEPS × global_batch
total_tokens = total_sequences × 16384
平均每个 train genotype 的抽样数 ≈ total_sequences / 23
```

推荐先根据 16K benchmark 计算预计时长，再冻结 `MAX_STEPS`、warmup 和 checkpoint 间隔。正式提交示例：

```bash
export RUN_ROOT=/shared/onemaize/runs/phase2_nam26_production
export INIT_CKPT=/shared/onemaize/runs/phase1_b73_8k_production/train/checkpoints_best/val_loss.ckpt
export NUM_DEVICES=8
export BATCH_SIZE=1
export BATCH_SIZE_EVAL=1
export NUM_WORKERS=8
export MAX_STEPS=<根据benchmark确定>
export WARMUP_STEPS=<通常为MAX_STEPS的约5%>
export CHECKPOINT_INTERVAL=<根据队列时限和磁盘确定>
export REQUIRE_FORMAL=1

sbatch --export=ALL,STAGE=16k \
  slurm_scripts/run_onemaize_h200.slurm
```

### 7.8 Phase II 断点续训和测试

断点续训使用 `RESUME_CKPT`，它恢复模型、optimizer、scheduler 和 global step。恢复时不要同时设置 `INIT_CKPT`：

```bash
unset INIT_CKPT
export RESUME_CKPT="$RUN_ROOT/16k/checkpoints_resume/last.ckpt"

sbatch --export=ALL,STAGE=16k \
  slurm_scripts/run_onemaize_h200.slurm
```

held-out genotype 测试：

```bash
unset INIT_CKPT RESUME_CKPT
export EVAL_CKPT="$RUN_ROOT/16k/checkpoints_best/val_loss.ckpt"

sbatch --export=ALL,STAGE=test16k \
  slurm_scripts/run_onemaize_h200.slurm
```

## 8. 当前实测结果

### 8.1 Phase I B73 全基因组切片

当前 B73 FASTA 已完成固定窗口构建和逐坐标核验：260,239 条序列覆盖 chr1–10 的 2,131,846,805 个有效 bp，覆盖率为 100%。保留 10 个尾部窗口后需要 31,083 个 PAD token；若直接丢弃尾部，则为 260,229 条序列，覆盖率约 99.9976%。详细统计见：

- `B73_8192_FULL_GENOME_SLICING_STATS.md`
- `B73_8192_FULL_GENOME_SLICING_STATS.csv`

### 8.2 单张 A100 80GB smoke 和吞吐

在完整 24 层、hidden size 864、8,192 token、BF16、batch size 1 的配置下：

- 20-step smoke test 的 train loss 约从 5.10 降至 2.03。
- A100 短 benchmark 平均约 0.441 秒/step。
- 吞吐约 2.27 sequences/s，约 18,564 tokens/s。
- 按单卡、batch size 1 的短 benchmark 线性估算，完整遍历 260,239 条序列约需 31.9 小时。

该时间只用于量级参考。8×H200 的实际耗时必须在对应节点完成 benchmark 后重新计算，因为分布式通信、I/O、验证和 checkpoint 写入都会影响总时间。

### 8.3 早期 B73 region-aware 训练记录

项目早期曾使用 B73 region-aware 8K 数据完成 7 个 epoch，用于证明代码、模型和 loss 链路可以稳定运行：

| epoch | global step | 末段 train loss | val loss | 耗时 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 25,000 | 1.080 | 1.09709 | 6:00:18 |
| 1 | 50,000 | 0.901 | 1.00976 | 6:01:42 |
| 2 | 75,000 | 0.808 | 0.96155 | 6:02:11 |
| 3 | 100,000 | 0.747 | 0.92955 | 6:02:04 |
| 4 | 125,000 | 0.700 | 0.90695 | 6:00:41 |
| 5 | 150,000 | 0.670 | 0.89139 | 6:00:12 |
| 6 | 175,000 | 0.651 | 0.87862 | 6:01:22 |

这组结果对应旧的 region-aware 8K 数据定义，不能当作当前 Phase I 固定全基因组窗口的正式结果，也不能代表 NAM26 跨材料性能。它只说明训练过程稳定、loss 持续下降，并提供 A100 运行时间参考。

### 8.4 Phase II B73 候选区域审计

当前 B73 的 16K 候选区域为：

| 类型 | 数量 |
| --- | ---: |
| gene | 39,021 |
| TE | 143 |
| intergenic | 124,986 |
| 总计 | 164,150 |

这些候选区域均满足 16,384-bp 动态裁剪要求。正式 Phase II 仍需对全部 26 个材料重新构建并完成相同审计。

## 9. 停止标准与归档

### 9.1 建议停止的情况

- 连续 3 个完整验证周期的 val loss 绝对改善均小于约 0.005。
- 连续 3 个完整验证周期 val loss 回升。
- 出现持续的 NaN/Inf、数据损坏、checkpoint 无法恢复或无法解释的覆盖异常。
- 磁盘可用空间低于 10 GB。此时应先处理存储，不要继续生成 checkpoint。

不要只根据 train loss 判断是否继续；正式判断应以 validation loss、覆盖审计和下游测试为依据。

### 9.2 每次正式实验应归档

- Git commit hash
- 数据 manifest、schema 版本和 23/1/2 split
- 完整 Hydra 配置
- Slurm 提交脚本和 job ID
- benchmark JSON
- console log
- best checkpoint 和 resume checkpoint
- validation/test 指标
- GPU 型号、数量、PyTorch/CUDA/driver 版本
- 总步数、global batch、总 sequences、总 tokens 和实际运行时长

数据文件和 checkpoint 不提交到 GitHub；GitHub 只保存代码、配置、操作说明和小型审计报告。

## 10. 常见问题

### 10.1 缺少 `.fai` 或 `.gzi`

当前 dataset 需要对 BGZF FASTA 做随机访问。FASTA、`.fai` 和 `.gzi` 必须来自同一个文件版本，不能混用重新压缩前后的索引。

### 10.2 GFF3 seqid 不存在于 FASTA

先比较 FASTA contig 名与 GFF3 第一列。不要静默丢弃无法映射的 annotation；应显式建立映射或更换匹配的 annotation 版本。

### 10.3 `--formal` 拒绝 manifest

检查是否恰好 26 个指定材料、是否为 23/1/2、B73 是否在 train、每个材料的 FASTA/索引/GFF3 是否都可读。修正输入后重新构建 metadata。

### 10.4 H200 上找不到 Mamba CUDA kernel

确认当前 shell 激活的是安装了项目依赖的环境，并检查 `torch`、CUDA runtime、驱动及 `mamba-ssm`/`causal-conv1d` 的 ABI 是否匹配。环境变化后先重新运行 benchmark。

### 10.5 NCCL hang

检查每个 rank 的日志、`CUDA_VISIBLE_DEVICES`、Slurm task 数和网卡配置。Phase I Slurm 模板一次提交 8 个 task，启动器会在 allocation 内创建对应的 `srun` step。

### 10.6 CUDA OOM

先使用每卡 batch size 1；若仍 OOM，检查是否有其他进程占用显存、序列长度是否正确以及 kernel 是否退化。不要为通过 smoke test 而修改正式模型层数或 hidden size。

### 10.7 checkpoint 占满磁盘

降低保存频率，只保留最优 checkpoint、`last.ckpt` 和少量关键里程碑。删除前先确认文件可恢复且不属于正在运行的任务。

## 11. 尚未完成的正式工作

1. 收集并校验 26 个材料的 FASTA、索引、gene GFF3 和 TE GFF3。
2. 冻结 Phase II 的 23/1/2 genotype split。
3. 在目标 8×H200 节点运行 Phase I benchmark 和 smoke test，确定正式 batch size 与吞吐。
4. 完成当前 Phase I 固定全基因组 8K 正式训练并归档 best checkpoint。
5. 构建、校验和审计 NAM26 Phase II metadata。
6. 在目标节点完成 16K benchmark 与 smoke test，冻结 Phase II 总训练预算。
7. 从 Phase I best checkpoint 初始化 Phase II，并完成 held-out genotype 测试。

在以上步骤完成前，可以确认“代码和单材料数据链路已经跑通”，但不应表述为“26 材料完整训练已经完成”。
