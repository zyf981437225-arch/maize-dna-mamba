# Maize DNA-Mamba / OneMaize

本项目训练一个单碱基分辨率的玉米 DNA 掩码语言模型。正式设计使用 B73 与
25 个 NAM founder 基因组，在基因型层面划分训练、验证和测试集合，并在保留
现有双向 Caduceus/Mamba、BCW writer、跨层 Memory Bank 和轻量读取器的前提下，
实现注释感知的 OneMaize 数据与训练流程。

> 当前结论：B73 单材料的数据处理、8K/16K 读取、MLM、完整模型反向传播、
> checkpoint 和单张 A100 训练已经验证成功。正式 26 材料代码已经具备，但尚未
> 在真实 NAM26 数据及 8 x H200 超算上完成实机验收。不要把 B73 结果描述成
> 已完成的 26 材料正式实验。

## 1. 当前阶段与未完成事项

已完成：

- B73 v5 FASTA、基因 GFF3 和 TE GFF3 已构建为 OneMaize schema v3 元数据；
- 完成全 FASTA `A/C/G/T/N` 审计并剔除 `N > 10%` 的候选区域；
- 验证基因区、非重复区和 TE 富集区 `50/30/20` 动态采样；
- 验证 8,192 bp 和 16,384 bp 动态裁剪；
- 验证训练集 50% reverse-complement augmentation；
- 验证 15% MLM 与 80/10/10 corruption；
- 验证 121,191,553 参数正式模型的 BF16 forward/backward/optimizer/checkpoint；
- 在单张 A100 80GB 上完成 B73 的 7 个完整 8K epoch，并保存最佳模型。

尚未完成：

1. 准备其余 25 个 NAM 材料及其匹配的 FASTA、FAI、GZI、gene GFF3 和 TE GFF3；
2. 由课题组确定正式的 `23 train / 1 validation / 2 test` 基因型划分，B73 必须在
   train；
3. 用真实 26 材料执行 `--formal` 元数据构建与质量验收；
4. 在目标 8 x H200 节点上验证 PyTorch/CUDA/Mamba 内核与 8 卡 DDP；
5. 在 H200 实测吞吐后冻结 8K/16K 的全局 batch、总步数、warmup 和作业时限；
6. 完成 NAM26 的 8K 训练、16K 续训及 held-out genotype 测试。

本仓库与原 RNA-Mamba 项目相互独立。`scripts/run_formal_8gpu.sh` 和
`docs/FORMAL_8GPU_TRAINING.md` 属于旧 RNA/m6A 流程；OneMaize 超算训练必须使用
本文的 `scripts/run_onemaize_h200.sh`。

## 2. 模型与数据设计

| 项目 | 正式设置 |
| --- | --- |
| Backbone | 24 层、`d_model=864`、weight-tied bidirectional Mamba |
| 双向策略 | forward/reverse 输出相加，双向权重共享 |
| Memory | BCW writer + cross-layer Memory Bank + lightweight reader |
| Memory 维度 | `d_sum=64`、`d_mem=64`、4 heads |
| Memory stride | write 6、read 2、最多 32 个 slots |
| 参数量 | 121,191,553 |
| Token | A/C/G/T/N 单碱基字符级 token |
| 训练目标 | 15% MLM，80% mask、10% random、10% unchanged |
| Phase I | 8,192 bp |
| Phase II | 16,384 bp |
| 区域采样 | 50% gene-centered、30% non-repeat、20% TE-rich |
| 基因型采样 | 先在当前 split 内均匀采样 genotype，再采样区域类型 |
| Reverse complement | 仅训练集启用，概率 0.5 |
| 精度 | BF16 |
| 正式划分 | 23 train、1 validation、2 test，B73 在 train |

保留的实现是仓库当前的 `mamba_ssm.modules.mamba_simple.Mamba`，没有切换为
Mamba2。当前模型是双向模型，但带 memory 的路径不声明严格 reverse-complement
equivariance；具体边界见 [RC_MEMORY_COMPATIBILITY.md](RC_MEMORY_COMPATIBILITY.md)。

## 3. B73 数据处理结果

B73 pilot 使用 chr1-8 训练、chr9 验证、chr10 测试。这是无染色体泄漏的工程
验证划分，不是最终的 genotype-held-out population evaluation。

### 3.1 B73 全基因组审计

| 指标 | 结果 |
| --- | ---: |
| 总长度 | 2,131,846,805 bp |
| protein-coding genes | 39,035 |
| TE union | 1,820,540,623 bp |
| TE coverage | 85.3973% |
| A | 26.5279% |
| C | 23.3629% |
| G | 23.3738% |
| T | 26.5568% |
| N | 0.1786% |
| 非 A/C/G/T/N 字符 | 0 |

### 3.2 B73 候选区域

构建后共有 164,150 个通过质量门的候选区域：

| Split | Gene-centered | Non-repeat | TE-rich | Total |
| --- | ---: | ---: | ---: | ---: |
| train, chr1-8 | 33,329 | 101 | 106,505 | 139,935 |
| validation, chr9 | 2,988 | 40 | 9,507 | 12,535 |
| test, chr10 | 2,704 | 2 | 8,974 | 11,680 |
| **total** | **39,021** | **143** | **124,986** | **164,150** |

8K 和 16K 实际 FASTA random access、`N <= 10%`、reverse complement 和 MLM
mask 比例均已通过验证。

## 4. B73 单张 A100 实测结果

### 4.1 Phase-0 benchmark

硬件和配置：NVIDIA A100 80GB PCIe、BF16、batch size 1、8,192 bp、24 层、
`d_model=864`。

| 指标 | Backbone only | Backbone + BCW/memory |
| --- | ---: | ---: |
| 参数量 | 121,050,720 | 121,191,553 |
| mean forward | 0.05342 s | 0.05529 s |
| mean backward | 0.14671 s | 0.14975 s |
| mean optimizer step | 0.20880 s | 0.21374 s |
| throughput | 39,233.5 tokens/s | 38,327.5 tokens/s |
| peak allocated | 8.204 GiB | 8.220 GiB |
| peak reserved | 8.365 GiB | 8.375 GiB |

Memory 路径增加约 2.36% step time 和 0.19% allocated peak memory。benchmark
在同一批次上执行少量更新，只用于证明 forward/backward/optimizer 和数值有限，
不能替代正式收敛判断。

### 4.2 B73 8K 阶段性训练

训练设置：单张 A100 80GB、BF16、batch size 1、梯度累积 4、全局 batch 4、
每个 epoch 100,000 个训练窗口与 2,048 个验证窗口。每个完整 epoch 对应
25,000 次 optimizer update。

| Epoch | Global step | Train loss | Validation loss | Train perplexity | Validation perplexity | Wall time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 25,000 | 1.080 | 1.09709 | 2.95 | 3.00 | 6:00:18 |
| 1 | 50,000 | 0.901 | 1.00976 | 2.46 | 2.74 | 6:01:42 |
| 2 | 75,000 | 0.808 | 0.96155 | 2.24 | 2.62 | 6:02:11 |
| 3 | 100,000 | 0.747 | 0.92955 | 2.11 | 2.53 | 6:02:04 |
| 4 | 125,000 | 0.700 | 0.90695 | 2.01 | 2.48 | 6:00:41 |
| 5 | 150,000 | 0.670 | 0.89139 | 1.95 | 2.44 | 6:00:12 |
| 6 | 175,000 | 0.651 | **0.87862** | 1.92 | 2.41 | 6:01:22 |

7 个完整 epoch 合计约 42 小时 8 分。随后在 epoch 7 约 9% 处人工停止，因为该
实验已经充分证明数据与训练链路稳定。训练期间没有 NaN、Inf、OOM 或 traceback；
GPU 训练时约占用 9.7GB，利用率通常为 94-97%。最佳 checkpoint 与 resume
checkpoint 均成功读取，元数据为 epoch 6、global step 175,000、481 个 state
tensors，单个完整 checkpoint 约 1.45GB。

该 run 是阶段结果，不是收敛结果：validation loss 在停止时仍下降，因此不能把
`0.87862` 宣称为 B73 的最终最优值，也不要把 checkpoint 提交到 GitHub。

### 4.3 时间参考应如何解释

在上述单 A100 配置下，一个 100,000-window 的 8K epoch 约需 6 小时；机械跑满
100 epoch 约需 25 天。NAM26 不会自动变成“26 倍 epoch 时间”，因为
`train_samples_per_epoch` 默认仍为 100,000。变化的是每个 genotype 在一个 epoch
中看到的样本数量。

正式划分有 23 个训练 genotype。默认 100,000 windows/epoch 时，每个训练材料
平均约获得：

```text
100000 / 23 = 4348 windows per genotype per epoch
```

因此，不应使用“训练了多少 epoch”比较 B73 与 NAM26，而应报告：

```text
global_batch = GPU数 x 每卡batch x 梯度累积
total_train_windows = optimizer_steps x global_batch
windows_per_train_genotype ≈ total_train_windows / 训练材料数
total_tokens = total_train_windows x context_length
```

若希望 23 个训练材料平均各看到 100,000 个窗口，8 卡默认全局 batch 8 时：

```text
MAX_STEPS = ceil(100000 x 23 / 8) = 287500
```

这只是样本预算示例，不是已经验证的最终超参数。8 x H200 的 step time、I/O、
NCCL 开销和最佳 batch 必须先 benchmark。生产设计是一个模型联合训练所有选定
材料，不是每个材料各训练一个模型；少于 26 材料属于 ablation，且不能使用
`--formal` 声称完成正式 OneMaize。

H200 launcher 默认 `REQUIRE_FORMAL=1`，会在占用 GPU 前检查 26 材料、23/1/2
划分、正式 panel 标记和 B73 train 约束。只有明确标注的材料数量 ablation 才允许
设置 `REQUIRE_FORMAL=0`。

## 5. 代码目录

```text
configs/experiment/onemaize_b73_8k.yaml     121M Phase-I 配置
configs/experiment/onemaize_b73_16k.yaml    121M Phase-II 配置
scripts/build_onemaize_regions.py           构建 genomes/regions parquet
scripts/validate_onemaize_data.py           正式数据、采样与 MLM 验收
scripts/benchmark_onemaize.py               单卡设备与模型 benchmark
scripts/run_onemaize_a100.sh                单卡 A100 开发/验收入口
scripts/run_onemaize_h200.sh                单节点 8 x H200 正式入口
slurm_scripts/run_onemaize_h200.slurm       Slurm 提交模板
docs/ONEMAIZE.md                             数据实现细节
docs/PLAN_COMPLIANCE.md                      计划符合性矩阵
```

虽然生产配置文件名保留了早期的 `b73` 名称，它们并没有把数据锁定为 B73；实际
加载的数据由 `ONEMAIZE_DATA_DIR` 决定，正式 NAM26 metadata 可以直接使用同一
8K/16K 模型配置。

## 6. 8 x H200 完整操作流程

以下命令假设使用一个包含 8 张 H200 的单节点 Slurm 作业。集群的 partition、
account、module 名称和最长 wall time 需要按站点规则调整。

### 6.1 克隆代码并建立环境

```bash
git clone https://github.com/zyf981437225-arch/maize-dna-mamba.git
cd maize-dna-mamba

conda create -n onemaize python=3.10 -y
conda activate onemaize
```

先按照超算 CUDA module 安装与驱动匹配、支持 H200 `sm_90` 的 PyTorch。当前
A100 验证基线是 PyTorch 2.2.0 + CUDA 12.1；目标集群可以使用管理员提供的更新
版本，但必须重新编译/安装与该 PyTorch 匹配的 `causal-conv1d` 和 `mamba-ssm`。

示例顺序：

```bash
# 先安装集群推荐的 CUDA-enabled PyTorch，再执行：
pip install -r requirements-core.txt
pip install ninja packaging pysam==0.22.0
pip install causal-conv1d==1.2.0.post2 --no-build-isolation
pip install mamba-ssm==1.2.2 --no-build-isolation
```

验证环境：

```bash
python - <<'PY'
import torch
import mamba_ssm
import causal_conv1d
import pysam
import pyarrow

print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("visible GPUs", torch.cuda.device_count())
print("GPU names", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("BF16", torch.cuda.is_bf16_supported())
PY

python scripts/check_onemaize_model_budget.py
```

预期参数量必须为 `121,191,553`，且 H200 节点必须报告 8 张可见 GPU 与 BF16
支持。环境不满足时不要提交长训练。

### 6.2 组织 26 材料数据

每个 genotype 需要：

- genome FASTA；若为 `.fa.gz`，必须是 BGZF；
- 与 FASTA 完全匹配的 `.fa.gz.fai`；
- 与 FASTA 完全匹配的 `.fa.gz.gzi`；
- 同一 assembly version 的 protein-coding gene GFF3；
- 同一 assembly version 的 TE GFF3。

目录可以自由组织，因为 input manifest 使用绝对路径。推荐：

```text
/shared/onemaize/raw/
  B73/
    genome.fa.gz
    genome.fa.gz.fai
    genome.fa.gz.gzi
    genes.gff3.gz
    TE.gff3.gz
  B97/
    ...
  ...
/shared/onemaize/manifests/
/shared/onemaize/processed/
/shared/onemaize/runs/
```

不要在 8 个 rank 各复制一份 FASTA；所有 rank 应读取同一份共享的只读文件。
高并发 BGZF random access 对并行文件系统敏感，正式训练前必须实测 I/O。

### 6.3 冻结 23/1/2 manifest

由课题组先确定 1 个 validation genotype 和 2 个 test genotypes。B73 必须为
train，且不要根据模型结果反复更换 held-out 材料。

创建 `/shared/onemaize/manifests/onemaize_26.tsv`：

```text
genotype\tfasta\tgenes_gff3\tte_gff3\tsplit
B73\t/shared/onemaize/raw/B73/genome.fa.gz\t/shared/onemaize/raw/B73/genes.gff3.gz\t/shared/onemaize/raw/B73/TE.gff3.gz\ttrain
B97\t/shared/onemaize/raw/B97/genome.fa.gz\t/shared/onemaize/raw/B97/genes.gff3.gz\t/shared/onemaize/raw/B97/TE.gff3.gz\ttrain
...
```

正式模式只接受精确的 26 个名称：

```text
B73 B97 CML103 CML228 CML247 CML277 CML322 CML333 CML52 CML69
Hp301 Il14H Ki3 Ki11 Ky21 M162W M37W Mo18W MS71 NC350 NC358
Oh43 Oh7B P39 Tx303 Tzi8
```

### 6.4 构建正式 NAM26 metadata

该步骤做一次 GFF3 坐标转换、TE union、FASTA alphabet/N 审计和候选区域构建。
建议在大内存 CPU 节点运行，不需要占用 8 张 H200。

```bash
cd /path/to/maize-dna-mamba
conda activate onemaize

python scripts/build_onemaize_regions.py \
  --input-manifest /shared/onemaize/manifests/onemaize_26.tsv \
  --output-dir /shared/onemaize/processed/onemaize_nam26 \
  --max-n-fraction 0.1 \
  --formal \
  2>&1 | tee /shared/onemaize/processed/build_nam26.log
```

输出目录应包含：

```text
manifest.json
genomes.parquet
regions.parquet
```

`--formal` 会拒绝错误的 genotype panel、非 23/1/2 划分、B73 被 held out、跳过
FASTA audit、错误坐标或缺失区域类型。第一次对真实 26 材料运行时，应把任何失败
当作数据兼容性问题调查，不要用 `--allow-missing-class` 绕过正式门禁。

### 6.5 同时验收 8K 与 16K 数据

```bash
export ONEMAIZE_DATA_DIR=/shared/onemaize/processed/onemaize_nam26
export RUN_ROOT=/shared/onemaize/runs/preflight

bash scripts/run_onemaize_h200.sh validate
```

成功后生成：

```text
$RUN_ROOT/validate/validation_8k.json
$RUN_ROOT/validate/validation_16k.json
```

报告必须确认：26 个 genotype、23/1/2 split、每个 split 可读、50/30/20 抽样在
tolerance 内、8K/16K 长度正确、MLM masked fraction 约为 15%。

### 6.6 运行代码测试

```bash
python -m compileall -q caduceus src scripts train.py

python -m pytest -q -p no:cacheprovider \
  tests/test_onemaize_pipeline.py \
  tests/test_maize_dna_pipeline.py

bash -n scripts/run_onemaize_h200.sh
bash -n slurm_scripts/run_onemaize_h200.slurm
```

当前 A100 服务器对上述 OneMaize/maize 数据测试收集到 14 项，结果为 14 passed。
这不替代 H200 上的 Mamba CUDA smoke。

### 6.7 单张 H200 benchmark

先申请一张 H200，并运行单卡 benchmark。它会记录模型参数、finite loss、step
time、tokens/s、peak memory 和 memory overhead。

```bash
export ONEMAIZE_DATA_DIR=/shared/onemaize/processed/onemaize_nam26
export RUN_ROOT=/shared/onemaize/runs/h200_preflight
export BATCH_SIZE=1
export NUM_WORKERS=4

bash scripts/run_onemaize_h200.sh benchmark
```

结果位于：

```text
$RUN_ROOT/benchmark/phase0_h200.json
```

根据 peak memory 再尝试 `BATCH_SIZE=2`、4 等值。每改变 context、batch、PyTorch、
Mamba 或 CUDA 版本都应重新 benchmark。不要直接按 H200 显存容量线性放大 batch，
内核 workspace 和 16K backward 必须实测。

### 6.8 8 卡完整模型 smoke

首先只跑 10 个 optimizer steps。该 stage 使用正式 121M 8K 模型、8 卡 DDP、
BF16、真实 NAM26 数据，并在训练前执行小规模 validation。

```bash
cd /path/to/maize-dna-mamba

export PROJECT_DIR=$PWD
export ONEMAIZE_DATA_DIR=/shared/onemaize/processed/onemaize_nam26
export RUN_ROOT=/shared/onemaize/runs/h200_preflight
export CONDA_ENV_NAME=onemaize
export NUM_DEVICES=8
export BATCH_SIZE=1
export GRAD_ACCUM=1
export NUM_WORKERS=4

sbatch --export=ALL,STAGE=smoke slurm_scripts/run_onemaize_h200.slurm
```

Slurm 模板默认申请一个节点、8 个 task 和 8 张 H200。不同超算可能使用
`--gpus-per-node=h200:8`、不同 partition/account 或 module；只修改资源指令和
环境加载，不要改成多节点，除非另行验证。

验收条件：

- 8 个 rank 全部启动且各绑定正确 GPU；
- 参数量为 121,191,553；
- loss 有限，无 NaN/Inf；
- 每张 GPU 都有计算负载；
- DDP 无 hang、NCCL error 或 duplicated-rank error；
- `$RUN_ROOT/smoke_8gpu/checkpoints_resume/last.ckpt` 可读取。

### 6.9 根据 H200 实测冻结训练预算

默认 H200 launcher 使用：

```text
NUM_DEVICES=8
BATCH_SIZE=1
GRAD_ACCUM=1
global_batch=8
TRAIN_SAMPLES_PER_EPOCH=100000
approximate optimizer steps per epoch=12500
```

推荐先用“每个训练 genotype 的目标窗口数”设计预算，再根据 validation loss 决定
是否延长。例：每个训练 genotype 先看 100,000 个窗口时，`MAX_STEPS=287500`。

warmup 可以先设为约一个 metadata epoch，即 12,500 steps，再根据 benchmark 和
老师的实验计划调整：

```bash
export MAX_STEPS=287500
export WARMUP_STEPS=12500
export CHECKPOINT_INTERVAL=12500
```

这组数值是保守的起始计划，不是最终结论。若修改 batch 或 gradient accumulation，
必须用前述公式重新计算 MAX_STEPS，不能保持步数不变而无意中扩大训练 token 数。

### 6.10 提交 Phase I 8K

```bash
export PROJECT_DIR=/path/to/maize-dna-mamba
export ONEMAIZE_DATA_DIR=/shared/onemaize/processed/onemaize_nam26
export RUN_ROOT=/shared/onemaize/runs/onemaize_nam26
export CONDA_ENV_NAME=onemaize
export NUM_DEVICES=8
export BATCH_SIZE=1
export BATCH_SIZE_EVAL=1
export GRAD_ACCUM=1
export NUM_WORKERS=4
export TRAIN_SAMPLES_PER_EPOCH=100000
export VAL_SAMPLES_PER_EPOCH=2048
export TEST_SAMPLES_PER_EPOCH=2048
export MAX_STEPS=287500
export WARMUP_STEPS=12500
export CHECKPOINT_INTERVAL=12500

sbatch --export=ALL,STAGE=8k slurm_scripts/run_onemaize_h200.slurm
```

如果集群最长 wall time 不足，允许作业到时退出后从 last checkpoint 续跑。生产
脚本只保留：

- validation loss 最低的 best checkpoint；
- 定期覆盖更新的 last/resume checkpoint。

它显式关闭逐 epoch 累积的大型 periodic checkpoints，避免每份约 1.45GB 的文件
持续占用存储。

### 6.11 查看日志、loss 和 GPU

```bash
squeue -u "$USER"
nvidia-smi
```

实时查看 Slurm 输出：

```bash
tail -f slurm-onemaize-h200-<jobid>.out
```

从 console log 提取最新进度：

```bash
watch -n 5 'tail -c 200000 /shared/onemaize/runs/onemaize_nam26/8k/console.log | tr "\r" "\n" | grep "Epoch " | tail -n 1'
```

每个完整 epoch 记录 train loss、validation loss、perplexity、global step、wall time、
GPU memory 和 tokens/s。判断停止主要看 validation loss，而不是某一个随机 batch
的瞬时 loss。

### 6.12 8K 断点续训

`RESUME_CKPT` 用于同一 stage 的训练状态恢复，它会恢复模型、optimizer、scheduler、
epoch 和 global step。`MAX_STEPS` 仍填写整个实验的最终目标，不是额外追加的步数。

```bash
export RESUME_CKPT=/shared/onemaize/runs/onemaize_nam26/8k/checkpoints_resume/last.ckpt
export MAX_STEPS=287500
export WARMUP_STEPS=12500

sbatch --export=ALL,STAGE=8k slurm_scripts/run_onemaize_h200.slurm
```

动态 MLM/裁剪数据在恢复后不承诺逐样本 bitwise replay，但模型、优化器、scheduler
和 step 状态会从 checkpoint 恢复。不要同时设置 `INIT_CKPT` 与 `RESUME_CKPT`。

### 6.13 提交 Phase II 16K

16K 是新的 curriculum stage，应使用最佳 8K checkpoint 作为权重初始化，不是把
8K optimizer 状态强行恢复到新阶段。

```bash
unset RESUME_CKPT
export INIT_CKPT=/shared/onemaize/runs/onemaize_nam26/8k/checkpoints_best/val_loss.ckpt
export MAX_STEPS=<根据16K benchmark确定>
export WARMUP_STEPS=<根据16K计划确定>
export CHECKPOINT_INTERVAL=<恢复间隔>

sbatch --export=ALL,STAGE=16k slurm_scripts/run_onemaize_h200.slurm
```

若 16K 作业中断，再使用 16K 自己的 last checkpoint：

```bash
unset INIT_CKPT
export RESUME_CKPT=/shared/onemaize/runs/onemaize_nam26/16k/checkpoints_resume/last.ckpt
sbatch --export=ALL,STAGE=16k slurm_scripts/run_onemaize_h200.slurm
```

### 6.14 Held-out genotype 测试

最终测试必须使用 validation 选择的 best checkpoint，且只运行一次正式 test 报告。

测试 8K checkpoint：

```bash
export EVAL_CKPT=/shared/onemaize/runs/onemaize_nam26/8k/checkpoints_best/val_loss.ckpt
sbatch --export=ALL,STAGE=test8k slurm_scripts/run_onemaize_h200.slurm
```

测试最终 16K checkpoint：

```bash
export EVAL_CKPT=/shared/onemaize/runs/onemaize_nam26/16k/checkpoints_best/val_loss.ckpt
sbatch --export=ALL,STAGE=test16k slurm_scripts/run_onemaize_h200.slurm
```

测试日志分别写入 `$RUN_ROOT/test8k/test.log` 或 `$RUN_ROOT/test16k/test.log`。

## 7. 输出与归档

每个生产 stage 至少保留：

```text
run_manifest.txt
console.log
checkpoints_best/val_loss.ckpt
checkpoints_resume/last.ckpt
```

`run_manifest.txt` 记录 Git commit、GPU、context、数据路径、全局 batch、训练样本数、
总步数、warmup、初始化/恢复 checkpoint 和 Slurm job ID。最终报告还应记录：

- NAM26 manifest 与 split 的冻结版本；
- `manifest.json` 和输入文件 checksum；
- H200 型号、GPU 数、CUDA/PyTorch/Mamba 版本；
- 8K/16K total windows、tokens、optimizer steps 和 wall time；
- 每个 epoch 的 train/validation loss；
- best checkpoint 的 epoch、global step 和 validation loss；
- 两个 held-out test genotype 的 test loss/perplexity。

checkpoint 不提交到 GitHub。GitHub 只保存代码、配置、操作说明和小型 JSON/表格
结果；大型模型放在课题组规定的模型存储中。

## 8. 停止标准

不要机械跑满预设 epoch。每个完整 validation 后检查：

- validation loss 是否仍明显下降；
- train loss 下降但 validation loss 连续回升，是否出现过拟合；
- 连续 3 次 validation 的绝对改善是否都小于约 0.005；
- 是否出现 NaN/Inf、异常 loss spike、I/O timeout 或 checkpoint 停止更新。

建议使用 best-checkpoint selection，并在连续 3 个 validation 已平台或回升时停止。
阈值 0.005 是 B73 阶段的操作参考，不是跨数据规模固定不变的统计结论。

## 9. 常见问题

### `Missing .fai/.gzi`

压缩 FASTA 必须是 BGZF，且 `.fa.gz`、`.fa.gz.fai`、`.fa.gz.gzi` 三者必须来自
同一版本。不要让训练节点临时为正式数据重建索引。

### GFF3 seqid 不存在于 FASTA

检查 GFF3 第一列与 FASTA header 的染色体命名。不要简单删除报错行；先确认是否
混用了 assembly/annotation 版本。

### `--formal` 拒绝 manifest

检查是否精确包含 B73 + 25 NAM、split 是否为 23/1/2、B73 是否为 train，以及
每个 genotype/split 是否具有三类候选区域。

### H200 上找不到 Mamba CUDA kernel

确认 PyTorch CUDA build、加载的 CUDA module 和编译扩展时的 `CUDA_HOME` 一致；
删除错误环境后重新安装与当前 PyTorch 匹配的 `causal-conv1d` 与 `mamba-ssm`。
不要静默切换到 Mamba2。

### NCCL hang

先确认 smoke 是否为单节点 8 GPU、每 GPU 一个 Slurm task；查看 rank/GPU 绑定，
再按超算文档设置 NCCL 网络变量。不要直接开始多节点训练。

### CUDA OOM

先将 `BATCH_SIZE=1`，保持 BF16，再减少每卡 batch；不要通过改变 context 或关闭
BCW/memory 来掩盖问题，因为那会改变正式模型定义。

### 磁盘增长过快

使用 `scripts/run_onemaize_h200.sh`。该脚本关闭逐 epoch 累积的 periodic
checkpoint，只保存 best 与覆盖更新的 last。日志仍可能很大，应由集群 log
rotation 或定期归档处理。

## 10. 开发验证

本地数据测试不需要 CUDA Mamba extension：

```bash
python -m pytest -q -p no:cacheprovider \
  tests/test_onemaize_pipeline.py \
  tests/test_maize_dna_pipeline.py
```

完整模型 forward/backward、8 卡 DDP、BF16 和 H200 性能验收必须在 Linux GPU
环境完成。计划符合性见 [docs/PLAN_COMPLIANCE.md](docs/PLAN_COMPLIANCE.md)，数据
实现细节见 [docs/ONEMAIZE.md](docs/ONEMAIZE.md)，架构审计见
[ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md)。
