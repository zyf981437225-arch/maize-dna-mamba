# OneMaize

Population-aware genomic language modeling for maize. 本仓库的正式工作流是：B73 exhaustive 8K Phase-I 预训练，然后从同一个 Phase-I checkpoint 分叉到 B73 reference Phase-II 或 NAM26 population Phase-II。

> B73 数据管线、8K slicing、A100 smoke/benchmark 和 16K candidate compatibility 已验证。
>
> NAM26 schema-v3 代码已实现，但最终 26 个材料尚未完成端到端 H200 验收与正式训练。

## Project status

| Component | B73 | NAM26 |
| --- | --- | --- |
| FASTA/index 与 annotation validation | VALIDATED | IMPLEMENTED, DATA PENDING |
| Phase-I 8K exhaustive manifest | VALIDATED | 不适用；Phase-I 固定为 B73 |
| Phase-I A100 smoke/benchmark | VALIDATED | 不适用 |
| Phase-I 8×H200 formal training | READY | PENDING |
| Phase-II schema-v3 candidates | VALIDATED | IMPLEMENTED, DATA PENDING |
| Phase-II 8×H200 formal training/test | READY | PENDING |
| schema-v4 explicit variant route | EXPERIMENTAL | EXPERIMENTAL |

## Formal pipeline

```text
B73 FASTA -> Phase-I: 8192 bp exhaustive windows -> B73 Phase-I best checkpoint
                                                        |
                         +------------------------------+-------------------+
                         |                                                  |
                         v                                                  v
             B73 Phase-II: 16K                                  NAM26 Phase-II: 16K
        reference region-aware model                         population schema-v3 model
                         |                                                  |
                         +---------------- validation / test ---------------+
```

Phase-I 使用 `window=8192, stride=8192` 的固定全基因组切片。Phase-II 使用 32,768-bp candidate 分类，再动态裁剪 16,384 bp；抽样比例为 gene-centered/non-repeat/TE-rich = 50/30/20。

## Teacher quick start: 8×H200 PBS

目标超算参数已写入三个编号 PBS：queue `rt_HF`、project `gaa50089`、one node、CUDA 12.6.1、Conda env `rna-mamba`、8 张可见 GPU。仓库默认位置：

```text
/home/acd13855wx/projects/onemaize_project/onemaize
```

| Order | Job | Input | Output |
| ---: | --- | --- | --- |
| 1 | `pbs_scripts/01_b73_phase1_train.pbs` | B73 FASTA/index + Phase-I parquet | B73 Phase-I checkpoints |
| 2A | `pbs_scripts/02_b73_reference_phase2_train.pbs` | B73 schema-v3 + Phase-I best | B73 Phase-II checkpoints |
| 2B | `pbs_scripts/03_nam26_phase2_train.pbs` | NAM26 schema-v3 + same Phase-I best | NAM26 Phase-II checkpoints |

提交命令只有三条：

```bash
cd /home/acd13855wx/projects/onemaize_project/onemaize
qsub pbs_scripts/01_b73_phase1_train.pbs
qsub pbs_scripts/02_b73_reference_phase2_train.pbs
qsub pbs_scripts/03_nam26_phase2_train.pbs
```

先完成 Job 1。Job 2A 与 2B 都从 Job 1 的 best checkpoint 初始化，二者彼此独立；Model B 不得从 B73 Phase-II checkpoint 初始化。

## 0. Clone and environment

```bash
git clone https://github.com/zyf981437225-arch/maize-dna-mamba.git onemaize
cd onemaize
git rev-parse HEAD
```

超算已有环境时：

```bash
module load cuda/12.6/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rna-mamba
python -c "import torch, mamba_ssm, pyarrow, hydra; print(torch.__version__, torch.cuda.device_count())"
```

新机器可参考 `setup_linux_env.sh` 建立 Python 3.10 环境。当前模型使用 `mamba_ssm.modules.mamba_simple.Mamba`，不是 Mamba2。

## 1. Prepare data

每个 genotype 需要 BGZF FASTA、匹配的 `.fai/.gzi`、gene GFF3 和 TE GFF3。推荐目录：

```text
raw/B73/{genome.fa.gz,genome.fa.gz.fai,genome.fa.gz.gzi,genes.gff3.gz,TE.gff3.gz}
raw/B97/...
...
raw/Tzi8/...
```

NAM26 panel：B73, B97, CML103, CML228, CML247, CML277, CML322, CML333, CML52, CML69, Hp301, Il14H, Ki3, Ki11, Ky21, M162W, M37W, Mo18W, MS71, NC350, NC358, Oh43, Oh7B, P39, Tx303, Tzi8。

正式 split 必须为 23 train / 1 val / 2 test，B73 必须在 train。held-out genotypes 必须在构建 metadata 前冻结。

完整输入说明、TSV schema 与 gate 见 [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)。

## 2. Build B73 Phase-I manifest

```bash
mkdir -p data/processed/onemaize_b73_phase1_8k
python scripts/build_b73_phase1_8k_manifest.py \
  --fasta raw/B73/genome.fa.gz \
  --output data/processed/onemaize_b73_phase1_8k/b73_phase1_8k_full_genome.parquet \
  --genotype B73 --chromosomes chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 \
  --window-size 8192 --stride 8192
python scripts/validate_b73_phase1_8k_manifest.py \
  --manifest data/processed/onemaize_b73_phase1_8k/b73_phase1_8k_full_genome.parquet \
  --fasta raw/B73/genome.fa.gz
```

通过条件：260,239 sequences；260,229 full windows；10 padded tails；chr1–chr10 共 2,131,846,805 bp，coverage 100%。如实际 parquet 路径不同，只修改 `01_b73_phase1_train.pbs` 中标记的那一行。

## 3. Build B73 Phase-II candidates

```bash
python scripts/build_onemaize_regions.py \
  --genotype B73 --fasta raw/B73/genome.fa.gz \
  --genes-gff3 raw/B73/genes.gff3.gz --te-gff3 raw/B73/TE.gff3.gz \
  --output-dir data/processed/onemaize_b73 \
  --val-seqid chr9 --test-seqid chr10 --max-n-fraction 0.10
python scripts/validate_onemaize_data.py \
  --data-dir data/processed/onemaize_b73 --context-length 16384
```

通过条件：`manifest.json`, `genomes.parquet`, `regions.parquet`, `DATA_STATS.md` 均存在；train/val/test 与三个 region classes 非空；无短于 16K 的 candidate。

## 4. Build NAM26 schema-v3 candidates

建立 `data/manifests/onemaize_26.tsv`，header 必须为：

```text
genotype	fasta	genes_gff3	te_gff3	split
```

然后运行：

```bash
python scripts/build_onemaize_regions.py \
  --input-manifest data/manifests/onemaize_26.tsv \
  --output-dir data/processed/onemaize_nam26 \
  --primary-context 8192 --extended-context 16384 \
  --candidate-span 32768 --candidate-stride 16384 \
  --gene-flank 5000 --repeat-threshold 0.5 --max-n-fraction 0.10 --formal
python scripts/validate_onemaize_data.py \
  --data-dir data/processed/onemaize_nam26 --context-length 16384 --formal
python scripts/audit_onemaize_allcultivar.py \
  --data-dir data/processed/onemaize_nam26 \
  --output-dir runs/nam26_preflight/audit --context-length 16384 --formal
```

任何 genotype/file/class 缺失、染色体命名不一致、FASTA alphabet 异常、23/1/2 split 错误或 B73 不在 train，均不得提交 Job 3。

## 5. Checkpoint locations

默认输出：

```text
runs/b73_phase1_8k/train/checkpoints_best/val_loss.ckpt
runs/b73_phase1_8k/train/checkpoints_resume/last.ckpt
runs/b73_reference_phase2_16k/train/checkpoints_best/val_loss.ckpt
runs/b73_reference_phase2_16k/train/checkpoints_resume/last.ckpt
runs/nam26_phase2_16k/train/checkpoints_best/val_loss.ckpt
runs/nam26_phase2_16k/train/checkpoints_resume/last.ckpt
```

正式预算：Phase-I 10 个完整 B73 passes，`MAX_STEPS=325300`；Phase-II 10 个 100,000-sequence virtual epochs，`MAX_STEPS=15630`, `WARMUP_STEPS=782`, gradient accumulation 8。详见 [docs/TRAINING_DETAILS.md](docs/TRAINING_DETAILS.md)。

## 6. Resume and test

PBS 默认启动新实验。断点恢复时，在对应 PBS 的 train 命令前设置 `RESUME_CKPT`，并取消 `PHASE1_CKPT`；`MAX_STEPS` 保持最终目标 global step，不是追加步数。不要混用初始化 checkpoint 与 resume checkpoint。

训练完成后可调用内部 launcher 的 `test` stage：

```bash
export ONEMAIZE_DATA_DIR="$PWD/data/processed/onemaize_b73"
export RUN_ROOT="$PWD/runs/b73_reference_phase2_16k"
export EVAL_CKPT="$RUN_ROOT/train/checkpoints_best/val_loss.ckpt"
bash scripts/run_onemaize_b73_phase2_h200.sh test
```

NAM26 对应使用 `scripts/run_onemaize_allcultivar_phase2_h200.sh test`。评估输出为 `checkpoint_evaluation.csv` 与 `checkpoint_evaluation.md`。

## Pre-flight checklist

- [ ] Git commit 已记录；Python/CUDA/Mamba import 通过；8 GPUs 可见。
- [ ] B73/NAM26 FASTA、FAI、GZI、gene GFF3、TE GFF3 完整。
- [ ] chromosome names 与 annotation seqid 一致。
- [ ] Phase-I manifest validation 通过。
- [ ] schema-v3 builder/validator/audit 通过。
- [ ] NAM26 split 已冻结为 23/1/2 且 B73=train。
- [ ] Phase-I best checkpoint 可读，Phase-II 没有错误串联。
- [ ] `df -h` 显示 checkpoint/output filesystem 空间充足。
- [ ] smoke 无 exception、NaN 或 Inf。

清单未全部通过时，不要启动正式训练。

## B73 validated reference

当前 A100 Phase-I benchmark：约 0.441 s/step、2.27 seq/s、18,564 tokens/s，估算单张 A100 完成一个 B73 full pass 约 31.9 h。该结果用于容量参考，不是 8×H200 实测值。真实报告见 `docs/audits/onemaize_b73/`。

## Repository map

```text
configs/experiment/   three formal Hydra experiment configs
configs/dataset/      Phase-I, B73 Phase-II, NAM26 Phase-II data contracts
pbs_scripts/          numbered teacher-facing H200 entrypoints
scripts/              builders, validators, audits, launchers, evaluation
src/onemaize/         schema, region, coverage and checkpoint contracts
src/dataloaders/      formal OneMaize datasets/datamodules
caduceus/             preserved model, memory and tokenizer core
docs/audits/          real-data and implementation evidence
```

Repository cleanup decisions见 [docs/REPO_CLEANUP_AUDIT.md](docs/REPO_CLEANUP_AUDIT.md)。schema-v4 variant/TE route 仅见 [docs/experimental/VARIANT_EXTENSION.md](docs/experimental/VARIANT_EXTENSION.md)，不属于正式流程。

## Troubleshooting

- Missing `.fai/.gzi`：确认 FASTA 是 BGZF；用 `samtools faidx genome.fa.gz` 生成匹配索引。
- `chr1` vs `1`：统一 FASTA 与两个 GFF3 的 seqid；不要在已构建 parquet 后静默改名。
- CUDA/Mamba kernel failure：核对 active Conda、PyTorch/CUDA ABI 与 `mamba_ssm` wheel 后重新 smoke。
- OOM：先停止；不得自行改变正式 batch/context/model。记录峰值后再决定是否修改方案。
- Disk full：运行 `df -h` 与 `du -sh runs 2>/dev/null`；清理前先确认 checkpoint 是否需要保留。
- NaN/Inf：停止正式提交，保存 console log 与最近 checkpoint，先复现 smoke/validation gate。

## License

See [LICENSE](LICENSE).
