# OneMaize

OneMaize 是基于玉米全基因组序列训练的 DNA language model。B73 的两个训练阶段已经完成；当前主要任务是准备 26 个 NAM 材料并训练 population-scale Phase-II 模型。

## Current status

| Stage | Status |
| --- | --- |
| B73 Phase-I：8K full-genome MLM | 已完成 |
| B73 Phase-II：16K region-aware continuation | 已完成 |
| NAM26 metadata / candidate regions | 等待 26 个材料 |
| NAM26 Phase-II：16K population training | 代码与 PBS 已准备好，等待数据 |

NAM26 训练从 **B73 Phase-I best checkpoint** 初始化，不从 B73 Phase-II checkpoint 初始化。

## NAM26 workflow

```text
26 × (FASTA + FAI + GZI + gene GFF3 + TE GFF3)
                         |
                         v
                  onemaize_26.tsv
                         |
                         v
        manifest.json + genomes.parquet + regions.parquet
                         |
                         v
             validate / audit / benchmark / smoke
                         |
                         v
                 8×H200 formal training
                         |
                         v
                    validation / test
```

## 1. Required files

每个 genotype 必须准备以下文件：

| File | Purpose |
| --- | --- |
| BGZF FASTA | DNA sequence |
| `.fai` | FASTA index |
| `.gzi` | BGZF index |
| gene GFF3 | gene-centered candidates |
| TE GFF3 | non-repeat / TE-rich candidates |

推荐目录结构：

```text
raw/
├── B73/
│   ├── genome.fa.gz
│   ├── genome.fa.gz.fai
│   ├── genome.fa.gz.gzi
│   ├── genes.gff3.gz
│   └── TE.gff3.gz
├── B97/
│   └── ...
└── Tzi8/
    └── ...
```

文件名可以不同，真实路径由下一步的 TSV 指定。FASTA 必须是 BGZF，`.fai/.gzi` 必须与 FASTA 完全匹配。

NAM26 名单：

```text
B73 B97 CML103 CML228 CML247 CML277 CML322 CML333 CML52 CML69
Hp301 Il14H Ki3 Ki11 Ky21 M162W M37W Mo18W MS71 NC350 NC358 Oh43
Oh7B P39 Tx303 Tzi8
```

正式 split 为 `23 train / 1 val / 2 test`，B73 必须在 train。开始构建前先确定最终 val/test genotype。

## 2. Create the 26-genotype TSV

在仓库根目录运行：

```bash
mkdir -p data/manifests
nano data/manifests/onemaize_26.tsv
```

第一行必须完全一致：

```text
genotype	fasta	genes_gff3	te_gff3	split
```

随后填写 26 行，例如：

```text
B73	/home/acd13855wx/projects/onemaize_project/onemaize/raw/B73/genome.fa.gz	/home/acd13855wx/projects/onemaize_project/onemaize/raw/B73/genes.gff3.gz	/home/acd13855wx/projects/onemaize_project/onemaize/raw/B73/TE.gff3.gz	train
B97	/home/acd13855wx/projects/onemaize_project/onemaize/raw/B97/genome.fa.gz	/home/acd13855wx/projects/onemaize_project/onemaize/raw/B97/genes.gff3.gz	/home/acd13855wx/projects/onemaize_project/onemaize/raw/B97/TE.gff3.gz	train
```

> TSV 必须使用真实 Tab 分隔，不能把 `\t` 字符原样写进文件。

## 3. Build NAM26 metadata

```bash
cd /home/acd13855wx/projects/onemaize_project/onemaize
module load cuda/12.6/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rna-mamba

python scripts/build_onemaize_regions.py \
  --input-manifest data/manifests/onemaize_26.tsv \
  --output-dir data/processed/onemaize_nam26 \
  --primary-context 8192 \
  --extended-context 16384 \
  --candidate-span 32768 \
  --candidate-stride 16384 \
  --gene-flank 5000 \
  --repeat-threshold 0.5 \
  --max-n-fraction 0.10 \
  --formal
```

成功后必须出现：

```text
data/processed/onemaize_nam26/manifest.json
data/processed/onemaize_nam26/genomes.parquet
data/processed/onemaize_nam26/regions.parquet
data/processed/onemaize_nam26/DATA_STATS.md
```

如果命令报告文件缺失、染色体名称不匹配、某个 genotype 缺少 region class、split 不是 23/1/2，停止处理，不要继续训练。

## 4. Validate before training

```bash
python scripts/validate_onemaize_data.py \
  --data-dir data/processed/onemaize_nam26 \
  --context-length 16384 \
  --formal

python scripts/audit_onemaize_allcultivar.py \
  --data-dir data/processed/onemaize_nam26 \
  --output-dir runs/nam26_preflight/audit \
  --context-length 16384 \
  --formal
```

通过条件：检测到 26 个 genotype、23/1/2 split 正确、B73 在 train、三类 candidates 均存在、16K sequence fetch 正常且没有 schema error。

## 5. Check the Phase-I checkpoint

NAM26 使用已经完成的 B73 Phase-I best checkpoint：

```bash
test -s runs/b73_phase1_8k/train/checkpoints_best/val_loss.ckpt
```

如果 checkpoint 保存在其他位置，只修改 `pbs_scripts/03_nam26_phase2_train.pbs` 中的 `PHASE1_CKPT`。

## 6. Submit NAM26 training

先确认磁盘和 GPU 环境：

```bash
df -h
python -c "import torch, mamba_ssm; print(torch.__version__, torch.cuda.device_count())"
```

提交正式任务：

```bash
qsub pbs_scripts/03_nam26_phase2_train.pbs
```

该 PBS 会依次自动执行：

```text
audit -> validate -> benchmark -> smoke -> train
```

正式参数已写入 PBS：8×H200、16,384 bp、batch/GPU=1、gradient accumulation=8、50/30/20 region sampling、`MAX_STEPS=15630`、`WARMUP_STEPS=782`、BF16、MLM 15%、RC augmentation 0.5。

## 7. Monitor training

```bash
qstat -u "$USER"
tail -f onemaize_nam26_phase2.o<JOB_ID>
```

训练输出：

```text
runs/nam26_phase2_16k/train/console.log
runs/nam26_phase2_16k/train/checkpoints_best/val_loss.ckpt
runs/nam26_phase2_16k/train/checkpoints_resume/last.ckpt
```

出现 `NaN`、`Inf`、Traceback、checkpoint 长时间不更新或磁盘空间不足时，应停止并检查，不要直接重新提交。

## 8. Test the trained model

在 GPU allocation 内运行：

```bash
export ONEMAIZE_DATA_DIR="$PWD/data/processed/onemaize_nam26"
export RUN_ROOT="$PWD/runs/nam26_phase2_16k"
export EVAL_CKPT="$RUN_ROOT/train/checkpoints_best/val_loss.ckpt"
bash scripts/run_onemaize_allcultivar_phase2_h200.sh test
```

结果文件：

```text
runs/nam26_phase2_16k/test/checkpoint_evaluation.csv
runs/nam26_phase2_16k/test/checkpoint_evaluation.md
```

## Model summary

正式模型使用现有双向 Caduceus/Mamba 实现：24 layers、`d_model=864`、约 121.2M parameters。它使用 `mamba_ssm.modules.mamba_simple.Mamba`，不是 Mamba2。

更详细的数据规则见 [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)，训练和 checkpoint 说明见 [docs/TRAINING_DETAILS.md](docs/TRAINING_DETAILS.md)。
