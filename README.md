# OneMaize

OneMaize 是玉米全基因组 DNA language model。B73 Phase-I 8K 和 B73 Phase-II 16K 已训练完成；当前工作是准备 NAM26 数据并训练 16K population model。

> NAM26 必须从 **B73 Phase-I best checkpoint** 初始化，不能从 B73 Phase-II checkpoint 初始化。

## 当前状态

| Stage | Status |
| --- | --- |
| B73 Phase-I 8K full-genome | 已完成 |
| B73 Phase-II 16K region-aware | 已完成 |
| NAM26 schema-v3 metadata | 等待 26 个材料 |
| NAM26 Phase-II 16K | 代码和 8×H200 PBS 已就绪 |

## Step 1 — 准备文件

每个材料需要 5 类文件：

```text
raw/<genotype>/
├── genome.fa.gz       # BGZF FASTA
├── genome.fa.gz.fai   # FASTA index
├── genome.fa.gz.gzi   # BGZF index
├── genes.gff3.gz      # gene annotation
└── TE.gff3.gz         # TE annotation
```

文件名可以不同，后续 TSV 填写真实路径即可。`.fai/.gzi` 必须与 FASTA 完全匹配。

26 个材料：

```text
B73 B97 CML103 CML228 CML247 CML277 CML322 CML333 CML52 CML69
Hp301 Il14H Ki3 Ki11 Ky21 M162W M37W Mo18W MS71 NC350 NC358 Oh43
Oh7B P39 Tx303 Tzi8
```

正式划分为 `23 train / 1 val / 2 test`，B73 必须在 train。先确定最终 val/test 材料，再继续。

## Step 2 — 建立材料清单

```bash
cd /home/acd13855wx/projects/onemaize_project/onemaize
mkdir -p data/manifests
nano data/manifests/onemaize_26.tsv
```

TSV 第一行必须是：

```text
genotype	fasta	genes_gff3	te_gff3	split
```

然后填写 26 行，使用真实 Tab 分隔。例如：

```text
B73	/home/acd13855wx/projects/onemaize_project/onemaize/raw/B73/genome.fa.gz	/home/acd13855wx/projects/onemaize_project/onemaize/raw/B73/genes.gff3.gz	/home/acd13855wx/projects/onemaize_project/onemaize/raw/B73/TE.gff3.gz	train
B97	/home/acd13855wx/projects/onemaize_project/onemaize/raw/B97/genome.fa.gz	/home/acd13855wx/projects/onemaize_project/onemaize/raw/B97/genes.gff3.gz	/home/acd13855wx/projects/onemaize_project/onemaize/raw/B97/TE.gff3.gz	train
```

> 不要把 `\t` 两个字符写进文件；列之间必须是真正的 Tab。

## Step 3 — 构建 NAM26 metadata

```bash
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

成功后应生成：

```text
data/processed/onemaize_nam26/manifest.json
data/processed/onemaize_nam26/genomes.parquet
data/processed/onemaize_nam26/regions.parquet
data/processed/onemaize_nam26/DATA_STATS.md
```

## Step 4 — 校验数据

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

必须满足：26 个 genotype、23/1/2 split、B73=train、三类 candidate 均存在、16K sequence 可以读取。任何文件缺失、染色体名称不匹配或 schema error 都不要继续训练。

## Step 5 — 检查 B73 checkpoint

```bash
test -s runs/b73_phase1_8k/train/checkpoints_best/val_loss.ckpt
```

如果 checkpoint 不在这里，只修改 `pbs_scripts/03_nam26_phase2_train.pbs` 中的 `PHASE1_CKPT`。

## Step 6 — 提交 8×H200 训练

```bash
df -h
qsub pbs_scripts/03_nam26_phase2_train.pbs
```

PBS 会自动运行：

```text
audit -> validate -> benchmark -> smoke -> train
```

正式参数已经写好：16,384 bp、8×H200、batch/GPU=1、gradient accumulation=8、gene/non-repeat/TE-rich=`50/30/20`、`MAX_STEPS=15630`、`WARMUP_STEPS=782`、BF16、15% MLM、RC=0.5。

查看任务和日志：

```bash
qstat -u "$USER"
tail -f onemaize_nam26_phase2.o<JOB_ID>
```

训练结果位于：

```text
runs/nam26_phase2_16k/train/console.log
runs/nam26_phase2_16k/train/checkpoints_best/val_loss.ckpt
runs/nam26_phase2_16k/train/checkpoints_resume/last.ckpt
```

出现 NaN、Inf、Traceback、磁盘不足或 checkpoint 长时间不更新时，应停止检查。

## 训练完成后测试

在 GPU allocation 内运行：

```bash
export ONEMAIZE_DATA_DIR="$PWD/data/processed/onemaize_nam26"
export RUN_ROOT="$PWD/runs/nam26_phase2_16k"
export EVAL_CKPT="$RUN_ROOT/train/checkpoints_best/val_loss.ckpt"
bash scripts/run_onemaize_allcultivar_phase2_h200.sh test
```

输出：

```text
runs/nam26_phase2_16k/test/checkpoint_evaluation.csv
runs/nam26_phase2_16k/test/checkpoint_evaluation.md
```

详细规则见 [Data pipeline](docs/DATA_PIPELINE.md) 和 [Training details](docs/TRAINING_DETAILS.md)。
