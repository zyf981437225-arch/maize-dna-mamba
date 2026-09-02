# OneMaize

Population-aware genomic language modeling for maize.

本 README 是当前仓库的实验操作手册。项目现在明确分为两个从同一 B73 Phase-I checkpoint 分叉的模型；Model B **不得**从 B73 Phase-II checkpoint 初始化。命令以仓库当前 `main` 分支的真实脚本和参数为准。

> **B73 pipeline has been validated.**
>
> **NAM26 has not yet been run end-to-end on the final 26-genotype dataset.**
>
> **Current formal Model B status: READY FOR ALL-CULTIVAR DATA PREPARATION.**

## Project status

状态定义：

- **VALIDATED**：已使用真实 B73 数据或真实 A100 运行并留下审计记录。
- **IMPLEMENTED BUT NOT FORMALLY VALIDATED**：代码和测试存在，但最终 NAM26 数据或 8×H200 实验尚未完成。
- **PLANNED**：当前仓库没有可直接执行的完整入口。

| Component | B73 | NAM26 |
| --- | --- | --- |
| FASTA/FAI/GZI random access | **VALIDATED** | **IMPLEMENTED BUT NOT FORMALLY VALIDATED** |
| schema-v3 candidate metadata | **VALIDATED** | **IMPLEMENTED BUT NOT FORMALLY VALIDATED** |
| Phase-I 8K full-genome manifest | **VALIDATED** | **PLANNED**；没有多 genotype manifest/loader |
| Phase-I A100 benchmark/smoke | **VALIDATED** | **PLANNED** |
| Phase-I 8×H200 | **IMPLEMENTED BUT NOT FORMALLY VALIDATED**，仅 B73 | **PLANNED** |
| Phase-II 16K candidate compatibility | **VALIDATED** | **IMPLEMENTED BUT NOT FORMALLY VALIDATED** |
| Phase-II checkpoint continuation | **IMPLEMENTED BUT NOT FORMALLY VALIDATED** | **IMPLEMENTED BUT NOT FORMALLY VALIDATED** |
| genotype-held-out validation/test | pilot only | **IMPLEMENTED BUT NOT FORMALLY VALIDATED** |
| formal Model B schema-v3 population training | 不适用 | **IMPLEMENTED BUT NOT FORMALLY VALIDATED** |
| schema-v4 explicit variant extension | 不适用 | **EXPERIMENTAL，WAITING FOR REAL DATA** |
| final training | 未完成 | 未完成 |

当前 Phase-I production path 是 **B73-only**。当前正式设计从同一个 B73 Phase-I checkpoint 分叉为 **Model A：B73 Phase-II** 与 **Model B：all-cultivar schema-v3 region-aware Phase-II**。Model B 通过多 cultivar FASTA 暴露隐式学习自然 genomic variation；gene GFF3 和 TE GFF3 显式参与 candidate construction。schema-v4 explicit variant sampling 是未来实验扩展，不是正式 Model B 的必需入口。

## Training design: two independent models

```text
 B73 FASTA -> Phase-I 8K full genome -> B73 Phase-I best checkpoint
                                              |
                         +--------------------+--------------------+
                         |                                         |
                         v                                         v
 Model A: B73 Phase-II 16K region-aware       Model B: all-cultivar Phase-II 16K
 schema-v3 gene/non-repeat/TE-rich            schema-v3 gene/non-repeat/TE-rich
                         |                                         |
                         v                                         v
 Reference maize genome language model       Population-scale variation/TE-aware model
```

两条路线都使用 `train.pretrained_model_path` 从同一个 B73 Phase-I best checkpoint 严格加载模型权重，并新建 optimizer/scheduler。Model B 先均匀采 genotype，再按 0.5/0.3/0.2 采 gene/non-repeat/TE-rich；26 个材料中 23 个进入 optimizer，1 个 validation、2 个 test。这里的 variation-aware 指 multi-cultivar sequence exposure，不表示当前正式训练显式读取 SNP/SV/PAV annotation。

模型保持仓库现有的 `mamba_ssm.modules.mamba_simple.Mamba` 双向 Caduceus 实现，不是 Mamba2。正式模型为 24 layers、`d_model=864`、约 121,191,553 parameters，使用 BF16、15% MLM、80/10/10 corruption、single-base `A/C/G/T/N` tokenizer 和训练集 0.5 reverse-complement augmentation。

### Do not confuse these three lengths

| 参数 | 长度 | 用途 |
| --- | ---: | --- |
| Phase-I window/stride | 8,192 bp | B73 fixed full-genome model input |
| Phase-II candidate span | 32,768 bp | genome-wide repeat classification/candidate construction |
| Phase-II dynamic crop | 16,384 bp | 实际 Phase-II model input |

`32K candidate != 16K model input`。Phase-II 从 32K candidate 内动态选择 16K crop；gene-centered candidate 可以长于 32K，但必须至少为 16K。Phase-I 则严格要求 `window=8192` 且 `stride=8192`。

## Quick Start: from 26 raw genomes to training

本节给出执行顺序；每一步的输入、输出和通过条件在后文展开。

### Step 0 — Clone

```bash
git clone https://github.com/zyf981437225-arch/maize-dna-mamba.git
cd maize-dna-mamba
export PROJECT_DIR="$PWD"
git rev-parse HEAD
```

### Step 1 — Create the repository environment

仓库的可复现安装入口是 `setup_linux_env.sh`，要求 Python 3.10：

```bash
command -v python3.10
bash setup_linux_env.sh
source .venv/bin/activate
python --version
```

`caduceus_env.yml` 是完整环境快照，但包含导出机器的 site-specific prefix；不要直接依赖其中的 prefix。需要 Conda 时用命令行名称覆盖：

```bash
conda env create --name caduceus_env --file caduceus_env.yml
conda activate caduceus_env
```

### Step 2 — Set storage paths

只需在当前 shell 设置一次。把 `/data/onemaize` 改为目标并行文件系统上的实际目录：

```bash
export ONEMAIZE_ROOT=/data/onemaize
export RAW_ROOT="$ONEMAIZE_ROOT/raw"
export INPUT_MANIFEST="$ONEMAIZE_ROOT/manifests/onemaize_26.tsv"
export ONEMAIZE_DATA_DIR="$ONEMAIZE_ROOT/metadata/nam26_schema_v3"
export ONEMAIZE_B73_FASTA="$RAW_ROOT/B73/genome.fa.gz"
export ONEMAIZE_PHASE1_DIR="$ONEMAIZE_ROOT/metadata/b73_phase1_8k"
export ONEMAIZE_PHASE1_MANIFEST="$ONEMAIZE_PHASE1_DIR/b73_phase1_8k_full_genome.parquet"
export PHASE1_RUN_ROOT="$ONEMAIZE_ROOT/runs/b73_phase1_8k"
export PHASE2_RUN_ROOT="$ONEMAIZE_ROOT/runs/nam26_phase2_16k"

mkdir -p "$ONEMAIZE_ROOT/manifests" "$ONEMAIZE_PHASE1_DIR"
mkdir -p "$PHASE1_RUN_ROOT" "$PHASE2_RUN_ROOT"
```

### Step 3 — Prepare indexes and freeze `onemaize_26.tsv`

每个 genotype 需要 FASTA、匹配的 FAI/GZI、gene GFF3 和 TE GFF3。先确定最终 1 个 validation genotype 和 2 个 test genotypes；B73 必须保持为 train。按后文 “NAM26 preparation workflow” 建立精确为 26 行的输入表。

### Step 4 — Build B73 Phase-I manifest

```bash
python scripts/build_b73_phase1_8k_manifest.py \
  --fasta "$ONEMAIZE_B73_FASTA" \
  --output "$ONEMAIZE_PHASE1_MANIFEST" \
  --genotype B73 \
  --chromosomes chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 \
  --window-size 8192 \
  --stride 8192

python scripts/validate_b73_phase1_8k_manifest.py \
  --manifest "$ONEMAIZE_PHASE1_MANIFEST" \
  --fasta "$ONEMAIZE_B73_FASTA"
```

### Step 5 — Build and validate NAM26 schema-v3 candidates

```bash
python scripts/build_onemaize_regions.py \
  --input-manifest "$INPUT_MANIFEST" \
  --output-dir "$ONEMAIZE_DATA_DIR" \
  --max-n-fraction 0.10 \
  --formal

export RUN_ROOT="$ONEMAIZE_ROOT/runs/nam26_preflight"
export REQUIRE_FORMAL=1
bash scripts/run_onemaize_h200.sh validate

python scripts/audit_onemaize_phase2_candidate_lengths.py \
  --data-dir "$ONEMAIZE_DATA_DIR" \
  --context-length 16384 \
  --output-md "$ONEMAIZE_DATA_DIR/phase2_16k_candidate_audit.md"
```

### Step 6 — Run Phase-I benchmark, smoke, then train

```bash
export RUN_ROOT="$PHASE1_RUN_ROOT"
export NUM_DEVICES=8
export BATCH_SIZE=1
export BATCH_SIZE_EVAL=1
export NUM_WORKERS=8

sbatch --export=ALL,STAGE=benchmark slurm_scripts/run_onemaize_phase1_h200.slurm
sbatch --export=ALL,STAGE=smoke slurm_scripts/run_onemaize_phase1_h200.slurm
```

确认 benchmark 和 smoke 通过后，再按后文 “Phase I” 计算并设置 `MAX_STEPS`、`WARMUP_STEPS` 和 `CHECKPOINT_INTERVAL`，提交 `STAGE=train`。

### Step 7 — Fork the Phase-I checkpoint into Model A or Model B

```bash
export PHASE1_CKPT="$PHASE1_RUN_ROOT/train/checkpoints_best/val_loss.ckpt"
export NUM_DEVICES=8
export BATCH_SIZE=1
export BATCH_SIZE_EVAL=1
export GRAD_ACCUM=1
export NUM_WORKERS=8
```

Model A 将 `PHASE1_CKPT` 作为 B73 Phase-II 的 `INIT_CKPT`。Model B 在 schema-v3 all-cultivar audit 通过后，把同一个 `PHASE1_CKPT` 交给 `run_onemaize_allcultivar_phase2_h200.sh`。两条分支分别建立 optimizer/scheduler，互不 resume。schema-v4/VCF 不属于本步骤的前置条件。

## Environment and dependencies

`setup_linux_env.sh` 固定以下核心版本：

| Component | Repository environment |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.2.0 + cu121 |
| torchvision / torchaudio | 0.17.0 / 2.2.0 |
| mamba-ssm | 1.2.2，代码导入 `mamba_simple.Mamba` |
| causal-conv1d | 1.2.0.post2 |
| Hydra / OmegaConf | 1.3.2 / 2.3.0 |
| PyTorch Lightning | 1.8.6 |
| pyarrow | 14.0.2 |
| Biopython / pyfaidx | 1.81 / 0.8.1.1 |
| pytest | 8.0.2 |

`pysam` 是可选 FASTA backend，不在 `requirements-core.txt` 中；未安装时，BGZF 读取器使用 Biopython。`caduceus_env.yml` 快照包含 `pysam==0.22.0`。系统还需要 `samtools` 和 `bgzip` 来生成 FASTA 索引，它们不是 Python requirements 的一部分。

环境验收：

```bash
python - <<'PY'
import importlib
import torch

for name in ("mamba_ssm", "causal_conv1d", "pyarrow", "hydra", "pytorch_lightning", "Bio"):
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "imported"))
print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("BF16 supported", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
print("visible GPUs", torch.cuda.device_count())
PY
```

> 如果 Mamba/CUDA wheel 与目标集群 driver 或 PyTorch ABI 不兼容，应停在环境阶段处理；不要进入数据构建或训练。

## Data preparation

### Required file types and recommended layout

文件名不由代码固定；真实路径由 `onemaize_26.tsv` 指定。以下只是推荐布局：

```text
/data/onemaize/raw/
├── B73/
│   ├── genome.fa.gz
│   ├── genome.fa.gz.fai
│   ├── genome.fa.gz.gzi
│   ├── genes.gff3.gz
│   └── TE.gff3.gz
├── B97/
│   └── ...
├── CML103/
│   └── ...
└── Tzi8/
    └── ...
```

### NAM26 genotype panel

正式 `--formal` 模式按大小写不敏感方式验证以下 26 个名称；建议保持源码中的标准拼写：

|  |  |  |  |
| --- | --- | --- | --- |
| B73 | B97 | CML103 | CML228 |
| CML247 | CML277 | CML322 | CML333 |
| CML52 | CML69 | Hp301 | Il14H |
| Ki3 | Ki11 | Ky21 | M162W |
| M37W | Mo18W | MS71 | NC350 |
| NC358 | Oh43 | Oh7B | P39 |
| Tx303 | Tzi8 |  |  |

正式 split 必须为 `23 train / 1 val / 2 test`，且 B73 必须在 train。仓库当前没有已经冻结的正式 held-out genotype 文件。

> **The final held-out genotypes must be specified before formal manifest construction.**

### What each input file is used for

| 文件 | 用途 | Phase I | Phase II |
| --- | --- | :---: | :---: |
| FASTA | DNA sequence | ✓，当前仅 B73 | ✓，26 genotypes |
| `.fai` | contig length/offset random-access index | ✓ | ✓ |
| `.gzi` | BGZF compressed-offset index | ✓ | ✓ |
| gene GFF3 | protein-coding gene coordinates | — | ✓ |
| TE GFF3 | TE union coverage | — | ✓ |

当前 region builder 只接受 9-column GFF3，不直接接受 BED。

Gene candidate 使用 `feature_type == gene` 的坐标；若 GFF3 提供 `biotype` 或 `gene_biotype`，非 `protein_coding` gene 会被排除。候选区间为 gene body 上下游各 5 kb，并在必要时扩展到至少 16,384 bp。

TE-rich 不按 TE 条目数判断。`src/onemaize/regions.py` 先合并重叠 TE 区间，再计算：

```text
repeat coverage = unique TE-covered bp / candidate length
repeat coverage >= 0.5  -> te_rich
repeat coverage <  0.5  -> non_repeat
```

non-repeat 32K tile 默认排除与 protein-coding gene body 重叠的候选。所有候选的 `N` 比例默认不得超过 0.10。

## NAM26 preparation workflow

### Step 1 — Build or verify FASTA indexes

**Input**

```text
One FASTA per genotype. Compressed inputs must be BGZF, not ordinary gzip.
```

**Command**

已有发布方提供且与 FASTA 完全匹配的 `.fai/.gzi` 时直接保留。需要从 plain FASTA 建立 BGZF 和索引时：

```bash
command -v bgzip
command -v samtools
bgzip -@ 8 -c genome.fa > genome.fa.gz
samtools faidx genome.fa.gz
test -s genome.fa.gz.fai
test -s genome.fa.gz.gzi
```

**Expected output**

```text
genome.fa.gz
genome.fa.gz.fai
genome.fa.gz.gzi
```

**Pass condition**

三个文件非空，且 `samtools faidx genome.fa.gz chr1:1-100` 能返回序列。

**Do not continue if**

FASTA 是普通 gzip、索引来自另一版 FASTA、缺失 `.fai/.gzi`，或主染色体名称不能映射到 `chr1`–`chr10`。

### Step 2 — Freeze the 26-row input manifest

**Input**

```text
26 genotypes × (FASTA, gene GFF3, TE GFF3, split)
```

**Command / file format**

建立 TSV，header 必须精确为：

```text
genotype	fasta	genes_gff3	te_gff3	split
```

示例行：

```text
B73	/data/onemaize/raw/B73/genome.fa.gz	/data/onemaize/raw/B73/genes.gff3.gz	/data/onemaize/raw/B73/TE.gff3.gz	train
B97	/data/onemaize/raw/B97/genome.fa.gz	/data/onemaize/raw/B97/genes.gff3.gz	/data/onemaize/raw/B97/TE.gff3.gz	train
```

列名是 `genes_gff3`，不是 `gene_gff`。路径可使用任意文件名，但必须是实际可访问路径。TSV 内的 `$VARIABLE` 不会被 builder 自动展开，应写展开后的真实路径。

**Expected output**

```text
$INPUT_MANIFEST
```

**Pass condition**

精确 26 个非空数据行、名称与 NAM26 panel 一致、23/1/2 split、B73=train，且 held-out genotype 已在实验开始前冻结。

**Do not continue if**

held-out genotypes 尚未确定，或同一 genotype 重复出现。

### Step 3 — Run the raw-file gate

**Input**

```text
$INPUT_MANIFEST and every path referenced by it
```

**Command**

```bash
set -euo pipefail
expected_header=$'genotype\tfasta\tgenes_gff3\tte_gff3\tsplit'
[[ "$(head -n 1 "$INPUT_MANIFEST")" == "$expected_header" ]]
[[ "$(tail -n +2 "$INPUT_MANIFEST" | sed '/^[[:space:]]*$/d' | wc -l)" -eq 26 ]]

while IFS=$'\t' read -r genotype fasta genes_gff3 te_gff3 split; do
  [[ -n "$genotype" ]] || continue
  required=("$fasta" "$fasta.fai" "$genes_gff3" "$te_gff3")
  [[ "$fasta" == *.gz ]] && required+=("$fasta.gzi")
  for path in "${required[@]}"; do
    [[ -s "$path" ]] || { echo "MISSING: $genotype $path" >&2; exit 1; }
  done
done < <(tail -n +2 "$INPUT_MANIFEST")

echo "RAW FILE GATE PASSED"
```

**Expected output**

```text
RAW FILE GATE PASSED
```

**Pass condition**

命令返回 0，所有 26 行文件和压缩 FASTA 索引都存在。

**Do not continue if**

出现任何 `MISSING`。该 gate 只检查路径；FASTA alphabet、GFF3 坐标和 region class 完整性由下一步 builder 严格检查。

### Step 4 — Build schema-v3 candidates

**Input**

```text
$INPUT_MANIFEST
```

**Command**

```bash
python scripts/build_onemaize_regions.py \
  --input-manifest "$INPUT_MANIFEST" \
  --output-dir "$ONEMAIZE_DATA_DIR" \
  --primary-context 8192 \
  --extended-context 16384 \
  --candidate-span 32768 \
  --candidate-stride 16384 \
  --gene-flank 5000 \
  --repeat-threshold 0.5 \
  --max-n-fraction 0.10 \
  --formal \
  | tee "$ONEMAIZE_ROOT/metadata/nam26_build.log"
```

首次构建时 output directory 必须不存在。只有明确要替换旧 metadata 时才添加 `--overwrite`。

**Expected output**

```text
$ONEMAIZE_DATA_DIR/manifest.json
$ONEMAIZE_DATA_DIR/genomes.parquet
$ONEMAIZE_DATA_DIR/regions.parquet
$ONEMAIZE_DATA_DIR/DATA_STATS.md
```

`manifest.json` 的 `schema_version` 应为 3。

**Pass condition**

- `genotype_count == 26`
- `genotype_split_counts == {train: 23, val: 1, test: 2}`
- `formal_split_validated == true`
- `expected_genotype_panel_validated == true`
- `required_train_genotypes` 包含 B73
- 每个 represented genotype/split 都有 gene-centered、non-repeat、TE-rich 三类候选
- FASTA alphabet audit 通过，无非 `A/C/G/T/N` 字符

**Do not continue if**

builder 报 chromosome/seqid mismatch、GFF3 malformed、缺少 region class、N 比例问题或 panel/split 错误。builder 使用临时目录并在成功后原子发布输出；失败时不要手工拼接不完整 parquet。

### Step 5 — Validate schema-v3 reads at both lengths

**Input**

```text
$ONEMAIZE_DATA_DIR/{manifest.json,genomes.parquet,regions.parquet}
26 indexed FASTA files
```

**Command**

```bash
export RUN_ROOT="$ONEMAIZE_ROOT/runs/nam26_preflight"
export REQUIRE_FORMAL=1
bash scripts/run_onemaize_h200.sh validate
```

该 stage 不需要 GPU；它分别调用 `validate_onemaize_data.py` 验证 8K 和 16K random access、50/30/20 sampling、MLM masking 和 formal metadata gate。

**Expected output**

```text
$RUN_ROOT/validate/validation_8k.json
$RUN_ROOT/validate/validation_16k.json
```

**Pass condition**

命令返回 0；两个 JSON 均包含 26 genotypes 和 train/val/test 三个 split，抽样比例在默认容差内，读取长度分别为 8,192 和 16,384。

**Do not continue if**

任何 FASTA fetch、schema、split、sampling fraction 或 MLM check 失败。

### Step 6 — Audit 16K candidate compatibility

**Input**

```text
$ONEMAIZE_DATA_DIR/regions.parquet
```

**Command**

```bash
python scripts/audit_onemaize_phase2_candidate_lengths.py \
  --data-dir "$ONEMAIZE_DATA_DIR" \
  --context-length 16384 \
  --output-md "$ONEMAIZE_DATA_DIR/phase2_16k_candidate_audit.md"
```

**Expected output**

```text
$ONEMAIZE_DATA_DIR/phase2_16k_candidate_audit.md
```

**Pass condition**

表格中 `all.short == 0`，且三种 region class 的总数均大于 0。

**Do not continue if**

任何 candidate 小于 16,384 bp。当前审计脚本的结论句仍写作 B73，但数值会统计传入 metadata 的全部行；对 NAM26 应以表格中的 `short` 数值为 gate。

## Phase I — exhaustive 8K full-genome MLM (B73 only)

> Phase I guarantees B73 sequence coverage by deterministic genome slicing rather than probabilistic candidate sampling.

当前 production Phase-I 不读取 `regions.parquet`，只读取 B73 fixed-window parquet。它不等价于旧的 100,000 random windows/epoch。

### 1 — Build manifest

**Input**：B73 BGZF FASTA、同名 `.fai/.gzi`。

**Command**

```bash
python scripts/build_b73_phase1_8k_manifest.py \
  --fasta "$ONEMAIZE_B73_FASTA" \
  --output "$ONEMAIZE_PHASE1_MANIFEST" \
  --genotype B73 \
  --chromosomes chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 \
  --window-size 8192 \
  --stride 8192
```

**Expected output**：`$ONEMAIZE_PHASE1_MANIFEST`。

**Pass condition**：stdout 报告 260,239 keep/PAD windows、10 tail windows、coverage 1.0。

**Do not continue if**：B73 主染色体不是 `chr1`–`chr10`、总 bp 与 B73 v5 审计不符，或 output 已存在但来源不明确。

### 2 — Validate manifest and slicing statistics

**Command**

```bash
python scripts/validate_b73_phase1_8k_manifest.py \
  --manifest "$ONEMAIZE_PHASE1_MANIFEST" \
  --fasta "$ONEMAIZE_B73_FASTA" \
  | tee "$ONEMAIZE_PHASE1_DIR/validation.json"

python scripts/stat_b73_8192_full_genome_slicing.py \
  --fasta "$ONEMAIZE_B73_FASTA" \
  --output-md "$ONEMAIZE_PHASE1_DIR/slicing_stats.md" \
  --output-csv "$ONEMAIZE_PHASE1_DIR/slicing_stats.csv"
```

**Expected output**：validation JSON、slicing Markdown/CSV 和 manifest parquet。

**Pass condition**：

| Metric | Expected B73 v5 value |
| --- | ---: |
| chr1–chr10 bp | 2,131,846,805 |
| full 8,192-bp windows | 260,229 |
| tail windows | 10 |
| total sequences | 260,239 |
| tail valid bp | 50,837 |
| padded bp | 31,083 |
| valid-bp coverage | 100% |

**Do not continue if**：validator 返回非 0、存在 overlap/gap、tail/padding 不一致或 FASTA/manifest 长度不一致。

### 3 — Benchmark on the target node

**Command**

```bash
export RUN_ROOT="$PHASE1_RUN_ROOT"
export NUM_DEVICES=8
export BATCH_SIZE=1
export NUM_WORKERS=8
export BENCHMARK_WARMUP_STEPS=5
export BENCHMARK_STEPS=100
export BENCHMARK_IO_STEPS=32

sbatch --export=ALL,STAGE=benchmark \
  slurm_scripts/run_onemaize_phase1_h200.slurm
```

交互式单 GPU 节点可用：

```bash
NUM_DEVICES=1 bash scripts/run_onemaize_phase1_h200.sh benchmark
```

**Expected output**：`$PHASE1_RUN_ROOT/benchmark.json`。

重点读取以下字段：

- `mean_step_seconds_max_rank`
- `global_samples_per_second`
- `global_tokens_per_second`
- `estimated_epoch_seconds`
- `io.tokens_per_second`
- `losses`

Phase-I benchmark 当前不记录 peak GPU memory；显存应在 smoke 阶段使用 `nvidia-smi` 监控。

**Pass condition**：loss 全部有限，8 个 rank 正常结束，吞吐和 I/O 字段为正。

**Do not continue if**：出现 NaN/Inf、CUDA/Mamba kernel error、I/O 明显停滞或 rank failure。

### 4 — Eight-GPU smoke test

**Command**

```bash
export RUN_ROOT="$PHASE1_RUN_ROOT"
export NUM_DEVICES=8
export BATCH_SIZE=1
export BATCH_SIZE_EVAL=1
export NUM_WORKERS=8
export SMOKE_STEPS=20
export SMOKE_WARMUP_STEPS=2

sbatch --export=ALL,STAGE=smoke \
  slurm_scripts/run_onemaize_phase1_h200.slurm
```

**Expected output**：`$PHASE1_RUN_ROOT/smoke/console.log` 和 Slurm stdout/stderr。smoke launcher 会关闭完整周期 checkpoint。

**Pass condition**：8 ranks 正常启动；无 NCCL hang、OOM、NaN/Inf 或 traceback；train loss 有下降趋势；GPU 利用率合理。

**Do not continue if**：任一 rank 失败，或实际输入不是 8,192 tokens。

### 5 — Freeze epoch and train

> **Phase-I epoch is not a virtual random epoch. One epoch corresponds to one full pass over the fixed full-genome manifest.**

B73 有 260,239 unique slices。`DistributedSampler(drop_last=False)` 在 world size 8 时：

```text
per rank              = ceil(260239 / 8) = 32,530 samples
global sampler draws  = 32,530 × 8       = 260,240
unique slices         = 260,239
padding duplicate     = 1
```

Phase-I launcher 固定 `accumulate_grad_batches=1`。步数公式为：

```text
optimizer steps per epoch = ceil(32,530 / batch_size_per_gpu)
```

先完成 benchmark；未测得目标 8×H200 吞吐前，不要冻结正式 `MAX_STEPS`。以下仅是 `BATCH_SIZE=1` 的一个完整 B73 epoch：

```bash
export RUN_ROOT="$PHASE1_RUN_ROOT"
export NUM_DEVICES=8
export BATCH_SIZE=1
export BATCH_SIZE_EVAL=1
export NUM_WORKERS=8
export MAX_STEPS=32530
export WARMUP_STEPS=1626
export CHECKPOINT_INTERVAL=2000

df -h "$RUN_ROOT"
sbatch --export=ALL,STAGE=train \
  slurm_scripts/run_onemaize_phase1_h200.slurm
```

**Expected output**

```text
$PHASE1_RUN_ROOT/train/console.log
$PHASE1_RUN_ROOT/train/checkpoints_best/val_loss.ckpt
$PHASE1_RUN_ROOT/train/checkpoints_resume/last.ckpt
```

**Pass condition**：validation loss 有限；checkpoint 持续更新；完整 epoch 后 coverage metrics 接近：

```text
train/phase1_unique_regions = 260239
train/phase1_genomic_bp_coverage = 1.0
```

1 个 DDP padding duplicate 是预期行为，不代表额外 genomic coverage。

**Do not continue if**：coverage 不完整、checkpoint 停止更新、磁盘空间不足或 validation loss 数值异常。

### 6 — Resume exact Phase-I state

```bash
export RUN_ROOT="$PHASE1_RUN_ROOT"
export RESUME_CKPT="$PHASE1_RUN_ROOT/train/checkpoints_resume/last.ckpt"
export MAX_STEPS=32530
export WARMUP_STEPS=1626

sbatch --export=ALL,STAGE=train \
  slurm_scripts/run_onemaize_phase1_h200.slurm
```

`MAX_STEPS` 是最终目标 global step，不是追加步数。`RESUME_CKPT` 恢复 model、optimizer、scheduler 和 global step。

### 7 — Evaluate the Phase-I checkpoint

```bash
unset RESUME_CKPT
export RUN_ROOT="$PHASE1_RUN_ROOT"
export EVAL_CKPT="$PHASE1_RUN_ROOT/train/checkpoints_best/val_loss.ckpt"

sbatch --export=ALL,STAGE=test \
  slurm_scripts/run_onemaize_phase1_h200.slurm
```

该 test 是 B73 fixed-manifest 的确定性技术检查，不是 genotype-held-out population evaluation。

## Model A Phase II — B73 16K annotation-aware context adaptation

Phase-II 不是从头训练。`INIT_CKPT` 通过 `train.pretrained_model_path` 严格加载 Phase-I model state，并建立新的 Phase-II optimizer/scheduler；`RESUME_CKPT` 则恢复已经开始的 Phase-II 完整训练状态。两者不能同时设置。Model A 正式数据目录只包含 B73；正式 all-cultivar Model B 使用后文独立 launcher。

Phase-II region sampling：

```text
gene GFF3 -> protein-coding gene ±5 kb -> gene-centered
TE GFF3   -> merged union coverage     -> non-repeat / TE-rich
                                              |
uniform genotype -> class 50/30/20 -> candidate -> random 16K crop -> MLM
```

当前 `scripts/run_onemaize_h200.sh 16k` 实际调用 `configs/experiment/onemaize_b73_16k.yaml`。虽然配置名含 `b73`，dataset 从 `$ONEMAIZE_DATA_DIR` 读取 formal NAM26 schema-v3；launcher 不调用新增的 `onemaize_b73_phase2_16k_region_aware.yaml`。两份 Phase-II config 的 optimizer/accumulation defaults 不完全相同，因此正式实验必须按 launcher 的实际命令归档配置，不能把另一份 YAML 当成已运行配置。

### 1 — Single-H200 16K benchmark

**Prerequisites**：formal validation 通过，candidate audit `short=0`。

```bash
export RUN_ROOT="$PHASE2_RUN_ROOT"
mkdir -p "$RUN_ROOT"

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
  --output-json "$RUN_ROOT/benchmark_16k_single_h200.json"
```

**Expected output**：`$PHASE2_RUN_ROOT/benchmark_16k_single_h200.json`，包含 input-pipeline throughput、forward/backward/step time、peak allocated/reserved memory、loss 和 parameter count。

**Pass condition**：`parameter_target_met=true`、loss 全部有限、batch 1 不 OOM。

当前仓库没有专门的 **distributed 8×H200 Phase-II benchmark JSON** 入口；8 卡吞吐需通过下一步短 `STAGE=16k` 作业的日志实测。

### 2 — Eight-GPU Phase-II smoke

```bash
export RUN_ROOT="$PHASE2_RUN_ROOT"
export INIT_CKPT="$PHASE1_RUN_ROOT/train/checkpoints_best/val_loss.ckpt"
unset RESUME_CKPT EVAL_CKPT
export NUM_DEVICES=8
export BATCH_SIZE=1
export BATCH_SIZE_EVAL=1
export GRAD_ACCUM=1
export NUM_WORKERS=8
export TRAIN_SAMPLES_PER_EPOCH=100000
export VAL_SAMPLES_PER_EPOCH=2048
export TEST_SAMPLES_PER_EPOCH=2048
export MAX_STEPS=20
export WARMUP_STEPS=2
export CHECKPOINT_INTERVAL=20
export REQUIRE_FORMAL=1

df -h "$RUN_ROOT"
sbatch --export=ALL,STAGE=16k \
  slurm_scripts/run_onemaize_h200.slurm
```

**Expected output**

```text
$PHASE2_RUN_ROOT/16k/run_manifest.txt
$PHASE2_RUN_ROOT/16k/console.log
$PHASE2_RUN_ROOT/16k/checkpoints_best/
$PHASE2_RUN_ROOT/16k/checkpoints_resume/
```

**Pass condition**：Phase-I state strict-load 成功，无 missing key/shape mismatch；8 ranks 正常；16K loss 有限；无 OOM/NCCL hang；日志记录实际 global batch 和 context。

**Do not continue if**：checkpoint 加载失败、formal gate 失败、任何 rank 失败或磁盘不足。该 launcher 会保存 checkpoint；smoke 前也要确认空间。

### 3 — Freeze the Phase-II budget and train

Phase-II 是 virtual random epoch，默认 `TRAIN_SAMPLES_PER_EPOCH=100000`；它与 Phase-I full-manifest epoch 无关。

```text
global batch = NUM_DEVICES × BATCH_SIZE × GRAD_ACCUM
optimizer steps per virtual epoch = ceil(TRAIN_SAMPLES_PER_EPOCH / global batch)
total sampled sequences = MAX_STEPS × global batch
total model tokens = total sampled sequences × 16384
```

根据 single-H200 benchmark 和 8-card smoke 冻结以下变量后再提交：

```bash
read -r -p "Approved Phase-II MAX_STEPS: " MAX_STEPS
read -r -p "Approved Phase-II WARMUP_STEPS: " WARMUP_STEPS
read -r -p "Approved CHECKPOINT_INTERVAL: " CHECKPOINT_INTERVAL
[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]
[[ "$WARMUP_STEPS" =~ ^[0-9]+$ ]]
[[ "$CHECKPOINT_INTERVAL" =~ ^[1-9][0-9]*$ ]]
(( WARMUP_STEPS < MAX_STEPS ))
export MAX_STEPS WARMUP_STEPS CHECKPOINT_INTERVAL
: "${INIT_CKPT:?Set the Phase-I best checkpoint}"

export RUN_ROOT="$PHASE2_RUN_ROOT"
export REQUIRE_FORMAL=1
sbatch --export=ALL,STAGE=16k \
  slurm_scripts/run_onemaize_h200.slurm
```

### 4 — Resume exact Phase-II state

```bash
unset INIT_CKPT EVAL_CKPT
export RESUME_CKPT="$PHASE2_RUN_ROOT/16k/checkpoints_resume/last.ckpt"
export RUN_ROOT="$PHASE2_RUN_ROOT"

sbatch --export=ALL,STAGE=16k \
  slurm_scripts/run_onemaize_h200.slurm
```

保持正式实验原来的 `MAX_STEPS`、`WARMUP_STEPS`、batch、gradient accumulation 和 data directory。不要同时设置 `INIT_CKPT`。

### 5 — Validation and held-out test

训练期间 model selection 使用 formal validation genotype。最终 test：

```bash
unset INIT_CKPT RESUME_CKPT
export RUN_ROOT="$PHASE2_RUN_ROOT"
export EVAL_CKPT="$PHASE2_RUN_ROOT/16k/checkpoints_best/val_loss.ckpt"
export REQUIRE_FORMAL=1

sbatch --export=ALL,STAGE=test16k \
  slurm_scripts/run_onemaize_h200.slurm
```

**Expected output**：`$PHASE2_RUN_ROOT/test16k/run_manifest.txt` 和 `test.log`。

**Pass condition**：test 只读取 TSV 中冻结的 2 个 test genotypes；无 train genotype 混入；指标有限且 checkpoint 与 run manifest 对应。

## Model B Phase II — formal all-cultivar 16K region-aware continuation

> **Current status: READY FOR ALL-CULTIVAR DATA PREPARATION.** 正式 Model B 不要求 VCF、SV/PAV 或 cultivar-specific TE insertion 文件；目前仍缺最终 26-genotype FASTA/GFF3 corpus 的完整构建与实机 audit。

正式 Model B 只使用 schema-v3：

```text
23 train genotypes (B73 included)
          |
uniform genotype sampling
          |
gene / non-repeat / TE-rich = 0.5 / 0.3 / 0.2
          |
dynamic 16,384-bp crop
          |
MLM continuation from B73 Phase-I best
```

不同 cultivar FASTA 自然携带 SNP、indel、SV、PAV 和 TE-related sequence differences，模型通过 population-scale sequence exposure 隐式学习这些差异。TE GFF3 则被显式用于 union repeat coverage 和 `te_rich` candidate classification。candidate pool 中各类数量不需要等于 50/30/20；训练 sampler 才执行 50/30/20。

### 1 — Audit and validate schema-v3

```bash
export RUN_ROOT="$ONEMAIZE_ROOT/runs/allcultivar_phase2"
export ONEMAIZE_DATA_DIR="$ONEMAIZE_ROOT/metadata/nam26_schema_v3"

bash scripts/run_onemaize_allcultivar_phase2_h200.sh audit
bash scripts/run_onemaize_allcultivar_phase2_h200.sh validate
```

audit 输出每个 genotype 的 FASTA/FAI/GZI/gene GFF3/TE GFF3 状态、split、三类 candidate count、candidate length、N fraction、repeat fraction 和空 pool。少量 candidate 只产生 warning；空 pool、short candidate、缺文件或 split leakage 才会阻止训练。

### 2 — Benchmark and smoke

```bash
bash scripts/run_onemaize_allcultivar_phase2_h200.sh benchmark

export PHASE1_CKPT="$PHASE1_RUN_ROOT/train/checkpoints_best/val_loss.ckpt"
unset RESUME_CKPT
export MAX_STEPS=10 WARMUP_STEPS=2
bash scripts/run_onemaize_allcultivar_phase2_h200.sh smoke
```

launcher 会拒绝非 B73、非 8K、非 `full_genome` 的初始化 checkpoint。Model A 的 B73 Phase-II checkpoint 不能作为 Model B 初始化 checkpoint。

### 3 — Formal train and exact resume

先根据 benchmark/smoke 冻结 `MAX_STEPS`、warmup、batch 和 checkpoint interval。0.5/0.3/0.2 是当前设计，**sampling ratio should later be evaluated by ablation**。

```bash
export PHASE1_CKPT="$PHASE1_RUN_ROOT/train/checkpoints_best/val_loss.ckpt"
export MAX_STEPS=<benchmark-approved-steps>
export WARMUP_STEPS=<approved-warmup>
export CHECKPOINT_INTERVAL=<approved-interval>
bash scripts/run_onemaize_allcultivar_phase2_h200.sh train

# Exact Model-B resume only; restores optimizer/scheduler/global step.
unset PHASE1_CKPT
export RESUME_CKPT="$RUN_ROOT/train/checkpoints_resume/last.ckpt"
bash scripts/run_onemaize_allcultivar_phase2_h200.sh train
```

resume gate 要求 checkpoint 内记录的 16K region-aware `dataset.data_dir` 与当前 all-cultivar metadata 相同，避免把 Model A checkpoint 误当作 Model B resume。

### 4 — PBS 8×H200

编辑并提交 [`pbs_scripts/run_onemaize_allcultivar_phase2.pbs`](pbs_scripts/run_onemaize_allcultivar_phase2.pbs)：

```bash
qsub -v ONEMAIZE_DATA_DIR="$ONEMAIZE_DATA_DIR",PHASE1_CKPT="$PHASE1_CKPT",MAX_STEPS="$MAX_STEPS",WARMUP_STEPS="$WARMUP_STEPS",CHECKPOINT_INTERVAL="$CHECKPOINT_INTERVAL" pbs_scripts/run_onemaize_allcultivar_phase2.pbs
```

PBS 示例使用 `rt_HF`、project `gaa50089`、8 GPUs、CUDA 12.6.1 和 `rna-mamba` 环境；路径只存在于 PBS example，不进入 Python dataset/model。

### 5 — Fair schema-v3-only evaluation

不传 `--variant-data-dir` 即可在相同 deterministic schema-v3 16K set 上比较三条 checkpoint：

```bash
python scripts/evaluate_onemaize_checkpoints.py \
  --checkpoint "phase1=$PHASE1_CKPT" \
  --checkpoint "reference_phase2=$MODEL_A_CKPT" \
  --checkpoint "population_phase2=$MODEL_B_CKPT" \
  --base-data-dir "$ONEMAIZE_DATA_DIR" \
  --split test --context-length 16384 --samples-per-class 256 \
  --output-csv "$RUN_ROOT/checkpoint_comparison.csv" \
  --output-markdown "$RUN_ROOT/checkpoint_comparison.md"
```

当前正式输出包括 overall、gene-centered、non-repeat、TE-rich、per-genotype、macro-class 和 macro-genotype metrics。

## Future / Experimental Extension — explicit variant-aware schema-v4

> **EXPERIMENTAL / WAITING FOR REAL DATA. Not required for the current formal Model B training path.** 仓库没有真实 NAM26 VCF/SV/PAV/TE-polymorphism 文件，因此本扩展不能正式训练，但不会阻塞 schema-v3 Model B。

该未来扩展在 schema-v3 之外增加独立 schema-v4：

```text
schema-v3: manifest.json + genomes.parquet + regions.parquet
schema-v4: variant_manifest.json + variant_regions.parquet
```

schema-v4 中 `coordinate_genotype` 必须等于 `genotype`，表示坐标可以直接索引该 genotype FASTA。若 VCF 坐标仍在 B73 reference space，必须先完成经过验证的 liftover/assembly mapping；builder 会拒绝把 B73 坐标直接套到其他 cultivar FASTA。字段定义见 [`docs/ONEMAIZE_VARIANT_SCHEMA_V4.md`](docs/ONEMAIZE_VARIANT_SCHEMA_V4.md)。

### 1 — Prepare the variant input control table

当前已实现的真实文件 parser 只支持标准 `VCF`/`VCF.GZ`。控制表只映射文件，不规定实验室专用 PAV/TE 表字段：

```tsv
genotype	variant_file	source	coordinate_genotype	reference_genotype
B97	/path/to/B97.genotype_coordinates.vcf.gz	caller-and-version	B97	B73
```

TE/PAV 专用文件必须在确认真实列定义和坐标空间后再加 adapter；不要把普通 TE GFF3 或 assembly FASTA 差异写成 TE insertion polymorphism。完整缺失项见 [`docs/audits/onemaize_variant_te/MISSING_VARIANT_INPUTS.md`](docs/audits/onemaize_variant_te/MISSING_VARIANT_INPUTS.md)。

### 2 — Build schema-v4

```bash
export ONEMAIZE_VARIANT_INPUTS="$ONEMAIZE_ROOT/manifests/onemaize_variant_inputs.tsv"
export ONEMAIZE_VARIANT_DATA_DIR="$ONEMAIZE_ROOT/metadata/nam26_variant_te_v4"

python scripts/build_onemaize_variant_metadata.py \
  --base-data-dir "$ONEMAIZE_DATA_DIR" \
  --input-manifest "$ONEMAIZE_VARIANT_INPUTS" \
  --output-dir "$ONEMAIZE_VARIANT_DATA_DIR" \
  --fasta-root "$ONEMAIZE_FASTA_ROOT"
```

输出：`variant_manifest.json` 和 `variant_regions.parquet`。内部坐标只转换一次并统一为 0-based half-open。SNP/small indel 采 event context；超长 SV/PAV 采 left/right breakpoint；TE insertion/deletion 只有在真实 annotation 明确标注后才进入 `te_variant`。

### 3 — Audit and validate

```bash
export RUN_ROOT="$ONEMAIZE_ROOT/runs/model_b_preflight"
bash scripts/run_onemaize_variant_te_phase2_h200.sh audit
bash scripts/run_onemaize_variant_te_phase2_h200.sh validate
```

正式 audit 检查 genotype/split/B73、FASTA/FAI/GZI、gene/TE/variant 文件、variant 和 candidate counts、N fraction、重复 ID、越界、split leakage、TE family、PAV/SV 可用性、缺失 class 和 event coverage。任一 error 时停止。

### 4 — Benchmark and smoke

```bash
bash scripts/run_onemaize_variant_te_phase2_h200.sh benchmark

export PHASE1_CKPT="$PHASE1_RUN_ROOT/train/checkpoints_best/val_loss.ckpt"
unset RESUME_CKPT
export MAX_STEPS=10 WARMUP_STEPS=2
bash scripts/run_onemaize_variant_te_phase2_h200.sh smoke
```

launcher 会检查 `PHASE1_CKPT` 的 stored config 必须是 B73、8,192 bp、`mode=full_genome`。不得传 Model A 的 B73 Phase-II checkpoint。

### 5 — Experimental train or exact resume

sampling pilot default 为 `gene/non-repeat/TE-rich/small-variant/SV-PAV/TE-variant = 0.20/0.15/0.15/0.20/0.20/0.10`。该比例只是可运行默认值，**sampling ratio requires ablation**；所有概率均在 YAML 中配置并强制和为 1。

```bash
export PHASE1_CKPT="$PHASE1_RUN_ROOT/train/checkpoints_best/val_loss.ckpt"
export MAX_STEPS=<benchmark-approved-steps>
export WARMUP_STEPS=<approved-warmup>
export CHECKPOINT_INTERVAL=<approved-interval>
bash scripts/run_onemaize_variant_te_phase2_h200.sh train

# Exact Model-B resume; restores optimizer/scheduler/global step.
unset PHASE1_CKPT
export RESUME_CKPT="$RUN_ROOT/train/checkpoints_resume/last.ckpt"
bash scripts/run_onemaize_variant_te_phase2_h200.sh train
```

在 PBS 8×H200 上可从 [`pbs_scripts/run_onemaize_variant_te_phase2.pbs`](pbs_scripts/run_onemaize_variant_te_phase2.pbs) 开始；launcher 不依赖 PBS/Slurm Python API，在 PBS allocation 内直接由 Lightning 启动 8 个本地进程。

### 6 — Fair checkpoint evaluation

不得比较不同训练日志里的 `val_loss`。在同一个 deterministic 16K set 上比较：

```bash
python scripts/evaluate_onemaize_checkpoints.py \
  --checkpoint "phase1=$PHASE1_CKPT" \
  --checkpoint "reference_phase2=$MODEL_A_CKPT" \
  --checkpoint "variant_te_phase2=$MODEL_B_CKPT" \
  --base-data-dir "$ONEMAIZE_DATA_DIR" \
  --variant-data-dir "$ONEMAIZE_VARIANT_DATA_DIR" \
  --split test --context-length 16384 --samples-per-class 256 \
  --output-csv "$RUN_ROOT/checkpoint_comparison.csv" \
  --output-markdown "$RUN_ROOT/checkpoint_comparison.md"
```

输出 overall、gene-centered、non-repeat、TE-rich、SNP、indel、SV、PAV、TE insertion/deletion、per-genotype、macro-class 和 macro-genotype loss/perplexity/token/sample counts。无数据的类别明确写 `N/A`。

## Pre-flight checklist

正式训练前逐项确认：

- [ ] exact 26-genotype panel present
- [ ] final 23/1/2 split frozen
- [ ] B73 assigned to train
- [ ] every FASTA/FAI/GZI present and matched
- [ ] every gene GFF3 and TE GFF3 present
- [ ] chromosome/seqid names consistent
- [ ] schema-v3 build completed with `--formal`
- [ ] `validation_8k.json` and `validation_16k.json` passed
- [ ] all genotype/class pools present
- [ ] TE union-coverage classification audited
- [ ] Phase-II candidate audit reports `short=0`
- [ ] B73 Phase-I manifest validation passed
- [ ] Phase-I target-node benchmark completed
- [ ] 8×H200 Phase-I smoke passed
- [ ] Phase-I best checkpoint exists
- [ ] single-H200 16K benchmark completed
- [ ] 8×H200 Phase-II smoke passed
- [ ] output/checkpoint filesystem has sufficient free space
- [ ] final `MAX_STEPS`, warmup, batch and checkpoint interval frozen

Formal Model B additionally requires:

- [ ] `ALLCULTIVAR_INPUT_AUDIT.json` reports PASS
- [ ] 23/1/2 split is frozen and B73 is train
- [ ] every genotype has FASTA/FAI/GZI, gene GFF3 and TE GFF3
- [ ] every genotype × gene/non-repeat/TE-rich pool is non-empty
- [ ] candidate length, N fraction and repeat coverage audit reviewed
- [ ] `PHASE1_CKPT` gate confirms B73 8K `full_genome`

Only the experimental schema-v4 extension additionally requires:

- [ ] real per-genotype SNP/indel and SV/PAV inputs available
- [ ] real cultivar-specific TE insertion/deletion annotation available
- [ ] every event coordinate explicitly mapped to its genotype FASTA
- [ ] schema-v4 `variant_manifest.json` and `variant_regions.parquet` built
- [ ] variant audit reports no duplicate ID, out-of-FASTA event or split leakage
- [ ] every enabled genotype/class pool is non-empty
- [ ] sampling probabilities and ablation plan frozen

> **任一必要项未通过，不要启动对应阶段的正式训练。**

通用磁盘检查：

```bash
df -h "$ONEMAIZE_ROOT"
du -sh "$ONEMAIZE_ROOT"/runs "$ONEMAIZE_ROOT"/runs/*/checkpoints* 2>/dev/null || true
```

## Expected outputs

| Stage | Files actually produced by current code |
| --- | --- |
| NAM26 region build | `manifest.json`, `genomes.parquet`, `regions.parquet`, `DATA_STATS.md` |
| NAM26 validation | `validate/validation_8k.json`, `validate/validation_16k.json` |
| Candidate audit | `phase2_16k_candidate_audit.md` |
| Phase-I index | B73 fixed-window parquet, validation JSON, optional slicing MD/CSV |
| Phase-I benchmark | `benchmark.json` |
| Phase-I train | `train/console.log`, `train/checkpoints_best/`, `train/checkpoints_resume/` |
| Phase-II benchmark | `benchmark_16k_single_h200.json` |
| Phase-II train | `16k/run_manifest.txt`, `16k/console.log`, best/resume checkpoint directories |
| Phase-II test | `test16k/run_manifest.txt`, `test16k/test.log` |
| Formal Model B audit | `ALLCULTIVAR_INPUT_AUDIT.md/.json`, `ALLCULTIVAR_CANDIDATE_COUNTS.csv` |
| Formal Model B benchmark/train | `allcultivar_phase2_16k_h200.json`, `run_manifest.txt`, logs/checkpoints |
| Experimental schema-v4 | `variant_manifest.json`, `variant_regions.parquet` |
| Experimental variant audit | `VARIANT_INPUT_AUDIT.md/.json`, `VARIANT_SAMPLER_AUDIT.md`, `VARIANT_SAMPLER_COUNTS.csv` |
| Fair checkpoint evaluation | `checkpoint_comparison.csv`, `checkpoint_comparison.md` |

原始 FASTA、annotation 和 checkpoints 不提交 Git。正式归档至少包括 Git commit、冻结 TSV、schema-v3 metadata、完整 Hydra config、Slurm job ID、benchmark JSON、console log、best/last checkpoint、split 和 validation/test 指标。

## B73 validated results

### Current deterministic Phase-I

真实 B73 v5 chr1–chr10 审计结果：

| Metric | Result |
| --- | ---: |
| genomic bp | 2,131,846,805 |
| full windows | 260,229 |
| tail windows | 10 |
| total keep/PAD sequences | 260,239 |
| padding | 31,083 bp |
| valid-bp coverage | 100% |

单张 A100 80GB、BF16、batch 1 的已记录结果：

- 20-step smoke train loss 约从 5.10 降至 2.03；无 NaN/Inf。
- 2 warmup + 10 measured steps：0.441 s/step。
- 2.27 sequences/s；18,564 tokens/s。
- 线性估算完整 B73 full-manifest epoch 约 31.9 h。

这些是 A100 参考，不是 8×H200 实测。证据见：

- [`docs/audits/onemaize_b73/PHASE1_FULL_GENOME_VALIDATION.md`](docs/audits/onemaize_b73/PHASE1_FULL_GENOME_VALIDATION.md)
- [`docs/audits/onemaize_b73/PHASE1_A100_BENCHMARK.json`](docs/audits/onemaize_b73/PHASE1_A100_BENCHMARK.json)
- [`B73_8192_FULL_GENOME_SLICING_STATS.md`](B73_8192_FULL_GENOME_SLICING_STATS.md)

### Phase-II candidate compatibility

B73 schema-v3 metadata 的 16K candidate audit：

| Class | Count | Shorter than 16K |
| --- | ---: | ---: |
| gene-centered | 39,021 | 0 |
| non-repeat | 143 | 0 |
| TE-rich | 124,986 | 0 |
| total | 164,150 | 0 |

该结果只证明 B73 candidates 可提供 16K crop，不证明 NAM26 已完成。证据见 [`docs/audits/onemaize_b73/PHASE2_16K_CANDIDATE_LENGTH_AUDIT.md`](docs/audits/onemaize_b73/PHASE2_16K_CANDIDATE_LENGTH_AUDIT.md)。

## Historical B73 pilot results

> These results were produced before the deterministic full-genome Phase-I design was adopted. They validate the model/data stack but do not define the current Phase-I epoch semantics.

旧 B73 region-aware 8K pilot 使用 100,000 random windows/epoch，在单张 A100 上每个虚拟 epoch 约 6 小时：

| epoch | global step | train loss | val loss | elapsed |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 25,000 | 1.080 | 1.09709 | 6:00:18 |
| 1 | 50,000 | 0.901 | 1.00976 | 6:01:42 |
| 2 | 75,000 | 0.808 | 0.96155 | 6:02:11 |
| 3 | 100,000 | 0.747 | 0.92955 | 6:02:04 |
| 4 | 125,000 | 0.700 | 0.90695 | 6:00:41 |
| 5 | 150,000 | 0.670 | 0.89139 | 6:00:12 |
| 6 | 175,000 | 0.651 | 0.87862 | 6:01:22 |

这组时间不能用于宣称当前 full-genome Phase-I 为 6 h/epoch，也不能代表 NAM26 population performance。

## NAM26 status and operational blockers

NAM26 schema-v3 builder、formal audit、region-aware loader、新的独立 all-cultivar 16K launcher、PBS entry 和 held-out deterministic evaluator 均已实现并有 synthetic tests；最终 26-genotype 数据尚未正式构建或运行。因此当前状态是 **READY FOR ALL-CULTIVAR DATA PREPARATION**，不是被 VCF 缺失阻塞。

当前限制：

1. **NAM26 Phase-I full-genome training is not implemented.** 当前 production Phase-I validator/config/launcher 是 B73-only；没有 26-genotype fixed-manifest aggregator 或 population Phase-I loader。
2. **Formal held-out genotypes are not frozen in the repository.** 运行者必须在 `onemaize_26.tsv` 构建前确定 1 val + 2 test。
3. **8×H200 throughput has not been measured.** 新 launcher 的 benchmark stage 是单 GPU；真实 8 卡吞吐仍需从 smoke/train log 获取。
4. **Final all-cultivar schema-v3 metadata has not been audited.** 只有 audit PASS 后，状态才能升级为 READY FOR BENCHMARK。
5. **Explicit variant annotations remain unavailable.** 这只阻塞 experimental schema-v4 extension，不阻塞正式 Model B。

正式 Model B 只等待 26-genotype FASTA/FAI/GZI、gene GFF3、TE GFF3 和冻结 split；不等待 VCF。上述限制阻止将仓库描述为“Model B 已完成真实数据验证”或“已完成 8×H200 benchmark/formal training”。

## Repository map

```text
configs/
├── dataset/
│   ├── onemaize_b73_phase1_8k_full_genome.yaml   # B73 fixed-manifest dataset
│   ├── onemaize_b73_phase2_16k_region_aware.yaml # explicit 16K region dataset
│   ├── onemaize_dna_mlm.yaml                     # schema-v3 region-aware defaults
│   ├── onemaize_allcultivar_phase2_16k_region_aware.yaml # formal Model B
│   └── onemaize_allcultivar_phase2_variant_te_16k.yaml # experimental extension
└── experiment/
    ├── onemaize_b73_phase1_8k_full_genome.yaml   # current Phase-I experiment
    ├── onemaize_b73_16k.yaml                     # config used by H200 16K launcher
    ├── onemaize_b73_phase2_16k_region_aware.yaml # explicit Phase-II config, not launcher default
    ├── onemaize_allcultivar_phase2_16k_region_aware.yaml # formal Model B
    └── onemaize_allcultivar_phase2_variant_te_16k.yaml # experimental extension

scripts/
├── build_b73_phase1_8k_manifest.py               # B73 fixed 8K index
├── validate_b73_phase1_8k_manifest.py            # B73 v5 exact validator
├── stat_b73_8192_full_genome_slicing.py           # B73 slicing MD/CSV
├── benchmark_b73_phase1_8k.py                     # Phase-I benchmark
├── run_onemaize_phase1_h200.sh                    # Phase-I stages
├── build_onemaize_regions.py                      # schema-v3 NAM26 builder
├── validate_onemaize_data.py                      # region metadata/read validator
├── audit_onemaize_phase2_candidate_lengths.py     # 16K candidate-length audit
├── benchmark_onemaize.py                          # single-process region benchmark
├── run_onemaize_h200.sh                           # schema-v3 region-aware H200 stages
├── audit_onemaize_allcultivar.py                  # formal Model B schema-v3 audit
├── check_onemaize_checkpoint_contract.py          # branch/resume provenance gate
├── run_onemaize_allcultivar_phase2_h200.sh         # formal Model B launcher
├── build_onemaize_variant_metadata.py             # schema-v4 VCF builder
├── audit_onemaize_variant_te.py                   # experimental preflight gate
├── validate_onemaize_variant_te.py                # deterministic reads/crops
├── evaluate_onemaize_checkpoints.py               # fair checkpoint comparison
└── run_onemaize_variant_te_phase2_h200.sh          # experimental launcher

slurm_scripts/
├── run_onemaize_phase1_h200.slurm
└── run_onemaize_h200.slurm

src/
├── onemaize/regions.py                            # schema-v3, TE union, candidate construction
├── onemaize/population_audit.py                   # formal Model B audit
├── onemaize/checkpoint_contracts.py               # checkpoint branch gates
├── onemaize/variants.py                           # schema-v4 and coordinate conversion
├── onemaize/variant_audit.py                      # experimental audit implementation
├── onemaize/phase1_coverage.py                    # fixed-manifest DDP coverage metrics
├── dataloaders/onemaize_mlm.py                    # Phase-I/Phase-II data module
├── dataloaders/onemaize_variant_mlm.py            # experimental data module
└── dataloaders/datasets/
    ├── onemaize_phase1_dataset.py                 # fixed B73 windows and tail PAD
    ├── onemaize_dataset.py                        # 50/30/20 dynamic region sampler
    └── onemaize_variant_dataset.py                # explicit event-centered sampler

docs/audits/onemaize_b73/                          # real B73 validation evidence
docs/audits/onemaize_variant_te/                   # status and missing-input contract
pbs_scripts/run_onemaize_variant_te_phase2.pbs     # 8×H200 PBS example
pbs_scripts/run_onemaize_allcultivar_phase2.pbs    # formal Model B PBS example
```

## Troubleshooting

### Missing `.fai`

对 BGZF FASTA 执行 `samtools faidx genome.fa.gz`。若 `.fai` 来自另一文件版本，删除错误索引后用当前 FASTA 重建；不要混用不同下载或重新压缩版本。

### Missing `.gzi`

确认文件是 BGZF。对 plain FASTA 用 `bgzip` 重新压缩后执行 `samtools faidx`。普通 gzip 不能替代 BGZF random access。

### Chromosome naming mismatch (`chr1` vs `1`)

默认 `--seqid-regex` 只接收 `chr1`–`chr10`。优先使用 FASTA 与 annotation 匹配的官方版本；如果材料确实采用另一套一致命名，可在 region builder 中显式传入经过审查的 `--seqid-regex`。B73 exact Phase-I validator 仍固定要求 `chr1`–`chr10`。

### Schema validation failure

检查 TSV header、26 个标准名称、23/1/2 split、B73=train、`--formal`、三个 parquet/JSON 文件和 FASTA indexes。不要用 `REQUIRE_FORMAL=0` 绕过正式训练 gate。

### Candidate shorter than 16K

停止 Phase-II。先确认 chromosome lengths、gene coordinates 和 builder 参数。当前 loader 会拒绝 split 中最短 candidate 小于 context length；不要静默降低 context 或改变 50/30/20 权重。

### GFF3 or TE coverage failure

当前 parser 要求 9-column GFF3。gene 只读取 `feature_type=gene`；TE intervals 按 1-based inclusive 输入转换为 0-based half-open 后做 union。先修复 malformed rows 和 seqid mapping，再重建整个 output directory。

### CUDA/Mamba kernel failure

重新确认 Python 3.10、PyTorch 2.2.0、CUDA runtime、driver、`mamba-ssm==1.2.2` 和 `causal-conv1d==1.2.0.post2` ABI。运行环境 import gate 和小 benchmark；不要把当前实现误换成 Mamba2 来绕过错误。

### CUDA OOM

先保持 `BATCH_SIZE=1`，检查其他 GPU 进程和实际 context length。通过 `GRAD_ACCUM` 调整 Phase-II effective batch；Phase-I launcher 当前固定 accumulation=1。不要缩小正式模型层数或 hidden size 来伪造 production smoke 通过。

### NCCL hang

两个 Slurm 模板均请求 8 tasks/8 GPUs；应直接 `sbatch` 模板，不要在外层再加 `srun`。检查 Slurm stdout 中 rank、visible GPU、node 和 NCCL 错误。

### Disk full

训练前运行 `df -h` 和 `du -sh`。降低 `CHECKPOINT_INTERVAL` 的频率值含义是增大步数间隔；至少保留 best checkpoint 和 `last.ckpt`。不要在作业写入期间删除 checkpoint。

### Checkpoint continuation or resume

- Phase-I → Phase-II：使用 `INIT_CKPT`，开始新的 optimizer/scheduler。
- 同一阶段中断恢复：使用 `RESUME_CKPT`，恢复完整状态。
- 评估：清空前两者并使用 `EVAL_CKPT`。

三种变量不要混用。launcher 会拒绝同时设置 `INIT_CKPT` 和 `RESUME_CKPT`。
