# Phase-I Full-Genome Implementation Audit

审计基线：仓库 commit `f68afc8`。本文件先记录修改前真实代码，再说明本次
Phase-I deterministic full-genome 方案的最小改动位置。Phase-II 的
region-aware sampler 保持独立、继续保留。

## 1. 现有调用链（修改前）

```text
train.py:train(config)
  -> SequenceLightningModule.__init__/setup
  -> SequenceDataset.registry["onemaize_dna_mlm"]
  -> src/dataloaders/onemaize_mlm.py:OneMaizeDNAMLM.setup
  -> OneMaizeRegionMLMDataset
  -> regions.parquet candidate rows
  -> OneMaizeRegionMLMDataset.__getitem__
  -> _FastaStore.fetch / pysam.FastaFile.fetch
  -> reverse_complement
  -> _mask
  -> collate_onemaize_mlm
  -> LMTask.forward
  -> DNAEmbeddingModelCaduceus/Caduceus
  -> cross_entropy(ignore_index=4)
```

修改前训练数据是虚拟随机 epoch：`__len__()` 返回配置的
`samples_per_epoch`（默认 100,000），每次 `__getitem__(i)` 都会随机选择
genotype、region class、candidate row 和 crop 起点。它适合 Phase-II，不能保证
单个 epoch 覆盖每个 genomic locus。

## 2. Requirement → current implementation → modification → file/function

| Requirement | Current implementation at `f68afc8` | Modification | File / function |
| --- | --- | --- | --- |
| Phase-I 只覆盖 B73 chr1–chr10 | 动态 metadata 由 `regions.parquet` 提供，按 split/genotype 取候选区 | 新增从 B73 `.fai` 生成固定坐标 manifest；只接受 chr1–chr10 | `scripts/build_b73_phase1_8k_manifest.py` |
| 8192 bp、stride 8192、连续非重叠 | `candidate_span=32768`，运行时在 candidate 内 random crop | manifest 每条记录一个 `[start,end)` 固定窗口；`stride == window_size` 强制校验 | `scripts/build_b73_phase1_8k_manifest.py` |
| 每条染色体尾部必须保留并 pad | 现有 `_FastaStore.fetch` 只返回真实区间，动态 candidate 要求长度足够 | full-genome dataset fetch `start:end`，再在右侧补 `[PAD]`；不跨 chromosome | `src/dataloaders/datasets/onemaize_phase1_dataset.py:OneMaizePhase1FullGenomeMLMDataset.__getitem__` |
| `len(dataset) == 260239` | 动态 dataset 长度等于 `samples_per_epoch` | Phase-I train dataset 长度严格等于 manifest 行数；val/test 可选固定前缀子集 | `OneMaizePhase1FullGenomeMLMDataset.__len__` |
| index 稳定指向 locus | `_rng_for_index` 只在 deterministic split 使用；train 会随机 region/crop | Phase-I index 直接映射 manifest 第 i 行；MLM/RC RNG 与坐标分离 | `OneMaizePhase1FullGenomeMLMDataset.__getitem__` |
| 允许的 augmentation | 现有 tokenizer 为 `CaduceusTokenizer`，MLM 15%，train RC 默认 0.5 | Phase-I 沿用 single-nucleotide tokenizer、MLM 和显式 `phase1_rc_probability`；报告 RC 配置 | `src/dataloaders/onemaize_phase1_dataset.py`, `src/dataloaders/onemaize_mlm.py` |
| PAD 不参与 attention/loss | 动态样本无 PAD；collate 根据 `input_ids != pad_id` 生成 attention mask；labels 用 pad id=4 ignore | full-genome collate 同时返回 `attention_mask`、`valid_mask` 和 `valid_bp`；mask eligibility 只取真实 A/C/G/T；PAD label 永远为 4 | `collate_onemaize_phase1_mlm`, `OneMaizePhase1FullGenomeMLMDataset._mask` |
| current FASTA fetch | `_FastaStore.fetch` 选择 pysam、Biopython BGZF 或 pyfaidx | 复用 `_FastaStore.fetch`，不导出超大 sequence 文本 | `src/dataloaders/datasets/onemaize_dataset.py:_FastaStore.fetch` |
| current tokenizer | `OneMaizeDNAMLM.setup` 构造 `CaduceusTokenizer(sequence_type="dna")` | Phase-I 使用同一 tokenizer 实例；不改 vocabulary | `src/dataloaders/onemaize_mlm.py:setup` |
| current MLM masking | `OneMaizeRegionMLMDataset._mask`：A/C/G/T eligible，15%，80/10/10，labels 非 target 为 pad id | Phase-I 独立实现完全相同的规则，并额外排除 PAD；Phase-II 原函数不动 | `src/dataloaders/datasets/onemaize_dataset.py`, `onemaize_phase1_dataset.py` |
| current RC | `reverse_complement()`；动态 train `__getitem__` 按概率执行 | Phase-I 只对 sequence orientation 做 augmentation；manifest region_id 不变 | `src/dataloaders/datasets/onemaize_dataset.py:reverse_complement` |
| current region parquet loading | `OneMaizeRegionMLMDataset.__init__` 读取 regions parquet，并构建 genotype/class pools | Phase-I 不读取 candidate-region sampler；Phase-II 路径不改 | `src/dataloaders/datasets/onemaize_dataset.py:OneMaizeRegionMLMDataset.__init__` |
| 50/30/20 sampler | `_sample_pool()` 先均匀 genotype，再按 gene/non-repeat/TE-rich 概率选择；`_sample_region()` 随机 row | 保留原实现；仅在 Phase-II `mode=region_aware` 使用 | `src/dataloaders/datasets/onemaize_dataset.py:_sample_pool`, `_sample_region` |
| 16K Phase-II | `configs/experiment/onemaize_b73_16k.yaml` 已设 context 16384；candidate span 32768 | 新增显式 `mode=region_aware` dataset config，并生成短 candidate audit；不重定义区域 | `configs/dataset/onemaize_b73_phase2_16k_region_aware.yaml`, `scripts/audit_onemaize_phase2_candidate_lengths.py` |
| Phase-I config | 8K experiment 当前默认动态 dataset，`train_samples_per_epoch=100000` | 新增 full-genome dataset/experiment 配置，manifest、stride、tail policy 都显式写出 | `configs/dataset/onemaize_b73_phase1_8k_full_genome.yaml`, `configs/experiment/onemaize_b73_phase1_8k_full_genome.yaml` |
| Lightning / DDP | `SequenceLightningModule.train_dataloader()` 调用 dataset dataloader；`train.py:create_trainer()` 在 devices>1 自动设 DDPStrategy | Phase-I 显式启用 `replace_sampler_ddp=true`；coverage tracker 跨 rank 汇总 sampler 实际 IDs | `train.py:SequenceLightningModule`, `src/onemaize/phase1_coverage.py` |
| 每 epoch coverage | 当前没有 full-genome fixed-region tracker | 记录 samples seen、unique IDs、duplicate fraction、valid bp coverage、tail count | `src/onemaize/phase1_coverage.py:Phase1CoverageTracker` |
| checkpoint / model | 121M Caduceus/BCW/memory 架构与 checkpoint callback 已存在 | 不改模型；Phase-I 只更换 dataset/config/callback metrics | `train.py`, existing model/config files |

## 3. Dataset semantics

Phase-I manifest 每一行是一个固定的 chromosome-local interval。普通行满足
`valid_bp=8192, padded_bp=0`；尾行满足 `valid_bp < 8192`、
`padded_bp=8192-valid_bp`、`is_tail=true`。`end` 是真实 FASTA end，不是 pad
后的虚拟坐标。dataset 只 fetch `[start,end)`，然后在右端补 `pad_token_id`，因此
不会读取到下一条染色体。

Phase-I train dataset 的 `__len__` 就是 manifest 行数 `260239`。`dataset[i]`
的 `region_id/chromosome/start/end` 永远不变；随机性只影响 MLM mask 和（若配置
启用）reverse-complement orientation。Phase-II 继续使用原来的随机
`__getitem__` 和 `samples_per_epoch=100000` 虚拟 epoch。

## 4. DDP semantics

Lightning 的 `replace_sampler_ddp=true` 为 map-style Phase-I dataset 注入
`DistributedSampler(drop_last=False)`。设 `N=260239`、`W=world_size`：

```text
per-rank samples = ceil(N / W)
global sampler draws = W * ceil(N / W)
unique regions = N
sampler padding duplicates = global sampler draws - N
```

所以 world size 1 为 260239 samples、0 duplicate；world size 8 为每 rank
32530 samples、global draws 260240、1 个 sampler padding duplicate、unique
regions 仍为 260239。coverage tracker 以 region_id 去重，不能把这个 duplicate
报告成额外 genomic coverage。

optimizer steps 取决于 per-rank batch 和 gradient accumulation：

```text
batches/rank = ceil(per_rank samples / batch_size_per_gpu)
optimizer steps/epoch = ceil(batches/rank / grad_accum)
```

`drop_last=false` 保证尾 batch 不被静默丢弃；正式日志同时记录 rank-local
samples、global draws、unique regions 和 duplicate count。

## 5. Padding and loss semantics

`attention_mask`/`valid_mask` 在真实窗口位置为 true，在右侧 PAD 位置为 false。
MLM eligible positions 只来自真实的 A/C/G/T token；PAD、N 和 mask special token
都不会成为 target。labels 的非 target 值继续使用正式 tokenizer `pad_token_id=4`，
与 `configs/.../task.loss.ignore_index=4` 和 perplexity metric 的 ignore index
一致。因此 PAD 不进入 forward attention，也不进入 MLM cross-entropy 或 token
count。

## 6. Phase-II boundary

Phase-II 继续读取 `genomes.parquet`/`regions.parquet`，保留 gene-centered、
non-repeat、TE-rich、TE union、repeat threshold=0.5、50/30/20、dynamic crop
和已有 coverage tracking。16K candidate audit 单独统计每种 region class 的
`length >= 16384` 与 `<16384`；若真实 26 材料出现短 candidate，报告 blocker，
不擅自改变 gene 定义或剪裁规则。

## 7. Expected implementation call chains

### Phase I

```text
train.py:train
 -> configs/experiment/onemaize_b73_phase1_8k_full_genome.yaml
 -> OneMaizeDNAMLM(mode=full_genome)
 -> b73_phase1_8k_full_genome.parquet
 -> fixed chromosome/start/end by index
 -> _FastaStore.fetch
 -> right PAD for tail
 -> RC (configured probability)
 -> CaduceusTokenizer
 -> 15% MLM (PAD ignored)
 -> Caduceus/BCW/memory model
```

### Phase II

```text
train.py:train
 -> configs/experiment/onemaize_b73_phase2_16k_region_aware.yaml
 -> OneMaizeDNAMLM(mode=region_aware)
 -> OneMaizeRegionMLMDataset
 -> 50/30/20 region pool
 -> random candidate and 16K crop
 -> _FastaStore.fetch
 -> RC
 -> CaduceusTokenizer
 -> 15% MLM
 -> Caduceus/BCW/memory model
```
