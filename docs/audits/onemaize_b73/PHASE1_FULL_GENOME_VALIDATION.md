# B73 Phase-I 8K Full-Genome Validation

本报告对应老师确定的 Phase-I：B73 `chr1`--`chr10`，窗口和 stride 均为
8192 bp，坐标为 0-based half-open，尾部窗口保留并在 dataset 中右侧 PAD。
验证由 `scripts/validate_b73_phase1_8k_manifest.py` 独立完成；FASTA、FAI、GZI
只在服务器数据目录保存，不进入 Git。

## 结果

| 项目 | 值 |
|---|---:|
| 有效基因组 bp | 2,131,846,805 |
| 完整 8192-bp windows | 260,229 |
| 尾部 windows | 10 |
| 尾部有效 bp | 50,837 |
| 保留/PAD 总 sequence | 260,239 |
| 直接丢弃尾部 sequence | 260,229 |
| PAD bp | 31,083 |
| 保留/PAD 覆盖率 | 100% |
| 丢弃尾部覆盖率 | 99.997615354% |

逐条染色体长度、完整窗口、尾长和覆盖率见仓库根目录的
`B73_8192_FULL_GENOME_SLICING_STATS.md/.csv`。验证同时检查了 region_id 连续性、
同染色体窗口无 gap/overlap、窗口不跨染色体、`valid_bp+padded_bp=8192`，以及
FAI 染色体长度一致性。

## 运行

```bash
python scripts/build_b73_phase1_8k_manifest.py \
  --fasta "$ONEMAIZE_B73_FASTA" \
  --output "$ONEMAIZE_PHASE1_MANIFEST"
python scripts/validate_b73_phase1_8k_manifest.py \
  --manifest "$ONEMAIZE_PHASE1_MANIFEST" \
  --fasta "$ONEMAIZE_B73_FASTA"
```

## 训练语义

Phase-I train dataset 的 `len()` 就是 260,239；不再使用 Phase-II 的虚拟
100,000 samples/epoch。`DistributedSampler(drop_last=False)` 在 world=1 时每 epoch
抽取 260,239 条；world=8 时每卡 32,530 条、全局 draws 260,240，唯一窗口
260,239，理论重复 1 条。训练日志增加 unique region、unique fraction、duplicate
fraction、有效 genomic bp coverage 和 tail sequences seen。

## 短 smoke / 时间参考

在当前学校服务器单张 NVIDIA A100 80GB、BF16、121M 参数、batch=1 上完成
20-step Phase-I smoke：loss 约从 `5.10` 降到 `2.03`，没有 NaN/Inf；smoke
明确关闭逐步 checkpoint，避免占满共享磁盘。

随后进行 2 warmup + 10 measured steps：平均 `0.441 s/step`、约 `2.27
sequences/s`（`18,564 tokens/s`），按完整 260,239-window epoch 约 `114,841 s`
（31.9 h）。该值仅作为 A100 参考；8×H200 必须在目标节点先复测。
原始短 benchmark 摘要保存在 `PHASE1_A100_BENCHMARK.json`。
