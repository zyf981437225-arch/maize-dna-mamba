# Phase-II 16K Candidate Length Audit

本审计只读检查 B73 现有 `regions.parquet`，不修改现有 region-aware sampler。
候选区间长度必须至少为 16,384 bp，运行脚本为
`scripts/audit_onemaize_phase2_candidate_lengths.py`。

| class | total | >=16,384 | short | min bp | max bp |
|---|---:|---:|---:|---:|---:|
| gene_centered | 39,021 | 39,021 | 0 | 16,384 | 761,401 |
| non_repeat | 143 | 143 | 0 | 32,768 | 32,768 |
| te_rich | 124,986 | 124,986 | 0 | 32,768 | 32,768 |
| all | 164,150 | 164,150 | 0 | - | - |

当前 B73 没有短候选区间，因此 16K dynamic crop 可直接运行。其他材料在建立
索引后必须分别执行同一审计；若 `short > 0`，应显式报告并决定处理方式，不能
静默改变 Phase-II 的 50/30/20 采样语义。
