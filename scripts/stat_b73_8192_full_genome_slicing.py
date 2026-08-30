#!/usr/bin/env python
"""Report continuous non-overlapping B73 chromosome slicing statistics."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CHROMOSOMES = tuple(f"chr{i}" for i in range(1, 11))
DEFAULT_WINDOW_SIZE = 8192
DEFAULT_MD = Path("B73_8192_FULL_GENOME_SLICING_STATS.md")
DEFAULT_CSV = Path("B73_8192_FULL_GENOME_SLICING_STATS.csv")


@dataclass(frozen=True)
class SlicingStats:
    chromosome: str
    length_bp: int
    window_size: int

    @property
    def full_windows(self) -> int:
        return self.length_bp // self.window_size

    @property
    def tail_bp(self) -> int:
        return self.length_bp % self.window_size

    @property
    def sequences_drop_tail(self) -> int:
        return self.full_windows

    @property
    def sequences_keep_pad(self) -> int:
        return self.full_windows + int(self.tail_bp > 0)

    @property
    def covered_bp_drop_tail(self) -> int:
        return self.full_windows * self.window_size

    @property
    def covered_bp_keep_pad(self) -> int:
        return self.length_bp

    @property
    def discarded_bp(self) -> int:
        return self.tail_bp

    @property
    def padding_bp(self) -> int:
        return 0 if self.tail_bp == 0 else self.window_size - self.tail_bp

    @property
    def coverage_drop_tail_pct(self) -> float:
        return 100.0 * self.covered_bp_drop_tail / self.length_bp

    @property
    def coverage_keep_pad_pct(self) -> float:
        return 100.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--fasta",
        type=Path,
        help="B73 FASTA path; the script reads the matching <FASTA>.fai index.",
    )
    source.add_argument(
        "--fai",
        type=Path,
        help="B73 FASTA .fai path (useful when only the index is mounted).",
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument(
        "--chromosomes",
        nargs="+",
        default=list(DEFAULT_CHROMOSOMES),
        help="Ordered reference names to include (default: chr1 through chr10).",
    )
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    return parser


def resolve_fai(args: argparse.Namespace) -> tuple[Path, str]:
    if args.fasta is not None:
        fasta = args.fasta.expanduser().resolve()
        if not fasta.is_file():
            raise FileNotFoundError(f"FASTA does not exist: {fasta}")
        fai = Path(f"{fasta}.fai")
        source_name = fasta.name
    else:
        fai = args.fai.expanduser().resolve()
        source_name = fai.name[:-4] if fai.name.endswith(".fai") else fai.name
    if not fai.is_file():
        raise FileNotFoundError(f"FASTA index does not exist: {fai}")
    return fai, source_name


def read_fai_lengths(path: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    with path.open("rt", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                raise ValueError(f"Malformed FAI row {line_number}: {line!r}")
            name = fields[0]
            if name in lengths:
                raise ValueError(f"Duplicate reference name in FAI: {name}")
            try:
                length = int(fields[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid length in FAI row {line_number}: {fields[1]!r}"
                ) from exc
            if length <= 0:
                raise ValueError(f"Non-positive length for {name}: {length}")
            lengths[name] = length
    if not lengths:
        raise ValueError(f"FAI is empty: {path}")
    return lengths


def calculate_stats(
    lengths: dict[str, int], chromosomes: list[str], window_size: int
) -> list[SlicingStats]:
    if window_size <= 0:
        raise ValueError(f"window-size must be positive, got {window_size}")
    if not chromosomes:
        raise ValueError("At least one chromosome is required")
    if len(set(chromosomes)) != len(chromosomes):
        raise ValueError("--chromosomes contains duplicate names")
    missing = [name for name in chromosomes if name not in lengths]
    if missing:
        raise ValueError("Chromosomes missing from FAI: " + ", ".join(missing))
    return [
        SlicingStats(name, lengths[name], window_size) for name in chromosomes
    ]


def total_stats(rows: list[SlicingStats]) -> dict[str, int | float]:
    total_length = sum(row.length_bp for row in rows)
    full_windows = sum(row.full_windows for row in rows)
    keep_pad_sequences = sum(row.sequences_keep_pad for row in rows)
    covered_drop = sum(row.covered_bp_drop_tail for row in rows)
    covered_keep = sum(row.covered_bp_keep_pad for row in rows)
    discarded = sum(row.discarded_bp for row in rows)
    padding = sum(row.padding_bp for row in rows)
    return {
        "length_bp": total_length,
        "full_windows": full_windows,
        "tail_bp": discarded,
        "sequences_drop_tail": full_windows,
        "sequences_keep_pad": keep_pad_sequences,
        "covered_bp_drop_tail": covered_drop,
        "covered_bp_keep_pad": covered_keep,
        "discarded_bp": discarded,
        "padding_bp": padding,
        "coverage_drop_tail_pct": 100.0 * covered_drop / total_length,
        "coverage_keep_pad_pct": 100.0 * covered_keep / total_length,
        "allocated_bp_keep_pad": keep_pad_sequences * rows[0].window_size,
        "tail_sequence_count": sum(row.tail_bp > 0 for row in rows),
    }


CSV_FIELDS = (
    "row_type",
    "chromosome",
    "length_bp",
    "window_size_bp",
    "full_window_count",
    "tail_remainder_bp",
    "sequence_count_drop_tail",
    "sequence_count_keep_pad",
    "covered_bp_drop_tail",
    "covered_bp_keep_pad",
    "discarded_bp_drop_tail",
    "padding_bp_keep_pad",
    "coverage_drop_tail_pct",
    "coverage_keep_pad_pct",
)


def write_csv(
    path: Path, rows: list[SlicingStats], totals: dict[str, int | float]
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "row_type": "chromosome",
                    "chromosome": row.chromosome,
                    "length_bp": row.length_bp,
                    "window_size_bp": row.window_size,
                    "full_window_count": row.full_windows,
                    "tail_remainder_bp": row.tail_bp,
                    "sequence_count_drop_tail": row.sequences_drop_tail,
                    "sequence_count_keep_pad": row.sequences_keep_pad,
                    "covered_bp_drop_tail": row.covered_bp_drop_tail,
                    "covered_bp_keep_pad": row.covered_bp_keep_pad,
                    "discarded_bp_drop_tail": row.discarded_bp,
                    "padding_bp_keep_pad": row.padding_bp,
                    "coverage_drop_tail_pct": f"{row.coverage_drop_tail_pct:.9f}",
                    "coverage_keep_pad_pct": f"{row.coverage_keep_pad_pct:.9f}",
                }
            )
        writer.writerow(
            {
                "row_type": "total",
                "chromosome": "TOTAL_CHR1_CHR10",
                "length_bp": totals["length_bp"],
                "window_size_bp": rows[0].window_size,
                "full_window_count": totals["full_windows"],
                "tail_remainder_bp": totals["tail_bp"],
                "sequence_count_drop_tail": totals["sequences_drop_tail"],
                "sequence_count_keep_pad": totals["sequences_keep_pad"],
                "covered_bp_drop_tail": totals["covered_bp_drop_tail"],
                "covered_bp_keep_pad": totals["covered_bp_keep_pad"],
                "discarded_bp_drop_tail": totals["discarded_bp"],
                "padding_bp_keep_pad": totals["padding_bp"],
                "coverage_drop_tail_pct": (
                    f"{totals['coverage_drop_tail_pct']:.9f}"
                ),
                "coverage_keep_pad_pct": (
                    f"{totals['coverage_keep_pad_pct']:.9f}"
                ),
            }
        )


def write_markdown(
    path: Path,
    rows: list[SlicingStats],
    totals: dict[str, int | float],
    source_name: str,
    fai_name: str,
    fai_sequence_count: int,
) -> None:
    window_size = rows[0].window_size
    lines = [
        "# B73 8192-bp 连续全基因组切片统计",
        "",
        "## 统计范围与口径",
        "",
        f"- 输入 FASTA：`{source_name}`",
        f"- 输入索引：`{fai_name}`",
        f"- window size / stride：`{window_size} bp` / `{window_size} bp`",
        "- 切片方式：每条染色体从 bp 0 开始，连续、非重叠切片",
        "- 纳入序列：`chr1` 至 `chr10`",
        f"- FAI 共包含 {fai_sequence_count:,} 条 reference；本统计排除 scaffold",
        "- “有效基因组 bp”定义为 chr1-chr10 的染色体长度之和",
        "",
        "本报告只做计数，不实际生成 sequence 文件，也不修改现有训练代码或 "
        "Phase-II region-aware sampler。",
        "",
        "## 逐染色体统计",
        "",
        "| 染色体 | 长度 (bp) | 完整 8192-bp windows | 尾部剩余 (bp) | 丢弃尾部 sequence 数 | 保留/pad sequence 数 | 丢弃尾部覆盖率 | pad bp |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.chromosome} | {row.length_bp:,} | {row.full_windows:,} | "
            f"{row.tail_bp:,} | {row.sequences_drop_tail:,} | "
            f"{row.sequences_keep_pad:,} | {row.coverage_drop_tail_pct:.6f}% | "
            f"{row.padding_bp:,} |"
        )
    lines.extend(
        [
            f"| **合计 chr1-chr10** | **{totals['length_bp']:,}** | "
            f"**{totals['full_windows']:,}** | **{totals['tail_bp']:,}** | "
            f"**{totals['sequences_drop_tail']:,}** | "
            f"**{totals['sequences_keep_pad']:,}** | "
            f"**{totals['coverage_drop_tail_pct']:.6f}%** | "
            f"**{totals['padding_bp']:,}** |",
            "",
            "## 全基因组汇总",
            "",
            f"- B73 有效染色体基因组：**{totals['length_bp']:,} bp**",
            f"- 完整 {window_size}-bp windows：**{totals['full_windows']:,}**",
            f"- 10 条染色体尾部剩余长度之和：**{totals['tail_bp']:,} bp**",
            f"- 尾部长度非 0 的染色体数：**{totals['tail_sequence_count']}**",
            "",
            "### 方案 A：直接丢弃每条染色体尾部",
            "",
            f"- sequence 数量：**{totals['sequences_drop_tail']:,}**",
            f"- 覆盖的有效基因组 bp：**{totals['covered_bp_drop_tail']:,} bp**",
            f"- 丢弃 bp：**{totals['discarded_bp']:,} bp**",
            f"- 有效基因组覆盖比例：**{totals['coverage_drop_tail_pct']:.9f}%**",
            "",
            "### 方案 B：每条染色体保留一个尾部 sequence 并 pad",
            "",
            f"- sequence 数量：**{totals['sequences_keep_pad']:,}**",
            f"- 覆盖的有效基因组 bp：**{totals['covered_bp_keep_pad']:,} bp**",
            f"- 所有尾部 sequence 新增的 padding：**{totals['padding_bp']:,} bp**",
            f"- 分配的 sequence slot 总长度：**{totals['allocated_bp_keep_pad']:,} bp**",
            f"- 有效基因组覆盖比例：**{totals['coverage_keep_pad_pct']:.9f}%**",
            "",
            "覆盖率以 chr1-chr10 的真实染色体 bp 为分母；padding 不计入基因组覆盖。",
            "",
            "## 复现命令",
            "",
            "```bash",
            "python scripts/stat_b73_8192_full_genome_slicing.py \\",
            f"  --fasta /path/to/{source_name} \\",
            "  --output-md B73_8192_FULL_GENOME_SLICING_STATS.md \\",
            "  --output-csv B73_8192_FULL_GENOME_SLICING_STATS.csv",
            "```",
            "",
        ]
    )
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    fai, source_name = resolve_fai(args)
    lengths = read_fai_lengths(fai)
    rows = calculate_stats(lengths, args.chromosomes, args.window_size)
    totals = total_stats(rows)
    write_csv(args.output_csv, rows, totals)
    write_markdown(
        args.output_md,
        rows,
        totals,
        source_name,
        fai.name,
        len(lengths),
    )
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")
    print(
        "total_length_bp={length_bp} full_windows={full_windows} "
        "drop_tail_sequences={sequences_drop_tail} "
        "keep_pad_sequences={sequences_keep_pad} "
        "drop_tail_coverage_pct={coverage_drop_tail_pct:.9f}".format(**totals)
    )


if __name__ == "__main__":
    main()
