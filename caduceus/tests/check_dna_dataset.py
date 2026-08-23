import sys
from pathlib import Path

VALID = set("ATCG")

def read_sequences(path: str):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in [".fa", ".fasta", ".fna"]:
        seqs = []
        cur = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur:
                        seqs.append("".join(cur))
                        cur = []
                else:
                    cur.append(line)
            if cur:
                seqs.append("".join(cur))
        return seqs

    else:
        # 默认按 txt：一行一条序列
        seqs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seqs.append(line)
        return seqs


def main(path: str):
    seqs = read_sequences(path)

    total = len(seqs)
    empty_count = 0
    bad_count = 0
    good_count = 0
    bad_examples = []

    for i, seq in enumerate(seqs):
        seq = seq.strip().upper()

        if len(seq) == 0:
            empty_count += 1
            bad_count += 1
            if len(bad_examples) < 10:
                bad_examples.append((i, seq, {"EMPTY"}))
            continue

        bad_chars = set(seq) - VALID
        if bad_chars:
            bad_count += 1
            if len(bad_examples) < 10:
                bad_examples.append((i, seq[:100], bad_chars))
        else:
            good_count += 1

    print(f"文件: {path}")
    print(f"总序列数: {total}")
    print(f"合法序列数(仅ATCG): {good_count}")
    print(f"非法序列数: {bad_count}")
    print(f"空序列数: {empty_count}")

    if bad_count == 0:
        print("结论: 这个数据集全是正常DNA序列（只含ATCG）")
    else:
        print("结论: 这个数据集里还有非法序列")
        print("\n前10个非法样本示例：")
        for idx, seq, bad_chars in bad_examples:
            print(f"- 第 {idx+1} 条: 非法字符={sorted(list(bad_chars))} | 序列前100bp={seq}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python check_dna_dataset.py <你的数据文件>")
        sys.exit(1)
    main(sys.argv[1])