import gzip
import importlib.util
import json
from pathlib import Path

import torch

DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "dataloaders"
    / "datasets"
    / "full_transcript_mlm_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("full_transcript_mlm_dataset", DATASET_PATH)
DATASET_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DATASET_MODULE)
FullTranscriptMLMDataset = DATASET_MODULE.FullTranscriptMLMDataset
_stable_source_split = DATASET_MODULE._stable_source_split
collate_full_transcript_mlm = DATASET_MODULE.collate_full_transcript_mlm


class DummyTokenizer:
    ids = {"A": 7, "C": 8, "G": 9, "U": 10, "N": 11}
    pad_token_id = 4
    mask_token_id = 3
    unk_token_id = 6

    def get_vocab(self):
        return self.ids


def _write_jsonl(path: Path, records) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _record(sequence, mrna_reliable=True, cds_reliable=True):
    length = len(sequence)
    cds_start = min(1, length)
    cds_end = max(cds_start, length - 1)
    return {
        "sequence": sequence,
        "mrna_coordinate_system_reliable": mrna_reliable,
        "cds_boundary_reliable": cds_reliable,
        "utr5_start": 0,
        "utr5_end": cds_start,
        "cds_start": cds_start,
        "cds_end": cds_end,
        "utr3_start": cds_end,
        "utr3_end": length,
    }


def test_complete_mrna_mlm_is_untruncated_deterministic_and_padding_aware(tmp_path):
    path = tmp_path / "train.jsonl.gz"
    _write_jsonl(
        path,
        [
            _record("ATCGA"),
            _record("AAA"),
            _record("NNA"),
            _record("A" * 9),
            _record("AAAA", mrna_reliable=False),
        ],
    )
    dataset = FullTranscriptMLMDataset(
        path,
        tokenizer=DummyTokenizer(),
        split="train",
        max_sequence_length=8,
        include_ncrna=False,
        mlm_probability=1.0,
        deterministic_mlm=True,
        seed=11,
    )

    assert dataset.sequences == ["AUCGA", "AAA", "NNA"]
    assert dataset.excluded_overlength == 1
    assert dataset.excluded_unreliable_mrna == 1

    first_a = dataset[0]
    first_b = dataset[0]
    assert torch.equal(first_a[0], first_b[0])
    assert torch.equal(first_a[1], first_b[1])
    assert first_a[1].tolist() == [7, 10, 8, 9, 7]
    assert dataset[2][1].tolist() == [4, 4, 7]

    inputs, labels, metadata = collate_full_transcript_mlm(
        [dataset[0], dataset[1]], pad_token_id=4, pad_to_multiple=8
    )
    assert inputs.shape == labels.shape == (2, 8)
    assert metadata["attention_mask"].sum(dim=1).tolist() == [5, 3]
    assert torch.all(labels[0, 5:] == 4)
    assert torch.all(labels[1, 3:] == 4)


def test_ncrna_split_is_stable_and_disjoint():
    identifiers = [f"RNA_{index}" for index in range(1000)]
    assignment_a = {identifier: _stable_source_split(identifier, 2357) for identifier in identifiers}
    assignment_b = {identifier: _stable_source_split(identifier, 2357) for identifier in identifiers}
    assert assignment_a == assignment_b
    assert set(assignment_a.values()) == {"train", "val", "test"}
