#!/usr/bin/env python
"""Evaluate OneMaize checkpoints on identical deterministic 16K sets."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloaders.datasets.onemaize_dataset import (
    OneMaizeRegionMLMDataset,
    collate_onemaize_mlm,
)
from src.dataloaders.datasets.onemaize_variant_dataset import (
    OneMaizeVariantTEMLMDataset,
)


BASE_CLASSES = ("gene_centered", "non_repeat", "te_rich")
def _load_module(checkpoint_path: Path, base_data_dir: Path, context_length: int):
    from omegaconf import DictConfig, OmegaConf
    from train import SequenceLightningModule

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" not in checkpoint or "hyper_parameters" not in checkpoint:
        raise ValueError(f"Expected a complete Lightning checkpoint: {checkpoint_path}")
    stored = checkpoint["hyper_parameters"]
    if isinstance(stored, DictConfig):
        stored = OmegaConf.to_container(stored, resolve=False)
    config = OmegaConf.create(stored)
    updates = {
        "dataset._name_": "onemaize_dna_mlm",
        "dataset.mode": "region_aware",
        "dataset.data_dir": str(base_data_dir.resolve()),
        "dataset.context_length": int(context_length),
        "dataset.train_samples_per_epoch": 1,
        "dataset.val_samples_per_epoch": 1,
        "dataset.test_samples_per_epoch": 1,
        "dataset.batch_size": 1,
        "dataset.batch_size_eval": 1,
        "dataset.shuffle": False,
        "dataset.reverse_complement_probability": 0.0,
        "dataset.gene_probability": 0.5,
        "dataset.non_repeat_probability": 0.3,
        "dataset.te_rich_probability": 0.2,
        "train.pretrained_model_path": None,
        "train.ckpt": None,
    }
    for key, value in updates.items():
        OmegaConf.update(config, key, value, force_add=True)
    OmegaConf.update(
        config, "train.pretrained_model_state_hook._name_", None, force_add=True
    )
    module = SequenceLightningModule(config)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.eval()
    module.requires_grad_(False)
    return module


def _base_dataset(args, tokenizer, sampling_class: str):
    probabilities = {item: 0.0 for item in BASE_CLASSES}
    probabilities[sampling_class] = 1.0
    return OneMaizeRegionMLMDataset(
        args.base_data_dir,
        tokenizer=tokenizer,
        split=args.split,
        context_length=args.context_length,
        samples_per_epoch=args.samples_per_class,
        gene_probability=probabilities["gene_centered"],
        non_repeat_probability=probabilities["non_repeat"],
        te_rich_probability=probabilities["te_rich"],
        reverse_complement_probability=0.0,
        deterministic=True,
        seed=args.seed,
        fasta_root=args.fasta_root,
    )


def _variant_dataset(args, tokenizer, sampling_class: str, variant_types):
    probabilities = {
        "gene_probability": 0.0,
        "non_repeat_probability": 0.0,
        "te_rich_probability": 0.0,
        "small_variant_probability": 0.0,
        "structural_variant_probability": 0.0,
        "te_variant_probability": 0.0,
    }
    probabilities[f"{sampling_class}_probability"] = 1.0
    return OneMaizeVariantTEMLMDataset(
        args.base_data_dir,
        args.variant_data_dir,
        tokenizer=tokenizer,
        split=args.split,
        context_length=args.context_length,
        samples_per_epoch=args.samples_per_class,
        reverse_complement_probability=0.0,
        deterministic=True,
        seed=args.seed,
        fasta_root=args.fasta_root,
        missing_class_policy="error",
        return_metadata=True,
        variant_type_filter=variant_types,
        **probabilities,
    )


def _move_batch(batch, device):
    return tuple(
        {
            key: item.to(device) if isinstance(item, torch.Tensor) else item
            for key, item in value.items()
        }
        if isinstance(value, dict)
        else value.to(device)
        for value in batch
    )


def _evaluate_dataset(module, dataset, device: torch.device, evaluation_set: str):
    totals = defaultdict(lambda: {"nll": 0.0, "masked_tokens": 0, "sample_count": 0})
    tokenizer = module.dataset.tokenizer
    for index in range(len(dataset)):
        item = dataset[index]
        if len(item) == 3:
            input_ids, labels, metadata = item
        else:
            input_ids, labels = item
            metadata = dataset.sample_metadata(index)
        batch = collate_onemaize_mlm(
            [(input_ids, labels)], pad_token_id=int(tokenizer.pad_token_id)
        )
        batch = _move_batch(batch, device)
        with torch.inference_mode():
            logits, targets, _ = module.forward(batch)
            nll = F.cross_entropy(
                logits.float(),
                targets,
                ignore_index=int(tokenizer.pad_token_id),
                reduction="sum",
            )
        masked_tokens = int(targets.ne(int(tokenizer.pad_token_id)).sum().item())
        if masked_tokens <= 0 or not torch.isfinite(nll):
            raise FloatingPointError(
                f"Non-finite evaluation result for {evaluation_set} sample {index}"
            )
        for key in ("overall", f"class:{evaluation_set}", f"genotype:{metadata['genotype']}"):
            totals[key]["nll"] += float(nll.cpu())
            totals[key]["masked_tokens"] += masked_tokens
            totals[key]["sample_count"] += 1
    return totals


def _merge_totals(target, source):
    for key, values in source.items():
        for metric, value in values.items():
            target[key][metric] += value


def _row(checkpoint_label: str, evaluation_set: str, totals: dict):
    tokens = totals["masked_tokens"]
    loss = totals["nll"] / tokens if tokens else None
    return {
        "evaluation_set": evaluation_set,
        "checkpoint": checkpoint_label,
        "loss": loss,
        "perplexity": math.exp(loss) if loss is not None else None,
        "masked_tokens": tokens,
        "sample_count": totals["sample_count"],
    }


def evaluate_checkpoint(args, label: str, checkpoint_path: Path) -> list[dict]:
    module = _load_module(checkpoint_path, args.base_data_dir, args.context_length)
    device = torch.device(args.device)
    module.to(device)
    totals = defaultdict(lambda: {"nll": 0.0, "masked_tokens": 0, "sample_count": 0})
    unavailable = []
    evaluation_specs = [
        ("gene-centered", "gene_centered", None),
        ("non-repeat", "non_repeat", None),
        ("TE-rich", "te_rich", None),
    ]
    if args.variant_data_dir:
        evaluation_specs.extend(
            [
                ("SNP", "small_variant", {"snp"}),
                ("indel", "small_variant", {"indel", "insertion", "deletion"}),
                ("SV", "structural_variant", {"deletion", "insertion", "inversion", "duplication", "sv"}),
                ("PAV", "structural_variant", {"pav"}),
                ("TE insertion", "te_variant", {"te_insertion"}),
                ("TE deletion", "te_variant", {"te_deletion"}),
            ]
        )
    for evaluation_set, sampling_class, variant_types in evaluation_specs:
        try:
            dataset = (
                _base_dataset(args, module.dataset.tokenizer, sampling_class)
                if variant_types is None
                else _variant_dataset(
                    args,
                    module.dataset.tokenizer,
                    sampling_class,
                    variant_types,
                )
            )
        except (ValueError, FileNotFoundError) as exc:
            unavailable.append((evaluation_set, str(exc)))
            continue
        _merge_totals(
            totals,
            _evaluate_dataset(module, dataset, device, evaluation_set),
        )
        dataset.close()

    rows = [_row(label, key, value) for key, value in sorted(totals.items())]
    for evaluation_set, reason in unavailable:
        rows.append(
            {
                "evaluation_set": f"class:{evaluation_set}",
                "checkpoint": label,
                "loss": None,
                "perplexity": None,
                "masked_tokens": 0,
                "sample_count": 0,
                "note": f"N/A: {reason}",
            }
        )
    class_losses = [
        row["loss"]
        for row in rows
        if row["evaluation_set"].startswith("class:") and row["loss"] is not None
    ]
    genotype_losses = [
        row["loss"]
        for row in rows
        if row["evaluation_set"].startswith("genotype:") and row["loss"] is not None
    ]
    for name, values in (
        ("macro:classes", class_losses),
        ("macro:genotypes", genotype_losses),
    ):
        loss = sum(values) / len(values) if values else None
        rows.append(
            {
                "evaluation_set": name,
                "checkpoint": label,
                "loss": loss,
                "perplexity": math.exp(loss) if loss is not None else None,
                "masked_tokens": 0,
                "sample_count": len(values),
            }
        )
    return rows


def _write_outputs(rows: list[dict], csv_path: Path, markdown_path: Path) -> None:
    fields = [
        "evaluation_set",
        "checkpoint",
        "loss",
        "perplexity",
        "masked_tokens",
        "sample_count",
        "note",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    lines = [
        "# OneMaize deterministic checkpoint evaluation",
        "",
        "| Evaluation set | Checkpoint | Loss | Perplexity | Masked tokens | Samples | Note |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        loss = "N/A" if row.get("loss") is None else f"{row['loss']:.6f}"
        perplexity = "N/A" if row.get("perplexity") is None else f"{row['perplexity']:.6f}"
        lines.append(
            f"| {row['evaluation_set']} | {row['checkpoint']} | {loss} | "
            f"{perplexity} | {row['masked_tokens']} | {row['sample_count']} | "
            f"{row.get('note', '')} |"
        )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="LABEL=PATH; repeat as needed")
    parser.add_argument("--base-data-dir", type=Path, required=True)
    parser.add_argument("--variant-data-dir", type=Path)
    parser.add_argument("--fasta-root")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--samples-per-class", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.samples_per_class <= 0:
        raise ValueError("samples-per-class must be positive")
    rows = []
    for item in args.checkpoint:
        if "=" not in item:
            raise ValueError("--checkpoint must use LABEL=PATH")
        label, raw_path = item.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(evaluate_checkpoint(args, label, path))
    _write_outputs(rows, args.output_csv, args.output_markdown)


if __name__ == "__main__":
    main()
