import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.dataloaders.genomics import HG38
from caduceus.configuration_caduceus import CaduceusConfig
from caduceus.modeling_caduceus import CaduceusForMaskedLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from the fixed validation split of mixed_rna and run PCA/t-SNE."
    )
    parser.add_argument("--text_file", type=str, required=True, help="Path to coding RNA TXT file")
    parser.add_argument("--fasta_file", type=str, required=True, help="Path to non-coding RNA FASTA file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint (.ckpt)")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--max_len", type=int, default=1024, help="Sequence max length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding extraction")
    parser.add_argument("--plot_prefix", type=str, default="rna", help="Prefix for output figure names")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Which fixed split to visualize"
    )
    parser.add_argument("--tsne_perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--tsne_random_state", type=int, default=42, help="Random seed for t-SNE")
    return parser.parse_args()


def get_real_dataset(dataset):
    real_dataset = dataset
    while hasattr(real_dataset, "dataset"):
        real_dataset = real_dataset.dataset
    return real_dataset


def build_datamodule(args):
    print("Loading datamodule...")

    datamodule = HG38(
        bed_file=None,
        fasta_file=None,
        dataset_name="mixed_rna",
        text_file=args.text_file,
        rna_fasta_file=args.fasta_file,
        tokenizer_name="char",
        max_length=args.max_len,
        batch_size=16,
        batch_size_eval=64,
        mlm=False,  # embedding extraction must use unmasked inputs
    )
    datamodule.setup()

    if args.split == "val":
        dataset = datamodule.dataset_val
    else:
        dataset = datamodule.dataset_test

    print(f"{args.split} dataset size:", len(dataset))
    return datamodule, dataset


def load_split_sequences(dataset):
    real_dataset = get_real_dataset(dataset)

    if not hasattr(real_dataset, "sources") or not hasattr(real_dataset, "sequences"):
        raise AttributeError("Expected underlying dataset to have 'sources' and 'sequences' attributes.")

    split_indices = dataset.indices
    split_sources = [real_dataset.sources[i] for i in split_indices]
    split_sequences = [real_dataset.sequences[i] for i in split_indices]

    print(f"Loaded {len(split_sequences)} sequences from fixed split")
    return split_sources, split_sequences


def build_model(checkpoint_path, device):
    print("Loading checkpoint...")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    # IMPORTANT:
    # training used vocab_size = 12, so analysis must match it
    config = CaduceusConfig(
        d_model=768,
        n_layer=12,
        vocab_size=12,
    )

    model = CaduceusForMaskedLM(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    model = model.to(device)
    model.eval()

    print("Model loaded")
    return model


def extract_embeddings(model, tokenizer, seqs, batch_size, max_len, device):
    embeddings = []

    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]

        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )

        input_ids = tokens["input_ids"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        hidden = outputs.hidden_states[-1]
        emb = hidden.mean(dim=1)
        embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def main():
    args = parse_args()

    print("========== PCA / t-SNE RNA embedding analysis ==========")
    print("Checkpoint:", args.checkpoint_path)
    print("TXT file   :", args.text_file)
    print("FASTA file :", args.fasta_file)
    print("Split      :", args.split)
    print("Device     :", args.device)
    print("mlm        : False (required for embedding extraction)")
    print("========================================================")

    datamodule, dataset = build_datamodule(args)
    tokenizer = datamodule.tokenizer

    split_sources, split_sequences = load_split_sequences(dataset)
    print("Tokenizer vocab size:", len(tokenizer))

    model = build_model(args.checkpoint_path, args.device)

    print("Extracting embeddings...")
    X = extract_embeddings(
        model=model,
        tokenizer=tokenizer,
        seqs=split_sequences,
        batch_size=args.batch_size,
        max_len=args.max_len,
        device=args.device,
    )
    print("Embedding shape:", X.shape)

    labels = np.array([1 if s == "txt" else 0 for s in split_sources])
    print("Coding    :", int(np.sum(labels == 1)))
    print("Noncoding :", int(np.sum(labels == 0)))

    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)

    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=args.tsne_random_state,
    )
    X_tsne = tsne.fit_transform(X)
    print("t-SNE finished")

    pca_out = f"{args.plot_prefix}_pca_{args.split}set.png"
    tsne_out = f"{args.plot_prefix}_tsne_{args.split}set.png"

    plt.figure(figsize=(7, 6))
    plt.scatter(
        X_pca[labels == 0, 0],
        X_pca[labels == 0, 1],
        c="blue",
        label="noncoding RNA",
        alpha=0.6,
    )
    plt.scatter(
        X_pca[labels == 1, 0],
        X_pca[labels == 1, 1],
        c="red",
        label="coding RNA",
        alpha=0.6,
    )
    plt.legend()
    plt.title(f"RNA Embedding PCA ({args.split} set)")
    plt.savefig(pca_out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved PCA  ->", os.path.abspath(pca_out))

    plt.figure(figsize=(7, 6))
    plt.scatter(
        X_tsne[labels == 0, 0],
        X_tsne[labels == 0, 1],
        c="blue",
        label="noncoding RNA",
        alpha=0.6,
    )
    plt.scatter(
        X_tsne[labels == 1, 0],
        X_tsne[labels == 1, 1],
        c="red",
        label="coding RNA",
        alpha=0.6,
    )
    plt.legend()
    plt.title(f"RNA Embedding t-SNE ({args.split} set)")
    plt.savefig(tsne_out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved t-SNE ->", os.path.abspath(tsne_out))


if __name__ == "__main__":
    main()