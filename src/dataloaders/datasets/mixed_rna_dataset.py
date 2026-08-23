import torch
from torch.utils.data import Dataset


class MixedRNADataset(Dataset):
    """
    Mixed RNA dataset:
    - TXT format: each line like "SEQUENCE,..."
    - FASTA format: standard FASTA

    Unified preprocessing:
    - upper()
    - T -> U
    - non-A/U/C/G -> N

    Returns:
        (input_ids, labels)

    Notes:
    - Can load txt only, fasta only, or both.
    - Suitable for MLM pretraining.
    - self.sources stores the original source for each sequence:
        "txt"   -> coding RNA source
        "fasta" -> non-coding RNA source
    - When mlm=False, labels are returned as a copy of input_ids only for
      interface compatibility; they are NOT biological class labels.
    """

    def __init__(
        self,
        tokenizer,
        text_file: str | None = None,
        fasta_file: str | None = None,
        max_length: int = 1024,
        add_eos: bool = True,
        mlm: bool = False,
        mlm_probability: float = 0.15,
        ignore_id: int | None = None,
        kmer: int = 1,
        frame: int | str = 0,
        max_text_sequences: int | None = None,
        max_fasta_sequences: int | None = None,
        deterministic_mlm: bool = False,
        mlm_seed: int = 0,
    ) -> None:
        super().__init__()

        if text_file is None and fasta_file is None:
            raise ValueError("At least one of text_file or fasta_file must be provided.")

        self.tokenizer = tokenizer
        self.text_file = text_file
        self.fasta_file = fasta_file
        self.max_length = max_length
        self.add_eos = add_eos
        self.mlm = mlm
        self.mlm_probability = float(mlm_probability)
        self.ignore_id = ignore_id
        self.kmer = kmer
        self.frame = frame
        self.max_text_sequences = max_text_sequences
        self.max_fasta_sequences = max_fasta_sequences
        self.deterministic_mlm = deterministic_mlm
        self.mlm_seed = int(mlm_seed)

        tok_pad = getattr(self.tokenizer, "pad_token_id", None)
        self.pad_id = tok_pad if tok_pad is not None else 4
        self.ignore_id = self.pad_id if ignore_id is None else int(ignore_id)
        self.mask_id = getattr(self.tokenizer, "mask_token_id", None)
        self.eos_id = getattr(self.tokenizer, "eos_token_id", None)
        vocab = self.tokenizer.get_vocab()
        self.random_token_ids = torch.tensor(
            [vocab[token] for token in ("A", "C", "G", "U", "N") if token in vocab],
            dtype=torch.long,
        )

        if self.mlm and self.mask_id is None:
            raise ValueError("Tokenizer has no mask_token_id but mlm=True")
        if self.mlm and self.random_token_ids.numel() == 0:
            raise ValueError("Tokenizer has no nucleotide tokens for random MLM replacement")

        self.sequences: list[str] = []
        self.sources: list[str] = []

        if self.text_file is not None:
            txt_seqs = self._load_txt_sequences(
                self.text_file,
                max_sequences=self.max_text_sequences
            )
            self.sequences.extend(txt_seqs)
            self.sources.extend(["txt"] * len(txt_seqs))
            print(f"[MixedRNADataset] Loaded {len(txt_seqs)} TXT sequences from {self.text_file}")

        if self.fasta_file is not None:
            fasta_seqs = self._load_fasta_sequences(
                self.fasta_file,
                max_sequences=self.max_fasta_sequences
            )
            self.sequences.extend(fasta_seqs)
            self.sources.extend(["fasta"] * len(fasta_seqs))
            print(f"[MixedRNADataset] Loaded {len(fasta_seqs)} FASTA sequences from {self.fasta_file}")

        print(f"[MixedRNADataset] Total loaded sequences = {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def _normalize_seq(self, seq: str) -> str:
        """
        Normalize sequence:
        - uppercase
        - T -> U
        - ONLY keep AUCG sequences
        - if any illegal char exists → drop entire sequence
        """
        seq = seq.strip().upper().replace("T", "U")
        allowed = {"A", "U", "C", "G"}

        # 如果有非法字符 → 整条丢弃
        if any(ch not in allowed for ch in seq):
            return ""

        return seq

    def _load_txt_sequences(self, text_file: str, max_sequences: int | None = None) -> list[str]:
        sequences = []

        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                seq = line.split(",")[0].strip()
                seq = self._normalize_seq(seq)

                if len(seq) == 0:
                    continue

                sequences.append(seq)

                if max_sequences is not None and len(sequences) >= max_sequences:
                    break

        return sequences

    def _load_fasta_sequences(self, fasta_file: str, max_sequences: int | None = None) -> list[str]:
        sequences = []
        current_seq = []

        with open(fasta_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith(">"):
                    if len(current_seq) > 0:
                        seq = "".join(current_seq)
                        seq = self._normalize_seq(seq)
                        if len(seq) > 0:
                            sequences.append(seq)

                        if max_sequences is not None and len(sequences) >= max_sequences:
                            break

                    current_seq = []
                else:
                    current_seq.append(line)

            if (max_sequences is None or len(sequences) < max_sequences) and len(current_seq) > 0:
                seq = "".join(current_seq)
                seq = self._normalize_seq(seq)
                if len(seq) > 0:
                    sequences.append(seq)

        return sequences

    def _apply_mlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        generator: torch.Generator | None = None,
    ):
        labels = torch.full_like(input_ids, fill_value=self.ignore_id)

        can_mask = input_ids != self.pad_id
        if self.eos_id is not None:
            can_mask = can_mask & (input_ids != self.eos_id)
        if attention_mask is not None:
            can_mask = can_mask & attention_mask.to(dtype=torch.bool)

        if can_mask.sum().item() == 0:
            return input_ids, labels

        prob = torch.full_like(input_ids, self.mlm_probability, dtype=torch.float)
        prob = prob * can_mask.float()
        mask_positions = torch.bernoulli(prob, generator=generator).to(dtype=torch.bool)

        if mask_positions.sum().item() == 0:
            idxs = torch.nonzero(can_mask, as_tuple=False).view(-1)
            mask_positions[idxs[0]] = True

        labels[mask_positions] = input_ids[mask_positions]

        rand = torch.rand(
            input_ids.shape,
            dtype=torch.float,
            device=input_ids.device,
            generator=generator,
        )
        input_ids = input_ids.clone()

        mask_mask = mask_positions & (rand < 0.8)
        input_ids[mask_mask] = self.mask_id

        random_mask = mask_positions & (rand >= 0.8) & (rand < 0.9)
        random_indices = torch.randint(
            low=0,
            high=self.random_token_ids.numel(),
            size=input_ids.shape,
            dtype=torch.long,
            device=input_ids.device,
            generator=generator,
        )
        random_tokens = self.random_token_ids.to(input_ids.device)[random_indices]
        input_ids[random_mask] = random_tokens[random_mask]

        return input_ids, labels

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]

        if self.frame == "random":
            if self.kmer > 1:
                offset = torch.randint(low=0, high=self.kmer, size=(1,)).item()
            else:
                offset = 0
        else:
            offset = int(self.frame)

        seq = seq[offset:]

        if self.kmer > 1:
            L = (len(seq) // self.kmer) * self.kmer
            seq = seq[:L]

        encoded = self.tokenizer(
            seq,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=self.add_eos,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0).long()
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0).long()

        if self.mlm:
            generator = None
            if self.deterministic_mlm:
                generator = torch.Generator().manual_seed(self.mlm_seed + int(idx))
            input_ids, labels = self._apply_mlm(
                input_ids,
                attention_mask,
                generator=generator,
            )
        else:
            labels = input_ids.clone()

        return input_ids, labels
