import torch
from torch.utils.data import Dataset


class RNACentralFastaDataset(Dataset):
    """
    RNAcentral FASTA dataset

    处理流程:
        upper()
        T -> U
        非 AUCG -> N
    """

    def __init__(
        self,
        fasta_file: str,
        tokenizer,
        max_length: int = 1024,
        add_eos: bool = True,
        mlm: bool = False,
        mlm_probability: float = 0.15,
        ignore_id: int | None = None,
        kmer: int = 1,
        frame: int | str = 0,
    ) -> None:

        super().__init__()

        self.fasta_file = fasta_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_eos = add_eos
        self.mlm = mlm
        self.mlm_probability = float(mlm_probability)
        self.kmer = kmer
        self.frame = frame

        tok_pad = getattr(self.tokenizer, "pad_token_id", None)
        self.pad_id = tok_pad if tok_pad is not None else 4
        self.ignore_id = self.pad_id if ignore_id is None else int(ignore_id)

        self.mask_id = getattr(self.tokenizer, "mask_token_id", None)
        self.eos_id = getattr(self.tokenizer, "eos_token_id", None)

        if self.mlm and self.mask_id is None:
            raise ValueError("Tokenizer has no mask_token_id but mlm=True")

        self.headers = []
        self.sequences = []

        allowed = {"A", "U", "C", "G"}

        current_header = None
        current_seq = []

        with open(self.fasta_file, "r") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                if line.startswith(">"):

                    if current_header is not None and len(current_seq) > 0:

                        seq = "".join(current_seq).upper()
                        seq = seq.replace("T", "U")

                        if any(ch not in allowed for ch in seq):
                            pass  # 丢弃
                        else:
                            self.headers.append(current_header)
                            self.sequences.append(seq)

                    current_header = line[1:]
                    current_seq = []

                else:

                    current_seq.append(line)

        if current_header is not None and len(current_seq) > 0:

            seq = "".join(current_seq).upper()
            seq = seq.replace("T", "U")

            if any(ch not in allowed for ch in seq):
                pass
            else:
                self.headers.append(current_header)
                self.sequences.append(seq)

        print(f"[RNACentralFastaDataset] Loaded {len(self.sequences)} sequences from {self.fasta_file}")

    def __len__(self):
        return len(self.sequences)

    def _apply_mlm(self, input_ids, attention_mask):

        labels = torch.full_like(input_ids, fill_value=self.ignore_id)

        can_mask = input_ids != self.pad_id

        if self.eos_id is not None:
            can_mask = can_mask & (input_ids != self.eos_id)

        if attention_mask is not None:
            can_mask = can_mask & attention_mask.bool()

        if can_mask.sum() == 0:
            return input_ids, labels

        prob = torch.full_like(input_ids, self.mlm_probability, dtype=torch.float)
        prob = prob * can_mask.float()

        mask_positions = torch.bernoulli(prob).bool()

        if mask_positions.sum() == 0:
            idxs = torch.nonzero(can_mask).view(-1)
            mask_positions[idxs[0]] = True

        labels[mask_positions] = input_ids[mask_positions]

        rand = torch.rand_like(input_ids.float())

        input_ids = input_ids.clone()

        mask_mask = mask_positions & (rand < 0.8)
        input_ids[mask_mask] = self.mask_id

        random_mask = mask_positions & (rand >= 0.8) & (rand < 0.9)

        vocab_size = getattr(self.tokenizer, "vocab_size", None)

        if vocab_size is None:
            vocab_size = len(self.tokenizer)

        random_tokens = torch.randint(
            0,
            vocab_size,
            size=input_ids.shape,
            dtype=torch.long
        )

        input_ids[random_mask] = random_tokens[random_mask]

        return input_ids, labels

    def __getitem__(self, idx):

        seq = self.sequences[idx]

        if self.frame == "random":

            if self.kmer > 1:
                offset = torch.randint(0, self.kmer, (1,)).item()
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
            input_ids, labels = self._apply_mlm(input_ids, attention_mask)
        else:
            labels = input_ids.clone()

        return input_ids, labels