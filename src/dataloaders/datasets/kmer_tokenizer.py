import itertools
from typing import Dict, List, Optional, Sequence
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


def build_kmer_vocab(k: int = 3, alphabet: str = "ACGU") -> List[str]:
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]


class KmerTokenizer(PreTrainedTokenizer):
    """
    非重叠 k-mer tokenizer：每 k 个碱基作为一个 token（步长=k）
    例：k=3: ACGTTA -> ["ACG","TTA"]
    """
    def __init__(self, k: int = 3, alphabet: str = "ACGU", model_max_length: int = 1024, padding_side: str = "right", **kwargs):
        self.k = int(k)
        self.alphabet = alphabet
        self.model_max_length = model_max_length

        # special tokens
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[EOS]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        kmers = build_kmer_vocab(self.k, self.alphabet)

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{tok: i + 7 for i, tok in enumerate(kmers)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
                     # ===== 给 RCPS 用的：token id -> 反向互补 token id 映射 =====
        base_comp = {"A": "U", "C": "G", "G": "C", "U": "A"}

        def rc_kmer(tok: str) -> str:
            # tok 形如 "ACG"
            return "".join(base_comp.get(b, "N") for b in reversed(tok))

        self.complement_map = {}
        for tok, tid in self._vocab_str_to_int.items():
            if tok.startswith("["):  # special token: 映射到自己
                self.complement_map[tid] = tid
            else:
                rctok = rc_kmer(tok)
                self.complement_map[tid] = self._vocab_str_to_int.get(rctok, self._vocab_str_to_int["[UNK]"])


        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    def _tokenize(self, text: str) -> List[str]:
        text = text.upper()
        L = (len(text) // self.k) * self.k
        text = text[:L]

        toks = []
        for i in range(0, L, self.k):
            tok = text[i:i+self.k]
            # 如果含有非字母表字符，标为 [UNK]
            if any(ch not in self.alphabet for ch in tok):
                toks.append("[UNK]")
            else:
                toks.append(tok)
        return toks

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        # 反拼接
        return "".join([t if t.startswith("[") else t for t in tokens])

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result
