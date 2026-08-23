"""Single-nucleotide DNA tokenizer for Maize DNA-Mamba."""

from typing import List, Optional, Dict, Sequence, Tuple

from transformers import PreTrainedTokenizer


class CaduceusTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids"]

    DNA_CHARACTERS = ("A", "C", "G", "T", "N")
    DNA_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

    def __init__(
        self,
        model_max_length: int,
        characters: Optional[Sequence[str]] = None,
        sequence_type: str = "dna",
        complement_map=None,
        bos_token="[BOS]",
        eos_token="[SEP]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs
    ):
        """Character tokenizer for same-position nucleotide modelling.

        Special tokens and ids:
            "[CLS]": 0
            "[SEP]": 1
            "[BOS]": 2
            "[MASK]": 3
            "[PAD]": 4
            "[RESERVED]": 5
            "[UNK]": 6

        Character ids start from 7.

        Args:
            model_max_length (int): Model maximum sequence length.
            characters (Sequence[str]): Optional explicit nucleotide alphabet.
            sequence_type (str): Must be ``dna``. The explicit argument keeps
                Hydra configurations self-documenting and rejects accidental
                RNA input early.
            complement_map (Optional[Dict[str, str]]): Complement dictionary.
        """
        sequence_type = str(sequence_type).strip().lower()
        if sequence_type != "dna":
            raise ValueError("Maize DNA-Mamba requires sequence_type='dna'")
        if characters is None:
            characters = self.DNA_CHARACTERS
        characters = tuple(str(character).upper() for character in characters)
        if len(set(characters)) != len(characters):
            raise ValueError("Tokenizer characters must be unique")

        if complement_map is None:
            complement_map = self.DNA_COMPLEMENT

        self.sequence_type = sequence_type
        self.characters = characters
        self.model_max_length = model_max_length

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        add_prefix_space = kwargs.pop("add_prefix_space", False)
        padding_side = kwargs.pop("padding_side", "left")

        self._complement_map = {}
        for token_str, token_id in self._vocab_str_to_int.items():
            if token_str in complement_map and complement_map[token_str] in self._vocab_str_to_int:
                complement_token_id = self._vocab_str_to_int[complement_map[token_str]]
            else:
                complement_token_id = token_id
            self._complement_map[token_id] = complement_token_id

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=add_prefix_space,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    @property
    def complement_map(self) -> Dict[int, int]:
        return self._complement_map

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text.upper())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        result = token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple:
        return ()
