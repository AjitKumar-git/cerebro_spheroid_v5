"""
Tokenizer and Special Tokens for Minimal Tool MVP
=================================================
A self-contained, deterministic tokenizer tailored for the minimal tool experiment.
Supports ASCII characters, special control tokens, and bi-directional encoding.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|user|>",
    "<|user_end|>",
    "<|tool_call|>",
    "<|tool_call_end|>",
    "<|tool_result|>",
    "<|tool_result_end|>",
    "<|answer|>",
    "<|answer_end|>",
    "<|unk|>",
]


class MinimalTokenizer:
    """Deterministic character/subword tokenizer with special control tokens."""

    def __init__(self):
        self.special_tokens = list(SPECIAL_TOKENS)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # 1. Register special tokens
        for idx, tok in enumerate(self.special_tokens):
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok

        # 2. Register printable ASCII characters
        offset = len(self.special_tokens)
        for i in range(32, 127):  # printable ascii 32 to 126
            ch = chr(i)
            self.token_to_id[ch] = offset
            self.id_to_token[offset] = ch
            offset += 1

        # Common whitespace
        for ch in ["\n", "\t", "\r"]:
            if ch not in self.token_to_id:
                self.token_to_id[ch] = offset
                self.id_to_token[offset] = ch
                offset += 1

        self.pad_token_id = self.token_to_id["<|pad|>"]
        self.unk_token_id = self.token_to_id["<|unk|>"]
        self.vocab_size = len(self.token_to_id)

    def encode(self, text: str) -> List[int]:
        """Encodes string containing text and special tokens into token IDs."""
        tokens: List[int] = []
        i = 0
        n = len(text)

        while i < n:
            matched_special = False
            for st in self.special_tokens:
                if text.startswith(st, i):
                    tokens.append(self.token_to_id[st])
                    i += len(st)
                    matched_special = True
                    break

            if not matched_special:
                ch = text[i]
                tokens.append(self.token_to_id.get(ch, self.unk_token_id))
                i += 1

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs into a string."""
        chars = []
        for tid in token_ids:
            tok = self.id_to_token.get(tid, "")
            if skip_special_tokens and tok in self.special_tokens:
                continue
            chars.append(tok)
        return "".join(chars)
