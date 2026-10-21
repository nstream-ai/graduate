"""Training: per-call SFT examples, chat-template encoding with target-only loss, LoRA training
with an overfit check and probe logging, and decoding student output back into actions.

`data`, `encode` and `decode` are pure Python; `sft` needs the `train` dependency group.
"""

from graduate.train.data import SFTExample, build_examples, to_chat, tools_spec
from graduate.train.decode import parse_completion
from graduate.train.encode import Encoded, TemplateMismatchError, encode, ensure_template, render

__all__ = [
    "Encoded",
    "SFTExample",
    "TemplateMismatchError",
    "build_examples",
    "encode",
    "ensure_template",
    "parse_completion",
    "render",
    "to_chat",
    "tools_spec",
]
