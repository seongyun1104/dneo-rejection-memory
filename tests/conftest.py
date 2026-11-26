import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent.parent
DNEO_SRC = Path(os.environ.get("DNEO_SRC", HERE.parent / "external" / "DNeo" / "src"))
sys.path[:0] = [str(HERE), str(DNEO_SRC)]

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from dneo import DNeoConfig, DNeoDraftModel  # noqa: E402


def tiny_pair(seed=0):
    torch.manual_seed(seed)
    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    target = Qwen3ForCausalLM(config).eval()
    for layer in target.model.layers:
        layer.input_layernorm.weight.data.uniform_(0.5, 1.5)
    draft_config = DNeoConfig(
        target_config=config,
        num_hidden_layers=2,
        target_layer_ids=[0, 3],
        block_size=4,
        head_rank=8,
        folded_rank=8,
        conv_group_size=8,
        mask_token_id=5,
        shared_layer_mofa=True,
    )
    model = DNeoDraftModel(draft_config).eval()
    for layer in model.draft.layers:
        for conv in (layer.attention_conv, layer.mlp_conv):
            conv.base_kernel.data[:, 0].fill_(1.0)
    model.bind_target(target)
    return target, model


@pytest.fixture
def pair():
    return tiny_pair()


@pytest.fixture(autouse=True)
def no_cuda_sync(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
