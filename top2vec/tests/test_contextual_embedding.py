from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from top2vec.embedding import (
    average_embeddings,
    contextual_token_embeddings,
    load_contextual_model,
    validate_contextual_model,
)


class FakeTokenizer:
    model_max_length = 32
    all_special_ids = [0, 101, 102]

    def __call__(
        self,
        texts,
        padding,
        max_length,
        truncation,
        return_special_tokens_mask,
        return_tensors,
    ):
        encoded = []
        for index, _ in enumerate(texts):
            content = [10 + index, 20 + index] if index == 0 else [10 + index]
            encoded.append([101, *content, 102])

        sequence_length = max(len(row) for row in encoded)
        input_ids = []
        attention_mask = []
        special_tokens_mask = []
        for row in encoded:
            padding_length = sequence_length - len(row)
            input_ids.append(row + [0] * padding_length)
            attention_mask.append([1] * len(row) + [0] * padding_length)
            special_tokens_mask.append(
                [1] + [0] * (len(row) - 2) + [1] + [1] * padding_length
            )

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "special_tokens_mask": torch.tensor(special_tokens_mask),
        }

    def convert_ids_to_tokens(self, token_ids):
        return [f"token-{token_id}" for token_id in token_ids]


class FakeModel(torch.nn.Module):
    def __init__(self, name_or_path="fake/model", max_position_embeddings=16):
        super().__init__()
        self.config = SimpleNamespace(
            _name_or_path=name_or_path,
            max_position_embeddings=max_position_embeddings,
        )
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        values = input_ids.to(torch.float32)
        hidden = torch.stack((values, values + 1, values + 2), dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_loads_identifier_without_alias_mapping(monkeypatch):
    model = FakeModel()
    tokenizer = FakeTokenizer()
    loaded_sources = []

    monkeypatch.setattr(
        "top2vec.embedding.AutoModel.from_pretrained",
        lambda source: loaded_sources.append(("model", source)) or model,
    )
    monkeypatch.setattr(
        "top2vec.embedding.AutoTokenizer.from_pretrained",
        lambda source: loaded_sources.append(("tokenizer", source)) or tokenizer,
    )

    loaded_model, loaded_tokenizer, reference = load_contextual_model("org/model-name")

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert reference == "org/model-name"
    assert loaded_sources == [
        ("model", "org/model-name"),
        ("tokenizer", "org/model-name"),
    ]


def test_loads_local_path(monkeypatch, tmp_path):
    model = FakeModel()
    tokenizer = FakeTokenizer()
    loaded_sources = []

    monkeypatch.setattr(
        "top2vec.embedding.AutoModel.from_pretrained",
        lambda source: loaded_sources.append(source) or model,
    )
    monkeypatch.setattr(
        "top2vec.embedding.AutoTokenizer.from_pretrained",
        lambda source: loaded_sources.append(source) or tokenizer,
    )

    loaded_model, loaded_tokenizer, reference = load_contextual_model(Path(tmp_path))

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert reference == str(tmp_path)
    assert loaded_sources == [str(tmp_path), str(tmp_path)]


def test_accepts_model_and_tokenizer_instances():
    model = FakeModel()
    tokenizer = FakeTokenizer()

    loaded_model, loaded_tokenizer, reference = load_contextual_model(
        model,
        embedding_tokenizer=tokenizer,
    )

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert reference == "fake/model"


def test_model_instance_requires_resolvable_tokenizer():
    model = FakeModel(name_or_path="")

    with pytest.raises(ValueError, match="tokenizer could not be inferred"):
        load_contextual_model(model)


def test_validation_caps_sequence_length_to_model_limit(monkeypatch):
    monkeypatch.setattr("top2vec.embedding._device", lambda: torch.device("cpu"))

    max_length = validate_contextual_model(
        FakeModel(max_position_embeddings=16),
        FakeTokenizer(),
        requested_max_length=64,
    )

    assert max_length == 16


def test_contextual_embeddings_exclude_padding_and_special_tokens(monkeypatch):
    monkeypatch.setattr("top2vec.embedding._device", lambda: torch.device("cpu"))

    embeddings, tokens, labels = contextual_token_embeddings(
        ["first", "second"],
        batch_size=2,
        model_max_length=16,
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
    )

    assert [embedding.shape for embedding in embeddings] == [(2, 3), (1, 3)]
    assert tokens == [["token-10", "token-20"], ["token-11"]]
    assert labels == [0, 0, 1]


def test_vocabulary_pooling_excludes_padding_and_special_tokens(monkeypatch):
    monkeypatch.setattr("top2vec.embedding._device", lambda: torch.device("cpu"))

    vectors = average_embeddings(
        ["first", "second"],
        batch_size=2,
        model_max_length=16,
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
    )

    expected = np.array([[15, 16, 17], [11, 12, 13]], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(vectors, expected, rtol=1e-6)


class PooledOnlyModel(FakeModel):
    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(pooler_output=torch.ones(input_ids.shape[0], 3))


def test_rejects_models_without_token_hidden_states(monkeypatch):
    monkeypatch.setattr("top2vec.embedding._device", lambda: torch.device("cpu"))

    with pytest.raises(ValueError, match="must return last_hidden_state"):
        validate_contextual_model(PooledOnlyModel(), FakeTokenizer())
