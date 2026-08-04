from types import SimpleNamespace

import numpy as np

from top2vec import embedding
from top2vec.top2vec import Top2Vec


def _bare_model(embedding_model, embed, show_progress_bar=True):
    model = Top2Vec.__new__(Top2Vec)
    model.embedding_model = embedding_model
    model.embed = embed
    model.show_progress_bar = show_progress_bar
    model._check_import_status = lambda: None
    model._check_model_status = lambda: None
    return model


def test_sentence_transformer_receives_progress_setting():
    calls = []

    def encode(documents, **kwargs):
        calls.append(kwargs)
        return np.ones((len(documents), 2))

    model = _bare_model("all-MiniLM-L6-v2", encode)

    model._embed_documents(["one", "two"], batch_size=2)

    assert calls == [{"batch_size": 2, "show_progress_bar": True}]


def test_custom_callable_does_not_receive_progress_keyword():
    calls = []

    def encode(documents):
        calls.append(list(documents))
        return np.ones((len(documents), 2))

    model = _bare_model("custom", encode)

    model._embed_documents(["one", "two", "three"], batch_size=2)

    assert calls == [["one", "two"], ["three"]]


def test_query_embedding_is_silent_for_sentence_transformers():
    calls = []

    def encode(documents, **kwargs):
        calls.append(kwargs)
        return np.ones((len(documents), 2))

    model = _bare_model("all-MiniLM-L6-v2", encode)

    model._embed_query("query")

    assert calls == [{"show_progress_bar": False}]


def test_contextual_embedding_progress_can_be_disabled(monkeypatch):
    tqdm_calls = []

    def fake_tqdm(iterable, **kwargs):
        tqdm_calls.append(kwargs)
        return iterable

    class Tokenizer:
        all_special_ids = []

        def __call__(self, texts, **kwargs):
            import torch

            batch_size = len(texts)
            return {
                "input_ids": torch.ones((batch_size, 2), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, 2), dtype=torch.long),
                "special_tokens_mask": torch.zeros((batch_size, 2), dtype=torch.long),
            }

    class Model:
        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            import torch

            shape = kwargs["input_ids"].shape
            return SimpleNamespace(last_hidden_state=torch.ones((*shape, 3)))

    monkeypatch.setattr(embedding, "tqdm", fake_tqdm)
    embedding.average_embeddings(
        ["one", "two"],
        batch_size=2,
        model_max_length=8,
        model=Model(),
        tokenizer=Tokenizer(),
        show_progress_bar=False,
    )

    assert tqdm_calls == [{"desc": "Embedding vocabulary", "disable": True}]


def test_contextual_postprocessing_uses_progress_setting(monkeypatch):
    tqdm_calls = []

    def fake_tqdm(iterable, **kwargs):
        tqdm_calls.append(kwargs)
        return iterable

    monkeypatch.setattr(embedding, "tqdm", fake_tqdm)
    token_embeddings = [np.ones((2, 3))]
    tokens = [["one", "two"]]

    embedding.sliding_window_average(
        token_embeddings,
        tokens,
        window_size=2,
        stride=1,
        show_progress_bar=True,
    )
    embedding.smooth_document_token_embeddings(
        token_embeddings,
        show_progress_bar=False,
    )

    assert tqdm_calls[0]["disable"] is False
    assert tqdm_calls[1]["disable"] is True
