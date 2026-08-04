import os

import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def _source_name(source):
    if isinstance(source, (str, os.PathLike)):
        return os.fspath(source)
    return None


def _model_name_or_path(model):
    name_or_path = getattr(model, "name_or_path", None)
    if not name_or_path:
        name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None)
    return name_or_path or None


def _load_tokenizer(tokenizer):
    tokenizer_source = _source_name(tokenizer)
    if tokenizer_source is not None:
        return AutoTokenizer.from_pretrained(tokenizer_source)
    if callable(tokenizer):
        return tokenizer
    raise ValueError(
        "embedding_tokenizer must be a Hugging Face tokenizer instance, "
        "model identifier, or local model path."
    )


def load_contextual_model(embedding_model, embedding_tokenizer=None):
    """Load a contextual model and its tokenizer.

    ``embedding_model`` may be a Hugging Face identifier, a local path, or an
    instantiated Transformers model. For model instances, a tokenizer can be
    supplied explicitly or inferred from the model's ``name_or_path``.
    """
    model_source = _source_name(embedding_model)

    if model_source is not None:
        model = AutoModel.from_pretrained(model_source)
        tokenizer_source = embedding_tokenizer or model_source
        tokenizer = _load_tokenizer(tokenizer_source)
        model_reference = model_source
    else:
        if not callable(getattr(embedding_model, "forward", None)):
            raise ValueError(
                "For contextual Top2Vec, embedding_model must be a Hugging "
                "Face model instance, model identifier, or local model path."
            )
        model = embedding_model
        inferred_source = _model_name_or_path(model)
        if embedding_tokenizer is None:
            if inferred_source is None:
                raise ValueError(
                    "A tokenizer could not be inferred from the model instance. "
                    "Pass embedding_tokenizer as a tokenizer instance, model "
                    "identifier, or local path."
                )
            embedding_tokenizer = inferred_source
        tokenizer = _load_tokenizer(embedding_tokenizer)
        model_reference = inferred_source or model.__class__.__name__

    return model, tokenizer, model_reference


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _effective_max_length(model, tokenizer, requested_max_length=512):
    """Choose a finite length supported by both the tokenizer and model."""
    limits = [requested_max_length]
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    model_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)

    # Hugging Face uses very large sentinel integers when no limit is known.
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        limits.append(tokenizer_limit)
    if isinstance(model_limit, int) and model_limit > 0:
        limits.append(model_limit)

    return min(limits)


def _tokenize(tokenizer, texts, max_length):
    try:
        inputs = tokenizer(
            texts,
            padding=True,
            max_length=max_length,
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
    except Exception as error:
        raise ValueError(
            "The contextual embedding tokenizer must support batched padding, "
            "truncation, and PyTorch tensor output. If it has no padding token, "
            "configure one before passing it to Top2Vec."
        ) from error

    missing = {"input_ids", "attention_mask"}.difference(inputs.keys())
    if missing:
        raise ValueError(
            "The contextual embedding tokenizer did not return required fields: "
            f"{', '.join(sorted(missing))}."
        )
    return inputs


def _last_hidden_state(outputs):
    hidden_state = getattr(outputs, "last_hidden_state", None)
    if hidden_state is None and isinstance(outputs, dict):
        hidden_state = outputs.get("last_hidden_state")
    if hidden_state is None:
        raise ValueError(
            "The contextual embedding model must return last_hidden_state with "
            "shape (batch_size, sequence_length, embedding_dimension)."
        )
    return hidden_state


def _special_tokens_mask(tokenizer, inputs):
    provided_mask = inputs.get("special_tokens_mask")
    if provided_mask is not None:
        return provided_mask.bool()

    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    if not special_ids:
        return torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
    return torch.tensor(
        [
            [token_id.item() in special_ids for token_id in row]
            for row in inputs["input_ids"]
        ],
        dtype=torch.bool,
    )


def validate_contextual_model(model, tokenizer, requested_max_length=512):
    """Run a small capability check and return the usable sequence length."""
    max_length = _effective_max_length(model, tokenizer, requested_max_length)
    device = _device()
    model.eval()
    model.to(device)

    inputs = _tokenize(
        tokenizer,
        ["Contextual embedding compatibility test.", "Short test."],
        max_length,
    )
    model_inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if key != "special_tokens_mask"
    }

    with torch.no_grad():
        hidden_state = _last_hidden_state(model(**model_inputs))

    expected_shape = tuple(inputs["input_ids"].shape)
    if hidden_state.ndim != 3 or tuple(hidden_state.shape[:2]) != expected_shape:
        raise ValueError(
            "The contextual embedding model returned an incompatible hidden "
            f"state shape {tuple(hidden_state.shape)}; expected "
            f"({expected_shape[0]}, {expected_shape[1]}, embedding_dimension)."
        )
    if hidden_state.shape[2] == 0 or not torch.isfinite(hidden_state).all():
        raise ValueError(
            "The contextual embedding model returned empty, NaN, or infinite "
            "token embeddings."
        )

    valid_tokens = inputs["attention_mask"].bool() & ~_special_tokens_mask(tokenizer, inputs)
    if not valid_tokens.any(dim=1).all():
        raise ValueError(
            "The contextual tokenizer produced no non-special tokens for a test input."
        )

    return max_length


def average_embeddings(
    documents,
    batch_size,
    model_max_length,
    model,
    tokenizer,
    show_progress_bar=False,
):
    device = _device()
    data_loader = DataLoader(documents, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to(device)
    pooled_embeddings = []

    with torch.no_grad():
        for batch in tqdm(
            data_loader,
            desc="Embedding vocabulary",
            disable=not show_progress_bar,
        ):
            inputs = _tokenize(tokenizer, batch, model_max_length)
            special_tokens_mask = _special_tokens_mask(tokenizer, inputs)
            valid_tokens = inputs["attention_mask"].bool() & ~special_tokens_mask
            model_inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if key != "special_tokens_mask"
            }
            hidden_state = _last_hidden_state(model(**model_inputs))
            mask = valid_tokens.to(device).unsqueeze(-1).to(hidden_state.dtype)
            pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled_embeddings.append(pooled.float().cpu().numpy())

    return normalize(np.vstack(pooled_embeddings))


def contextual_token_embeddings(
    documents,
    batch_size,
    model_max_length,
    model,
    tokenizer,
    show_progress_bar=False,
):
    device = _device()
    data_loader = DataLoader(documents, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to(device)

    document_token_embeddings = []
    document_tokens = []
    document_labels = []
    document_index = 0

    with torch.no_grad():
        for batch in tqdm(
            data_loader,
            desc="Embedding documents",
            disable=not show_progress_bar,
        ):
            inputs = _tokenize(tokenizer, batch, model_max_length)
            special_tokens_mask = _special_tokens_mask(tokenizer, inputs)
            valid_tokens = inputs["attention_mask"].bool() & ~special_tokens_mask
            model_inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if key != "special_tokens_mask"
            }
            hidden_states = _last_hidden_state(model(**model_inputs)).cpu()

            for hidden_state, valid, token_ids in zip(
                hidden_states, valid_tokens, inputs["input_ids"]
            ):
                embeddings = hidden_state[valid]
                ids = token_ids[valid].tolist()
                if not ids:
                    raise ValueError(
                        "The contextual tokenizer produced no non-special tokens "
                        f"for document {document_index}."
                    )
                if hasattr(tokenizer, "convert_ids_to_tokens"):
                    tokens = tokenizer.convert_ids_to_tokens(ids)
                else:
                    tokens = [tokenizer.decode(token_id) for token_id in ids]

                document_token_embeddings.append(embeddings.float().numpy())
                document_tokens.append(tokens)
                document_labels.extend([document_index] * len(tokens))
                document_index += 1

    return document_token_embeddings, document_tokens, document_labels


def sliding_window_average(
    document_token_embeddings,
    document_tokens,
    window_size,
    stride,
    show_progress_bar=False,
):
    averaged_embeddings = []
    chunk_tokens = []

    documents = zip(document_token_embeddings, document_tokens)
    for doc, tokens in tqdm(
        documents,
        total=len(document_token_embeddings),
        desc="Creating document chunks",
        disable=not show_progress_bar,
    ):
        doc_averages = []

        for i in range(0, len(doc), stride):
            start = i
            end = i + window_size

            if start != 0 and end > len(doc):
                start = len(doc) - window_size
                end = len(doc)

            window = doc[start:end]
            doc_averages.append(np.mean(window, axis=0))
            chunk_tokens.append(" ".join(tokens[start:end]))

        averaged_embeddings.append(doc_averages)

    return normalize(np.vstack(averaged_embeddings)), chunk_tokens


def average_adjacent_tokens(token_embeddings, window_size):
    num_tokens = token_embeddings.shape[0]
    averaged_embeddings = np.zeros_like(token_embeddings)
    token_embeddings = normalize(token_embeddings)

    for i in range(num_tokens):
        start_idx = max(0, i - window_size)
        end_idx = min(num_tokens, i + window_size + 1)
        averaged_embeddings[i] = np.mean(token_embeddings[start_idx:end_idx], axis=0)

    return averaged_embeddings


def smooth_document_token_embeddings(
    document_token_embeddings,
    window_size=2,
    show_progress_bar=False,
):
    smoothed_document_embeddings = []

    for doc in tqdm(
        document_token_embeddings,
        desc="Smoothing document token embeddings",
        disable=not show_progress_bar,
    ):
        smoothed_doc = average_adjacent_tokens(doc, window_size=window_size)
        smoothed_document_embeddings.append(smoothed_doc)

    return smoothed_document_embeddings
