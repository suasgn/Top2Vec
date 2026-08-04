Contextual Top2Vec embedding models
===================================

Contextual Top2Vec accepts a Hugging Face model identifier, a local model
directory, or an instantiated Transformers model. Model names and paths are
passed directly to Hugging Face; there is no contextual model whitelist or
short-name alias mapping.

Hugging Face identifier
-----------------------

.. code-block:: python

   from top2vec import Top2Vec

   topic_model = Top2Vec(
       documents=documents,
       contextual_top2vec=True,
       embedding_model="sentence-transformers/all-MiniLM-L6-v2",
   )

Local model directory
---------------------

A local directory that contains both model and tokenizer files can be passed
directly:

.. code-block:: python

   topic_model = Top2Vec(
       documents=documents,
       contextual_top2vec=True,
       embedding_model="/path/to/local/model",
   )

Model and tokenizer instances
-----------------------------

For an instantiated model, pass ``embedding_tokenizer`` when the tokenizer
cannot be inferred from the model's ``name_or_path`` metadata. The tokenizer
may itself be an instance, identifier, or local path.

.. code-block:: python

   from transformers import AutoModel, AutoTokenizer
   from top2vec import Top2Vec

   model_id = "sentence-transformers/all-MiniLM-L6-v2"
   encoder = AutoModel.from_pretrained(model_id)
   tokenizer = AutoTokenizer.from_pretrained(model_id)

   topic_model = Top2Vec(
       documents=documents,
       contextual_top2vec=True,
       embedding_model=encoder,
       embedding_tokenizer=tokenizer,
   )

Required capabilities
---------------------

A compatible model must:

* Return a finite three-dimensional ``last_hidden_state`` with shape
  ``(batch_size, sequence_length, embedding_dimension)``.
* Have a tokenizer that supports batched padding and truncation and returns
  ``input_ids`` and ``attention_mask`` as PyTorch tensors.
* Produce non-special tokens for normal text.
* Use the same hidden-state space for vocabulary and document-token
  embeddings.

Top2Vec checks these capabilities with a small forward pass before processing
the corpus. Padding and special tokens are excluded from vocabulary pooling and
document topic assignment.

Recommended specifications
--------------------------

For best results, use a bidirectional encoder such as BERT, RoBERTa, MiniLM,
MPNet, or XLM-R that covers the corpus language and has been trained for useful
semantic similarity. Prefer a model with:

* A configured padding token.
* A positional limit appropriate for the document length.
* An embedding dimension and model size that fit the available memory.
* Stable token-level semantic representations, not only a pooled sentence
  representation.

Decoder-only models are not recommended because their contextual states are
causal and therefore asymmetric. Models that expose only a pooled sentence
embedding are incompatible because Contextual Top2Vec requires one vector for
each token.

The embedding dimension is the number of values in each token vector.
Contextual Top2Vec does not require 768 dimensions: 384 is a useful
lower-memory starting point, 768 is a strong default similar to the model used
in the paper, and 1024 may preserve more information but costs more memory and
compute. A larger dimension is not automatically more accurate; training data,
language coverage, tokenizer quality, and token-space geometry matter more.

Bahasa Indonesia semantic-similarity candidates
------------------------------------------------

The Indonesian values currently displayed by the
`MTEB Multilingual v2 language view
<https://mteb-leaderboard.hf.space/benchmark/MTEB%28Multilingual%2C%20v2%29?s.lang=lang%3AIndonesian&d.lang=desc&st=1&openreq=weights&tab=perf_language&minSize=1&maxSize=1995.262314968881>`_
are explicitly labelled as simulated examples. They are therefore not used to
rank models here.

This is an ordered evaluation shortlist, not a claim that model 1 has a higher
verified Indonesian score than model 2. Selection prioritizes bidirectional
encoders, Indonesian or broad multilingual coverage, sentence-similarity,
paraphrase, contrastive, or retrieval training, and practical use with raw
contextual token states. This follows the model requirements described in the
`Contextual Top2Vec paper
<https://aclanthology.org/2024.findings-emnlp.790/>`_.

Compatibility was checked on 2026-08-04 using the project's locked Transformers
version. Every entry resolves through ``AutoConfig``, ``AutoTokenizer``, and the
standard ``AutoModel`` mapping without ``trust_remote_code=True``. This proves
loader compatibility, not topic quality; Top2Vec also performs a forward-pass
capability check before training.

.. list-table:: 20 candidates for Indonesian semantic similarity
   :header-rows: 1
   :widths: 5 43 9 17 26

   * - Order
     - Hugging Face model
     - Token dim
     - Evidence category
     - Guidance
   * - 1
     - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2 <https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2>`_
     - 768
     - Multilingual paraphrase
     - Best first semantic-similarity baseline.
   * - 2
     - `intfloat/multilingual-e5-base <https://huggingface.co/intfloat/multilingual-e5-base>`_
     - 768
     - Multilingual contrastive
     - Strong balance of quality, memory, and Indonesian evidence.
   * - 3
     - `intfloat/multilingual-e5-large <https://huggingface.co/intfloat/multilingual-e5-large>`_
     - 1024
     - Multilingual contrastive
     - Quality-first E5 option; higher memory cost.
   * - 4
     - `BAAI/bge-m3 <https://huggingface.co/BAAI/bge-m3>`_
     - 1024
     - Multilingual retrieval
     - Modern long-context candidate; benchmark on topic spans.
   * - 5
     - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 <https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2>`_
     - 384
     - Multilingual paraphrase
     - Best lightweight semantic-similarity baseline.
   * - 6
     - `intfloat/multilingual-e5-small <https://huggingface.co/intfloat/multilingual-e5-small>`_
     - 384
     - Multilingual contrastive
     - Lightweight E5 alternative.
   * - 7
     - `Snowflake/snowflake-arctic-embed-l-v2.0 <https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0>`_
     - 1024
     - Multilingual retrieval
     - Long-context, high-memory candidate.
   * - 8
     - `intfloat/multilingual-e5-large-instruct <https://huggingface.co/intfloat/multilingual-e5-large-instruct>`_
     - 1024
     - Instructed contrastive
     - Strong pooled embeddings; instructions are not applied automatically.
   * - 9
     - `sentence-transformers/LaBSE <https://huggingface.co/sentence-transformers/LaBSE>`_
     - 768
     - Cross-lingual similarity
     - Useful alignment baseline; its Sentence-Transformers dense head is not used.
   * - 10
     - `sentence-transformers/distiluse-base-multilingual-cased-v2 <https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2>`_
     - 768
     - Multilingual similarity
     - Established lower-compute baseline.
   * - 11
     - `Fatahillah01/indobert-sts-indonesian <https://huggingface.co/Fatahillah01/indobert-sts-indonesian>`_
     - 768
     - Indonesian STS
     - Direct Indonesian STS candidate; maximum training length is 128.
   * - 12
     - `LazarusNLP/all-indo-e5-small-v4 <https://huggingface.co/LazarusNLP/all-indo-e5-small-v4>`_
     - 384
     - Indonesian contrastive
     - Practical Indonesian-specific retrieval and clustering candidate.
   * - 13
     - `rzkamalia/fine-tune-paraphrase-multilingual-MiniLM-L12-v2-version-2 <https://huggingface.co/rzkamalia/fine-tune-paraphrase-multilingual-MiniLM-L12-v2-version-2>`_
     - 384
     - Indonesian STS
     - Lightweight Indonesian STS fine-tune; validate beyond translated STS data.
   * - 14
     - `alphaedge-ai/multilingual-e5-small-ind-32768 <https://huggingface.co/alphaedge-ai/multilingual-e5-small-ind-32768>`_
     - 384
     - Indonesian-adapted E5
     - Vocabulary-trimmed option for Indonesian efficiency.
   * - 15
     - `naufalihsan/indonesian-sbert-large <https://huggingface.co/naufalihsan/indonesian-sbert-large>`_
     - 1024
     - Indonesian SBERT
     - High-memory candidate; model-card evaluation detail is limited.
   * - 16
     - `sentence-transformers/stsb-xlm-r-multilingual <https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual>`_
     - 768
     - Multilingual STS
     - Direct STS-oriented legacy baseline.
   * - 17
     - `sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking <https://huggingface.co/sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking>`_
     - 768
     - NLI, STS, paraphrase
     - Broad similarity-training baseline.
   * - 18
     - `sentence-transformers/paraphrase-xlm-r-multilingual-v1 <https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1>`_
     - 768
     - Multilingual paraphrase
     - Older XLM-R paraphrase baseline.
   * - 19
     - `sentence-transformers/use-cmlm-multilingual <https://huggingface.co/sentence-transformers/use-cmlm-multilingual>`_
     - 768
     - Multilingual similarity
     - Additional bidirectional semantic baseline.
   * - 20
     - `sentence-transformers/quora-distilbert-multilingual <https://huggingface.co/sentence-transformers/quora-distilbert-multilingual>`_
     - 768
     - Multilingual paraphrase
     - Quora-focused legacy baseline.

Start by comparing orders 1, 2, and 5. They provide a useful 768-versus-384
dimension comparison without beginning with the largest models. Add order 11
when the corpus is predominantly Indonesian, then try orders 3 or 4 if memory
allows. Evaluate topic coherence, topic diversity, and topic-span quality on the
actual corpus; sentence-level leaderboard performance does not guarantee useful
token-level topic geometry.

Contextual Top2Vec reads the transformer's token states and does not reproduce
every Sentence-Transformers pooling, projection, normalization, prompt, or
instruction step. Those models remain useful candidates because semantic
training shapes the encoder, but their published sentence-level results are not
directly transferable to this pipeline.

Input length and memory
-----------------------

The effective input length is the smallest of
``contextual_model_max_length`` (512 by default), the tokenizer limit, and the
model positional limit. Longer text is truncated. Memory use grows with batch
size, sequence length, hidden dimension, and model depth; lower
``embedding_batch_size`` when necessary.
