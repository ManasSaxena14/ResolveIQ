# RAG Pipeline

This document describes Phase 1 of ResolveIQ: the assistant that drafts a reply to a
support question from the help center and cites the pages it used.

## Overview

Retrieval-augmented generation (RAG) finds the relevant passages first and then asks a
language model to answer using only those passages. This keeps answers grounded in the
documentation and lets every fact be traced to a source page.

```
Offline (when documents change)
  data/kb/*  ->  load  ->  clean  ->  chunk  ->  embed  ->  Chroma + BM25 index

Online (per question)
  question  ->  embed  ->  vector search (+ BM25, fused with RRF)  ->  optional rerank
            ->  refusal rule  ->  prompt  ->  language model  ->  answer + sources
```

## Components

| Module | Responsibility |
|---|---|
| `rag/loaders.py` | Reads PDF (per page), Markdown, text and HTML files from `data/kb/` |
| `rag/cleaning.py` | Removes running headers and footers, repairs line breaks, drops empty pages |
| `rag/chunking.py` | Token-limited chunks with overlap, split on word boundaries |
| `rag/embeddings.py` | sentence-transformers wrapper with normalized vectors |
| `rag/store.py` | Persistent ChromaDB collection (cosine) and a JSON-backed BM25 index |
| `rag/ingest.py` | Builds the index and writes `index_manifest.json` |
| `rag/retrieve.py` | Vector search, reciprocal rank fusion with BM25, optional cross-encoder reranking |
| `rag/prompts.py` | Grounded prompt with citation format and prompt-injection guard |
| `rag/llm.py` | HTTP clients for Ollama, Gemini, Groq and OpenRouter behind one interface |
| `rag/generate.py` | `AnswerEngine`: retrieval, refusal rule, generation, source collection |
| `rag/evaluation.py` | Test set loading, automatic checks and metric aggregation |

## Design decisions

**Embedding model.** `BAAI/bge-small-en-v1.5` accepts 512 tokens per input. The default
chunk size of 500 tokens therefore fits without truncation, which would not be true for
`all-MiniLM-L6-v2` (256 tokens). Chunk sizes are measured with the embedding model's own
tokenizer, so the configured limit is exact.

**Refusal rule.** Before calling the language model, the engine compares the highest
cosine similarity from vector search with `rag.refusal_threshold`. Questions below the
threshold are refused immediately with the fixed sentence
"I could not find this in the documents." Fused and reranked scores are not used for this
rule because they are not comparable across questions. Questions above the threshold can
still be refused by the model itself, which the prompt instructs to do when the context
does not contain the answer; such replies are detected and returned as refusals.

**Threshold choice.** The threshold was set from the distribution of best similarity
scores on the dev split, comparing answerable and out-of-scope questions. The two
distributions overlap, so no threshold separates them perfectly: the value is set just
below the lowest score of an answerable question, which refuses clearly unrelated
questions without refusing answerable ones, and leaves the overlapping cases to the
prompt-level refusal. Revisit it whenever the embedding model or the knowledge base
changes.

**Hybrid search.** BM25 catches exact terms that embeddings can miss, such as product
names and error codes. The two ranked lists are merged with reciprocal rank fusion
(`k = 60`). Hybrid search is kept only if it improves the evaluation.

**Provider switch.** `llm.provider` selects Ollama for local development or a hosted free
tier for the public demo. Model names live in the config, not in code. API keys are read
from environment variables named in the config.

**Untrusted input.** Ticket text and document content are treated as data. The system
prompt tells the model to ignore instructions found inside them, and question length is
capped.

## Running it

```bash
make index                                              # build the index
python eval/run_rag_eval.py run --split dev --retrieval-only
python eval/run_rag_eval.py run --split dev --run-name baseline
python eval/run_rag_eval.py score results/rag/baseline.csv
```

A full run needs a language model. For local use install [Ollama](https://ollama.com),
then run `ollama pull llama3.2`. To use a hosted provider instead, set `llm.provider` and
add the matching key to `.env`.

## Evaluation protocol

The test set in `eval/rag_testset.csv` has 40 hand-written questions:

| Split | Answerable | Out of scope | Purpose |
|---|---|---|---|
| dev | 23 | 7 | Tuning chunking, retrieval and the refusal threshold |
| holdout | 7 | 3 | Untouched until the final run |

Rules:

1. The test set is frozen. Questions are never edited to improve a score.
2. Settings are changed one at a time and every run is logged to MLflow (experiment `rag`).
3. The holdout split is used once, for the final configuration.
4. Source hit rate and refusal behaviour are scored automatically. Answer correctness is
   labeled by hand in the `correct` column of the results file, then scored.

### Settings to compare

| Setting | Values |
|---|---|
| `chunk_tokens` | 300, 500, 800 |
| `chunk_overlap` | 0, 50, 100 |
| `top_k` | 3, 5, 8 |
| `hybrid` | off, on |
| `rerank` | off, on |
| `refusal_threshold` | three values from the score histogram |
| language model | llama3.2 3B, a larger local model, a hosted model |

Measured results are recorded in the evaluation report at the end of the project. Only
numbers produced by logged runs are reported.

## Limitations

- The knowledge base is synthetic and small, so results describe the method, not
  performance on a real company's documentation.
- Articles are short, so most produce a single chunk at the default size.
- English only.
