# Text-to-SQL RAG Library

A Retrieval-Augmented Generation (RAG) utility for managing and querying SQL query libraries. This library provides both programmatic and command-line interfaces for integrating similarity-based query matching into your codebase.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. All commands should be run via `uv run`. First initialise the proper venv:

```bash
uv venv
uv sync
```

## Quick Start (Programmatic)

```python
from rag_helper.query_library import QueryLibrary
from rag_helper.rag import RAGProcessor, RAGConfig, QueryLibraryManager

# Load an existing query library
ql = QueryLibrary.load("path/to/querylib.db")

# Initialize the RAG processor
config = RAGConfig(sim_threshold=0.5)
processor = RAGProcessor(config)

# Inject the library
manager = QueryLibraryManager.get_instance()
manager.querylib = ql
ql.load_embedding_model("BAAI/bge-large-en-v1.5")

# Get similar queries
results = processor.get_similar_queries("How many patients last year?", top_k=5)
print(results)
```

---

## CLI Reference (`rag-cli`)

The pipeline CLI is designed for automation and integration into data workflows. All commands support an optional `--output` / `-o` flag to export results to a file (`.csv`, `.json`, `.xlsx`).

### Global Arguments

| Argument | Description |
|---|---|
| `--file`, `-f` | Path to the `.db` query library file. |
| `--overview` | Show a quick usage guide and exit. |

### Commands

#### `view`

View the first N rows of the library.

| Argument | Description | Default |
|---|---|---|
| `--limit`, `-l` | Number of records to show. | 10 |
| `--output`, `-o` | Path to export results. | None |

**Example:**
```bash
rag-cli -f lib.db view --limit 20 -o results.csv
```

---

#### `info`

View full details for a specific record ID.

| Argument | Description |
|---|---|
| `id` | The record ID to inspect. |
| `--output`, `-o` | Path to export results. |

**Example:**
```bash
rag-cli -f lib.db info 42 -o record.json
```

---

#### `search`

Search questions for a keyword (case-insensitive). Note: see first point below in personal remarks. it is currently very unintuitive and hardcoded what columns is searches / matches against. this is an implementation remainder of its origins in a very specific pipeline. will change.

| Argument | Description |
|---|---|
| `term` | The keyword to search for. |
| `--output`, `-o` | Path to export results. |

**Example:**
```bash
rag-cli -f lib.db search "biopsy" -o search_results.xlsx
```

---

#### `match`

Perform similarity matching for one or more natural language queries. Note: same thing about unintuitive what cols it matches and what the output col is...

| Argument | Description | Default |
|---|---|---|
| `query` | The question(s) to match. | (required) |
| `--k`, `-k` | Number of top results to return. | 5 |
| `--threshold`, `-t` | Minimum similarity score (0.0 to 1.0). | 0.0 |
| `--output`, `-o` | Path to export results. | None |

**Example:**
```bash
rag-cli -f lib.db match "How many patients?" -k 3 -t 0.6 -o matches.csv
```

---

#### `docs`

Show auto-generated documentation for all CLI commands. This documentation is dynamically generated from the function docstrings and argparse configuration, ensuring it is always up-to-date with the source code.

**Example:**
```bash
rag-cli docs
```

---

## Interactive CLI (`rag-cli-interactive`)

For manual exploration, a menu-driven interactive CLI is available. It provides all the same functionality as the pipeline CLI, plus database scanning and system overview features.

```bash
uv run rag-cli-interactive
```

> **Note:** The "View Full CLI Documentation" option (and the `--docs` flag) displays documentation that is automatically synchronized with the docstrings in the source code. This means the displayed information is always accurate and reflects the current implementation.

---

## Testing

```bash
uv run pytest
```

---

## Architecture

- `query_library.py`: Core data management (SQLite storage, embeddings).
- `rag.py`: RAG flow coordinator with singleton manager.
- `cli.py`: Pipeline CLI entry point.
- `interactive_cli.py`: Menu-driven research tool.

## personal remarks & ToDos

- seperate QA and cohort generator entries to different dbs instead of having to handle that aftwards -> that makes the querylib initialisation easier
    - could then just specify [these columns i want to find a match with (naturally includes paraphrases)] and [this columns i want as an output for a match] ((or you just put all of this into the RAG.py idk))
        - instead of specifying four columns int he querlib cuz they insisted on having Q&A question samples and cohort generation question samples in the same db...
    - then you can also add this to the CLI so you are fully configurable straight from command line and very flexible with different qlib files etc

    - i think its most sensible to use one knowledbge base for each 'thing' you do.

    -> TODO: completly decouple the knowledgebases and for the querymanager / rag stuff do not assume any prior knowledge about the knowledgebase!

- Only the QA questions (which is about 'How many' and not cohorts) have their questions in natural language; the cohort generation stuff is already saved as processed {inclusion:..., exclusion:...} criteria. Which i suppose is sensible because we want to RAG search against masked questions anyway and having that in a standard form (via a pre-processing LLM) is quite sensible. Just want to point this out for anyone who is exploring the knowledgebase. These are all kind of details which would be adressed when building pipelines, e.g. via makefiles. Thats why kept the RAG tool flexible and CLI based

- the main point of this whole exercise to eventually have a really clean start for a RAG system which we actually understand. and for reproducible testing via eg snakemake i really like having cli's. the (more or less) decoupled nature of the codebase should make debugging and expansion actually nice instead of it being a pain

- this spring clean was the prelude for checking how language affects the reliability of the rag system. actually measure difference between
    - original english input matching against original english knowledge base (with hold out testing; this will require a TODO: allow querymanager to temporarily alter the knowledgebase)
    - original english input translated into german matching against orignal english kb translated into german
    - german input translated into english matching against orignal english kb




