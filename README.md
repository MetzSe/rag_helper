# Text-to-SQL RAG Library

A Retrieval-Augmented Generation (RAG) utility for managing and querying SQL query libraries. This library provides both programmatic and command-line interfaces for integrating similarity-based query matching into your codebase.

This work is based on work by [Ziletti et al.](https://github.com/Bayer-Group/text-to-sql-epi-ehr-naacl2024).

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. All commands should be run via `uv run`. First initialise the proper venv:

```bash
uv venv
uv sync
```

## Quick Start (Programmatic)

The simplified API provides a high-level interface that handles library loading and RAG configuration in a single call.

```python
import rag_helper as rh

# Match queries directly
results = rh.match(
    file="path/to/querylib.db", 
    query="How many patients last year?", 
    k=5,
    threshold=0.5
)
print(results)
```

For advanced usage or direct access to the underlying manager:
```python
from rag_helper.query_library_manager import QueryLibraryManager
from rag_helper.rag import RAGProcessor, RAGConfig

ql = QueryLibraryManager.load("lib.db")
config = RAGConfig(sim_threshold=0.5)
processor = RAGProcessor(config)
results = processor.get_similar_queries("How many patients?")
```

---

---

## Query Library Structure & Matching Logic

The query library stores pairs of natural language questions and their corresponding SQL queries. To understand what is being searched or matched, it's helpful to know the relevant column roles:

| Column | Description |
| :--- | :--- |
| `INDEX` | The unique identifier of the record in the library. |
| `QUESTION_TYPE` | Category of the query (e.g., `QA` or `COHORT_GENERATOR`). |
| `QUESTION` | The original natural language question (e.g., "How many patients with diabetes?"). Searchable via keyword matching. |
| `QUESTION_MASKED` | **[EXPERIMENTAL BASE]** Generalized version with placeholders (e.g., "How many patients with [CONDITION]?"). This is the column primarily used for embedding-based similarity matching. |
| `QUERY_MASKED` | SQL query template with placeholders (e.g., `SELECT ... WHERE condition = {CONDITION}`). |
| `QUERY_EXECUTABLE` | The valid SQL query ready to run. |
---

## CLI Reference (`rag-cli`)

The pipeline CLI is designed for automation and integration into data workflows.

### Global Options

| Flag | Description |
|---|---|
| `--file`, `-f` | Path to the `.db` query library file. |

---

### Commands

#### `view`
View the first N rows of the library.

| Argument | Description |
|---|---|
| `--limit`, `-l` | Number of records to show. |
| `--columns`, `-c` | Space-separated list of [canonical columns](#available-columns) to display. |
| `--output`, `-o` | Path to export results (`.csv`, `.json`, `.xlsx`). |

**Example:**
```bash
rag-cli -f lib.db view --limit 20 -o results.csv
```

---

#### `info`
View full details for specific record IDs.

| Argument | Description |
|---|---|
| `id` | One or more record IDs to inspect. |
| `--columns`, `-c` | Space-separated list of [canonical columns](#available-columns) to display. |
| `--output`, `-o` | Path to export results (`.csv`, `.json`, `.xlsx`). |

**Example:**
```bash
rag-cli -f lib.db info 42 43 -o records.json
```

---

#### `search`
Search questions for a keyword (case-insensitive).

| Argument | Description |
|---|---|
| `term` | The keyword to search for. |
| `--columns`, `-c` | Space-separated list of [canonical columns](#available-columns) to display. |
| `--output`, `-o` | Path to export results (`.csv`, `.json`, `.xlsx`). |

**Example:**
```bash
rag-cli -f lib.db search "biopsy" -o search_results.xlsx
```

---

#### `match`
Perform similarity matching for one or more natural language queries.

| Argument | Description |
|---|---|
| `query` | One or more questions to match. |
| `--k`, `-k` | Number of top results to return. |
| `--threshold`, `-t` | Minimum similarity score (0.0 to 1.0). |
| `--type`, `-y` | Filter by question type. Choices: `QA`, `COHORT`. |
| `--embedding-model`, `-e` | Model name. Common: `BAAI/bge-large-en-v1.5`, `intfloat/multilingual-e5-large`. |
| `--columns`, `-c` | Space-separated list of [canonical columns](#available-columns) to display. |
| `--output`, `-o` | Path to export results (`.csv`, `.json`, `.xlsx`). |

**Example:**
```bash
rag-cli -f lib.db match "How many patients?" --type QA -k 3 -t 0.6 -o matches.csv
```

---

#### `embeddings`
Manage stored embedding models.

**Subcommands:**
- `list`: Show all models stored in the library.
- `add --model <name>`: Calculate and store embeddings for a new model.

| Argument | Description |
|---|---|
| `--model`, `-m` | HuggingFace model identifier (required for `add`). |

**Example:**
```bash
rag-cli -f lib.db embeddings list
rag-cli -f lib.db embeddings add --model intfloat/multilingual-e5-large
```

---

### Available Embedding Models

You can use any model available on HuggingFace by passing its ID to the `--model` flag. However, the following models are recommended and used as defaults:

- **Default**: `BAAI/bge-large-en-v1.5` (Best for English)
- **Multilingual**: `intfloat/multilingual-e5-large` (Recommended for non-English or mixed content)

---

#### `docs`
Show auto-generated documentation for all CLI commands.

---

### Available Columns

When using the `--columns` or `-c` flag, you can specify any of the following:

| Column | Description |
|---|---|
| `INDEX` | The unique identifier of the record. |
| `QUESTION_TYPE` | Category (e.g., `QA` or `COHORT_GENERATOR`). |
| `QUESTION` | The original natural language question. |
| `QUESTION_MASKED` | Generalized version with placeholders (what is embedded). |
| `QUERY_MASKED` | SQL query template with placeholders. |
| `QUERY_EXECUTABLE` | Valid SQL query ready to run. |
| `SourceQuery` | (Only for `match`) The input question provided to the CLI. |
| `Score` | (Only for `match`) Similarity score. |
| `SearchTerm` | (Only for `search`) The term searched for. |

---

## Interactive CLI (`rag-cli-interactive`)

For manual exploration, a menu-driven interactive CLI is available. It provides all the same functionality as the pipeline CLI, plus database scanning and system overview features.

```bash
uv run rag-cli-interactive
```

> **Note:** The "View Full CLI Documentation" option (and the `docs` command) displays documentation that is automatically synchronized with the docstrings in the source code. This means the displayed information is always accurate and reflects the current implementation.

---

## Testing

```bash
uv run pytest tests
```

---

## Documentation

### Command Line
You can view detailed documentation for all CLI commands directly:
```bash
rag-cli docs
```

### Python API (Standalone HTML)
For the full Pythonic API documentation (docstrings, signatures, and examples), you can generate a standalone HTML site:

```bash
uv run pdoc src/rag_helper -o docs/api
```
This will create a searchable, ReadTheDocs-style documentation site in `docs/api`.

## Python API Equivalents

All CLI functionalities are available via a simplified Python API in `rag_helper`.
The API functions mirror the CLI command names and prioritize Pythonic naming (avoiding shadowing built-ins).

**1. Load Library**
```python
import rag_helper as rh
ql = rh.load("lib.db")
```

**2. View Data (`view`, `info`)**
```python
# View first 10 rows
df = rh.view(ql, limit=10)

# Get specific records (using record_id instead of id to avoid shadowing)
df_info = rh.info(ql, record_id=[42, 43])
```

**3. Search (`search`)**
```python
results = rh.search(ql, "biopsy")
```

**4. Match (`match`)**
```python
# Perform similarity matching (using question_type instead of type)
results = rh.match(
    file=ql, 
    query="How many patients?", 
    k=5, 
    threshold=0.5,
    question_type="QA",
    embedding_model="intfloat/multilingual-e5-large"
)
```

**5. Manage Embeddings (`embeddings`)**
```python
# List models
models = rh.embeddings(ql, action="list")

# Add model
rh.embeddings(ql, action="add", model="BAAI/bge-large-en-v1.5")
```

---

## Design & Best Practices

- **Logging Over Printing**: The Python API uses the standard `logging` module. It does not print to `stdout` (except in the CLI entry points), making it safe for use in servers, notebooks, and pipelines.
- **Pythonic Naming**: Parameter names like `record_id` and `question_type` are used to avoid shadowing Python built-ins like `id()` and `type()`.
- **Singleton Management**: Functions like `rh.match()` automatically manage the `QueryLibraryManager` singleton state for the duration of the call.


---

## Architecture

- `api.py`: Simplified high-level wrapper for programmatic use.
- `query_library_manager.py`: Core logic for SQLite storage and embedding management.
- `rag.py`: Coordination layer for the RAG matching flow.
- `cli.py`: Entry point for the pipeline CLI (`rag-cli`).
- `interactive_cli.py`: Entry point for the interactive explorer (`rag-cli-interactive`).
- `utils.py`: Shared utilities for dataframe normalization and file exports.
