"""
# RAG Helper Usage Guide

`rag_helper` is a utility for managing and querying SQL query libraries using Retrieval-Augmented Generation (RAG). 
It provides a simplified Python API and a powerful Command Line Interface (CLI).

## Quick Start

The easiest way to perform similarity matching is using the `match` function directly:

```python
import rag_helper as rh

# Load a library and match a query in one go
results = rh.match(
    file="path/to/querylib.db", 
    query="How many patients had a biopsy?", 
    k=3
)
print(results[["Score", "QUESTION_MASKED", "QUERY_MASKED"]])
```

## Core API Functions

### Loading Libraries
Use `load()` to get a `QueryLibraryManager` instance for manual inspection:
```python
ql = rh.load("lib.db")
print(f"Loaded {len(ql.df_querylib)} records.")
```

### Viewing and Inspecting
```python
# View first 5 rows
df = rh.view(ql, limit=5)

# Inspect specific records by ID
df_info = rh.info(ql, record_id=[10, 25])
```

### Keyword Searching
Perform non-semantic, case-insensitive searches on the original questions:
```python
results = rh.search(ql, term="biopsy")
```

### Similarity Matching (RAG)
Powerful semantic matching using stored embeddings:
```python
results = rh.match(
    file=ql,
    query="Show me cohort generator queries",
    question_type="COHORT_GENERATOR" # Filter by type
)
```

## Embedding Management
Manage which models are stored in your library:
```python
# List available models
models = rh.embeddings(ql, action="list")

# Add and calculate embeddings for a new model
rh.embeddings(ql, action="add", model="BAAI/bge-small-en-v1.5")
```

## Integration & Best Practices
- **DataFrames**: All API functions return `pandas.DataFrame` objects for easy analysis.
- **Logging**: Configure logging via `logging.basicConfig()` to see warnings and info messages.
- **Clean Indexing**: The `INDEX` or `ID` from your database is preserved throughout all transformations.
"""

# Expose simplified API
from rag_helper.api import load, view, info, search, match, embeddings

__all__ = ["load", "view", "info", "search", "match", "embeddings"]
