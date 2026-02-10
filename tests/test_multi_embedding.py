"""
Comprehensive tests for multi-embedding support.

Tests cover:
- Deduplication when adding embeddings
- Listing stored embedding models
- Looking up embeddings by model name
- Round-trip save/load of multiple embeddings
- Model selection in get_similar_questions
- Default model behavior (backwards compatible)
- CLI --embedding-model flag parsing
- CLI embeddings list/add subcommands
- RAGConfig embedding_model_name passthrough
"""

import tempfile
import sqlite3
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from rag_helper.query_library_manager import (
    QueryLibraryManager,
    DEFAULT_EMBEDDING_MODEL,
    MULTILINGUAL_E5_MODEL,
)
from rag_helper.rag import RAGConfig, RAGProcessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ql(tmp_path):
    """Create a QueryLibraryManager with sample data and one default embedding."""
    ql = QueryLibraryManager.__new__(QueryLibraryManager)
    ql.querylib_name = "test_library"
    ql.source = "test"
    ql.col_question = "QUESTION"
    ql.col_question_masked = "QUESTION_MASKED"
    ql.col_query_w_placeholders = "QUERY_PLACEHOLDERS"
    ql.col_query_executable = "QUERY_EXECUTABLE"
    ql.date_live = None
    ql.embedding_model = None
    ql.is_loaded = False

    df = pd.DataFrame({
        "QUESTION": ["How many patients?", "What is the diagnosis?", "Count of visits"],
        "QUESTION_MASKED": ["How many patients?", "What is the diagnosis?", "Count of visits"],
        "QUESTION_TYPE": ["QA", "QA", "COHORT_GENERATOR"],
        "QUERY_PLACEHOLDERS": ["SELECT COUNT(*)", "SELECT diagnosis", "SELECT COUNT(*)"],
        "QUERY_EXECUTABLE": ["SELECT COUNT(*)", "SELECT diagnosis", "SELECT COUNT(*)"],
    })
    ql.df_querylib = df

    # Create one default embedding
    embed_matrix_default = np.random.rand(3, 1024).astype(np.float32)
    ql.embeddings = [
        {"model_name": DEFAULT_EMBEDDING_MODEL, "embed_matrix": embed_matrix_default}
    ]

    return ql


@pytest.fixture
def db_file(sample_ql, tmp_path):
    """Save the sample_ql to a .db file and return its path."""
    db_path = str(tmp_path / "test_querylib.db")
    sample_ql.save(db_path)
    return db_path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_constants():
    """Verify model constants are defined."""
    assert DEFAULT_EMBEDDING_MODEL == "BAAI/bge-large-en-v1.5"
    assert MULTILINGUAL_E5_MODEL == "intfloat/multilingual-e5-large"


# ---------------------------------------------------------------------------
# list_embedding_models
# ---------------------------------------------------------------------------

def test_list_embedding_models(sample_ql):
    """list_embedding_models returns model names from the embeddings list."""
    models = sample_ql.list_embedding_models()
    assert models == [DEFAULT_EMBEDDING_MODEL]


def test_list_embedding_models_empty():
    """list_embedding_models returns empty list if no embeddings."""
    ql = QueryLibraryManager.__new__(QueryLibraryManager)
    ql.embeddings = []
    assert ql.list_embedding_models() == []


def test_list_embedding_models_multiple(sample_ql):
    """list_embedding_models returns all model names when multiple exist."""
    sample_ql.embeddings.append(
        {"model_name": MULTILINGUAL_E5_MODEL, "embed_matrix": np.random.rand(3, 1024).astype(np.float32)}
    )
    models = sample_ql.list_embedding_models()
    assert models == [DEFAULT_EMBEDDING_MODEL, MULTILINGUAL_E5_MODEL]


# ---------------------------------------------------------------------------
# get_embedding_by_model
# ---------------------------------------------------------------------------

def test_get_embedding_by_model_found(sample_ql):
    """get_embedding_by_model returns the correct embedding dict."""
    result = sample_ql.get_embedding_by_model(DEFAULT_EMBEDDING_MODEL)
    assert result["model_name"] == DEFAULT_EMBEDDING_MODEL
    assert result["embed_matrix"].shape == (3, 1024)


def test_get_embedding_by_model_not_found(sample_ql):
    """get_embedding_by_model raises ValueError for unknown model."""
    with pytest.raises(ValueError, match="No embedding found for model"):
        sample_ql.get_embedding_by_model("nonexistent/model")


# ---------------------------------------------------------------------------
# add_embedding (with deduplication)
# ---------------------------------------------------------------------------

@patch("rag_helper.query_library_manager.make_sentence_transformer")
def test_add_embedding_new_model(mock_st, sample_ql):
    """add_embedding calculates and stores a new embedding."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(1024).astype(np.float32)
    mock_st.return_value = mock_model

    result = sample_ql.add_embedding(MULTILINGUAL_E5_MODEL)
    assert result is not None
    assert len(sample_ql.embeddings) == 2
    assert sample_ql.embeddings[1]["model_name"] == MULTILINGUAL_E5_MODEL


@patch("rag_helper.query_library_manager.make_sentence_transformer")
def test_add_embedding_dedup(mock_st, sample_ql):
    """add_embedding returns None and does not duplicate if model already exists."""
    result = sample_ql.add_embedding(DEFAULT_EMBEDDING_MODEL)
    assert result is None
    assert len(sample_ql.embeddings) == 1  # No duplicate added
    mock_st.assert_not_called()  # No model download attempted


# ---------------------------------------------------------------------------
# save_embedding_to_db (with DB-level deduplication)
# ---------------------------------------------------------------------------

def test_save_embedding_to_db_new(sample_ql, db_file):
    """save_embedding_to_db inserts a new model into the DB."""
    # Add a second embedding in memory
    sample_ql.embeddings.append(
        {"model_name": MULTILINGUAL_E5_MODEL, "embed_matrix": np.random.rand(3, 1024).astype(np.float32)}
    )
    result = sample_ql.save_embedding_to_db(db_file, MULTILINGUAL_E5_MODEL)
    assert result is True

    # Verify it's in the DB
    with sqlite3.connect(db_file) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE model_name = ?",
            (MULTILINGUAL_E5_MODEL,)
        ).fetchone()[0]
        assert count == 1


def test_save_embedding_to_db_dedup(sample_ql, db_file):
    """save_embedding_to_db skips if model already in DB."""
    result = sample_ql.save_embedding_to_db(db_file, DEFAULT_EMBEDDING_MODEL)
    assert result is False  # Already saved during fixture creation


# ---------------------------------------------------------------------------
# Round-trip: save multiple → load → verify both present
# ---------------------------------------------------------------------------

def test_round_trip_multiple_embeddings(sample_ql, tmp_path):
    """Save a library with two embeddings, load it, verify both are present."""
    # Add second embedding
    e5_matrix = np.random.rand(3, 1024).astype(np.float32)
    sample_ql.embeddings.append(
        {"model_name": MULTILINGUAL_E5_MODEL, "embed_matrix": e5_matrix}
    )

    db_path = str(tmp_path / "multi_embed.db")
    sample_ql.save(db_path)

    loaded = QueryLibraryManager.load(db_path)
    assert loaded is not None
    models = loaded.list_embedding_models()
    assert DEFAULT_EMBEDDING_MODEL in models
    assert MULTILINGUAL_E5_MODEL in models
    assert len(loaded.embeddings) == 2
    # Verify shapes are preserved
    for emb in loaded.embeddings:
        assert emb["embed_matrix"].shape == (3, 1024)


# ---------------------------------------------------------------------------
# get_similar_questions with model selection
# ---------------------------------------------------------------------------

def test_get_similar_questions_default_model(sample_ql):
    """Default (no model specified) uses embeddings[0]."""
    # Mock loaded model to avoid real loading
    sample_ql.embedding_model = MagicMock()
    sample_ql.current_embedding_model_name = DEFAULT_EMBEDDING_MODEL
    
    with patch.object(sample_ql, "extract_embed_matrix") as mock_extract, \
         patch.object(sample_ql, "add_separator_to_input_entities", return_value=["test"]), \
         patch.object(sample_ql, "get_similar_names", return_value=(pd.DataFrame(), [])):
        mock_extract.return_value = (np.random.rand(3, 1024), ["a", "b", "c"], [0, 1, 2])

        sample_ql.get_similar_questions(["test"], top_k=1)

        # Verify the default embedding was passed
        call_kwargs = mock_extract.call_args
        assert call_kwargs[1]["embedding"]["model_name"] == DEFAULT_EMBEDDING_MODEL


def test_get_similar_questions_with_model_selection(sample_ql):
    """Specifying embedding_model_name selects the correct embedding."""
    # Add second embedding
    e5_matrix = np.random.rand(3, 1024).astype(np.float32)
    sample_ql.embeddings.append(
        {"model_name": MULTILINGUAL_E5_MODEL, "embed_matrix": e5_matrix}
    )
    
    # Mock loaded model
    sample_ql.embedding_model = MagicMock()
    sample_ql.current_embedding_model_name = MULTILINGUAL_E5_MODEL

    with patch.object(sample_ql, "extract_embed_matrix") as mock_extract, \
         patch.object(sample_ql, "add_separator_to_input_entities", return_value=["test"]), \
         patch.object(sample_ql, "get_similar_names", return_value=(pd.DataFrame(), [])):
        mock_extract.return_value = (np.random.rand(3, 1024), ["a", "b", "c"], [0, 1, 2])

        sample_ql.get_similar_questions(
            ["test"], top_k=1, embedding_model_name=MULTILINGUAL_E5_MODEL
        )

        call_kwargs = mock_extract.call_args
        assert call_kwargs[1]["embedding"]["model_name"] == MULTILINGUAL_E5_MODEL


def test_get_similar_questions_invalid_model(sample_ql):
    """Specifying a non-existent model raises ValueError."""
    with pytest.raises(ValueError, match="No embedding found for model"):
        sample_ql.get_similar_questions(
            ["test"], top_k=1, embedding_model_name="nonexistent/model"
        )


# ---------------------------------------------------------------------------
# get_df_recs passthrough
# ---------------------------------------------------------------------------

def test_get_df_recs_passes_model_name(sample_ql):
    """get_df_recs forwards embedding_model_name to get_similar_questions."""
    with patch.object(sample_ql, "get_similar_questions") as mock_gsq:
        # Return a tuple: (DataFrame, list of DataFrames)
        # The dataframe in the list must have 'QUESTION' column for the merge in get_df_recs
        mock_gsq.return_value = (
            pd.DataFrame({"Score": [0.9], "QUESTION_MASKED": ["test"]}),
            [pd.DataFrame({"Score": [0.9], "QUESTION": ["How many patients?"]})]
        )

        sample_ql.get_df_recs(
            ["test"], top_k=5, sim_threshold=0.0,
            question_type=None, embedding_model_name=MULTILINGUAL_E5_MODEL
        )

        _, call_kwargs = mock_gsq.call_args
        assert call_kwargs["embedding_model_name"] == MULTILINGUAL_E5_MODEL


def test_get_df_recs_default_model_name(sample_ql):
    """get_df_recs defaults to embedding_model_name=None."""
    with patch.object(sample_ql, "get_similar_questions") as mock_gsq:
        mock_gsq.return_value = (
            pd.DataFrame({"Score": [0.9], "QUESTION_MASKED": ["test"]}),
            [pd.DataFrame({"Score": [0.9], "QUESTION": ["How many patients?"]})]
        )

        sample_ql.get_df_recs(
            ["test"], top_k=5, sim_threshold=0.0, question_type=None
        )

        _, call_kwargs = mock_gsq.call_args
        assert call_kwargs["embedding_model_name"] is None


# ---------------------------------------------------------------------------
# RAGConfig embedding_model_name
# ---------------------------------------------------------------------------

def test_rag_config_default_embedding_model():
    """RAGConfig defaults embedding_model_name to None."""
    config = RAGConfig()
    assert config.embedding_model_name is None


def test_rag_config_custom_embedding_model():
    """RAGConfig accepts a custom embedding_model_name."""
    config = RAGConfig(embedding_model_name=MULTILINGUAL_E5_MODEL)
    assert config.embedding_model_name == MULTILINGUAL_E5_MODEL


# ---------------------------------------------------------------------------
# CLI: match --embedding-model flag parsing
# ---------------------------------------------------------------------------

def test_match_cli_embedding_model_flag():
    """CLI parser correctly parses --embedding-model flag."""
    from rag_helper.cli import get_parser
    parser = get_parser()

    args = parser.parse_args([
        "-f", "test.db", "match", "How many?",
        "--embedding-model", MULTILINGUAL_E5_MODEL
    ])
    assert args.embedding_model == MULTILINGUAL_E5_MODEL


def test_match_cli_embedding_model_default():
    """CLI parser defaults --embedding-model to None."""
    from rag_helper.cli import get_parser
    parser = get_parser()

    args = parser.parse_args(["-f", "test.db", "match", "How many?"])
    assert args.embedding_model is None


def test_match_cli_embedding_model_short_flag():
    """CLI parser accepts -e short flag."""
    from rag_helper.cli import get_parser
    parser = get_parser()

    args = parser.parse_args([
        "-f", "test.db", "match", "How many?",
        "-e", MULTILINGUAL_E5_MODEL
    ])
    assert args.embedding_model == MULTILINGUAL_E5_MODEL


# ---------------------------------------------------------------------------
# CLI: embeddings list / add subcommands
# ---------------------------------------------------------------------------

def test_embeddings_list_cli_parser():
    """CLI parser recognises 'embeddings list' subcommand."""
    from rag_helper.cli import get_parser
    parser = get_parser()

    args = parser.parse_args(["-f", "test.db", "embeddings", "list"])
    assert args.command == "embeddings"
    assert args.emb_command == "list"


def test_embeddings_add_cli_parser():
    """CLI parser recognises 'embeddings add --model' subcommand."""
    from rag_helper.cli import get_parser
    parser = get_parser()

    args = parser.parse_args([
        "-f", "test.db", "embeddings", "add",
        "--model", MULTILINGUAL_E5_MODEL
    ])
    assert args.command == "embeddings"
    assert args.emb_command == "add"
    assert args.model == MULTILINGUAL_E5_MODEL


def test_embeddings_list_function(sample_ql, capsys):
    """embeddings_list prints stored models without error."""
    from rag_helper.cli import embeddings_list
    embeddings_list(sample_ql)
    # Should not raise; output goes to rich console


def test_embeddings_list_empty(capsys):
    """embeddings_list handles no embeddings gracefully."""
    from rag_helper.cli import embeddings_list
    ql = QueryLibraryManager.__new__(QueryLibraryManager)
    ql.embeddings = []
    embeddings_list(ql)
    # Should not raise


@patch("rag_helper.query_library_manager.make_sentence_transformer")
def test_embeddings_add_function(mock_st, sample_ql, db_file):
    """embeddings_add calculates, stores, and saves to DB."""
    from rag_helper.cli import embeddings_add

    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(1024).astype(np.float32)
    mock_st.return_value = mock_model

    # Ensure api.embeddings triggers save
    sample_ql.source = db_file
    
    embeddings_add(sample_ql, MULTILINGUAL_E5_MODEL, db_file)

    # Verify it is now in the in-memory list
    assert MULTILINGUAL_E5_MODEL in sample_ql.list_embedding_models()

    # Verify it is persisted in the DB
    with sqlite3.connect(db_file) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE model_name = ?",
            (MULTILINGUAL_E5_MODEL,)
        ).fetchone()[0]
        assert count == 1


@patch("rag_helper.query_library_manager.make_sentence_transformer")
def test_embeddings_add_dedup_function(mock_st, sample_ql, db_file):
    """embeddings_add skips gracefully when model already exists."""
    from rag_helper.cli import embeddings_add

    embeddings_add(sample_ql, DEFAULT_EMBEDDING_MODEL, db_file)

    # Should still have only 1 embedding
    assert len(sample_ql.embeddings) == 1
    mock_st.assert_not_called()
