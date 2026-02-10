import os
import shutil
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

import rag_helper.api as api
from rag_helper.query_library_manager import QueryLibraryManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_db(tmp_path):
    """Create a temporary SQLite DB for testing."""
    db_path = tmp_path / "test_api.db"
    
    ql = QueryLibraryManager(
        querylib_name="api_test",
        source="internal",
        querylib_source_file=None,
        col_question="QUESTION",
        col_question_masked="QUESTION_MASKED",
        col_query_w_placeholders="QUERY_MASKED"
    )
    
    # Create test data
    data = {
        "QUESTION": ["What is A?", "How many B?", "Find C"],
        "QUESTION_MASKED": ["What is [A]?", "How many [B]?", "Find [C]"],
        "QUERY_MASKED": ["SELECT * FROM A", "SELECT COUNT(*) FROM B", "SELECT * FROM C"],
        "QUESTION_TYPE": ["QA", "COHORT_GENERATOR", "QA"]
    }
    ql.df_querylib = pd.DataFrame(data)
    ql.df_querylib.index.name = "ID"
    # Create a 1-based index to better test ID lookup
    ql.df_querylib.index = [1, 2, 3]

    # Add dummy embeddings
    matrix = np.ones((3, 384), dtype=np.float32)
    ql.embeddings = [{
        "model_name": "test-model",
        "embed_matrix": matrix
    }]
    
    ql.save(str(db_path))
    return db_path

# ---------------------------------------------------------------------------
# LOAD Tests
# ---------------------------------------------------------------------------

def test_load_explicit_path(sample_db):
    """api.load() works with explicit file path."""
    ql = api.load(str(sample_db))
    assert isinstance(ql, QueryLibraryManager)
    assert len(ql.df_querylib) == 3

def test_load_auto_discovery(sample_db):
    """api.load() finds latest file in cwd if none provided."""
    # We must be in the dir where the file is for this logic to work as written in api.load
    # or we mock os.getcwd / get_latest_querylib_file
    with patch("rag_helper.api.get_latest_querylib_file", return_value=str(sample_db)) as mock_find:
        ql = api.load()
        assert isinstance(ql, QueryLibraryManager)
        mock_find.assert_called_once()

def test_load_missing(tmp_path):
    """api.load() raises FileNotFoundError if no file found."""
    with patch("rag_helper.api.get_latest_querylib_file", return_value=None):
        with pytest.raises(FileNotFoundError):
            api.load()

# ---------------------------------------------------------------------------
# VIEW Tests
# ---------------------------------------------------------------------------

def test_view_basic(sample_db):
    """api.view() returns correct DataFrame."""
    df = api.view(str(sample_db), limit=2)
    assert len(df) == 2
    assert "INDEX" in df.columns
    # Verify canonical column names
    assert "QUESTION" in df.columns
    assert "QUESTION_MASKED" in df.columns
    assert "QUESTION_TYPE" in df.columns

def test_view_with_output(sample_db, tmp_path):
    """api.view() exports to file if output provided."""
    out_file = tmp_path / "view.csv"
    api.view(str(sample_db), limit=2, output=str(out_file))
    assert out_file.exists()
    df = pd.read_csv(out_file)
    assert len(df) == 2

def test_view_with_loaded_object(sample_db):
    """api.view() accepts pre-loaded QueryLibraryManager."""
    ql = api.load(str(sample_db))
    df = api.view(ql, limit=1)
    assert len(df) == 1

# ---------------------------------------------------------------------------
# INFO Tests
# ---------------------------------------------------------------------------

def test_info_single_id(sample_db):
    """api.info() retrieves single record."""
    df = api.info(str(sample_db), record_id=2)
    assert len(df) == 1
    assert df.iloc[0]["QUESTION"] == "How many B?"

def test_info_multiple_ids(sample_db):
    """api.info() retrieves multiple records."""
    df = api.info(str(sample_db), record_id=[1, 3])
    assert len(df) == 2
    assert df.iloc[0]["QUESTION"] == "What is A?"
    assert df.iloc[1]["QUESTION"] == "Find C"

def test_info_missing_id(sample_db, caplog):
    """api.info() handles missing IDs gracefully via logging."""
    import logging
    with caplog.at_level(logging.WARNING):
        df = api.info(str(sample_db), record_id=[1, 99])
        assert len(df) == 1 # Should return what it found
        assert "Some IDs not found" in caplog.text

def test_info_all_missing(sample_db):
    """api.info() returns empty DF if all IDs missing."""
    df = api.info(str(sample_db), record_id=[99, 100])
    assert df.empty

# ---------------------------------------------------------------------------
# SEARCH Tests
# ---------------------------------------------------------------------------

def test_search_found(sample_db):
    """api.search() finds matching records."""
    df = api.search(str(sample_db), term="How many")
    assert len(df) == 1
    assert df.iloc[0]["QUESTION"] == "How many B?"
    assert df.iloc[0]["SearchTerm"] == "How many"

def test_search_case_insensitive(sample_db):
    """api.search() is case-insensitive."""
    df = api.search(str(sample_db), term="how many")
    assert len(df) == 1

def test_search_not_found(sample_db):
    """api.search() returns empty DF for no matches."""
    df = api.search(str(sample_db), term="NonExistent")
    assert df.empty

# ---------------------------------------------------------------------------
# MATCH Tests
# ---------------------------------------------------------------------------

def test_match_basic(sample_db):
    """api.match() performs similarity search."""
    # We need to mock the embedding model to avoid loading/downloading
    ql = api.load(str(sample_db))
    ql.embedding_model = MagicMock()
    ql.embedding_model.encode.return_value = np.ones((1, 384), dtype=np.float32)
    ql.current_embedding_model_name = "test-model"

    df = api.match(ql, query="Something similar to A", k=2)
    assert len(df) >= 1
    assert "Score" in df.columns
    assert "SourceQuery" in df.columns

def test_match_threshold(sample_db):
    """api.match() filters by threshold."""
    ql = api.load(str(sample_db))
    ql.embedding_model = MagicMock()
    ql.embedding_model.encode.return_value = np.ones((1, 384), dtype=np.float32)
    ql.current_embedding_model_name = "test-model"

    # With constant embeddings, score is 1.0 (cosine sim of identical vectors) or close
    # Set threshold high -> should pass
    df = api.match(ql, query="A", threshold=0.9)
    print(f"\nDEBUG: df empty? {df.empty}")
    if not df.empty:
        print(f"DEBUG: Scores: {df['Score'].tolist()}")
    assert not df.empty

    # If we could force low score... mocking get_similar_queries easier?
    with patch("rag_helper.rag.RAGProcessor.get_similar_queries") as mock_rag:
        # Mock returning low score results
        mock_res = pd.DataFrame({"Score": [0.1], "QUESTION": ["Q"]})
        mock_rag.return_value = mock_res
        
        df = api.match(ql, query="A", threshold=0.9)
        assert df.empty

def test_match_question_type(sample_db):
    """api.match() filters by question_type."""
    ql = api.load(str(sample_db))
    ql.embedding_model = MagicMock()
    ql.embedding_model.encode.return_value = np.ones((1, 384), dtype=np.float32)
    ql.current_embedding_model_name = "test-model"

    df = api.match(ql, query="A", question_type="COHORT")
    assert all(df["QUESTION_TYPE"] == "COHORT_GENERATOR")

# ---------------------------------------------------------------------------
# EMBEDDINGS Tests
# ---------------------------------------------------------------------------

def test_embeddings_list(sample_db):
    """api.embeddings('list') returns model list."""
    models = api.embeddings(str(sample_db), action="list")
    assert len(models) == 1
    assert models[0]["model_name"] == "test-model"

def test_embeddings_add(sample_db):
    """api.embeddings('add') calls add_embedding and save."""
    # Ensure source looks like a DB file to trigger save
    ql = api.load(str(sample_db))
    ql.source = str(sample_db)
    
    # Mocking add_embedding on the instance
    with patch.object(ql, "add_embedding") as mock_add, \
         patch.object(ql, "save_embedding_to_db") as mock_save:
        
        api.embeddings(ql, action="add", model="new-model")
        mock_add.assert_called_with(embedding_model_name="new-model")
        mock_save.assert_called_with(ql.source, "new-model")

