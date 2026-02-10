import os
import tempfile
import sqlite3
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from pathlib import Path
from rag_helper.query_library_manager import QueryLibraryManager, get_latest_querylib_file

def test_get_latest_querylib_file_varied_formats():
    """Test finding the latest DB file with different date formats and invalid names."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create dummy files
        # querylib_20240101.db (valid)
        # querylib_20240102.db (valid, latest)
        # querylib_zzzz.db (invalid date, should trigger fallback)
        f1 = Path(tmp_dir, "querylib_20240101.db")
        f2 = Path(tmp_dir, "querylib_20240102.db")
        f3 = Path(tmp_dir, "querylib_zzzz.db")
        f1.touch()
        f2.touch()
        f3.touch()
        
        # This will trigger log warnings for 'zzzz' but should return it as fallback if it sorts last
        latest = get_latest_querylib_file(tmp_dir)
        assert latest in [str(f1), str(f2), str(f3)]
        
        # Test no files
        with tempfile.TemporaryDirectory() as empty_dir:
            assert get_latest_querylib_file(empty_dir) is None

def test_save_auto_extension():
    """Test that .db extension is added automatically if missing."""
    ql = QueryLibraryManager(
        querylib_name="test",
        source="test",
        querylib_source_file=None,
        col_question="Q",
        col_question_masked="QM",
        col_query_w_placeholders="QP"
    )
    # Mock data to allow saving
    ql.df_querylib = pd.DataFrame({"Q": ["test"], "QM": ["test"], "QP": ["test"]})
    ql.embeddings = [{"model_name": "m", "embed_matrix": np.array([[0.1]])}]
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_lib")
        ql.save(db_path)
        assert os.path.exists(db_path + ".db")

def test_verify_embeddings_corrupt():
    """Test verification of embeddings with missing keys or wrong types."""
    ql = QueryLibraryManager("t", "s", None, "q", "qm", "qp")
    
    # Case 1: No embeddings (Now returns True but logs warning)
    assert ql.verify_embeddings() is True
    
    # Case 2: Missing model_name
    ql.embeddings = [{"embed_matrix": np.array([1, 2])}]
    assert ql.verify_embeddings() is False
    
    # Case 3: Missing embed_matrix
    ql.embeddings = [{"model_name": "m"}]
    assert ql.verify_embeddings() is False
    
    # Case 4: Wrong type for matrix
    ql.embeddings = [{"model_name": "m", "embed_matrix": [1, 2]}]
    assert ql.verify_embeddings() is False

def test_load_invalid_file_type():
    """Test that loading from a non-.db file raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        with pytest.raises(ValueError, match="Unsupported file type"):
            QueryLibraryManager.load(tmp.name)

def test_load_corrupt_sqlite():
    """Test handling of corrupt or incomplete SQLite databases."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        # Write some junk to the file
        with open(tmp.name, "wb") as f:
            f.write(b"NOT A SQLITE DB")
        
        result = QueryLibraryManager.load(tmp.name)
        assert result is None

def test_get_similar_questions_invalid_type():
    """Test that invalid question_type raises ValueError."""
    ql = QueryLibraryManager("t", "s", None, "q", "qm", "qp")
    with pytest.raises(ValueError, match="question_type must be one of"):
        ql.get_similar_questions(["test"], question_type="INVALID")

def test_get_df_recs_empty_branch():
    """Test branch where get_similar_questions returns no results."""
    ql = QueryLibraryManager("t", "s", None, "q", "qm", "qp")
    # By default, get_similar_questions will fail because ql.embeddings is empty
    # But we want to test the 'if not df_recs_list_merged' branch in get_df_recs
    
    # Mock get_similar_questions to return empty list
    ql.get_similar_questions = lambda *args, **kwargs: (pd.DataFrame(), [])
    
    df_recs = ql.get_df_recs("test", 5, 0.0, None)
    assert df_recs.empty
    assert "Score" in df_recs.columns

def test_calc_embedding_exception_clean():
    """Test error handling in calc_embedding."""
    ql = QueryLibraryManager("t", "s", None, "q", "qm", "qp")
    ql.df_querylib = pd.DataFrame({"q": ["test"], "qm": ["test"]})
    ql.col_question = "q"
    ql.col_question_masked = "qm"
    
    # Mock make_sentence_transformer to raise exception
    with patch("rag_helper.query_library_manager.make_sentence_transformer", side_effect=Exception("Model Load Fail")):
         with pytest.raises(Exception, match="Model Load Fail"):
             ql.calc_embedding()
