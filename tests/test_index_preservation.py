import os
import pandas as pd
import pytest
from rag_helper.query_library_manager import QueryLibraryManager
from rag_helper.cli import normalize_library_df

def test_index_preservation_sqlite(tmp_path):
    """Verify that original indices are preserved when saving/loading from SQLite."""
    # Create a DF with non-contiguous indices
    df = pd.DataFrame({
        "question": ["Q1", "Q3", "Q5"],
        "masked": ["M1", "M3", "M5"],
        "qp": ["P1", "P3", "P5"]
    }, index=[1, 3, 5])
    
    ql = QueryLibraryManager(
        querylib_name="index_test",
        source="unit_test",
        querylib_source_file=None,
        col_question="question",
        col_question_masked="masked",
        col_query_w_placeholders="qp"
    )
    ql.df_querylib = df
    
    db_path = str(tmp_path / "index_test.db")
    ql.save(db_path)
    
    # Load it back
    loaded_ql = QueryLibraryManager.load(db_path)
    
    assert list(loaded_ql.df_querylib.index) == [1, 3, 5]
    assert loaded_ql.df_querylib.loc[3]["question"] == "Q3"

def test_normalize_df_uses_index_for_index_column():
    """Verify that normalize_library_df picks up the actual DF index for the 'INDEX' column."""
    df = pd.DataFrame({
        "question": ["Q1"],
        "masked": ["M1"],
        "qp": ["P1"]
    }, index=[123])
    
    ql = MagicMock()
    ql.col_question = "question"
    ql.col_question_masked = "masked"
    ql.col_query_w_placeholders = "qp"
    ql.col_query_executable = "qe"
    
    df_norm = normalize_library_df(ql, df)
    
    assert "INDEX" in df_norm.columns
    assert df_norm.iloc[0]["INDEX"] == 123

from unittest.mock import MagicMock
