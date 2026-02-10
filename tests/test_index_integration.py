import os
import sqlite3
import pandas as pd
import pytest
from rag_helper.query_library_manager import QueryLibraryManager
from rag_helper.cli import normalize_library_df
from unittest.mock import patch

def test_index_preservation_real_sqlite(tmp_path):
    """
    Detailed integration test:
    1. Create a SQLite DB manually with an 'id' column starting at 100.
    2. Load it with QueryLibraryManager.
    3. Verify normalize_library_df uses these IDs as INDEX.
    """
    db_path = str(tmp_path / "integration_index.db")
    
    # 1. Manual SQLite setup
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE queries (id INTEGER PRIMARY KEY, question TEXT, question_masked TEXT, query_with_placeholders TEXT)")
    conn.execute("INSERT INTO queries (id, question, question_masked, query_with_placeholders) VALUES (101, 'Q1', 'M1', 'P1')")
    conn.execute("INSERT INTO queries (id, question, question_masked, query_with_placeholders) VALUES (202, 'Q2', 'M2', 'P2')")
    
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value BLOB)")
    conn.execute("INSERT INTO metadata (key, value) VALUES ('querylib_name', 'integration')")
    conn.execute("INSERT INTO metadata (key, value) VALUES ('source', 'manual')")
    conn.execute("INSERT INTO metadata (key, value) VALUES ('col_question', 'question')")
    conn.execute("INSERT INTO metadata (key, value) VALUES ('col_question_masked', 'question_masked')")
    conn.execute("INSERT INTO metadata (key, value) VALUES ('col_query_w_placeholders', 'query_with_placeholders')")
    conn.execute("INSERT INTO metadata (key, value) VALUES ('date_live', NULL)")
    
    conn.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, model_name TEXT, embed_matrix BLOB, matrix_shape TEXT)")
    conn.commit()
    conn.close()
    
    # 2. Load via QLM
    ql = QueryLibraryManager.load(db_path)
    assert ql.df_querylib.index.tolist() == [101, 202], "Index should be restored from 'id' column"
    
    # 3. Verify CLI normalization
    df_norm = normalize_library_df(ql, ql.df_querylib)
    assert df_norm["INDEX"].tolist() == [101, 202], "CLI INDEX column should match restored index"
    
def test_index_preservation_capital_id(tmp_path):
    """Verify loading from a DB that uses 'ID' (all caps) as the primary key name."""
    db_path = str(tmp_path / "integration_index_caps.db")
    
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE queries (ID INTEGER PRIMARY KEY, question TEXT, question_masked TEXT, query_with_placeholders TEXT)")
    conn.execute("INSERT INTO queries (ID, question, question_masked, query_with_placeholders) VALUES (500, 'Q', 'M', 'P')")
    
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value BLOB)")
    conn.execute("INSERT INTO metadata (key, value) VALUES ('querylib_name', 'integration')")
    conn.execute("INSERT INTO metadata (key, value) VALUES ('source', 'manual')")
    conn.execute("INSERT INTO metadata (key, value) VALUES ('col_question', 'question')")
    conn.execute("INSERT INTO metadata (key, value) VALUES ('col_question_masked', 'question_masked')")
    conn.execute("INSERT INTO metadata (key, value) VALUES ('col_query_w_placeholders', 'query_with_placeholders')")
    conn.execute("INSERT INTO metadata (key, value) VALUES ('date_live', NULL)")
    
    conn.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, model_name TEXT, embed_matrix BLOB, matrix_shape TEXT)")
    conn.commit()
    conn.close()
    
    ql = QueryLibraryManager.load(db_path)
    assert ql.df_querylib.index.tolist() == [500], "Index should be restored from 'ID' column"
    
    df_norm = normalize_library_df(ql, ql.df_querylib)
    assert df_norm["INDEX"].tolist() == [500]

def test_index_preservation_excel_logic(tmp_path):
    """Verify that QueryLibraryManager detects an 'ID' column in an Excel DF and sets it as index."""
    # We can't easily write excel without openpyxl/xlsxwriter, which might not be in env.
    # But we can test the logic in __init__ by passing a mock or just verifying the code paths.
    
    # Let's mock pd.read_excel
    mock_df = pd.DataFrame({
        "ID": [10, 20],
        "question": ["Q1", "Q2"],
        "masked": ["M1", "M2"],
        "qp": ["P1", "P2"]
    })
    
    with patch("pandas.read_excel", return_value=mock_df):
        ql = QueryLibraryManager(
            querylib_name="excel_test",
            source="excel",
            querylib_source_file="fake.xlsx",
            col_question="question",
            col_question_masked="masked",
            col_query_w_placeholders="qp"
        )
        
        assert ql.df_querylib.index.tolist() == [10, 20]
        assert ql.df_querylib.index.name == "ID"
