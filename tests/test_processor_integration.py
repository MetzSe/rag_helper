import os
import pytest
from rag_helper.rag import QueryLibraryManager, RAGProcessor, RAGConfig
from rag_helper.query_library import QueryLibrary
import pandas as pd
import numpy as np

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "querylib_20240101.db"
    df = pd.DataFrame({
        "QUESTION": ["test question"],
        "QUESTION_MASKED": ["test [ENTITY]"],
        "QUERY_WITH_PLACEHOLDERS": ["SELECT *"],
        "QUERY_RUNNABLE": ["SELECT *"],
        "QUESTION_TYPE": ["QA"]
    })
    ql = QueryLibrary("test", "test", None, "QUESTION", "QUESTION_MASKED", "QUERY_WITH_PLACEHOLDERS", "QUERY_RUNNABLE")
    ql.df_querylib = df
    ql.embeddings.append({
        "model_name": "BAAI/bge-large-en-v1.5",
        "embed_matrix": np.zeros((1, 384), dtype=np.float32)
    })
    ql.save(str(db_path))
    return db_path

def test_manager_singleton():
    m1 = QueryLibraryManager.get_instance()
    m2 = QueryLibraryManager.get_instance()
    assert m1 is m2

def test_end_to_end_matching(temp_db, monkeypatch):
    manager = QueryLibraryManager.get_instance()
    # Reset manager state
    manager.querylib = None
    
    # Mock embedding model to avoid downloading
    from unittest.mock import MagicMock, patch
    with patch("rag_helper.query_library.make_sentence_transformer") as mock_transformer:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)
        mock_transformer.return_value = mock_model
        
        # load_querylib expects main_path or querylib_file
        # It now uses get_latest_querylib_file if querylib_file is None
        manager.load_querylib(querylib_file=str(temp_db))
        
        config = RAGConfig()
        processor = RAGProcessor(config)
        results = processor.get_similar_queries("test question", top_k=1)
        
        assert not results.empty
        assert results.iloc[0]["QUESTION"] == "test question"
        assert "Score" in results.columns
