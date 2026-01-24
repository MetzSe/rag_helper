import pytest
import pandas as pd
import numpy as np
from rag_helper.query_library import QueryLibrary
from unittest.mock import MagicMock, patch

def test_very_long_input(tmp_path):
    long_question = "a" * 10000
    ql = QueryLibrary("test", "test", None, "Q", "QM", "QP")
    ql.df_querylib = pd.DataFrame({"Q": [long_question], "QM": [long_question], "QP": ["SELECT 1"]})
    
    with patch("rag_helper.query_library.make_sentence_transformer") as mock_transformer:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)
        mock_transformer.return_value = mock_model
        
        # Test that encoding doesn't crash
        ql.calc_embedding()
        assert len(ql.embeddings) == 1

def test_non_ascii_input():
    # Test emoji and foreign characters
    question = "How many patients have 🧬 and 🧪?"
    ql = QueryLibrary("test", "test", None, "Q", "QM", "QP")
    ql.df_querylib = pd.DataFrame({"Q": [question], "QM": [question], "QP": ["SELECT 1"]})
    
    with patch("rag_helper.query_library.make_sentence_transformer") as mock_transformer:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)
        mock_transformer.return_value = mock_model
        
        ql.calc_embedding()
        assert len(ql.embeddings) == 1

def test_empty_library_search():
    ql = QueryLibrary("test", "test", None, "Q", "QM", "QP")
    ql.df_querylib = pd.DataFrame(columns=["Q", "QM", "QP"])
    ql.embeddings = [{"model_name": "test", "embed_matrix": np.array([]).reshape(0, 384)}]
    
    with patch("rag_helper.query_library.make_sentence_transformer") as mock_transformer:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)
        mock_transformer.return_value = mock_model
        ql.load_embedding_model("test")
        
        with pytest.raises(Exception): # partition on empty array should fail or we should handle it
            ql.get_df_recs("test", 1, 0, "QA")

def test_sql_injection_string_safety():
    # Although RAG doesn't execute SQL on the querylib, we should ensure 
    # that malicious strings in questions don't break the similarity search logic
    malicious = "'; DROP TABLE queries; --"
    ql = QueryLibrary("test", "test", None, "Q", "QM", "QP")
    ql.df_querylib = pd.DataFrame({"Q": [malicious], "QM": [malicious], "QP": ["SELECT 1"]})
    
    with patch("rag_helper.query_library.make_sentence_transformer") as mock_transformer:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)
        mock_transformer.return_value = mock_model
        
        ql.calc_embedding()
        assert ql.df_querylib.iloc[0]["Q"] == malicious
