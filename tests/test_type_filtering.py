import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from rag_helper.query_library_manager import QueryLibraryManager
from rag_helper.rag import RAGConfig, RAGProcessor

@pytest.fixture
def filter_db(tmp_path):
    db_path = tmp_path / "filter_test.db"
    ql = QueryLibraryManager(
        querylib_name="filter_test",
        source="internal",
        querylib_source_file=None,
        col_question="QUESTION",
        col_question_masked="QUESTION_MASKED",
        col_query_w_placeholders="QUERY_MASKED"
    )
    
    # Create test data with lowercase 'question_type' to test normalization
    data = {
        "QUESTION": ["What is A?", "How many B?", "Find C"],
        "QUESTION_MASKED": ["What is [A]?", "How many [B]?", "Find [C]"],
        "QUERY_MASKED": ["SELECT * FROM A", "SELECT COUNT(*) FROM B", "SELECT * FROM C"],
        "question_type": ["QA", "COHORT_GENERATOR", "QA"]
    }
    ql.df_querylib = pd.DataFrame(data)
    ql.df_querylib.index.name = "ID"
    
    # Add dummy embeddings (3 rows, 384 dims)
    matrix = np.ones((3, 384), dtype=np.float32)
    ql.embeddings = [{
        "model_name": "test-model",
        "embed_matrix": matrix
    }]
    
    ql.save(str(db_path))
    return db_path

def test_type_filtering_logic(filter_db):
    ql = QueryLibraryManager.load(str(filter_db))
    
    mock_model = MagicMock()
    mock_model.encode.return_value = np.ones((1, 384), dtype=np.float32)
    ql.embedding_model = mock_model
    ql.current_embedding_model_name = "test-model"
    
    # Test QA filter
    # Passing single query as string is now supported robustly
    df_qa = ql.get_df_recs("test", top_k=5, sim_threshold=0.0, question_type="QA", embedding_model_name="test-model")
    assert len(df_qa) == 2
    assert "QUESTION_TYPE" in df_qa.columns
    assert all(df_qa["QUESTION_TYPE"] == "QA")
    
    # Test COHORT alias
    df_cohort = ql.get_df_recs("test", top_k=5, sim_threshold=0.0, question_type="COHORT", embedding_model_name="test-model")
    assert len(df_cohort) == 1
    assert all(df_cohort["QUESTION_TYPE"] == "COHORT_GENERATOR")
    
    # Test case insensitivity
    df_qa_lower = ql.get_df_recs("test", top_k=5, sim_threshold=0.0, question_type="qa", embedding_model_name="test-model")
    assert len(df_qa_lower) == 2

def test_type_filtering_no_filter(filter_db):
    ql = QueryLibraryManager.load(str(filter_db))
    
    mock_model = MagicMock()
    mock_model.encode.return_value = np.ones((1, 384), dtype=np.float32)
    ql.embedding_model = mock_model
    ql.current_embedding_model_name = "test-model"
    
    # Test no filter
    df_all = ql.get_df_recs("test", top_k=5, sim_threshold=0.0, question_type=None, embedding_model_name="test-model")
    assert len(df_all) == 3

def test_invalid_type_raises_error(filter_db):
    ql = QueryLibraryManager.load(str(filter_db))
    with pytest.raises(ValueError, match="question_type must be one of"):
        ql.get_similar_questions(["test"], question_type="INVALID")
