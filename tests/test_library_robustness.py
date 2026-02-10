import sqlite3
import pandas as pd
import numpy as np
import pytest
from rag_helper.query_library_manager import QueryLibraryManager

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "QUESTION": ["What is the patient count?", "List all medications"],
        "QUESTION_MASKED": ["What is the [ENTITY] count?", "List all [ENTITY]"],
        "QUERY_WITH_PLACEHOLDERS": ["SELECT count(*) FROM patients", "SELECT * FROM drugs"],
        "QUERY_RUNNABLE": ["SELECT count(*) FROM patients", "SELECT * FROM drugs"],
        "QUESTION_TYPE": ["QA", "QA"]
    })

def test_library_save_load_cycle(sample_df, tmp_path):
    db_file = tmp_path / "test_cycle.db"
    ql = QueryLibraryManager(
        querylib_name="cycle_test",
        source="unit",
        querylib_source_file=None,
        col_question="QUESTION",
        col_question_masked="QUESTION_MASKED",
        col_query_w_placeholders="QUERY_WITH_PLACEHOLDERS",
        col_query_executable="QUERY_RUNNABLE"
    )
    ql.df_querylib = sample_df
    
    # Add dummy embedding
    import numpy as np
    ql.embeddings.append({
        "model_name": "test-model",
        "embed_matrix": np.random.rand(2, 10).astype(np.float32)
    })
    
    ql.save(str(db_file))
    
    # Load back
    loaded_ql = QueryLibraryManager.load(db_file)
    assert loaded_ql.querylib_name == "cycle_test"
    assert len(loaded_ql.df_querylib) == 2
    assert loaded_ql.col_question == "QUESTION"
    assert len(loaded_ql.embeddings) == 1
    assert loaded_ql.embeddings[0]["model_name"] == "test-model"

def test_handling_of_null_questions(tmp_path):
    df = pd.DataFrame({
        "QUESTION": ["Valid question", None],
        "QUESTION_MASKED": ["Valid [ENTITY]", None],
        "QUERY_WITH_PLACEHOLDERS": ["SELECT 1", "SELECT 2"]
    })
    
    ql = QueryLibraryManager(
        querylib_name="null_test",
        source="unit",
        querylib_source_file=None,
        col_question="QUESTION",
        col_question_masked="QUESTION_MASKED",
        col_query_w_placeholders="QUERY_WITH_PLACEHOLDERS"
    )
    ql.df_querylib = df
    
    # calc_embedding should drop the null row
    from unittest.mock import MagicMock, patch
    with patch("rag_helper.query_library_manager.make_sentence_transformer") as mock_transformer:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)
        mock_transformer.return_value = mock_model
        
        ql.calc_embedding(use_masked=False)
        
    assert len(ql.df_querylib) == 1
    assert ql.df_querylib.iloc[0]["QUESTION"] == "Valid question"

def test_storage_type_initialization():
    ql = QueryLibraryManager("test", "test", None, "Q", "QM", "QP")
    assert ql.storage_type == "sqlite"
